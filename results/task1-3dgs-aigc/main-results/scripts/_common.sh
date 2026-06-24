#!/usr/bin/env bash
set -Eeuo pipefail

export HW3_TASK1_ROOT="${HW3_TASK1_ROOT:-/root/autodl-tmp/hw3_task1_3dgs_aigc}"
export PATH="/root/miniconda3/bin:$PATH"
export HF_HOME="$HW3_TASK1_ROOT/models/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export TORCH_HOME="$HW3_TASK1_ROOT/models/torch"
export XDG_CACHE_HOME="$HW3_TASK1_ROOT/models/xdg_cache"
export CONDA_PKGS_DIRS="$HW3_TASK1_ROOT/envs/conda_pkgs"
export TMPDIR="$HW3_TASK1_ROOT/tmp"
export PIP_CACHE_DIR="$HW3_TASK1_ROOT/envs/pip_cache"

mkdir -p "$HW3_TASK1_ROOT"/{scripts,logs,status,data,third_party,envs,models,outputs,results,docs,tmp}
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$TORCH_HOME" "$XDG_CACHE_HOME" "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR"

now_iso() {
  date '+%Y-%m-%dT%H:%M:%S%z'
}

write_status() {
  local run_id="$1"
  local status="$2"
  local exit_code="${3:-null}"
  local note="${4:-}"
  export STATUS_RUN_ID="$run_id"
  export STATUS_STATE="$status"
  export STATUS_EXIT_CODE="$exit_code"
  export STATUS_NOTE="$note"
  /root/miniconda3/bin/python - <<'PY'
import json, os, datetime, pathlib, subprocess
root = pathlib.Path(os.environ["HW3_TASK1_ROOT"])
status_dir = root / "status"
status_dir.mkdir(parents=True, exist_ok=True)
exit_raw = os.environ.get("STATUS_EXIT_CODE", "null")
try:
    exit_code = None if exit_raw in ("", "null", "None") else int(exit_raw)
except ValueError:
    exit_code = exit_raw
payload = {
    "run_id": os.environ["STATUS_RUN_ID"],
    "status": os.environ["STATUS_STATE"],
    "exit_code": exit_code,
    "updated_at": datetime.datetime.now().astimezone().isoformat(),
    "workspace": str(root),
    "note": os.environ.get("STATUS_NOTE", ""),
}
try:
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()
    payload["gpu"] = gpu
except Exception:
    pass
with open(status_dir / f"{payload['run_id']}.json", "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, ensure_ascii=False)
PY
}

activate_conda() {
  local env_prefix="$1"
  source /root/miniconda3/etc/profile.d/conda.sh
  set +u
  conda activate "$env_prefix"
  local rc=$?
  set -u
  return "$rc"
}

log_section() {
  echo
  echo "===== $* ====="
  date '+%F %T %z'
}
