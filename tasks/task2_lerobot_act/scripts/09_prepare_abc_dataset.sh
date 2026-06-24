#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs results
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"

status_file="results/prepare_abc.status.json"
repo_id="fywang/calvin-task-ABC-D-lerobot"
cache_root="$HOME/.cache/huggingface/lerobot"
dataset_root="$cache_root/fywang/calvin-task-ABC-D-lerobot"
attempts="${ABC_DOWNLOAD_ATTEMPTS:-12}"
sleep_seconds="${ABC_RETRY_SLEEP_SECONDS:-60}"
max_workers="${HF_MAX_WORKERS:-2}"

write_status() {
  local status="$1"
  local exit_code="${2:-}"
  python - "$status_file" "$status" "$exit_code" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
status = sys.argv[2]
exit_code = sys.argv[3]
payload = {}
if path.exists():
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        payload = {}
payload.update({
    "run_id": "prepare_abc",
    "status": status,
    "updated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
    "repo_id": "fywang/calvin-task-ABC-D-lerobot",
    "hf_endpoint": "https://hf-mirror.com",
    "dataset_root": str(Path.home() / ".cache" / "huggingface" / "lerobot" / "fywang" / "calvin-task-ABC-D-lerobot"),
})
if status == "running" and "start_time" not in payload:
    payload["start_time"] = payload["updated_at"]
if status in {"finished", "failed"}:
    payload["end_time"] = payload["updated_at"]
if exit_code:
    payload["exit_code"] = int(exit_code)
path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
PY
}

if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
fi

echo "Preparing ${repo_id}"
echo "HF_ENDPOINT=${HF_ENDPOINT}"
echo "dataset_root=${dataset_root}"
echo "download_attempts=${attempts}"
echo "retry_sleep_seconds=${sleep_seconds}"
echo "hf_max_workers=${max_workers}"
write_status running

for attempt in $(seq 1 "${attempts}"); do
  echo "download_attempt=${attempt}/${attempts}"
  if HF_MAX_WORKERS="${max_workers}" python - <<'PY'
import os
from pathlib import Path
from huggingface_hub import snapshot_download

repo_id = "fywang/calvin-task-ABC-D-lerobot"
root = Path.home() / ".cache" / "huggingface" / "lerobot" / "fywang" / "calvin-task-ABC-D-lerobot"
root.mkdir(parents=True, exist_ok=True)
max_workers = int(os.environ.get("HF_MAX_WORKERS", "2"))
path = snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=str(root),
    max_workers=max_workers,
)
print(f"downloaded_to={path}")
PY
  then
    echo "download_complete"
    break
  fi

  if [ "${attempt}" -eq "${attempts}" ]; then
    echo "download_failed_after_${attempts}_attempts"
    write_status failed 1
    exit 1
  fi

  echo "download_failed_retrying_in_${sleep_seconds}s"
  sleep "${sleep_seconds}"
done

python - <<'PY'
import json
from pathlib import Path

info_path = Path.home() / ".cache" / "huggingface" / "lerobot" / "fywang" / "calvin-task-ABC-D-lerobot" / "meta" / "info.json"
info = json.loads(info_path.read_text())
print("codebase_version_before", info.get("codebase_version"))
PY

python -m lerobot.datasets.v30.convert_dataset_v21_to_v30 \
  --repo-id "${repo_id}" \
  --root "${cache_root}" \
  --push-to-hub false \
  --force-conversion

python - <<'PY'
import json
from pathlib import Path

root = Path.home() / ".cache" / "huggingface" / "lerobot" / "fywang" / "calvin-task-ABC-D-lerobot"
info = json.loads((root / "meta" / "info.json").read_text())
print("codebase_version_after", info.get("codebase_version"))
print("total_episodes", info.get("total_episodes"))
print("total_frames", info.get("total_frames"))
PY

du -sh "${dataset_root}"
write_status finished 0
