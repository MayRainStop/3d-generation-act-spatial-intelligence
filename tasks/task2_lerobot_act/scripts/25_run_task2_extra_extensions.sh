#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs results outputs/20260603_extensions
chain_id="task2_extra_extensions_chain"
status_file="results/${chain_id}.status.json"
log_file="logs/${chain_id}.log"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export SENTENCE_TRANSFORMERS_HOME="${SENTENCE_TRANSFORMERS_HOME:-${HF_HOME}/sentence_transformers}"
export HW3_LANG_EMBED_MODEL="${HW3_LANG_EMBED_MODEL:-sentence-transformers/all-MiniLM-L6-v2}"
timestamp() { date -Iseconds; }
write_status() {
  local status="$1"
  local exit_code="${2:-null}"
  local note="${3:-}"
  cat > "$status_file" <<JSON
{
  "run_id": "${chain_id}",
  "status": "${status}",
  "exit_code": ${exit_code},
  "updated_at": "$(timestamp)",
  "log_file": "${log_file}",
  "note": "${note}"
}
JSON
}
exec > >(tee -a "$log_file") 2>&1
trap 'rc=$?; if [ "$rc" -ne 0 ]; then write_status failed "$rc" "extra extension chain failed; inspect log_file"; fi' EXIT
. .venv/bin/activate
export HYDRA_FULL_ERROR=1

ensure_config_rows() {
  python - <<'PY'
import csv
from pathlib import Path

path = Path("configs/runs.csv")
rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
fieldnames = list(rows[0].keys())
existing = {row["run_id"] for row in rows}
extra_rows = [
    {
        "run_id": "act_abc_taskcond_100k_seed0",
        "dataset_repo": "fywang/calvin-task-ABC-D-lerobot",
        "output_dir": "outputs/20260603_extensions/act_abc_taskcond_100k_seed0",
        "chunk_size": "50",
        "n_action_steps": "50",
        "seed": "0",
        "batch_size": "8",
        "learning_rate": "0.0001",
        "steps": "100000",
        "priority": "extension",
        "notes": "task-index one-hot conditioned ACT, longer 100k training",
    },
    {
        "run_id": "act_abc_lang_hash_seed0",
        "dataset_repo": "fywang/calvin-task-ABC-D-lerobot",
        "output_dir": "outputs/20260603_extensions/act_abc_lang_hash_seed0",
        "chunk_size": "50",
        "n_action_steps": "50",
        "seed": "0",
        "batch_size": "8",
        "learning_rate": "0.0001",
        "steps": "50000",
        "priority": "extension",
        "notes": "hash language-conditioned ACT ablation",
    },
]
changed = False
for row in extra_rows:
    if row["run_id"] not in existing:
        rows.append(row)
        existing.add(row["run_id"])
        changed = True
if changed:
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
PY
}

checkpoint_for() {
  local run_id="$1"
  python - "$run_id" <<'PY'
import csv, sys
from pathlib import Path
run_id=sys.argv[1]
root=Path.cwd()
with open(root/'configs'/'runs.csv', newline='', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        if row['run_id'] == run_id:
            print(str(root/row['output_dir']/ 'checkpoints' / f"{int(row['steps']):06d}" / 'pretrained_model'))
            raise SystemExit(0)
raise SystemExit(f'unknown run_id: {run_id}')
PY
}

wait_for_main_extension_chain() {
  local pid_file="results/task2_extensions_chain.pid"
  if [ -f "$pid_file" ]; then
    local pid
    pid="$(tr -cd '0-9' < "$pid_file")"
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      write_status waiting null "waiting for existing task2_extensions_chain pid=${pid}"
      echo "[$(timestamp)] Waiting for existing task2_extensions_chain pid=${pid} before extra experiments."
      while kill -0 "$pid" 2>/dev/null; do
        sleep 1800
        write_status waiting null "still waiting for existing task2_extensions_chain pid=${pid}"
      done
      echo "[$(timestamp)] Existing task2_extensions_chain pid=${pid} finished."
    fi
  fi
}

eval_condition_if_needed() {
  local condition_type="$1"
  local run_id="$2"
  local ckpt="$3"
  local summary="results/calvin_eval_extensions/${run_id}/D/calvin_${run_id}_${condition_type}_D_eval_summary.json"
  if [ -f "$summary" ]; then
    echo "[$(timestamp)] skip eval ${run_id}/${condition_type}; summary exists: ${summary}"
  else
    write_status running null "evaluating ${run_id} condition=${condition_type} on CALVIN D"
    bash scripts/20_run_conditioned_eval.sh "$condition_type" "$run_id" "$ckpt" full
  fi
}

eval_unconditioned_if_needed() {
  local run_id="$1"
  local ckpt="$2"
  local summary="results/calvin_eval_extensions/${run_id}/D/calvin_${run_id}_unconditioned_D_eval_summary.json"
  if [ -f "$summary" ]; then
    echo "[$(timestamp)] skip unconditioned eval ${run_id}; summary exists: ${summary}"
  else
    write_status running null "evaluating ${run_id} unconditioned baseline on CALVIN D"
    bash scripts/24_run_unconditioned_eval.sh "$run_id" "$ckpt" full
  fi
}

train_onehot_if_needed() {
  local run_id="$1"
  CKPT_PATH="$(checkpoint_for "$run_id")"
  if [ -d "$CKPT_PATH" ]; then
    echo "[$(timestamp)] skip training ${run_id}; checkpoint exists: ${CKPT_PATH}"
  else
    write_status running null "training ${run_id}"
    bash scripts/11_task_conditioned_run.sh "$run_id"
  fi
}

train_hash_language_if_needed() {
  local run_id="$1"
  CKPT_PATH="$(checkpoint_for "$run_id")"
  if [ -d "$CKPT_PATH" ]; then
    echo "[$(timestamp)] skip training ${run_id}; checkpoint exists: ${CKPT_PATH}"
  else
    write_status running null "training hash language-conditioned ${run_id}"
    HW3_LANG_EMBED_BACKEND=hash bash scripts/18_language_conditioned_run.sh "$run_id"
  fi
}

write_status running null "starting Task 2 extra extension chain"
echo "[$(timestamp)] HW3 Task2 extra extension chain queued."
echo "Estimated extra runtime after the main extension chain: 5-7 hours on current RTX 4080."
ensure_config_rows
wait_for_main_extension_chain

base_uncond_ckpt="outputs/20260529_acts/act_abc_to_d_seed0/checkpoints/050000/pretrained_model"
taskcond_seed0_ckpt="outputs/20260601_task_conditioned/act_abc_taskcond_seed0/checkpoints/050000/pretrained_model"
eval_unconditioned_if_needed "act_abc_to_d_seed0" "$base_uncond_ckpt"
eval_condition_if_needed "zero_onehot" "act_abc_taskcond_seed0" "$taskcond_seed0_ckpt"
eval_condition_if_needed "wrong_onehot" "act_abc_taskcond_seed0" "$taskcond_seed0_ckpt"
python scripts/22_analyze_extension_results.py || true

train_onehot_if_needed "act_abc_taskcond_100k_seed0"
eval_condition_if_needed "onehot" "act_abc_taskcond_100k_seed0" "$CKPT_PATH"
python scripts/22_analyze_extension_results.py || true

train_hash_language_if_needed "act_abc_lang_hash_seed0"
eval_condition_if_needed "hash" "act_abc_lang_hash_seed0" "$CKPT_PATH"
python scripts/22_analyze_extension_results.py || true

write_status finished 0 "Task 2 extra extension chain completed"
echo "[$(timestamp)] HW3 Task2 extra extension chain finished."
