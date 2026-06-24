#!/usr/bin/env bash
set -euo pipefail

mode="${1:-full}"
cd "$(dirname "$0")/.."
mkdir -p logs results results/calvin_eval_taskcond

run_id="calvin_taskcond_eval_chain"
status_file="results/${run_id}.status.json"
log_file="logs/${run_id}.log"
checkpoint="outputs/20260601_task_conditioned/act_abc_taskcond_seed0/checkpoints/050000/pretrained_model"
task_meta="${HOME}/.cache/huggingface/lerobot/fywang/calvin-task-ABC-D-lerobot/meta/tasks.parquet"
debug_dataset="data/calvin_raw/datasets/calvin_debug_dataset"
d_dataset="data/calvin_raw/datasets/task_D_D"

timestamp() {
  date -Iseconds
}

write_status() {
  local status="$1"
  local exit_code="${2:-null}"
  local note="${3:-}"
  cat > "$status_file" <<JSON
{
  "run_id": "${run_id}",
  "status": "${status}",
  "exit_code": ${exit_code},
  "updated_at": "$(timestamp)",
  "mode": "${mode}",
  "checkpoint": "${checkpoint}",
  "debug_dataset": "${debug_dataset}",
  "d_dataset": "${d_dataset}",
  "log_file": "${log_file}",
  "note": "${note}"
}
JSON
}

exec > >(tee -a "$log_file") 2>&1
trap 'rc=$?; if [ "$rc" -ne 0 ]; then write_status failed "$rc" "CALVIN eval chain failed; inspect log_file"; fi' EXIT

. .venv/bin/activate
export HYDRA_FULL_ERROR=1

write_status started null "running short debug eval before formal D eval"
echo "[$(timestamp)] CALVIN task-conditioned ACT eval chain started, mode=${mode}"

echo "[$(timestamp)] Step 1: debug online eval smoke"
python scripts/15_calvin_eval_taskcond.py \
  --dataset-path "$debug_dataset" \
  --checkpoint "$checkpoint" \
  --task-meta "$task_meta" \
  --run-id "calvin_taskcond_debug_eval" \
  --num-sequences 2 \
  --sequence-workers 1 \
  --ep-len 80 \
  --eval-log-dir "results/calvin_eval_taskcond/debug" \
  --status-file "results/calvin_taskcond_debug_eval.status.json"

if [ "$mode" = "debug-only" ]; then
  write_status finished 0 "debug-only eval finished"
  echo "[$(timestamp)] Debug-only mode finished."
  exit 0
fi

if [ ! -d "${d_dataset}/validation" ]; then
  write_status waiting_for_D null "debug eval passed; waiting for CALVIN D dataset"
  echo "[$(timestamp)] ${d_dataset}/validation is not ready yet; waiting."
  while [ ! -d "${d_dataset}/validation" ]; do
    sleep 300
  done
fi

write_status running_D_eval null "running formal CALVIN D online eval"
echo "[$(timestamp)] Step 2: formal D online eval; estimated several hours to >10h depending on env-step speed."
python scripts/15_calvin_eval_taskcond.py \
  --dataset-path "$d_dataset" \
  --checkpoint "$checkpoint" \
  --task-meta "$task_meta" \
  --run-id "calvin_taskcond_D_eval" \
  --num-sequences 1000 \
  --sequence-workers 1 \
  --ep-len 360 \
  --eval-log-dir "results/calvin_eval_taskcond/D" \
  --status-file "results/calvin_taskcond_D_eval.status.json"

write_status finished 0 "debug and formal D eval finished"
echo "[$(timestamp)] CALVIN task-conditioned ACT eval chain finished."
