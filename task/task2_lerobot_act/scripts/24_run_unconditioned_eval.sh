#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
  echo "Usage: bash scripts/24_run_unconditioned_eval.sh <model_run_id> <checkpoint> [full|debug-only]" >&2
  exit 2
fi

model_run_id="$1"
checkpoint="$2"
mode="${3:-full}"
cd "$(dirname "$0")/.."
mkdir -p logs results results/calvin_eval_extensions
chain_id="${model_run_id}_unconditioned_eval_chain"
status_file="results/${chain_id}.status.json"
log_file="logs/${chain_id}.log"
debug_dataset="data/calvin_raw/datasets/calvin_debug_dataset"
d_dataset="data/calvin_raw/datasets/task_D_D"
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
  "mode": "${mode}",
  "condition_type": "unconditioned",
  "model_run_id": "${model_run_id}",
  "checkpoint": "${checkpoint}",
  "debug_dataset": "${debug_dataset}",
  "d_dataset": "${d_dataset}",
  "log_file": "${log_file}",
  "note": "${note}"
}
JSON
}
exec > >(tee -a "$log_file") 2>&1
trap 'rc=$?; if [ "$rc" -ne 0 ]; then write_status failed "$rc" "unconditioned eval chain failed; inspect log_file"; fi' EXIT
. .venv/bin/activate
export HYDRA_FULL_ERROR=1
if [ ! -d "$checkpoint" ]; then
  write_status failed 4 "checkpoint directory missing"
  exit 4
fi
write_status started null "running debug eval before formal D eval"
echo "[$(timestamp)] Unconditioned eval chain started: model=${model_run_id}, mode=${mode}"
python scripts/23_calvin_eval_unconditioned.py \
  --dataset-path "$debug_dataset" \
  --checkpoint "$checkpoint" \
  --run-id "calvin_${model_run_id}_unconditioned_debug_eval" \
  --num-sequences 2 \
  --sequence-workers 1 \
  --ep-len 80 \
  --eval-log-dir "results/calvin_eval_extensions/${model_run_id}/debug" \
  --status-file "results/calvin_${model_run_id}_unconditioned_debug_eval.status.json"
if [ "$mode" = "debug-only" ]; then
  write_status finished 0 "debug-only eval finished"
  exit 0
fi
if [ ! -d "${d_dataset}/validation" ]; then
  write_status waiting_for_D null "debug eval passed; waiting for D dataset"
  while [ ! -d "${d_dataset}/validation" ]; do sleep 300; done
fi
write_status running_D_eval null "running formal CALVIN D eval"
python scripts/23_calvin_eval_unconditioned.py \
  --dataset-path "$d_dataset" \
  --checkpoint "$checkpoint" \
  --run-id "calvin_${model_run_id}_unconditioned_D_eval" \
  --num-sequences 1000 \
  --sequence-workers 1 \
  --ep-len 360 \
  --eval-log-dir "results/calvin_eval_extensions/${model_run_id}/D" \
  --status-file "results/calvin_${model_run_id}_unconditioned_D_eval.status.json"
write_status finished 0 "debug and formal D eval finished"
echo "[$(timestamp)] Unconditioned eval chain finished: model=${model_run_id}"
