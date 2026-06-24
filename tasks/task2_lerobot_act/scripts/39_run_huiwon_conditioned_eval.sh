#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ] || [ "$#" -gt 4 ]; then
  echo "Usage: bash scripts/39_run_huiwon_conditioned_eval.sh <condition_type> <model_run_id> <checkpoint> [full|debug-only]" >&2
  exit 2
fi

condition_type="$1"
model_run_id="$2"
checkpoint="$3"
mode="${4:-full}"
cd "$(dirname "$0")/.."
mkdir -p logs results results/calvin_eval_extensions
chain_id="${model_run_id}_${condition_type}_huiwon_eval_chain"
status_file="results/${chain_id}.status.json"
log_file="logs/${chain_id}.log"
task_meta="results/huiwon_tasks.parquet"
debug_dataset="data/calvin_raw/datasets/calvin_debug_dataset"
d_dataset="data/calvin_raw/datasets/task_D_D"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false

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
  "condition_type": "${condition_type}",
  "model_run_id": "${model_run_id}",
  "checkpoint": "${checkpoint}",
  "task_meta": "${task_meta}",
  "debug_dataset": "${debug_dataset}",
  "d_dataset": "${d_dataset}",
  "log_file": "${log_file}",
  "note": "${note}"
}
JSON
}

exec > >(tee -a "$log_file") 2>&1
trap 'rc=$?; if [ "$rc" -ne 0 ]; then write_status failed "$rc" "huiwon conditioned eval failed; inspect log_file"; fi' EXIT

. .venv/bin/activate
export HYDRA_FULL_ERROR=1
if [ ! -d "$checkpoint" ]; then
  write_status failed 4 "checkpoint directory missing"
  exit 4
fi
if [ ! -f "$task_meta" ]; then
  python scripts/40_create_huiwon_task_meta.py --output "$task_meta"
fi

write_status started null "running huiwon task-meta debug eval before formal D eval"
echo "[$(timestamp)] Huiwon conditioned eval started: model=${model_run_id}, condition=${condition_type}, mode=${mode}"
python scripts/19_calvin_eval_conditioned.py \
  --dataset-path "$debug_dataset" \
  --checkpoint "$checkpoint" \
  --task-meta "$task_meta" \
  --condition-type "$condition_type" \
  --run-id "calvin_${model_run_id}_${condition_type}_debug_eval" \
  --num-sequences 2 \
  --sequence-workers 1 \
  --ep-len 80 \
  --eval-log-dir "results/calvin_eval_extensions/${model_run_id}/debug" \
  --status-file "results/calvin_${model_run_id}_${condition_type}_debug_eval.status.json"
if [ "$mode" = "debug-only" ]; then
  write_status finished 0 "debug-only eval finished"
  exit 0
fi
if [ ! -d "${d_dataset}/validation" ]; then
  write_status waiting_for_D null "debug eval passed; waiting for D dataset"
  while [ ! -d "${d_dataset}/validation" ]; do sleep 300; done
fi
write_status running_D_eval null "running formal CALVIN D eval with huiwon task-meta"
python scripts/19_calvin_eval_conditioned.py \
  --dataset-path "$d_dataset" \
  --checkpoint "$checkpoint" \
  --task-meta "$task_meta" \
  --condition-type "$condition_type" \
  --run-id "calvin_${model_run_id}_${condition_type}_D_eval" \
  --num-sequences 1000 \
  --sequence-workers 1 \
  --ep-len 360 \
  --eval-log-dir "results/calvin_eval_extensions/${model_run_id}/D" \
  --status-file "results/calvin_${model_run_id}_${condition_type}_D_eval.status.json"
write_status finished 0 "debug and formal D eval finished"
echo "[$(timestamp)] Huiwon conditioned eval finished: model=${model_run_id}"
