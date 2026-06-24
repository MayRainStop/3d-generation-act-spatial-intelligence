#!/usr/bin/env bash
set -Eeuo pipefail

if [ "$#" -lt 3 ] || [ "$#" -gt 4 ]; then
  echo "Usage: bash scripts/45_run_ta_conditioned_eval.sh <condition_type> <model_run_id> <checkpoint> [full|debug-only]" >&2
  exit 2
fi

condition_type="$1"
model_run_id="$2"
checkpoint="$3"
mode="${4:-full}"

cd "$(dirname "$0")/.."
mkdir -p logs results results/calvin_eval_extensions

chain_id="${model_run_id}_${condition_type}_ta_eval_chain"
status_file="results/${chain_id}.status.json"
log_file="logs/${chain_id}.log"
task_meta="${HW3_TA_TASK_META:-data/xiaoma_calvin_lerobot_v30/xiaoma26/calvin_lerobot_splitA/meta/tasks.parquet}"
debug_dataset="data/calvin_raw/datasets/calvin_debug_dataset"
d_dataset="data/calvin_raw/datasets/task_D_D"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export SENTENCE_TRANSFORMERS_HOME="${SENTENCE_TRANSFORMERS_HOME:-${HF_HOME}/sentence_transformers}"
export HW3_LANG_EMBED_MODEL="${HW3_LANG_EMBED_MODEL:-sentence-transformers/all-MiniLM-L6-v2}"
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
trap 'rc=$?; if [ "$rc" -ne 0 ]; then write_status failed "$rc" "TA conditioned eval failed; inspect log_file"; fi' EXIT

. .venv/bin/activate
if [ "$condition_type" = "sbert" ]; then
  python -m pip show sentence-transformers >/dev/null 2>&1 || python -m pip install 'sentence-transformers==3.4.1'
fi
export HYDRA_FULL_ERROR=1

for required in "$checkpoint" "$task_meta" "$debug_dataset"; do
  if [ ! -e "$required" ]; then
    write_status failed 4 "missing required path: $required"
    exit 4
  fi
done

write_status started null "running debug eval before formal D eval"
echo "[$(timestamp)] TA conditioned eval started: model=${model_run_id}, condition=${condition_type}, mode=${mode}"
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

write_status running_D_eval null "running formal CALVIN D eval"
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
echo "[$(timestamp)] TA conditioned eval finished: model=${model_run_id}"
