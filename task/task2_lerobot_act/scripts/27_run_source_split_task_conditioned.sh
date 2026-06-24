#!/usr/bin/env bash
set -Eeuo pipefail

if [ "$#" -ne 2 ]; then
  echo "Usage: bash scripts/27_run_source_split_task_conditioned.sh <run_id> <A|B|C|ABC>" >&2
  exit 2
fi

run_id="$1"
source_split="$2"
cd "$(dirname "$0")/.."
mkdir -p logs results outputs/20260604_final

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export HW3_SOURCE_SPLIT="${source_split}"
export HW3_SOURCE_SPLIT_MODE="${HW3_SOURCE_SPLIT_MODE:-episode_order_thirds}"

if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
elif command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate hw3_task2_lerobot
fi

cmd_file="logs/${run_id}.command.txt"
log_file="logs/${run_id}.source_split_task_conditioned.train.log"
status_file="results/${run_id}.status.json"

base_cmd="$(python scripts/04_build_train_command.py "$run_id" --print-shell)"
cmd="${base_cmd/#lerobot-train/python scripts/26_train_source_split_task_conditioned_act.py}"
printf '%s\n' "$cmd" > "$cmd_file"

cat > "$status_file" <<JSON
{
  "run_id": "${run_id}",
  "status": "started",
  "source_split": "${source_split}",
  "source_split_mode": "${HW3_SOURCE_SPLIT_MODE}",
  "task_conditioning": "task_index_one_hot_appended_to_observation_state",
  "start_time": "$(date -Iseconds)",
  "command_file": "${cmd_file}",
  "log_file": "${log_file}",
  "hf_endpoint": "${HF_ENDPOINT}"
}
JSON

echo "Starting source-split task-conditioned run ${run_id} split=${source_split}"
echo "Command: $(cat "$cmd_file")"
echo "Log: ${log_file}"

set +e
bash -lc "$(cat "$cmd_file")" 2>&1 | tee "$log_file"
exit_code="${PIPESTATUS[0]}"
set -e

final_status="finished"
if [ "$exit_code" -ne 0 ]; then
  final_status="failed"
fi

cat > "$status_file" <<JSON
{
  "run_id": "${run_id}",
  "status": "${final_status}",
  "source_split": "${source_split}",
  "source_split_mode": "${HW3_SOURCE_SPLIT_MODE}",
  "task_conditioning": "task_index_one_hot_appended_to_observation_state",
  "exit_code": ${exit_code},
  "end_time": "$(date -Iseconds)",
  "command_file": "${cmd_file}",
  "log_file": "${log_file}",
  "hf_endpoint": "${HF_ENDPOINT}"
}
JSON

exit "$exit_code"
