#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: bash scripts/11_task_conditioned_run.sh <run_id>" >&2
  exit 2
fi

run_id="$1"
cd "$(dirname "$0")/.."
mkdir -p logs results outputs/20260601_task_conditioned

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export WANDB_MODE="${WANDB_MODE:-offline}"

if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate hw3_task2_lerobot
elif [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
fi

cmd_file="logs/${run_id}.command.txt"
log_file="logs/${run_id}.task_conditioned.train.log"
status_file="results/${run_id}.status.json"

base_cmd="$(python scripts/04_build_train_command.py "$run_id" --print-shell)"
cmd="${base_cmd/#lerobot-train/python scripts/10_train_task_conditioned_act.py}"
printf '%s\n' "$cmd" > "$cmd_file"

{
  echo "{"
  echo "  \"run_id\": \"${run_id}\","
  echo "  \"status\": \"started\","
  echo "  \"task_conditioning\": \"task_index_one_hot_appended_to_observation_state\","
  echo "  \"start_time\": \"$(date -Iseconds)\","
  echo "  \"command_file\": \"${cmd_file}\","
  echo "  \"log_file\": \"${log_file}\","
  echo "  \"hf_endpoint\": \"${HF_ENDPOINT}\""
  echo "}"
} > "$status_file"

echo "Starting task-conditioned run ${run_id}"
echo "Command: $(cat "$cmd_file")"
echo "HF_ENDPOINT: ${HF_ENDPOINT}"
echo "Log: ${log_file}"

set +e
bash -lc "$(cat "$cmd_file")" 2>&1 | tee "$log_file"
exit_code="${PIPESTATUS[0]}"
set -e

final_status="finished"
if [ "$exit_code" -ne 0 ]; then
  final_status="failed"
fi

{
  echo "{"
  echo "  \"run_id\": \"${run_id}\","
  echo "  \"status\": \"${final_status}\","
  echo "  \"task_conditioning\": \"task_index_one_hot_appended_to_observation_state\","
  echo "  \"exit_code\": ${exit_code},"
  echo "  \"end_time\": \"$(date -Iseconds)\","
  echo "  \"command_file\": \"${cmd_file}\","
  echo "  \"log_file\": \"${log_file}\","
  echo "  \"hf_endpoint\": \"${HF_ENDPOINT}\""
  echo "}"
} > "$status_file"

exit "$exit_code"
