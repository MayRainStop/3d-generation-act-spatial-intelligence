#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: bash scripts/18_language_conditioned_run.sh <run_id>" >&2
  exit 2
fi

run_id="$1"
cd "$(dirname "$0")/.."
mkdir -p logs results outputs/20260603_extensions

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export SENTENCE_TRANSFORMERS_HOME="${SENTENCE_TRANSFORMERS_HOME:-${HF_HOME}/sentence_transformers}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TOKENIZERS_PARALLELISM=false
export HW3_LANG_EMBED_BACKEND="${HW3_LANG_EMBED_BACKEND:-sbert}"
export HW3_LANG_EMBED_MODEL="${HW3_LANG_EMBED_MODEL:-sentence-transformers/all-MiniLM-L6-v2}"

if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate hw3_task2_lerobot
elif [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
fi

if [ "$HW3_LANG_EMBED_BACKEND" = "sbert" ]; then
  python -m pip show sentence-transformers >/dev/null 2>&1 || python -m pip install 'sentence-transformers==3.4.1'
fi

cmd_file="logs/${run_id}.command.txt"
log_file="logs/${run_id}.language_conditioned.train.log"
status_file="results/${run_id}.status.json"

base_cmd="$(python scripts/04_build_train_command.py "$run_id" --print-shell)"
cmd="${base_cmd/#lerobot-train/python scripts/17_train_language_conditioned_act.py}"
printf '%s\n' "$cmd" > "$cmd_file"

cat > "$status_file" <<JSON
{
  "run_id": "${run_id}",
  "status": "started",
  "task_conditioning": "sentence_transformer_language_embedding_appended_to_observation_state",
  "language_backend": "${HW3_LANG_EMBED_BACKEND}",
  "language_model": "${HW3_LANG_EMBED_MODEL}",
  "start_time": "$(date -Iseconds)",
  "command_file": "${cmd_file}",
  "log_file": "${log_file}",
  "hf_endpoint": "${HF_ENDPOINT}"
}
JSON

echo "Starting language-conditioned run ${run_id}"
echo "Command: $(cat "$cmd_file")"
echo "HF_ENDPOINT: ${HF_ENDPOINT}"
echo "Language backend: ${HW3_LANG_EMBED_BACKEND} ${HW3_LANG_EMBED_MODEL}"
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
  "task_conditioning": "sentence_transformer_language_embedding_appended_to_observation_state",
  "language_backend": "${HW3_LANG_EMBED_BACKEND}",
  "language_model": "${HW3_LANG_EMBED_MODEL}",
  "exit_code": ${exit_code},
  "end_time": "$(date -Iseconds)",
  "command_file": "${cmd_file}",
  "log_file": "${log_file}",
  "hf_endpoint": "${HF_ENDPOINT}"
}
JSON

exit "$exit_code"
