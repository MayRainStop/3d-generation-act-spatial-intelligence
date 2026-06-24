#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs results outputs/20260603_extensions
chain_id="task2_extensions_chain"
status_file="results/${chain_id}.status.json"
log_file="logs/${chain_id}.log"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export SENTENCE_TRANSFORMERS_HOME="${SENTENCE_TRANSFORMERS_HOME:-${HF_HOME}/sentence_transformers}"
export HW3_LANG_EMBED_BACKEND="${HW3_LANG_EMBED_BACKEND:-sbert}"
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
trap 'rc=$?; if [ "$rc" -ne 0 ]; then write_status failed "$rc" "extension chain failed; inspect log_file"; fi' EXIT
. .venv/bin/activate
export HYDRA_FULL_ERROR=1
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
train_onehot() {
  local run_id="$1"
  CKPT_PATH="$(checkpoint_for "$run_id")"
  if [ -d "$CKPT_PATH" ]; then
    echo "[$(timestamp)] skip training ${run_id}; checkpoint exists: ${CKPT_PATH}"
  else
    write_status running null "training ${run_id}"
    bash scripts/11_task_conditioned_run.sh "$run_id"
  fi
}
train_language() {
  local run_id="$1"
  CKPT_PATH="$(checkpoint_for "$run_id")"
  if [ -d "$CKPT_PATH" ]; then
    echo "[$(timestamp)] skip training ${run_id}; checkpoint exists: ${CKPT_PATH}"
  else
    write_status running null "training language-conditioned ${run_id}"
    bash scripts/18_language_conditioned_run.sh "$run_id"
  fi
}
eval_if_needed() {
  local condition_type="$1"
  local run_id="$2"
  local ckpt="$3"
  local summary="results/calvin_eval_extensions/${run_id}/D/calvin_${run_id}_${condition_type}_D_eval_summary.json"
  if [ -f "$summary" ]; then
    echo "[$(timestamp)] skip eval ${run_id}; summary exists: ${summary}"
  else
    write_status running null "evaluating ${run_id} condition=${condition_type} on CALVIN D"
    bash scripts/20_run_conditioned_eval.sh "$condition_type" "$run_id" "$ckpt" full
  fi
}
write_status running null "starting Task 2 extension chain"
echo "[$(timestamp)] HW3 Task2 extension chain started."
echo "Estimated total runtime: 10-16 hours on current RTX 4080 with qwen_server occupying part of VRAM."
for run_id in act_abc_taskcond_chunk10_seed0 act_abc_taskcond_chunk100_seed0 act_abc_taskcond_seed1; do
  train_onehot "$run_id"
  eval_if_needed onehot "$run_id" "$CKPT_PATH"
  python scripts/22_analyze_extension_results.py || true
done
train_language lang_sbert_debug_smoke_seed0
bash scripts/20_run_conditioned_eval.sh sbert lang_sbert_debug_smoke_seed0 "$CKPT_PATH" debug-only || true
train_language act_abc_lang_sbert_seed0
eval_if_needed sbert act_abc_lang_sbert_seed0 "$CKPT_PATH"
python scripts/22_analyze_extension_results.py || true
write_status finished 0 "Task 2 extension chain completed"
echo "[$(timestamp)] HW3 Task2 extension chain finished."
