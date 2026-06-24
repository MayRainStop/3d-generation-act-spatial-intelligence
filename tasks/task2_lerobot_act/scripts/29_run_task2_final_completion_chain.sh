#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs results outputs/20260604_final
chain_id="task2_final_completion_chain"
status_file="results/${chain_id}.status.json"
log_file="logs/${chain_id}.log"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export WANDB_MODE="${WANDB_MODE:-offline}"
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
  "log_file": "${log_file}",
  "note": "${note}"
}
JSON
}

exec > >(tee -a "$log_file") 2>&1
trap 'rc=$?; if [ "$rc" -ne 0 ]; then write_status failed "$rc" "final completion chain failed; inspect log_file"; fi' EXIT

if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
fi

write_status running null "source split analysis and A/B/C policy comparisons running"
echo "[$(timestamp)] Task2 final completion chain started"
echo "[$(timestamp)] Running source split visual/action analysis before training"
python scripts/28_analyze_source_split_and_shift.py || true

run_one() {
  local split="$1"
  local lower
  lower="$(printf '%s' "$split" | tr 'A-Z' 'a-z')"
  local run_id="act_${lower}_proxy_taskcond_seed0"
  local checkpoint="outputs/20260604_final/${run_id}/checkpoints/050000/pretrained_model"
  local eval_summary="results/calvin_eval_extensions/${run_id}/D/calvin_${run_id}_onehot_D_eval_summary.json"

  echo "[$(timestamp)] Considering ${run_id} split=${split}"
  if [ ! -d "$checkpoint" ]; then
    write_status running null "training ${run_id}"
    bash scripts/27_run_source_split_task_conditioned.sh "$run_id" "$split"
  else
    echo "[$(timestamp)] Checkpoint exists; skip training ${run_id}: ${checkpoint}"
  fi

  if [ ! -d "$checkpoint" ]; then
    echo "[$(timestamp)] Missing checkpoint after train: ${checkpoint}" >&2
    return 5
  fi

  if [ ! -f "$eval_summary" ]; then
    write_status running null "CALVIN D eval ${run_id}"
    bash scripts/20_run_conditioned_eval.sh onehot "$run_id" "$checkpoint" full
  else
    echo "[$(timestamp)] Eval summary exists; skip eval ${run_id}: ${eval_summary}"
  fi

  python scripts/22_analyze_extension_results.py
  python scripts/28_analyze_source_split_and_shift.py || true
}

run_one A
run_one B
run_one C

python scripts/22_analyze_extension_results.py
python scripts/28_analyze_source_split_and_shift.py
bash scripts/08_download_important_results.sh > results/task2_latest_package_path.txt

python - <<'INNERPY'
from pathlib import Path
root=Path.cwd()
section = """
## 2026-06-04 Final Completion Experiments

To cover the assignment request to compare training on A, B, C, and ABC before zero-shot evaluation on D, the final queue adds source-domain proxy experiments. The public converted LeRobot dataset available here is `fywang/calvin-task-ABC-D-lerobot`; it does not expose raw A/B/C environment labels, so A/B/C are deterministic episode-order thirds of the ABC dataset. The report artifacts explicitly label this as a proxy split and include visual-distribution statistics against raw CALVIN D.

Additional runs:

| Run ID | Train split | Condition | Steps | Evaluation |
| --- | --- | --- | ---: | --- |
| `act_a_proxy_taskcond_seed0` | ABC episode third A | task one-hot | 50000 | CALVIN D online SR |
| `act_b_proxy_taskcond_seed0` | ABC episode third B | task one-hot | 50000 | CALVIN D online SR |
| `act_c_proxy_taskcond_seed0` | ABC episode third C | task one-hot | 50000 | CALVIN D online SR |

Final artifacts:

- `results/task2_source_split_map.csv/json`: exact episode/frame counts for A/B/C/ABC proxy splits.
- `results/task2_source_split_visual_action_analysis.csv/json`: source-to-D visual RGB shift and action smoothness.
- `results/task2_final_completion_summary.md`: report-ready table combining the source split analysis and CALVIN D eval summaries.
- `results/task2_extension_eval_summary.csv`: all CALVIN D online evaluation results, including the new A/B/C proxy policies.
- `results/package/`: timestamped curated archive with README, configs, status JSON, CSV/JSON summaries, and log tails.
"""
readme=root/'README.md'
text=readme.read_text(encoding='utf-8', errors='replace') if readme.exists() else ''
if '2026-06-04 Final Completion Experiments' not in text:
    readme.write_text(text.rstrip()+section+'\n', encoding='utf-8')
INNERPY

write_status finished 0 "Task 2 final completion chain completed"
echo "[$(timestamp)] Task2 final completion chain finished"
