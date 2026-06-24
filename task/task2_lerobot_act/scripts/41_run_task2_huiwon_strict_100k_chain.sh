#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs results outputs/20260605_huiwon_strict_100k

chain_id="task2_huiwon_strict_100k_chain"
status_file="results/${chain_id}.status.json"
log_file="logs/${chain_id}.log"
dataset_root="data/huiwon_calvin_task_ABC_D_v30/huiwon/calvin_task_ABC_D"
source_dataset_root="data/huiwon_calvin_task_ABC_D/calvin_task_ABC_D_lerobot_0_4"
split_map="results/task2_huiwon_official_scene_split_map.csv"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TOKENIZERS_PARALLELISM=false
export HW3_HUIWON_SPLIT_MAP="$split_map"
export HW3_HUIWON_BATCH_SIZE="${HW3_HUIWON_BATCH_SIZE:-4}"
export HW3_HUIWON_VIDEO_BACKEND="${HW3_HUIWON_VIDEO_BACKEND:-pyav}"
export HW3_HUIWON_TOLERANCE_S="${HW3_HUIWON_TOLERANCE_S:-0.01}"

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
  "dataset_root": "${dataset_root}",
  "split_map": "${split_map}",
  "batch_size": "${HW3_HUIWON_BATCH_SIZE}",
  "log_file": "${log_file}",
  "note": "${note}"
}
JSON
}

exec > >(tee -a "$log_file") 2>&1
trap 'rc=$?; if [ "$rc" -ne 0 ]; then write_status failed "$rc" "strict 100k chain failed; inspect log_file"; fi' EXIT

source .venv/bin/activate

for required in "$dataset_root/meta/info.json" "$split_map" "data/calvin_raw/datasets/task_D_D/validation"; do
  if [ ! -e "$required" ]; then
    write_status failed 3 "missing required path: $required"
    exit 3
  fi
done

if [ ! -f results/huiwon_tasks.parquet ]; then
  write_status running null "creating huiwon task metadata"
  python scripts/40_create_huiwon_task_meta.py \
    --huiwon-tasks "${source_dataset_root}/meta/tasks.jsonl" \
    --output results/huiwon_tasks.parquet \
    --summary results/huiwon_tasks_summary.json
fi

run_train() {
  local split="$1"
  local steps="$2"
  local run_id="$3"
  local out_dir="outputs/20260605_huiwon_strict_100k/${run_id}"
  local checkpoint="${out_dir}/checkpoints/$(printf '%06d' "$steps")/pretrained_model"
  export HW3_HUIWON_SOURCE_SPLIT="$split"
  echo "[$(timestamp)] Train ${run_id} split=${split} steps=${steps} batch=${HW3_HUIWON_BATCH_SIZE}"
  if [ ! -d "$checkpoint" ]; then
    python scripts/37_train_huiwon_official_task_conditioned_act.py \
      --policy.type=act \
      --dataset.repo_id=huiwon/calvin_task_ABC_D \
      --dataset.root="$dataset_root" \
      --dataset.video_backend="${HW3_HUIWON_VIDEO_BACKEND}" \
      --output_dir="$out_dir" \
      --seed=0 \
      --steps="$steps" \
      --batch_size="${HW3_HUIWON_BATCH_SIZE}" \
      --tolerance_s="${HW3_HUIWON_TOLERANCE_S}" \
      --optimizer.lr=0.0001 \
      --policy.chunk_size=50 \
      --policy.n_action_steps=50 \
      --policy.push_to_hub=false \
      --wandb.mode=offline \
      2>&1 | tee "logs/${run_id}.huiwon_official_100k.train.log"
  else
    echo "[$(timestamp)] Checkpoint exists, skip ${run_id}: ${checkpoint}"
  fi
}

run_eval() {
  local run_id="$1"
  local checkpoint="outputs/20260605_huiwon_strict_100k/${run_id}/checkpoints/100000/pretrained_model"
  local summary="results/calvin_eval_extensions/${run_id}/D/calvin_${run_id}_onehot_D_eval_summary.json"
  echo "[$(timestamp)] Eval ${run_id}"
  if [ ! -f "$summary" ]; then
    bash scripts/39_run_huiwon_conditioned_eval.sh onehot "$run_id" "$checkpoint" full
  else
    echo "[$(timestamp)] Eval exists, skip ${run_id}: ${summary}"
  fi
}

write_status running null "training official A-only 100k"
run_train A 100000 act_huiwon_a_official_taskcond_100k_seed0

write_status running null "training official ABC-mixed 100k"
run_train ABC 100000 act_huiwon_abc_official_taskcond_100k_seed0

write_status running null "evaluating official A-only and ABC-mixed 100k on CALVIN D"
run_eval act_huiwon_a_official_taskcond_100k_seed0
run_eval act_huiwon_abc_official_taskcond_100k_seed0

write_status running null "summarizing strict 100k results"
python - <<'PY'
from pathlib import Path
import json

root = Path.cwd()
runs = [
    "act_huiwon_a_official_taskcond_100k_seed0",
    "act_huiwon_abc_official_taskcond_100k_seed0",
]
rows = {}
for run in runs:
    path = root / "results" / "calvin_eval_extensions" / run / "D" / f"calvin_{run}_onehot_D_eval_summary.json"
    rows[run] = json.loads(path.read_text()) if path.exists() else {"missing": str(path)}

payload = {
    "source": "huiwon/calvin_task_ABC_D with official CALVIN scene_info + original_frame_idx",
    "purpose": "strict official-label 100k extension for A-only vs ABC-mixed",
    "split_map": "results/task2_huiwon_official_scene_split_map.csv",
    "steps": 100000,
    "chunk_size": 50,
    "batch_size": 4,
    "eval": rows,
}
(root / "results/task2_huiwon_strict_100k_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
lines = [
    "# HW3 Task 2 Huiwon Strict 100k Summary",
    "",
    "A/B/C labels are assigned by official CALVIN `task_ABC_D` scene intervals and `original_frame_idx`.",
    "",
    "| Run | Avg Len | SR@1 | SR@2 | SR@3 | SR@4 | SR@5 |",
    "|---|---:|---:|---:|---:|---:|---:|",
]
for run, data in rows.items():
    sr = data.get("success_rates", {})
    lines.append(
        f"| {run} | {data.get('avg_successful_sequence_len')} | {sr.get('1')} | {sr.get('2')} | {sr.get('3')} | {sr.get('4')} | {sr.get('5')} |"
    )
(root / "results/task2_huiwon_strict_100k_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2, ensure_ascii=False))
PY

bash scripts/08_download_important_results.sh > results/task2_huiwon_strict_100k_latest_package_path.txt || true
write_status finished 0 "strict official A/ABC 100k chain completed"
echo "[$(timestamp)] Strict official A/ABC 100k chain finished"
