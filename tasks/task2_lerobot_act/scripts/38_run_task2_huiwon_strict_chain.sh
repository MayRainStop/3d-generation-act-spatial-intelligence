#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs results outputs/20260604_huiwon_strict

chain_id="task2_huiwon_strict_chain"
status_file="results/${chain_id}.status.json"
log_file="logs/${chain_id}.log"
source_dataset_root="data/huiwon_calvin_task_ABC_D/calvin_task_ABC_D_lerobot_0_4"
v30_root_base="data/huiwon_calvin_task_ABC_D_v30"
dataset_root="${v30_root_base}/huiwon/calvin_task_ABC_D"
split_map="results/task2_huiwon_official_scene_split_map.csv"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TOKENIZERS_PARALLELISM=false
export HW3_HUIWON_SPLIT_MAP="$split_map"

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
  "source_dataset_root": "${source_dataset_root}",
  "dataset_root": "${dataset_root}",
  "split_map": "${split_map}",
  "log_file": "${log_file}",
  "note": "${note}"
}
JSON
}

exec > >(tee -a "$log_file") 2>&1
trap 'rc=$?; if [ "$rc" -ne 0 ]; then write_status failed "$rc" "huiwon strict chain failed; inspect log_file"; fi' EXIT

source .venv/bin/activate

write_status running null "downloading huiwon videos"
echo "[$(timestamp)] Step 1/8: download huiwon videos"
python scripts/36_download_huiwon_videos.py --env ABC --workers "${HW3_HUIWON_VIDEO_WORKERS:-8}"

write_status running null "converting huiwon dataset from LeRobot v2.1 to v3.0"
echo "[$(timestamp)] Step 2/8: convert huiwon dataset to LeRobot v3.0"
if python - <<'PY'
from pathlib import Path
import json
import sys

info_path = Path("data/huiwon_calvin_task_ABC_D_v30/huiwon/calvin_task_ABC_D/meta/info.json")
if info_path.exists():
    info = json.loads(info_path.read_text())
    if info.get("codebase_version") == "v3.0":
        print("huiwon_v30_already_ready")
        sys.exit(0)
sys.exit(1)
PY
then
  :
else
  mkdir -p "${v30_root_base}/huiwon"
  if [ ! -e "${dataset_root}" ]; then
    ln -s "$(pwd)/${source_dataset_root}" "${dataset_root}"
  fi
  python -m lerobot.datasets.v30.convert_dataset_v21_to_v30 \
    --repo-id=huiwon/calvin_task_ABC_D \
    --root="${v30_root_base}" \
    --push-to-hub=false \
    --force-conversion
fi

write_status running null "creating huiwon task metadata for D eval"
echo "[$(timestamp)] Step 3/8: create huiwon task metadata"
python scripts/40_create_huiwon_task_meta.py \
  --huiwon-tasks "${source_dataset_root}/meta/tasks.jsonl" \
  --output results/huiwon_tasks.parquet \
  --summary results/huiwon_tasks_summary.json

run_train() {
  local split="$1"
  local steps="$2"
  local run_id="$3"
  local out_dir="outputs/20260604_huiwon_strict/${run_id}"
  local checkpoint="${out_dir}/checkpoints/$(printf '%06d' "$steps")/pretrained_model"
  export HW3_HUIWON_SOURCE_SPLIT="$split"
  echo "[$(timestamp)] Train ${run_id} split=${split} steps=${steps}"
  if [ ! -d "$checkpoint" ]; then
    python scripts/37_train_huiwon_official_task_conditioned_act.py \
      --policy.type=act \
      --dataset.repo_id=huiwon/calvin_task_ABC_D \
      --dataset.root="$dataset_root" \
      --dataset.video_backend="${HW3_HUIWON_VIDEO_BACKEND:-pyav}" \
      --output_dir="$out_dir" \
      --seed=0 \
      --steps="$steps" \
      --batch_size="${HW3_HUIWON_BATCH_SIZE:-4}" \
      --tolerance_s="${HW3_HUIWON_TOLERANCE_S:-0.01}" \
      --optimizer.lr=0.0001 \
      --policy.chunk_size=50 \
      --policy.n_action_steps=50 \
      --policy.push_to_hub=false \
      --wandb.mode=offline \
      2>&1 | tee "logs/${run_id}.huiwon_official.train.log"
  else
    echo "[$(timestamp)] Checkpoint exists, skip ${run_id}: ${checkpoint}"
  fi
}

write_status running null "smoke training huiwon official A"
run_train A 200 huiwon_a_official_smoke_seed0

write_status running null "training huiwon official A-only"
run_train A 50000 act_huiwon_a_official_taskcond_seed0

write_status running null "training huiwon official ABC-mixed"
run_train ABC 50000 act_huiwon_abc_official_taskcond_seed0

run_eval() {
  local run_id="$1"
  local checkpoint="outputs/20260604_huiwon_strict/${run_id}/checkpoints/050000/pretrained_model"
  local summary="results/calvin_eval_extensions/${run_id}/D/calvin_${run_id}_onehot_D_eval_summary.json"
  if [ ! -f "$summary" ]; then
    bash scripts/39_run_huiwon_conditioned_eval.sh onehot "$run_id" "$checkpoint" full
  else
    echo "[$(timestamp)] Eval exists, skip ${run_id}: ${summary}"
  fi
}

write_status running null "evaluating huiwon official A and ABC on CALVIN D"
run_eval act_huiwon_a_official_taskcond_seed0
run_eval act_huiwon_abc_official_taskcond_seed0

write_status running null "summarizing huiwon strict results"
python scripts/22_analyze_extension_results.py || true
python - <<'PY'
from pathlib import Path
import json

root = Path.cwd()
rows = {}
for run in ["act_huiwon_a_official_taskcond_seed0", "act_huiwon_abc_official_taskcond_seed0"]:
    path = root / "results" / "calvin_eval_extensions" / run / "D" / f"calvin_{run}_onehot_D_eval_summary.json"
    rows[run] = json.loads(path.read_text()) if path.exists() else {"missing": str(path)}
payload = {
    "source": "huiwon/calvin_task_ABC_D with official CALVIN scene_info + original_frame_idx",
    "split_map": "results/task2_huiwon_official_scene_split_map.csv",
    "selection_A": "results/task2_huiwon_A_selection_summary.json",
    "selection_ABC": "results/task2_huiwon_ABC_selection_summary.json",
    "eval": rows,
}
(root / "results/task2_huiwon_strict_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
lines = ["# HW3 Task 2 Huiwon Strict Summary", "", "A/B/C labels are assigned by official CALVIN `task_ABC_D` scene intervals and `original_frame_idx`.", "", "| Run | Avg Len | SR@1 | SR@2 | SR@3 | SR@4 | SR@5 |", "|---|---:|---:|---:|---:|---:|---:|"]
for run, data in rows.items():
    lines.append(f"| {run} | {data.get('avg_successful_sequence_len')} | {data.get('success_len1')} | {data.get('success_len2')} | {data.get('success_len3')} | {data.get('success_len4')} | {data.get('success_len5')} |")
(root / "results/task2_huiwon_strict_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2, ensure_ascii=False))
PY

bash scripts/08_download_important_results.sh > results/task2_huiwon_strict_latest_package_path.txt || true
write_status finished 0 "huiwon strict A/ABC chain completed"
echo "[$(timestamp)] Huiwon strict chain finished"
