#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs results outputs/20260604_strict

chain_id="task2_strict_a_chain"
status_file="results/${chain_id}.status.json"
log_file="logs/${chain_id}.log"
scene_info_json="results/official_task_ABC_D_scene_info.json"
split_map_csv="results/task2_official_scene_split_map.csv"
run_id="act_a_official_taskcond_seed0"
checkpoint="outputs/20260604_strict/${run_id}/checkpoints/050000/pretrained_model"
eval_summary="results/calvin_eval_extensions/${run_id}/D/calvin_${run_id}_onehot_D_eval_summary.json"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TOKENIZERS_PARALLELISM=false
export HW3_OFFICIAL_SOURCE_SPLIT="A"
export HW3_OFFICIAL_SCENE_SPLIT_MAP="${split_map_csv}"

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
  "strict_run_id": "${run_id}",
  "scene_info_json": "${scene_info_json}",
  "split_map_csv": "${split_map_csv}",
  "checkpoint": "${checkpoint}",
  "eval_summary": "${eval_summary}",
  "log_file": "${log_file}",
  "note": "${note}"
}
JSON
}

exec > >(tee -a "$log_file") 2>&1
trap 'rc=$?; if [ "$rc" -ne 0 ]; then write_status failed "$rc" "strict A chain failed; inspect log_file"; fi' EXIT

if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
fi

write_status running null "extracting official task_ABC_D scene_info"
echo "[$(timestamp)] Strict A chain started"
echo "[$(timestamp)] Step 1/5: official task_ABC_D scene_info extraction"
if [ ! -s "$scene_info_json" ]; then
  python -u scripts/31_extract_remote_zip_scene_info.py \
    http://calvin.cs.uni-freiburg.de/dataset/task_ABC_D.zip \
    "$scene_info_json"
else
  echo "[$(timestamp)] scene_info already exists: ${scene_info_json}"
fi

write_status running null "building official scene split map"
echo "[$(timestamp)] Step 2/5: build official A/B/C episode map"
python scripts/32_build_official_scene_split_map.py \
  --dataset-root "${HF_HOME}/lerobot/fywang/calvin-task-ABC-D-lerobot" \
  --scene-info-json "$scene_info_json" \
  --out-csv "$split_map_csv" \
  --out-json "results/task2_official_scene_split_map.json" \
  --a-episodes-out "results/task2_official_A_episode_indices.txt"

write_status running null "training strict official A-only ACT"
echo "[$(timestamp)] Step 3/5: train strict official A-only task-conditioned ACT"
if [ ! -d "$checkpoint" ]; then
  cmd=(
    python scripts/33_train_official_scene_task_conditioned_act.py
    --policy.type=act
    --dataset.repo_id=fywang/calvin-task-ABC-D-lerobot
    --output_dir="outputs/20260604_strict/${run_id}"
    --seed=0
    --steps=50000
    --batch_size=8
    --optimizer.lr=0.0001
    --policy.chunk_size=50
    --policy.n_action_steps=50
    --policy.push_to_hub=false
    --wandb.mode=offline
  )
  printf '%q ' "${cmd[@]}" > "logs/${run_id}.command.txt"
  printf '\n' >> "logs/${run_id}.command.txt"
  "${cmd[@]}" 2>&1 | tee "logs/${run_id}.official_scene_task_conditioned.train.log"
else
  echo "[$(timestamp)] Checkpoint exists; skip training: ${checkpoint}"
fi

if [ ! -d "$checkpoint" ]; then
  echo "[$(timestamp)] Missing checkpoint after training: ${checkpoint}" >&2
  exit 5
fi

write_status running null "running CALVIN D eval for strict official A-only"
echo "[$(timestamp)] Step 4/5: CALVIN D eval"
if [ ! -f "$eval_summary" ]; then
  bash scripts/20_run_conditioned_eval.sh onehot "$run_id" "$checkpoint" full
else
  echo "[$(timestamp)] Eval summary exists; skip eval: ${eval_summary}"
fi

write_status running null "refreshing summaries and package"
echo "[$(timestamp)] Step 5/5: refresh summaries"
python scripts/22_analyze_extension_results.py || true
python scripts/28_analyze_source_split_and_shift.py || true

python - <<'PY'
from pathlib import Path
import csv, json

root = Path.cwd()
results = root / "results"
strict_eval = results / "calvin_eval_extensions/act_a_official_taskcond_seed0/D/calvin_act_a_official_taskcond_seed0_onehot_D_eval_summary.json"
payload = {
    "strict_source": "official CALVIN task_ABC_D training/scene_info.npy",
    "split_map": str(results / "task2_official_scene_split_map.csv"),
    "selection_summary": str(results / "task2_official_A_selection_summary.json"),
    "eval_summary": str(strict_eval),
}
if strict_eval.exists():
    payload["eval"] = json.loads(strict_eval.read_text())
out = results / "task2_strict_a_summary.json"
out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
md = results / "task2_strict_a_summary.md"
lines = [
    "# HW3 Task 2 Strict A-only Summary",
    "",
    "A-only data is selected using the official CALVIN `task_ABC_D/training/scene_info.npy` scene intervals extracted from the official zip by HTTP Range requests.",
    "",
]
if "eval" in payload:
    e = payload["eval"]
    lines += [
        "| Run | Avg Len | SR@1 | SR@2 | SR@3 | SR@4 | SR@5 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| act_a_official_taskcond_seed0 | {e.get('avg_successful_sequence_len')} | {e.get('success_len1')} | {e.get('success_len2')} | {e.get('success_len3')} | {e.get('success_len4')} | {e.get('success_len5')} |",
    ]
md.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2, ensure_ascii=False))
PY

bash scripts/08_download_important_results.sh > results/task2_strict_latest_package_path.txt || true

write_status finished 0 "strict official A-only chain completed"
echo "[$(timestamp)] Strict A chain finished"
