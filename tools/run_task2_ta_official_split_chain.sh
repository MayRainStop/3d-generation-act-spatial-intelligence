#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs results outputs/20260606_ta_official_split data

chain_id="task2_ta_official_split_chain"
status_file="results/${chain_id}.status.json"
log_file="logs/${chain_id}.log"
source_root="data/xiaoma_calvin_lerobot"
v30_root="data/xiaoma_calvin_lerobot_v30"
out_root="outputs/20260606_ta_official_split"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TOKENIZERS_PARALLELISM=false
export SENTENCE_TRANSFORMERS_HOME="${SENTENCE_TRANSFORMERS_HOME:-${HF_HOME}/sentence_transformers}"
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
  "source_root": "${source_root}",
  "v30_root": "${v30_root}",
  "out_root": "${out_root}",
  "log_file": "${log_file}",
  "note": "${note}"
}
JSON
}

exec > >(tee -a "$log_file") 2>&1
trap 'rc=$?; if [ "$rc" -ne 0 ]; then write_status failed "$rc" "course split chain failed; inspect log_file"; fi' EXIT

. .venv/bin/activate
export HYDRA_FULL_ERROR=1

write_audit() {
  python - <<'PY'
from pathlib import Path
import json

root = Path("data/xiaoma_calvin_lerobot")
rows = []
for label in ["A", "B", "C", "D"]:
    info_path = root / f"split{label}" / "meta" / "info.json"
    if not info_path.exists():
        rows.append({"split": label, "missing": str(info_path)})
        continue
    info = json.loads(info_path.read_text())
    rows.append({
        "split": label,
        "scene": info.get("scene"),
        "codebase_version": info.get("codebase_version"),
        "total_episodes": info.get("total_episodes"),
        "total_frames": info.get("total_frames"),
        "total_videos": info.get("total_videos"),
        "fps": info.get("fps"),
    })
payload = {
    "source": "TA-provided official split dataset xiaoma26/calvin-lerobot",
    "purpose": "replace proxy A/B/C split with explicit splitA/splitB/splitC/splitD scenes",
    "rows": rows,
}
Path("results/task2_ta_official_split_audit.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
lines = [
    "# HW3 Task 2 Course Split Audit",
    "",
    "Dataset: xiaoma26/calvin-lerobot. Each split has an explicit `scene` in meta/info.json.",
    "",
    "| Split | Scene | Version | Episodes | Frames | Videos | FPS |",
    "|---|---:|---:|---:|---:|---:|---:|",
]
for row in rows:
    lines.append(f"| {row.get('split')} | {row.get('scene')} | {row.get('codebase_version')} | {row.get('total_episodes')} | {row.get('total_frames')} | {row.get('total_videos')} | {row.get('fps')} |")
Path("results/task2_ta_official_split_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2, ensure_ascii=False))
PY
}

download_ta_splits() {
  write_status running null "downloading course split dataset"
  echo "[$(timestamp)] Download xiaoma26/calvin-lerobot splitA/B/C/D"
  python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="xiaoma26/calvin-lerobot",
    repo_type="dataset",
    local_dir="data/xiaoma_calvin_lerobot",
    local_dir_use_symlinks=False,
    resume_download=True,
    allow_patterns=["splitA/**", "splitB/**", "splitC/**", "splitD/**", ".gitattributes"],
)
PY
  write_audit
}

convert_split() {
  local label="$1"
  local source_dir="${source_root}/split${label}"
  local repo_leaf="calvin_lerobot_split${label}"
  local target_dir="${v30_root}/xiaoma26/${repo_leaf}"
  local repo_id="xiaoma26/${repo_leaf}"
  if python - "$target_dir/meta/info.json" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
if path.exists() and json.loads(path.read_text()).get("codebase_version") == "v3.0":
    sys.exit(0)
sys.exit(1)
PY
  then
    echo "[$(timestamp)] split${label} already converted: ${target_dir}"
    return 0
  fi
  write_status running null "converting split${label} from LeRobot v2.1 to v3.0"
  mkdir -p "${v30_root}/xiaoma26"
  if [ ! -e "$target_dir" ]; then
    ln -s "$(pwd)/${source_dir}" "$target_dir"
  fi
  python -m lerobot.datasets.v30.convert_dataset_v21_to_v30 \
    --repo-id="${repo_id}" \
    --root="${v30_root}" \
    --push-to-hub=false \
    --force-conversion
}

checkpoint_path() {
  local run_id="$1"
  local step="$2"
  printf "%s/%s/checkpoints/%06d/pretrained_model" "$out_root" "$run_id" "$step"
}

train_model() {
  local split="$1"
  local condition="$2"
  local run_id="$3"
  local steps="$4"
  local chunk_size="$5"
  local batch_size="${HW3_TA_BATCH_SIZE:-8}"
  local checkpoint
  checkpoint="$(checkpoint_path "$run_id" "$steps")"
  if [ -d "$checkpoint" ]; then
    echo "[$(timestamp)] checkpoint exists, skip training ${run_id}: ${checkpoint}"
    return 0
  fi
  write_status running null "training ${run_id} split=${split} condition=${condition}"
  echo "[$(timestamp)] Train ${run_id}: split=${split}, condition=${condition}, steps=${steps}, chunk=${chunk_size}, batch=${batch_size}"
  export HW3_TA_SOURCE_SPLIT="$split"
  export HW3_TA_CONDITION_TYPE="$condition"
  export HW3_LANG_EMBED_BACKEND="sbert"
  local train_log="logs/${run_id}.ta_official.train.log"
  set +e
  python scripts/43_train_ta_official_split_conditioned_act.py \
    --policy.type=act \
    --dataset.repo_id=xiaoma26/calvin_lerobot_splitA \
    --dataset.root="${v30_root}/xiaoma26/calvin_lerobot_splitA" \
    --output_dir="${out_root}/${run_id}" \
    --seed=0 \
    --steps="${steps}" \
    --batch_size="${batch_size}" \
    --save_freq=50000 \
    --log_freq=200 \
    --eval_freq=0 \
    --tolerance_s="${HW3_TA_TOLERANCE_S:-0.01}" \
    --optimizer.lr=0.0001 \
    --policy.chunk_size="${chunk_size}" \
    --policy.n_action_steps="${chunk_size}" \
    --policy.push_to_hub=false \
    --wandb.mode=offline \
    2>&1 | tee "$train_log"
  local rc="${PIPESTATUS[0]}"
  set -e
  if [ "$rc" -ne 0 ] && grep -Eiq "out of memory|cuda.*memory|cublas.*alloc" "$train_log"; then
    echo "[$(timestamp)] OOM detected for ${run_id}; retrying with batch_size=4"
    set +e
    python scripts/43_train_ta_official_split_conditioned_act.py \
      --policy.type=act \
      --dataset.repo_id=xiaoma26/calvin_lerobot_splitA \
      --dataset.root="${v30_root}/xiaoma26/calvin_lerobot_splitA" \
      --output_dir="${out_root}/${run_id}" \
      --seed=0 \
      --steps="${steps}" \
      --batch_size=4 \
      --save_freq=50000 \
      --log_freq=200 \
      --eval_freq=0 \
      --tolerance_s="${HW3_TA_TOLERANCE_S:-0.01}" \
      --optimizer.lr=0.0001 \
      --policy.chunk_size="${chunk_size}" \
      --policy.n_action_steps="${chunk_size}" \
      --policy.push_to_hub=false \
      --wandb.mode=offline \
      2>&1 | tee -a "$train_log"
    rc="${PIPESTATUS[0]}"
    set -e
  fi
  if [ "$rc" -ne 0 ]; then
    echo "Training failed for ${run_id}; see ${train_log}" >&2
    exit "$rc"
  fi
}

eval_model() {
  local condition="$1"
  local run_id="$2"
  local step="$3"
  local eval_run="$4"
  local checkpoint
  checkpoint="$(checkpoint_path "$run_id" "$step")"
  local summary="results/calvin_eval_extensions/${eval_run}/D/calvin_${eval_run}_${condition}_D_eval_summary.json"
  if [ ! -d "$checkpoint" ]; then
    echo "[$(timestamp)] missing checkpoint for eval, skip ${eval_run}: ${checkpoint}"
    return 0
  fi
  if [ -f "$summary" ]; then
    echo "[$(timestamp)] summary exists, skip eval ${eval_run}: ${summary}"
    return 0
  fi
  write_status running null "evaluating ${eval_run} on CALVIN D"
  bash scripts/45_run_ta_conditioned_eval.sh "$condition" "$eval_run" "$checkpoint" full
}

summarize_ta() {
  write_status running null "summarizing course split results"
  python - <<'PY'
from pathlib import Path
import csv
import json

root = Path.cwd()
candidates = [
    ("ta_A_taskcond_50k", "act_ta_a_taskcond_100k_seed0_ckpt050k", "onehot"),
    ("ta_A_taskcond_100k", "act_ta_a_taskcond_100k_seed0", "onehot"),
    ("ta_B_taskcond_50k", "act_ta_b_taskcond_100k_seed0_ckpt050k", "onehot"),
    ("ta_B_taskcond_100k", "act_ta_b_taskcond_100k_seed0", "onehot"),
    ("ta_C_taskcond_50k", "act_ta_c_taskcond_100k_seed0_ckpt050k", "onehot"),
    ("ta_C_taskcond_100k", "act_ta_c_taskcond_100k_seed0", "onehot"),
    ("ta_ABC_taskcond_50k", "act_ta_abc_taskcond_100k_seed0_ckpt050k", "onehot"),
    ("ta_ABC_taskcond_100k", "act_ta_abc_taskcond_100k_seed0", "onehot"),
    ("ta_ABC_sbert_50k", "act_ta_abc_lang_sbert_100k_seed0_ckpt050k", "sbert"),
    ("ta_ABC_sbert_100k", "act_ta_abc_lang_sbert_100k_seed0", "sbert"),
    ("ta_ABC_chunk100_100k", "act_ta_abc_taskcond_chunk100_100k_seed0", "onehot"),
]
rows = []
for family, run_id, cond in candidates:
    path = root / "results" / "calvin_eval_extensions" / run_id / "D" / f"calvin_{run_id}_{cond}_D_eval_summary.json"
    if not path.exists():
        rows.append({"family": family, "run_id": run_id, "condition_type": cond, "missing": str(path)})
        continue
    data = json.loads(path.read_text())
    sr = data.get("success_rates", {})
    rows.append({
        "family": family,
        "run_id": run_id,
        "condition_type": cond,
        "avg_successful_sequence_len": data.get("avg_successful_sequence_len"),
        "success_len1": sr.get("1"),
        "success_len2": sr.get("2"),
        "success_len3": sr.get("3"),
        "success_len4": sr.get("4"),
        "success_len5": sr.get("5"),
        "num_sequences": data.get("num_sequences"),
        "summary_file": str(path),
    })
scored = [r for r in rows if r.get("avg_successful_sequence_len") is not None]
scored.sort(key=lambda r: r["avg_successful_sequence_len"], reverse=True)
payload = {
    "source": "course split dataset xiaoma26/calvin-lerobot",
    "purpose": "official splitA/splitB/splitC single-source and splitA+B+C combined training, evaluated zero-shot on splitD/CALVIN D",
    "best": scored[0] if scored else None,
    "rows": rows,
}
(root / "results/task2_ta_official_split_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
fieldnames = ["family", "run_id", "condition_type", "avg_successful_sequence_len", "success_len1", "success_len2", "success_len3", "success_len4", "success_len5", "num_sequences", "summary_file", "missing"]
with (root / "results/task2_ta_official_split_summary.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
lines = [
    "# HW3 Task 2 Course Split Summary",
    "",
    "The TA dataset provides explicit splitA/splitB/splitC/splitD scene labels. Single-source models use A, B, or C; the ABC model is the manual union of splitA+splitB+splitC. All models are evaluated zero-shot on CALVIN D.",
    "",
    "| Family | Run | Cond | AvgLen | SR@1 | SR@2 | SR@3 | SR@4 | SR@5 |",
    "|---|---|---|---:|---:|---:|---:|---:|---:|",
]
for r in scored:
    lines.append(f"| {r['family']} | {r['run_id']} | {r['condition_type']} | {r['avg_successful_sequence_len']} | {r['success_len1']} | {r['success_len2']} | {r['success_len3']} | {r['success_len4']} | {r['success_len5']} |")
missing = [r for r in rows if r.get("missing")]
if missing:
    lines += ["", "Missing or not yet evaluated:", ""]
    lines.extend(f"- {r['run_id']}: {r['missing']}" for r in missing)
(root / "results/task2_ta_official_split_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps(payload["best"], indent=2, ensure_ascii=False))
PY
}

echo "[$(timestamp)] course split chain started."
echo "Estimated runtime: download/convert 0.5-2h; six 100k trainings plus evaluations about 30-45h total."

download_ta_splits
for label in A B C D; do convert_split "$label"; done
write_audit

write_status running null "course split smoke training"
train_model A onehot act_ta_a_taskcond_smoke_seed0 200 50

train_model A onehot act_ta_a_taskcond_100k_seed0 100000 50
eval_model onehot act_ta_a_taskcond_100k_seed0 50000 act_ta_a_taskcond_100k_seed0_ckpt050k
eval_model onehot act_ta_a_taskcond_100k_seed0 100000 act_ta_a_taskcond_100k_seed0
summarize_ta || true

train_model B onehot act_ta_b_taskcond_100k_seed0 100000 50
eval_model onehot act_ta_b_taskcond_100k_seed0 50000 act_ta_b_taskcond_100k_seed0_ckpt050k
eval_model onehot act_ta_b_taskcond_100k_seed0 100000 act_ta_b_taskcond_100k_seed0
summarize_ta || true

train_model C onehot act_ta_c_taskcond_100k_seed0 100000 50
eval_model onehot act_ta_c_taskcond_100k_seed0 50000 act_ta_c_taskcond_100k_seed0_ckpt050k
eval_model onehot act_ta_c_taskcond_100k_seed0 100000 act_ta_c_taskcond_100k_seed0
summarize_ta || true

train_model ABC onehot act_ta_abc_taskcond_100k_seed0 100000 50
eval_model onehot act_ta_abc_taskcond_100k_seed0 50000 act_ta_abc_taskcond_100k_seed0_ckpt050k
eval_model onehot act_ta_abc_taskcond_100k_seed0 100000 act_ta_abc_taskcond_100k_seed0
summarize_ta || true

train_model ABC sbert act_ta_abc_lang_sbert_100k_seed0 100000 50
eval_model sbert act_ta_abc_lang_sbert_100k_seed0 50000 act_ta_abc_lang_sbert_100k_seed0_ckpt050k
eval_model sbert act_ta_abc_lang_sbert_100k_seed0 100000 act_ta_abc_lang_sbert_100k_seed0
summarize_ta || true

train_model ABC onehot act_ta_abc_taskcond_chunk100_100k_seed0 100000 100
eval_model onehot act_ta_abc_taskcond_chunk100_100k_seed0 100000 act_ta_abc_taskcond_chunk100_100k_seed0
summarize_ta || true

bash scripts/08_download_important_results.sh > results/task2_ta_official_split_latest_package_path.txt || true
write_status finished 0 "course split chain completed"
echo "[$(timestamp)] course split chain finished"
