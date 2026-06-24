#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs results outputs/20260606_high_score

chain_id="task2_high_score_chain"
status_file="results/${chain_id}.status.json"
log_file="logs/${chain_id}.log"
score_root="outputs/20260606_high_score"
package_note="results/task2_high_score_latest_package_path.txt"

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
  "score_root": "${score_root}",
  "log_file": "${log_file}",
  "note": "${note}"
}
JSON
}

exec > >(tee -a "$log_file") 2>&1
trap 'rc=$?; if [ "$rc" -ne 0 ]; then write_status failed "$rc" "high-score chain failed; inspect log_file"; fi' EXIT

. .venv/bin/activate
export HYDRA_FULL_ERROR=1

edit_resume_config() {
  local train_config="$1"
  local output_dir="$2"
  local target_steps="$3"
  local batch_size="$4"
  python - "$train_config" "$output_dir" "$target_steps" "$batch_size" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text())
data["output_dir"] = sys.argv[2]
data["steps"] = int(sys.argv[3])
data["batch_size"] = int(sys.argv[4])
data["save_freq"] = 20000
data["log_freq"] = 200
data["eval_freq"] = 0
data["resume"] = False
if data.get("wandb") and isinstance(data["wandb"], dict):
    data["wandb"]["mode"] = "offline"
path.write_text(json.dumps(data, indent=4), encoding="utf-8")
PY
}

prepare_resume_dir() {
  local src_dir="$1"
  local dst_dir="$2"
  local source_step="$3"
  local target_steps="$4"
  local batch_size="$5"
  local source_ckpt
  source_ckpt="$(printf "%06d" "$source_step")"

  if [ ! -d "${src_dir}/checkpoints/${source_ckpt}/pretrained_model" ]; then
    echo "Missing source checkpoint: ${src_dir}/checkpoints/${source_ckpt}/pretrained_model" >&2
    exit 5
  fi
  if [ ! -d "$dst_dir" ]; then
    mkdir -p "$(dirname "$dst_dir")"
    cp -a "$src_dir" "$dst_dir"
  fi
  edit_resume_config \
    "${dst_dir}/checkpoints/${source_ckpt}/pretrained_model/train_config.json" \
    "$dst_dir" "$target_steps" "$batch_size"
}

run_train_resume() {
  local kind="$1"
  local run_id="$2"
  local src_dir="$3"
  local source_step="$4"
  local target_steps="$5"
  local batch_size="$6"
  local dst_dir="${score_root}/${run_id}"
  local final_ckpt="${dst_dir}/checkpoints/$(printf "%06d" "$target_steps")/pretrained_model"
  local source_ckpt
  local train_config
  local train_log
  source_ckpt="$(printf "%06d" "$source_step")"
  train_config="${dst_dir}/checkpoints/${source_ckpt}/pretrained_model/train_config.json"
  train_log="logs/${run_id}.high_score.train.log"

  if [ -d "$final_ckpt" ]; then
    echo "[$(timestamp)] skip training ${run_id}; final checkpoint exists: ${final_ckpt}"
    return 0
  fi

  write_status running null "training ${run_id} from ${source_step} to ${target_steps}"
  prepare_resume_dir "$src_dir" "$dst_dir" "$source_step" "$target_steps" "$batch_size"
  echo "[$(timestamp)] Resume-train ${run_id}: kind=${kind}, source_step=${source_step}, target_steps=${target_steps}, batch=${batch_size}"

  local cmd=(python "scripts/10_train_task_conditioned_act.py" "--config_path=${train_config}" "--resume=true" "--wandb.mode=offline")
  if [ "$kind" = "sbert" ]; then
    cmd=(python "scripts/17_train_language_conditioned_act.py" "--config_path=${train_config}" "--resume=true" "--wandb.mode=offline")
  fi

  set +e
  "${cmd[@]}" 2>&1 | tee "$train_log"
  local rc="${PIPESTATUS[0]}"
  set -e

  if [ "$rc" -ne 0 ] && grep -Eiq "out of memory|cuda.*memory|cublas.*alloc" "$train_log"; then
    echo "[$(timestamp)] OOM detected for ${run_id}; retrying once with batch_size=4"
    edit_resume_config "$train_config" "$dst_dir" "$target_steps" 4
    set +e
    "${cmd[@]}" 2>&1 | tee -a "$train_log"
    rc="${PIPESTATUS[0]}"
    set -e
  fi

  if [ "$rc" -ne 0 ]; then
    echo "Training failed for ${run_id}; see ${train_log}" >&2
    exit "$rc"
  fi
}

eval_public_if_needed() {
  local condition_type="$1"
  local model_run_id="$2"
  local checkpoint="$3"
  local summary="results/calvin_eval_extensions/${model_run_id}/D/calvin_${model_run_id}_${condition_type}_D_eval_summary.json"
  if [ -f "$summary" ]; then
    echo "[$(timestamp)] skip eval ${model_run_id}/${condition_type}; summary exists: ${summary}"
    return 0
  fi
  write_status running null "evaluating ${model_run_id} on CALVIN D"
  bash scripts/20_run_conditioned_eval.sh "$condition_type" "$model_run_id" "$checkpoint" full
}

summarize_high_score() {
  write_status running null "summarizing high-score candidates"
  python - <<'PY'
from pathlib import Path
import csv
import json

root = Path.cwd()
candidates = [
    ("existing_best", "act_abc_taskcond_100k_seed0", "onehot"),
    ("checkpoint_sweep", "act_abc_taskcond_100k_seed0_ckpt060k", "onehot"),
    ("checkpoint_sweep", "act_abc_taskcond_100k_seed0_ckpt080k", "onehot"),
    ("long_train", "act_abc_taskcond_200k_seed0_ckpt120k", "onehot"),
    ("long_train", "act_abc_taskcond_200k_seed0_ckpt160k", "onehot"),
    ("long_train", "act_abc_taskcond_200k_seed0_ckpt200k", "onehot"),
    ("chunk100_long", "act_abc_taskcond_chunk100_100k_seed0", "onehot"),
    ("sbert_long", "act_abc_lang_sbert_100k_seed0", "sbert"),
    ("seed1_long", "act_abc_taskcond_seed1_100k", "onehot"),
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
    "purpose": "HW3 Task 2 high-score extension candidates for CALVIN D",
    "selection_rule": "Report the highest avg_successful_sequence_len, with SR@1..SR@5 as secondary metrics.",
    "best": scored[0] if scored else None,
    "rows": rows,
}
(root / "results/task2_high_score_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

fieldnames = ["family", "run_id", "condition_type", "avg_successful_sequence_len", "success_len1", "success_len2", "success_len3", "success_len4", "success_len5", "num_sequences", "summary_file", "missing"]
with (root / "results/task2_high_score_summary.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

lines = [
    "# HW3 Task 2 High-Score Summary",
    "",
    "Candidate models are evaluated on CALVIN D. The main score is average successful sequence length.",
    "",
    "| Family | Run | Cond | AvgLen | SR@1 | SR@2 | SR@3 | SR@4 | SR@5 |",
    "|---|---|---|---:|---:|---:|---:|---:|---:|",
]
for r in scored:
    lines.append(
        f"| {r['family']} | {r['run_id']} | {r['condition_type']} | {r['avg_successful_sequence_len']} | "
        f"{r['success_len1']} | {r['success_len2']} | {r['success_len3']} | {r['success_len4']} | {r['success_len5']} |"
    )
missing = [r for r in rows if r.get("missing")]
if missing:
    lines += ["", "Missing summaries:", ""]
    for r in missing:
        lines.append(f"- {r['run_id']}: {r['missing']}")
(root / "results/task2_high_score_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps(payload["best"], indent=2, ensure_ascii=False))
PY
}

echo "[$(timestamp)] HW3 Task2 high-score chain started."
echo "Estimated runtime: 15-18 hours if batch_size=8 fits; up to 24-30 hours if OOM forces batch_size=4."
write_status running null "starting high-score candidate sweep"

# Existing best checkpoint sweep: cheap, no extra training.
eval_public_if_needed onehot "act_abc_taskcond_100k_seed0_ckpt060k" \
  "outputs/20260603_extensions/act_abc_taskcond_100k_seed0/checkpoints/060000/pretrained_model"
eval_public_if_needed onehot "act_abc_taskcond_100k_seed0_ckpt080k" \
  "outputs/20260603_extensions/act_abc_taskcond_100k_seed0/checkpoints/080000/pretrained_model"
summarize_high_score || true

# Longer one-hot training from the current best public converted ABC model.
run_train_resume onehot "act_abc_taskcond_200k_seed0" \
  "outputs/20260603_extensions/act_abc_taskcond_100k_seed0" 100000 200000 8
eval_public_if_needed onehot "act_abc_taskcond_200k_seed0_ckpt120k" \
  "${score_root}/act_abc_taskcond_200k_seed0/checkpoints/120000/pretrained_model"
eval_public_if_needed onehot "act_abc_taskcond_200k_seed0_ckpt160k" \
  "${score_root}/act_abc_taskcond_200k_seed0/checkpoints/160000/pretrained_model"
eval_public_if_needed onehot "act_abc_taskcond_200k_seed0_ckpt200k" \
  "${score_root}/act_abc_taskcond_200k_seed0/checkpoints/200000/pretrained_model"
summarize_high_score || true

# Long horizon and language-conditioned variants.
run_train_resume onehot "act_abc_taskcond_chunk100_100k_seed0" \
  "outputs/20260603_extensions/act_abc_taskcond_chunk100_seed0" 50000 100000 8
eval_public_if_needed onehot "act_abc_taskcond_chunk100_100k_seed0" \
  "${score_root}/act_abc_taskcond_chunk100_100k_seed0/checkpoints/100000/pretrained_model"
summarize_high_score || true

HW3_LANG_EMBED_BACKEND=sbert run_train_resume sbert "act_abc_lang_sbert_100k_seed0" \
  "outputs/20260603_extensions/act_abc_lang_sbert_seed0" 50000 100000 8
eval_public_if_needed sbert "act_abc_lang_sbert_100k_seed0" \
  "${score_root}/act_abc_lang_sbert_100k_seed0/checkpoints/100000/pretrained_model"
summarize_high_score || true

# Seed-1 extension gives a second high-score candidate without starting from scratch.
run_train_resume onehot "act_abc_taskcond_seed1_100k" \
  "outputs/20260603_extensions/act_abc_taskcond_seed1" 50000 100000 8
eval_public_if_needed onehot "act_abc_taskcond_seed1_100k" \
  "${score_root}/act_abc_taskcond_seed1_100k/checkpoints/100000/pretrained_model"

summarize_high_score
python scripts/22_analyze_extension_results.py || true
bash scripts/08_download_important_results.sh > "$package_note" || true
write_status finished 0 "high-score chain completed"
echo "[$(timestamp)] HW3 Task2 high-score chain finished."
