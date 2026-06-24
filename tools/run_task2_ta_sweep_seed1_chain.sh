#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs results outputs/20260606_ta_official_split

chain_id="task2_ta_sweep_seed1_chain"
status_file="results/${chain_id}.status.json"
pid_file="results/${chain_id}.pid"
log_file="logs/${chain_id}.log"
out_root="outputs/20260606_ta_official_split"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TOKENIZERS_PARALLELISM=false
export SENTENCE_TRANSFORMERS_HOME="${SENTENCE_TRANSFORMERS_HOME:-${HF_HOME}/sentence_transformers}"
export HW3_LANG_EMBED_MODEL="${HW3_LANG_EMBED_MODEL:-sentence-transformers/all-MiniLM-L6-v2}"
export HYDRA_FULL_ERROR=1

timestamp() { date -Iseconds; }

write_status() {
  local status="$1"
  local exit_code="${2:-null}"
  local note="${3:-}"
  local current="${4:-}"
  cat > "$status_file" <<JSON
{
  "run_id": "${chain_id}",
  "status": "${status}",
  "exit_code": ${exit_code},
  "updated_at": "$(timestamp)",
  "out_root": "${out_root}",
  "log_file": "${log_file}",
  "pid_file": "${pid_file}",
  "current": "${current}",
  "note": "${note}"
}
JSON
}

exec > >(tee -a "$log_file") 2>&1
trap 'rc=$?; if [ "$rc" -ne 0 ]; then write_status failed "$rc" "TA sweep/seed1 chain failed; inspect log_file"; fi' EXIT

echo "$$" > "$pid_file"
. .venv/bin/activate

checkpoint_dir() {
  local run_id="$1"
  local step="$2"
  printf "%s/%s/checkpoints/%06d" "$out_root" "$run_id" "$step"
}

checkpoint_model() {
  local run_id="$1"
  local step="$2"
  printf "%s/pretrained_model" "$(checkpoint_dir "$run_id" "$step")"
}

summary_path() {
  local eval_run="$1"
  local condition="$2"
  printf "results/calvin_eval_extensions/%s/D/calvin_%s_%s_D_eval_summary.json" "$eval_run" "$eval_run" "$condition"
}

train_run() {
  local split="$1"
  local condition="$2"
  local run_id="$3"
  local seed="$4"
  local steps="$5"
  local chunk_size="$6"
  local resume="${7:-false}"
  local target
  target="$(checkpoint_model "$run_id" "$steps")"
  if [ -d "$target" ]; then
    echo "[$(timestamp)] checkpoint exists, skip train ${run_id} step ${steps}: ${target}"
    return 0
  fi

  local batch_size="${HW3_TA_BATCH_SIZE:-8}"
  local train_log="logs/${run_id}.ta_sweep_seed1.train.log"
  local resume_args=()
  if [ "$resume" = "true" ]; then
    local last_ckpt="${out_root}/${run_id}/checkpoints/last"
    if [ ! -e "$last_ckpt" ]; then
      echo "[$(timestamp)] resume requested but last checkpoint missing for ${run_id}; training from scratch"
    else
      resume_args=(--resume=true --config_path="${last_ckpt}/pretrained_model/train_config.json")
    fi
  fi

  write_status running null "training ${run_id}" "${run_id}:train:${steps}"
  echo "[$(timestamp)] Train ${run_id}: split=${split}, condition=${condition}, seed=${seed}, steps=${steps}, chunk=${chunk_size}, resume=${resume}, batch=${batch_size}"
  export HW3_TA_SOURCE_SPLIT="$split"
  export HW3_TA_CONDITION_TYPE="$condition"
  export HW3_LANG_EMBED_BACKEND="sbert"

  set +e
  python scripts/43_train_ta_official_split_conditioned_act.py \
    --policy.type=act \
    --dataset.repo_id=xiaoma26/calvin_lerobot_splitA \
    --dataset.root=data/xiaoma_calvin_lerobot_v30/xiaoma26/calvin_lerobot_splitA \
    --output_dir="${out_root}/${run_id}" \
    --seed="${seed}" \
    --steps="${steps}" \
    --batch_size="${batch_size}" \
    --save_freq=20000 \
    --log_freq=200 \
    --eval_freq=0 \
    --tolerance_s="${HW3_TA_TOLERANCE_S:-0.01}" \
    --optimizer.lr=0.0001 \
    --policy.chunk_size="${chunk_size}" \
    --policy.n_action_steps="${chunk_size}" \
    --policy.push_to_hub=false \
    --wandb.mode=offline \
    "${resume_args[@]}" \
    2>&1 | tee "$train_log"
  local rc="${PIPESTATUS[0]}"
  set -e

  if [ "$rc" -ne 0 ] && grep -Eiq "out of memory|cuda.*memory|cublas.*alloc" "$train_log"; then
    echo "[$(timestamp)] OOM for ${run_id}; retry with batch_size=4"
    set +e
    python scripts/43_train_ta_official_split_conditioned_act.py \
      --policy.type=act \
      --dataset.repo_id=xiaoma26/calvin_lerobot_splitA \
      --dataset.root=data/xiaoma_calvin_lerobot_v30/xiaoma26/calvin_lerobot_splitA \
      --output_dir="${out_root}/${run_id}" \
      --seed="${seed}" \
      --steps="${steps}" \
      --batch_size=4 \
      --save_freq=20000 \
      --log_freq=200 \
      --eval_freq=0 \
      --tolerance_s="${HW3_TA_TOLERANCE_S:-0.01}" \
      --optimizer.lr=0.0001 \
      --policy.chunk_size="${chunk_size}" \
      --policy.n_action_steps="${chunk_size}" \
      --policy.push_to_hub=false \
      --wandb.mode=offline \
      "${resume_args[@]}" \
      2>&1 | tee -a "$train_log"
    rc="${PIPESTATUS[0]}"
    set -e
  fi

  if [ "$rc" -ne 0 ]; then
    echo "[$(timestamp)] training failed for ${run_id}; see ${train_log}" >&2
    exit "$rc"
  fi
}

eval_run() {
  local condition="$1"
  local source_run="$2"
  local step="$3"
  local eval_name="$4"
  local ckpt
  ckpt="$(checkpoint_model "$source_run" "$step")"
  local summary
  summary="$(summary_path "$eval_name" "$condition")"
  if [ ! -d "$ckpt" ]; then
    echo "[$(timestamp)] missing checkpoint, skip eval ${eval_name}: ${ckpt}"
    return 0
  fi
  if [ -f "$summary" ]; then
    echo "[$(timestamp)] summary exists, skip eval ${eval_name}: ${summary}"
    return 0
  fi
  write_status running null "evaluating ${eval_name}" "${eval_name}:eval:${step}"
  bash scripts/45_run_ta_conditioned_eval.sh "$condition" "$eval_name" "$ckpt" full
}

summarize() {
  write_status running null "summarizing TA sweep/seed1 results" "summarize"
  python - <<'PY'
from pathlib import Path
import csv
import json

root = Path.cwd()
candidates = [
    ("ta_A_seed0_100k", "act_ta_a_taskcond_100k_seed0", "onehot"),
    ("ta_A_seed0_120k", "act_ta_a_taskcond_160k_seed0_ckpt120k", "onehot"),
    ("ta_A_seed0_140k", "act_ta_a_taskcond_160k_seed0_ckpt140k", "onehot"),
    ("ta_A_seed0_160k", "act_ta_a_taskcond_160k_seed0", "onehot"),
    ("ta_A_seed1_50k", "act_ta_a_taskcond_100k_seed1_ckpt050k", "onehot"),
    ("ta_A_seed1_100k", "act_ta_a_taskcond_100k_seed1", "onehot"),
    ("ta_ABC_sbert_seed0_100k", "act_ta_abc_lang_sbert_100k_seed0", "sbert"),
    ("ta_ABC_sbert_seed0_120k", "act_ta_abc_lang_sbert_160k_seed0_ckpt120k", "sbert"),
    ("ta_ABC_sbert_seed0_140k", "act_ta_abc_lang_sbert_160k_seed0_ckpt140k", "sbert"),
    ("ta_ABC_sbert_seed0_160k", "act_ta_abc_lang_sbert_160k_seed0", "sbert"),
    ("ta_ABC_sbert_seed1_50k", "act_ta_abc_lang_sbert_100k_seed1_ckpt050k", "sbert"),
    ("ta_ABC_sbert_seed1_100k", "act_ta_abc_lang_sbert_100k_seed1", "sbert"),
]
rows = []
for family, run_id, cond in candidates:
    path = root / "results" / "calvin_eval_extensions" / run_id / "D" / f"calvin_{run_id}_{cond}_D_eval_summary.json"
    row = {"family": family, "run_id": run_id, "condition_type": cond, "summary_file": str(path)}
    if path.exists():
        data = json.loads(path.read_text())
        sr = data.get("success_rates", {})
        row.update({
            "avg_successful_sequence_len": data.get("avg_successful_sequence_len"),
            "success_len1": sr.get("1"),
            "success_len2": sr.get("2"),
            "success_len3": sr.get("3"),
            "success_len4": sr.get("4"),
            "success_len5": sr.get("5"),
            "num_sequences": data.get("num_sequences"),
        })
    else:
        row["missing"] = str(path)
    rows.append(row)

scored = [r for r in rows if r.get("avg_successful_sequence_len") is not None]
scored.sort(key=lambda r: r["avg_successful_sequence_len"], reverse=True)
payload = {
    "source": "course splitA/B/C/D sweep and seed1 extension",
    "best": scored[0] if scored else None,
    "rows": rows,
}
(root / "results/task2_ta_sweep_seed1_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
fieldnames = [
    "family", "run_id", "condition_type", "avg_successful_sequence_len",
    "success_len1", "success_len2", "success_len3", "success_len4", "success_len5",
    "num_sequences", "summary_file", "missing",
]
with (root / "results/task2_ta_sweep_seed1_summary.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
lines = [
    "# HW3 Task 2 TA Sweep/Seed1 Summary",
    "",
    "| Family | Run | Cond | AvgLen | SR@1 | SR@2 | SR@3 | SR@4 | SR@5 |",
    "|---|---|---|---:|---:|---:|---:|---:|---:|",
]
for r in scored:
    lines.append(
        f"| {r['family']} | {r['run_id']} | {r['condition_type']} | "
        f"{r['avg_successful_sequence_len']} | {r['success_len1']} | {r['success_len2']} | "
        f"{r['success_len3']} | {r['success_len4']} | {r['success_len5']} |"
    )
missing = [r for r in rows if r.get("missing")]
if missing:
    lines += ["", "Missing or not yet evaluated:", ""]
    lines.extend(f"- {r['run_id']}: {r['missing']}" for r in missing)
(root / "results/task2_ta_sweep_seed1_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps(payload["best"], indent=2, ensure_ascii=False))
PY
}

write_status running null "starting TA sweep/seed1 chain" "start"
echo "[$(timestamp)] TA sweep/seed1 chain started"
echo "[$(timestamp)] Scope: resume seed0 TA A and TA ABC SBERT to 160k; train seed1 TA A and TA ABC SBERT to 100k; evaluate checkpoints."

# Seed0 checkpoint sweep. Existing 100k run directories are resumed in place.
train_run A onehot act_ta_a_taskcond_100k_seed0 0 160000 50 true
eval_run onehot act_ta_a_taskcond_100k_seed0 120000 act_ta_a_taskcond_160k_seed0_ckpt120k
eval_run onehot act_ta_a_taskcond_100k_seed0 140000 act_ta_a_taskcond_160k_seed0_ckpt140k
eval_run onehot act_ta_a_taskcond_100k_seed0 160000 act_ta_a_taskcond_160k_seed0
summarize || true

train_run ABC sbert act_ta_abc_lang_sbert_100k_seed0 0 160000 50 true
eval_run sbert act_ta_abc_lang_sbert_100k_seed0 120000 act_ta_abc_lang_sbert_160k_seed0_ckpt120k
eval_run sbert act_ta_abc_lang_sbert_100k_seed0 140000 act_ta_abc_lang_sbert_160k_seed0_ckpt140k
eval_run sbert act_ta_abc_lang_sbert_100k_seed0 160000 act_ta_abc_lang_sbert_160k_seed0
summarize || true

# Seed1 robustness checks.
train_run A onehot act_ta_a_taskcond_100k_seed1 1 100000 50 false
eval_run onehot act_ta_a_taskcond_100k_seed1 50000 act_ta_a_taskcond_100k_seed1_ckpt050k
eval_run onehot act_ta_a_taskcond_100k_seed1 100000 act_ta_a_taskcond_100k_seed1
summarize || true

train_run ABC sbert act_ta_abc_lang_sbert_100k_seed1 1 100000 50 false
eval_run sbert act_ta_abc_lang_sbert_100k_seed1 50000 act_ta_abc_lang_sbert_100k_seed1_ckpt050k
eval_run sbert act_ta_abc_lang_sbert_100k_seed1 100000 act_ta_abc_lang_sbert_100k_seed1
summarize || true

write_status finished 0 "TA sweep/seed1 chain completed" "finished"
echo "[$(timestamp)] TA sweep/seed1 chain finished"
