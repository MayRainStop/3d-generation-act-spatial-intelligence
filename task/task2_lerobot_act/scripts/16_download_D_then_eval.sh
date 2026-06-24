#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs results

run_id="download_calvin_D_then_eval"
status_file="results/${run_id}.status.json"
log_file="logs/${run_id}.log"
eval_script="scripts/15_run_calvin_eval_taskcond.sh"

timestamp() {
  date -Iseconds
}

write_status() {
  local status="$1"
  local exit_code="${2:-null}"
  local note="${3:-}"
  cat > "$status_file" <<JSON
{
  "run_id": "${run_id}",
  "status": "${status}",
  "exit_code": ${exit_code},
  "updated_at": "$(timestamp)",
  "download_status": "results/download_calvin_D.status.json",
  "download_log": "logs/download_calvin_D.log",
  "eval_script": "${eval_script}",
  "log_file": "${log_file}",
  "note": "${note}"
}
JSON
}

exec > >(tee -a "$log_file") 2>&1
trap 'rc=$?; if [ "$rc" -ne 0 ]; then write_status failed "$rc" "chain failed; inspect log_file"; fi' EXIT

write_status started null "downloading CALVIN D; will run eval script after success"
echo "[$(timestamp)] Starting CALVIN D download + unzip. Estimated download about 14h, unzip about 1-3h based on prior probe."

echo "[$(timestamp)] Step 1/2: download/unzip D"
bash scripts/13_download_calvin_dataset.sh D

echo "[$(timestamp)] D dataset ready. Disk snapshot:"
df -h . data || df -h .
du -sh data/calvin_raw/datasets/task_D_D || true

if [ ! -x "$eval_script" ]; then
  write_status waiting_for_eval_script null "D ready, but eval script is not executable yet"
  echo "[$(timestamp)] ${eval_script} is missing or not executable; waiting for adapter script."
  while [ ! -x "$eval_script" ]; do
    sleep 300
  done
fi

write_status running_eval null "D ready; running task-conditioned ACT CALVIN eval"
echo "[$(timestamp)] Step 2/2: running ${eval_script}"
bash "$eval_script"
write_status finished 0 "download and eval chain finished"
echo "[$(timestamp)] Chain finished."
