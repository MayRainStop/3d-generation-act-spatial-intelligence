#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs results

queue_id="task2_ta_after_high_score_queue"
status_file="results/${queue_id}.status.json"
log_file="logs/${queue_id}.log"
high_status="results/task2_high_score_chain.status.json"
high_pid_file="results/task2_high_score_chain.pid"
ta_status="results/task2_ta_official_split_chain.status.json"

timestamp() { date -Iseconds; }

write_status() {
  local status="$1"
  local exit_code="${2:-null}"
  local note="${3:-}"
  cat > "$status_file" <<JSON
{
  "run_id": "${queue_id}",
  "status": "${status}",
  "exit_code": ${exit_code},
  "updated_at": "$(timestamp)",
  "wait_for": "task2_high_score_chain",
  "then_run": "scripts/44_run_task2_ta_official_split_chain.sh",
  "log_file": "${log_file}",
  "note": "${note}"
}
JSON
}

json_field() {
  local file="$1"
  local field="$2"
  python3 - "$file" "$field" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
if not path.exists():
    print("")
    raise SystemExit(0)
data = json.loads(path.read_text())
print(data.get(sys.argv[2], ""))
PY
}

exec > >(tee -a "$log_file") 2>&1
trap 'rc=$?; if [ "$rc" -ne 0 ]; then write_status failed "$rc" "queue failed; inspect log_file"; fi' EXIT

echo "[$(timestamp)] Queue started; waiting for task2_high_score_chain to finish."
write_status waiting null "waiting for high-score chain"

while true; do
  high_state="$(json_field "$high_status" status)"
  high_note="$(json_field "$high_status" note)"
  high_pid=""
  [ -f "$high_pid_file" ] && high_pid="$(cat "$high_pid_file" 2>/dev/null || true)"
  if [ "$high_state" = "failed" ]; then
    write_status blocked 4 "high-score chain failed; not starting TA chain automatically"
    echo "[$(timestamp)] High-score chain failed; queue stops. note=${high_note}"
    exit 4
  fi
  if [ "$high_state" = "finished" ]; then
    break
  fi
  if [ -n "$high_pid" ] && kill -0 "$high_pid" 2>/dev/null; then
    echo "[$(timestamp)] Still waiting: high_score status=${high_state:-unknown}; note=${high_note:-none}; pid=${high_pid}"
    sleep 1800
    continue
  fi
  if [ "$high_state" != "running" ] && [ "$high_state" != "started" ] && [ -n "$high_state" ]; then
    break
  fi
  echo "[$(timestamp)] Still waiting: high_score status=${high_state:-unknown}; note=${high_note:-none}"
  sleep 1800
done

if [ -f "$ta_status" ] && [ "$(json_field "$ta_status" status)" = "running" ]; then
  write_status finished 0 "TA official split chain already running"
  echo "[$(timestamp)] TA official split chain is already running; queue exits."
  exit 0
fi

write_status running null "starting TA official split chain"
echo "[$(timestamp)] Starting scripts/44_run_task2_ta_official_split_chain.sh"
bash scripts/44_run_task2_ta_official_split_chain.sh
write_status finished 0 "TA official split chain finished"
