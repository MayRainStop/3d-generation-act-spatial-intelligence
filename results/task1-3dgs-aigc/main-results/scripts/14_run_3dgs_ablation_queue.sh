#!/usr/bin/env bash
set -Eeuo pipefail
cd /root/autodl-tmp/hw3_task1_3dgs_aigc
source scripts/_common.sh
RUN_ID="task1_3dgs_ablation_queue"
LOG_FILE="$HW3_TASK1_ROOT/logs/${RUN_ID}.log"
PID_FILE="$HW3_TASK1_ROOT/status/${RUN_ID}.pid"
write_status "$RUN_ID" running null "queued; waiting for main Task1 chain if it is still running"
exec > >(tee -a "$LOG_FILE") 2>&1

wait_for_main_chain() {
  local pid_file="$HW3_TASK1_ROOT/status/task1_full_chain.pid"
  while true; do
    local pid=""
    local state="missing"
    if [ -f "$pid_file" ]; then
      pid="$(tr -cd '0-9' < "$pid_file")"
    fi
    state="$(/root/miniconda3/bin/python - <<PY
import json, pathlib
p=pathlib.Path('$HW3_TASK1_ROOT/status/task1_full_chain.json')
try:
    print(json.loads(p.read_text()).get('status','missing'))
except Exception:
    print('missing')
PY
)"
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      echo "[$(now_iso)] Waiting for task1_full_chain pid=$pid before 3DGS ablations."
      write_status "$RUN_ID" waiting null "waiting for task1_full_chain pid=$pid"
      sleep 1800
    elif [ "$state" = "running" ] || [ "$state" = "started" ]; then
      echo "[$(now_iso)] task1_full_chain status is $state but pid is not alive; waiting for monitor/restart."
      write_status "$RUN_ID" waiting null "task1_full_chain status=$state but pid not alive; waiting"
      sleep 1800
    else
      break
    fi
  done
}

require_main_success_or_continue_reasonably() {
  local status_file="$HW3_TASK1_ROOT/status/task1_full_chain.json"
  if [ -f "$status_file" ]; then
    local state
    state="$(/root/miniconda3/bin/python - <<PY
import json, pathlib
p=pathlib.Path('$status_file')
try:
    print(json.loads(p.read_text()).get('status','unknown'))
except Exception:
    print('unknown')
PY
)"
    if [ "$state" != "finished" ]; then
      write_status "$RUN_ID" failed 2 "main Task1 chain did not finish; skip ablations to preserve failure signal"
      echo "Main chain status is $state; skipping ablations."
      exit 2
    fi
  fi
}

run_variant() {
  local scene="$1"
  local iters="$2"
  local res="$3"
  local out="$HW3_TASK1_ROOT/outputs/3dgs/${scene}_iter${iters}_r${res}"
  if [ -d "$out/point_cloud" ] || [ -f "$HW3_TASK1_ROOT/results/3dgs/${scene}_iter${iters}_r${res}/metrics.txt" ]; then
    echo "[$(now_iso)] Skip existing 3DGS variant ${scene}_iter${iters}_r${res}"
  else
    write_status "$RUN_ID" running null "running ${scene}_iter${iters}_r${res}"
    bash scripts/04_run_3dgs_scene.sh "$scene" "$iters" "$res"
  fi
  /root/miniconda3/bin/python scripts/15_extract_3dgs_metrics.py || true
  bash scripts/09_collect_results.sh || true
}

echo $$ > "$PID_FILE"
echo "[$(now_iso)] 3DGS ablation queue started. Estimated runtime after main chain: 2-4 hours."
wait_for_main_chain
require_main_success_or_continue_reasonably
run_variant garden 7000 4
run_variant garden 15000 4
run_variant garden 30000 8
/root/miniconda3/bin/python scripts/15_extract_3dgs_metrics.py || true
/root/miniconda3/bin/python scripts/13_analyze_task1_results.py || true
bash scripts/12_export_artifact_check.sh || true
bash scripts/09_collect_results.sh || true
write_status "$RUN_ID" finished 0 "3DGS ablation queue completed"
echo "[$(now_iso)] 3DGS ablation queue finished."
