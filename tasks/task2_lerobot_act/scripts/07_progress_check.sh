#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p results
out="results/progress_$(date +%Y%m%d_%H%M%S).txt"

{
  echo "== date =="
  date
  echo
  echo "== tmux sessions =="
  tmux ls || true
  echo
  echo "== gpu =="
  nvidia-smi || true
  echo
  echo "== disk =="
  df -h .
  echo
  echo "== status files =="
  ls -lh results/*.status.json 2>/dev/null || true
  echo
  echo "== recent logs =="
  for log in logs/*.train.log; do
    [ -f "$log" ] || continue
    echo "--- $log ---"
    tail -n 30 "$log"
  done
  echo
  echo "== output dirs =="
  find outputs -maxdepth 3 -type f \( -name "*.pt" -o -name "*.safetensors" -o -name "*.json" -o -name "*.csv" \) 2>/dev/null | tail -n 80 || true
} | tee "$out"
