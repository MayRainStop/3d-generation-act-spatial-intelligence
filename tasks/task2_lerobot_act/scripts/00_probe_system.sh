#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs results
out="logs/system_probe_$(date +%Y%m%d_%H%M%S).log"

{
  echo "== date =="
  date
  echo
  echo "== host =="
  hostname
  echo
  echo "== user =="
  whoami
  echo
  echo "== pwd =="
  pwd
  echo
  echo "== disk =="
  df -h .
  echo
  echo "== memory =="
  free -h || true
  echo
  echo "== python =="
  command -v python3 || true
  python3 --version || true
  echo
  echo "== conda =="
  command -v conda || true
  conda --version || true
  echo
  echo "== gpu =="
  command -v nvidia-smi || true
  nvidia-smi || true
  echo
  echo "== tmux =="
  command -v tmux || true
  tmux -V || true
} | tee "$out"

echo "$out" > results/latest_system_probe_path.txt
