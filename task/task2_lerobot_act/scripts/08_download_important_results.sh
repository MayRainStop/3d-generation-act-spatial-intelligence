#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p results/shareable-snapshot
snapshot_dir="results/shareable-snapshot/task2-shareable-$(date +%Y%m%d_%H%M%S)"
mkdir -p "$snapshot_dir"

cp -f README.md "$snapshot_dir/" || true
cp -f environment.yml "$snapshot_dir/" || true
cp -f .gitignore "$snapshot_dir/" || true
mkdir -p "$snapshot_dir/configs" "$snapshot_dir/logs-tail" "$snapshot_dir/report-figures" "$snapshot_dir/model-paths"
cp -f configs/runs.csv "$snapshot_dir/configs/" 2>/dev/null || true
cp -f results/*.csv "$snapshot_dir/" 2>/dev/null || true
cp -f results/*.json "$snapshot_dir/" 2>/dev/null || true
cp -f results/*.md "$snapshot_dir/" 2>/dev/null || true
cp -f logs/*.command.txt "$snapshot_dir/" 2>/dev/null || true

for log in logs/*.train.log logs/*.eval_d.log logs/*.calvin_eval.log; do
  [ -f "$log" ] || continue
  tail -n 300 "$log" > "$snapshot_dir/logs-tail/$(basename "$log").tail.txt"
done

if [ -d results/report-figures ]; then
  cp -f results/report-figures/* "$snapshot_dir/report-figures/" 2>/dev/null || true
fi

find outputs -maxdepth 5 -type f \( -name "*.safetensors" -o -name "*.pt" -o -name "*.pth" \) \
  > "$snapshot_dir/model-paths/checkpoints_on_server.txt" 2>/dev/null || true

echo "$snapshot_dir"
