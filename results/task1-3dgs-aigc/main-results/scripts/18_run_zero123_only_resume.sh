#!/usr/bin/env bash
set -Eeuo pipefail
cd /root/autodl-tmp/hw3_task1_3dgs_aigc
source scripts/_common.sh
RUN_ID=task1_zero123_only_resume
trap 'rc=$?; if [ "$rc" -ne 0 ]; then write_status "$RUN_ID" failed "$rc" "Zero123-only resume chain failed"; fi' EXIT
write_status "$RUN_ID" running null "running Zero123 smoke/full with 24GB-safe config"
echo "[$(date -Is)] Zero123-only resume started. Estimated runtime: smoke 15-40 min, full 45-120 min after cached model load."
bash scripts/08_run_zero123_image_to_3d.sh smoke
bash scripts/08_run_zero123_image_to_3d.sh full
bash scripts/09_collect_results.sh || true
bash scripts/12_export_artifact_check.sh || true
/root/miniconda3/bin/python scripts/13_analyze_task1_results.py || true
write_status "$RUN_ID" finished 0 "Zero123 smoke/full completed with 24GB-safe config"
