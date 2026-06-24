#!/usr/bin/env bash
set -Eeuo pipefail
cd /root/autodl-tmp/hw3_task1_3dgs_aigc
source scripts/_common.sh
RUN_ID=task1_remaining_ablation_then_zero123
trap 'rc=$?; if [ "$rc" -ne 0 ]; then write_status "$RUN_ID" failed "$rc" "remaining ablations + Zero123 chain failed"; fi' EXIT
write_status "$RUN_ID" waiting null "waiting for currently active 3DGS ablation to finish"
while pgrep -af "scripts/04_run_3dgs_scene.sh|outputs/3dgs/.*train.py|python train.py -s .*/data/mipnerf360" | grep -v pgrep >/dev/null; do sleep 60; done
write_status "$RUN_ID" running null "running queued 3DGS ablations"
bash scripts/14_run_3dgs_ablation_queue.sh
/root/miniconda3/bin/python scripts/15_extract_3dgs_metrics.py || true
bash scripts/09_collect_results.sh || true
/root/miniconda3/bin/python scripts/13_analyze_task1_results.py || true
write_status "$RUN_ID" running null "running patched Zero123 smoke/full"
bash scripts/08_run_zero123_image_to_3d.sh smoke
bash scripts/08_run_zero123_image_to_3d.sh full
bash scripts/09_collect_results.sh || true
bash scripts/12_export_artifact_check.sh || true
/root/miniconda3/bin/python scripts/13_analyze_task1_results.py || true
write_status "$RUN_ID" finished 0 "remaining ablations + Zero123 completed"
