#!/usr/bin/env bash
set -Eeuo pipefail
cd /root/autodl-tmp/hw3_task1_3dgs_aigc
source scripts/_common.sh
RUN_ID=task1_zero123_then_ablation_resume
trap 'rc=$?; if [ "$rc" -ne 0 ]; then write_status "$RUN_ID" failed "$rc" "Zero123 + ablation resume chain failed"; fi' EXIT
write_status "$RUN_ID" running null "running Zero123 resume before 3DGS ablations"
bash scripts/08_run_zero123_image_to_3d.sh smoke
bash scripts/08_run_zero123_image_to_3d.sh full
bash scripts/09_collect_results.sh || true
bash scripts/12_export_artifact_check.sh || true
/root/miniconda3/bin/python scripts/13_analyze_task1_results.py || true
write_status "$RUN_ID" running null "running queued 3DGS ablations"
bash scripts/14_run_3dgs_ablation_queue.sh
bash scripts/09_collect_results.sh || true
/root/miniconda3/bin/python scripts/15_extract_3dgs_metrics.py || true
/root/miniconda3/bin/python scripts/13_analyze_task1_results.py || true
write_status "$RUN_ID" finished 0 "Zero123 + ablation resume chain completed"
