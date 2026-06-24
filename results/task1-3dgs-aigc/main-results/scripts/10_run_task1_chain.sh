#!/usr/bin/env bash
set -Eeuo pipefail
cd /root/autodl-tmp/hw3_task1_3dgs_aigc
source scripts/_common.sh
RUN_ID=task1_full_chain
trap 'rc=$?; if [ "$rc" -ne 0 ]; then write_status "$RUN_ID" failed "$rc" "HW3 Task1 chain failed; inspect logs/task1_full_chain.log"; fi' EXIT
write_status "$RUN_ID" running null "starting HW3 Task1 full chain"
bash scripts/01_setup_workspace.sh
bash scripts/00_check_server.sh > logs/00_check_server_$(date +%Y%m%d_%H%M%S).log 2>&1 || true
bash scripts/02_setup_gaussian_splatting.sh
for scene in garden bicycle counter; do
  write_status "$RUN_ID" running null "downloading and training 3DGS scene: $scene"
  bash scripts/03_download_mipnerf360.sh "$scene"
  bash scripts/11_colmap_scene_check.sh "$scene" || true
  bash scripts/04_run_3dgs_scene.sh "$scene" 30000 4
  /root/miniconda3/bin/python scripts/15_extract_3dgs_metrics.py || true
  bash scripts/09_collect_results.sh || true
done
write_status "$RUN_ID" running null "setting up and running threestudio text-to-3D"
bash scripts/06_setup_threestudio.sh
bash scripts/07_run_text_to_3d.sh smoke || { write_status text_to_3d_smoke failed 1 "smoke failed; see logs/task1_full_chain.log"; exit 1; }
bash scripts/07_run_text_to_3d.sh full
bash scripts/09_collect_results.sh || true
bash scripts/12_export_artifact_check.sh || true
/root/miniconda3/bin/python scripts/13_analyze_task1_results.py || true
write_status "$RUN_ID" running null "running Zero123 image-to-3D"
if bash scripts/08_run_zero123_image_to_3d.sh smoke; then
  bash scripts/08_run_zero123_image_to_3d.sh full || true
else
  echo "Zero123 smoke did not complete; likely model auth/network issue. Continuing to collection."
fi
bash scripts/09_collect_results.sh
bash scripts/12_export_artifact_check.sh || true
/root/miniconda3/bin/python scripts/13_analyze_task1_results.py || true
write_status "$RUN_ID" finished 0 "HW3 Task1 chain completed"
