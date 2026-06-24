#!/usr/bin/env bash
set -Eeuo pipefail
cd /root/autodl-tmp/hw3_task1_3dgs_aigc
source scripts/_common.sh
SCENE="${1:-garden}"
ITERS="${2:-30000}"
RESOLUTION="${3:-4}"
RUN_ID="3dgs_${SCENE}_iter${ITERS}_r${RESOLUTION}"
write_status "$RUN_ID" running null "training $SCENE for $ITERS iterations at resolution divisor $RESOLUTION"
ENV_PREFIX="$HW3_TASK1_ROOT/envs/3dgs"
GS_DIR="$HW3_TASK1_ROOT/third_party/gaussian-splatting"
SCENE_DIR="$HW3_TASK1_ROOT/data/mipnerf360/$SCENE"
OUT_DIR="$HW3_TASK1_ROOT/outputs/3dgs/${SCENE}_iter${ITERS}_r${RESOLUTION}"
if [ ! -d "$SCENE_DIR" ]; then
  write_status "$RUN_ID" failed 1 "dataset missing: $SCENE_DIR"
  exit 1
fi
activate_conda "$ENV_PREFIX"
export CUDA_HOME="${CUDA_HOME:-$ENV_PREFIX}"
export PATH="$ENV_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$ENV_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$GS_DIR/submodules/simple-knn:$GS_DIR/submodules/diff-gaussian-rasterization:${PYTHONPATH:-}"
mkdir -p "$OUT_DIR"
cd "$GS_DIR"
CHECKPOINT="$OUT_DIR/point_cloud/iteration_${ITERS}/point_cloud.ply"
log_section "train 3DGS $SCENE iter=$ITERS r=$RESOLUTION"
if [ -f "$CHECKPOINT" ]; then
  echo "[$(now_iso)] Reusing existing 3DGS checkpoint: $CHECKPOINT"
else
  python train.py -s "$SCENE_DIR" -m "$OUT_DIR" --eval --iterations "$ITERS" -r "$RESOLUTION" --data_device cuda
fi
log_section "render and metrics $SCENE iter=$ITERS r=$RESOLUTION"
bash "$HW3_TASK1_ROOT/scripts/05_render_3dgs_scene.sh" "$SCENE" "$OUT_DIR"
write_status "$RUN_ID" finished 0 "training/render completed for $SCENE iter=$ITERS r=$RESOLUTION"
