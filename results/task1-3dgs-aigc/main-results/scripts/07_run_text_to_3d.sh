#!/usr/bin/env bash
set -Eeuo pipefail
cd /root/autodl-tmp/hw3_task1_3dgs_aigc
source scripts/_common.sh
MODE="${1:-smoke}"
PROMPT="${2:-a small red ceramic teapot, high quality 3d asset}"
RUN_ID="text_to_3d_${MODE}"
write_status "$RUN_ID" running null "running threestudio DreamFusion SD $MODE"
ENV_PREFIX="$HW3_TASK1_ROOT/envs/threestudio"
TS_DIR="$HW3_TASK1_ROOT/third_party/threestudio"
activate_conda "$ENV_PREFIX"
export CUDA_HOME="${CUDA_HOME:-$ENV_PREFIX}"
export PATH="$ENV_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$ENV_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="$HW3_TASK1_ROOT/models/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HUGGINGFACE_CO_RESOLVE_ENDPOINT="${HUGGINGFACE_CO_RESOLVE_ENDPOINT:-$HF_ENDPOINT}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-300}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-300}"
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
cd "$TS_DIR"
SD_MODEL="${SD_MODEL:-runwayml/stable-diffusion-v1-5}"
if [ "$MODE" = "smoke" ]; then
  python launch.py --config configs/dreamfusion-sd.yaml --train --gpu 0 \
    system.prompt_processor.prompt="$PROMPT" \
    name=hw3_task1_text_to_3d tag=smoke exp_root_dir="$HW3_TASK1_ROOT/outputs/threestudio" \
    trainer.max_steps=200 trainer.val_check_interval=100 checkpoint.every_n_train_steps=200 \
    system.prompt_processor.pretrained_model_name_or_path="$SD_MODEL" \
    system.guidance.pretrained_model_name_or_path="$SD_MODEL"
else
  python launch.py --config configs/dreamfusion-sd.yaml --train --gpu 0 \
    system.prompt_processor.prompt="$PROMPT" \
    name=hw3_task1_text_to_3d tag=full exp_root_dir="$HW3_TASK1_ROOT/outputs/threestudio" \
    system.prompt_processor.pretrained_model_name_or_path="$SD_MODEL" \
    system.guidance.pretrained_model_name_or_path="$SD_MODEL"
fi
mkdir -p "$HW3_TASK1_ROOT/results/threestudio_text_to_3d"
find "$HW3_TASK1_ROOT/outputs/threestudio" -maxdepth 5 -type f \( -name '*.mp4' -o -name '*.png' -o -name '*.jpg' -o -name '*.yaml' \) | tail -n 100 | while read -r f; do
  rel="${f#$HW3_TASK1_ROOT/outputs/threestudio/}"
  mkdir -p "$HW3_TASK1_ROOT/results/threestudio_text_to_3d/$(dirname "$rel")"
  cp -f "$f" "$HW3_TASK1_ROOT/results/threestudio_text_to_3d/$rel" 2>/dev/null || true
done
write_status "$RUN_ID" finished 0 "text-to-3D $MODE completed"
