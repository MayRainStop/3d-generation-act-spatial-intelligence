#!/usr/bin/env bash
set -Eeuo pipefail
cd /root/autodl-tmp/hw3_task1_3dgs_aigc
source scripts/_common.sh
MODE="${1:-smoke}"
RUN_ID="zero123_${MODE}"
write_status "$RUN_ID" running null "running Stable Zero123 $MODE"
ENV_PREFIX="$HW3_TASK1_ROOT/envs/threestudio"
TS_DIR="$HW3_TASK1_ROOT/third_party/threestudio"
activate_conda "$ENV_PREFIX"
export HF_HOME="$HW3_TASK1_ROOT/models/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HUGGINGFACE_CO_RESOLVE_ENDPOINT="${HUGGINGFACE_CO_RESOLVE_ENDPOINT:-$HF_ENDPOINT}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-900}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-900}"
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_HOME="${CUDA_HOME:-$ENV_PREFIX}"
export PATH="$ENV_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$ENV_PREFIX/lib:${LD_LIBRARY_PATH:-}"
INPUT_DIR="$HW3_TASK1_ROOT/data/zero123_inputs"
mkdir -p "$INPUT_DIR" "$TS_DIR/load/images" "$TS_DIR/load/zero123"
INPUT="$INPUT_DIR/red_teapot_rgba.png"
python - <<PY
from PIL import Image, ImageDraw
img = Image.new('RGBA', (512, 512), (0, 0, 0, 0))
d = ImageDraw.Draw(img)
d.ellipse((135, 175, 365, 395), fill=(190, 35, 35, 255), outline=(85, 15, 15, 255), width=10)
d.rectangle((210, 115, 300, 190), fill=(210, 55, 55, 255), outline=(85, 15, 15, 255), width=8)
d.ellipse((190, 88, 320, 135), fill=(160, 25, 25, 255), outline=(85, 15, 15, 255), width=8)
d.polygon([(360,235),(470,190),(380,285)], fill=(190,35,35,255), outline=(85,15,15,255))
d.arc((42,185,170,330), 80, 280, fill=(85,15,15,255), width=18)
d.ellipse((172, 110, 335, 145), fill=(255,255,255,48))
img.save('$INPUT')
PY
cp -f "$INPUT" "$TS_DIR/load/images/red_teapot_rgba.png"
cd "$TS_DIR"
if [ ! -f load/zero123/stable-zero123.ckpt ]; then
  if command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download stabilityai/stable-zero123 stable-zero123.ckpt --local-dir load/zero123 --local-dir-use-symlinks False || true
  fi
fi
zero_config="configs/stable-zero123.yaml"
zero_note="stable-zero123"
if [ ! -f load/zero123/stable-zero123.ckpt ]; then
  echo "stable-zero123.ckpt unavailable; trying Zero123-XL fallback"
  if [ ! -f load/zero123/zero123-xl.ckpt ]; then
    if command -v wget >/dev/null 2>&1; then
      wget -c -O load/zero123/zero123-xl.ckpt https://zero123.cs.columbia.edu/assets/zero123-xl.ckpt || true
    else
      curl -L -C - -o load/zero123/zero123-xl.ckpt https://zero123.cs.columbia.edu/assets/zero123-xl.ckpt || true
    fi
  fi
  if [ -f load/zero123/zero123-xl.ckpt ]; then
    zero_config="configs/hw3_zero123_24gb.yaml"
    zero_note="zero123-xl-fallback-24gb"
  else
    write_status "$RUN_ID" failed 3 "stable-zero123 and zero123-xl checkpoints unavailable"
    exit 3
  fi
fi
echo "Using Zero123 route: ${zero_note} config=${zero_config}"
if [ "$MODE" = "smoke" ]; then
  python launch.py --config "$zero_config" --train --gpu 0 \
    data.image_path=./load/images/red_teapot_rgba.png \
    name=hw3_task1_zero123 tag=smoke exp_root_dir="$HW3_TASK1_ROOT/outputs/threestudio" \
    trainer.max_steps=200 system.freq.val=100 system.freq.test=200 system.freq.save=200
else
  python launch.py --config "$zero_config" --train --gpu 0 \
    data.image_path=./load/images/red_teapot_rgba.png \
    name=hw3_task1_zero123 tag=full exp_root_dir="$HW3_TASK1_ROOT/outputs/threestudio"
fi
mkdir -p "$HW3_TASK1_ROOT/results/zero123_image_to_3d"
find "$HW3_TASK1_ROOT/outputs/threestudio" -maxdepth 5 -type f \( -name '*.mp4' -o -name '*.png' -o -name '*.jpg' -o -name '*.yaml' \) | tail -n 100 | while read -r f; do
  rel="${f#$HW3_TASK1_ROOT/outputs/threestudio/}"
  mkdir -p "$HW3_TASK1_ROOT/results/zero123_image_to_3d/$(dirname "$rel")"
  cp -f "$f" "$HW3_TASK1_ROOT/results/zero123_image_to_3d/$rel" 2>/dev/null || true
done
write_status "$RUN_ID" finished 0 "Zero123 $MODE completed"
