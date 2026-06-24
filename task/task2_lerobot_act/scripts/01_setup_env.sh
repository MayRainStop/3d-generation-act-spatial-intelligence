#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p results
trap 'echo $? > results/setup_env.exit' EXIT

if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  if conda env list | awk '{print $1}' | grep -qx "hw3_task2_lerobot"; then
    echo "Conda env hw3_task2_lerobot already exists."
  else
    conda env create -f environment.yml
  fi
  conda activate hw3_task2_lerobot
else
  if [ ! -d .venv ]; then
    python3 -m venv .venv
  fi
  source .venv/bin/activate
fi

python -m pip install --upgrade pip setuptools wheel
python -m pip install \
  numpy \
  pandas \
  matplotlib \
  seaborn \
  pyyaml \
  tqdm \
  scikit-learn \
  opencv-python \
  pybullet \
  "huggingface_hub[hf_xet]" \
  datasets \
  safetensors \
  wandb \
  swanlab \
  pytest

remove_stale_torch() {
  python - <<'PY'
from pathlib import Path
import shutil
import site

patterns = [
    "torch",
    "torch-*.dist-info",
    "torchvision",
    "torchvision-*.dist-info",
    "triton",
    "triton-*.dist-info",
]

for root in map(Path, site.getsitepackages()):
    for pattern in patterns:
        for path in root.glob(pattern):
            if path.exists():
                print(f"removing stale package path: {path}")
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
PY
}

if ! python -c "import torch, torchvision; raise SystemExit(0 if getattr(torch, '__file__', None) and hasattr(torch, 'cuda') and '+cu128' in torch.__version__ else 1)" >/dev/null 2>&1; then
  python -m pip uninstall -y torch torchvision torchaudio triton || true
  remove_stale_torch
  python -m pip install --no-cache-dir --force-reinstall \
    torch==2.7.1 \
    torchvision==0.22.1 \
    --index-url https://download.pytorch.org/whl/cu128
fi

python -m pip install fsspec==2026.2.0 setuptools==80.10.2

if ! python -c "import lerobot" >/dev/null 2>&1; then
  python -m pip install "lerobot==0.4.4"
fi

python scripts/02_verify_env.py
