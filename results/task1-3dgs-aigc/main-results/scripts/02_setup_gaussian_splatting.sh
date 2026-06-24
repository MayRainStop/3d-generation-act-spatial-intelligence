#!/usr/bin/env bash
set -Eeuo pipefail
cd /root/autodl-tmp/hw3_task1_3dgs_aigc
source scripts/_common.sh
RUN_ID=setup_gaussian_splatting
trap 'rc=$?; if [ "$rc" -ne 0 ]; then write_status "$RUN_ID" failed "$rc" "3DGS setup failed; inspect logs/task1_full_chain.log"; fi' EXIT
write_status "$RUN_ID" running null "setting up official 3D Gaussian Splatting"
GS_DIR="$HW3_TASK1_ROOT/third_party/gaussian-splatting"
ENV_PREFIX="$HW3_TASK1_ROOT/envs/3dgs"
log_section "clone gaussian-splatting"
if [ ! -d "$GS_DIR/.git" ]; then
  git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting "$GS_DIR"
else
  git -C "$GS_DIR" submodule update --init --recursive
fi
log_section "create conda env"
source /root/miniconda3/etc/profile.d/conda.sh
if [ ! -x "$ENV_PREFIX/bin/python" ]; then
  set +u
  CONDA_PKGS_DIRS="$CONDA_PKGS_DIRS" conda create -p "$ENV_PREFIX" python=3.10 pip -y
  conda_rc=$?
  set -u
  if [ "$conda_rc" -ne 0 ]; then
    write_status "$RUN_ID" failed "$conda_rc" "conda create failed"
    exit "$conda_rc"
  fi
fi
activate_conda "$ENV_PREFIX"
python -m pip install --upgrade pip setuptools wheel ninja cmake
log_section "install pytorch and python deps"
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install tqdm plyfile scipy opencv-python imageio imageio-ffmpeg matplotlib tensorboard lpips scikit-image
log_section "ensure cuda toolkit"
if ! command -v nvcc >/dev/null 2>&1; then
  set +u
  conda install -p "$ENV_PREFIX" -y -c nvidia cuda-toolkit=12.8
  conda_rc=$?
  set -u
  if [ "$conda_rc" -ne 0 ]; then
    python -m pip install nvidia-cuda-nvcc-cu12
  fi
fi
if [ -x "$ENV_PREFIX/bin/nvcc" ]; then
  export CUDA_HOME="$ENV_PREFIX"
  export PATH="$ENV_PREFIX/bin:$PATH"
else
  NVCC_PATH=$(python - <<'PY' 2>/dev/null || true
import os, site, glob
for base in site.getsitepackages():
    hits = glob.glob(os.path.join(base, 'nvidia', 'cuda_nvcc', 'bin', 'nvcc'))
    if hits:
        print(hits[0])
        break
PY
)
  if [ -n "$NVCC_PATH" ]; then
    export CUDA_HOME="$(dirname "$(dirname "$NVCC_PATH")")"
    export PATH="$CUDA_HOME/bin:$PATH"
  fi
fi
export LD_LIBRARY_PATH="$ENV_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$GS_DIR/submodules/simple-knn:$GS_DIR/submodules/diff-gaussian-rasterization:${PYTHONPATH:-}"

log_section "select compiler for CUDA extension build"
if [ -x /usr/bin/gcc ] && [ -x /usr/bin/g++ ]; then
  export CC=/usr/bin/gcc
  export CXX=/usr/bin/g++
else
  export CC="$ENV_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
  export CXX="$ENV_PREFIX/bin/x86_64-conda-linux-gnu-c++"
fi
echo "CC=$CC"
echo "CXX=$CXX"
$CXX --version | head -1 || true
log_section "verify torch cuda"
python - <<'PY'
import torch
print('torch', torch.__version__)
print('cuda_available', torch.cuda.is_available())
print('cuda', torch.version.cuda)
print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
log_section "install gaussian-splatting cuda extensions"
export CC="${CC:-/usr/bin/gcc}"
export CXX="${CXX:-/usr/bin/g++}"
cd "$GS_DIR"
python -m pip install --no-build-isolation -e submodules/diff-gaussian-rasterization
python -m pip install --no-build-isolation -e submodules/simple-knn
python - <<'PY'
import torch
import diff_gaussian_rasterization
import simple_knn._C
print('3dgs_imports_ok', torch.cuda.is_available())
PY
write_status "$RUN_ID" finished 0 "3DGS environment ready"
