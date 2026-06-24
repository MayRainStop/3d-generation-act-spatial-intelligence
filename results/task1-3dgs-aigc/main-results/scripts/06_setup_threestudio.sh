#!/usr/bin/env bash
set -Eeuo pipefail
cd /root/autodl-tmp/hw3_task1_3dgs_aigc
source scripts/_common.sh
RUN_ID=setup_threestudio
write_status "$RUN_ID" running null "setting up threestudio"
TS_DIR="$HW3_TASK1_ROOT/third_party/threestudio"
ENV_PREFIX="$HW3_TASK1_ROOT/envs/threestudio"
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"
export NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-}"
safe_conda_install() {
  set +u
  CONDA_PKGS_DIRS="$CONDA_PKGS_DIRS" conda install -p "$ENV_PREFIX" -y "$@"
  local rc=$?
  set -u
  return "$rc"
}
if [ ! -d "$TS_DIR/.git" ]; then
  git clone https://github.com/threestudio-project/threestudio "$TS_DIR"
fi
source /root/miniconda3/etc/profile.d/conda.sh
if [ ! -x "$ENV_PREFIX/bin/python" ]; then
  set +u
  CONDA_PKGS_DIRS="$CONDA_PKGS_DIRS" conda create -p "$ENV_PREFIX" python=3.10 pip -y
  set -u
fi
activate_conda "$ENV_PREFIX"
python -m pip install --upgrade pip setuptools wheel ninja cmake
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install pybind11
safe_conda_install -c conda-forge eigen || true
export CPLUS_INCLUDE_PATH="$ENV_PREFIX/include/eigen3:${CPLUS_INCLUDE_PATH:-}"
export CPATH="$ENV_PREFIX/include/eigen3:${CPATH:-}"
# nerfacc v0.5.2 imports torch from setup.py, so PEP 517 isolation hides
# the torch wheel that was just installed in this environment. GitHub clones
# can also be flaky from this server, so retry the full requirements install.
git config --global http.version HTTP/1.1 || true
git config --global http.postBuffer 524288000 || true
install_threestudio_requirements() {
  local attempt
  for attempt in 1 2 3 4 5 6 7 8 9 10 11 12; do
    echo "[$(date -Iseconds)] Installing threestudio requirements attempt ${attempt}/12"
    rm -rf "$HW3_TASK1_ROOT"/tmp/pip-req-build-* "$HW3_TASK1_ROOT"/tmp/pip-build-env-* "$HW3_TASK1_ROOT"/tmp/pip-install-* 2>/dev/null || true
    if GIT_HTTP_VERSION=HTTP/1.1 python -m pip install --no-build-isolation --retries 10 --timeout 120 -r "$TS_DIR/requirements.txt"; then
      return 0
    fi
    echo "[$(date -Iseconds)] threestudio requirements attempt ${attempt}/12 failed; retrying after backoff"
    sleep_seconds=$((attempt * 45))
    if [ "$sleep_seconds" -gt 300 ]; then sleep_seconds=300; fi
    sleep "$sleep_seconds"
  done
  return 1
}
install_threestudio_requirements
python -m pip install "huggingface_hub==0.19.4"
if ! command -v nvcc >/dev/null 2>&1; then
  safe_conda_install -c nvidia cuda-toolkit=12.8 || true
fi
export CUDA_HOME="${CUDA_HOME:-$ENV_PREFIX}"
export PATH="$ENV_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$ENV_PREFIX/lib:${LD_LIBRARY_PATH:-}"
python - <<'PY'
import torch
print('torch', torch.__version__)
print('cuda_available', torch.cuda.is_available())
print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
write_status "$RUN_ID" finished 0 "threestudio environment ready"
