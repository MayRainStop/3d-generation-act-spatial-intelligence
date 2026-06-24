#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/root/autodl-tmp/hw3_task1_3dgs_aigc
mkdir -p "$ROOT"/{scripts,logs,status,data,third_party,envs,models,outputs,results,docs,tmp}
cd "$ROOT"
source scripts/_common.sh
write_status setup_workspace running null "creating directories and cache config"
cat > "$ROOT/.env.sh" <<'EOF'
export HW3_TASK1_ROOT=/root/autodl-tmp/hw3_task1_3dgs_aigc
export PATH=/root/miniconda3/bin:$PATH
export HF_HOME=$HW3_TASK1_ROOT/models/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export TORCH_HOME=$HW3_TASK1_ROOT/models/torch
export XDG_CACHE_HOME=$HW3_TASK1_ROOT/models/xdg_cache
export CONDA_PKGS_DIRS=$HW3_TASK1_ROOT/envs/conda_pkgs
export TMPDIR=$HW3_TASK1_ROOT/tmp
export PIP_CACHE_DIR=$HW3_TASK1_ROOT/envs/pip_cache
EOF
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$TORCH_HOME" "$XDG_CACHE_HOME" "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR"
write_status setup_workspace finished 0 "workspace ready"
