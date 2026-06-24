#!/usr/bin/env bash
set -Eeuo pipefail
cd /root/autodl-tmp/hw3_task1_3dgs_aigc
source scripts/_common.sh
write_status check_server running null "checking server resources"
log_section "identity"
whoami
hostname
pwd
date '+%F %T %z'
uname -a
log_section "gpu"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu,driver_version --format=csv,noheader,nounits || true
nvidia-smi || true
log_section "disk"
df -hT / /root/autodl-tmp /autodl-pub 2>/dev/null || df -hT
log_section "memory_cpu"
free -h || true
echo "nproc: $(nproc)"
lscpu | egrep 'Model name|CPU\(s\)|Thread|Core|Socket|NUMA' || true
log_section "env"
/root/miniconda3/bin/conda --version || true
/root/miniconda3/bin/conda env list || true
/root/miniconda3/bin/python --version || true
/root/miniconda3/bin/python - <<'PY' || true
import torch
print('torch', torch.__version__)
print('cuda_available', torch.cuda.is_available())
print('cuda', torch.version.cuda)
print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
log_section "processes"
pgrep -af '10_run_task1_chain|02_setup_gaussian|03_download_mipnerf|04_run_3dgs|05_render_3dgs|06_setup_threestudio|07_run_text|08_run_zero123|train.py|launch.py|aria2c|wget|curl|unzip' || true
write_status check_server finished 0 "server check completed"
