#!/usr/bin/env bash
set -Eeuo pipefail
cd /root/autodl-tmp/hw3_task1_3dgs_aigc
source scripts/_common.sh
RUN_ID=collect_results
write_status "$RUN_ID" running null "collecting curated results"
find "$HW3_TASK1_ROOT" -path '*/__pycache__' -type d -not -path "$HW3_TASK1_ROOT/envs/*" -prune -exec rm -rf {} + 2>/dev/null || true
find "$HW3_TASK1_ROOT/tmp" -mindepth 1 -maxdepth 2 -type f -delete 2>/dev/null || true
find "$HW3_TASK1_ROOT/data/downloads" -type f \( -name '*.aria2' -o -name '*.part' -o -name '*.tmp' \) -delete 2>/dev/null || true
ARCHIVE="$HW3_TASK1_ROOT/results/hw3_task1_results_$(date +%Y%m%d_%H%M%S).tar.gz"
tar -czf "$ARCHIVE" \
  --exclude='results/*.tar.gz' \
  README.md scripts docs status results logs
/root/miniconda3/bin/python - <<PY
import json, pathlib, datetime, os, subprocess
root=pathlib.Path(os.environ['HW3_TASK1_ROOT'])
archive=pathlib.Path('$ARCHIVE')
payload={'run_id':'collect_results','status':'finished','exit_code':0,'updated_at':datetime.datetime.now().astimezone().isoformat(),'archive':str(archive)}
try: payload['archive_size']=subprocess.check_output(['du','-h',str(archive)], text=True).split()[0]
except Exception: pass
with open(root/'status'/'collect_results.json','w',encoding='utf-8') as f: json.dump(payload,f,indent=2)
PY
