#!/usr/bin/env bash
set -Eeuo pipefail
cd /root/autodl-tmp/hw3_task1_3dgs_aigc
source scripts/_common.sh
chain_id="task1_final_completion_chain"
status_file="status/${chain_id}.json"
log_file="logs/${chain_id}.log"
mkdir -p logs status results/report_assets

timestamp() { date -Iseconds; }
write_chain_status() {
  local status="$1"
  local exit_code="${2:-null}"
  local note="${3:-}"
  cat > "$status_file" <<JSON
{
  "run_id": "${chain_id}",
  "status": "${status}",
  "exit_code": ${exit_code},
  "updated_at": "$(timestamp)",
  "log_file": "${log_file}",
  "note": "${note}"
}
JSON
}

exec > >(tee -a "$log_file") 2>&1
trap 'rc=$?; if [ "$rc" -ne 0 ]; then write_chain_status failed "$rc" "inspect ${log_file}"; fi' EXIT

write_chain_status running null "building final report artifacts"
echo "[$(timestamp)] Task1 final completion chain started"

bash scripts/12_export_artifact_check.sh
/root/miniconda3/bin/python scripts/15_extract_3dgs_metrics.py
/root/miniconda3/bin/python scripts/19_make_report_artifacts.py
/root/miniconda3/bin/python scripts/13_analyze_task1_results.py
bash scripts/09_collect_results.sh

/root/miniconda3/bin/python - <<'PY'
from pathlib import Path
import json, datetime
root = Path('/root/autodl-tmp/hw3_task1_3dgs_aigc')
archives = sorted((root/'results').glob('hw3_task1_results_*.tar.gz'), key=lambda p: p.stat().st_mtime)
deleted = []
if len(archives) > 1:
    keep = archives[-1]
    for path in archives[:-1]:
        if path.parent == root/'results' and path.name.startswith('hw3_task1_results_') and path.suffixes[-2:] == ['.tar', '.gz']:
            deleted.append({'path': str(path), 'size_bytes': path.stat().st_size})
            path.unlink()
else:
    keep = archives[-1] if archives else None
payload = {
    'run_id': 'task1_archive_cleanup',
    'status': 'finished',
    'updated_at': datetime.datetime.now().astimezone().isoformat(),
    'kept_archive': str(keep) if keep else None,
    'deleted_count': len(deleted),
    'deleted_archives': deleted,
}
(root/'status'/'task1_archive_cleanup.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
print(json.dumps(payload, indent=2))
PY

/root/miniconda3/bin/python - <<'PY'
from pathlib import Path
root = Path('/root/autodl-tmp/hw3_task1_3dgs_aigc')
section = """
## 2026-06-04 Final Completion Artifacts

Final report-oriented artifacts were generated after all long runs completed.

- `results/task1_final_report_artifacts.md/json`: final metric table, artifact index, and report notes.
- `results/report_assets/`: contact sheets for 3DGS GT/render comparisons and AIGC snapshots.
- `results/3dgs_metrics_summary.csv/json`: consolidated PSNR/SSIM/LPIPS for main scenes and ablations.
- `results/task1_artifact_index.json`: mesh/image/video/checkpoint artifact inventory and Blender availability.
- `status/task1_archive_cleanup.json`: cleanup record; only the latest curated result archive is retained under `results/`.

The final archive is produced by `scripts/09_collect_results.sh` and older duplicate archives are removed after the new archive is confirmed.
"""
readme = root/'README.md'
text = readme.read_text(encoding='utf-8', errors='replace') if readme.exists() else ''
if '2026-06-04 Final Completion Artifacts' not in text:
    readme.write_text(text.rstrip()+section+'\n', encoding='utf-8')
PY

write_chain_status finished 0 "Task1 final completion chain finished"
echo "[$(timestamp)] Task1 final completion chain finished"
