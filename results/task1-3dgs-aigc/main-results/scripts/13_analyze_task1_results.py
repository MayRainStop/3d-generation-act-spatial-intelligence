#!/usr/bin/env python
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import datetime

ROOT = Path(os.environ.get("HW3_TASK1_ROOT", "/root/autodl-tmp/hw3_task1_3dgs_aigc"))
OUT = ROOT / "results" / "task1_progress_summary.json"


def du(path: Path) -> str:
    if not path.exists():
        return "missing"
    try:
        return subprocess.check_output(["du", "-sh", str(path)], text=True).split()[0]
    except Exception:
        return "unknown"


summary = {
    "run_id": "task1_progress_summary",
    "updated_at": datetime.datetime.now().astimezone().isoformat(),
    "workspace": str(ROOT),
    "sizes": {
        "data": du(ROOT / "data"),
        "third_party": du(ROOT / "third_party"),
        "envs": du(ROOT / "envs"),
        "models": du(ROOT / "models"),
        "outputs": du(ROOT / "outputs"),
        "results": du(ROOT / "results"),
    },
    "status_files": {},
    "metrics_files": [],
    "important_artifacts": [],
}
for path in sorted((ROOT / "status").glob("*.json")):
    try:
        summary["status_files"][path.name] = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        summary["status_files"][path.name] = {"error": repr(exc)}
for pattern in ["metrics.txt", "*summary*.json", "*.mp4", "*.ply", "*.obj", "*.glb"]:
    for path in sorted((ROOT / "results").rglob(pattern)):
        if path.is_file():
            summary["important_artifacts"].append(str(path))
for path in sorted((ROOT / "results").rglob("metrics.txt")):
    summary["metrics_files"].append(str(path))
OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps({"summary": str(OUT), "artifact_count": len(summary["important_artifacts"])}, indent=2))
