#!/usr/bin/env bash
set -Eeuo pipefail
cd /root/autodl-tmp/hw3_task1_3dgs_aigc
source scripts/_common.sh
run_id="export_artifact_check"
status_file="$HW3_TASK1_ROOT/status/${run_id}.json"
write_status "$run_id" running null "checking mesh/video/image export artifacts"
python - "$status_file" <<'PY'
import json, pathlib, shutil, subprocess, datetime, os
root=pathlib.Path(os.environ["HW3_TASK1_ROOT"])
patterns=["*.ply","*.obj","*.glb","*.mp4","*.png","*.jpg","*.jpeg"]
artifacts=[]
for base in [root/"outputs", root/"results"]:
    if not base.exists():
        continue
    for pattern in patterns:
        for path in base.rglob(pattern):
            if path.is_file():
                try:
                    size=path.stat().st_size
                except OSError:
                    size=0
                artifacts.append({"path": str(path), "size": size})
artifacts=sorted(artifacts, key=lambda x: x["path"])[:500]
payload={
    "run_id":"export_artifact_check",
    "status":"finished",
    "exit_code":0,
    "updated_at":datetime.datetime.now().astimezone().isoformat(),
    "blender_executable": shutil.which("blender"),
    "artifact_count": len(artifacts),
    "artifacts": artifacts,
    "note": "Blender is available for headless rendering." if shutil.which("blender") else "Blender executable not found; report will use generated renders/videos/PLY/OBJ if present.",
}
(root/"results"/"task1_artifact_index.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
pathlib.Path(__import__("sys").argv[1]).write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(json.dumps({k: payload[k] for k in ["run_id","status","artifact_count","blender_executable","note"]}, indent=2))
PY
