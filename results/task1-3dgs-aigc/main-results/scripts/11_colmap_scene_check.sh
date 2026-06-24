#!/usr/bin/env bash
set -Eeuo pipefail
cd /root/autodl-tmp/hw3_task1_3dgs_aigc
source scripts/_common.sh
scene="${1:-garden}"
run_id="colmap_check_${scene}"
scene_dir="$HW3_TASK1_ROOT/data/mipnerf360/$scene"
status_file="$HW3_TASK1_ROOT/status/${run_id}.json"
write_status "$run_id" running null "checking COLMAP/sparse evidence for $scene"
python - "$scene" "$scene_dir" "$status_file" <<'PY'
import json, pathlib, shutil, subprocess, sys, datetime
scene=sys.argv[1]
scene_dir=pathlib.Path(sys.argv[2])
status_file=pathlib.Path(sys.argv[3])
sparse_candidates = [scene_dir/"sparse"/"0", scene_dir/"sparse"]
image_dirs = [scene_dir/"images", scene_dir/"images_4", scene_dir/"images_2"]
payload = {
    "run_id": f"colmap_check_{scene}",
    "status": "finished",
    "exit_code": 0,
    "updated_at": datetime.datetime.now().astimezone().isoformat(),
    "scene": scene,
    "scene_dir": str(scene_dir),
    "colmap_executable": shutil.which("colmap"),
    "note": "",
}
payload["scene_exists"] = scene_dir.exists()
payload["image_dirs"] = {str(p): (len(list(p.glob("*"))) if p.exists() else 0) for p in image_dirs}
payload["sparse_dirs"] = {}
for p in sparse_candidates:
    payload["sparse_dirs"][str(p)] = {
        "exists": p.exists(),
        "files": sorted(x.name for x in p.glob("*"))[:30] if p.exists() else [],
    }
payload["has_preprocessed_sparse"] = any(
    info["exists"] and any(name.startswith(("cameras", "images", "points3D")) for name in info["files"])
    for info in payload["sparse_dirs"].values()
)
if payload["has_preprocessed_sparse"]:
    payload["note"] = "Mip-NeRF360 scene includes preprocessed COLMAP sparse camera/point files; 3DGS can train directly."
elif payload["colmap_executable"]:
    payload["note"] = "No sparse files found, but COLMAP executable exists; manual COLMAP reconstruction could be run if needed."
else:
    payload["note"] = "No sparse files found and COLMAP executable is absent; install COLMAP or use preprocessed dataset."
status_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(json.dumps(payload, indent=2))
PY
