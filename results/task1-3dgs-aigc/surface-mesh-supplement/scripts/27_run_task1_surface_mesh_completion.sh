#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs status results/surface_mesh_exports results/report_assets
run_id="task1_surface_mesh_completion"; ts="$(date +%Y%m%d_%H%M%S)"; log_file="logs/${run_id}_${ts}.log"; status_file="status/${run_id}.json"; py="${PYTHON:-envs/threestudio/bin/python}"
write_status(){ local st="$1"; local rc="${2:-null}"; local note="${3:-}"; cat > "$status_file" <<JSON
{"run_id":"${run_id}","status":"${st}","exit_code":${rc},"updated_at":"$(date -Iseconds)","log_file":"${log_file}","note":"${note}"}
JSON
}
exec > >(tee -a "$log_file") 2>&1
trap 'rc=$?; if [ "$rc" -ne 0 ]; then write_status failed "$rc" "surface mesh completion failed; inspect log_file"; fi' EXIT
write_status running null "checking Python dependencies"; echo "[$(date -Iseconds)] Task1 surface mesh completion started"
[ -x "$py" ] || { echo "Python env not found: $py" >&2; exit 3; }
if ! "$py" - <<'PY' >/dev/null 2>&1
import open3d
PY
then
  write_status running null "installing open3d"
  echo "[$(date -Iseconds)] Installing Open3D into ${py} environment"
  "$py" -m pip install --cache-dir "$PWD/envs/pip_cache" -i https://pypi.tuna.tsinghua.edu.cn/simple "open3d==0.18.0"
fi
write_status running null "reconstructing Poisson surface meshes"
"$py" scripts/24_reconstruct_3dgs_surface_mesh.py --input-dir results/mesh_blender_exports/ply_samples --output-dir results/surface_mesh_exports --target-points 50000 --depth 8 --threads 8 --density-quantile 0.045 --max-triangles 220000
write_status running null "rendering surface meshes in Blender"
xvfb-run -a blender --background --python scripts/25_render_task1_surface_meshes.py -- --mesh-dir results/surface_mesh_exports/meshes --out-dir results/surface_mesh_exports/blender_renders --resolution-x 1400 --resolution-y 950
write_status running null "creating contact sheet"
"$py" scripts/26_make_surface_mesh_contact_sheet.py --render-dir results/surface_mesh_exports/blender_renders --out results/report_assets/3dgs_surface_mesh_contact.jpg --mirror-out results/surface_mesh_exports/3dgs_surface_mesh_contact.jpg
cat > results/surface_mesh_exports/README.md <<'MD'
# Task 1 Surface Mesh Supplement

This directory contains a geometry-focused supplement for HW3 Task 1. The meshes are reconstructed from sampled final 3DGS Gaussian centers for the Mip-NeRF 360 garden, bicycle, and counter scenes. They are intended as an additional surface/Blender visualization, while the primary 3DGS evaluation remains image-space novel-view synthesis with PSNR, SSIM, and LPIPS.

- `meshes/*_surface_mesh_poisson.ply`: colored triangular surface meshes reconstructed by Open3D Poisson reconstruction.
- `meshes/*_surface_mesh_poisson.obj`: OBJ geometry exports for compatibility checks.
- `reconstruction_point_clouds/*_surface_reconstruction_points.ply`: downsampled point clouds used for reconstruction.
- `blender_renders/<scene>/*.png`: Blender orthographic renders of each reconstructed mesh.
- `3dgs_surface_mesh_contact.jpg`: report-ready contact sheet.
- `task1_surface_mesh_manifest.json`: reconstruction parameters and mesh statistics.

These meshes are not used to replace 3DGS rendering metrics; they are included to make the geometric representation explicit for the assignment's Mesh/Blender-related requirement.
MD
write_status running null "packaging surface mesh supplement"
archive="results/hw3_task1_surface_mesh_supplement_$(date +%Y%m%d_%H%M).tar.gz"
tar -czf "$archive" results/surface_mesh_exports results/report_assets/3dgs_surface_mesh_contact.jpg status/task1_surface_mesh_completion.json "$log_file"
printf '%s\n' "$archive" > results/task1_surface_mesh_latest_package_path.txt
python3 - <<'PY'
from pathlib import Path
section='''
## 2026-06-04 Surface Mesh Supplement

A final geometry supplement was added for the three Mip-NeRF 360 3DGS scenes. The supplement reconstructs colored triangular surface meshes from sampled final 3DGS Gaussian centers and renders them in Blender. This is a visualization supplement for the Mesh/Blender requirement; the primary 3DGS comparison still uses novel-view rendering metrics.

Important files:

- `results/surface_mesh_exports/meshes/*_surface_mesh_poisson.ply`
- `results/surface_mesh_exports/blender_renders/`
- `results/report_assets/3dgs_surface_mesh_contact.jpg`
- `results/task1_surface_mesh_latest_package_path.txt`
'''
p=Path('README.md'); text=p.read_text(encoding='utf-8',errors='replace') if p.exists() else ''
if '2026-06-04 Surface Mesh Supplement' not in text: p.write_text(text.rstrip()+section+'\n',encoding='utf-8')
PY
write_status finished 0 "surface mesh supplement completed"
echo "[$(date -Iseconds)] Task1 surface mesh completion finished: ${archive}"
