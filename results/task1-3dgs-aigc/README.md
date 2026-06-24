# HW3 Task 1 Result Package

This directory stores the final Task 1 artifacts downloaded from the AutoDL server. The real object A reconstruction and Blender object-fusion files are stored under `tasks/task1_3dgs_aigc/` because they include local notebooks, raw phone captures, meshes, and rendered videos.

## Contents

```text
main-results/
surface-mesh-supplement/
report-assets/
```

## Main 3DGS / AIGC Artifacts

```text
main-results/results/3dgs_metrics_summary.csv
main-results/results/task1_final_report_artifacts.md
main-results/results/task1_final_report_artifacts.json
main-results/results/task1_artifact_index.json
main-results/results/mesh_blender_exports/
main-results/results/report_assets/
main-results/results/threestudio_text_to_3d/
main-results/results/zero123_image_to_3d/
surface-mesh-supplement/results/surface_mesh_exports/
```

## Added Object A / Fusion Artifacts

The strict Task 1 object and scene-fusion supplement is organized outside this result package:

```text
tasks/task1_3dgs_aigc/object_a_reconstruction/
tasks/task1_3dgs_aigc/blender_fusion/
docs/report_assets/task1_object_a_reconstruction.png
docs/report_assets/task1_blender_fusion_video_frames.png
```

Object A uses 73 phone-captured images, COLMAP sparse reconstruction, and a 7000-iteration 3DGS export. The final Blender fusion video places objects A/B/C into the reconstructed `counter` background scene and renders a 150-frame orbit video.

## Verification Notes

The original compressed packages were verified and then removed. The extracted directories are the retained final local copies. Large raw datasets, checkpoints, meshes, and videos should be submitted through cloud links rather than committed directly to GitHub.
