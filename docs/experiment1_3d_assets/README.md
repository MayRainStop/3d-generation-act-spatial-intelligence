# Task 1 Deliverable Index

This directory is a lightweight index for Task 1 assets. It does not duplicate raw datasets, rendered frame dumps, or large external source trees. The actual files remain in `tasks/`, `results/`, and `docs/report_assets/`.

## Report-Ready Figures

| Item | Path | Purpose |
| --- | --- | --- |
| Scene 3DGS contact sheets | `docs/report_assets/3dgs_*_contact.jpg` | Qualitative renders for `garden`, `bicycle`, and `counter` scene reconstruction. |
| Surface mesh contact sheet | `docs/report_assets/3dgs_surface_mesh_contact.jpg` | Comparison of reconstructed surface meshes exported from 3DGS points. |
| AIGC 3D contact sheet | `docs/report_assets/aigc_output_contact.jpg` | Text-to-3D and image-to-3D generated object examples used to document the generation pipeline. |
| Zero123 input reference | `docs/report_assets/task1_object_c_zero123_input.png` | Single-image input retained from the Object C image-to-3D pipeline records. |
| Object A reconstruction | `docs/report_assets/task1_object_a_reconstruction.png` | Phone-captured object A, COLMAP/3DGS reconstruction, and exported point cloud overview. |
| Object A loss curve | `docs/report_assets/task1_object_a_loss_curve.png` | TensorBoard-derived training loss curve for the Object A 3DGS run. |
| Blender fusion frames | `docs/report_assets/task1_blender_fusion_video_frames.png` | Key frames from the final A/B/C object fusion walkthrough. |

## Scene Reconstruction Assets

| Item | Path |
| --- | --- |
| Metrics table | `results/task1-3dgs-aigc/main-results/results/3dgs_metrics_summary.csv` |
| Main Task 1 result notes | `results/task1-3dgs-aigc/main-results/README.md` |
| Surface mesh notes | `results/task1-3dgs-aigc/surface-mesh-supplement/README.md` |
| Garden mesh | `results/task1-3dgs-aigc/surface-mesh-supplement/results/surface_mesh_exports/meshes/garden_surface_mesh_poisson.obj` |
| Bicycle mesh | `results/task1-3dgs-aigc/surface-mesh-supplement/results/surface_mesh_exports/meshes/bicycle_surface_mesh_poisson.obj` |
| Counter mesh | `results/task1-3dgs-aigc/surface-mesh-supplement/results/surface_mesh_exports/meshes/counter_surface_mesh_poisson.obj` |
| Mesh render folder | `results/task1-3dgs-aigc/surface-mesh-supplement/results/surface_mesh_exports/blender_renders/` |

## AIGC 3D Assets

| Item | Path |
| --- | --- |
| Text-to-3D outputs | `results/task1-3dgs-aigc/main-results/results/threestudio_text_to_3d/` |
| Image-to-3D outputs | `results/task1-3dgs-aigc/main-results/results/zero123_image_to_3d/` |
| Fusion object B pear-like mesh | `tasks/task1_3dgs_aigc/blender_fusion/assets/input-assets/obj_b.obj` |
| Fusion object B pear-like texture | `tasks/task1_3dgs_aigc/blender_fusion/assets/input-assets/obj_b_text.jpg` |
| Fusion object C mesh | `tasks/task1_3dgs_aigc/blender_fusion/assets/input-assets/obj_c.obj` |
| Fusion object C texture | `tasks/task1_3dgs_aigc/blender_fusion/assets/input-assets/obj_c_text.jpg` |
| AIGC pipeline records | `tasks/task1_3dgs_aigc/aigc_pipeline_records/` |
| Object C Zero123 input | `tasks/task1_3dgs_aigc/aigc_pipeline_records/data/object_c_image3d/inputs/object_c_input.png` |
| Object B prompt records | `tasks/task1_3dgs_aigc/aigc_pipeline_records/configs/task1_pipeline_example.json`, `tasks/task1_3dgs_aigc/aigc_pipeline_records/outputs/object_b_text3d/object_b_summary.json` |
| Object C command record | `tasks/task1_3dgs_aigc/aigc_pipeline_records/outputs/object_c_zero123/object_c_summary.json` |

## Real Object A and Fusion Assets

| Item | Path |
| --- | --- |
| Object A local package | `tasks/task1_3dgs_aigc/object_a_reconstruction/` |
| Object A phone captures | `tasks/task1_3dgs_aigc/object_a_reconstruction/data/object_a_data/images/` |
| Object A 3DGS point cloud | `tasks/task1_3dgs_aigc/object_a_reconstruction/outputs/object_a_3dgs/point_cloud/iteration_7000/point_cloud.ply` |
| Object A loss CSV | `results/task1-3dgs-aigc/main-results/results/task1_object_a_3dgs_loss_curve.csv` |
| Object A fusion mesh | `tasks/task1_3dgs_aigc/blender_fusion/assets/input-assets/tissue.obj` |
| Object A fusion point cloud | `tasks/task1_3dgs_aigc/blender_fusion/assets/input-assets/tissue.ply` |
| Fusion background mesh | `tasks/task1_3dgs_aigc/blender_fusion/assets/input-assets/counter_surface_mesh_poisson.obj` |
| Final Blender render script | `tasks/task1_3dgs_aigc/blender_fusion/scripts/render_colored_orbit_v5.py` |
| Final fusion video | `tasks/task1_3dgs_aigc/blender_fusion/outputs/object_fusion_orbit_v5.mp4` |

## Regeneration Commands

Regenerate report figures:

```bash
python tools/plot_task1_loss_curves.py
python tools/plot_task1_object_fusion_figures.py
```

Render the final fusion video with Blender:

```bash
blender -b -P tasks/task1_3dgs_aigc/blender_fusion/scripts/render_colored_orbit_v5.py
```

Large raw inputs and training outputs are intentionally local-only. For GitHub submission, keep this index and upload large assets to external storage if the course submission requires them.
