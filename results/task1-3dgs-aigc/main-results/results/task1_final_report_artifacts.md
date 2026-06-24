# HW3 Task 1 Final Report Artifacts

Updated: 2026-06-04T09:48:45.368321+08:00

## 3DGS Metrics

| Variant | Scene | PSNR | SSIM | LPIPS |
|---|---|---:|---:|---:|
| bicycle_iter30000_r4 | bicycle | 25.6564693 | 0.7794347 | 0.2023928 |
| counter_iter30000_r4 | counter | 29.6144466 | 0.9289855 | 0.1006935 |
| garden_iter15000_r4 | garden | 27.3542805 | 0.8657238 | 0.1141754 |
| garden_iter30000_r4 | garden | 27.8552685 | 0.8751445 | 0.1026784 |
| garden_iter30000_r8 | garden | 29.7439518 | 0.9252081 | 0.0547893 |
| garden_iter7000_r4 | garden | 26.6160221 | 0.8394632 | 0.1527677 |

## Contact Sheets

- `results/report_assets/3dgs_bicycle_iter30000_r4_contact.jpg`
- `results/report_assets/3dgs_counter_iter30000_r4_contact.jpg`
- `results/report_assets/3dgs_garden_iter15000_r4_contact.jpg`
- `results/report_assets/3dgs_garden_iter30000_r4_contact.jpg`
- `results/report_assets/3dgs_garden_iter30000_r8_contact.jpg`
- `results/report_assets/3dgs_garden_iter7000_r4_contact.jpg`
- `results/report_assets/aigc_output_contact.jpg`
- `results/report_assets/3dgs_blender_pointcloud_contact.jpg`

## Notes

- The 3DGS route includes multi-scene reconstruction plus garden iteration/resolution ablations.
- The AIGC route includes text-to-3D SDS and image-to-3D Zero123 outputs through threestudio.
- Sampled final 3DGS point-cloud exports and Blender multi-view renders are stored in `results/mesh_blender_exports/`.
- Mesh/checkpoint/render/video artifacts are indexed in `results/task1_final_report_artifacts.json`.
