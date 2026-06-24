# Task 1 AIGC Pipeline Records

本目录保留题目一 AIGC 生成与融合脚本、配置、输入图和命令记录，用于补充主结果包的复现入口和 prompt 记录。最终报告中的定量表格、图像和融合视频对应 `results/task1-3dgs-aigc/`、`tasks/task1_3dgs_aigc/object_a_reconstruction/`、`tasks/task1_3dgs_aigc/blender_fusion/` 和 `docs/report_assets/` 中已经整理好的文件。

## Scope

该补充包覆盖题目一的四个实现环节：

- Object A：真实物体多视角图像整理、COLMAP 和 3DGS 命令模板。
- Object B：threestudio / DreamFusion-style SDS 文本生成 3D 的 prompt 和训练命令模板；最终融合中使用的是梨形纹理网格。
- Object C：stable-zero123 单图生成 3D 的输入图和训练命令模板。
- Fusion：Blender 场景融合 manifest 和脚本模板。

完整第三方源码不放在本目录中。需要复现实验时，请将依赖框架 clone 或安装到：

```text
external/gaussian-splatting/
external/threestudio/
```

## Directory Layout

```text
aigc_pipeline_records/
|-- README.md
|-- requirements.txt
|-- configs/
|   `-- task1_pipeline_example.json
|-- scripts/
|   |-- hw3_pipeline.py
|   |-- prepare_object_a.py
|   |-- prepare_object_b.py
|   |-- prepare_object_c.py
|   `-- blender_fusion.py
|-- data/
|   `-- object_c_image3d/inputs/object_c_input.png
`-- outputs/
    |-- object_b_text3d/
    |   |-- object_b_summary.json
    |   |-- run_object_b_threestudio.ps1
    |   `-- export_object_b_mesh.ps1
    |-- object_c_zero123/
    |   |-- object_c_summary.json
    |   |-- run_object_c_zero123.ps1
    |   `-- export_object_c_mesh.ps1
    `-- fusion/blender_manifest.json
```

## Notes on Prompt Records

`configs/task1_pipeline_example.json` records the pear text-to-3D example used for the final Object B description, while `outputs/object_b_text3d/object_b_summary.json` preserves an alternate command snapshot generated in the experiment environment. Both files are useful prompt-engineering and command provenance. The final fusion video uses the pear-like textured mesh stored in `tasks/task1_3dgs_aigc/blender_fusion/assets/input-assets/`. This directory is supplementary reproducibility material.

## Local Helper Environment

The lightweight helper scripts only need Python utilities:

```bash
python -m pip install -r requirements.txt
```

Full training additionally requires COLMAP, Blender, 3D Gaussian Splatting, threestudio, and the corresponding diffusion/Zero123 weights.

## Reproduction Entry Points

Generate a command summary from the example config:

```bash
cd tasks/task1_3dgs_aigc/aigc_pipeline_records
python scripts/hw3_pipeline.py --config configs/task1_pipeline_example.json
```

Prepare Object A COLMAP/3DGS inputs:

```bash
python scripts/prepare_object_a.py --config configs/task1_pipeline_example.json
```

Generate Object B and Object C threestudio command templates:

```bash
python scripts/prepare_object_b.py --config configs/task1_pipeline_example.json
python scripts/prepare_object_c.py --config configs/task1_pipeline_example.json
```

Run the retained PowerShell templates after installing/cloning `external/threestudio`:

```powershell
powershell -ExecutionPolicy Bypass -File outputs/object_b_text3d/run_object_b_threestudio.ps1
powershell -ExecutionPolicy Bypass -File outputs/object_c_zero123/run_object_c_zero123.ps1
```

The export scripts contain placeholder `PATH_TO_PARSED_YAML` and `PATH_TO_LAST_CKPT` values; replace them with the trial's real `parsed.yaml` and `last.ckpt` after training finishes.
