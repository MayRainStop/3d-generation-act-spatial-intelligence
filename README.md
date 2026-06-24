# 深度学习与空间智能 HW3

本仓库整理作业 3 的最终提交材料，包含任务 1 的 A/B/C 三维物体构建、Mip-NeRF360 场景重建、Blender 场景融合，以及任务 2 的 LeRobot ACT 跨环境模仿学习实验。目录按报告、任务代码、结果包和复现命令分开存放；GitHub 仓库只保留轻量代码、报告和复现脚本，大型数据、视频、mesh 和模型权重通过本地备份或外部存储提交。

## 报告与提交链接

实验报告：

- `docs/report.pdf`
- GitHub 项目地址：[MayRainStop/3d-generation-act-spatial-intelligence](https://github.com/MayRainStop/3d-generation-act-spatial-intelligence)
- 模型权重网盘地址：[Google Drive](https://drive.google.com/drive/folders/14n7YY6TVmrlqI1PMlGP65uwwRv75a2do?usp=drive_link)

其他报告文件：

- LaTeX 源文件：`docs/report.tex`
- 报告插图：`docs/report_assets/`
- 作业要求原文：`HW3_深度学习与空间智能.pdf`

## Project Contents

| Part | Topic | Local status |
| --- | --- | --- |
| Task 1 | Object A/B/C construction | A 为手机多视角湿巾 COLMAP+3DGS，B 为梨形 text-to-3D，C 为单图 image-to-3D |
| Task 1 | Mip-NeRF360 scene reconstruction | `garden`、`bicycle`、`counter` 已训练、评测并整理可视化 |
| Task 1 | Blender fusion | 重建 `counter` 背景中融合物体 A 湿巾、梨形物体 B 和单图生成物体 C |
| Task 1 | Supplementary AIGC route | 保留 text-to-3D / image-to-3D 流程记录，用于说明生成资产来源 |
| Task 2 | LeRobot ACT on CALVIN | A-only、ABC-mixed、course-provided split、seed/checkpoint 扩展实验均已整理 |

## Directory Layout

```text
.
|-- README.md
|-- environment.yml
|-- requirements.txt
|-- HW3_深度学习与空间智能.pdf
|-- docs/
|   |-- report.tex
|   |-- report.pdf
|   |-- report_assets/
|   `-- experiment1_3d_assets/     # Task 1 report notes and lightweight assets
|-- tasks/
|   |-- task1_3dgs_aigc/
|   |   |-- README.md
|   |   |-- object_a_reconstruction/
|   |   |   |-- notebooks/
|   |   |   |-- data/                  # local only: phone images + COLMAP workspace
|   |   |   `-- outputs/               # local only: object A 3DGS outputs
|   |   |-- blender_fusion/
|   |   |   |-- notebooks/
|   |   |   |-- scripts/
|   |   |   |-- assets/input-assets/    # local only: meshes/textures for fusion
|   |   |   `-- outputs/               # local only: rendered walkthrough videos
|   |   `-- aigc_pipeline_records/      # Task 1 helper scripts and prompt records
|   `-- task2_lerobot_act/
|       |-- README.md
|       |-- environment.yml
|       |-- configs/                    # Task 2 run table
|       |-- scripts/                    # Task 2 training/evaluation/retrieval scripts
|       `-- tests/                      # lightweight command and parsing tests
|-- results/
|   |-- task1-3dgs-aigc/
|   `-- task2-lerobot-act/
`-- tools/
    |-- plot_task1_loss_curves.py
    |-- plot_task1_object_fusion_figures.py
    |-- plot_task2_loss_curves.py
    |-- plot_task2_report_figures.py
    `-- ...
```

## Environment

For the report, plotting scripts, and result-table regeneration on a local machine:

```bash
python -m pip install -r requirements.txt
```

For a Linux/CUDA training environment, use the Conda environment file:

```bash
conda env create -f environment.yml
conda activate hw3_task2_lerobot
python -m pip install -r requirements.txt
```

The full training environments are heavier than the local report package. The task code and runnable entrypoints are under:

```text
results/task1-3dgs-aigc/main-results/scripts/
tasks/task2_lerobot_act/scripts/
tasks/task2_lerobot_act/configs/
tasks/task2_lerobot_act/tests/
```

Full third-party framework source trees such as 3D Gaussian Splatting and threestudio are intentionally not vendored in the final folder. Clone or install them separately when rerunning Task 1 training.

## Task 1 Summary

### Object A/B/C Construction

Task 1 follows the assignment order. First, it builds three object assets with three different input modes:

- Object A: a phone-captured tissue/wet-wipe package reconstructed from 73 images with COLMAP and 3DGS.
- Object B: a pear-like textured mesh generated from a text prompt with threestudio SDS.
- Object C: a single-image image-to-3D mesh generated from a red/yellow fruit reference.

The report includes a compact construction figure that shows the input source, generated 3D asset, and final video evidence for A/B/C.

### Mip-NeRF360 Scene Reconstruction

The scene-level 3DGS experiments use Mip-NeRF360 scenes and official COLMAP camera/sparse point inputs. Main quantitative results:

| Variant | Scene | PSNR | SSIM | LPIPS |
| --- | --- | ---: | ---: | ---: |
| `counter_iter30000_r4` | counter | 29.614 | 0.9290 | 0.1007 |
| `garden_iter30000_r4` | garden | 27.855 | 0.8751 | 0.1027 |
| `bicycle_iter30000_r4` | bicycle | 25.656 | 0.7794 | 0.2024 |
| `garden_iter30000_r8` | garden | 29.744 | 0.9252 | 0.0548 |
| `garden_iter15000_r4` | garden | 27.354 | 0.8657 | 0.1142 |
| `garden_iter7000_r4` | garden | 26.616 | 0.8395 | 0.1528 |

### Blender Fusion

The fusion stage inserts all three generated/reconstructed objects into the reconstructed `counter` scene in Blender:

- Object A training convergence: 3DGS total loss decreases from 0.1894 to 0.0186; L1 loss decreases from 0.1790 to 0.0136.
- Exported object A 3DGS point cloud: 146445 Gaussian centers.
- Background: reconstructed `counter` surface mesh.
- Object fusion: object A tissue mesh, object B pear-like textured mesh, and object C single-image mesh inserted into the reconstructed background in Blender.
- Final walkthrough: `object_fusion_orbit_v5.mp4`, 150 frames, 24 fps, 1280x720.
- Additional AIGC route examples are kept as method records for text-to-3D and image-to-3D generation.

Regenerate Task 1 report figures:

```bash
python tools/plot_task1_loss_curves.py
python tools/plot_task1_object_fusion_figures.py
```

Run the final Blender fusion render if Blender is installed:

```bash
blender -b -P tasks/task1_3dgs_aigc/blender_fusion/scripts/render_colored_orbit_v5.py
```

## Task 2 Summary

Task 2 evaluates ACT policies on CALVIN D zero-shot transfer. The final report treats the course-provided `splitA/splitB/splitC/splitD` data as the clean main comparison, and the ABC extended training run as a supplementary checkpoint-selection result.

### Course-Provided Split Main Results

| Run | Condition | Step | Avg Len | Len1 | Len2 | Len3 | Len4 | Len5 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| splitA A-only seed0 | one-hot | 100k | 0.855 | 0.525 | 0.225 | 0.072 | 0.025 | 0.008 |
| splitABC seed0 | one-hot | 100k | 0.649 | 0.402 | 0.165 | 0.052 | 0.021 | 0.009 |
| splitABC seed0 | SBERT | 100k | 0.787 | 0.477 | 0.195 | 0.071 | 0.032 | 0.012 |
| splitABC seed0 | chunk100 | 100k | 0.696 | 0.457 | 0.162 | 0.052 | 0.017 | 0.008 |

### Optional Robustness / Sweep Results

| Run | Step | Avg Len | Len1 | Len2 | Len3 | Len4 | Len5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| splitA A-only seed1 | 100k | 0.846 | 0.487 | 0.219 | 0.088 | 0.041 | 0.011 |
| splitABC SBERT seed1 | 100k | 0.923 | 0.496 | 0.235 | 0.112 | 0.059 | 0.021 |
| splitABC SBERT seed0 | 160k | 0.915 | 0.483 | 0.261 | 0.104 | 0.045 | 0.022 |
| ABC extended 120k | 120k ckpt | 1.028 | 0.561 | 0.277 | 0.119 | 0.049 | 0.022 |

Task 2 local result package:

```text
tasks/task2_lerobot_act/
results/task2-lerobot-act/final-results/
results/task2-lerobot-act/final-results/model-weights/
```

### Task 2 Reproduction Commands

The commands below expect a Linux/CUDA workspace with raw CALVIN data and checkpoints stored outside Git. In this repository, the runnable Task 2 entrypoints are under `tasks/task2_lerobot_act/`; large `data/`, `outputs/`, checkpoints, logs, and videos are intentionally excluded from Git and provided through the Google Drive package.

Prepare the Python environment:

```bash
conda env create -f tasks/task2_lerobot_act/environment.yml
conda activate hw3_task2_lerobot
python -m pip install -r requirements.txt
```

Reproduce the course-provided split main chain:

```bash
cd tasks/task2_lerobot_act
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export WANDB_MODE=offline
bash scripts/44_run_task2_ta_official_split_chain.sh
```

This chain downloads `xiaoma26/calvin-lerobot` splitA/splitB/splitC/splitD, converts LeRobot v2.1 data to v3.0, trains A-only and ABC-mixed ACT variants, evaluates on CALVIN D, and writes audit files, summaries, checkpoints, and evaluation outputs under the local `results/` and `outputs/` folders.

Run the optional checkpoint sweep and seed1 robustness chain:

```bash
cd tasks/task2_lerobot_act
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export WANDB_MODE=offline
bash scripts/46_run_task2_ta_sweep_seed1_chain.sh
```

This resumes or trains the seed/checkpoint extension runs and refreshes the sweep summaries used by the final report.

Evaluate a specific trained checkpoint on CALVIN D:

```bash
cd tasks/task2_lerobot_act
bash scripts/45_run_ta_conditioned_eval.sh \
  sbert \
  act_ta_abc_lang_sbert_100k_seed1 \
  outputs/20260606_ta_official_split/act_ta_abc_lang_sbert_100k_seed1/checkpoints/100000/pretrained_model \
  full
```

Regenerate Task 2 plots:

```bash
python tools/plot_task2_report_figures.py
python tools/plot_task2_loss_curves.py
```

## Reproducibility Notes

- Raw datasets, checkpoints, rendered videos, and large mesh assets are intentionally kept out of Git through `.gitignore`.
- Report-ready summaries, scripts, logs, selected model weights, tables, and figures are present locally.
- Large local-only data, mesh, video, and checkpoint assets are provided through the Google Drive link at the top of this README and on the report title page.

## Final Submission Checklist

- `docs/report.pdf` is the final report.
- `docs/report.tex` is the editable report source.
- `README.md` documents layout, results, and reproduction commands.
- Task 1 now includes scene 3DGS, AIGC generation, real object A reconstruction, and A/B/C Blender fusion.
- Task 2 now has its own code workspace under `tasks/task2_lerobot_act/`, plus clean official split results, public high-score supplement, seed/checkpoint robustness, loss curves, and reproduction scripts.
