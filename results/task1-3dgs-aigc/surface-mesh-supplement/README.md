# HW3 Task 1: 3DGS and AIGC 3D Experiments

This workspace runs HW3 Task 1 experiments on the AutoDL server.

## Server

- Host: AutoDL / SeeTaCloud SSH endpoint
- GPU: NVIDIA GeForce RTX 4090 D, 24GB VRAM
- CPU/RAM observed: 128 CPU threads, 503GB RAM
- Persistent workspace: `/root/autodl-tmp/hw3_task1_3dgs_aigc`
- Important rule: keep data, environments, model caches, and outputs under `/root/autodl-tmp`.
- Container root disk is only about 30GB and should not store large files.

## Assignment Coverage

Task 1 is treated as a 3DGS + AIGC 3D pipeline:

1. COLMAP/3DGS reconstruction route using official 3D Gaussian Splatting code.
2. Mip-NeRF 360 scenes: `garden`, `bicycle`, and `counter` when time and disk allow.
3. Text-to-3D SDS route with `threestudio` using DreamFusion/Stable Diffusion config.
4. Image-to-3D route with Zero123/Stable-Zero123 through `threestudio` when model download is available.
5. Curated report artifacts: status JSON, commands, concise logs, metrics, rendered images/videos, and final README.

## Directory Layout

```text
scripts/      reproducible setup, download, train, render, collect scripts
logs/         stdout/stderr logs for each long job
status/       machine-readable status JSON files
third_party/  cloned upstream repositories
envs/         conda environments and package cache
data/         datasets and small input images
models/       Hugging Face / torch / checkpoint caches
outputs/      raw training outputs
results/      curated outputs and final archives
```

## Main Commands

Check server state:

```bash
cd /root/autodl-tmp/hw3_task1_3dgs_aigc
bash scripts/00_check_server.sh
```

Start the full chain in the background:

```bash
cd /root/autodl-tmp/hw3_task1_3dgs_aigc
nohup bash scripts/10_run_task1_chain.sh > logs/task1_full_chain.log 2>&1 & echo $! > status/task1_full_chain.pid
```

Check running jobs:

```bash
pgrep -af '10_run_task1_chain|02_setup_gaussian|03_download_mipnerf|04_run_3dgs|train.py|launch.py|aria2c|wget|curl|unzip'
cat status/task1_full_chain.json
```

## Experiment Matrix

| ID | Route | Dataset/Input | Output | Status file |
| --- | --- | --- | --- | --- |
| 3dgs_garden | 3DGS reconstruction | Mip-NeRF360 garden | `outputs/3dgs/garden_iter30000_r4` | `status/3dgs_garden.json` |
| 3dgs_bicycle | 3DGS reconstruction | Mip-NeRF360 bicycle | `outputs/3dgs/bicycle_iter30000_r4` | `status/3dgs_bicycle.json` |
| 3dgs_counter | 3DGS reconstruction | Mip-NeRF360 counter | `outputs/3dgs/counter_iter30000_r4` | `status/3dgs_counter.json` |
| text_to_3d | threestudio SDS | prompt: small red ceramic teapot | `outputs/threestudio` | `status/text_to_3d_*.json` |
| zero123 | Stable-Zero123 | generated RGBA teapot image | `outputs/threestudio` | `status/zero123_*.json` |

## Current Notes

- Mip-NeRF 360 zips are downloaded with resume support and removed after successful extraction.
- The 3DGS scenes use preprocessed COLMAP sparse data included in the downloaded Mip-NeRF 360 scene archives when available.
- Large raw outputs stay on the server. Only curated artifacts are packed and downloaded locally.
- Temporary files such as incomplete downloads, pycache folders, and scratch files are cleaned by `scripts/09_collect_results.sh`.

## 2026-06-03 Extension Notes

Additional report-strengthening checks were added while the main chain was running:

- `scripts/11_colmap_scene_check.sh`: records whether each Mip-NeRF360 scene contains COLMAP sparse camera/point files.
- `scripts/12_export_artifact_check.sh`: indexes mesh, image, and video outputs and records Blender availability.
- `scripts/13_analyze_task1_results.py`: writes `results/task1_progress_summary.json`.
- `scripts/08_run_zero123_image_to_3d.sh`: tries Stable-Zero123 first, then Zero123-XL fallback if the Stable-Zero123 checkpoint is unavailable.

## 2026-06-03 Additional Task 1 Extension Queue

The main Task 1 chain remains the priority. A second queue waits for `task1_full_chain` to finish and then runs lightweight 3DGS ablations for report-quality comparisons.

Additional experiments:

| Run | Purpose | Runtime Estimate |
| --- | --- | --- |
| `garden_iter7000_r4` | Early 3DGS checkpoint quality/time point | about 15-40 minutes |
| `garden_iter15000_r4` | Mid-training quality/time point | about 30-80 minutes |
| `garden_iter30000_r8` | Lower-resolution training comparison against the main `garden_iter30000_r4` run | about 30-80 minutes |

New files and outputs:

- `scripts/14_run_3dgs_ablation_queue.sh`: waits for the main chain, then runs the ablations sequentially.
- `scripts/15_extract_3dgs_metrics.py`: parses 3DGS metrics into `results/3dgs_metrics_summary.csv` and `results/3dgs_metrics_summary.json`.
- `scripts/04_run_3dgs_scene.sh` and `scripts/05_render_3dgs_scene.sh`: status/result naming now includes `iter` and `r` so ablations do not overwrite main-scene result files.
- Queue status: `status/task1_3dgs_ablation_queue.json`; queue PID: `status/task1_3dgs_ablation_queue.pid`; queue log: `logs/task1_3dgs_ablation_queue.log`.

This queue is expected to add about 2-4 hours after the main Task 1 chain, depending on 3DGS training speed on the 4090D.

## 2026-06-03 Resume Fix: 3DGS Render Status

During `garden_iter30000_r4`, metrics completed successfully but the shell pipeline that copied a bounded set of result files exited with SIGPIPE under `set -o pipefail`. The scripts now avoid that SIGPIPE and reuse existing 3DGS checkpoints/render metrics when resuming, so a restarted chain continues from completed work instead of retraining `garden`.

Recorded `garden_iter30000_r4` metrics: PSNR 27.8553, SSIM 0.8751, LPIPS 0.1027.

The metrics summary parser was also updated to parse official 3DGS output lines such as `PSNR : value`, `SSIM : value`, and `LPIPS: value`.

## 2026-06-03 Resume Fix: threestudio Dependency Build

- Fixed `scripts/06_setup_threestudio.sh` to install threestudio requirements with `--no-build-isolation`.
- Root cause: `nerfacc==v0.5.2` imports `torch` from `setup.py`; isolated PEP 517 builds could not see the torch wheel already installed in the threestudio conda environment.
- After this change, the Task1 chain can resume from completed 3DGS scenes and continue with text-to-3D / Zero123 stages.

## 2026-06-03 Resume Fix: threestudio Network Retry

- Added retry/backoff around `scripts/06_setup_threestudio.sh` requirements installation.
- Configured Git to prefer HTTP/1.1 for the `nerfacc` dependency clone after GitHub HTTP/2 framing/time-out failures on AutoDL.
- Failed pip build directories under `tmp/` are cleaned before retry so restarts do not reuse partial checkouts.

## 2026-06-03 Resume Fix: pysdf Build Dependency

- Added explicit `pybind11` installation before threestudio requirements.
- Root cause: `pysdf` builds from source under `--no-build-isolation`; its build step imports `pybind11` from the active environment.

## 2026-06-03 Resume Fix: Longer threestudio Retry Window

- Increased `scripts/06_setup_threestudio.sh` requirements retries from 5 to 12 with capped backoff.
- Reason: GitHub VCS dependencies such as `nvdiffrast` and `tiny-cuda-nn` intermittently time out from AutoDL; longer retry keeps the chain alive instead of stopping on a transient network failure.

## 2026-06-04 Resume Fix: pysdf Eigen Headers

- Added `eigen` installation via conda before threestudio requirements.
- Exported `CPLUS_INCLUDE_PATH` and `CPATH` to include `$ENV_PREFIX/include/eigen3`, because `pysdf` includes headers as `Eigen/Core` during source builds.
- Added a system include symlink fallback when system Eigen headers are already present, so an already-running pip build has a better chance to reach `pysdf` without another restart.

## 2026-06-04 Resume Fix: libigl API Compatibility

- Patched `third_party/threestudio/threestudio/utils/ops.py` for `libigl 2.6.x` API changes.
- `fast_winding_number_for_meshes` now falls back to `igl.fast_winding_number`.
- `read_obj` now falls back to `trimesh.load(..., force="mesh")` and returns the 6-value tuple expected by the original threestudio code.

## 2026-06-04 Resume Fix: diffusers / huggingface_hub Compatibility

- Pinned `huggingface_hub==0.19.4` after threestudio requirements installation.
- Root cause: `diffusers 0.19.x` imports `cached_download`, which is removed from newer `huggingface_hub` releases.

## 2026-06-04 Resume Fix: CUDA Conda Activate under nounset

- Wrapped threestudio conda activation and conda package installs with `set +u` / `set -u` guards.
- Initialized `NVCC_PREPEND_FLAGS` and `NVCC_APPEND_FLAGS` before CUDA toolkit activation scripts run.
- Root cause: the `cuda-nvcc` activate script reads these variables, and `set -u` turned an unset optional variable into a hard failure.

## 2026-06-04 Resume Fix: DreamFusion Smoke Config Overrides

- Removed `system.freq.*` overrides from the DreamFusion text-to-3D smoke run.
- Replaced them with supported trainer/checkpoint fields: `trainer.val_check_interval` and `checkpoint.every_n_train_steps`.
- Root cause: `freq` is used by Zero123 configs, but the current `dreamfusion-system` config does not define `system.freq`.


## 2026-06-04 Resume Fix: Hugging Face Mirror for AIGC Models

- Added explicit Hugging Face mirror/default timeout exports to `scripts/07_run_text_to_3d.sh` and `scripts/08_run_zero123_image_to_3d.sh`.
- Root cause: the DreamFusion prompt embedding subprocess tried to reach `huggingface.co` directly for `stabilityai/stable-diffusion-2-1-base` and timed out before the checkpoint/tokenizer cache existed locally.
- Failed smoke-output folders from the timeout-only run are treated as intermediate artifacts; result-worthy files remain under `results/`.
## 2026-06-04 Resume Fix: Public SD v1.5 Model for DreamFusion

- Changed `scripts/07_run_text_to_3d.sh` to override DreamFusion prompt/guidance model with `runwayml/stable-diffusion-v1-5` by default.
- Root cause: the default `stabilityai/stable-diffusion-2-1-base` endpoint returned 404 on the available Hugging Face mirror, while `runwayml/stable-diffusion-v1-5` probe succeeded.
- Removed failed smoke directories that contained only copied source/config/log artifacts and no successful AIGC result media.
## 2026-06-04 Resume Fix: HF Large-File Timeout for AIGC Models

- Added `HF_HUB_DISABLE_XET=1`, disabled hf-transfer, and raised Hugging Face download/etag timeouts to 300 seconds in the text-to-3D and Zero123 scripts.
- Root cause: SD v1.5 reached model download but a large-file request through `cas-bridge.xethub.hf.co` timed out before the model cache completed.
- The failed `smoke@20260604-033823` directory only contained copied code/config/log artifacts and was removed before restarting.
## 2026-06-04 Resume Fix: Zero123 PyTorch Checkpoint Loading

- Added `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` to `scripts/08_run_zero123_image_to_3d.sh`.
- Root cause: PyTorch 2.6+ defaults checkpoint loading toward weights-only mode; the Zero123 checkpoint contains Lightning callback metadata and failed with an allowlist error.
- Added `scripts/16_run_zero123_then_ablation_resume.sh` to run Zero123 smoke/full first and then the queued 3DGS ablations without GPU contention.
## 2026-06-04 Resume Chain: Remaining Ablations then Zero123

- Added `scripts/17_run_remaining_ablation_then_zero123.sh` after the old ablation waiter had already launched `garden_iter7000_r4`.
- The new chain waits for the active 3DGS ablation process to finish, resumes the queued ablations, then runs the patched Zero123 smoke/full route.

## 2026-06-04 Resume Fix: Zero123 24GB OOM Recovery

- Added `third_party/threestudio/configs/hw3_zero123_24gb.yaml` for the Zero123-XL fallback on the RTX 4090D 24GB GPU.
- The reduced config keeps the same image-to-3D route but lowers random-camera batch size to 1, caps progressive render resolution at 128, lowers validation/test views, and reduces NeRF volume samples per ray.
- Patched `zero123_guidance.py` to call `torch.load(..., weights_only=False)` explicitly for the trusted local Zero123 checkpoint instead of relying only on an environment override.
- Added `scripts/18_run_zero123_only_resume.sh` so Zero123 smoke/full can be retried independently after all 3DGS runs and ablations have already completed.
- Root cause: Zero123-XL loaded successfully, but the default config started with random-camera batch size 12 and exhausted the 24GB GPU before the first optimization step.

## 2026-06-04 Resume Fix: Zero123 Config Milestones

- Corrected `hw3_zero123_24gb.yaml` input image resolution to scalar `128` with empty `resolution_milestones`.
- Root cause: threestudio requires `len(height) == len(resolution_milestones) + 1` when height/width are lists; the 24GB config intentionally does not progressively raise input supervision resolution.
- Removed the failed assertion-only Zero123 smoke output directory before restarting the dedicated Zero123-only chain.
## 2026-06-04 Final Completion Artifacts

Final report-oriented artifacts were generated after all long runs completed.

- `results/task1_final_report_artifacts.md/json`: final metric table, artifact index, and report notes.
- `results/report_assets/`: contact sheets for 3DGS GT/render comparisons and AIGC snapshots.
- `results/3dgs_metrics_summary.csv/json`: consolidated PSNR/SSIM/LPIPS for main scenes and ablations.
- `results/task1_artifact_index.json`: mesh/image/video/checkpoint artifact inventory and Blender availability.
- `status/task1_archive_cleanup.json`: cleanup record; only the latest curated result archive is retained under `results/`.

The final archive is produced by `scripts/09_collect_results.sh` and older duplicate archives are removed after the new archive is confirmed.## 2026-06-04 Surface Mesh Supplement

A final geometry supplement was added for the three Mip-NeRF 360 3DGS scenes. The supplement reconstructs colored triangular surface meshes from sampled final 3DGS Gaussian centers and renders them in Blender. This is a visualization supplement for the Mesh/Blender requirement; the primary 3DGS comparison still uses novel-view rendering metrics.

Important files:

- `results/surface_mesh_exports/meshes/*_surface_mesh_poisson.ply`
- `results/surface_mesh_exports/blender_renders/`
- `results/report_assets/3dgs_surface_mesh_contact.jpg`
- `results/task1_surface_mesh_latest_package_path.txt`
