# Task 2: LeRobot ACT on CALVIN

This directory is the lightweight runnable workspace for Task 2. It contains code, run tables, and sanity tests. Large runtime assets are intentionally not stored here: raw CALVIN/LeRobot data, training outputs, checkpoints, evaluation videos, and full logs are kept outside Git and referenced from the project README / Google Drive package.

## Layout

- `configs/`: experiment definitions and run table.
- `scripts/`: training, dataset preparation, evaluation, audit, and result-collection entrypoints.
- `tests/`: lightweight tests for command construction, result collection, and environment checks.
- `environment.yml`: Linux/CUDA Conda environment for ACT training and CALVIN evaluation.
- `.gitignore`: local-only exclusions for `data/`, `outputs/`, logs, caches, and archives.

The corresponding report-facing outputs are stored separately under:

```text
../../results/task2-lerobot-act/final-results/
```

This separation is intentional:

- `tasks/task2_lerobot_act/` is for code and reproduction entrypoints.
- `results/task2-lerobot-act/` is for final tables, figures, logs, summaries, and locally retained weights.

## Main Reproduction Commands

Create the environment:

```bash
conda env create -f environment.yml
conda activate hw3_task2_lerobot
python -m pip install -r ../../requirements.txt
```

Run the course-provided split main chain:

```bash
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export WANDB_MODE=offline
bash scripts/44_run_task2_ta_official_split_chain.sh
```

Run checkpoint sweep and seed1 robustness experiments:

```bash
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export WANDB_MODE=offline
bash scripts/46_run_task2_ta_sweep_seed1_chain.sh
```

Evaluate one trained checkpoint on CALVIN D:

```bash
bash scripts/45_run_ta_conditioned_eval.sh \
  sbert \
  act_ta_abc_lang_sbert_100k_seed1 \
  outputs/20260606_ta_official_split/act_ta_abc_lang_sbert_100k_seed1/checkpoints/100000/pretrained_model \
  full
```

## Important Final Experiment Areas

The final report is supported by these result groups:

- course-provided `splitA/splitB/splitC/splitD` training and evaluation;
- A-only baseline on CALVIN D;
- ABC-mixed one-hot, SBERT, and chunk-size variants;
- seed1 robustness and checkpoint sweep experiments;
- ABC extended high-score checkpoint selection;
- visual/action distribution shift and subtask breakdown analysis.

Key output files are kept under `../../results/task2-lerobot-act/final-results/results/`, including:

- `task2_ta_official_split_summary.csv`
- `task2_ta_sweep_seed1_summary.csv`
- `task2_high_score_summary.csv`
- `task2_seed_robustness_summary.csv`
- `task2_source_split_visual_action_analysis.csv`
- `task2_extension_eval_summary.csv`

## Tests

Run lightweight sanity tests from this directory:

```bash
pytest tests
```

These tests do not require full CALVIN data or model checkpoints; they check local command-building and parsing utilities.
