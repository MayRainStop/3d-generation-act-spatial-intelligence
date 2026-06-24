# Dataset Route Decision

Date: 2026-05-29

Chosen route: public LeRobot-format CALVIN datasets for debug, ABC-to-D, D evaluation metadata, and chunk-size ablations; source-specific A/B/C runs require official raw CALVIN conversion or another dataset source.

Evidence:
- `results/dataset_inspection/fywang__calvin-debug-lerobot.json`
- `results/dataset_inspection/fywang__calvin-task-ABC-D-lerobot.json`
- `results/dataset_inspection/fywang__calvin-task-D-D-lerobot.json`
- `fywang` currently exposes `calvin-debug-lerobot`, `calvin-task-ABC-D-lerobot`, `calvin-task-D-D-lerobot`, and `calvin-task-ABCD-D-lerobot` through the Hugging Face mirror.

Reason:
- The ABC LeRobot-format dataset does not expose an A/B/C source-environment key in `meta/episodes.jsonl`.
- A/B/C source-specific comparison requires reliable source environment labels.
- The server cannot reach `huggingface.co`, but `https://hf-mirror.com` works and should be used for Hugging Face dataset access.
- The immediate executable route is to run `debug_smoke_seed0`, then `act_abc_to_d_seed0`, `act_abc_chunk10_seed0`, and `act_abc_chunk100_seed0`.
- If the assignment report needs strict A-only, B-only, and C-only comparisons, add a later raw CALVIN conversion step before launching those runs.
