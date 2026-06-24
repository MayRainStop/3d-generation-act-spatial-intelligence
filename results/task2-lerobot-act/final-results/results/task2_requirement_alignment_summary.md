# HW3 Task 2 Requirement Alignment Summary

Updated: 2026-06-04T20:33:15+08:00

## Requirement Reading

The PDF asks for two ACT policies: one trained only on environment A, and one trained on mixed A/B/C, then both evaluated zero-shot on environment D.

## Dataset Audit

- LeRobot repo: `fywang/calvin-task-ABC-D-lerobot`
- Episodes: 18957 (6319 + 6319 + 6319)
- Explicit environment label column present: False
- Ordered equal thirds audit pass: True

The public converted LeRobot dataset does not expose an explicit A/B/C environment-label column. The first ordered third is therefore used as the requirement-aligned A-only baseline, while the full train split is used as ABC-mixed. This is stronger than a random split and matches the official ABC-D split structure, but the missing stored label is reported honestly.

## Requirement-Aligned CALVIN D Results

| Model | Training data | Steps | Chunk | Avg Len | SR@1 | SR@2 | SR@3 | SR@4 | SR@5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A-only inferred first ordered third | A inferred from first 6319 ordered ABC episodes | 50000 | 50 | 0.613 | 0.402 | 0.152 | 0.049 | 0.010 | 0.000 |
| ABC-mixed same hyperparameters | all ABC mixed episodes | 50000 | 50 | 0.671 | 0.411 | 0.167 | 0.060 | 0.027 | 0.006 |
| ABC-mixed extended training | all ABC mixed episodes | 100000 | 50 | 0.997 | 0.566 | 0.260 | 0.111 | 0.041 | 0.019 |

## Cleanup Note

- Historical compressed package copies were removed during server cleanup.
- Extracted results remain in the project tree under `results/` and `outputs/`.
