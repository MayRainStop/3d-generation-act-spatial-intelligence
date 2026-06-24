# Task 2 Seed Robustness Summary

| Policy | Condition | Steps | Seed0 AvgLen | Seed1 AvgLen | Mean AvgLen | Sample Std |
|---|---|---:|---:|---:|---:|---:|
| TA splitA A-only | one-hot | 100000 | 0.855 | 0.846 | 0.8505 | 0.0064 |
| TA splitABC SBERT | SBERT | 100000 | 0.787 | 0.923 | 0.8550 | 0.0962 |

The seed1 robustness check evaluates the saved 100k checkpoints. The training jobs save every 20k steps, so 50k checkpoints were not produced for seed1; any empty 50k placeholder rows in the sweep CSV are not failed runs.

The A-only baseline is stable across seed0/seed1, while ABC SBERT has larger variance but also gives the strongest official-split ABC result at seed1 100k. Together with the seed0 160k result of 0.915, this supports the report conclusion that SBERT task conditioning and checkpoint/seed selection materially improve the ABC policy on the TA official split.
