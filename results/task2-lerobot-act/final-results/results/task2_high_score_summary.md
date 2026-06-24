# HW3 Task 2 High-Score Summary

Candidate models are evaluated on CALVIN D. The main score is average successful sequence length.

| Family | Run | Cond | AvgLen | SR@1 | SR@2 | SR@3 | SR@4 | SR@5 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| long_train | act_abc_taskcond_200k_seed0_ckpt120k | onehot | 1.028 | 0.561 | 0.277 | 0.119 | 0.049 | 0.022 |
| long_train | act_abc_taskcond_200k_seed0_ckpt160k | onehot | 1.023 | 0.564 | 0.285 | 0.116 | 0.045 | 0.013 |
| sbert_long | act_abc_lang_sbert_100k_seed0 | sbert | 0.998 | 0.57 | 0.254 | 0.102 | 0.049 | 0.023 |
| existing_best | act_abc_taskcond_100k_seed0 | onehot | 0.997 | 0.566 | 0.26 | 0.111 | 0.041 | 0.019 |
| seed1_long | act_abc_taskcond_seed1_100k | onehot | 0.957 | 0.526 | 0.247 | 0.119 | 0.046 | 0.019 |
| long_train | act_abc_taskcond_200k_seed0_ckpt200k | onehot | 0.873 | 0.517 | 0.224 | 0.09 | 0.034 | 0.008 |
| chunk100_long | act_abc_taskcond_chunk100_100k_seed0 | onehot | 0.747 | 0.5 | 0.175 | 0.055 | 0.014 | 0.003 |
| checkpoint_sweep | act_abc_taskcond_100k_seed0_ckpt060k | onehot | 0.743 | 0.495 | 0.162 | 0.06 | 0.02 | 0.006 |
| checkpoint_sweep | act_abc_taskcond_100k_seed0_ckpt080k | onehot | 0.734 | 0.483 | 0.18 | 0.053 | 0.015 | 0.003 |
