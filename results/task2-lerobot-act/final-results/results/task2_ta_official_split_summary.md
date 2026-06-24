# HW3 Task 2 TA Official Split Summary

The TA dataset provides explicit splitA/splitB/splitC/splitD scene labels. Single-source models use A, B, or C; the ABC model is the manual union of splitA+splitB+splitC. All models are evaluated zero-shot on CALVIN D.

| Family | Run | Cond | AvgLen | SR@1 | SR@2 | SR@3 | SR@4 | SR@5 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| ta_A_taskcond_100k | act_ta_a_taskcond_100k_seed0 | onehot | 0.855 | 0.525 | 0.225 | 0.072 | 0.025 | 0.008 |
| ta_ABC_sbert_100k | act_ta_abc_lang_sbert_100k_seed0 | sbert | 0.787 | 0.477 | 0.195 | 0.071 | 0.032 | 0.012 |
| ta_A_taskcond_50k | act_ta_a_taskcond_100k_seed0_ckpt050k | onehot | 0.758 | 0.464 | 0.189 | 0.067 | 0.027 | 0.011 |
| ta_ABC_sbert_50k | act_ta_abc_lang_sbert_100k_seed0_ckpt050k | sbert | 0.731 | 0.45 | 0.19 | 0.059 | 0.026 | 0.006 |
| ta_ABC_chunk100_100k | act_ta_abc_taskcond_chunk100_100k_seed0 | onehot | 0.696 | 0.457 | 0.162 | 0.052 | 0.017 | 0.008 |
| ta_ABC_taskcond_100k | act_ta_abc_taskcond_100k_seed0 | onehot | 0.649 | 0.402 | 0.165 | 0.052 | 0.021 | 0.009 |
| ta_ABC_taskcond_50k | act_ta_abc_taskcond_100k_seed0_ckpt050k | onehot | 0.603 | 0.368 | 0.139 | 0.06 | 0.026 | 0.01 |
| ta_C_taskcond_100k | act_ta_c_taskcond_100k_seed0 | onehot | 0.473 | 0.315 | 0.108 | 0.034 | 0.011 | 0.005 |
| ta_C_taskcond_50k | act_ta_c_taskcond_100k_seed0_ckpt050k | onehot | 0.438 | 0.305 | 0.091 | 0.035 | 0.007 | 0.0 |
| ta_B_taskcond_100k | act_ta_b_taskcond_100k_seed0 | onehot | 0.338 | 0.266 | 0.059 | 0.012 | 0.001 | 0.0 |
| ta_B_taskcond_50k | act_ta_b_taskcond_100k_seed0_ckpt050k | onehot | 0.303 | 0.262 | 0.036 | 0.005 | 0.0 | 0.0 |
