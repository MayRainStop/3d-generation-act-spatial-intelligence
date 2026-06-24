# HW3 Task 2 TA Sweep/Seed1 Summary

| Family | Run | Cond | AvgLen | SR@1 | SR@2 | SR@3 | SR@4 | SR@5 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| ta_ABC_sbert_seed1_100k | act_ta_abc_lang_sbert_100k_seed1 | sbert | 0.923 | 0.496 | 0.235 | 0.112 | 0.059 | 0.021 |
| ta_ABC_sbert_seed0_160k | act_ta_abc_lang_sbert_160k_seed0 | sbert | 0.915 | 0.483 | 0.261 | 0.104 | 0.045 | 0.022 |
| ta_ABC_sbert_seed0_120k | act_ta_abc_lang_sbert_160k_seed0_ckpt120k | sbert | 0.898 | 0.474 | 0.238 | 0.11 | 0.051 | 0.025 |
| ta_A_seed0_100k | act_ta_a_taskcond_100k_seed0 | onehot | 0.855 | 0.525 | 0.225 | 0.072 | 0.025 | 0.008 |
| ta_A_seed1_100k | act_ta_a_taskcond_100k_seed1 | onehot | 0.846 | 0.487 | 0.219 | 0.088 | 0.041 | 0.011 |
| ta_ABC_sbert_seed0_100k | act_ta_abc_lang_sbert_100k_seed0 | sbert | 0.787 | 0.477 | 0.195 | 0.071 | 0.032 | 0.012 |
| ta_A_seed0_160k | act_ta_a_taskcond_160k_seed0 | onehot | 0.774 | 0.482 | 0.185 | 0.069 | 0.029 | 0.009 |
| ta_ABC_sbert_seed0_140k | act_ta_abc_lang_sbert_160k_seed0_ckpt140k | sbert | 0.759 | 0.45 | 0.2 | 0.066 | 0.028 | 0.015 |
| ta_A_seed0_120k | act_ta_a_taskcond_160k_seed0_ckpt120k | onehot | 0.726 | 0.487 | 0.161 | 0.053 | 0.02 | 0.005 |
| ta_A_seed0_140k | act_ta_a_taskcond_160k_seed0_ckpt140k | onehot | 0.681 | 0.451 | 0.159 | 0.051 | 0.016 | 0.004 |

Notes:

- The seed1 robustness check evaluates the 100k checkpoints for A-only and ABC SBERT. The training script saves every 20k steps, so there is no 50k checkpoint for these seed1 runs; the empty 50k rows in the CSV are placeholders, not failed evaluations.
- For the directly comparable 100k seed check, A-only gets 0.855/0.846 AvgLen for seed0/seed1, while ABC SBERT gets 0.787/0.923.
