# HW3 Task 2 Final Completion Summary

Updated: 2026-06-04T20:05:16.542166+08:00

## Source Split Caveat

The available public LeRobot dataset is `fywang/calvin-task-ABC-D-lerobot`, which exposes ABC as one converted dataset without raw A/B/C environment tags. For the required A/B/C comparison, this workspace uses deterministic episode-order thirds as proxy source domains and reports visual statistics to make the limitation explicit.

## Visual Shift and Action Smoothness

| Split | Episodes | Frames | RGB L2 top->D | RGB L2 wrist->D | Mean action delta | P95 action delta |
|---|---:|---:|---:|---:|---:|---:|
| A | 6319 | 379231 | 0.0288 | 0.0262 | 0.2260 | 0.5671 |
| B | 6319 | 377837 | 0.0263 | 0.0267 | 0.2257 | 0.5665 |
| C | 6319 | 379468 | 0.0199 | 0.0270 | 0.2237 | 0.5599 |
| ABC | 18957 | 1136536 | 0.0250 | 0.0265 | 0.2251 | 0.5646 |

## CALVIN D Online Evaluation Rows

| Run | Condition | Avg Len | SR@1 | SR@5 |
|---|---|---:|---:|---:|
| calvin_act_a_proxy_taskcond_seed0_onehot_D_eval | onehot | 0.613 | 0.402 | 0.0 |
| calvin_act_abc_lang_hash_seed0_hash_D_eval | hash | 0.492 | 0.373 | 0.001 |
| calvin_act_abc_lang_sbert_seed0_sbert_D_eval | sbert | 0.688 | 0.434 | 0.009 |
| calvin_act_abc_taskcond_100k_seed0_onehot_D_eval | onehot | 0.997 | 0.566 | 0.019 |
| calvin_act_abc_taskcond_chunk100_seed0_onehot_D_eval | onehot | 0.791 | 0.517 | 0.004 |
| calvin_act_abc_taskcond_chunk10_seed0_onehot_D_eval | onehot | 0.523 | 0.323 | 0.007 |
| calvin_act_abc_taskcond_seed0_wrong_onehot_D_eval | wrong_onehot | 0.076 | 0.075 | 0.0 |
| calvin_act_abc_taskcond_seed0_zero_onehot_D_eval | zero_onehot | 0.034 | 0.033 | 0.0 |
| calvin_act_abc_taskcond_seed1_onehot_D_eval | onehot | 0.794 | 0.489 | 0.01 |
| calvin_act_abc_to_d_seed0_unconditioned_D_eval | unconditioned | 0.097 | 0.094 | 0.0 |
| calvin_act_b_proxy_taskcond_seed0_onehot_D_eval | onehot | 0.741 | 0.462 | 0.004 |
| calvin_act_c_proxy_taskcond_seed0_onehot_D_eval | onehot | 0.493 | 0.347 | 0.0 |
| calvin_taskcond_D_eval | unknown | 0.671 | 0.411 | 0.006 |
