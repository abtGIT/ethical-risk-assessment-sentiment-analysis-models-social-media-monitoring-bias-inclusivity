# Chapter 5 figures vs saved outputs of the locked notebooks

Source: the printed cell outputs saved inside each notebook in the Drive folder **Implementation**
(folder ID 1TFli-yb7EOSv9RT5FIlDviLHteOM4B5m), read through the Google Drive connector on 12 Oct 2026.
Values below were transcribed from those outputs. To check any row, open the notebook link and look at the named cell's output.
The result CSV files in LJMU_FINAL_EXPERIMENT_RUN/results were NOT accessed (not visible to the connector).

| Fig | Notebook (Drive file ID) | Cell output used | Values in notebook output | Value shown in figure | Result |
|---|---|---|---|---|---|
| 5.1 | FINAL_02_Model_Development_and_Lock (17FJGGvBitcrvCGPD9wZ057GrHgW4xOwq) | Stage A table (baseline_val, "Validation-selected baseline: LinearSVM") | SVM 0.6860/0.646191; LR 0.6705/0.609972; XGB 0.6560/0.596196; MNB 0.6260/0.577419; RF 0.6200/0.511543 (accuracy/Macro F1) | 0.6860/0.6462; 0.6705/0.6100; 0.6560/0.5962; 0.6260/0.5774; 0.6200/0.5115 | PASS |
| 5.2 | FINAL_03_RQ1_Subgroup_Performance (19k3_Ia6jV_MBpgH5bNwL3KmvUEpYd7gY) | perf table (Macro F1, CI Low, CI High) | ref 0.596926 [0.587504, 0.606141] N 10388; emoji 0.635388 [0.598448, 0.672985] N 710; slang 0.619431 [0.498602, 0.731746] N 60; sarcasm 0.789744 [0.475308, 1.000000] N 14 | 0.5969, 0.6354, 0.6194, 0.7897 with matching interval bars and N | PASS |
| 5.3 | FINAL_04_RQ2_Disparity_Audit (1cGR8TpGumyPS6BMBX34V6HkJnqvghqya) | parity table (CPD column) and Fairlearn/AIF360 cross-check table | LR 0.016052/0.143987/-0.072680; SVM 0.010134/0.115768/-0.012803; MNB 0.025137/0.004480/0.009241; RF 0.005815/0.052529/-0.030805; XGB -0.014482/0.058523/-0.058144 (emoji/slang/sarcasm); all toolkit Agree = True | Bars correct; labels XGB emoji "-0.015" and RF slang "+0.052" were wrong (should be -0.014, +0.053); LR slang label clipped | FIXED (see 4_figure_5_3_fix) |
| 5.4 | FINAL_05_RQ3_Errors_Confidence_XAI (1UP3CmHmGnTfXlYGllHQGkbM2Q6XoQrix) | "overall" error-direction table | 1755 (37.733821%), 1040 (22.360783%), 730 (15.695549%), 684 (14.706515%), 374 (8.041281%), 68 (1.462051%) | 1,755 (37.73%) ... 68 (1.46%) | PASS |
| 5.5 | FINAL_05 (same file) | "cc" confidence/calibration table | HCER ref 0.145677, emoji 0.198381, slang 0.173913, sarcasm 0.000000; ECE ref 0.007797, emoji 0.051766, slang 0.104879, sarcasm 0.179006 | 0.1457, 0.1984, 0.1739, 0.0000; 0.0078, 0.0518, 0.1049, 0.1790 | PASS |
| 5.6 | FINAL_06_v2_RQ4_External_Robustness (1FjqWCohOptPFIUFYKSyDtX1VK2H-CpFI) | regression-guard asserts and printed results | SemEval Macro F1 0.5735511881960479; Sentiment140 comparator 0.677815530444217; Sentiment140 target 0.5804; source 3-class 0.6053 (FINAL_02 test output 0.6053146) | 0.6053 / 0.5736 (delta -0.0318); 0.6778 / 0.5804 (delta -0.0974) | PASS |
| 3.9 | FINAL_01_Data_Preparation_and_Subgroups (1mfSQBgCZsQvaPhaa3p1a5WxiRnVv3p_Q) and FINAL_06_v2 | subgroup counts output; S140/SemEval counts | test subgroups 10388 / 710 / 60 / 14 / 1112; SemEval 1,758 rows / 1,011 sentences; S140 100,000 rows | same | PASS |

Notebook links: https://colab.research.google.com/drive/<file ID>
