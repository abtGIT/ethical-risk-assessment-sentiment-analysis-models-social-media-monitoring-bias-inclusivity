# Ethical Risk Assessment of Sentiment Analysis Models

MSc thesis implementation repository for:

**An Ethical Risk Assessment of Sentiment Analysis Models in Social Media Monitoring with a Focus on Bias, Inclusivity, and Misclassification Patterns**

**Abhishek Tomar — Liverpool John Moores University — October 2026**

This repository is aligned to the final thesis submission candidate **V12.2.7**. The authoritative implementation is the frozen `FINAL_00`–`FINAL_07` notebook chain plus the shared utility module. Earlier 25-experiment material is retained only under `archive/legacy-25-experiment/` for provenance and must not be used as the final thesis implementation.

## Final study design

The final thesis uses **nine analytical stages**:

1. protocol/configuration freeze;
2. five-model common-representation comparison;
3. feature comparison, tuning and final pipeline lock;
4. one held-out TweetEval test evaluation;
5. RQ1 linguistic-subgroup performance;
6. RQ2 linguistic-disparity audit;
7. RQ3 error, confidence/calibration and SHAP/LIME analysis;
8. RQ4 external robustness;
9. multi-dimensional ethical-risk evidence synthesis.

Stages 2–8 are the empirical/evaluative core. Stages 1 and 9 are control and synthesis stages.

## Authoritative final-run order

```text
Implementation/
  final_thesis_utils.py
  FINAL_00_Config_and_Freeze.ipynb
  FINAL_01_Data_Preparation_and_Subgroups.ipynb
  FINAL_02_Model_Development_and_Lock.ipynb
  FINAL_03_RQ1_Subgroup_Performance.ipynb
  FINAL_04_RQ2_Disparity_Audit.ipynb
  FINAL_05_RQ3_Errors_Confidence_XAI.ipynb
  FINAL_06_v2_RQ4_External_Robustness.ipynb
  FINAL_07_Ethical_Risk_Synthesis.ipynb
  AUX_01_SarcOji_Proxy_Validation.ipynb
```

`AUX_01` is auxiliary construct-validity/limitation evidence only. It does not alter the frozen sarcasm-indicated rule, model selection, features or thresholds.

## Supporting EDA

```text
EDA/
  EDA_01_TweetEval_FINAL_v2.ipynb
  EDA_02_SemEval2014_FINAL_v2.ipynb
  EDA_03_Sentiment140_FINAL_v3.ipynb
  EDA_04_CrossDataset_FINAL_v2.ipynb
  EDA_05_SarcOji_ScopeGate_v1.ipynb
```

These notebooks document exploratory/data-integrity work and the frozen external-dataset roles. They are supporting records, not a separate experimental chain.

## Dataset roles

| Dataset | Final role |
|---|---|
| **TweetEval** | Primary model development, validation-based selection and one held-out final test evaluation |
| **Sentiment140** | Binary external-robustness condition; frozen 100,000-row balanced unique-text sample |
| **SemEval-2014 Task 4** | Domain/task-shift stress condition; not clean same-task validation |
| **SarcOji** | Auxiliary validation/limitation evidence for the frozen sarcasm-indicated surface-cue proxy only |

## Frozen linguistic groups

Priority order:

`sarcasm-indicated → emoji-heavy → slang-heavy → reference-unmarked → other`

Key rules:

- emoji-heavy: emoji density > 0.05;
- slang-heavy: slang ratio > 0.10 using the frozen 51-term lexicon;
- sarcasm-indicated: predefined surface-cue regex; **not gold sarcasm**;
- reference-unmarked: no focal flag and word count ≥ 8;
- stable quantitative interpretation requires **N ≥ 30**.

The final reference group is **reference-unmarked**, not the superseded `formal` label.

## Locked headline results

### Held-out TweetEval

- Accuracy: **0.6214**
- Macro F1: **0.6053**
- Test N: **12,284**
- Total errors: **4,651**

### RQ1 subgroup performance

| Subgroup | N | Macro F1 | Interpretation |
|---|---:|---:|---|
| Emoji-heavy | 710 | 0.6354 | Stable |
| Slang-heavy | 60 | 0.6194 | Stable |
| Reference-unmarked | 10,388 | 0.5969 | Stable reference |
| Other | 1,112 | 0.5915 | Residual |
| Sarcasm-indicated | 14 | 0.7897 | Exploratory only |

### RQ2 disparity

The final study reports:

- **Correctness Parity Difference (CPD)**
- **Correctness Rate Ratio (CRR)**
- one-vs-rest class-conditional **TPR/FPR gaps**

These are descriptive linguistic-group outcome disparities. They are **not** demographic statistical-parity/disparate-impact claims and do not produce a universal fair/unfair label.

### RQ3 error / reliability / XAI

- negative → neutral: **1,755 errors (37.73%)**
- neutral → positive: **1,040**
- positive → neutral: **730**
- neutral → negative: **684**
- negative → positive: **374**
- positive → negative: **68**
- SHAP/LIME specific-cue mean Jaccard: **0.5269**
- explanation sample: **N = 33**
- SHAP/LIME interpretation is local/descriptive, not causal.

### RQ4 external robustness

**Sentiment140**

- frozen target N: **100,000**
- Accuracy: **0.4913**
- Macro F1: **0.5804**
- matched-source Macro F1: **0.6778**
- Δ Macro F1: **−0.0974**
- neutral prediction rate: **0.3353**
- emoji-heavy external subgroup: N=72, Macro F1=0.3690, gap vs reference **−0.2123**

**SemEval-2014**

- aspect rows: **1,758**
- unique sentences: **1,011**
- Accuracy: **0.6485**
- Macro F1: **0.5736**
- Δ Macro F1 vs TweetEval source: **−0.0318**

## Reproducibility controls

- random seed: **42**
- Python: **3.13.15**
- scikit-learn: **1.6.1**
- TweetEval test excluded from model-family, feature and hyperparameter selection;
- external datasets never used for source-model selection or retuning;
- subgroup thresholds frozen before final evaluation;
- no total ethical-risk score, low/medium/high risk category or overall model ranking.

Locked fingerprints:

- final model SHA-256: `bf92c6162dd4664c4a3445d8ce806cc28b09809f75613d6c70f4ef8b6ad954e2`
- frozen Sentiment140 artefact SHA-256: `d25327ccb390945e945f687c38f84e3973e81199c7b7977f02d1f5a08593b8ed`

See `manifests/` for the final run manifest and subgroup freeze configuration.

## Evidence workbook

The thesis-aligned evidence workbook is maintained separately as:

**Abhishek_Tomar_Thesis_FINAL_Evidence_Workbook.xlsx**

Drive copy:
https://docs.google.com/spreadsheets/d/1RwNHZ9TZmWnmG-Ihm1xjD7AGZVl6pg7l/edit?usp=drivesdk

It maps the final nine-stage design, RQs, notebooks, thesis tables, locked results and the original 25-experiment plan to the final methodology.

## Repository layout

```text
Implementation/               authoritative final-run notebooks
EDA/                          supporting exploratory/data-integrity notebooks
manifests/                    frozen run and subgroup configuration
archive/legacy-25-experiment/ superseded original implementation and old README-era figures
README.md                     this file
```

## Legacy archive

The original repository was built around a **25-experiment** plan containing items such as a `formal` subgroup, SPD/DIR-oriented fairness framing, imbalance intervention, ensemble weighting and older external-robustness outputs.

Those files are retained only under:

`archive/legacy-25-experiment/`

They are useful for provenance but **must not be cited or executed as the authoritative final thesis implementation**.

## Interpretation boundaries

This repository does **not** claim:

- demographic fairness guarantees;
- causal explanations from SHAP/LIME;
- universal cross-domain fairness;
- a single ethical-risk score;
- an overall model ranking;
- gold-standard sarcasm detection from the surface-cue proxy.

The final thesis instead reports a traceable, multi-dimensional evidence profile with explicit limitations and coverage conditions.
