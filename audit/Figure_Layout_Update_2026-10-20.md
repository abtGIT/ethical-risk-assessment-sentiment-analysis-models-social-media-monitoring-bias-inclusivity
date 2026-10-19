# Figure layout reconciliation — 20 October 2026

This change synchronises the revised dissertation presentation for Figures **3.6** and **3.8** with GitHub_Upload and the public implementation repository. The source is the embedded chart assets in `Abhishek_Tomar_Thesis_V13_5_LANGUAGE_POLISHED_WORKING.docx` (which carries forward the V13_4 figures-only changes). The dissertation document itself is **not** published to GitHub.

## Files updated
- `figures/Fig3_6_class_distribution.png` and `.svg`: move the legend clear of the plotted class labels; preserve the existing data and display rounding
- `figures/Fig3_8_markers.png` and `.svg`: separate percentage labels from the bars and move the legend outside the plotting area
- `Implementation/FIGURES_Regenerate.ipynb`: synchronise the latest Figure 3.8 code from the executed `FIGURES_Regenerate_v2.ipynb` and adjust Figure 3.6 legend/title positioning; remove the stale saved Figure 3.6 cell output

## Immutable scientific content
- TweetEval training/validation/test splits and all subgroup marker calculations are unchanged
- The 63 logged plotting values in `figures/FIGURE_VALUES.csv` are unchanged
- No files under `Implementation/FINAL_00`–`FINAL_07`, `EDA/`, `manifests/`, or the legacy archive were modified
- No claim is made that this presentation-only commit reruns or retunes models

## Embedded asset SHA-256 values

| Asset | SHA-256 |
| --- | --- |
| Fig3_6_class_distribution.png | 18f77b772c8b7c9ef4ff3f5d789be6ff43184a4a0f5f1b6614cb9f73f570e5fe |
| Fig3_6_class_distribution.svg | c5769e78f6847154ec65b33e059c3d77ba4bdaf82e7e9aea8e6489b01d212111 |
| Fig3_8_markers.png | 9200a7409da34a2eebba4acf47c837320783386c195a2cab96b27124047d8510 |
| Fig3_8_markers.svg | e006bcb6bbfa2524aba7c506058475fbd57870139aacfb1cae2a5ebe092ac429 |

The figure regeneration notebook uses the original frozen evidence paths; run the notebook in a configured Colab environment if fresh derived previews are required. The uploaded dissertation's content and its version history remain outside this repository.
