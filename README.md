# Leakage-Free Evaluation of Weakly Supervised Shoplifting Detection

Official benchmark implementation and reproducibility package for **"Leakage-free evaluation of weakly supervised shoplifting detection: a split-rule audit and a zero-shot baseline for frozen foundation-model features"** by **Naveen Kumar Gottipalli** ([University of Wollongong in Dubai](https://www.uowdubai.ac.ae/)).

- **Code & Benchmark Repository:** [https://github.com/NaveenKumarGottipalli/Shopliftingdetector](https://github.com/NaveenKumarGottipalli/Shopliftingdetector)
- **Curated Multi-Corpus Dataset Index:** [https://huggingface.co/datasets/NaveenKumarGottipalli/Shoplifting_DataCollection](https://huggingface.co/datasets/NaveenKumarGottipalli/Shoplifting_DataCollection)
- **License:** [MIT License](LICENSE)

## Overview & Key Findings

Reported performance figures in video-based retail shoplifting detection frequently exceed 0.85–0.95 AUROC under random clip splits, yet degrade substantially when deployed on unseen retail cameras. This repository audits **2,353 deduplicated surveillance clips** across four public shoplifting corpora ([DCSASS](https://www.kaggle.com/datasets/mateohervas/dcsass-dataset), [MNNIT-S3](https://data.mendeley.com/datasets/r3yjf35hzr/1), [Shop-Dataset](https://www.kaggle.com/datasets/omarelg/shoplifting-videos-dataset), and the retail subset of [UCF-Crime](https://openaccess.thecvf.com/content_cvpr_2018/html/Sultani_Real-World_Anomaly_Detection_CVPR_2018_paper.html)) after removing **255 exact SHA-256 byte-for-byte duplicate files** (15 in MNNIT-S3, 218 in Shop-Dataset, and 22 in UCF-Crime) and **28 cross-corpus parent recordings** in UCF-Crime whose sliced sub-clips constitute the 28 incident folders of DCSASS.

1. **Pooled-Metric Distortion Audit:** Because positive class prevalence varies from **7.3%** ([UCF-Crime](https://openaccess.thecvf.com/content_cvpr_2018/html/Sultani_Real-World_Anomaly_Detection_CVPR_2018_paper.html)) to **55.8%** ([Shop-Dataset](https://www.kaggle.com/datasets/omarelg/shoplifting-videos-dataset)), we show that a trivial predictor outputting only each clip's dataset base rate without inspecting any video frames achieves **0.725 pooled AUROC** (and **0.472 pooled AUPRC**) despite having **0.500 AUROC** on every individual dataset (**+0.225 pooled AUROC bias**).
2. **In-Domain Split-Rule Audit (Experiment 1):** Holding the visual backbone ([CLIP ViT-B/16](https://proceedings.mlr.press/v139/radford21a.html)) and classifier (Top-3 MIL MLP) fixed, moving from random clip splits to strict scene-disjoint 5-fold cross-validation reduces AUROC by **0.341** on [DCSASS](https://www.kaggle.com/datasets/mateohervas/dcsass-dataset) (`0.907 -> 0.566`), **0.122** on [Shop-Dataset](https://www.kaggle.com/datasets/omarelg/shoplifting-videos-dataset) (`1.000 -> 0.877`), and **0.043** on [MNNIT-S3](https://data.mendeley.com/datasets/r3yjf35hzr/1) (`0.971 -> 0.928`).
3. **Leave-One-Dataset-Out Transfer & Statistical Tie (Experiment 2):** Under leave-one-dataset-out (LODO) evaluation across all four corpora using OpenAI's exact CLIP normalization constants (`0.4578275` green-channel mean) and `duration > 20s` timestamp seeking, training-free zero-shot text-prompt scoring (`0.644–0.690` macro AUROC; best: `0.690`, 95% scene-cluster bootstrap CI: `0.631–0.748`) and the best supervised MIL head trained on the remaining three corpora (`0.670` on CLIP ViT-B/16, 95% CI: `0.609–0.722`) show **no statistically significant difference** (paired difference `+0.020`, 95% CI: `[-0.051, +0.085]`, raw $p = 0.565$, Holm-Bonferroni $p_{\text{Holm}} = 1.000$).
4. **Scene Regularizers & Preprocessing Sensitivity (Experiment 3):** Unsupervised scene-mean centering and Group-DRO yield `0.663–0.672` macro AUROC on CLIP ViT-B/16 and partially mitigate cross-domain overfitting on CLIP ViT-L/14 (`0.512 -> 0.549`). Critically, zero-shot prompt scoring is nearly invariant to minor preprocessing perturbations ($\Delta = +0.002$, `0.690 -> 0.692`), whereas the supervised CLIP ViT-B/16 MIL head drops by `-0.055` (`0.670 -> 0.615`), demonstrating that supervised heads fitted to small surveillance corpora are fragile to minor feature and decode shifts.

## Benchmark Corpora Summary

| Dataset | Raw Candidates | SHA-256 Duplicates Removed | Cross-Corpus Parents Removed | Deduplicated Clips | Positive (%) | Negative | Nominal Groups | Strict Scenes | Scene Size (Min–Max) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| [DCSASS](https://www.kaggle.com/datasets/mateohervas/dcsass-dataset) | 896 | 0 | 0 | 896 | 155 (17.3%) | 741 | 28 | 28 | 32–32 |
| [MNNIT-S3](https://data.mendeley.com/datasets/r3yjf35hzr/1) ([Kaggle Mirror](https://www.kaggle.com/datasets/callmekio/shopliftingdataset)) | 589 | 15 | 0 | 574 | 207 (36.1%) | 367 | 366 | 64 | 2–32 |
| [Shop-Dataset](https://www.kaggle.com/datasets/omarelg/shoplifting-videos-dataset) | 799 | 218 | 0 | 581 | 324 (55.8%) | 257 | — | 15 | 12–70 |
| [UCF-Crime (Retail Subset)](https://openaccess.thecvf.com/content_cvpr_2018/html/Sultani_Real-World_Anomaly_Detection_CVPR_2018_paper.html) | 352 | 22 | 28 | 302 | 22 (7.3%) | 280 | 302 | 68 | 1–15 |
| **Total** | **2,636** | **255** | **28** | **2,353** | **708 (30.1%)** | **1,645** | — | **175** | **1–70** |

## Main Experimental Results

### Experiment 1: In-Domain 5-Fold Cross-Validation AUROC (`CLIP ViT-B/16`, Trained MIL Head, 2-Seed Mean ± SD)

| Dataset | Random Clips | Nominal Groups | Strict Scenes | Drop (`Random -> Strict`) |
| :--- | :---: | :---: | :---: | :---: |
| [DCSASS](https://www.kaggle.com/datasets/mateohervas/dcsass-dataset) | `0.907 ± 0.003` | `0.566 ± 0.041` | `0.566 ± 0.041` | `-0.341` |
| [MNNIT-S3](https://data.mendeley.com/datasets/r3yjf35hzr/1) | `0.971 ± 0.000` | `0.954 ± 0.001` | `0.928 ± 0.001` | `-0.043` |
| [Shop-Dataset](https://www.kaggle.com/datasets/omarelg/shoplifting-videos-dataset) | `1.000 ± 0.000` | — | `0.877 ± 0.005` | `-0.122` |

### Experiment 2: Leave-One-Dataset-Out (LODO) Cross-Dataset Benchmark

| Backbone | Method | DCSASS | MNNIT-S3 | Shop-Dataset | UCF-Crime | Macro AUROC | 95% Bootstrap CI | Macro AUPRC | Macro FAR@80%R |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| [CLIP ViT-B/16](https://proceedings.mlr.press/v139/radford21a.html) | `zs_single_max` (2 prompts) | 0.575 | 0.684 | **0.645** | 0.837 | 0.685 | (0.602–0.754) | 0.456 | 0.534 |
| [CLIP ViT-B/16](https://proceedings.mlr.press/v139/radford21a.html) | `zs_ens8_mean` (8 prompts) | 0.528 | **0.810** | 0.539 | **0.884** | **0.690** | **(0.631–0.748)** | **0.479** | **0.512** |
| [CLIP ViT-B/16](https://proceedings.mlr.press/v139/radford21a.html) | `zs_ens8_max` (8 prompts) | 0.582 | 0.806 | 0.577 | 0.790 | 0.689 | (0.613–0.765) | 0.468 | 0.521 |
| [SigLIP ViT-B/16](https://openaccess.thecvf.com/content/ICCV2023/html/Zhai_Sigmoid_Loss_for_Language_Image_Pre-Training_ICCV_2023_paper.html) | `zs_single_max` (2 prompts) | 0.554 | 0.691 | 0.557 | 0.843 | 0.661 | (0.605–0.709) | 0.438 | 0.562 |
| [SigLIP ViT-B/16](https://openaccess.thecvf.com/content/ICCV2023/html/Zhai_Sigmoid_Loss_for_Language_Image_Pre-Training_ICCV_2023_paper.html) | `zs_ens8_mean` (8 prompts) | 0.527 | 0.749 | 0.439 | 0.862 | 0.644 | (0.594–0.695) | 0.429 | 0.584 |
| [SigLIP ViT-B/16](https://openaccess.thecvf.com/content/ICCV2023/html/Zhai_Sigmoid_Loss_for_Language_Image_Pre-Training_ICCV_2023_paper.html) | `zs_ens8_max` (8 prompts) | 0.503 | 0.792 | 0.424 | 0.878 | 0.649 | (0.600–0.700) | 0.435 | 0.575 |
| [CLIP ViT-L/14](https://proceedings.mlr.press/v139/radford21a.html) | `zs_single_max` (2 prompts) | **0.611** | 0.650 | 0.582 | 0.766 | 0.652 | (0.589–0.710) | 0.431 | 0.579 |
| [CLIP ViT-L/14](https://proceedings.mlr.press/v139/radford21a.html) | `zs_ens8_mean` (8 prompts) | 0.486 | 0.802 | 0.495 | 0.837 | 0.655 | (0.610–0.704) | 0.442 | 0.568 |
| [CLIP ViT-L/14](https://proceedings.mlr.press/v139/radford21a.html) | `zs_ens8_max` (8 prompts) | 0.521 | 0.801 | 0.462 | 0.856 | 0.660 | (0.599–0.718) | 0.446 | 0.559 |
| [CLIP ViT-B/16](https://proceedings.mlr.press/v139/radford21a.html) | `trained_head` (Top-3 MIL) | 0.595 | 0.723 | 0.626 | 0.737 | 0.670 | (0.609–0.722) | 0.448 | 0.546 |
| [CLIP ViT-L/14](https://proceedings.mlr.press/v139/radford21a.html) | `trained_head` (Top-3 MIL) | 0.505 | 0.798 | 0.328 | 0.416 | 0.512 | (0.435–0.576) | 0.334 | 0.732 |

### Experiment 3: Scene Regularizers (Centering & Group-DRO) and Preprocessing Sensitivity

| Backbone | Variant | DCSASS | MNNIT-S3 | Shop-Dataset | UCF-Crime | Macro AUROC | 95% Bootstrap CI | Macro AUPRC | Macro FAR@80%R |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| CLIP ViT-B/16 | `trained_base` | 0.595 | 0.723 | 0.626 | 0.737 | 0.670 | (0.609–0.722) | 0.448 | 0.546 |
| CLIP ViT-B/16 | `trained_center` | 0.588 | 0.715 | 0.618 | 0.729 | 0.663 | (0.601–0.718) | 0.441 | 0.554 |
| CLIP ViT-B/16 | `trained_dro` | 0.599 | 0.726 | 0.621 | 0.742 | 0.672 | (0.611–0.726) | 0.450 | 0.542 |
| CLIP ViT-B/16 | `trained_center+dro` | 0.591 | 0.719 | 0.614 | 0.734 | 0.665 | (0.603–0.720) | 0.443 | 0.551 |
| CLIP ViT-L/14 | `trained_base` | 0.505 | 0.798 | 0.328 | 0.416 | 0.512 | (0.435–0.576) | 0.334 | 0.732 |
| CLIP ViT-L/14 | `trained_center` | 0.518 | 0.784 | 0.392 | 0.468 | 0.541 | (0.468–0.605) | 0.352 | 0.698 |
| CLIP ViT-L/14 | `trained_dro` | 0.512 | 0.801 | 0.341 | 0.432 | 0.522 | (0.446–0.585) | 0.340 | 0.719 |
| CLIP ViT-L/14 | `trained_center+dro` | 0.524 | 0.789 | 0.405 | 0.479 | 0.549 | (0.476–0.612) | 0.358 | 0.686 |

## Repository Structure

```text
Shopliftingdetector/
├── README.md                                     # Project documentation & experimental summary
├── LICENSE                                       # MIT License
├── requirements.txt                              # Python dependencies
├── .gitignore                                    # Clean git ignore rules
├── configs/
│   └── benchmark_config.json                     # Backbones, CLIP constants, hyperparameters & prompts
├── src/
│   ├── __init__.py
│   ├── dataset.py                                # Corpus discovery, SHA-256 deduplication & scene clustering
│   ├── features.py                               # Canonical video decoding (>20s seek) & VLM extraction
│   ├── models.py                                 # Top-K MIL head, zero-shot scorer, scene centering & Group-DRO
│   ├── metrics.py                                # Rank AUROC, AUPRC, FAR@80%R, bootstrap & Holm-Bonferroni
│   └── plotting.py                               # Vector PDF & 300-DPI PNG figure generation
├── scripts/
│   └── run_benchmark.py                          # Single-source CLI runner for tables, figures & LaTeX macros
├── notebooks/
│   └── Shoplifting_Detection_Benchmark.ipynb     # Self-contained Google Colab notebook with full outputs
├── results/
│   ├── tables/
│   │   ├── corpus_summary.csv                    # Table 1: Deduplicated corpus statistics & scene cluster sizes
│   │   ├── duplicate_audit_summary.csv           # Per-dataset SHA-256 duplicate & cross-corpus parent audit
│   │   ├── pooled_vs_macro_audit.csv             # Table 3: Dataset-identity baseline pooled vs. macro audit
│   │   ├── split_rule_audit.csv                  # Table 4: Experiment 1 in-domain 5-fold CV AUROC
│   │   ├── lodo_benchmark_results.csv            # Table 5: Experiment 2 LODO AUROC, CIs, AUPRC & FAR@80%R
│   │   ├── ablation_centering_dro.csv            # Table 6A: Scene-mean centering & Group-DRO LODO ablations
│   │   ├── preprocessing_sensitivity.csv         # Table 6B: Canonical vs. perturbed preprocessing sensitivity
│   │   ├── paired_differences.csv                # Table 7: Paired bootstrap differences & Holm-Bonferroni p-values
│   │   └── per_clip_predictions.csv              # Clip-level out-of-domain predictions across all 2,353 clips
│   └── figures/
│       ├── fig_split_rules.pdf / .png            # Figure 1: In-domain split rules comparison
│       ├── fig_macro_lodo.pdf / .png             # Figure 2: Macro LODO AUROC forest plot
│       └── fig_per_dataset_lodo.pdf / .png       # Figure 3: Per-dataset LODO AUROC comparison
└── paper/
    ├── main.tex                                  # Springer Nature LaTeX manuscript source
    ├── generated_macros.tex                      # Auto-generated LaTeX macros from CSV tables
    ├── generated_tables.tex                      # Auto-generated LaTeX tables from CSV tables
    ├── sn-jnl.cls                                # Springer Nature journal class file
    ├── refs.bib                                  # Verified BibTeX bibliography
    ├── main.bbl                                  # Compiled BibTeX bibliography
    ├── fig_split_rules.pdf                       # Vector figure 1
    ├── fig_macro_lodo.pdf                        # Vector figure 2
    ├── fig_per_dataset_lodo.pdf                  # Vector figure 3
    └── main.pdf                                  # Compiled camera-ready manuscript PDF
```

## Citation

```bibtex
@misc{gottipalli2026leakage,
  title  = {Leakage-free evaluation of weakly supervised shoplifting detection: a split-rule audit and a zero-shot baseline for frozen foundation-model features},
  author = {Gottipalli, Naveen Kumar},
  year   = {2026},
  note   = {Preprint},
  url    = {https://github.com/NaveenKumarGottipalli/Shopliftingdetector}
}
```
