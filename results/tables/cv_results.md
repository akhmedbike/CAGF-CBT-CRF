# Cross-validation results (k-fold, official CoNLL-2018 metrics)

Source: `results_cv/stratified/cv_raw.json`
Jack-knifed F1: every sentence scored exactly once (by the fold whose test set held it).
Internal metrics (Acc/macro-F1) are mean +/- std across folds; official F1 is computed on the concatenated out-of-fold predictions.

## Official CoNLL-2018 F1 (jack-knifed over all sentences)
| Configuration | Lemmas | UPOS | UFeats | AllTags | N folds |
|---|---|---|---|---|---|
| Transformer-only | 72.27 | 72.39 | 49.09 | 43.79 | 10 |
| w/o char encoder | 72.08 | 73.61 | 48.65 | 42.82 | 10 |
| w/o gated fusion | 74.90 | 82.75 | 62.15 | 56.57 | 10 |
| w/o CRF | 74.83 | 83.16 | 64.15 | 58.59 | 10 |
| CAGF-CBT+CRF (full) | 75.04 | 83.06 | 64.43 | 59.13 | 10 |

## Per-fold internal metrics (mean +/- std across folds)
### Lemma
| Configuration | ACCURACY | PRECISION | RECALL | F1 | N folds |
|---|---|---|---|---|---|
| Transformer-only | 63.80 ± 1.76 | 50.11 ± 2.91 | 46.04 ± 2.88 | 46.62 ± 2.84 | 10 |
| w/o char encoder | 63.27 ± 1.59 | 48.67 ± 2.24 | 44.74 ± 2.88 | 45.26 ± 2.64 | 10 |
| w/o gated fusion | 69.05 ± 1.57 | 45.58 ± 2.38 | 45.99 ± 2.71 | 43.97 ± 2.47 | 10 |
| w/o CRF | 69.18 ± 1.68 | 44.69 ± 2.37 | 45.62 ± 2.62 | 43.37 ± 2.40 | 10 |
| CAGF-CBT+CRF (full) | 69.33 ± 1.89 | 44.87 ± 2.24 | 45.76 ± 2.77 | 43.53 ± 2.48 | 10 |

### UPOS
| Configuration | ACCURACY | PRECISION | RECALL | F1 | N folds |
|---|---|---|---|---|---|
| Transformer-only | 72.30 ± 3.73 | 71.81 ± 2.82 | 60.36 ± 3.33 | 63.62 ± 2.51 | 10 |
| w/o char encoder | 73.61 ± 3.75 | 72.86 ± 5.71 | 61.75 ± 6.33 | 65.23 ± 5.93 | 10 |
| w/o gated fusion | 82.77 ± 2.03 | 77.45 ± 3.97 | 72.13 ± 5.88 | 73.19 ± 4.86 | 10 |
| w/o CRF | 83.13 ± 1.80 | 78.14 ± 4.01 | 71.85 ± 4.06 | 73.80 ± 3.85 | 10 |
| CAGF-CBT+CRF (full) | 83.02 ± 1.85 | 77.06 ± 3.49 | 72.84 ± 5.29 | 73.84 ± 4.17 | 10 |

### Grammeme
| Configuration | ACCURACY | PRECISION | RECALL | F1 | N folds |
|---|---|---|---|---|---|
| Transformer-only | 47.28 ± 2.74 | 43.09 ± 5.08 | 17.44 ± 2.35 | 22.44 ± 2.86 | 10 |
| w/o char encoder | 46.63 ± 2.86 | 43.21 ± 8.21 | 19.05 ± 4.61 | 23.70 ± 5.28 | 10 |
| w/o gated fusion | 60.66 ± 2.44 | 57.31 ± 4.14 | 39.25 ± 5.10 | 43.44 ± 4.91 | 10 |
| w/o CRF | 62.73 ± 4.17 | 58.23 ± 6.88 | 39.53 ± 6.71 | 44.27 ± 6.78 | 10 |
| CAGF-CBT+CRF (full) | 62.96 ± 4.23 | 60.54 ± 4.73 | 40.25 ± 5.79 | 45.23 ± 5.62 | 10 |
