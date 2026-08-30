## Table 2. Transformed TSC across annotation budgets, mean ± SD (%)
| Language | Budget | Random | Entropy | Novelty | Support-Aware |
|---|---|---|---|---|---|
| Kazakh | 10% | 41.87 ± 2.21 | 40.08 ± 2.44 | 41.23 ± 2.27 | 43.44 ± 2.03 |
| Kazakh | 20% | 54.85 ± 2.75 | 52.52 ± 1.33 | 55.67 ± 2.12 | 56.62 ± 1.32 |
| Kazakh | 30% | 61.78 ± 2.10 | 59.90 ± 1.11 | 63.12 ± 1.19 | 64.44 ± 0.99 |
| Kazakh | 50% | 70.89 ± 1.72 | 69.12 ± 1.43 | 71.54 ± 0.69 | 72.52 ± 1.06 |
| Turkish | 10% | 58.94 ± 0.90 | 54.94 ± 0.86 | 59.53 ± 0.71 | 61.09 ± 0.49 |
| Turkish | 20% | 70.20 ± 0.92 | 66.81 ± 0.80 | 71.53 ± 0.59 | 72.93 ± 0.49 |
| Turkish | 30% | 75.46 ± 0.60 | 72.87 ± 0.65 | 77.26 ± 0.56 | 78.24 ± 0.47 |
| Turkish | 50% | 81.05 ± 0.44 | 80.10 ± 0.52 | 82.52 ± 0.45 | 83.18 ± 0.66 |
| Kyrgyz | 10% | 60.53 ± 1.18 | 56.56 ± 1.56 | 59.42 ± 1.63 | 61.36 ± 1.31 |
| Kyrgyz | 20% | 72.63 ± 1.07 | 69.11 ± 1.59 | 73.38 ± 1.05 | 75.07 ± 0.94 |
| Kyrgyz | 30% | 78.67 ± 1.16 | 77.02 ± 1.45 | 80.05 ± 0.66 | 81.56 ± 1.04 |
| Kyrgyz | 50% | 85.08 ± 0.84 | 83.85 ± 1.03 | 86.45 ± 0.63 | 87.11 ± 0.86 |

## Table 3. Normalized AULC over the 5–50% annotation range (%)
| Language | Lemma: Random | Lemma: Entropy | Lemma: Support-aware | TSC: Random | TSC: Support-aware | UPOS: Entropy | UPOS: Support-aware |
|---|---|---|---|---|---|---|---|
| Kazakh | 59.83 | 58.64 | 60.03 | 57.13 | 59.03 | 67.51 | 67.78 |
| Turkish | 68.31 | 66.61 | 68.88 | 71.13 | 73.50 | 79.02 | 79.35 |
| Kyrgyz | 71.06 | 68.90 | 70.96 | 73.95 | 76.04 | 83.42 | 83.71 |

## Table 4. Paired SA vs Entropy, transformed TSC @20% (Wilcoxon + Holm)
| Language | SA − Entropy (pp) | Holm-adjusted p | n seeds |
|---|---|---|---|
| Kazakh | +4.10 | 0.0059 | 10 |
| Turkish | +6.11 | 0.0059 | 10 |
| Kyrgyz | +5.97 | 0.0059 | 10 |

## b95: median budget to reach 95% of full-pool lemma accuracy (%)
| Language | Random | Entropy | Novelty | Support-Aware | full-pool lemma acc |
|---|---|---|---|---|---|
| Kazakh | 50.0 | 50.0 | 50.0 | 50.0 | 73.36 |
| Turkish | 50.0 | 50.0 | 48.6 | 46.4 | 78.60 |
| Kyrgyz | 49.7 | 50.0 | 48.7 | 48.5 | 82.27 |

## Kazakh Eq. (7)-(8) weight sensitivity @20% (liblinear_ovr surrogate)
| Config | Transformed TSC @20% | Δ | Lemma acc @20% | Δ |
|---|---|---|---|---|
| baseline | 56.62 ± 1.32 | +0.00 | 58.48 ± 1.03 | +0.00 |
| equal | 56.01 ± 1.52 | -0.61 | 57.98 ± 0.79 | -0.50 |
| support-heavy | 57.71 ± 1.19 | +1.09 | 59.00 ± 0.57 | +0.52 |
| novelty-heavy | 56.22 ± 1.73 | -0.40 | 57.76 ± 1.36 | -0.72 |
| uncertainty-heavy | 54.52 ± 1.59 | -2.10 | 57.62 ± 0.93 | -0.86 |
| mean-only | 56.65 ± 1.33 | +0.03 | 58.69 ± 0.97 | +0.21 |
| balanced | 55.71 ± 1.58 | -0.91 | 57.64 ± 0.70 | -0.84 |
