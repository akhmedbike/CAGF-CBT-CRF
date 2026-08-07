# Ablation statistics vs full model (paired t-test, 5 seeds)

p_raw = uncorrected paired t-test p-value; p_Holm = Holm-Bonferroni adjusted;
d = Cohen's d (paired). * = significant after Holm correction at alpha=0.05.

## Lemma (macro F1)
| Configuration | F1 (mean +/- std) | Delta vs full | p_raw | p_Holm | Cohen's d |
|---|---|---|---|---|---|---|
| Transformer-only | 42.93 +/- 3.48 | -2.02 | 0.4623 | 1.0000 | 0.363 |
| w/o char encoder | 42.21 +/- 2.27 | -2.74 | 0.0596 | 0.5368 | 1.166 |
| w/o gated fusion | 43.65 +/- 1.76 | -1.30 | 0.3176 | 1.0000 | 0.510 |
| w/o CRF | 44.31 +/- 1.91 | -0.64 | 0.6351 | 1.0000 | 0.229 |
| CAGF-CBT+CRF (full) | 44.95 +/- 1.71 | +0.00 | -- | -- | -- |

## UPOS (macro F1)
| Configuration | F1 (mean +/- std) | Delta vs full | p_raw | p_Holm | Cohen's d |
|---|---|---|---|---|---|---|
| Transformer-only | 62.84 +/- 3.25 | -6.80 | 0.0452 | 0.4519 | 1.286 |
| w/o char encoder | 63.70 +/- 3.42 | -5.94 | 0.0658 | 0.5368 | 1.124 |
| w/o gated fusion | 70.56 +/- 1.36 | +0.93 | 0.5284 | 1.0000 | -0.308 |
| w/o CRF | 70.38 +/- 0.30 | +0.75 | 0.4911 | 1.0000 | -0.339 |
| CAGF-CBT+CRF (full) | 69.64 +/- 1.85 | +0.00 | -- | -- | -- |

## Grammeme (macro F1)
| Configuration | F1 (mean +/- std) | Delta vs full | p_raw | p_Holm | Cohen's d |
|---|---|---|---|---|---|---|
| Transformer-only | 24.36 +/- 6.39 | -21.00 | 0.0018 | 0.0201 * | 3.284 |
| w/o char encoder | 26.64 +/- 4.08 | -18.72 | 0.0010 | 0.0116 * | 3.883 |
| w/o gated fusion | 47.21 +/- 1.99 | +1.84 | 0.2654 | 1.0000 | -0.579 |
| w/o CRF | 42.92 +/- 5.01 | -2.44 | 0.3463 | 1.0000 | 0.477 |
| CAGF-CBT+CRF (full) | 45.36 +/- 1.26 | +0.00 | -- | -- | -- |
