# Sentence-level cluster bootstrap 95% CIs

B = 1000 resamples, seed = 42, resampling unit = sentence (CRF: 162 test sentences / 1,594 tokens; CAGF-CBT+CRF: 109 test sentences / 1,094 tokens). Temperatures are the archived dev-fitted values (T = 2.55 NLL, T = 2.60 ECE grid) and are not refitted. CAGF conditions share each iteration's resample, so the delta rows are paired. Point estimates are computed on the full test data and pass the sanity gate against docs/bali.md.

| Metric | Point | Sentence-bootstrap 95% CI |
|---|---|---|
| CRF accuracy (%) | 94.48 | 93.26–95.54 |
| CRF ECE (%) | 2.71 | 1.91–3.97 |
| CRF Brier (top-1) | 0.04 | 0.035–0.053 |
| CRF Brier (multiclass, Eq. 3) | 0.09 | 0.072–0.111 |
| CRF coverage@0.85 (%) | 92.97 | 91.74–94.23 |
| CRF risk@0.85 (%) | 3.44 | 2.48–4.50 |
| CRF review@0.70 (%) | 3.26 | 2.47–4.12 |
| CAGF accuracy (%) | 84.00 | 81.45–86.77 |
| CAGF ECE raw (%) | 12.79 | 10.52–15.17 |
| CAGF Brier (top-1) raw | 0.14 | 0.114–0.157 |
| CAGF Brier (multiclass, Eq. 3) raw | 0.29 | 0.240–0.332 |
| CAGF ECE T=2.55 (%) | 2.99 | 2.01–5.13 |
| CAGF ECE T=2.60 (%) | 2.79 | 2.03–4.98 |
| CAGF Brier (top-1) T=2.60 | 0.10 | 0.089–0.116 |
| CAGF Brier (multiclass, Eq. 3) T=2.60 | 0.24 | 0.208–0.279 |
| CAGF risk@0.85 raw (%) | 13.04 | 10.52–15.42 |
| CAGF risk@0.85 T=2.60 (%) | 5.17 | 3.40–6.83 |
| CAGF coverage@0.85 raw (%) | 92.50 | 90.74–94.22 |
| CAGF coverage@0.85 T=2.60 (%) | 67.18 | 63.02–71.13 |
| Δ risk@0.85 (T=2.60 − raw, pp) | -7.87 | -10.04–-5.93 |
| Δ coverage@0.85 (T=2.60 − raw, pp) | -25.32 | -28.52–-21.98 |

BCa intervals for the headline metrics, full replicate arrays and the sanity-gate report are in bootstrap_cis.json; histograms in figs/.

