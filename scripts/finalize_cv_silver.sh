#!/usr/bin/env bash
# Finalize the silver-transfer cross-validation analysis (T7) after the
# overnight run_cv_silver.py completes. Answers the headline question:
#   "Does silver-corpus pretraining help morphological analysis under CV?"
#
# Reference config is ktb_only (no silver). Comparisons are
# silver_filtered_pretrain_ktb_finetune (and silver_unfiltered_* if run).
# Family of hypotheses = n_configs_without_silver x 4 official metrics.
#
# Usage: bash scripts/finalize_cv_silver.sh [results_cv_silver/stratified]
set -e
cd "$(dirname "$0")/.."

CV_DIR="${1:-results_cv_silver/stratified}"

if [ ! -f "$CV_DIR/cv_raw.json" ]; then
  echo "ERROR: $CV_DIR/cv_raw.json not found. Is the silver CV run complete?"
  exit 1
fi

echo "=== 0. Per-config jack-knife summary (T7 headline) ==="
PYTHONPATH=. .venv/bin/python3 -c "
import json
from pathlib import Path
cv_dir = Path('$CV_DIR')
for cfg_dir in sorted(cv_dir.iterdir()):
    if not cfg_dir.is_dir() or cfg_dir.name.startswith('_'):
        continue
    jk = cfg_dir / 'jackknifed.json'
    pf = cfg_dir / 'per_fold.json'
    if jk.exists():
        j = json.loads(jk.read_text())['official_jackknifed']
        n = json.loads(jk.read_text())['n_folds']
        print(f'  {cfg_dir.name:<42} (n_folds={n})  '
              f'Lemmas={j[\"Lemmas\"]*100:6.2f}  UPOS={j[\"UPOS\"]*100:6.2f}  '
              f'UFeats={j[\"UFeats\"]*100:6.2f}  AllTags={j[\"AllTags\"]*100:6.2f}')
"
echo ""

echo "=== 1. Significance: does silver help? (fold-wise, Holm-Bonferroni) ==="
echo "    Reference = ktb_only (no silver). Family = (n_silver_configs x 4 metrics)."
PYTHONPATH=. .venv/bin/python3 scripts/significance_test.py \
  --results "$CV_DIR/cv_raw.json" \
  --unit fold --holm --alpha 0.05 \
  --reference ktb_only \
  --tasks official.UPOS official.UFeats official.Lemmas official.AllTags \
  --json-out "$CV_DIR/significance_fold.json"
echo ""

echo "=== 2. Effect-size + CI summary (the numbers to quote in the paper) ==="
PYTHONPATH=. .venv/bin/python3 -c "
import json
from pathlib import Path
sig_path = Path('$CV_DIR/significance_fold.json')
if not sig_path.exists():
    print('  (significance_fold.json not found)')
else:
    fam = json.loads(sig_path.read_text())
    print(f'  {\"metric\":<18} {\"mean_diff pp\":>13} {\"95% CI pp\":>22} {\"Cohen d\":>9} {\"effect\":<11} {\"Holm p\":>8} {\"sig\"}')
    for r in fam:
        ci = f'[{r[\"bootstrap_ci_lo\"]*100:+.2f}, {r[\"bootstrap_ci_hi\"]*100:+.2f}]'
        star = '*' if r['significant_holm'] else ''
        print(f'  {r[\"task\"]:<18} {r[\"mean_diff\"]*100:>+13.2f} {ci:>22} '
              f'{r[\"cohens_d\"]:>9.3f} {r[\"effect\"]:<11} {r[\"p_holm\"]:>8.4f} {star}')
"
echo ""

echo "=== Done. Review $CV_DIR/significance_fold.json ==="
echo "=== Noise-floor check: any |mean_diff| < 0.40pp is within MPS nondeterminism ==="
echo ""
echo "=== (Optional) regenerate the silver CV table into results/tables/ ==="
echo "    PYTHONPATH=. .venv/bin/python3 scripts/make_tables.py \\"
echo "      --ablation results/ablation_raw.json --baselines results/baseline_raw.json \\"
echo "      --cv $CV_DIR/cv_raw.json --out-dir results/tables"
