#!/usr/bin/env bash
# Finalize CV analysis after the ablation CV run completes.
# Runs: significance testing (fold-wise, Holm), table regeneration, feat constraints.
# Usage: bash scripts/finalize_cv.sh
set -e
cd "$(dirname "$0")/.."

CV_DIR="results_cv/stratified"
echo "=== 1. Significance testing (fold-wise, Holm-Bonferroni) ==="
PYTHONPATH=. .venv/bin/python3 scripts/significance_test.py \
  --results "$CV_DIR/cv_raw.json" \
  --unit fold --holm --alpha 0.05 \
  --tasks official.UPOS official.UFeats official.Lemmas official.AllTags \
  --json-out "$CV_DIR/significance_fold.json"
echo ""
echo "=== 2. Regenerate CV results table ==="
PYTHONPATH=. .venv/bin/python3 scripts/make_tables.py \
  --ablation results/ablation_raw.json \
  --baselines results/baseline_raw.json \
  --significance results/significance.json \
  --cv "$CV_DIR/cv_raw.json" \
  --out-dir results/tables
echo ""
echo "=== 3. Per-config jack-knife summary ==="
PYTHONPATH=. .venv/bin/python3 -c "
import json, statistics
from pathlib import Path
cv_dir = Path('$CV_DIR')
for cfg_dir in sorted(cv_dir.iterdir()):
    if not cfg_dir.is_dir() or cfg_dir.name.startswith('_'):
        continue
    jk = cfg_dir / 'jackknifed.json'
    if jk.exists():
        j = json.loads(jk.read_text())['official_jackknifed']
        print(f'  {cfg_dir.name:<28} Lemmas={j[\"Lemmas\"]*100:.2f} UPOS={j[\"UPOS\"]*100:.2f} '
              f'UFeats={j[\"UFeats\"]*100:.2f} AllTags={j[\"AllTags\"]*100:.2f}')
"
echo ""
echo "=== Done. Review $CV_DIR/significance_fold.json and results/tables/cv_results.md ==="
