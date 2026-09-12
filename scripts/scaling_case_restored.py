"""Reproduce the case-restored Lemmas column of the scaling table (paper Section 4.3).

The scaling campaign (results_cv_silver/scaling_{100k,210k,500k,1000k}) was run
from a code snapshot that predated the ``restore_lemma_case`` post-processing
step documented in Section 2.2 of the paper: edit scripts decode lower-cased
lemmas, and the official CoNLL-2018 scorer compares lemma strings byte-for-byte,
so raw scores under-report Lemmas by several points while UPOS/UFeats/AllTags
are unaffected. The published Lemmas column therefore applies the documented,
deterministic restoration rule to the *stored* predictions before scoring. The
rule uses only the surface form and the model-predicted UPOS of the same
prediction file -- no gold information enters it.

This script re-derives every number in that column from the released artifacts:

  * pooled and per-fold case-restored Lemmas per budget (Table 6, Lemmas column;
    other columns come from scaling_curve.py unchanged);
  * paired t-tests for the three adjacent scaling steps with Holm correction
    over the 4 metrics x 3 steps family (the "statistical analysis" paragraph).

Verification performed with it (pooled, %): 100K 80.82, 210K 82.55,
500K 84.38, 1M 84.96; per-fold means 80.80/82.54/84.37/84.96. Holm-corrected
210K->500K: UPOS 0.0009, Lemmas 0.0009, AllTags 0.030; all 500K->1M >= 0.18.

Usage
-----
    PYTHONPATH=. .venv/bin/python scripts/scaling_case_restored.py \
        --root results_cv_silver
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Dict, List

from cagf.data import restore_lemma_case

BUDGETS = ['100k', '210k', '500k', '1000k']
CONFIG = 'silver_filtered_pretrain_ktb_finetune'


def _rows(path: Path):
    """Word rows (ID without '-' or '.') of a CoNLL-U file."""
    for ln in path.read_text(encoding='utf-8').splitlines():
        if not ln or ln.startswith('#'):
            continue
        cols = ln.split('\t')
        if len(cols) >= 10 and '-' not in cols[0] and '.' not in cols[0]:
            yield cols


def _restored_accuracy(gold_path: Path, pred_path: Path) -> float:
    """Strict lemma accuracy after applying the Section 2.2 case rule."""
    gold = list(_rows(gold_path))
    preds = list(_rows(pred_path))
    ok = 0
    for g, p in zip(gold, preds):
        lemma = restore_lemma_case(p[1], p[2], p[3])
        if g[2] == lemma:
            ok += 1
    return ok / len(gold)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default='results_cv_silver',
                    help='directory holding scaling_<budget>k/ subdirs')
    ap.add_argument('--out', default=None,
                    help='optional JSON output (e.g. results_cv_silver/scaling_lemmas_case_restored.json)')
    args = ap.parse_args()

    root = Path(args.root)
    pooled: Dict[str, float] = {}
    per_fold: Dict[str, List[float]] = {}
    for b in BUDGETS:
        run = root / f'scaling_{b}' / CONFIG
        pooled[b] = _restored_accuracy(run / 'gold_all.conllu', run / 'pred_all.conllu')
        per_fold[b] = [
            _restored_accuracy(root / f'scaling_{b}' / '_gold_folds' / f'fold_{i}.conllu',
                               run / f'fold_{i}.conllu')
            for i in range(10)
        ]
        mean = statistics.mean(per_fold[b])
        sd = statistics.pstdev(per_fold[b])
        print(f'{b:>5}: pooled={100*pooled[b]:.2f}  per-fold mean±sd={100*mean:.2f}±{100*sd:.2f}')

    # paired t-tests for adjacent steps; Lemmas from the restored values above,
    # the other three metrics from scaling_perfold_stats.json (official scorer)
    try:
        from scipy import stats as sps
    except ImportError:  # pragma: no cover
        sps = None
    steps = list(zip(BUDGETS, BUDGETS[1:]))
    metrics: Dict[str, Dict[str, List[float]]] = {'Lemmas': per_fold}
    stats_path = root / 'scaling_perfold_stats.json'
    if stats_path.exists():
        ps = json.loads(stats_path.read_text(encoding='utf-8'))
        for m in ('UPOS', 'UFeats', 'AllTags'):
            metrics[m] = {b: ps['budgets'][i]['per_fold'][m]
                          for i, b in enumerate(BUDGETS)}
    tests = []
    if sps is not None:
        for m in ('UPOS', 'UFeats', 'Lemmas', 'AllTags'):
            for a, c in steps:
                t, p = sps.ttest_rel(metrics[m][a], metrics[m][c])
                tests.append({'metric': m, 'step': f'{a}->{c}',
                              'delta_pp': round(100 * (statistics.mean(metrics[m][c])
                                                       - statistics.mean(metrics[m][a])), 2),
                              'p': p})
        n = len(tests)
        running = 0.0
        for rank, i in enumerate(sorted(range(n), key=lambda k: tests[k]['p'])):
            running = max(running, min(1.0, (n - rank) * tests[i]['p']))
            tests[i]['p_holm'] = running
        print('\npaired t-tests (Holm over 12):')
        for t in tests:
            print(f"  {t['metric']:>8} {t['step']:>13}: Δ={t['delta_pp']:+5.2f} pp  "
                  f"p={t['p']:.4g}  pHolm={t['p_holm']:.4g}")
    else:
        print('\n(scipy not installed -- skipping t-tests; '
              'pooled/per-fold values above do not need it)')

    if args.out:
        Path(args.out).write_text(json.dumps(
            {'pooled': pooled, 'per_fold': per_fold, 'tests': tests},
            ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'\nJSON: {args.out}')


if __name__ == '__main__':
    main()
