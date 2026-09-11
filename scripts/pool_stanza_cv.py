"""Pool the per-fold Stanza predictions into one jack-knife evaluation.

The paper's headline protocol scores every gold sentence exactly once by
concatenating the out-of-fold predictions of all 10 folds and running the
official CoNLL-2018 scorer once (same as run_cv.py does for CAGF and the
encoder baselines). run_stanza_cv.py reports per-fold means; this script
re-aggregates the same artifacts into the pooled numbers so the Stanza row
is directly comparable. Pure CPU post-processing — no model is re-run.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from cagf.official_eval import evaluate_conllu


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--in-dir', default='results_stanza/stratified')
    ap.add_argument('--folds', default='0,1,2,3,4,5,6,7,8,9')
    args = ap.parse_args()

    root = Path(args.in_dir)
    fold_ids = [int(x) for x in args.folds.split(',') if x.strip()]
    gold_parts, pred_parts = [], []
    n_sent = 0
    for i in fold_ids:
        fold = root / f'fold_{i}'
        gold_parts.append((fold / 'gold.conllu').read_text(encoding='utf-8'))
        pred_parts.append((fold / 'pred.conllu').read_text(encoding='utf-8'))
        n_sent += sum(1 for line in gold_parts[-1].splitlines()
                      if line.startswith('# sent_id'))
        assert (fold / 'result.json').exists(), f'fold {i} not finished'

    gold_all = root / 'gold_all.conllu'
    pred_all = root / 'pred_all.conllu'
    # the official parser only accepts comment lines at block starts, so folds
    # must be separated by a blank line
    gold_all.write_text('\n\n'.join(p.rstrip('\n') for p in gold_parts) + '\n\n',
                        encoding='utf-8')
    pred_all.write_text('\n\n'.join(p.rstrip('\n') for p in pred_parts) + '\n\n',
                        encoding='utf-8')

    official = evaluate_conllu(str(gold_all), str(pred_all))
    per_fold = json.loads((root / 'stanza_results.json').read_text(
        encoding='utf-8'))
    out = {
        'protocol': 'pooled out-of-fold jack-knife (one sentence scored once)',
        'n_folds': len(fold_ids),
        'n_sentences': n_sent,
        'pooled': {k: v * 100 for k, v in official.items()},
        'per_fold_mean': per_fold['summary'],
        'computed': datetime.now().isoformat(timespec='seconds'),
        'driver': 'scripts/pool_stanza_cv.py',
        'sources': ['fold_{i}/gold.conllu + fold_{i}/pred.conllu'.format(i=i)
                    for i in fold_ids],
    }
    dest = root / 'stanza_pooled.json'
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                    encoding='utf-8')
    print(f'pooled ({n_sent} sentences): '
          + ' '.join(f'{k}={v:.2f}' for k, v in out['pooled'].items()))
    print(f'wrote {dest}')


if __name__ == '__main__':
    main()
