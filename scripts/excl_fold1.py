"""E.2 (revision plan, R3.2): pooled out-of-fold scores excluding fold 1.

Hyperparameters were selected on the dev block of fold 0, which later served as
the TEST block of fold 1. To quantify the concern without retraining, this
script rebuilds the pooled prediction set from the stored per-fold CoNLL-U
files, leaving fold 1 out, and re-scores it with the official CoNLL-2018
scorer. If the deltas vs. the full pooled scores stay under the 0.40-pp
run-to-run floor, the non-nested selection did not inflate the reported
results through that fold.

CPU-only, reads stored predictions, trains nothing.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/excl_fold1.py
"""
from __future__ import annotations

import json
from pathlib import Path

from cagf.official_eval import evaluate_conllu
from scripts.run_cv import _concat_conllu

# condition name -> (config dir with fold_<k>.conllu, sibling _gold_folds dir)
CONDITIONS = {
    'cagf_gold': Path('results_cv_silver/stratified/ktb_only'),
    'cagf_silver': Path('results_cv_silver/stratified/silver_unfiltered_pretrain_ktb_finetune'),
    'kazroberta_gold': Path('results_cv_kazroberta/gold_only'),
    'kazroberta_silver': Path('results_cv_kazroberta/silver_finetune'),
}


def gold_folds_for(cfg_dir: Path) -> Path:
    for cand in (cfg_dir.parent / '_gold_folds', cfg_dir / '_gold_folds'):
        if cand.exists():
            return cand
    raise FileNotFoundError(f'no _gold_folds next to {cfg_dir}')


def main() -> None:
    out: dict = {}
    for name, cfg_dir in CONDITIONS.items():
        gold_dir = gold_folds_for(cfg_dir)
        folds = sorted(int(p.stem.split('_')[1]) for p in cfg_dir.glob('fold_*.conllu')
                       if p.stem != 'fold_all')
        keep = [f for f in folds if f != 1]
        pred_parts = [cfg_dir / f'fold_{f}.conllu' for f in keep]
        gold_parts = [gold_dir / f'fold_{f}.conllu' for f in keep]
        missing = [p for p in pred_parts + gold_parts if not p.exists()]
        if missing:
            print(f'[{name}] SKIP: missing {missing[:3]}')
            continue
        tmp_pred = cfg_dir / '_excl_fold1_pred.conllu'
        tmp_gold = cfg_dir / '_excl_fold1_gold.conllu'
        n_pred = _concat_conllu(pred_parts, tmp_pred)
        n_gold = _concat_conllu(gold_parts, tmp_gold)
        assert n_pred == n_gold
        excl = evaluate_conllu(str(tmp_gold), str(tmp_pred))
        full = json.loads((cfg_dir / 'jackknifed.json').read_text())['official_jackknifed']
        deltas = {m: round((full[m] - excl[m]) * 100, 3) for m in excl}
        out[name] = {
            'folds_used': keep,
            'n_sentences': n_pred,
            'official_excl_fold1': excl,
            'official_full_pooled': full,
            'delta_pp_full_minus_excl': deltas,
        }
        print(f'[{name}] folds={keep} n={n_pred}')
        print(f'  excl fold1 : ' + ' '.join(f'{m}={v*100:.2f}' for m, v in excl.items()))
        print(f'  full pooled: ' + ' '.join(f'{m}={v*100:.2f}' for m, v in full.items()))
        print(f'  delta pp   : ' + ' '.join(f'{m}={d:+.2f}' for m, d in deltas.items()))
        tmp_pred.unlink()
        tmp_gold.unlink()

    dest = Path('results_cv/excl_fold1.json')
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\nWrote {dest}')


if __name__ == '__main__':
    main()
