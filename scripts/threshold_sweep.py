"""E.5 (R2.1): sigmoid-threshold sweep for the grammeme head, from CV dumps.

The inference rule in both train loops is `sigmoid > 0.5` for every grammeme
(cagf/train_loop.py, cagf/hf_train_loop.py). This script asks whether a
different threshold helps, tuning ONLY on each fold's dev split:

  * global: one threshold for all grammemes, swept over 0.30..0.70 step 0.05,
    best-on-dev applied to test;
  * per-feature: one threshold per grammeme, each maximising that feature's
    dev binary F1 over the same grid, applied to test unchanged.

Bundle metric mirrors the official CoNLL-2018 UFeats exactly: a token scores
correct iff its predicted Feature=Value set equals the gold set (empty ==
empty counts). This is validated against the official per-fold UFeats in
per_fold.json for every fold (max discrepancy reported).

Inputs are the fold_N_gramprobs.npz dumps written by the patched drivers
(train seeds 13/2024 for CAGF 10-fold and KazRoBERTa folds {0,3,7}; the
paper's seed-42 runs predate the dump patch). Output:
results_cv/threshold_sweep.json. A null result is fine: if 0.5 is near the
dev optimum, that is the answer.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, 'third_party')
from conll18_ud_eval import UNIVERSAL_FEATURES  # noqa: E402


def eval_columns(grammemes):
    """Columns that the official UFeats metric actually scores: the vendored
    scorer filters FEATS to UNIVERSAL_FEATURES, which EXCLUDES the
    language-specific psor features (Number[psor], Person[psor]) — the
    prediction writer drops them too (finding of E.3). '<unk>' is the vocab
    pad, never a feature."""
    return [i for i, g in enumerate(grammemes)
            if g != '<unk>' and str(g).split('=', 1)[0] in UNIVERSAL_FEATURES]


GRID = [round(0.30 + 0.05 * i, 2) for i in range(9)]  # 0.30 .. 0.70

CONDITIONS = {
    'cagf_gold_seed13': 'results_cv_silver/seed13/ktb_only',
    'cagf_silver_seed13': 'results_cv_silver/seed13/silver_unfiltered_pretrain_ktb_finetune',
    'cagf_gold_seed2024': 'results_cv_silver/seed2024/ktb_only',
    'cagf_silver_seed2024': 'results_cv_silver/seed2024/silver_unfiltered_pretrain_ktb_finetune',
    'kazr_gold_seed13': 'results_cv_kazroberta/seed13_gold/gold_only',
    'kazr_silver_seed13': 'results_cv_kazroberta/seed13_silver/silver_finetune',
    'kazr_gold_seed2024': 'results_cv_kazroberta/seed2024_gold/gold_only',
    'kazr_silver_seed2024': 'results_cv_kazroberta/seed2024_silver/silver_finetune',
}


def bundle_f1(pred: np.ndarray, gold: np.ndarray) -> float:
    """Official-UFeats-style: fraction of tokens with exactly equal sets."""
    return float((pred == gold).all(axis=1).mean())


def binary_f1(pred: np.ndarray, gold: np.ndarray) -> float:
    tp = int((pred & gold).sum())
    fp = int((pred & ~gold.astype(bool)).sum())
    fn = int((~pred.astype(bool) & gold).sum())
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def sweep_fold(npz_path: Path) -> dict:
    d = np.load(npz_path, allow_pickle=True)
    grammemes = [str(g) for g in d['grammemes']]
    keep = eval_columns(grammemes)
    n_psor = len(grammemes) - 1 - len(keep)  # columns dropped: unk + psor
    probs_dev = d['probs_dev'][:, keep].astype(np.float32)
    gold_dev = d['gold_dev'][:, keep].astype(bool)
    probs_test = d['probs_test'][:, keep].astype(np.float32)
    gold_test = d['gold_test'][:, keep].astype(bool)

    def pred_at(probs, tau):
        return probs > tau

    # -- global threshold: best dev bundle F1, ties resolved toward 0.5
    dev_curve = {t: bundle_f1(pred_at(probs_dev, t), gold_dev) for t in GRID}
    tau_star = min(GRID, key=lambda t: (-dev_curve[t], abs(t - 0.5)))

    # -- per-feature thresholds: best dev binary F1 per column, ties toward 0.5
    taus_feat = []
    for j in range(probs_dev.shape[1]):
        best_t, best_f = 0.5, -1.0
        for t in GRID:
            f = binary_f1(pred_at(probs_dev[:, j], t), gold_dev[:, j])
            if f > best_f or (f == best_f and abs(t - 0.5) < abs(best_t - 0.5)):
                best_t, best_f = t, f
        taus_feat.append(best_t)
    taus_feat = np.array(taus_feat, dtype=np.float32)

    return {
        'fold': int(str(npz_path).split('fold_')[1].split('_')[0]),
        'best_epoch': int(d['best_epoch']),
        'n_dev': int(gold_dev.shape[0]),
        'n_test': int(gold_test.shape[0]),
        'n_columns_scored': len(keep),
        'n_columns_dropped_psor': n_psor,
        'dev_f1_at_0.5': dev_curve[0.5],
        'dev_f1_at_tau_star': dev_curve[tau_star],
        'tau_star': tau_star,
        'test_f1_at_0.5': bundle_f1(pred_at(probs_test, 0.5), gold_test),
        'test_f1_global': bundle_f1(pred_at(probs_test, tau_star), gold_test),
        'test_f1_perfeat': bundle_f1(probs_test > taus_feat, gold_test),
        'taus_feat': [float(t) for t in taus_feat],
        # pooled aggregation helpers — per-token match flags, not matrices
        # (the grammeme vocabulary differs across folds, so columns don't align)
        '_match_05': (pred_at(probs_test, 0.5) == gold_test).all(axis=1),
        '_match_global': (pred_at(probs_test, tau_star) == gold_test).all(axis=1),
        '_match_perfeat': ((probs_test > taus_feat) == gold_test).all(axis=1),
    }


def official_fold_ufeats(dir_path: Path) -> dict:
    pf = dir_path / 'per_fold.json'
    if not pf.exists():
        return {}
    return {e['fold']: e['official']['UFeats'] for e in json.loads(pf.read_text())}


def main() -> None:
    report = {'grid': GRID, 'conditions': {}, 'validation': {}}
    for name, path in CONDITIONS.items():
        dir_path = Path(path)
        npzs = sorted(dir_path.glob('fold_*_gramprobs.npz'))
        if not npzs:
            print(f'[{name}] NO DUMPS — skipped')
            continue
        folds = [sweep_fold(p) for p in npzs]
        official = official_fold_ufeats(dir_path)

        # validation vs official per-fold UFeats (bundle F1 at 0.5 must mirror it)
        disc = max(abs(f['test_f1_at_0.5'] - official[f['fold']]) for f in folds
                   if f['fold'] in official) if official else None
        report['validation'][name] = {'max_abs_diff_vs_official': disc,
                                      'official_ufeats_available': bool(official)}

        pooled = {
            'test_pooled_at_0.5': float(np.concatenate([f['_match_05'] for f in folds]).mean()),
            'test_pooled_global': float(np.concatenate([f['_match_global'] for f in folds]).mean()),
            'test_pooled_perfeat': float(np.concatenate([f['_match_perfeat'] for f in folds]).mean()),
        }
        for f in folds:  # strip numpy payloads before serialising
            for k in ('_match_05', '_match_global', '_match_perfeat'):
                del f[k]

        n = len(folds)
        agg = {k: sum(f[k] for f in folds) / n for k in
               ('dev_f1_at_0.5', 'dev_f1_at_tau_star', 'test_f1_at_0.5',
                'test_f1_global', 'test_f1_perfeat')}
        taus = [f['tau_star'] for f in folds]
        agg['tau_star_distribution'] = {str(t): taus.count(t) for t in GRID if taus.count(t)}
        report['conditions'][name] = {'n_folds': n, 'folds': folds,
                                      'mean': agg, 'pooled': pooled}
        print(f'[{name}] n={n} | val_disc={disc if disc is None else f"{disc:.4f}"} | '
              f'test mean: 0.5→{agg["test_f1_at_0.5"]*100:.2f} '
              f'global→{agg["test_f1_global"]*100:.2f} '
              f'perfeat→{agg["test_f1_perfeat"]*100:.2f} | '
              f'pooled: {pooled["test_pooled_at_0.5"]*100:.2f} → '
              f'{pooled["test_pooled_global"]*100:.2f} / {pooled["test_pooled_perfeat"]*100:.2f}')

    out = Path('results_cv/threshold_sweep.json')
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\nWrote {out}')


if __name__ == '__main__':
    main()
