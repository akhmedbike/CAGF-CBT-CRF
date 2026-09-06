"""E.3 (revision plan, R2.4): UFeats over the full trained inventory vs the
official universal-feature subset.

The CoNLL-2018 scorer keeps only universal FEATS at load time
(third_party/conll18_ud_eval.py: ``feat.split("=", 1)[0] in UNIVERSAL_FEATURES``),
while the models were trained on the complete UD_Kazakh-KTB inventory,
including language-specific attributes (e.g. Number[psor], Person[psor]).
The reviewer asked whether the model behaves differently on the discarded
attributes. This script re-scores the stored pooled predictions three ways:

  * universal  -- official UFeats (unchanged scorer behaviour)
  * full       -- no attribute filtering (bundle-exact-match over all pairs)
  * langspec   -- only attributes OUTSIDE the scorer's universal set

The filter is bypassed by substituting the module-level UNIVERSAL_FEATURES
container with a permissive/dual one; alignment and metric code are untouched.

CPU-only, reads stored predictions, trains nothing.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/feats_full_vs_universal.py
"""
from __future__ import annotations

import json
from pathlib import Path

from cagf.official_eval import conll18_ud_eval  # vendored, path-injected

CONDITIONS = {
    'cagf_gold': Path('results_cv_silver/stratified/ktb_only'),
    'cagf_silver': Path('results_cv_silver/stratified/silver_unfiltered_pretrain_ktb_finetune'),
    'kazroberta_gold': Path('results_cv_kazroberta/gold_only'),
    'kazroberta_silver': Path('results_cv_kazroberta/silver_finetune'),
}


class _AnyAttr:
    """Stands in for UNIVERSAL_FEATURES; mode 'all' keeps everything,
    'langspec' keeps exactly the non-universal attributes."""
    def __init__(self, original, mode: str):
        self.original = original
        self.mode = mode

    def __contains__(self, attr: str) -> bool:
        if self.mode == 'all':
            return True
        return attr not in self.original   # mode == 'langspec'


def _ufeats(gold_path: Path, pred_path: Path) -> dict:
    gold = conll18_ud_eval.load_conllu_file(str(gold_path))
    system = conll18_ud_eval.load_conllu_file(str(pred_path))
    scores = conll18_ud_eval.evaluate(gold, system)
    return {m: float(scores[m].f1) for m in ('UFeats', 'AllTags')}


def main() -> None:
    original = conll18_ud_eval.UNIVERSAL_FEATURES
    out: dict = {'universal_feature_attributes': sorted(original)}
    for name, cfg_dir in CONDITIONS.items():
        gold = cfg_dir / 'gold_all.conllu'
        pred = cfg_dir / 'pred_all.conllu'
        if not (gold.exists() and pred.exists()):
            print(f'[{name}] SKIP: missing pooled files')
            continue
        res = {}
        try:
            conll18_ud_eval.UNIVERSAL_FEATURES = original
            res['universal'] = _ufeats(gold, pred)
            conll18_ud_eval.UNIVERSAL_FEATURES = _AnyAttr(original, 'all')
            res['full_inventory'] = _ufeats(gold, pred)
            conll18_ud_eval.UNIVERSAL_FEATURES = _AnyAttr(original, 'langspec')
            res['language_specific_only'] = _ufeats(gold, pred)
        finally:
            conll18_ud_eval.UNIVERSAL_FEATURES = original
        out[name] = res
        print(f'[{name}]')
        for mode in ('universal', 'full_inventory', 'language_specific_only'):
            u, a = res[mode]['UFeats'] * 100, res[mode]['AllTags'] * 100
            print(f'  {mode:22s} UFeats={u:6.2f}  AllTags={a:6.2f}')

    # inventory classification: which trained attributes are language-specific
    # and how many gold tokens carry them
    gold_all = CONDITIONS['cagf_gold'] / 'gold_all.conllu'
    attrs_universal, attrs_langspec = {}, {}
    for line in gold_all.read_text(encoding='utf-8').splitlines():
        if not line or line.startswith('#') or '\t' not in line:
            continue
        cols = line.split('\t')
        if '-' in cols[0] or '.' in cols[0]:
            continue
        for feat in cols[5].split('|'):
            if feat == '_':
                continue
            attr = feat.split('=', 1)[0]
            d = attrs_universal if attr in original else attrs_langspec
            d[attr] = d.get(attr, 0) + 1
    out['gold_inventory'] = {
        'universal_attrs': attrs_universal,
        'language_specific_attrs': attrs_langspec,
    }
    print('\nGold inventory: universal attrs '
          f'{ {k: v for k, v in sorted(attrs_universal.items())} }')
    print('Gold inventory: LANGUAGE-SPECIFIC attrs '
          f'{ {k: v for k, v in sorted(attrs_langspec.items())} }')

    # token-restricted view: among tokens whose GOLD bundle carries at least
    # one language-specific pair, how often does the predicted psor bundle
    # match exactly / per-pair P-R-F1. Avoids the ~87% trivial "_" matches
    # that inflate the langspec-only bundle F1 above.
    def _feats_by_token(path: Path):
        tokens = []
        for line in path.read_text(encoding='utf-8').splitlines():
            if not line or line.startswith('#') or '\t' not in line:
                continue
            cols = line.split('\t')
            if '-' in cols[0] or '.' in cols[0]:
                continue
            tokens.append(set(f for f in cols[5].split('|')
                              if f != '_' and f.split('=', 1)[0] not in original))
        return tokens

    for name, cfg_dir in CONDITIONS.items():
        gold_p, pred_p = cfg_dir / 'gold_all.conllu', cfg_dir / 'pred_all.conllu'
        if not (gold_p.exists() and pred_p.exists()):
            continue
        g_toks, p_toks = _feats_by_token(gold_p), _feats_by_token(pred_p)
        assert len(g_toks) == len(p_toks), f'{name}: token count mismatch'
        idx = [i for i, g in enumerate(g_toks) if g]
        exact = sum(1 for i in idx if g_toks[i] == p_toks[i])
        tp = sum(len(g_toks[i] & p_toks[i]) for i in idx)
        gold_n = sum(len(g_toks[i]) for i in idx)
        pred_n = sum(len(p_toks[i]) for i in idx)
        prec = tp / pred_n if pred_n else 0.0
        rec = tp / gold_n if gold_n else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out[name]['langspec_tokens_with_gold_psor'] = {
            'n_tokens': len(idx),
            'bundle_exact_acc': round(exact / len(idx), 4) if idx else None,
            'pair_precision': round(prec, 4), 'pair_recall': round(rec, 4),
            'pair_f1': round(f1, 4),
        }
        print(f'[{name}] psor-bearing tokens={len(idx)} '
              f'bundle-exact={exact / len(idx) * 100 if idx else 0:.2f}% '
              f'pair-P={prec * 100:.2f} R={rec * 100:.2f} F1={f1 * 100:.2f}')

    dest = Path('results_cv/feats_full_vs_universal.json')
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\nWrote {dest}')


if __name__ == '__main__':
    main()
