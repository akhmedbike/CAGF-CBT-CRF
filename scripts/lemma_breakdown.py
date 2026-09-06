"""E.1 (revision plan, R3.9): lemmatization breakdown from stored predictions.

Lemma accuracy stratified by, per token of the pooled out-of-fold evaluation:

  (a) gold edit script present vs absent in that fold's training inventory
      (the 91.08% / 8.92% rule-inventory coverage already reported in §2.4);
  (b) surface form in-vocabulary vs OOV w.r.t. the fold's gold training data;
  (c) training-frequency bins: 1, 2-5, 6+ occurrences, unseen;
  (d) token length in characters (proxy for morphological complexity).

Computed for the four main conditions from their stored fold_<k>.conllu
predictions; fold train inventories are rebuilt deterministically with
cagf.folds.make_folds (stratified, seed 42) + cagf.data.build_vocabs -- the
same code path the training runs used, so the inventories are identical.

CPU-only, trains nothing.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/lemma_breakdown.py
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import yaml

from cagf.data import build_vocabs, form_to_edit_script, read_conllu
from cagf.folds import make_folds

CONDITIONS = {
    'cagf_gold': Path('results_cv_silver/stratified/ktb_only'),
    'cagf_silver': Path('results_cv_silver/stratified/silver_unfiltered_pretrain_ktb_finetune'),
    'kazroberta_gold': Path('results_cv_kazroberta/gold_only'),
    'kazroberta_silver': Path('results_cv_kazroberta/silver_finetune'),
}


def _tokens(path: Path):
    """Yield (form, lemma) for every syntactic word, in file order."""
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line or line.startswith('#') or '\t' not in line:
            continue
        cols = line.split('\t')
        if '-' in cols[0] or '.' in cols[0]:
            continue
        yield cols[1], cols[2]


def main() -> None:
    cfg = yaml.safe_load(Path('configs/default.yaml').read_text(encoding='utf-8'))
    sents = (read_conllu(cfg['data']['train_path']) +
             read_conllu(cfg['data']['dev_path']) +
             read_conllu(cfg['data']['test_path']))
    folds = make_folds(sents, k=10, seed=42, strategy='stratified')

    # per-fold training statistics, rebuilt exactly as during training
    fold_stats = []
    for f in folds:
        train_s = [sents[i] for i in f.train_idx]
        vocabs = build_vocabs(train_s, min_word_freq=1, max_word_vocab=50000)
        rules = set(vocabs.lemma_rule_vocab.itos)
        freq = Counter(tok.form for s in train_s for tok in s.tokens)
        fold_stats.append({'rules': rules, 'freq': freq})

    strata = {
        'rule_seen': Counter(), 'rule_unseen': Counter(),
        'iv': Counter(), 'oov': Counter(),
        'freq_1': Counter(), 'freq_2_5': Counter(), 'freq_6plus': Counter(),
        'freq_unseen': Counter(),
        'len_1_5': Counter(), 'len_6_8': Counter(), 'len_9_11': Counter(),
        'len_12plus': Counter(),
        'all': Counter(),
    }
    rows: list[dict] = []
    for name, cfg_dir in CONDITIONS.items():
        gold_dir = cfg_dir.parent / '_gold_folds'
        counters = {k: Counter() for k in
                    ('rule_seen', 'rule_unseen', 'iv', 'oov', 'freq_1', 'freq_2_5',
                     'freq_6plus', 'freq_unseen', 'len_1_5', 'len_6_8', 'len_9_11',
                     'len_12plus', 'all')}
        for k in range(10):
            gold = list(_tokens(gold_dir / f'fold_{k}.conllu'))
            pred = list(_tokens(cfg_dir / f'fold_{k}.conllu'))
            assert len(gold) == len(pred), f'{name} fold {k}: length mismatch'
            st = fold_stats[k]
            for (gform, glemma), (_, plemma) in zip(gold, pred):
                ok = int(plemma == glemma)
                rule = form_to_edit_script(gform, glemma)
                seen = rule in st['rules']
                n = st['freq'].get(gform, 0)
                L = len(gform)
                for key, _ in (
                    ('rule_seen' if seen else 'rule_unseen', True),
                    ('iv' if n > 0 else 'oov', True),
                    ('freq_unseen' if n == 0 else 'freq_1' if n == 1
                     else 'freq_2_5' if n <= 5 else 'freq_6plus', True),
                    ('len_1_5' if L <= 5 else 'len_6_8' if L <= 8 else
                     'len_9_11' if L <= 11 else 'len_12plus', True),
                    ('all', True),
                ):
                    counters[key]['n'] += 1
                    counters[key]['ok'] += ok
                rows.append({'condition': name, 'fold': k, 'form': gform,
                             'gold_lemma': glemma, 'pred_lemma': plemma,
                             'correct': ok, 'rule_seen': int(seen),
                             'train_freq': n, 'length': L})
        for key, c in counters.items():
            strata[key][name] = {'n': c['n'], 'acc': round(c['ok'] / c['n'], 4)
                                 if c['n'] else None}
        pct = lambda key: strata[key][name]['acc'] * 100
        print(f'[{name}] all={pct("all"):.2f} '
              f'rule_seen={pct("rule_seen"):.2f} rule_unseen={pct("rule_unseen"):.2f} '
              f'iv={pct("iv"):.2f} oov={pct("oov"):.2f}')
        print(f'          freq1={pct("freq_1"):.2f} freq2-5={pct("freq_2_5"):.2f} '
              f'freq6+={pct("freq_6plus"):.2f} unseen={pct("freq_unseen"):.2f} '
              f'len<=5={pct("len_1_5"):.2f} len6-8={pct("len_6_8"):.2f} '
              f'len9-11={pct("len_9_11"):.2f} len12+={pct("len_12plus"):.2f}')

    Path('results_cv/lemma_breakdown.json').write_text(
        json.dumps(strata, ensure_ascii=False, indent=2), encoding='utf-8')
    with open('results_cv/lemma_breakdown_tokens.csv', 'w', newline='',
              encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print('\nWrote results_cv/lemma_breakdown.json and lemma_breakdown_tokens.csv')


if __name__ == '__main__':
    main()
