"""Measure POS-conditional feature-impossibility rate, before and after masking.

Loads a CoNLL-U prediction file (the output of cagf.predict_writer /
scripts.run_cv), learns the allowed (upos, feature=value) set from the TRAINING
partition only, then reports:

  * how many predicted Feature=Value combos are "impossible" (never seen with
    that UPOS in training) -- the bug class exemplified by
    ``шыға / VERB / Case=Dat``;
  * how many remain after post-hoc masking (cagf.feat_constraints);
  * the change in the official UFeats / AllTags F1 from applying the mask.

This is a Discussion-section result: a quantitative answer to the reviewer
note about morphologically implausible predictions, obtained without
retraining. It applies to any already-written prediction file, so it composes
with the CV runner (point it at ``results_cv/stratified/full_model/pred_all.conllu``).

Usage
-----
    PYTHONPATH=. python scripts/count_impossible_feats.py \\
        --gold data/gold_merged/gold_train.conllu \\
        --pred results_cv/stratified/full_model/pred_all.conllu \\
        --gold-test data/gold_merged/gold_test.conllu
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cagf.data import Sentence, read_conllu
from cagf.feat_constraints import (
    apply_constraints_to_feats,
    build_constraints,
    count_impossible,
)
from cagf.official_eval import REPORTED_METRICS, evaluate_conllu
from cagf.predict_writer import write_conllu


def _rows_from_conllu(path: str) -> list[Sentence]:
    """Prediction-side sentences: we read the predicted LEMMA/UPOS/FEATS as
    the 'prediction' rows. For the impossibility count we only need UPOS and
    FEATS, both of which read_conllu gives us."""
    return read_conllu(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--gold', required=True,
                    help='TRAIN partition CoNLL-U (constraints learned from this only)')
    ap.add_argument('--pred', required=True,
                    help='prediction CoNLL-U to evaluate (e.g. CV pred_all.conllu)')
    ap.add_argument('--gold-test', default=None,
                    help='gold test CoNLL-U matching --pred; if given, recompute '
                         'official UFeats before and after masking')
    ap.add_argument('--out', default=None,
                    help='optional JSON output path')
    args = ap.parse_args()

    train = read_conllu(args.gold)
    constraints = build_constraints(train)
    print(f'Learned constraints from {len(train)} training sentences: '
          f'{len(constraints)} (upos, feature) pairs allowed.')

    pred_sents = _rows_from_conllu(args.pred)
    # flatten to per-token rows for the impossibility tally
    rows_before = [{'upos': t.upos,
                    'feats': t.feats.split('|') if t.feats != '_' else []}
                   for s in pred_sents for t in s.tokens]
    before = count_impossible(rows_before, constraints)
    print(f'\nPrediction file: {args.pred}')
    print(f'  total predicted Feature=Value tokens: {before["total_predicted_feats"]}')
    print(f'  impossible (upos, feat=val) combos:   {before["impossible_before"]} '
          f'({100*before["impossible_rate_before"]:.2f}%)')

    # apply masking and re-tally
    masked = 0
    for s in pred_sents:
        for t in s.tokens:
            feats = t.feats.split('|') if t.feats != '_' else []
            kept = apply_constraints_to_feats(feats, t.upos, constraints)
            if len(kept) != len(feats):
                masked += (len(feats) - len(kept))
            t.feats = '|'.join(kept) if kept else '_'
    rows_after = [{'upos': t.upos,
                   'feats': t.feats.split('|') if t.feats != '_' else []}
                  for s in pred_sents for t in s.tokens]
    after = count_impossible(rows_after, constraints)
    print(f'  after masking:                        {after["impossible_before"]} '
          f'({100*after["impossible_rate_before"]:.2f}%)  [{masked} feature predictions removed]')

    # official UFeats before/after if gold-test given
    report = {
        'pred_file': args.pred,
        'n_train_sents': len(train),
        'n_constraint_pairs': len(constraints),
        'before': before,
        'after': after,
        'features_removed': masked,
    }
    if args.gold_test:
        # BEFORE: re-score the original prediction file (unmodified copy on disk)
        scores_before = evaluate_conllu(args.gold_test, args.pred)
        # AFTER: write the masked predictions to a temp file and score it
        masked_path = Path(args.pred).with_suffix('.masked.conllu')
        # rebuild predictions in writer format from the (now masked) sentences
        preds = [{'lemma': [t.lemma for t in s.tokens],
                  'upos': [t.upos for t in s.tokens],
                  'feats': [t.feats.split('|') if t.feats != '_' else [] for t in s.tokens]}
                 for s in pred_sents]
        write_conllu(pred_sents, preds, masked_path)
        scores_after = evaluate_conllu(args.gold_test, str(masked_path))
        print(f'\nOfficial CoNLL-2018 metrics (gold test = {args.gold_test}):')
        for m in REPORTED_METRICS:
            delta = scores_after[m] - scores_before[m]
            sign = '+' if delta >= 0 else ''
            print(f'  {m:<9} before={scores_before[m]:.4f}  after={scores_after[m]:.4f}  '
                  f'({sign}{delta:.4f})')
        report['official_before'] = scores_before
        report['official_after'] = scores_after
        report['official_delta'] = {m: scores_after[m] - scores_before[m] for m in REPORTED_METRICS}
        report['masked_pred_file'] = str(masked_path)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                  encoding='utf-8')
        print(f'\nReport written to {args.out}')


if __name__ == '__main__':
    main()
