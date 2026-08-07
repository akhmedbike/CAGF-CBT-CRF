"""Measure the MPS nondeterminism noise floor.

Runs the same configuration (full_model, fold_0 of the stratified CV, seed 42)
N times and reports the spread of the resulting metrics. Every run uses the
same data, same seed, same code -- any difference is pure hardware/framework
nondeterminism (MPS atomics, CUDA-style nondeterministic reductions).

This number is what goes in the Methods section: 'differences below X pp are
not provable on this hardware'. It is also the guard against over-claiming
small ablation effects (e.g. the +2.3pp gated-fusion point estimate sits near
this floor and must be reported as 'not isolable').

Usage:
    PYTHONPATH=. python scripts/measure_noise_floor.py --runs 3
"""
from __future__ import annotations
import argparse
import json
import statistics
from pathlib import Path

from cagf.data import build_vocabs, read_conllu
from cagf.folds import make_folds
from cagf.model import AblationConfig, ModelHParams
from cagf.train_loop import train_one_run


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--runs', type=int, default=3)
    ap.add_argument('--k', type=int, default=10)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--out', default='results_cv/noise_floor.json')
    ap.add_argument('--max-epochs', type=int, default=200)
    args = ap.parse_args()

    print(f'Noise floor measurement: {args.runs} identical runs of full_model, '
          f'fold {args.fold} of {args.k}-fold stratified CV, seed {args.seed}')
    sents = (read_conllu('data/gold_merged/gold_train.conllu') +
             read_conllu('data/gold_merged/gold_dev.conllu') +
             read_conllu('data/gold_merged/gold_test.conllu'))
    folds = make_folds(sents, k=args.k, seed=args.seed, strategy='stratified')
    fold = folds[args.fold]
    train_s = [sents[i] for i in fold.train_idx]
    dev_s = [sents[i] for i in fold.dev_idx]
    test_s = [sents[i] for i in fold.test_idx]
    vocabs = build_vocabs(train_s, min_word_freq=1, max_word_vocab=50000)
    print(f'fold {args.fold}: train={len(train_s)} dev={len(dev_s)} test={len(test_s)}')

    hp = ModelHParams()
    ablation = AblationConfig()  # full model
    results = []
    for run_idx in range(args.runs):
        print(f'\n=== run {run_idx + 1}/{args.runs} ===')
        r = train_one_run(
            train_sentences=train_s, dev_sentences=dev_s, test_sentences=test_s,
            vocabs=vocabs, ablation=ablation, hparams=hp, seed=args.seed,
            max_epochs=args.max_epochs, batch_size=32, learning_rate=3e-4,
            weight_decay=0.01, grad_clip_norm=5.0, early_stopping_patience=15,
            verbose=True)
        results.append({
            'run': run_idx,
            'lemma_accuracy': r.lemma['accuracy'],
            'lemma_f1': r.lemma['f1'],
            'upos_accuracy': r.upos['accuracy'],
            'upos_f1': r.upos['f1'],
            'grammeme_accuracy': r.grammeme['accuracy'],
            'grammeme_f1': r.grammeme['f1'],
            'best_epoch': r.best_epoch,
        })

    # summarize spread
    print('\n=== NOISE FLOOR SUMMARY ===')
    summary = {}
    for key in ['lemma_accuracy', 'lemma_f1', 'upos_accuracy', 'upos_f1',
                'grammeme_accuracy', 'grammeme_f1']:
        vals = [r[key] for r in results]
        mean = statistics.mean(vals)
        std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        spread = max(vals) - min(vals)
        summary[key] = {'mean': mean, 'std': std, 'min': min(vals),
                        'max': max(vals), 'spread_pp': spread * 100}
        print(f'  {key:<20} mean={mean*100:.3f}  std={std*100:.4f}pp  '
              f'spread={spread*100:.3f}pp  range=[{min(vals)*100:.3f}, {max(vals)*100:.3f}]')

    out = {'config': 'full_model', 'fold': args.fold, 'seed': args.seed,
           'n_runs': args.runs, 'runs': results, 'summary': summary}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\nWritten to {args.out}')
    # overall noise floor = max spread across metrics, in pp
    max_spread = max(s['spread_pp'] for s in summary.values())
    print(f'\nNOISE FLOOR (max spread across metrics): {max_spread:.3f} pp')
    print(f'Rule of thumb: any ablation difference below ~{max_spread:.1f}pp is not provable.')


if __name__ == '__main__':
    main()
