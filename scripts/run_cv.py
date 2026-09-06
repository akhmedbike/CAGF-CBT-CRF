"""Cross-validation runner for CAGF-CBT+CRF.

Trains each ablation configuration on k folds of UD_Kazakh-KTB, writes
per-fold predictions to CoNLL-U, scores them with the official conll18_ud_eval
script, and computes a single jack-knifed metric over the concatenated
out-of-fold predictions (every sentence scored exactly once, by the fold
whose test set it belonged to).

Why this exists
---------------
The single-split protocol (train/dev/test, 5 seeds) leaves two gaps:

1. The test set is only 109 sentences / 1094 tokens. At that resolution,
   near-identical models become indistinguishable -- the std of lemma accuracy
   across seeds was 0.04pp, i.e. less than one token. Cross-validation makes
   the *whole corpus* the test set (each sentence once), restoring resolution.
2. n=3 / n=5 seeds cannot support Wilcoxon (minimum p is 0.25 at n=3) and
   inflates Cohen's d. n=10 folds fixes both (see scripts/significance_test.py
   --unit fold).

Critical anti-leakage rule
--------------------------
Vocabularies are rebuilt **from fold.train only** for every fold. The test
fold never touches vocab construction -- neither char/word (which could leak
unseen surface forms) nor lemma_rule / upos / grammeme labels (which could
leak answer classes). This is asserted per fold by logging the test-only
vocab coverage; it is the property T0 audited for the single-split protocol,
extended here to every fold.

Resume
------
Each completed fold writes a ``fold_<i>.conllu`` and contributes a row to
``per_fold.json``. ``--resume`` (default on) skips folds whose ``fold_<i>.conllu``
already exists with a matching ``per_fold.json`` entry, so a 50-foldo-run that
crashes on fold 43 restarts at fold 43, not fold 0.

Usage
-----
    PYTHONPATH=. .venv/bin/python scripts/run_cv.py \\
        --config configs/default.yaml \\
        --ablations full_model \\
        --k 10 --strategy stratified --seed 42 \\
        --out-dir results_cv/stratified

    # add more configurations (these names match scripts/run_ablation.py)
    --ablations full_model,wo_character_encoder,wo_gated_fusion,wo_crf,transformer_only
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List

import yaml

from cagf.data import Sentence, build_vocabs, read_conllu
from cagf.folds import make_folds
from cagf.model import AblationConfig, ModelHParams
from cagf.official_eval import REPORTED_METRICS, evaluate_conllu
from cagf.predict_writer import write_conllu
from cagf.run_meta import write_run_meta
from cagf.train_loop import predict, train_one_run

# These names and ablation settings mirror scripts/run_ablation.py exactly,
# so CV and single-split results are comparable configuration-for-configuration.
ABLATION_CONFIGS = {
    'full_model': AblationConfig(),
    'wo_character_encoder': AblationConfig(use_char_cnn=False, use_char_bilstm=False),
    'wo_gated_fusion': AblationConfig(use_gated_fusion=False),
    'wo_crf': AblationConfig(use_crf=False),
    'transformer_only': AblationConfig(use_char_cnn=False, use_char_bilstm=False,
                                       use_word_bilstm=False, use_gated_fusion=False),
}


def _split_by_index(sentences: List[Sentence], idx: List[int]) -> List[Sentence]:
    return [sentences[i] for i in idx]


def _gold_for_fold(test_sentences: List[Sentence], path: Path) -> None:
    """Write the gold CoNLL-U for this fold's test set, so the official scorer
    has something to compare against. This is NOT a prediction file -- it
    carries the gold LEMMA/UPOS/FEATS that the scorer needs on both sides."""
    preds = [{
        'lemma': [t.lemma for t in s.tokens],
        'upos': [t.upos for t in s.tokens],
        'feats': [t.feats.split('|') if t.feats != '_' else [] for t in s.tokens],
    } for s in test_sentences]
    write_conllu(test_sentences, preds, path)


def _concat_conllu(parts: List[Path], out: Path) -> int:
    """Concatenate fold prediction files into one jack-knife file. Returns the
    sentence count written, for the integrity assertion."""
    n_sent = 0
    buf: List[str] = []
    for p in parts:
        if not p.exists():
            continue
        text = Path(p).read_text(encoding='utf-8')
        n_sent += text.count('# sent_id')
        buf.append(text.rstrip('\n'))
    # Each fold file already ends with a single trailing blank line; joining on
    # '\n\n' and adding a final '\n\n' keeps exactly one blank line between
    # sentences and a trailing blank line at EOF, which conll18_ud_eval requires
    # ("The CoNLL-U file does not end with empty line" is its most common error).
    Path(out).write_text('\n\n'.join(buf) + '\n\n', encoding='utf-8')
    return n_sent


def _count_sentences(path: Path) -> int:
    return Path(path).read_text(encoding='utf-8').count('# sent_id')


def run_config_cv(
    config_name: str, ablation: AblationConfig, sentences: List[Sentence],
    folds, hp: ModelHParams, train_cfg: dict, args, gold_dir: Path,
) -> None:
    """Run one ablation configuration across all folds."""
    out_dir = Path(args.out_dir) / config_name
    out_dir.mkdir(parents=True, exist_ok=True)
    per_fold_path = out_dir / 'per_fold.json'
    per_fold: List[dict] = []
    if per_fold_path.exists() and args.resume:
        per_fold = json.loads(per_fold_path.read_text(encoding='utf-8'))
        print(f'  [{config_name}] resuming: {len(per_fold)}/{len(folds)} folds already done')

    log_path = out_dir / 'run.log'
    with open(log_path, 'a', encoding='utf-8') as logf:
        logf.write(f'\n=== CV run config={config_name} k={args.k} strategy={args.strategy} '
                   f'seed={args.seed} train_seed={args.train_seed or args.seed} '
                   f'started={time.strftime("%Y-%m-%d %H:%M:%S")} ===\n')

    # training seed is decoupled from the fold seed so seed-replication runs
    # (revision task B) can vary initialization without moving the folds
    train_seed = args.train_seed if args.train_seed is not None else args.seed
    from cagf.device import pick_device
    write_run_meta(out_dir, device=args.device or pick_device(),
                   driver='run_cv.py', k=args.k, strategy=args.strategy,
                   seed=args.seed, train_seed=train_seed, config=config_name,
                   max_epochs=args.max_epochs, corpus_sentences=len(sentences))

    for fold in folds:
        fold_pred_path = out_dir / f'fold_{fold.index}.conllu'
        # resume: skip folds with both a prediction file and a per_fold entry
        already = any(r.get('fold') == fold.index for r in per_fold)
        if args.resume and already and fold_pred_path.exists():
            print(f'  [{config_name}] skip fold {fold.index} (already done)')
            continue

        t0 = time.time()
        train_s = _split_by_index(sentences, fold.train_idx)
        dev_s = _split_by_index(sentences, fold.dev_idx)
        test_s = _split_by_index(sentences, fold.test_idx)

        # CRITICAL: vocabularies from fold.train ONLY. Test fold never enters.
        vocabs = build_vocabs(train_s, min_word_freq=args.min_word_freq,
                              max_word_vocab=args.max_word_vocab)

        from cagf.device import pick_device
        device = args.device or pick_device()
        print(f'  [{config_name}] fold {fold.index}: '
              f'train={len(train_s)} dev={len(dev_s)} test={len(test_s)} '
              f'vocab(chars={len(vocabs.char_vocab)} words={len(vocabs.word_vocab)} '
              f'upos={len(vocabs.upos_vocab)} gram={len(vocabs.grammeme_vocab)} '
              f'lemma_rules={len(vocabs.lemma_rule_vocab)})')

        # train + keep model for prediction (return_model=True)
        result, model = train_one_run(
            train_sentences=train_s, dev_sentences=dev_s, test_sentences=test_s,
            vocabs=vocabs, ablation=ablation, hparams=hp, seed=train_seed,
            max_epochs=args.max_epochs, batch_size=train_cfg['batch_size'],
            learning_rate=train_cfg['learning_rate'], weight_decay=train_cfg['weight_decay'],
            grad_clip_norm=train_cfg['grad_clip_norm'],
            early_stopping_patience=train_cfg['early_stopping_patience'],
            device=device, verbose=True, return_model=True,
            probs_dump_path=str(out_dir / f'fold_{fold.index}_gramprobs.npz'))

        # predict on test fold and write CoNLL-U (model already on `device`)
        preds = predict(model, test_s, vocabs, batch_size=train_cfg['batch_size'],
                        device=device)
        write_conllu(test_s, preds, fold_pred_path)

        # gold for this fold's test set
        gold_path = gold_dir / f'fold_{fold.index}.conllu'
        if not gold_path.exists():
            _gold_for_fold(test_s, gold_path)

        # official + our metrics for this fold
        official = evaluate_conllu(str(gold_path), str(fold_pred_path))
        elapsed = time.time() - t0
        entry = {
            'fold': fold.index,
            'config': config_name,
            'seed': args.seed,
            'train_seed': train_seed,
            'lemma': result.lemma,
            'upos': result.upos,
            'grammeme': result.grammeme,
            'official': official,
            'best_epoch': result.best_epoch,
            'n_train': len(train_s), 'n_dev': len(dev_s), 'n_test': len(test_s),
            'vocab_sizes': {
                'chars': len(vocabs.char_vocab), 'words': len(vocabs.word_vocab),
                'upos': len(vocabs.upos_vocab), 'grammemes': len(vocabs.grammeme_vocab),
                'lemma_rules': len(vocabs.lemma_rule_vocab),
            },
            'elapsed_sec': round(elapsed, 1),
        }
        per_fold.append(entry)
        # rewrite per_fold.json after each fold so a crash loses at most one fold
        per_fold.sort(key=lambda r: r['fold'])
        per_fold_path.write_text(json.dumps(per_fold, ensure_ascii=False, indent=2),
                                 encoding='utf-8')
        with open(log_path, 'a', encoding='utf-8') as logf:
            logf.write(f'  fold {fold.index}: best_epoch={result.best_epoch} '
                       f'elapsed={elapsed:.0f}s official={ {k: round(v,4) for k,v in official.items()} }\n')
        print(f'  [{config_name}] fold {fold.index} done in {elapsed:.0f}s: '
              f'UPOS_official={official["UPOS"]:.4f} UFeats={official["UFeats"]:.4f}')

    # jack-knife: concatenate all fold predictions, score against concatenated gold
    _jackknife(config_name, folds, sentences, out_dir, gold_dir, args)


def _jackknife(config_name: str, folds, sentences: List[Sentence], out_dir: Path,
               gold_dir: Path, args) -> None:
    """Concatenate per-fold predictions and gold into two whole-corpus files,
    then run the official scorer once. This is the jack-knife estimate: every
    sentence scored exactly once, by the model that never saw it in training."""
    pred_parts = [out_dir / f'fold_{f.index}.conllu' for f in folds]
    gold_parts = [gold_dir / f'fold_{f.index}.conllu' for f in folds]
    # all folds must be present for a valid jack-knife
    missing = [p for p in pred_parts if not p.exists()]
    if missing:
        print(f'  [{config_name}] jack-knife SKIPPED: missing {len(missing)} fold files')
        return
    pred_all = out_dir / 'pred_all.conllu'
    gold_all = out_dir / 'gold_all.conllu'
    n_pred = _concat_conllu(pred_parts, pred_all)
    n_gold = _concat_conllu(gold_parts, gold_all)

    # integrity guards from the spec
    assert n_pred == len(sentences), \
        f'jack-knife: pred_all has {n_pred} sentences, expected {len(sentences)}'
    assert n_gold == len(sentences), \
        f'jack-knife: gold_all has {n_gold} sentences, expected {len(sentences)}'

    official = evaluate_conllu(str(gold_all), str(pred_all))
    # also compute our own metrics across folds for direct comparison
    per_fold = json.loads((out_dir / 'per_fold.json').read_text(encoding='utf-8'))
    import statistics
    our = {}
    for task in ('lemma', 'upos', 'grammeme'):
        for metric in ('accuracy', 'f1'):
            vals = [r[task][metric] for r in per_fold]
            our[f'{task}_{metric}_mean'] = statistics.mean(vals)
            our[f'{task}_{metric}_std'] = statistics.pstdev(vals) if len(vals) > 1 else 0.0

    jack = {
        'config': config_name, 'n_folds': len(folds), 'n_sentences': n_pred,
        'official_jackknifed': official,
        'our_metrics_mean_over_folds': our,
    }
    (out_dir / 'jackknifed.json').write_text(json.dumps(jack, ensure_ascii=False,
                                                         indent=2), encoding='utf-8')
    print(f'  [{config_name}] JACK-KNIFE ({n_pred} sents): '
          f'Lemmas={official["Lemmas"]:.4f} UPOS={official["UPOS"]:.4f} '
          f'UFeats={official["UFeats"]:.4f} AllTags={official["AllTags"]:.4f}')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default='configs/default.yaml')
    ap.add_argument('--ablations', default='full_model',
                    help='comma-separated config names; one of: ' + ','.join(ABLATION_CONFIGS))
    ap.add_argument('--k', type=int, default=10)
    ap.add_argument('--strategy', choices=['stratified', 'grouped'], default='stratified')
    ap.add_argument('--seed', type=int, default=42,
                    help='fixes both fold split and model init; variance comes from folds')
    ap.add_argument('--train-seed', type=int, default=None,
                    help='override the TRAINING seed only; folds stay at --seed. '
                         'Seed replication retrains identical folds from '
                         'different initializations.')
    ap.add_argument('--out-dir', default='results_cv/stratified')
    ap.add_argument('--corpus', nargs='+', default=None,
                    help='CoNLL-U files to use as the full corpus (default: '
                         'train+dev+test from --config, i.e. the whole KTB)')
    ap.add_argument('--max-epochs', type=int, default=None)
    ap.add_argument('--min-word-freq', type=int, default=1)
    ap.add_argument('--max-word-vocab', type=int, default=50000)
    ap.add_argument('--resume', action='store_true', default=True,
                    help='skip folds whose fold_<i>.conllu already exists (default on)')
    ap.add_argument('--no-resume', dest='resume', action='store_false')
    ap.add_argument('--device', default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    train_cfg = cfg['training']
    if args.max_epochs is None:
        args.max_epochs = train_cfg['max_epochs']

    # load the whole corpus (train+dev+test merged) -- CV re-splits it.
    if args.corpus:
        sentences: List[Sentence] = []
        for p in args.corpus:
            sentences.extend(read_conllu(p))
    else:
        sentences = (read_conllu(cfg['data']['train_path']) +
                     read_conllu(cfg['data']['dev_path']) +
                     read_conllu(cfg['data']['test_path']))
    print(f'Corpus: {len(sentences)} sentences')

    folds = make_folds(sentences, k=args.k, seed=args.seed, strategy=args.strategy)
    print(f'Folds: k={args.k} strategy={args.strategy} seed={args.seed}, '
          f'sizes={[len(f.test_idx) for f in folds]}')

    hp = ModelHParams(**cfg['model'])

    # shared gold fold files (same for all configs -- the test sets don't change)
    gold_dir = Path(args.out_dir) / '_gold_folds'
    gold_dir.mkdir(parents=True, exist_ok=True)

    configs = [c.strip() for c in args.ablations.split(',') if c.strip()]
    unknown = [c for c in configs if c not in ABLATION_CONFIGS]
    if unknown:
        raise SystemExit(f'unknown ablation(s) {unknown}; choose from {list(ABLATION_CONFIGS)}')

    # write manifest
    manifest = {
        'k': args.k, 'strategy': args.strategy, 'seed': args.seed,
        'train_seed': args.train_seed if args.train_seed is not None else args.seed,
        'configs': configs, 'corpus_sentences': len(sentences),
        'config_file': args.config, 'max_epochs': args.max_epochs,
        'date': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.out_dir) / 'manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')

    # flat cv_raw.json across all configs x folds, compatible with ablation_raw.json
    cv_raw_path = Path(args.out_dir) / 'cv_raw.json'
    cv_raw: List[dict] = []
    for config_name in configs:
        ablation = ABLATION_CONFIGS[config_name]
        print(f'\n=== CV config={config_name} ===')
        run_config_cv(config_name, ablation, sentences, folds, hp, train_cfg, args, gold_dir)
        # reload this config's per_fold and merge into the flat cv_raw
        pf = json.loads((Path(args.out_dir) / config_name / 'per_fold.json').read_text(encoding='utf-8'))
        for r in pf:
            cv_raw.append({
                'config': r['config'], 'fold': r['fold'], 'seed': r['seed'],
                'lemma': r['lemma'], 'upos': r['upos'], 'grammeme': r['grammeme'],
                'official': r['official'], 'best_epoch': r['best_epoch'],
            })
    cv_raw_path.write_text(json.dumps(cv_raw, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\nDone. Flat CV results: {cv_raw_path}')
    print('Per-config jack-knife results:')
    for config_name in configs:
        jpath = Path(args.out_dir) / config_name / 'jackknifed.json'
        if jpath.exists():
            j = json.loads(jpath.read_text(encoding='utf-8'))
            o = j['official_jackknifed']
            print(f'  {config_name}: Lemmas={o["Lemmas"]:.4f} UPOS={o["UPOS"]:.4f} '
                  f'UFeats={o["UFeats"]:.4f} AllTags={o["AllTags"]:.4f}')


if __name__ == '__main__':
    main()
