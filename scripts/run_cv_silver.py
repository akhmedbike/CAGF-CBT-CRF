"""Cross-validation for the silver-transfer experiment (T7).

This is the CV analogue of scripts/run_silver_ablation.py: it re-runs the
silver-pretrain -> gold-finetune experiment under 10-fold cross-validation,
so the headline silver-transfer claim (+10.1pp lemma accuracy under the
single-split protocol) can be tested at n=10 folds with real error bars and
Holm-corrected significance.

Why a separate runner
---------------------
The silver protocol has a step the plain ablation CV does not: pretraining on
the silver corpus before fine-tuning on the gold fold. The critical
anti-leakage invariant (audited for single-split in docs/leakage_audit.md, and
required by the reviewer protocol) is:

    vocabularies are built from union(silver, fold.train) -- NEVER fold.test.

The silver corpus is large and shared across folds, so the temptation to build
the char/word vocabulary once on union(silver, all_gold) is real and must be
resisted: that would leak test-fold surface forms into the vocabulary. This
script rebuilds vocabularies per fold, anchoring labels to fold.train and
widening only char/word from silver.

Configurations (mirroring run_silver_ablation.py):
  * ktb_only                          -- train from scratch on fold.train only
  * silver_filtered_pretrain_ktb_finetune   -- pretrain on filtered silver, finetune on fold.train
  * silver_unfiltered_pretrain_ktb_finetune -- same, unfiltered silver (filter-ablation)

All evaluated on fold.test, scored with the official conll18_ud_eval, and
jack-knifed across all 1078 sentences like run_cv.py.

Usage
-----
    PYTHONPATH=. .venv/bin/python scripts/run_cv_silver.py \\
        --silver-filtered data/silver/silver_subset_210k.conllu \\
        --k 10 --strategy stratified --seed 42 \\
        --out-dir results_cv_silver/stratified
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import List

import yaml

from cagf.data import Sentence, read_conllu
from cagf.folds import make_folds
from cagf.model import AblationConfig, ModelHParams
from cagf.official_eval import REPORTED_METRICS, evaluate_conllu
from cagf.predict_writer import write_conllu
from cagf.run_meta import write_run_meta
from cagf.train_loop import predict, train_one_run
from scripts.pretrain_finetune import build_shared_vocabs
from scripts.run_cv import _concat_conllu, _gold_for_fold

KTB_ONLY = "ktb_only"
SILVER_FILTERED = "silver_filtered_pretrain_ktb_finetune"
SILVER_UNFILTERED = "silver_unfiltered_pretrain_ktb_finetune"


def _split_by_index(sentences: List[Sentence], idx: List[int]) -> List[Sentence]:
    return [sentences[i] for i in idx]


def run_ktb_only_fold(train_s, dev_s, test_s, hp, ablation, seed, train_cfg, device,
                      probs_dump_path=None):
    """Train from scratch on fold.train only. Vocab from fold.train only."""
    from cagf.data import build_vocabs
    vocabs = build_vocabs(train_s, min_word_freq=1, max_word_vocab=50000)
    result, model = train_one_run(
        train_sentences=train_s, dev_sentences=dev_s, test_sentences=test_s,
        vocabs=vocabs, ablation=ablation, hparams=hp, seed=seed,
        max_epochs=train_cfg['max_epochs'], batch_size=train_cfg['batch_size'],
        learning_rate=train_cfg['learning_rate'], weight_decay=train_cfg['weight_decay'],
        grad_clip_norm=train_cfg['grad_clip_norm'],
        early_stopping_patience=train_cfg['early_stopping_patience'],
        device=device, verbose=True, return_model=True,
        probs_dump_path=probs_dump_path)
    return result, model, vocabs


def run_silver_pretrain_finetune_fold(config_name, silver, train_s, dev_s, test_s,
                                      hp, ablation, seed, args, device,
                                      probs_dump_path=None):
    """Pretrain on silver (early-stop on gold dev), finetune on fold.train.

    CRITICAL: vocab from union(silver, fold.train) -- fold.test never enters.
    """
    vocabs = build_shared_vocabs(train_s, silver,
                                 min_word_freq=args.min_word_freq,
                                 max_word_vocab=args.max_word_vocab)
    import torch
    ckpt_path = Path(args.out_dir) / f"_tmp_{config_name}_fold_pretrained.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    # pretrain on silver, early-stop on gold dev (the fold's dev)
    train_one_run(
        train_sentences=silver, dev_sentences=dev_s, test_sentences=test_s,
        vocabs=vocabs, ablation=ablation, hparams=hp, seed=seed,
        max_epochs=args.pretrain_epochs, batch_size=args.batch_size,
        learning_rate=args.pretrain_lr, weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm, early_stopping_patience=args.pretrain_patience,
        device=device, verbose=True, save_checkpoint_path=str(ckpt_path))
    pretrained_state = torch.load(ckpt_path, map_location="cpu", weights_only=False)["model_state_dict"]
    ckpt_path.unlink(missing_ok=True)  # cleanup

    # finetune on fold.train, evaluate on fold.test
    result, model = train_one_run(
        train_sentences=train_s, dev_sentences=dev_s, test_sentences=test_s,
        vocabs=vocabs, ablation=ablation, hparams=hp, seed=seed,
        max_epochs=args.finetune_epochs, batch_size=args.batch_size,
        learning_rate=args.finetune_lr, weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm, early_stopping_patience=args.finetune_patience,
        device=device, verbose=True, init_state=pretrained_state, return_model=True,
        probs_dump_path=probs_dump_path)
    return replace(result, config_name=config_name), model, vocabs


def run_config_cv(config_name, run_fn, silver, sentences, folds, hp, train_cfg, args, gold_dir):
    out_dir = Path(args.out_dir) / config_name
    out_dir.mkdir(parents=True, exist_ok=True)
    per_fold_path = out_dir / 'per_fold.json'
    per_fold: List[dict] = []
    if per_fold_path.exists() and args.resume:
        per_fold = json.loads(per_fold_path.read_text(encoding='utf-8'))
        print(f'  [{config_name}] resuming: {len(per_fold)}/{len(folds)} folds done')

    from cagf.device import pick_device
    device = args.device or pick_device()

    log_path = out_dir / 'run.log'
    with open(log_path, 'a', encoding='utf-8') as logf:
        logf.write(f'\n=== CV-silver config={config_name} k={args.k} strategy={args.strategy} '
                   f'seed={args.seed} train_seed={args.train_seed or args.seed} '
                   f'started={time.strftime("%Y-%m-%d %H:%M:%S")} ===\n')

    # training seed decoupled from fold seed (revision task B: seed replication
    # must retrain the SAME folds from a different initialization)
    train_seed = args.train_seed if args.train_seed is not None else args.seed
    write_run_meta(out_dir, device=device, driver='run_cv_silver.py',
                   k=args.k, strategy=args.strategy, seed=args.seed,
                   train_seed=train_seed, config=config_name,
                   pretrain_epochs=args.pretrain_epochs,
                   corpus_sentences=len(sentences))

    for fold in folds:
        fold_pred_path = out_dir / f'fold_{fold.index}.conllu'
        already = any(r.get('fold') == fold.index for r in per_fold)
        if args.resume and already and fold_pred_path.exists():
            print(f'  [{config_name}] skip fold {fold.index} (done)')
            continue

        t0 = time.time()
        train_s = _split_by_index(sentences, fold.train_idx)
        dev_s = _split_by_index(sentences, fold.dev_idx)
        test_s = _split_by_index(sentences, fold.test_idx)
        print(f'  [{config_name}] fold {fold.index}: train={len(train_s)} dev={len(dev_s)} test={len(test_s)}')

        ablation = AblationConfig()  # full model; data is what changes
        probs_dump = str(out_dir / f'fold_{fold.index}_gramprobs.npz')
        if config_name == KTB_ONLY:
            result, model, vocabs = run_ktb_only_fold(train_s, dev_s, test_s, hp, ablation,
                                                      train_seed, train_cfg, device,
                                                      probs_dump_path=probs_dump)
        else:
            result, model, vocabs = run_fn(config_name, silver, train_s, dev_s, test_s,
                                           hp, ablation, train_seed, args, device,
                                           probs_dump_path=probs_dump)

        preds = predict(model, test_s, vocabs, batch_size=train_cfg['batch_size'], device=device)
        write_conllu(test_s, preds, fold_pred_path)
        gold_path = gold_dir / f'fold_{fold.index}.conllu'
        if not gold_path.exists():
            _gold_for_fold(test_s, gold_path)

        official = evaluate_conllu(str(gold_path), str(fold_pred_path))
        elapsed = time.time() - t0
        entry = {
            'fold': fold.index, 'config': config_name, 'seed': args.seed,
            'train_seed': train_seed,
            'lemma': result.lemma, 'upos': result.upos, 'grammeme': result.grammeme,
            'official': official, 'best_epoch': result.best_epoch,
            'n_train': len(train_s), 'n_dev': len(dev_s), 'n_test': len(test_s),
            'elapsed_sec': round(elapsed, 1),
        }
        per_fold.append(entry)
        per_fold.sort(key=lambda r: r['fold'])
        per_fold_path.write_text(json.dumps(per_fold, ensure_ascii=False, indent=2), encoding='utf-8')
        with open(log_path, 'a', encoding='utf-8') as logf:
            logf.write(f'  fold {fold.index}: best_epoch={result.best_epoch} elapsed={elapsed:.0f}s '
                       f'official={ {k: round(v,4) for k,v in official.items()} }\n')
        print(f'  [{config_name}] fold {fold.index} done in {elapsed:.0f}s: '
              f'UPOS={official["UPOS"]:.4f} UFeats={official["UFeats"]:.4f} Lemmas={official["Lemmas"]:.4f}')

    # jack-knife
    pred_parts = [out_dir / f'fold_{f.index}.conllu' for f in folds]
    gold_parts = [gold_dir / f'fold_{f.index}.conllu' for f in folds]
    missing = [p for p in pred_parts if not p.exists()]
    if missing:
        print(f'  [{config_name}] jack-knife SKIPPED: missing {len(missing)} fold files')
        return
    pred_all = out_dir / 'pred_all.conllu'
    gold_all = out_dir / 'gold_all.conllu'
    n_pred = _concat_conllu(pred_parts, pred_all)
    n_gold = _concat_conllu(gold_parts, gold_all)
    assert n_pred == len(sentences) and n_gold == len(sentences)
    official = evaluate_conllu(str(gold_all), str(pred_all))
    import statistics
    pf = json.loads(per_fold_path.read_text(encoding='utf-8'))
    our = {}
    for task in ('lemma', 'upos', 'grammeme'):
        for metric in ('accuracy', 'f1'):
            vals = [r[task][metric] for r in pf]
            our[f'{task}_{metric}_mean'] = statistics.mean(vals)
            our[f'{task}_{metric}_std'] = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    jack = {'config': config_name, 'n_folds': len(folds), 'n_sentences': n_pred,
            'official_jackknifed': official, 'our_metrics_mean_over_folds': our}
    (out_dir / 'jackknifed.json').write_text(json.dumps(jack, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'  [{config_name}] JACK-KNIFE ({n_pred} sents): '
          f'Lemmas={official["Lemmas"]:.4f} UPOS={official["UPOS"]:.4f} '
          f'UFeats={official["UFeats"]:.4f} AllTags={official["AllTags"]:.4f}')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default='configs/default.yaml')
    ap.add_argument('--silver-filtered', default='data/silver/silver_subset_210k.conllu')
    ap.add_argument('--silver-unfiltered', default=None,
                    help='optional unfiltered silver for filter-ablation config')
    ap.add_argument('--k', type=int, default=10)
    ap.add_argument('--strategy', choices=['stratified', 'grouped'], default='stratified')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--train-seed', type=int, default=None,
                    help='override the TRAINING seed only; folds stay at --seed '
                         '(seed replication retrains identical folds)')
    ap.add_argument('--out-dir', default='results_cv_silver/stratified')
    ap.add_argument('--max-epochs', type=int, default=None)
    ap.add_argument('--min-word-freq', type=int, default=1)
    ap.add_argument('--max-word-vocab', type=int, default=50000)
    ap.add_argument('--pretrain-epochs', type=int, default=30)
    ap.add_argument('--pretrain-lr', type=float, default=3e-4)
    ap.add_argument('--pretrain-patience', type=int, default=5)
    ap.add_argument('--finetune-lr', type=float, default=1e-4)
    ap.add_argument('--finetune-patience', type=int, default=15)
    ap.add_argument('--resume', action='store_true', default=True)
    ap.add_argument('--no-resume', dest='resume', action='store_false')
    ap.add_argument('--configs', default='ktb_only,silver_filtered_pretrain_ktb_finetune',
                    help='comma-separated subset of: ktb_only, '
                         'silver_filtered_pretrain_ktb_finetune, '
                         'silver_unfiltered_pretrain_ktb_finetune')
    ap.add_argument('--device', default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    train_cfg = cfg['training']
    if args.max_epochs is None:
        args.max_epochs = train_cfg['max_epochs']
    args.finetune_epochs = args.max_epochs
    args.batch_size = train_cfg['batch_size']
    args.weight_decay = train_cfg['weight_decay']
    args.grad_clip_norm = train_cfg['grad_clip_norm']

    sentences = (read_conllu(cfg['data']['train_path']) +
                 read_conllu(cfg['data']['dev_path']) +
                 read_conllu(cfg['data']['test_path']))
    print(f'Corpus: {len(sentences)} sentences')

    folds = make_folds(sentences, k=args.k, seed=args.seed, strategy=args.strategy)
    print(f'Folds: k={args.k} strategy={args.strategy} seed={args.seed}, '
          f'sizes={[len(f.test_idx) for f in folds]}')

    hp = ModelHParams(**cfg['model'])
    gold_dir = Path(args.out_dir) / '_gold_folds'
    gold_dir.mkdir(parents=True, exist_ok=True)

    requested = [c.strip() for c in args.configs.split(',') if c.strip()]
    silver_filtered = read_conllu(args.silver_filtered) if args.silver_filtered else None
    silver_unfiltered = read_conllu(args.silver_unfiltered) if args.silver_unfiltered else None

    manifest = {'k': args.k, 'strategy': args.strategy, 'seed': args.seed,
                'train_seed': args.train_seed if args.train_seed is not None else args.seed,
                'configs': requested, 'corpus_sentences': len(sentences),
                'silver_filtered_sentences': len(silver_filtered) if silver_filtered else 0,
                'silver_unfiltered_sentences': len(silver_unfiltered) if silver_unfiltered else 0,
                'date': time.strftime('%Y-%m-%d %H:%M:%S')}
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.out_dir) / 'manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')

    cv_raw: List[dict] = []
    for config_name in requested:
        print(f'\n=== CV-silver config={config_name} ===')
        if config_name == KTB_ONLY:
            run_config_cv(config_name, None, None, sentences, folds, hp, train_cfg, args, gold_dir)
        elif config_name == SILVER_FILTERED:
            run_config_cv(config_name, run_silver_pretrain_finetune_fold,
                          silver_filtered, sentences, folds, hp, train_cfg, args, gold_dir)
        elif config_name == SILVER_UNFILTERED:
            run_config_cv(config_name, run_silver_pretrain_finetune_fold,
                          silver_unfiltered, sentences, folds, hp, train_cfg, args, gold_dir)
        else:
            raise SystemExit(f'unknown config {config_name}')
        pf = json.loads((Path(args.out_dir) / config_name / 'per_fold.json').read_text(encoding='utf-8'))
        for r in pf:
            cv_raw.append({'config': r['config'], 'fold': r['fold'], 'seed': r['seed'],
                           'lemma': r['lemma'], 'upos': r['upos'], 'grammeme': r['grammeme'],
                           'official': r['official'], 'best_epoch': r['best_epoch']})
    cv_raw_path = Path(args.out_dir) / 'cv_raw.json'
    cv_raw_path.write_text(json.dumps(cv_raw, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\nDone. Flat CV-silver results: {cv_raw_path}')
    for config_name in requested:
        jpath = Path(args.out_dir) / config_name / 'jackknifed.json'
        if jpath.exists():
            j = json.loads(jpath.read_text(encoding='utf-8'))
            o = j['official_jackknifed']
            print(f'  {config_name}: Lemmas={o["Lemmas"]:.4f} UPOS={o["UPOS"]:.4f} '
                  f'UFeats={o["UFeats"]:.4f} AllTags={o["AllTags"]:.4f}')


if __name__ == '__main__':
    main()
