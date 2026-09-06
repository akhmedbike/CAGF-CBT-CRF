"""Cross-validation for the KazRoBERTa morphological tagger (HFMorphModel).

This is the HF analogue of :mod:`scripts.run_cv_silver`: it trains the
HuggingFace-backed :class:`HFMorphModel` (KazRoBERTa encoder + the same three
task heads + optional CRF as CAGF) under k-fold cross-validation on the gold
UD_Kazakh-KTB corpus, and produces the same per-fold / jack-knifed / flat-CV
output artifacts so the two model families can be reported side by side.

Configurations
--------------
* ``gold_only`` (K4-A): train KazRoBERTa + heads from scratch on fold.train.
  Vocabulary built from fold.train ONLY (the same anti-leakage invariant as
  :mod:`scripts.run_cv` / :mod:`scripts.run_cv_silver`).
* ``silver_finetune`` (K4-B, the O4 silver-encoder-reuse optimisation): train
  the encoder on the silver corpus ONCE (early-stop on gold dev), save the
  encoder weights, then for each fold load those encoder weights and
  fine-tune with fold-fresh heads on fold.train. Heads are initialised from
  the fold.train vocab (NOT the silver vocab), so per-fold label spaces never
  leak across folds and never inherit silver-only labels.
* ``frozen`` (optional): same as ``gold_only`` but with the KazRoBERTa encoder
  frozen (only the projection + heads train). Useful as a parameter-efficiency
  ablation.

Anti-leakage
------------
Every fold's label vocabularies (upos / grammeme / lemma_rule) are built from
fold.train ONLY. The silver corpus is external: it trains ONLY the encoder
weights and never enters a gold fold's test set, dev set, or vocabulary.
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
from cagf.hf_train_loop import predict_hf, train_hf_one_run
from cagf.official_eval import REPORTED_METRICS, evaluate_conllu
from cagf.predict_writer import write_conllu
from cagf.run_meta import write_run_meta
from scripts.run_cv import _concat_conllu, _gold_for_fold

GOLD_ONLY = "gold_only"
SILVER_FINETUNE = "silver_finetune"
FROZEN = "frozen"

# The exact checkpoint the HF encoder module loads by default; overridable via
# CLI for experiments with a different KazRoBERTa revision.
DEFAULT_MODEL_NAME = "kz-transformers/kaz-roberta-conversational"
DEFAULT_REVISION = "43077c2fd0a163487ed468b5ec3b8750686a5888"


def _split_by_index(sentences: List[Sentence], idx: List[int]) -> List[Sentence]:
    return [sentences[i] for i in idx]


def _encoder_state_path(out_dir: str) -> Path:
    """Where the silver-pretrained encoder weights live (one for the whole run)."""
    return Path(out_dir) / "checkpoints" / "silver_encoder_kazroberta.pt"


def _save_interface_checkpoint(model, vocabs, result, args, config_name, fold):
    """Export a fold's trained HFMorphModel + vocab for the web demo.

    Writes a checkpoint in the exact schema
    :func:`cagf.hf_inference.load_hf_checkpoint` consumes (same keys
    :func:`train_hf_one_run` writes via ``save_checkpoint_path``, plus the
    corpus-size annotations the demo's meta panel reads). This lets a single CV
    run produce a frontend-loadable KazRoBERTa model without a separate
    ``scripts/train_hf_for_interface.py`` invocation.
    """
    import torch
    out_path = Path(args.save_interface_checkpoint)
    vocabs_path = Path(args.save_interface_vocabs) if args.save_interface_vocabs else out_path.with_suffix('.vocabs.json')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vocabs_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_name': args.model_name, 'revision': args.revision,
        'num_upos': len(vocabs.upos_vocab),
        'num_grammemes': len(vocabs.grammeme_vocab),
        'num_lemma_rules': len(vocabs.lemma_rule_vocab),
        'use_crf': True,
        'freeze_encoder': (config_name == FROZEN),
        'hidden_dim': 256,
        'test_metrics': {'lemma': result.lemma, 'upos': result.upos, 'grammeme': result.grammeme},
        'seed': args.seed, 'best_epoch': result.best_epoch,
        'train_sentences': len(fold.train_idx),
        'config': config_name, 'fold': fold.index,
    }, out_path)
    vocabs.save(vocabs_path)
    print(f'  [{config_name}] exported interface checkpoint -> {out_path} '
          f'(vocabs -> {vocabs_path})')


def _train_silver_encoder_once(silver: List[Sentence], dev_sentences: List[Sentence],
                               test_sentences: List[Sentence], args, device: str,
                               train_seed: int) -> Path:
    """Pretrain the KazRoBERTa encoder on the silver corpus exactly once.

    The silver corpus is external to the gold folds, so its vocab is built from
    silver ONLY (the encoder is vocabulary-agnostic -- it consumes raw surface
    forms -- but the heads need a label vocab, and using silver's label vocab
    here is correct because the resulting head weights are discarded: only the
    ``model.encoder.*`` tensors are saved and reloaded per fold). Early-stops on
    a gold dev set so the encoder stops when transfer to the gold domain peaks.

    Returns the path the encoder state was saved to.
    """
    ckpt_path = _encoder_state_path(args.out_dir)
    if ckpt_path.exists():
        print(f'  [silver_finetune] reusing cached silver encoder at {ckpt_path}')
        return ckpt_path
    print(f'  [silver_finetune] pretraining encoder on silver ({len(silver)} sentences)...')
    vocabs = build_vocabs(silver, min_word_freq=1, max_word_vocab=50000)
    # Train on silver; the test_sentences arg is only used for the final
    # evaluation logged in the RunResult, which we ignore (we only keep the
    # encoder weights). We DO use a gold dev set for early stopping so we stop
    # when the encoder transfers best to the target domain.
    train_hf_one_run(
        train_sentences=silver, dev_sentences=dev_sentences,
        test_sentences=test_sentences, vocabs=vocabs,
        model_name=args.model_name, revision=args.revision, seed=train_seed,
        max_epochs=args.silver_epochs, batch_size=args.batch_size,
        learning_rate=args.silver_lr, weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        early_stopping_patience=args.silver_patience,
        use_crf=True, freeze_encoder=False,
        device=device, verbose=True, return_model=False,
        save_checkpoint_path=str(ckpt_path),
        encoder_cache_dir=args.encoder_cache_dir,
        config_name='silver_pretrain')
    # Reload and keep ONLY the encoder tensors (drop the silver-vocab heads).
    import torch
    full_state = torch.load(ckpt_path, map_location="cpu", weights_only=False)["model_state_dict"]
    encoder_state = {k: v for k, v in full_state.items() if k.startswith("encoder.")}
    torch.save({"encoder_state_dict": encoder_state, "model_name": args.model_name,
                "revision": args.revision,
                "silver_sentences": len(silver),
                "date": time.strftime("%Y-%m-%d %H:%M:%S")}, ckpt_path)
    print(f'  [silver_finetune] saved {len(encoder_state)} encoder tensors to {ckpt_path}')
    return ckpt_path


def run_gold_only_fold(config_name, train_s, dev_s, test_s, seed, args, device,
                       freeze_encoder, probs_dump_path=None):
    """Train KazRoBERTa + heads from scratch on fold.train.

    CRITICAL: vocab from fold.train ONLY. Test fold never enters the vocab.
    """
    vocabs = build_vocabs(train_s, min_word_freq=1, max_word_vocab=50000)
    result, model = train_hf_one_run(
        train_sentences=train_s, dev_sentences=dev_s, test_sentences=test_s,
        vocabs=vocabs, model_name=args.model_name, revision=args.revision,
        seed=seed, max_epochs=args.max_epochs, batch_size=args.batch_size,
        learning_rate=args.lr, weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        early_stopping_patience=args.early_stopping_patience,
        use_crf=True, freeze_encoder=freeze_encoder,
        device=device, verbose=True, return_model=True,
        encoder_cache_dir=args.encoder_cache_dir, config_name=config_name,
        probs_dump_path=probs_dump_path)
    return result, model, vocabs


def run_silver_finetune_fold(config_name, encoder_ckpt, train_s, dev_s, test_s,
                             seed, args, device, probs_dump_path=None):
    """Load silver-pretrained encoder, init heads fresh from fold.train, finetune.

    CRITICAL: head vocabularies from fold.train ONLY (silver never enters the
    fold vocab, never enters fold.test). Only ``model.encoder.*`` tensors are
    loaded from the silver checkpoint; all head / CRF tensors are initialised
    fresh against the fold.train label vocab.
    """
    import torch
    vocabs = build_vocabs(train_s, min_word_freq=1, max_word_vocab=50000)
    encoder_state = torch.load(encoder_ckpt, map_location="cpu",
                               weights_only=False)["encoder_state_dict"]
    result, model = train_hf_one_run(
        train_sentences=train_s, dev_sentences=dev_s, test_sentences=test_s,
        vocabs=vocabs, model_name=args.model_name, revision=args.revision,
        seed=seed, max_epochs=args.max_epochs, batch_size=args.batch_size,
        learning_rate=args.lr, weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        early_stopping_patience=args.early_stopping_patience,
        use_crf=True, freeze_encoder=False,
        init_encoder_state=encoder_state,
        device=device, verbose=True, return_model=True,
        encoder_cache_dir=args.encoder_cache_dir, config_name=config_name,
        probs_dump_path=probs_dump_path)
    return result, model, vocabs


def run_config_cv(config_name, sentences, folds, args, gold_dir, silver,
                  encoder_ckpt):
    """Run one configuration across all folds, write per_fold.json + fold_N.conllu."""
    out_dir = Path(args.out_dir) / config_name
    out_dir.mkdir(parents=True, exist_ok=True)
    per_fold_path = out_dir / 'per_fold.json'
    per_fold: List[dict] = []
    if per_fold_path.exists() and args.resume:
        per_fold = json.loads(per_fold_path.read_text(encoding='utf-8'))
        print(f'  [{config_name}] resuming: {len(per_fold)}/{len(folds)} folds done')

    device = args.device or _pick_device()

    # training seed decoupled from fold seed (revision task B: seed replication
    # retrains the SAME folds from a different initialization)
    train_seed = args.train_seed if args.train_seed is not None else args.seed
    write_run_meta(out_dir, device=device, driver='run_cv_kazroberta.py',
                   k=args.k, strategy=args.strategy, seed=args.seed,
                   train_seed=train_seed, config=config_name,
                   model_name=args.model_name, revision=args.revision,
                   max_epochs=args.max_epochs, batch_size=args.batch_size,
                   lr=args.lr, corpus_sentences=len(sentences))

    log_path = out_dir / 'run.log'
    with open(log_path, 'a', encoding='utf-8') as logf:
        logf.write(f'\n=== CV-kazroberta config={config_name} k={args.k} '
                   f'strategy={args.strategy} seed={args.seed} '
                   f'train_seed={train_seed} '
                   f'started={time.strftime("%Y-%m-%d %H:%M:%S")} ===\n')

    # Whether this config is the one that should donate the interface checkpoint.
    interface_target_config = args.interface_config or (args.configs.split(',')[0].strip())
    is_interface_config = config_name == interface_target_config

    freeze_encoder = (config_name == FROZEN)
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
        print(f'  [{config_name}] fold {fold.index}: train={len(train_s)} '
              f'dev={len(dev_s)} test={len(test_s)}')

        if config_name == SILVER_FINETUNE:
            result, model, vocabs = run_silver_finetune_fold(
                config_name, encoder_ckpt, train_s, dev_s, test_s,
                train_seed, args, device,
                probs_dump_path=str(out_dir / f'fold_{fold.index}_gramprobs.npz'))
        else:
            result, model, vocabs = run_gold_only_fold(
                config_name, train_s, dev_s, test_s, train_seed, args, device,
                freeze_encoder=freeze_encoder,
                probs_dump_path=str(out_dir / f'fold_{fold.index}_gramprobs.npz'))

        # Optionally export this fold's trained model + vocab as the web demo's
        # KazRoBERTa checkpoint. Done while the model object is still in scope
        # (before it is released at the next loop iteration). Only the chosen
        # interface config + interface fold is exported, and only once.
        if (is_interface_config and fold.index == args.interface_fold
                and args.save_interface_checkpoint):
            _save_interface_checkpoint(model, vocabs, result, args, config_name, fold)

        preds = predict_hf(model, test_s, vocabs,
                           batch_size=args.batch_size, device=device)
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
        per_fold_path.write_text(json.dumps(per_fold, ensure_ascii=False, indent=2),
                                 encoding='utf-8')
        with open(log_path, 'a', encoding='utf-8') as logf:
            logf.write(f'  fold {fold.index}: best_epoch={result.best_epoch} '
                       f'elapsed={elapsed:.0f}s '
                       f'official={ {k: round(v, 4) for k, v in official.items()} }\n')
        print(f'  [{config_name}] fold {fold.index} done in {elapsed:.0f}s: '
              f'UPOS={official["UPOS"]:.4f} UFeats={official["UFeats"]:.4f} '
              f'Lemmas={official["Lemmas"]:.4f}')

    _jackknife(config_name, folds, sentences, out_dir, gold_dir)


def _jackknife(config_name, folds, sentences, out_dir, gold_dir):
    """Concatenate per-fold predictions and score the whole corpus at once."""
    n_expected = sum(len(f.test_idx) for f in folds)
    if n_expected != len(sentences):
        print(f'  [{config_name}] jack-knife SKIPPED: fold subset covers '
              f'{n_expected}/{len(sentences)} sentences')
        return
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
    pf = json.loads((out_dir / 'per_fold.json').read_text(encoding='utf-8'))
    our = {}
    for task in ('lemma', 'upos', 'grammeme'):
        for metric in ('accuracy', 'f1'):
            vals = [r[task][metric] for r in pf]
            our[f'{task}_{metric}_mean'] = statistics.mean(vals)
            our[f'{task}_{metric}_std'] = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    jack = {'config': config_name, 'n_folds': len(folds), 'n_sentences': n_pred,
            'official_jackknifed': official, 'our_metrics_mean_over_folds': our}
    (out_dir / 'jackknifed.json').write_text(
        json.dumps(jack, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'  [{config_name}] JACK-KNIFE ({n_pred} sents): '
          f'Lemmas={official["Lemmas"]:.4f} UPOS={official["UPOS"]:.4f} '
          f'UFeats={official["UFeats"]:.4f} AllTags={official["AllTags"]:.4f}')


def _pick_device():
    # Imported lazily so --help and arg errors don't require torch.
    from cagf.device import pick_device
    return pick_device()


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default='configs/default.yaml')
    ap.add_argument('--k', type=int, default=10)
    ap.add_argument('--strategy', choices=['stratified', 'grouped'], default='stratified')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--train-seed', type=int, default=None,
                    help='override the TRAINING seed only; folds stay at --seed '
                         '(seed replication retrains identical folds; also seeds '
                         'the one-time silver-encoder pretraining)')
    ap.add_argument('--folds', default=None,
                    help='comma-separated fold indices to run (default: all). Used '
                         'for the seed-replication run-to-run floor on a fixed, '
                         'pre-registered fold subset; the pooled jack-knife is '
                         'skipped automatically when folds are missing.')
    ap.add_argument('--out-dir', default='results_cv_kazroberta/stratified')
    ap.add_argument('--configs', default='gold_only',
                    help='comma-separated subset of: gold_only, silver_finetune, frozen')
    ap.add_argument('--model-name', default=DEFAULT_MODEL_NAME)
    ap.add_argument('--revision', default=DEFAULT_REVISION)
    ap.add_argument('--max-epochs', type=int, default=None,
                    help='fine-tune epochs per fold (default: min(YAML max_epochs, 40))')
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-5,
                    help='fine-tune learning rate for the gold folds')
    ap.add_argument('--weight-decay', type=float, default=0.01)
    ap.add_argument('--grad-clip-norm', type=float, default=5.0)
    ap.add_argument('--early-stopping-patience', type=int, default=5)
    ap.add_argument('--silver-path', default=None,
                    help='silver CoNLL-U (required for the silver_finetune config)')
    ap.add_argument('--silver-epochs', type=int, default=10,
                    help='encoder pretraining epochs on the silver corpus')
    ap.add_argument('--silver-lr', type=float, default=1e-5,
                    help='encoder pretraining learning rate on the silver corpus')
    ap.add_argument('--silver-patience', type=int, default=3,
                    help='encoder pretraining early-stopping patience (on gold dev)')
    ap.add_argument('--encoder-cache-dir', default=None,
                    help='HF cache dir for the KazRoBERTa download')
    ap.add_argument('--freeze-encoder', action='store_true', default=False,
                    help='freeze the KazRoBERTa encoder (only projection + heads train)')
    ap.add_argument('--resume', action='store_true', default=True)
    ap.add_argument('--no-resume', dest='resume', action='store_false')
    ap.add_argument('--device', default=None)
    ap.add_argument('--save-interface-checkpoint', default=None,
                    help='after the interface-source fold completes, also write a full '
                         'HFMorphModel checkpoint here (schema loadable by '
                         'cagf.hf_inference.load_hf_checkpoint) so webapp/app.py can '
                         'serve the KazRoBERTa backend')
    ap.add_argument('--save-interface-vocabs', default=None,
                    help='write the interface-source fold vocab JSON here (companion to '
                         '--save-interface-checkpoint, loadable by CorpusVocabs.load)')
    ap.add_argument('--interface-fold', type=int, default=0,
                    help='fold index whose trained model + vocab are exported via '
                         '--save-interface-checkpoint / --save-interface-vocabs (default 0)')
    ap.add_argument('--interface-config', default=None,
                    help='config whose folds are scanned for --interface-fold (defaults '
                         'to the first requested config)')
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    # max_epochs: --max-epochs takes precedence; otherwise the YAML's
    # training.max_epochs (clipped to 40 for HF -- 200 epochs on KazRoBERTa is
    # impractical and the default YAML is tuned for the small CAGF model).
    if args.max_epochs is None:
        args.max_epochs = min(int(cfg['training']['max_epochs']), 40)
    # batch_size / weight_decay / grad_clip_norm come from the CLI; the YAML's
    # training section is intentionally NOT consulted here because the HF
    # hyperparameters differ from the CAGF defaults (smaller LR, etc.).

    requested = [c.strip() for c in args.configs.split(',') if c.strip()]
    valid = {GOLD_ONLY, SILVER_FINETUNE, FROZEN}
    unknown = [c for c in requested if c not in valid]
    if unknown:
        raise SystemExit(f'unknown config(s) {unknown}; choose from {sorted(valid)}')
    if SILVER_FINETUNE in requested and not args.silver_path:
        raise SystemExit('--silver-path is required for the silver_finetune config')

    # load the whole gold corpus (train+dev+test merged) -- CV re-splits it.
    data_cfg = cfg.get('data', {})
    sentences = (read_conllu(data_cfg.get('train_path', 'data/gold_merged/gold_train.conllu')) +
                 read_conllu(data_cfg.get('dev_path', 'data/gold_merged/gold_dev.conllu')) +
                 read_conllu(data_cfg.get('test_path', 'data/gold_merged/gold_test.conllu')))
    print(f'Corpus: {len(sentences)} sentences')

    folds = make_folds(sentences, k=args.k, seed=args.seed, strategy=args.strategy)
    if args.folds:
        wanted = sorted({int(x) for x in args.folds.split(',') if x.strip()})
        folds = [f for f in folds if f.index in wanted]
        print(f'Fold subset: {wanted} (pooled jack-knife will be skipped)')
    print(f'Folds: k={args.k} strategy={args.strategy} seed={args.seed}, '
          f'sizes={[len(f.test_idx) for f in folds]}')

    gold_dir = Path(args.out_dir) / '_gold_folds'
    gold_dir.mkdir(parents=True, exist_ok=True)

    silver = None
    encoder_ckpt = None
    if SILVER_FINETUNE in requested:
        silver = read_conllu(args.silver_path)
        print(f'Silver: {len(silver)} sentences ({args.silver_path})')
        device = args.device or _pick_device()
        # ONE-TIME encoder pretraining. Done outside the fold loop so it is
        # shared across folds (the encoder is vocabulary-agnostic and the
        # silver corpus is external to every gold fold).
        # Use the first fold's dev/test as the early-stopping anchor -- the
        # silver corpus itself is the training data, so any gold dev set is
        # equally valid as the "transfer target" signal and fold 0 is just the
        # canonical choice.
        first = folds[0]
        dev_s = _split_by_index(sentences, first.dev_idx)
        test_s = _split_by_index(sentences, first.test_idx)
        train_seed = args.train_seed if args.train_seed is not None else args.seed
        encoder_ckpt = _train_silver_encoder_once(silver, dev_s, test_s, args, device,
                                                  train_seed)

    manifest = {'k': args.k, 'strategy': args.strategy, 'seed': args.seed,
                'train_seed': args.train_seed if args.train_seed is not None else args.seed,
                'folds': args.folds,
                'configs': requested, 'model_name': args.model_name,
                'revision': args.revision, 'corpus_sentences': len(sentences),
                'silver_sentences': len(silver) if silver else 0,
                'max_epochs': args.max_epochs, 'batch_size': args.batch_size,
                'lr': args.lr, 'date': time.strftime('%Y-%m-%d %H:%M:%S')}
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.out_dir) / 'manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')

    cv_raw: List[dict] = []
    for config_name in requested:
        print(f'\n=== CV-kazroberta config={config_name} ===')
        run_config_cv(config_name, sentences, folds, args, gold_dir, silver,
                      encoder_ckpt)
        pf = json.loads((Path(args.out_dir) / config_name / 'per_fold.json')
                        .read_text(encoding='utf-8'))
        for r in pf:
            cv_raw.append({'config': r['config'], 'fold': r['fold'], 'seed': r['seed'],
                           'lemma': r['lemma'], 'upos': r['upos'],
                           'grammeme': r['grammeme'], 'official': r['official'],
                           'best_epoch': r['best_epoch']})
    cv_raw_path = Path(args.out_dir) / 'cv_raw.json'
    cv_raw_path.write_text(json.dumps(cv_raw, ensure_ascii=False, indent=2),
                           encoding='utf-8')
    print(f'\nDone. Flat CV-kazroberta results: {cv_raw_path}')
    for config_name in requested:
        jpath = Path(args.out_dir) / config_name / 'jackknifed.json'
        if jpath.exists():
            j = json.loads(jpath.read_text(encoding='utf-8'))
            o = j['official_jackknifed']
            print(f'  {config_name}: Lemmas={o["Lemmas"]:.4f} UPOS={o["UPOS"]:.4f} '
                  f'UFeats={o["UFeats"]:.4f} AllTags={o["AllTags"]:.4f}')


if __name__ == '__main__':
    main()
