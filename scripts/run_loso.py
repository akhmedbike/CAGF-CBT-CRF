"""A.7: leave-one-source-out for Иран and wikipedia (R3.3, strictest protocol).

The whole held-out document is the test set; train = every other sentence
except a FIXED dev set; dev = a set of whole documents from sources other
than the two LOSO documents, ~10% of the corpus, chosen once (seed 42) and
identical in all 8 runs, exactly as the revision spec requires. akorda-random
and kdt are NOT re-run: grouped-5 baskets 0/1 of task A already hold them out
whole.

Four conditions per document, mirroring the CV protocol with identical
hyperparameters (nothing re-tuned): CAGF gold (configs/default.yaml training
section), CAGF silver-pretrain->finetune (transfer section), KazRoBERTa gold
(lr 5e-5 / batch 16 / 80 epochs — the grid selection used in every CV run)
and KazRoBERTa silver-finetune (silver encoder pretrained ONCE for LOSO with
the clean fixed dev — early stopping never sees a held-out document — then
finetuned per document; heads always from the document's train split).

Usage
-----
    PYTHONPATH=. .venv/bin/python scripts/run_loso.py \
        --silver data/silver/silver_unfiltered_subset_210k.conllu \
        --out-dir results_loso
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from types import SimpleNamespace

import yaml

from cagf.data import read_conllu
from cagf.device import pick_device
from cagf.folds import sent_id_of, source_of
from cagf.model import AblationConfig, ModelHParams
from cagf.official_eval import evaluate_conllu
from cagf.predict_writer import write_conllu
from cagf.run_meta import write_run_meta
from cagf.train_loop import predict, train_one_run
from cagf.hf_train_loop import predict_hf, train_hf_one_run
from scripts.run_cv import _gold_for_fold
from scripts.run_cv_silver import run_ktb_only_fold, run_silver_pretrain_finetune_fold
from scripts.run_cv_kazroberta import (_train_silver_encoder_once,
                                       run_silver_finetune_fold)

LOSO_DOCS = ['Иран.tagged.txt', 'wikipedia.tagged.txt']

KAZR = 'kz-transformers/kaz-roberta-conversational'
KAZR_REV = '43077c2fd0a163487ed468b5ec3b8750686a5888'


def build_loso_splits(sentences, docs, seed=42):
    """Held-out docs -> test; fixed whole-document dev (~10%); rest -> train.

    The dev pool excludes BOTH LOSO documents so one fixed dev set can serve
    all runs; documents are drawn whole in a seed-42 shuffle until the ~10%
    budget is reached."""
    by_src = {}
    for s in sentences:
        by_src.setdefault(source_of(sent_id_of(s)), []).append(s)
    pool = sorted(src for src in by_src if src not in docs)
    rng = random.Random(seed)
    rng.shuffle(pool)
    target = max(1, round(0.10 * len(sentences)))
    dev_docs, n = [], 0
    for src in pool:
        dev_docs.append(src)
        n += len(by_src[src])
        if n >= target:
            break
    dev = [s for src in dev_docs for s in by_src[src]]
    splits = {}
    for doc in docs:
        test = by_src[doc]
        train = [s for src in by_src if src not in dev_docs and src != doc for s in by_src[src]]
        splits[doc] = (train, dev, test)
    return splits, dev_docs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', default='configs/default.yaml')
    ap.add_argument('--silver', default='data/silver/silver_unfiltered_subset_210k.conllu')
    ap.add_argument('--out-dir', default='results_loso')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', default=None)
    ap.add_argument('--ktb-epochs', type=int, default=None,
                    help='override CAGF train epochs (smoke testing only)')
    ap.add_argument('--kazr-epochs', type=int, default=80)
    ap.add_argument('--configs', default='cagf_gold,cagf_silver,kazr_gold,kazr_silver')
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    train_cfg, transfer_cfg = cfg['training'], cfg['transfer']
    hp = ModelHParams()
    ablation = AblationConfig()
    device = args.device or pick_device()

    sentences = (read_conllu(cfg['data']['train_path']) +
                 read_conllu(cfg['data']['dev_path']) +
                 read_conllu(cfg['data']['test_path']))
    splits, dev_docs = build_loso_splits(sentences, LOSO_DOCS, seed=args.seed)
    n_dev = sum(1 for s in sentences if source_of(sent_id_of(s)) in dev_docs)
    print(f'LOSO docs: {LOSO_DOCS}')
    print(f'fixed dev docs ({n_dev} sentences, {100*n_dev/len(sentences):.1f}%): {dev_docs}')

    silver = None
    silver_cfg = SimpleNamespace(
        out_dir=str(Path(args.out_dir) / '_silver_tmp'),
        min_word_freq=1, max_word_vocab=50000,
        batch_size=train_cfg['batch_size'],
        pretrain_epochs=transfer_cfg['pretrain_epochs'],
        pretrain_lr=transfer_cfg['pretrain_lr'],
        pretrain_patience=transfer_cfg['pretrain_patience'],
        finetune_epochs=args.ktb_epochs or transfer_cfg['finetune_epochs'],
        finetune_lr=transfer_cfg['finetune_lr'],
        finetune_patience=transfer_cfg['finetune_patience'],
        weight_decay=train_cfg['weight_decay'],
        grad_clip_norm=train_cfg['grad_clip_norm'])
    kazr_cfg = SimpleNamespace(
        out_dir=str(Path(args.out_dir)), model_name=KAZR, revision=KAZR_REV,
        max_epochs=args.kazr_epochs, batch_size=16, lr=5e-5,
        weight_decay=0.01, grad_clip_norm=5.0, early_stopping_patience=5,
        silver_epochs=10, silver_lr=1e-5, silver_patience=3,
        encoder_cache_dir=None)

    requested = [c.strip() for c in args.configs.split(',')]
    results = {}
    for doc in LOSO_DOCS:
        train_s, dev_s, test_s = splits[doc]
        doc_stem = doc.replace('.tagged.txt', '')
        print(f'\n=== LOSO {doc}: train={len(train_s)} dev={len(dev_s)} test={len(test_s)} ===')
        for cond in requested:
            cond_dir = Path(args.out_dir) / doc_stem / cond
            done_path = cond_dir / 'result.json'
            if done_path.exists():
                results[f'{doc_stem}/{cond}'] = json.loads(done_path.read_text(encoding='utf-8'))
                print(f'  [{cond}] already done — skipped')
                continue
            cond_dir.mkdir(parents=True, exist_ok=True)
            write_run_meta(cond_dir, device=device, driver='run_loso.py',
                           held_out=doc, condition=cond, seed=args.seed,
                           n_train=len(train_s), n_dev=len(dev_s), n_test=len(test_s))
            t0 = time.time()
            preds_path = cond_dir / 'pred.conllu'
            gold_path = cond_dir / 'gold.conllu'

            if cond == 'cagf_gold':
                tc = dict(train_cfg)
                if args.ktb_epochs:
                    tc['max_epochs'] = args.ktb_epochs
                result, model, vocabs = run_ktb_only_fold(
                    train_s, dev_s, test_s, hp, ablation, args.seed, tc, device)
                preds = predict(model, test_s, vocabs,
                                batch_size=train_cfg['batch_size'], device=device)
            elif cond == 'cagf_silver':
                if silver is None:
                    silver = read_conllu(args.silver)
                silver_cfg.out_dir = str(cond_dir)  # tmp pretrain ckpt lives here
                result, model, vocabs = run_silver_pretrain_finetune_fold(
                    cond, silver, train_s, dev_s, test_s, hp, ablation,
                    args.seed, silver_cfg, device)
                preds = predict(model, test_s, vocabs,
                                batch_size=train_cfg['batch_size'], device=device)
            elif cond == 'kazr_gold':
                from cagf.data import build_vocabs
                vocabs = build_vocabs(train_s, min_word_freq=1, max_word_vocab=50000)
                result, model = train_hf_one_run(
                    train_sentences=train_s, dev_sentences=dev_s, test_sentences=test_s,
                    vocabs=vocabs, model_name=KAZR, revision=KAZR_REV, seed=args.seed,
                    max_epochs=args.kazr_epochs, batch_size=16, learning_rate=5e-5,
                    weight_decay=0.01, grad_clip_norm=5.0, early_stopping_patience=5,
                    use_crf=True, freeze_encoder=False, device=device,
                    verbose=True, return_model=True)
                preds = predict_hf(model, test_s, vocabs,
                                   batch_size=train_cfg['batch_size'], device=device)
            elif cond == 'kazr_silver':
                if silver is None:
                    silver = read_conllu(args.silver)
                # encoder pretrained ONCE with the clean fixed dev (cached on disk)
                encoder_ckpt = _train_silver_encoder_once(
                    silver, dev_s, dev_s, kazr_cfg, device, args.seed)
                result, model, vocabs = run_silver_finetune_fold(
                    cond, encoder_ckpt, train_s, dev_s, test_s, args.seed,
                    kazr_cfg, device)
                preds = predict_hf(model, test_s, vocabs,
                                   batch_size=train_cfg['batch_size'], device=device)
            else:
                raise ValueError(f'unknown condition {cond}')

            write_conllu(test_s, preds, preds_path)
            _gold_for_fold(test_s, gold_path)
            official = evaluate_conllu(str(gold_path), str(preds_path))
            entry = {'held_out': doc, 'condition': cond, 'seed': args.seed,
                     'n_train': len(train_s), 'n_dev': len(dev_s), 'n_test': len(test_s),
                     'official': official, 'elapsed_sec': time.time() - t0}
            done_path.write_text(json.dumps(entry, ensure_ascii=False, indent=2),
                                 encoding='utf-8')
            results[f'{doc_stem}/{cond}'] = entry
            print(f"  [{cond}] official: "
                  + ' '.join(f'{k}={v:.4f}' for k, v in official.items()))
            del model

    summary_path = Path(args.out_dir) / 'loso_results.json'
    summary_path.write_text(json.dumps(
        {'docs': LOSO_DOCS, 'fixed_dev_docs': dev_docs, 'seed': args.seed,
         'results': results}, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\nWrote {summary_path}')


if __name__ == '__main__':
    main()
