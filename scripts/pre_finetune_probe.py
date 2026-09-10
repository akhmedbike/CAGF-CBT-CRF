"""P1a: what does the pure silver-pretrained model predict before fine-tuning?

§2.5 reports the silver corpus carries 17.99% X-tagged tokens against 1.08%
in gold KTB; R2.2's answer needs the model-side view of that prior shift. This
script takes fold 0 of the standard 10-fold stratified protocol (seed 42),
runs the exact silver-pretraining stage of the CV silver condition (30 epochs,
early stop on the fold's gold dev), then — WITHOUT any gold fine-tuning —
predicts the fold's gold test set and reports the official metrics, the
predicted UPOS distribution and the X share, next to the gold test's own
distribution.

Output: results_cv_silver/p1a_pre_finetune.json (+ pred/gold conllu).
"""
from __future__ import annotations

import argparse
import json
import tempfile
from collections import Counter
from pathlib import Path

import torch
import yaml

from cagf.data import read_conllu
from cagf.device import pick_device
from cagf.folds import make_folds
from cagf.model import AblationConfig, ModelHParams
from cagf.official_eval import evaluate_conllu
from cagf.predict_writer import write_conllu
from cagf.run_meta import write_run_meta
from cagf.train_loop import predict, train_one_run
from scripts.pretrain_finetune import build_shared_vocabs
from scripts.run_cv import _gold_for_fold


def upos_distribution(conllu_path: Path) -> dict:
    counts = Counter()
    for line in conllu_path.read_text(encoding='utf-8').splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        cols = line.split('\t')
        if '-' in cols[0] or '.' in cols[0]:
            continue
        counts[cols[3]] += 1
    total = sum(counts.values())
    return {'total': total,
            'x_share': counts.get('X', 0) / total if total else None,
            'distribution': dict(counts.most_common())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', default='configs/default.yaml')
    ap.add_argument('--silver', default='data/silver/silver_unfiltered_subset_210k.conllu')
    ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--out-dir', default='results_cv_silver/p1a_pre_finetune')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    train_cfg, transfer = cfg['training'], cfg['transfer']
    device = args.device or pick_device()

    sentences = (read_conllu(cfg['data']['train_path']) +
                 read_conllu(cfg['data']['dev_path']) +
                 read_conllu(cfg['data']['test_path']))
    folds = make_folds(sentences, k=10, seed=args.seed, strategy='stratified')
    fold = folds[args.fold]
    train_s = [sentences[i] for i in fold.train_idx]
    dev_s = [sentences[i] for i in fold.dev_idx]
    test_s = [sentences[i] for i in fold.test_idx]
    silver = read_conllu(args.silver)

    vocabs = build_shared_vocabs(train_s, silver, min_word_freq=1, max_word_vocab=50000)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_run_meta(out_dir, device=device, driver='pre_finetune_probe.py',
                   fold=args.fold, seed=args.seed, stage='silver_pretrain_only')

    print(f'fold {args.fold}: train={len(train_s)} dev={len(dev_s)} test={len(test_s)} '
          f'silver={len(silver)}')
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        ckpt = f.name
    result, model = train_one_run(
        train_sentences=silver, dev_sentences=dev_s, test_sentences=test_s,
        vocabs=vocabs, ablation=AblationConfig(), hparams=ModelHParams(), seed=args.seed,
        max_epochs=transfer['pretrain_epochs'], batch_size=train_cfg['batch_size'],
        learning_rate=transfer['pretrain_lr'], weight_decay=train_cfg['weight_decay'],
        grad_clip_norm=train_cfg['grad_clip_norm'],
        early_stopping_patience=transfer['pretrain_patience'],
        device=device, verbose=True, save_checkpoint_path=ckpt, return_model=True)
    Path(ckpt).unlink(missing_ok=True)

    preds = predict(model, test_s, vocabs, batch_size=train_cfg['batch_size'], device=device)
    pred_path = out_dir / 'pred.conllu'
    gold_path = out_dir / 'gold.conllu'
    write_conllu(test_s, preds, pred_path)
    _gold_for_fold(test_s, gold_path)
    official = evaluate_conllu(str(gold_path), str(pred_path))

    report = {
        'stage': 'silver pretraining only (no gold fine-tuning)',
        'fold': args.fold, 'seed': args.seed,
        'silver_sentences': len(silver),
        'best_epoch_on_gold_dev': result.best_epoch,
        'official_on_gold_test': official,
        'predicted_upos': upos_distribution(pred_path),
        'gold_upos': upos_distribution(gold_path),
    }
    out = out_dir / 'p1a_pre_finetune.json'
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\nofficial (silver-only model, gold test): '
          + ' '.join(f'{k}={v:.4f}' for k, v in official.items()))
    print(f"predicted X share: {report['predicted_upos']['x_share']:.4f} "
          f"(gold test: {report['gold_upos']['x_share']:.4f})")
    print(f'Wrote {out}')


if __name__ == '__main__':
    main()
