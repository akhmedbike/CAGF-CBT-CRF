"""Train a KazRoBERTa (:class:`HFMorphModel`) checkpoint for the web demo.

This is the HuggingFace analogue of :mod:`scripts.train_for_interface`: it
trains one HF model on the gold train/dev/test split and writes a checkpoint +
vocab pair that :func:`cagf.hf_inference.load_hf_checkpoint` can load, so the
Flask demo (``webapp/app.py``) can serve the KazRoBERTa backend instead of the
CAGF-CBT+CRF backend.

The resulting artifacts default to ``models/interface_hf_model.pt`` and
``models/interface_hf_vocabs.json`` -- exactly what ``webapp/app.py``'s
``HF_CHECKPOINT`` / ``HF_VOCABS`` point at. Train once, then the demo picks the
HF backend up automatically (the app prefers CAGF when both exist; delete or
move the CAGF checkpoint to serve HF).

Optionally initialise the KazRoBERTa encoder from the silver-pretrained weights
written by ``scripts/run_cv_kazroberta.py`` (the ``silver_finetune`` config):
pass ``--init-encoder results_cv_kazroberta/checkpoints/silver_encoder_kazroberta.pt``.
This reuses the same encoder-reuse path the silver_finetune CV config uses, so
the demo model can benefit from the same silver transfer without a separate
CV run.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import torch
import yaml

from cagf.data import build_vocabs, read_conllu
from cagf.hf_train_loop import train_hf_one_run
from scripts.run_cv_kazroberta import DEFAULT_MODEL_NAME, DEFAULT_REVISION


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default='configs/default.yaml')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--max-epochs', type=int, default=40)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-5)
    ap.add_argument('--weight-decay', type=float, default=0.01)
    ap.add_argument('--grad-clip-norm', type=float, default=5.0)
    ap.add_argument('--early-stopping-patience', type=int, default=5)
    ap.add_argument('--model-name', default=DEFAULT_MODEL_NAME)
    ap.add_argument('--revision', default=DEFAULT_REVISION)
    ap.add_argument('--encoder-cache-dir', default=None)
    ap.add_argument('--init-encoder', default=None,
                    help='silver-pretrained encoder .pt (silver_encoder_kazroberta.pt) '
                         'to initialise the KazRoBERTa encoder from')
    ap.add_argument('--out', default='models/interface_hf_model.pt')
    ap.add_argument('--vocabs-out', default='models/interface_hf_vocabs.json')
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    train = read_conllu(cfg['data']['train_path'])
    dev = read_conllu(cfg['data']['dev_path'])
    test = read_conllu(cfg['data']['test_path'])
    # HF label vocab: same fold-train-only invariant as the CV runner. min_word_freq=1
    # because the encoder is vocabulary-agnostic (only the heads need labels).
    vocabs = build_vocabs(train, min_word_freq=1, max_word_vocab=50000)

    Path(args.vocabs_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    vocabs.save(args.vocabs_out)

    init_encoder_state = None
    if args.init_encoder:
        # The silver checkpoint stores encoder tensors keyed ``encoder.*`` --
        # exactly the shape train_hf_one_run's init_encoder_state path expects.
        init_encoder_state = torch.load(args.init_encoder, map_location='cpu',
                                        weights_only=False)['encoder_state_dict']
        print(f'Initialising encoder from {args.init_encoder} '
              f'({len(init_encoder_state)} tensors)')

    print(f'Training KazRoBERTa, seed={args.seed}, max_epochs={args.max_epochs}, '
          f'on {len(train)} train / {len(dev)} dev / {len(test)} test sentences.')
    result = train_hf_one_run(
        train_sentences=train, dev_sentences=dev, test_sentences=test,
        vocabs=vocabs, model_name=args.model_name, revision=args.revision,
        seed=args.seed, max_epochs=args.max_epochs, batch_size=args.batch_size,
        learning_rate=args.lr, weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        early_stopping_patience=args.early_stopping_patience,
        use_crf=True, freeze_encoder=False,
        init_encoder_state=init_encoder_state,
        verbose=True, return_model=False,
        save_checkpoint_path=args.out,
        encoder_cache_dir=args.encoder_cache_dir,
        config_name='kazroberta_interface')

    # train_hf_one_run already wrote the checkpoint; annotate it with the
    # corpus sizes so the demo can show "trained on N sentences" (mirrors what
    # scripts/train_for_interface.py does for the CAGF checkpoint).
    ckpt = torch.load(args.out, weights_only=False)
    ckpt['train_sentences'] = len(train)
    ckpt['dev_sentences'] = len(dev)
    ckpt['test_sentences'] = len(test)
    torch.save(ckpt, args.out)

    print(f'\nTest metrics -- lemma: {result.lemma}')
    print(f'Test metrics -- upos:  {result.upos}')
    print(f'Test metrics -- grammeme: {result.grammeme}')
    print(f'\nCheckpoint saved to {args.out}')
    print(f'Vocabs saved to {args.vocabs_out}')
    print('\nThe web demo (webapp/app.py) auto-detects this checkpoint and serves '
          'the KazRoBERTa backend. It prefers CAGF when both exist; to serve HF '
          'exclusively, move models/interface_model.pt aside.')


if __name__ == '__main__':
    main()
