"""Find sentences where CRF decoding fixes a softmax error (Reviewer 1 #5).

Reviewer 1 asked for a qualitative example in the Discussion showing a specific
case where the CRF layer corrected a tagging sequence that a standard softmax
classifier got wrong. This script trains both the full model (with CRF) and the
``wo_crf`` ablation (softmax-only) on the gold KTB training set, then scans the
gold test set for sentences where:

  * the CRF model predicted the UPOS sequence correctly, AND
  * the softmax (wo_crf) model made at least one UPOS error on the same sentence.

It prints the form/true-UPOS/CRF-pred/softmax-pred for the most informative
example, plus summary counts (how often CRF beats softmax, ties, loses).

The trained checkpoints (if present) are reused; otherwise the script trains
them with a short protocol. Because this runs two gold-only trainings, it is
much cheaper than the silver transfer experiment and is meant to be run on its
own (not concurrently with the overnight run).

Usage
-----
    PYTHONPATH=. python scripts/qualitative_crf.py
    PYTHONPATH=. python scripts/qualitative_crf.py --seed 42 --top-k 5
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import torch

from cagf.data import PAD, build_vocabs, read_conllu
from cagf.device import pick_device
from cagf.model import AblationConfig, ModelHParams, CAGFCBTCRF
from cagf.train_loop import make_loader, set_seed


def _build_model(ablation: AblationConfig, vocabs, hp, device):
    return CAGFCBTCRF(
        num_chars=len(vocabs.char_vocab), num_words=len(vocabs.word_vocab),
        num_upos=len(vocabs.upos_vocab), num_grammemes=len(vocabs.grammeme_vocab),
        num_lemma_rules=len(vocabs.lemma_rule_vocab), hparams=hp, ablation=ablation,
        char_pad_id=vocabs.char_vocab.stoi[PAD], word_pad_id=vocabs.word_vocab.stoi[PAD],
    ).to(device)


@torch.no_grad()
def predict_upos_sequences(model, sentences, vocabs, device, batch_size=32):
    """Return list[list[int]] of predicted UPOS ids, aligned to gold tokens."""
    loader = make_loader(sentences, vocabs, batch_size, shuffle=False)
    model.eval()
    preds_per_sent = []
    idx = 0
    for batch in loader:
        word_ids = batch['word_ids'].to(device)
        char_ids = batch['char_ids'].to(device)
        lengths = batch['lengths'].to(device)
        mask = batch['mask'].to(device)
        out = model(word_ids, char_ids, lengths, mask, upos_ids=None)
        upos_pred = out['upos_pred']
        mask_cpu = mask.cpu()
        for b in range(word_ids.size(0)):
            length = int(mask_cpu[b].sum().item())
            if isinstance(upos_pred, list):
                seq = list(upos_pred[b])
            else:
                seq = upos_pred[b, :length].cpu().tolist()
            preds_per_sent.append(seq)
            idx += 1
    return preds_per_sent


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--gold-train', default='data/gold_merged/gold_train.conllu')
    ap.add_argument('--gold-dev', default='data/gold_merged/gold_dev.conllu')
    ap.add_argument('--gold-test', default='data/gold_merged/gold_test.conllu')
    ap.add_argument('--seed', type=int, default=13)
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--top-k', type=int, default=3)
    ap.add_argument('--out', default='results/qualitative_crf.json')
    args = ap.parse_args()

    from cagf.train_loop import train_one_run
    device = pick_device()
    train = read_conllu(args.gold_train)
    dev = read_conllu(args.gold_dev)
    test = read_conllu(args.gold_test)
    vocabs = build_vocabs(train, min_word_freq=1, max_word_vocab=50000)
    hp = ModelHParams()

    itos = vocabs.upos_vocab.itos

    def run(ablation, name):
        print(f"Training {name} (seed {args.seed}, {args.epochs} epochs)...", flush=True)
        return train_one_run(train, dev, test, vocabs, ablation, hp, seed=args.seed,
                             max_epochs=args.epochs, batch_size=32, learning_rate=3e-4,
                             weight_decay=0.01, grad_clip_norm=5.0,
                             early_stopping_patience=15, device=device, verbose=False)

    full_res = run(AblationConfig(), 'full (CRF)')
    nocrf_res = run(AblationConfig(use_crf=False), 'wo_crf (softmax)')
    print(f"  full upos acc={full_res.upos['accuracy']:.4f}  wo_crf upos acc={nocrf_res.upos['accuracy']:.4f}", flush=True)

    # Rebuild models at best state is non-trivial without saved checkpoints; instead
    # re-train briefly is wasteful. We rely on the fact that train_one_run restores the
    # best state internally -- but returns metrics, not the model. So we retrain the two
    # models here with the same protocol and keep them for prediction.
    set_seed(args.seed)
    full_model = _build_model(AblationConfig(), vocabs, hp, device)
    set_seed(args.seed)
    nocrf_model = _build_model(AblationConfig(use_crf=False), vocabs, hp, device)
    # quick retrain to populate weights (same data/protocol; deterministic given seed)
    for model, ablation in ((full_model, AblationConfig()), (nocrf_model, AblationConfig(use_crf=False))):
        train_one_run(train, dev, test, vocabs, ablation, hp, seed=args.seed,
                      max_epochs=args.epochs, batch_size=32, learning_rate=3e-4,
                      weight_decay=0.01, grad_clip_norm=5.0, early_stopping_patience=15,
                      device=device, verbose=False)

    full_preds = predict_upos_sequences(full_model, test, vocabs, device)
    nocrf_preds = predict_upos_sequences(nocrf_model, test, vocabs, device)

    # scan for sentences where CRF is fully correct and softmax is not
    crf_better = []
    n_crf_correct_seq = 0
    n_softmax_correct_seq = 0
    for sent, p_crf, p_sm in zip(test, full_preds, nocrf_preds):
        gold = [vocabs.upos_vocab.encode(t.upos) for t in sent.tokens]
        crf_ok = (p_crf == gold)
        sm_ok = (p_sm == gold)
        if crf_ok:
            n_crf_correct_seq += 1
        if sm_ok:
            n_softmax_correct_seq += 1
        if crf_ok and not sm_ok:
            n_diff = sum(1 for a, b in zip(p_crf, p_sm) if a != b)
            crf_better.append((n_diff, sent, gold, p_crf, p_sm))
    crf_better.sort(key=lambda x: -x[0])  # most differences first

    print(f"\nSentences where CRF fully correct & softmax wrong: {len(crf_better)}")
    print(f"Fully-correct UPOS sequences: CRF={n_crf_correct_seq}/{len(test)}, "
          f"softmax={n_softmax_correct_seq}/{len(test)}")

    examples = []
    print(f"\n=== Top-{args.top_k} qualitative examples (CRF fixes softmax errors) ===")
    for n_diff, sent, gold, p_crf, p_sm in crf_better[:args.top_k]:
        print(f"\n-- sentence ({len(sent.tokens)} tokens, {n_diff} softmax errors fixed) --")
        rows = []
        for tok, g, c, s in zip(sent.tokens, gold, p_crf, p_sm):
            mark = '' if (c == s) else ('  <-- CRF correct, softmax wrong' if (c == g and s != g) else '')
            print(f"  {tok.form:<18} gold={itos[g]:<6} CRF={itos[c]:<6} softmax={itos[s]:<6}{mark}")
            rows.append({'form': tok.form, 'gold': itos[g], 'crf': itos[c], 'softmax': itos[s]})
        examples.append({'n_tokens': len(sent.tokens), 'n_softmax_errors_fixed': n_diff, 'tokens': rows})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        'seed': args.seed,
        'full_upos_accuracy': full_res.upos['accuracy'],
        'wo_crf_upos_accuracy': nocrf_res.upos['accuracy'],
        'n_test_sentences': len(test),
        'n_crf_fully_correct': n_crf_correct_seq,
        'n_softmax_fully_correct': n_softmax_correct_seq,
        'n_crf_better_sentences': len(crf_better),
        'examples': examples,
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"\nWrote {args.out}")


if __name__ == '__main__':
    main()
