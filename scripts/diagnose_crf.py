"""D1 diagnostic: measure gradient norms before clipping, full_model vs wo_crf.

The hypothesis from the reviewer analysis: CRF neg_log_likelihood is summed
over a sequence and then meaned over the batch, while the lemma/grammeme
losses are meaned over tokens. If the CRF loss magnitude is much larger,
the global clip_grad_norm_ (applied to ALL parameters together) clips the
lemma/grammeme gradients more aggressively, starving those heads.

This script measures, for 30 training steps on fold 0, for both full_model
and wo_crf:
  - per-component loss magnitudes (lemma, upos, grammeme)
  - total loss
  - gradient norm BEFORE clipping
  - gradient norm of each head's parameters separately
  - whether clip_grad_norm_ actually fired (grad_norm > max_norm)

No checkpoint is saved; this is pure instrumentation.
"""
from __future__ import annotations
import statistics
from pathlib import Path

import torch
from cagf.data import build_vocabs, read_conllu
from cagf.device import pick_device
from cagf.folds import make_folds
from cagf.losses import UncertaintyWeightedLoss, masked_lemma_ce, masked_multilabel_bce
from cagf.model import AblationConfig, ModelHParams
from cagf.train_loop import make_loader

GRAD_CLIP = 5.0  # from configs/default.yaml


def head_grad_norm(model, prefix: str) -> float:
    total = 0.0
    found = False
    for name, p in model.named_parameters():
        if name.startswith(prefix) and p.grad is not None:
            total += float(p.grad.norm(2).item()) ** 2
            found = True
    return total ** 0.5 if found else 0.0


def run_config(ablation: AblationConfig, label: str, train_sents, dev_sents, vocabs, n_steps: int = 30):
    device = pick_device()
    torch.manual_seed(42)
    model = type('M', (), {})()  # placeholder, replaced below
    from cagf.model import CAGFCBTCRF
    model = CAGFCBTCRF(
        num_chars=len(vocabs.char_vocab), num_words=len(vocabs.word_vocab),
        num_upos=len(vocabs.upos_vocab), num_grammemes=len(vocabs.grammeme_vocab),
        num_lemma_rules=len(vocabs.lemma_rule_vocab),
        hparams=ModelHParams(), ablation=ablation,
        char_pad_id=vocabs.char_vocab.stoi['<pad>'],
        word_pad_id=vocabs.word_vocab.stoi['<pad>']).to(device)
    loss_weighter = UncertaintyWeightedLoss(num_tasks=3).to(device)
    params = list(model.parameters()) + list(loss_weighter.parameters())
    optimizer = torch.optim.AdamW(params, lr=3e-4, weight_decay=0.01)
    loader = make_loader(train_sents, vocabs, batch_size=32, shuffle=True)

    rows = []
    step = 0
    for batch in loader:
        if step >= n_steps:
            break
        word_ids = batch['word_ids'].to(device)
        char_ids = batch['char_ids'].to(device)
        lengths = batch['lengths'].to(device)
        mask = batch['mask'].to(device)
        upos_ids = batch['upos_ids'].to(device)
        lemma_ids = batch['lemma_rule_ids'].to(device)
        gram_targets = batch['grammeme_targets'].to(device)
        optimizer.zero_grad()
        out = model(word_ids, char_ids, lengths, mask, upos_ids=upos_ids)
        lemma_loss = masked_lemma_ce(out['lemma_logits'], lemma_ids)
        gram_loss = masked_multilabel_bce(out['grammeme_logits'], gram_targets, mask)
        upos_loss = out['upos_nll']
        total_loss, w = loss_weighter([lemma_loss, upos_loss, gram_loss])
        total_loss.backward()
        # measure BEFORE clipping
        total_grad_norm = float(torch.nn.utils.clip_grad_norm_(params, float('inf')).item())
        lemma_gn = head_grad_norm(model, 'lemma_head')
        gram_gn = head_grad_norm(model, 'grammeme_head')
        upos_gn = head_grad_norm(model, 'upos_head')
        crf_gn = head_grad_norm(model, 'crf')
        clipped = total_grad_norm > GRAD_CLIP
        rows.append({
            'step': step, 'lemma_loss': float(lemma_loss.item()),
            'upos_loss': float(upos_loss.item()), 'gram_loss': float(gram_loss.item()),
            'total_loss': float(total_loss.item()),
            'uw_w0_lemma': w['weight_0'], 'uw_w1_upos': w['weight_1'], 'uw_w2_gram': w['weight_2'],
            'total_grad_norm': total_grad_norm, 'clipped': clipped,
            'lemma_head_gn': lemma_gn, 'gram_head_gn': gram_gn,
            'upos_head_gn': upos_gn, 'crf_gn': crf_gn,
        })
        # actually clip + step now so the model learns (mimic real training)
        torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
        optimizer.step()
        step += 1

    return rows


def summarize(rows: list, label: str) -> None:
    n = len(rows)
    clip_frac = sum(r['clipped'] for r in rows) / n

    def med(key):
        return statistics.median(r[key] for r in rows)

    print(f'\n=== {label} (n={n} steps) ===')
    print('  loss magnitudes (median):')
    print(f'    lemma:  {med("lemma_loss"):.4f}')
    print(f'    upos:   {med("upos_loss"):.4f}')
    print(f'    gramm:  {med("gram_loss"):.4f}')
    print(f'    total:  {med("total_loss"):.4f}')
    print('  UncertaintyWeighting (median precision = 1/var):')
    print(f'    lemma w: {med("uw_w0_lemma"):.4f}')
    print(f'    upos  w: {med("uw_w1_upos"):.4f}')
    print(f'    gramm w: {med("uw_w2_gram"):.4f}')
    print('  gradient norms (median, BEFORE clipping):')
    print(f'    TOTAL:        {med("total_grad_norm"):.4f}   (clip fired on {100*clip_frac:.0f}% of steps)')
    print(f'    lemma_head:   {med("lemma_head_gn"):.4f}')
    print(f'    gram_head:    {med("gram_head_gn"):.4f}')
    print(f'    upos_head:    {med("upos_head_gn"):.4f}')
    print(f'    crf params:   {med("crf_gn"):.4f}')


def main():
    print('D1 diagnostic: gradient norms before clipping, full_model vs wo_crf')
    print(f'(GRAD_CLIP = {GRAD_CLIP}, 30 steps on fold 0 of CV)\n')
    sents = (read_conllu('data/gold_merged/gold_train.conllu') +
             read_conllu('data/gold_merged/gold_dev.conllu') +
             read_conllu('data/gold_merged/gold_test.conllu'))
    folds = make_folds(sents, k=10, seed=42, strategy='stratified')
    fold0 = folds[0]
    train_s = [sents[i] for i in fold0.train_idx]
    dev_s = [sents[i] for i in fold0.dev_idx]
    vocabs = build_vocabs(train_s, min_word_freq=1, max_word_vocab=50000)
    print(f'fold 0: train={len(train_s)} dev={len(dev_s)} vocab(lemma_rules={len(vocabs.lemma_rule_vocab)})')

    rows_full = run_config(AblationConfig(), 'full_model', train_s, dev_s, vocabs)
    rows_crf = run_config(AblationConfig(use_crf=False), 'wo_crf', train_s, dev_s, vocabs)
    summarize(rows_full, 'full_model (with CRF)')
    summarize(rows_crf, 'wo_crf (softmax UPOS)')

    print('\n=== HEAD-TO-HEAD: lemma_head + gram_head grad norms ===')

    def med2(rows, key):
        return statistics.median(r[key] for r in rows)

    lh_full = med2(rows_full, 'lemma_head_gn')
    lh_crf = med2(rows_crf, 'lemma_head_gn')
    gh_full = med2(rows_full, 'gram_head_gn')
    gh_crf = med2(rows_crf, 'gram_head_gn')
    tg_full = med2(rows_full, 'total_grad_norm')
    tg_crf = med2(rows_crf, 'total_grad_norm')
    clip_full = 100 * sum(r['clipped'] for r in rows_full) / len(rows_full)
    clip_crf = 100 * sum(r['clipped'] for r in rows_crf) / len(rows_crf)
    print(f'  lemma_head grad norm: full={lh_full:.4f}  wo_crf={lh_crf:.4f}  ratio={lh_full/max(lh_crf,1e-12):.2f}x')
    print(f'  gram_head grad norm:  full={gh_full:.4f}  wo_crf={gh_crf:.4f}  ratio={gh_full/max(gh_crf,1e-12):.2f}x')
    print(f'  total grad norm:      full={tg_full:.4f}  wo_crf={tg_crf:.4f}')
    print(f'  clip fired:           full={clip_full:.0f}%  wo_crf={clip_crf:.0f}%')
    print('\nD1 verdict: if full_model clips more often AND its lemma/gram heads have')
    print('smaller grad norms than wo_crf, gradient starvation is the mechanism.')


if __name__ == '__main__':
    main()
