"""Training loop for :class:`HFMorphModel` (KazRoBERTa-backed morphological tagger).

This module mirrors :mod:`cagf.train_loop` but adapts the train / evaluate /
predict functions to the HF encoder's input contract. The only structural
difference from ``CAGFCBTCRF`` is that the HF encoder consumes the raw surface
forms (``word_strings_batch: List[List[str]]``) rather than ``word_ids`` /
``char_ids`` tensors -- the HF tokenizer is responsible for subword
segmentation, so the vocab IDs the CAGF model needs are irrelevant here.

Everything else is intentionally identical to ``train_loop``:

* the same :class:`UncertaintyWeightedLoss` (Kendall 2018) over the three task
  losses (lemma / upos / grammeme);
* the same AdamW + ReduceLROnPlateau scheduler;
* the same early-stopping on dev macro-F1 mean;
* the same :func:`apply_edit_script` lemma decoding with UPOS-aware case
  restoration, so downstream :mod:`cagf.predict_writer` output is byte-for-byte
  compatible with the CAGF pipeline and the official ``conll18_ud_eval``
  scorer.

The function :func:`train_hf_one_run` is the HF analogue of
:func:`cagf.train_loop.train_one_run`; it returns the same :class:`RunResult`
so the two model families can be reported side by side with no extra glue.
"""
from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import PAD, CorpusVocabs, Sentence
from .dataset import MorphDataset, collate_batch
from .device import pick_device
from .hf_model import HFMorphModel
from .losses import UncertaintyWeightedLoss, masked_multilabel_bce, masked_lemma_ce
from .metrics import compute_classification_metrics, compute_multilabel_metrics
from .train_loop import RunResult, set_seed


def make_hf_loader(sentences: list[Sentence], vocabs: CorpusVocabs,
                   batch_size: int, shuffle: bool) -> DataLoader:
    """DataLoader whose collate emits the ``word_strings`` the HF encoder needs.

    This reuses :class:`MorphDataset` + :func:`collate_batch` unchanged: the
    dataset already returns ``word_strings`` (the raw surface forms per token),
    and the collate passes the list-of-lists through without tensorising it.
    The ``pad_word_id`` / ``pad_char_id`` arguments are still required by the
    collate function (it builds ``word_ids`` / ``char_ids`` tensors that the HF
    model ignores but that the dataset contract still emits), so we supply them
    from the vocab as in :func:`cagf.train_loop.make_loader`.
    """
    ds = MorphDataset(sentences, vocabs)
    collate = partial(collate_batch,
                      pad_word_id=vocabs.word_vocab.stoi[PAD],
                      pad_char_id=vocabs.char_vocab.stoi[PAD])
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)


@torch.no_grad()
def evaluate_hf(model: HFMorphModel, loader: DataLoader, device: str) -> dict:
    """Evaluate ``model`` on ``loader`` and return per-task metric dicts.

    Same return schema as :func:`cagf.train_loop.evaluate`
    (``{'lemma': ..., 'upos': ..., 'grammeme': ...}`` with accuracy / precision
    / recall / f1). Calls ``model(word_strings, mask, upos_ids=None)`` -- the
    HF model needs neither ``word_ids`` nor ``char_ids``.
    """
    model.eval()
    all_lemma_true, all_lemma_pred = ([], [])
    all_upos_true, all_upos_pred = ([], [])
    all_gram_true, all_gram_pred = ([], [])
    for batch in loader:
        word_strings = batch['word_strings']
        mask = batch['mask'].to(device)
        upos_ids = batch['upos_ids'].to(device)
        lemma_ids = batch['lemma_rule_ids'].to(device)
        gram_targets = batch['grammeme_targets'].to(device)
        out = model(word_strings, mask, upos_ids=None)
        lemma_pred = out['lemma_logits'].argmax(dim=-1)
        gram_pred = (torch.sigmoid(out['grammeme_logits']) > 0.5).float()
        upos_pred = out['upos_pred']
        mask_cpu = mask.cpu()
        batch_size = len(word_strings)
        for b in range(batch_size):
            length = int(mask_cpu[b].sum().item())
            all_lemma_true.extend(lemma_ids[b, :length].cpu().tolist())
            all_lemma_pred.extend(lemma_pred[b, :length].cpu().tolist())
            if isinstance(upos_pred, list):
                all_upos_pred.extend(upos_pred[b])
            else:
                all_upos_pred.extend(upos_pred[b, :length].cpu().tolist())
            all_upos_true.extend(upos_ids[b, :length].cpu().tolist())
            all_gram_true.append(gram_targets[b, :length].cpu().numpy())
            all_gram_pred.append(gram_pred[b, :length].cpu().numpy())
    lemma_metrics = compute_classification_metrics(all_lemma_true, all_lemma_pred)
    upos_metrics = compute_classification_metrics(all_upos_true, all_upos_pred)
    gram_metrics = compute_multilabel_metrics(
        np.concatenate(all_gram_true, axis=0), np.concatenate(all_gram_pred, axis=0))
    return {'lemma': lemma_metrics.as_dict(), 'upos': upos_metrics.as_dict(),
            'grammeme': gram_metrics.as_dict()}


@torch.no_grad()
def predict_hf(model: HFMorphModel, sentences: list[Sentence], vocabs: CorpusVocabs,
               batch_size: int, device: str) -> list[dict]:
    """Run a trained HF model over ``sentences`` and decode per-token strings.

    Same output schema as :func:`cagf.train_loop.predict`: one dict per input
    sentence with ``lemma`` / ``upos`` / ``feats`` lists aligned to
    ``sentence.tokens``. Lemmas are recovered with
    :func:`cagf.data.apply_edit_script` under the predicted UPOS, so the
    case-restoration rules (acronyms -> upper, PROPN -> capitalised, otherwise
    lower) match the CAGF pipeline exactly and the resulting CoNLL-U scores
    identically under the official ``conll18_ud_eval`` scorer.

    The loader is built with ``shuffle=False`` so the per-sentence output order
    matches ``sentences`` regardless of batching.
    """
    from .data import apply_edit_script
    model.eval()
    loader = make_hf_loader(sentences, vocabs, batch_size, shuffle=False)
    per_sentence: list[dict] = []
    sent_idx = 0
    for batch in loader:
        word_strings = batch['word_strings']
        mask = batch['mask'].to(device)
        out = model(word_strings, mask, upos_ids=None)
        lemma_pred = out['lemma_logits'].argmax(dim=-1)
        gram_pred = (torch.sigmoid(out['grammeme_logits']) > 0.5).float()
        upos_pred = out['upos_pred']
        mask_cpu = mask.cpu()
        batch_size = len(word_strings)
        for b in range(batch_size):
            length = int(mask_cpu[b].sum().item())
            sent = sentences[sent_idx]
            sent_idx += 1
            lemmas: list[str] = []
            uposes: list[str] = []
            feats_list: list[list[str]] = []
            for i in range(length):
                rule_id = int(lemma_pred[b, i].item())
                rule = vocabs.lemma_rule_vocab.decode(rule_id)
                form = sent.tokens[i].form
                if isinstance(upos_pred, list):
                    upos_id = int(upos_pred[b][i])
                else:
                    upos_id = int(upos_pred[b, i].item())
                upos_str = vocabs.upos_vocab.decode(upos_id)
                decoded = apply_edit_script(form, rule, upos=upos_str)
                lemmas.append(decoded if decoded else form)
                uposes.append(upos_str)
                active = gram_pred[b, i].cpu().tolist()
                feats = [vocabs.grammeme_vocab.decode(g) for g, a in enumerate(active)
                         if a > 0.5 and g < len(vocabs.grammeme_vocab)]
                feats = [f for f in feats if f and f != '<unk>']
                feats_list.append(feats)
            per_sentence.append({'lemma': lemmas, 'upos': uposes, 'feats': feats_list})
    return per_sentence


def train_hf_one_run(train_sentences: list[Sentence], dev_sentences: list[Sentence],
                     test_sentences: list[Sentence], vocabs: CorpusVocabs,
                     model_name: str, revision: str, seed: int = 42,
                     max_epochs: int = 40, batch_size: int = 32,
                     learning_rate: float = 1e-5, weight_decay: float = 0.01,
                     grad_clip_norm: float = 5.0, early_stopping_patience: int = 5,
                     use_crf: bool = True, freeze_encoder: bool = False,
                     init_encoder_state: Optional[dict] = None, device: Optional[str] = None,
                     verbose: bool = True, return_model: bool = False,
                     save_checkpoint_path: Optional[str] = None,
                     encoder_cache_dir: Optional[str] = None,
                     config_name: Optional[str] = None) -> RunResult | tuple:
    """Train :class:`HFMorphModel` and evaluate on the test split.

    Mirrors :func:`cagf.train_loop.train_one_run`'s contract: same optimizer
    (AdamW over model + uncertainty-weighter params), same ReduceLROnPlateau
    scheduler (``mode='max', factor=0.5, patience=5``), same early stopping on
    dev macro-F1 mean, same :class:`UncertaintyWeightedLoss` over the three
    task losses, same grad-norm clipping, same checkpoint schema. The only
    difference is that the forward call is
    ``model(word_strings, mask, upos_ids=upos_ids)`` -- the HF encoder consumes
    the raw surface forms.

    Parameters
    ----------
    init_encoder_state:
        If given, loads ONLY the encoder weights into ``model.encoder``
        (``strict=False``). This is the silver-pretrained encoder reuse path
        (K4-B / O4): pretrain the encoder on the silver corpus once, then
        initialise each fold's heads freshly from the fold's vocab and
        fine-tune. Head tensors that do not match are left randomly
        initialised, so folds with different vocab sizes stay compatible.
    config_name:
        Name recorded on the returned :class:`RunResult`. Defaults to the
        ``model_name`` (the HF checkpoint identifier); the CV runner overrides
        this with the experiment config name (``gold_only`` /
        ``silver_finetune`` / ...).
    return_model:
        When ``True``, returns ``(RunResult, model)`` with the best dev-epoch
        state loaded, so the caller can immediately run :func:`predict_hf`
        without reloading from a checkpoint.

    Returns
    -------
    RunResult
        With ``config_name``, ``seed``, ``lemma`` / ``upos`` / ``grammeme``
        metrics (computed on ``test_sentences``), ``best_epoch`` and
        ``history`` -- the same fields the CAGF runner returns.
    """
    device = device or pick_device()
    set_seed(seed)
    train_loader = make_hf_loader(train_sentences, vocabs, batch_size, shuffle=True)
    dev_loader = make_hf_loader(dev_sentences, vocabs, batch_size, shuffle=False)
    test_loader = make_hf_loader(test_sentences, vocabs, batch_size, shuffle=False)

    model = HFMorphModel(
        num_upos=len(vocabs.upos_vocab), num_grammemes=len(vocabs.grammeme_vocab),
        num_lemma_rules=len(vocabs.lemma_rule_vocab), model_name=model_name,
        revision=revision, hidden_dim=256, use_crf=use_crf,
        freeze_encoder=freeze_encoder, encoder_cache_dir=encoder_cache_dir).to(device)

    if init_encoder_state is not None:
        # Silver-pretrained encoder reuse: load only the tensors that fall
        # under ``model.encoder`` and whose name AND shape match. Head tensors
        # (upos_head / lemma_head / grammeme_head / crf) are left untouched so
        # folds whose fold.train vocab differs from the silver vocab still
        # train cleanly.
        #
        # Key-name reconciliation: the save side filters
        # ``k.startswith("encoder.")`` on the FULL model state dict, producing
        # keys like ``encoder.encoder.embeddings.*`` (double ``encoder.``:
        # HFMorphModel.encoder → HFWordEncoder.encoder). The load side uses
        # ``model.encoder.state_dict()`` which has keys ``encoder.embeddings.*``
        # (single prefix). Strip one ``encoder.`` level so the names align.
        own_encoder = model.encoder.state_dict()
        normalized_init = {}
        for k, v in init_encoder_state.items():
            # strip leading "encoder." once so keys align with model.encoder
            nk = k[len("encoder."):] if k.startswith("encoder.") else k
            normalized_init[nk] = v
        compatible = {k: v for k, v in normalized_init.items()
                      if k in own_encoder and own_encoder[k].shape == v.shape}
        own_encoder.update(compatible)
        model.encoder.load_state_dict(own_encoder, strict=False)
        if verbose:
            print(f'  loaded {len(compatible)}/{len(own_encoder)} encoder tensors '
                  f'from init_encoder_state for fine-tuning')

    loss_weighter = UncertaintyWeightedLoss(num_tasks=3).to(device)
    params = list(model.parameters()) + list(loss_weighter.parameters())
    optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5)

    best_score = -1.0
    best_state: Optional[dict] = None
    best_epoch = -1
    epochs_without_improvement = 0
    history: list = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            word_strings = batch['word_strings']
            mask = batch['mask'].to(device)
            upos_ids = batch['upos_ids'].to(device)
            lemma_ids = batch['lemma_rule_ids'].to(device)
            gram_targets = batch['grammeme_targets'].to(device)
            optimizer.zero_grad()
            out = model(word_strings, mask, upos_ids=upos_ids)
            lemma_loss = masked_lemma_ce(out['lemma_logits'], lemma_ids)
            gram_loss = masked_multilabel_bce(out['grammeme_logits'], gram_targets, mask)
            upos_loss = out['upos_nll']
            total_loss, _ = loss_weighter([lemma_loss, upos_loss, gram_loss])
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(params, grad_clip_norm)
            optimizer.step()
            epoch_loss += float(total_loss.item())
            n_batches += 1
        dev_metrics = evaluate_hf(model, dev_loader, device)
        dev_score = (dev_metrics['lemma']['f1'] + dev_metrics['upos']['f1']
                     + dev_metrics['grammeme']['f1']) / 3.0
        scheduler.step(dev_score)
        history.append({'epoch': epoch, 'train_loss': epoch_loss / max(n_batches, 1),
                        'dev_macro_f1_mean': dev_score})
        if verbose:
            tag = config_name or model_name
            print(f'  [{tag} seed={seed}] epoch {epoch}: '
                  f'train_loss={epoch_loss / max(n_batches, 1):.4f} dev_f1={dev_score:.4f}')
        if dev_score > best_score:
            best_score = dev_score
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                if verbose:
                    print(f'  early stopping at epoch {epoch} (best epoch {best_epoch})')
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate_hf(model, test_loader, device)
    if save_checkpoint_path is not None:
        Path(save_checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_name': model_name, 'revision': revision,
            'num_upos': len(vocabs.upos_vocab),
            'num_grammemes': len(vocabs.grammeme_vocab),
            'num_lemma_rules': len(vocabs.lemma_rule_vocab),
            'use_crf': use_crf, 'freeze_encoder': freeze_encoder,
            'hidden_dim': 256,
            'test_metrics': test_metrics, 'seed': seed, 'best_epoch': best_epoch,
        }, save_checkpoint_path)
        if verbose:
            print(f'  Saved checkpoint to {save_checkpoint_path}')
    result = RunResult(
        config_name=config_name or model_name, seed=seed,
        lemma=test_metrics['lemma'], upos=test_metrics['upos'],
        grammeme=test_metrics['grammeme'], best_epoch=best_epoch, history=history)
    if return_model:
        return result, model
    return result
