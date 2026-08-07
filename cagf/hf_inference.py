"""Inference helpers for :class:`HFMorphModel` (KazRoBERTa-backed tagger).

This is the HuggingFace analogue of :mod:`cagf.inference`: it reconstructs a
trained :class:`HFMorphModel` from a checkpoint saved by
:func:`cagf.hf_train_loop.train_hf_one_run` and exposes an ``analyze_hf_tokens``
entry point with the SAME output contract (:class:`TokenAnalysis`) as
:func:`cagf.inference.analyze_tokens`.

Sharing the ``TokenAnalysis`` shape lets the web demo render either backend
(CAGF-CBT+CRF or KazRoBERTa) with a single template: the only difference
between the two inference paths is how the encoder consumes the input -- the
CAGF model wants pre-bucketed ``word_ids`` / ``char_ids`` tensors, while the HF
model consumes the raw surface-form strings (its tokenizer does the subword
segmentation). Everything downstream (PAD/UNK filtering, grammeme thresholding,
confidence readout) is identical because the two model families share the same
three task heads and the same :class:`LinearChainCRF` decoder.
"""
from __future__ import annotations
import torch
from .data import CorpusVocabs, PAD, UNK
from .hf_model import HFMorphModel
from .inference import TokenAnalysis  # reused so the renderer is shared


def load_hf_checkpoint(checkpoint_path: str, vocabs_path: str, device: str = 'cpu'):
    """Reconstruct a trained :class:`HFMorphModel` from a checkpoint + vocabs.

    Reads the HF checkpoint schema written by
    :func:`cagf.hf_train_loop.train_hf_one_run` (``model_state_dict``,
    ``model_name``, ``revision``, ``num_upos``, ``num_grammemes``,
    ``num_lemma_rules``, ``use_crf``, ``hidden_dim``) and the matching
    ``CorpusVocabs`` JSON, rebuilds the model with the SAME constructor
    arguments it was trained with, and loads the weights.

    Mirrors :func:`cagf.inference.load_checkpoint`'s ``(model, vocabs, meta)``
    return contract; the ``meta`` dict additionally carries ``model_family='hf'``
    so a downstream consumer (e.g. the web demo) can brand itself correctly
    without re-inferring the backend from the checkpoint keys.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    vocabs = CorpusVocabs.load(vocabs_path)
    model = HFMorphModel(
        num_upos=ckpt['num_upos'],
        num_grammemes=ckpt['num_grammemes'],
        num_lemma_rules=ckpt['num_lemma_rules'],
        model_name=ckpt['model_name'],
        revision=ckpt['revision'],
        hidden_dim=ckpt.get('hidden_dim', 256),
        use_crf=ckpt.get('use_crf', True),
        freeze_encoder=ckpt.get('freeze_encoder', False),
        encoder_cache_dir=ckpt.get('encoder_cache_dir'),
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    meta = {
        'test_metrics': ckpt.get('test_metrics'),
        'seed': ckpt.get('seed'),
        'best_epoch': ckpt.get('best_epoch'),
        'model_name': ckpt.get('model_name'),
        'revision': ckpt.get('revision'),
        'model_family': 'hf',
        'ablation_name': 'kazroberta',
        'train_sentences': ckpt.get('train_sentences'),
        'dev_sentences': ckpt.get('dev_sentences'),
        'test_sentences': ckpt.get('test_sentences'),
    }
    return (model, vocabs, meta)


def analyze_hf_tokens(tokens: list[str], model: HFMorphModel,
                      vocabs: CorpusVocabs, grammeme_threshold: float = 0.5) -> list[TokenAnalysis]:
    """Tag a single sentence (a list of surface-form strings) with an HF model.

    Returns a list of :class:`TokenAnalysis` -- the SAME dataclass
    :func:`cagf.inference.analyze_tokens` returns -- so the demo's renderer
    works for both backends unchanged.

    The HF encoder consumes the raw surface forms (``word_strings_batch``):
    ``model([[...forms...]], mask, upos_ids=None)``. The HF tokenizer handles
    subword segmentation internally; there are no ``word_ids`` / ``char_ids``
    to build here, in contrast to the CAGF path.
    """
    if not tokens:
        return []
    device = next(model.parameters()).device
    word_strings = [list(tokens)]  # HF contract: list-of-lists, one per sentence
    max_len = len(tokens)
    mask = torch.ones(1, max_len, dtype=torch.bool, device=device)
    with torch.no_grad():
        out = model(word_strings, mask, upos_ids=None)
    lemma_pred = out['lemma_logits'].argmax(dim=-1)[0].tolist()
    gram_probs = torch.sigmoid(out['grammeme_logits'])[0]
    upos_pred = out['upos_pred']
    # CRF -> list of per-sentence tag lists; softmax -> tensor. Normalise both
    # to a flat list of int tag ids for this single-sentence call.
    if isinstance(upos_pred, list):
        upos_ids = upos_pred[0]
    else:
        upos_ids = upos_pred[0].tolist()
    emission_softmax = torch.softmax(out['upos_emissions'], dim=-1)[0]

    # PAD/UNK occupy real class indices in upos_vocab / grammeme_vocab
    # (num_upos == len(upos_vocab) includes the PAD special token). Same hazard
    # as in the CAGF path: the CRF / argmax can select the PAD index for a
    # genuine token and otherwise leak the literal "<pad>" into the UI. Filter
    # explicitly rather than relying on it never happening.
    upos_pad_idx = vocabs.upos_vocab.stoi.get(PAD)
    grammeme_unk_idx = vocabs.grammeme_vocab.stoi.get(UNK)

    results = []
    for i, form in enumerate(tokens):
        lemma_rule = vocabs.lemma_rule_vocab.decode(lemma_pred[i]) if lemma_pred[i] < len(vocabs.lemma_rule_vocab) else '<unk>'
        raw_upos_id = upos_ids[i]
        upos_is_known = raw_upos_id < len(vocabs.upos_vocab) and raw_upos_id != upos_pad_idx
        upos = vocabs.upos_vocab.decode(raw_upos_id) if upos_is_known else '<unk>'
        grams = [vocabs.grammeme_vocab.decode(j) for j in range(len(vocabs.grammeme_vocab))
                 if j != grammeme_unk_idx and gram_probs[i, j].item() > grammeme_threshold]
        gram_prob_map = {g: round(gram_probs[i, vocabs.grammeme_vocab.stoi[g]].item(), 4) for g in grams}
        emission_conf = round(emission_softmax[i, raw_upos_id].item(), 4) if raw_upos_id < emission_softmax.shape[-1] else 0.0
        results.append(TokenAnalysis(form=form, lemma_rule_guess=lemma_rule, upos=upos,
                                     grammemes=grams, upos_known=upos_is_known,
                                     upos_emission_confidence=emission_conf,
                                     grammeme_probs=gram_prob_map))
    return results
