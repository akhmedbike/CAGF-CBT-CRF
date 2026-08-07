"""Integration tests for the HuggingFace KazRoBERTa encoder.

These are marked ``slow`` because they load the real
``kz-transformers/kaz-roberta-conversational`` checkpoint (~300MB, ~83.5M
params) from the HuggingFace cache. Run with::

    pytest tests/test_hf_encoder.py -v -s -m slow

Skip with ``-m 'not slow'`` (the default in fast CI loops).

Coverage:
- output shape matches the ``CAGFCBTCRF.encode()`` contract
  ``(batch, max_len, proj_dim)``;
- a sentence with a multi-subtoken word still produces exactly ``n_words``
  positions (first-subword pooling, not one-per-subtoken);
- padded positions in the output are exactly zero (so heads see a neutral
  input where ``mask`` is False);
- the precompute-tokenization fast path yields the same forward result as the
  per-step tokenization path (determinism up to float epsilon in eval mode).
"""
from __future__ import annotations
import sys
from pathlib import Path

import pytest
import torch

# Make the repo root importable when pytest is run from anywhere.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cagf.data import Sentence, Token
from cagf.device import pick_device
from cagf.hf_encoder import HFWordEncoder
from cagf.hf_model import HFMorphModel

MODEL_NAME = 'kz-transformers/kaz-roberta-conversational'
REVISION = '43077c2fd0a163487ed468b5ec3b8750686a5888'
pytestmark = pytest.mark.slow


def _make_sentences():
    """Two short Kazakh sentences for batched tests.

    The second sentence contains 'тәуелсіздікке' (a long word) to exercise the
    multi-subtoken aggregation path -- under KazRoBERTa's BPE this word may
    split into >1 subtoken, which is exactly what first-subword pooling has to
    handle.
    """
    s1 = Sentence(tokens=[
        Token(form='Мен', lemma='мен', upos='PRON', feats='_'),
        Token(form='мектепке', lemma='мектеп', upos='NOUN', feats='_'),
        Token(form='барамын', lemma='бар', upos='VERB', feats='_'),
    ])
    s2 = Sentence(tokens=[
        Token(form='Ол', lemma='ол', upos='PRON', feats='_'),
        Token(form='тәуелсіздікке', lemma='тәуелсіздік', upos='NOUN', feats='_'),
    ])
    return [s1, s2]


def _batch_strings(sentences):
    return [[t.form for t in s.tokens] for s in sentences]


def _mask_for(sentences, max_len=None):
    lengths = [len(s.tokens) for s in sentences]
    if max_len is None:
        max_len = max(lengths)
    mask = torch.zeros(len(sentences), max_len, dtype=torch.bool)
    for b, l in enumerate(lengths):
        mask[b, :l] = True
    return mask


@pytest.fixture(scope='module')
def encoder():
    """Module-scoped fixture so we only pay the model-load cost once."""
    device = pick_device()
    enc = HFWordEncoder(model_name=MODEL_NAME, revision=REVISION, proj_dim=256)
    enc.to(device)
    enc.eval()
    return enc


def test_output_shape(encoder):
    """Output is ``(B, max_len, proj_dim)`` matching the encode() contract."""
    sents = _make_sentences()
    word_strings = _batch_strings(sents)
    mask = _mask_for(sents).to(encoder.device)
    with torch.no_grad():
        out = encoder(word_strings, mask)
    assert out.shape == (2, mask.size(1), 256)
    assert out.dtype == torch.float32


def test_first_subword_pooling(encoder):
    """A multi-subtoken word still collapses to exactly one output position.

    We assert the strong contract: the number of unpadded positions in the
    output equals the number of UD words in the sentence (NOT the number of
    BPE subtokens). This is what guarantees the per-word heads can index the
    encoder output positionally.
    """
    sents = _make_sentences()
    word_strings = _batch_strings(sents)
    mask = _mask_for(sents).to(encoder.device)
    with torch.no_grad():
        out = encoder(word_strings, mask)
    # For each sentence, the number of non-zero (in norm) positions should
    # equal the number of UD words, because padded positions are zeroed.
    norms = out.norm(dim=-1)  # (B, max_len)
    for b, sent in enumerate(sents):
        n_words = len(sent.tokens)
        n_nonzero = int((norms[b] > 0).sum().item())
        assert n_nonzero == n_words, (
            f'sentence {b}: expected {n_words} word positions, got {n_nonzero}'
        )
    # Also: total subtokens for the batch (with specials) is strictly greater
    # than total words, confirming we are actually aggregating.
    total_words = sum(len(s.tokens) for s in sents)
    total_subtok = 0
    for words in word_strings:
        enc_ids = encoder.tokenizer(words, is_split_into_words=True, add_special_tokens=True)['input_ids']
        total_subtok += len(enc_ids)
    assert total_subtok > total_words, 'test setup degenerate: no multi-subtoken word present'


def test_padded_positions_zero(encoder):
    """Positions where ``mask`` is False must be exactly zero in the output."""
    sents = _make_sentences()
    word_strings = _batch_strings(sents)
    # Pad to 8 even though the longest sentence has only 3 words, so there are
    # at least 5 padded columns to check.
    mask = _mask_for(sents, max_len=8).to(encoder.device)
    with torch.no_grad():
        out = encoder(word_strings, mask)
    assert out.shape == (2, 8, 256)
    pad_norms = out.norm(dim=-1)[~mask]
    assert torch.all(pad_norms == 0.0), (
        f'padded positions should be exactly zero, max norm = {float(pad_norms.max())}'
    )


def test_cache_matches_direct(encoder):
    """Precomputed tokenization must produce the same forward as direct.

    This validates optimization O1 (pre-tokenize-and-cache): we tokenize the
    corpus once via ``precompute_tokenization`` and reuse the aligned lists in
    forward() instead of re-invoking the tokenizer. The two paths must agree
    to within float epsilon, since they feed the SAME subtoken ids into the
    SAME encoder -- only the construction path differs.

    NB: requires ``eval()`` mode so dropout is disabled; otherwise the two
    forward passes would sample independent dropout masks and diverge.
    """
    sents = _make_sentences()
    word_strings = _batch_strings(sents)
    mask = _mask_for(sents).to(encoder.device)
    cache = encoder.precompute_tokenization(sents)
    # Sanity: the cache schema is exactly the documented four aligned lists.
    assert set(cache.keys()) == {'input_ids', 'attention_mask', 'word_to_first_subtok', 'lengths'}
    assert len(cache['input_ids']) == 2
    # Lengths must match the UD word counts.
    assert cache['lengths'] == [len(s.tokens) for s in sents]
    with torch.no_grad():
        out_direct = encoder(word_strings, mask)
        out_cached = encoder(word_strings, mask, cached=cache)
    # Same subtoken ids -> identical encoder inputs -> identical outputs (eval
    # mode disables dropout, so there is no stochasticity).
    assert torch.allclose(out_direct, out_cached, atol=1e-6), (
        f'cache vs direct divergence: max abs diff = '
        f'{float((out_direct - out_cached).abs().max())}'
    )


def test_hf_morph_model_forward_shapes(encoder):
    """HFMorphModel produces all the CAGFCBTCRF output keys with correct shapes.

    This is the integration check that the HF encoder + the three task heads +
    the optional CRF compose into the same forward contract the train loop
    expects from CAGFCBTCRF, so train_loop can swap models with minimal glue.
    """
    sents = _make_sentences()
    word_strings = _batch_strings(sents)
    mask = _mask_for(sents).to(encoder.device)
    device = encoder.device
    model = HFMorphModel(num_upos=17, num_grammemes=50, num_lemma_rules=100, model_name=MODEL_NAME, revision=REVISION, hidden_dim=256, use_crf=True)
    # Reuse the already-loaded encoder weights so we don't reload the 300MB
    # checkpoint a second time inside this module-scoped test session.
    model.encoder.load_state_dict(encoder.state_dict())
    model.to(device)
    model.eval()
    upos_ids = torch.zeros(2, mask.size(1), dtype=torch.long, device=device)
    upos_ids[0, :3] = torch.tensor([1, 2, 3])
    upos_ids[1, :2] = torch.tensor([4, 5])
    with torch.no_grad():
        out = model(word_strings, mask, upos_ids=upos_ids)
    for k in ('lemma_logits', 'grammeme_logits', 'upos_emissions', 'upos_nll', 'upos_pred'):
        assert k in out, f'missing output key: {k}'
    assert out['lemma_logits'].shape == (2, mask.size(1), 100)
    assert out['grammeme_logits'].shape == (2, mask.size(1), 50)
    assert out['upos_emissions'].shape == (2, mask.size(1), 17)
    assert torch.isfinite(out['upos_nll'])
    # CRF mode -> upos_pred is a list of per-sentence tag lists.
    assert isinstance(out['upos_pred'], list)
    assert len(out['upos_pred']) == 2
    assert len(out['upos_pred'][0]) == 3
    assert len(out['upos_pred'][1]) == 2
