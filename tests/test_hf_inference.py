"""Tests for the HF inference layer (``cagf.hf_inference``).

These mirror the load-and-analyze path the web demo uses for the KazRoBERTa
backend. They are skipped unless a trained HF interface checkpoint is present
(``models/interface_hf_model.pt`` + ``models/interface_hf_vocabs.json``),
exactly as ``tests/test_webapp.py`` skips on a missing CAGF checkpoint. Train
one with::

    python scripts/train_hf_for_interface.py
    # or, from a CV run:
    python scripts/run_cv_kazroberta.py --configs silver_finetune \
        --save-interface-checkpoint models/interface_hf_model.pt \
        --save-interface-vocabs   models/interface_hf_vocabs.json

They are ALSO marked ``slow`` because loading the 83.5M-param KazRoBERTa
encoder dominates runtime; skip with ``-m 'not slow'`` in fast CI loops.
"""
from __future__ import annotations
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CHECKPOINT = 'models/interface_hf_model.pt'
VOCABS = 'models/interface_hf_vocabs.json'

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not (Path(CHECKPOINT).exists() and Path(VOCABS).exists()),
        reason='no trained HF interface checkpoint -- run '
               'scripts/train_hf_for_interface.py first'),
]


@pytest.fixture(scope='module')
def backend():
    from cagf.hf_inference import load_hf_checkpoint
    model, vocabs, meta = load_hf_checkpoint(CHECKPOINT, VOCABS)
    return model, vocabs, meta


def test_load_returns_hf_model_and_family(backend):
    """The loaded model is an HFMorphModel and meta reports the HF family."""
    from cagf.hf_model import HFMorphModel
    model, vocabs, meta = backend
    assert isinstance(model, HFMorphModel)
    assert meta['model_family'] == 'hf'
    assert meta.get('model_name')
    # Vocabs must round-trip: upos/grammeme/lemma_rule spaces are non-empty.
    assert len(vocabs.upos_vocab) > 0
    assert len(vocabs.grammeme_vocab) > 0
    assert len(vocabs.lemma_rule_vocab) > 0


def test_analyze_returns_tokenanalysis_shape(backend):
    """analyze_hf_tokens returns the same TokenAnalysis shape as the CAGF path."""
    from cagf.inference import TokenAnalysis
    model, vocabs, _ = backend
    tokens = ['Бүгін', 'студенттер', 'мектепке', 'барды']
    results = __import__('cagf.hf_inference', fromlist=['analyze_hf_tokens']).analyze_hf_tokens(
        tokens, model, vocabs)
    assert len(results) == len(tokens)
    assert all(isinstance(r, TokenAnalysis) for r in results)
    # Forms are echoed verbatim (the model does not rewrite surface forms).
    assert [r.form for r in results] == tokens
    # Each result has a string UPOS and a float confidence in [0, 1].
    for r in results:
        assert isinstance(r.upos, str) and r.upos
        assert 0.0 <= r.upos_emission_confidence <= 1.0


def test_empty_input_returns_empty(backend):
    """Empty token list is a no-op (mirrors the CAGF path's guard)."""
    from cagf.hf_inference import analyze_hf_tokens
    model, vocabs, _ = backend
    assert analyze_hf_tokens([], model, vocabs) == []


def test_checkpoint_schema_matches_loader(backend):
    """The on-disk checkpoint keys are exactly what load_hf_checkpoint reads."""
    import torch
    ckpt = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    for key in ('model_state_dict', 'model_name', 'revision', 'num_upos',
                'num_grammemes', 'num_lemma_rules', 'use_crf', 'hidden_dim'):
        assert key in ckpt, f'checkpoint missing key: {key}'
