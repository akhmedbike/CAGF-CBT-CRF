"""Web demo tests for the KazRoBERTa (HF) backend.

These verify the dual-backend wiring in ``webapp/app.py``: when an HF interface
checkpoint is present, the same routes serve it, the page brands itself as
KazRoBERTa, and the shared TokenAnalysis rendering works for the HF path too.

Skipped when no HF checkpoint is on disk (mirroring the CAGF skip guard in
``tests/test_webapp.py``) and marked ``slow`` because loading KazRoBERTa is
heavy. The existing CAGF-focused ``tests/test_webapp.py`` stays unchanged and
remains the primary demo contract; this file layers HF on top without
weakening those assertions.
"""
from __future__ import annotations
import io
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

HF_CHECKPOINT = 'models/interface_hf_model.pt'
HF_VOCABS = 'models/interface_hf_vocabs.json'

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not (Path(HF_CHECKPOINT).exists() and Path(HF_VOCABS).exists()),
        reason='no trained HF interface checkpoint -- run '
               'scripts/train_hf_for_interface.py first'),
]


@pytest.fixture()
def hf_client(monkeypatch):
    """A test client with the app pointed ONLY at the HF checkpoint.

    Temporarily moves the CAGF checkpoint out of the way (by rebinding the
    module-level CAGF paths to nonexistent files) so ``_load_backend`` cannot
    pick CAGF first, then forces a reload so the HF backend is active. The
    fixture restores everything on teardown.
    """
    import webapp.app as wa
    # Point both env-derived paths; reload picks HF when CAGF is unavailable.
    monkeypatch.setattr(wa, 'CAGF_CHECKPOINT', '/nonexistent/cagf.pt', raising=True)
    monkeypatch.setattr(wa, 'CAGF_VOCABS', '/nonexistent/cagf.json', raising=True)
    monkeypatch.setattr(wa, 'HF_CHECKPOINT', HF_CHECKPOINT, raising=True)
    monkeypatch.setattr(wa, 'HF_VOCABS', HF_VOCABS, raising=True)
    wa._load_backend()
    yield wa.app.test_client()
    # Restore the real backend for any subsequent test module.
    wa._load_backend()


def test_hf_backend_is_loaded(hf_client):
    """The app reports MODEL_FAMILY='hf' on the page."""
    import webapp.app as wa
    assert wa.MODEL_FAMILY == 'hf'
    assert wa.MODEL is not None


def test_hf_index_get_brands_kazroberta(hf_client):
    """GET / still contains the project label AND now the KazRoBERTa badge."""
    r = hf_client.get('/')
    assert r.status_code == 200
    html = r.get_data(as_text=True)
    assert 'CAGF-CBT+CRF' in html  # project title stays
    assert 'KazRoBERTa' in html    # backend badge is HF


def test_hf_analyze_raw_text(hf_client):
    """POST raw text produces a 200 page echoing the input tokens."""
    r = hf_client.post('/', data={'raw_text': 'Бүгін студенттер мектепке барды.'})
    assert r.status_code == 200
    html = r.get_data(as_text=True)
    assert 'студенттер' in html
    assert 'Error' not in html


def test_hf_confidence_thresholds_unchanged():
    """Confidence bucketing is a pure function independent of backend."""
    from webapp.app import _confidence_css_class
    assert _confidence_css_class(0.95) == 'conf-high'
    assert _confidence_css_class(0.8) == 'conf-high'
    assert _confidence_css_class(0.6) == 'conf-mid'
    assert _confidence_css_class(0.5) == 'conf-mid'
    assert _confidence_css_class(0.486) == 'conf-low'
    assert _confidence_css_class(0.1) == 'conf-low'


def test_hf_charts_shown_for_long_input(hf_client):
    """The >=2-sentence chart gate works for the HF backend too."""
    long_text = 'Бүгін студенттер мектепке барды. Олар кітап оқыды және дәріс тыңдады.'
    r = hf_client.post('/', data={'raw_text': long_text})
    html = r.get_data(as_text=True)
    assert 'uposDistChart' in html
    assert 'confHistChart' in html
    assert 'grammemeDistChart' in html
    assert 'exportConllu' in html
    assert 'data-upos=' in html
    assert 'data-conf=' in html


def test_hf_empty_input_returns_error(hf_client):
    """Empty input still surfaces the literal 'Error' string for the HF backend."""
    r = hf_client.post('/', data={'raw_text': ''})
    assert r.status_code == 200
    assert 'Error' in r.get_data(as_text=True)
