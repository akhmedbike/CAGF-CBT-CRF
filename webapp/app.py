"""Standalone Flask demo for the Kazakh morphological tagger.

Dual-backend: serves whichever trained model is available at startup -- the
CAGF-CBT+CRF model (:class:`cagf.model.CAGFCBTCRF`, loaded via
:func:`cagf.inference.load_checkpoint`) or the KazRoBERTa model
(:class:`cagf.hf_model.HFMorphModel`, loaded via
:func:`cagf.hf_inference.load_hf_checkpoint`). Both backends return the same
:class:`cagf.inference.TokenAnalysis` shape, so a single template renders
either.

The backend is auto-detected from the checkpoint keys: CAGF checkpoints store
``'hparams'`` (the model hyperparameters), HF checkpoints store
``'model_name'`` (the HuggingFace identifier). ``MODEL_FAMILY`` exposes the
result ('cagf' / 'hf' / 'none') so the page can brand itself.

This module is intentionally self-contained (no blueprint / factory app): the
test contract in ``tests/test_webapp.py`` imports module-level attributes
(``app``, ``MODEL``, ``VOCABS``, ``META``, ``ABLATION_RESULTS_PATH``) and
mutates them directly. Keep those names stable.
"""
from __future__ import annotations
import json
import math
import os
import re
from collections import Counter
from pathlib import Path

from flask import Flask, jsonify, render_template, request

# Default checkpoint / vocab locations. The CAGF pair is produced by
# scripts/train_for_interface.py; the HF pair by scripts/train_hf_for_interface.py.
# Either may be absent -- the app still serves the form page and reports
# MODEL_FAMILY='none' so the user sees an explicit "no model loaded" state
# rather than a 500.
CAGF_CHECKPOINT = os.environ.get('CAGF_CHECKPOINT', 'models/interface_model.pt')
CAGF_VOCABS = os.environ.get('CAGF_VOCABS', 'models/interface_vocabs.json')
HF_CHECKPOINT = os.environ.get('HF_CHECKPOINT', 'models/interface_hf_model.pt')
HF_VOCABS = os.environ.get('HF_VOCABS', 'models/interface_hf_vocabs.json')
# Aggregated ablation run results (CAGF 5x5 grid). The /training_status endpoint
# reads this fresh on every call so the test harness can rebind it.
ABLATION_RESULTS_PATH = Path(os.environ.get('ABLATION_RESULTS_PATH', 'results/ablation_raw.json'))
# 5 ablation configs (full_model, wo_character_encoder, wo_gated_fusion, wo_crf,
# transformer_only) x 5 seeds ([13, 42, 123, 777, 2026]) -- see
# scripts/run_ablation.py and configs/default.yaml. The /training_status
# "expected" count is a constant derived from this grid.
EXPECTED_TRAINING_RUNS = 25

app = Flask(__name__)
# Loaded lazily / overridable by tests. ``MODEL is None`` => no backend loaded.
MODEL = None
VOCABS = None
META: dict | None = None
MODEL_FAMILY = 'none'  # 'cagf', 'hf', or 'none'

# ---------------------------------------------------------------------------
# Tokenisation / sentence splitting (shared by both backends).
# ---------------------------------------------------------------------------

# Match runs of Unicode letters / apostrophes -- this covers Cyrillic (Kazakh)
# and Latin word forms. Punctuation becomes its own token so sentence boundaries
# can still be detected, but only letter-runs are sent to the model.
_TOKEN_RE = re.compile(r"[^\W\d_]+|[^\s]", re.UNICODE)
# Sentence-final punctuation (Latin + Cyrillic ellipsis). A sentence is a run
# of tokens up to and including one of these.
_SENT_END_RE = re.compile(r'[.!?…]+')

# Charts (UPOS distribution, confidence histogram, grammeme distribution) are
# only meaningful once there is enough input to aggregate. A single short
# sentence produces a trivial bar chart, so we gate the charts block on having
# at least this many analysed sentences.
MIN_SENTENCES_FOR_CHARTS = 2


def _tokenise(text: str) -> list[str]:
    """Split ``text`` into word / punctuation tokens (Unicode-letter aware)."""
    return _TOKEN_RE.findall(text)


def _split_sentences(text: str) -> list[list[str]]:
    """Group tokens into sentences on sentence-final punctuation.

    Returns one list of word-tokens per sentence (punctuation dropped). A
    trailing run of tokens with no final punctuation still forms a sentence, so
    unterminated input is analysed rather than discarded.
    """
    tokens = _tokenise(text)
    sentences: list[list[str]] = []
    current: list[str] = []
    for tok in tokens:
        if _SENT_END_RE.fullmatch(tok):
            if current:
                sentences.append(current)
                current = []
        else:
            current.append(tok)
    if current:
        sentences.append(current)
    return sentences


# ---------------------------------------------------------------------------
# Backend loading + analysis.
# ---------------------------------------------------------------------------

def _detect_family(checkpoint_path: str) -> str:
    """Inspect a checkpoint's top-level keys to decide CAGF vs HF.

    CAGF checkpoints (cagf/train_loop.py) store ``'hparams'``; HF checkpoints
    (cagf/hf_train_loop.py) store ``'model_name'``. Returns 'cagf' / 'hf' /
    'none' (none = missing or unreadable).
    """
    if not os.path.exists(checkpoint_path):
        return 'none'
    try:
        import torch
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except Exception:
        return 'none'
    if isinstance(ckpt, dict):
        if 'model_name' in ckpt:
            return 'hf'
        if 'hparams' in ckpt:
            return 'cagf'
    return 'none'


def _load_backend():
    """Load whichever trained backend is on disk.

    CAGF is preferred when both exist (it is the lighter, always-available demo
    model and matches the original test contract). Sets the module-level
    ``MODEL`` / ``VOCABS`` / ``META`` / ``MODEL_FAMILY``. Safe to call when no
    checkpoint exists -- leaves everything at ``None`` / ``'none'``.
    """
    global MODEL, VOCABS, META, MODEL_FAMILY
    try:
        cagf_family = _detect_family(CAGF_CHECKPOINT)
        if cagf_family == 'cagf' and os.path.exists(CAGF_VOCABS):
            from cagf.inference import load_checkpoint
            MODEL, VOCABS, META = load_checkpoint(CAGF_CHECKPOINT, CAGF_VOCABS)
            MODEL_FAMILY = 'cagf'
            META = dict(META or {})
            META.setdefault('model_family', 'cagf')
            return
        hf_family = _detect_family(HF_CHECKPOINT)
        if hf_family == 'hf' and os.path.exists(HF_VOCABS):
            from cagf.hf_inference import load_hf_checkpoint
            MODEL, VOCABS, META = load_hf_checkpoint(HF_CHECKPOINT, HF_VOCABS)
            MODEL_FAMILY = 'hf'
            META = dict(META or {})
            META.setdefault('model_family', 'hf')
            return
    except Exception as exc:  # pragma: no cover - startup logging path
        # A failed load must NOT crash the server: the form page still renders
        # and the user sees the explicit "no model loaded" state.
        print(f'[webapp] backend load failed: {exc!r}')
    MODEL, VOCABS, META, MODEL_FAMILY = None, None, None, 'none'


def _analyze_sentence(words: list[str]) -> list:
    """Run the loaded backend over one sentence's surface forms.

    Dispatches to ``analyze_hf_tokens`` for HF / ``analyze_tokens`` for CAGF.
    Returns ``[]`` when no model is loaded (the caller renders an error).
    """
    if MODEL is None or VOCABS is None or not words:
        return []
    if MODEL_FAMILY == 'hf':
        from cagf.hf_inference import analyze_hf_tokens
        return analyze_hf_tokens(words, MODEL, VOCABS)
    from cagf.inference import analyze_tokens
    return analyze_tokens(words, MODEL, VOCABS)


def _predicted_lemma(form: str, rule: str):
    """Decode a lemma rule into a surface lemma, flagging unparsable rules.

    Returns ``(lemma, parsed)`` where ``parsed`` is False when the rule could
    not be interpreted (``apply_edit_script_verbose`` signals this). The UI
    renders unparsable lemmas as the literal token 'unparsed' rather than
    silently echoing the surface form, so a broken lemma prediction is visible
    to the user instead of masked.
    """
    from cagf.data import apply_edit_script_verbose
    try:
        lemma, valid = apply_edit_script_verbose(form, rule)
    except Exception:
        return (form, False)
    return (lemma, bool(valid))


# ---------------------------------------------------------------------------
# Confidence bucketing (pure function, unit-tested directly).
# ---------------------------------------------------------------------------

def _confidence_css_class(confidence: float) -> str:
    """Bucket a UPOS emission confidence into a CSS class.

    Boundaries (from the test contract):
      >= 0.8 -> 'conf-high'   (0.8 included; 0.95 included)
      >= 0.5 -> 'conf-mid'    (0.5 included; 0.6 included)
      <  0.5 -> 'conf-low'    (0.486, 0.1)
    """
    if confidence >= 0.8:
        return 'conf-high'
    if confidence >= 0.5:
        return 'conf-mid'
    return 'conf-low'


# ---------------------------------------------------------------------------
# Helpers for the token table + charts.
# ---------------------------------------------------------------------------

def _build_rows(sentences_analyses: list[list]) -> list[dict]:
    """Flatten per-sentence analyses into renderable token rows.

    Each row carries the surface form, decoded lemma (+ 'unparsed' flag), UPOS,
    grammemes, the raw emission confidence, and the CSS confidence class, plus
    ``data-upos`` / ``data-conf`` attributes used by the client-side filter /
    sort controls.
    """
    rows: list[dict] = []
    for sent_idx, analyses in enumerate(sentences_analyses):
        for tok_idx, ta in enumerate(analyses):
            lemma, parsed = _predicted_lemma(ta.form, ta.lemma_rule_guess)
            conf = float(ta.upos_emission_confidence or 0.0)
            rows.append({
                'form': ta.form,
                'lemma': lemma if parsed else 'unparsed',
                'parsed': parsed,
                'upos': ta.upos if ta.upos_known else '<unk>',
                'grammemes': ta.grammemes,
                'confidence': conf,
                'confidence_class': _confidence_css_class(conf),
                'sentence': sent_idx + 1,
                'token': tok_idx + 1,
            })
    return rows


def _aggregate(rows: list[dict]) -> dict:
    """Compute the UPOS / grammeme / confidence distributions for the charts."""
    upos_counts = Counter(r['upos'] for r in rows)
    grammeme_counts: Counter = Counter()
    for r in rows:
        for g in r['grammemes']:
            grammeme_counts[g] += 1
    conf_buckets = {'conf-high': 0, 'conf-mid': 0, 'conf-low': 0}
    for r in rows:
        conf_buckets[r['confidence_class']] += 1
    return {
        'upos_counts': dict(upos_counts.most_common()),
        'grammeme_counts': dict(grammeme_counts.most_common()),
        'conf_buckets': conf_buckets,
        'n_tokens': len(rows),
    }


def _to_conllu(rows: list[dict]) -> str:
    """Serialise the rendered rows back to a CoNLL-U string for export.

    One sentence block per stored ``sentence`` group; columns 3+ (lemma, UPOS,
    feats) populated from the predictions, unspecified columns left as ``_``.
    """
    lines: list[str] = []
    sent_groups: dict[int, list[dict]] = {}
    for r in rows:
        sent_groups.setdefault(r['sentence'], []).append(r)
    for sent_idx in sorted(sent_groups):
        lines.append(f'# sent_id = {sent_idx}')
        for tok_idx, r in enumerate(sent_groups[sent_idx], start=1):
            feats = '|'.join(sorted(r['grammemes'])) if r['grammemes'] else '_'
            lemma = r['lemma'] if r['parsed'] else '_'
            upos = r['upos'] if r['upos'] != '<unk>' else '_'
            lines.append('\t'.join([
                str(tok_idx), r['form'], lemma, upos, '_', feats, '_', '_', '_', '_',
            ]))
        lines.append('')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Input extraction (raw text vs .txt vs .conllu uploads).
# ---------------------------------------------------------------------------

def _extract_input() -> tuple[str, str | None]:
    """Return ``(text_or_forms, error)`` from the current request.

    Handles three input modes:
      * ``raw_text`` form field -- raw Kazakh text,
      * ``file`` upload named ``*.txt`` (or any non-conllu name) -- decoded as
        raw text,
      * ``file`` upload named ``*.conllu`` -- parsed as CoNLL-U and reduced to
        its surface forms joined into a pseudo-text (so the downstream
        sentence splitter sees the same word sequence).
    Returns ``(text, None)`` on success or ``('', 'error message')`` when there
    is no usable input.
    """
    raw = request.form.get('raw_text', '').strip()
    if raw:
        return (raw, None)
    upload = request.files.get('file')
    if upload is None or not upload.filename:
        # "Error" must appear literally in the page so a caller can detect the
        # empty-input state by string match (see tests/test_webapp.py).
        return ('', 'Error: no input -- paste text or upload a .txt / .conllu file.')
    filename = upload.filename.lower()
    data = upload.read()
    try:
        text = data.decode('utf-8')
    except UnicodeDecodeError:
        return ('', 'Error: upload is not valid UTF-8 text.')
    if filename.endswith('.conllu'):
        # Reuse the gold reader to extract surface forms only -- we re-tag them
        # with the loaded model rather than trusting the file's annotations.
        # read_conllu opens its argument as a path, so spool the upload to a
        # temp file rather than handing it a StringIO.
        import tempfile
        from cagf.data import read_conllu
        try:
            with tempfile.NamedTemporaryFile('w', suffix='.conllu', encoding='utf-8', delete=False) as tf:
                tf.write(text)
                tmp_path = tf.name
            sentences = read_conllu(tmp_path)
        except Exception:
            return ('', 'Error: could not parse the uploaded CoNLL-U file.')
        finally:
            try:
                os.unlink(tmp_path)
            except (OSError, NameError):
                pass
        forms = ' '.join(t.form for s in sentences for t in s.tokens)
        return (forms, None)
    return (text.strip(), None)


# ---------------------------------------------------------------------------
# Routes.
# ---------------------------------------------------------------------------

@app.route('/', methods=['GET', 'POST'])
def index():
    """Render the form (GET) or the analysis result (POST)."""
    if request.method == 'GET':
        return render_template('index.html', model_family=MODEL_FAMILY, meta=META,
                               result=None, error=None)

    text, error = _extract_input()
    if error or not text:
        return render_template('index.html', model_family=MODEL_FAMILY, meta=META,
                               result=None, error=error or 'Error: empty input.'), 200

    if MODEL is None:
        return render_template('index.html', model_family=MODEL_FAMILY, meta=META,
                               result=None, error='Error: no model is loaded.'), 200

    sentences = _split_sentences(text)
    if not sentences:
        return render_template('index.html', model_family=MODEL_FAMILY, meta=META,
                               result=None,
                               error='Error: no tokens found in input.'), 200

    analyses = [_analyze_sentence(words) for words in sentences]
    rows = _build_rows(analyses)
    aggregate = _aggregate(rows)
    show_charts = len(sentences) >= MIN_SENTENCES_FOR_CHARTS
    conllu_export = _to_conllu(rows)
    result = {
        'sentences': sentences,
        'n_sentences': len(sentences),
        'rows': rows,
        'aggregate': aggregate,
        'show_charts': show_charts,
        'conllu_export': conllu_export,
    }
    return render_template('index.html', model_family=MODEL_FAMILY, meta=META,
                           result=result, error=None), 200


@app.route('/training_status')
def training_status():
    """Aggregate the ablation results file into a per-config progress summary.

    Reads ``ABLATION_RESULTS_PATH`` on every call (so rebinding the module
    attribute takes effect immediately, as the tests rely on). The file is a
    JSON list of per-run dicts with ``config`` / ``seed`` / ``lemma.f1`` /
    ``upos.f1`` / ``grammeme.f1`` (see ``results/ablation_raw.json``).
    """
    path = ABLATION_RESULTS_PATH
    if isinstance(path, str):
        path = Path(path)
    if not path.exists():
        return jsonify({'available': False, 'completed': 0, 'expected': EXPECTED_TRAINING_RUNS, 'runs': []})
    try:
        raw = json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return jsonify({'available': False, 'completed': 0, 'expected': EXPECTED_TRAINING_RUNS, 'runs': []})
    if not isinstance(raw, list):
        raw = []

    per_config: dict[str, list[dict]] = {}
    for row in raw:
        per_config.setdefault(row.get('config', '?'), []).append(row)
    runs = []
    for config, rows in sorted(per_config.items()):
        def _mean(field):
            vals = [float(r.get(field, {}).get('f1', 0.0)) for r in rows
                    if isinstance(r.get(field), dict)]
            return float(sum(vals) / len(vals)) if vals else 0.0
        runs.append({
            'config': config,
            'seeds_done': len(rows),
            'lemma_f1': _mean('lemma'),
            'upos_f1': _mean('upos'),
            'grammeme_f1': _mean('grammeme'),
        })
    return jsonify({
        'available': True,
        'completed': len(raw),
        'expected': EXPECTED_TRAINING_RUNS,
        'runs': runs,
    })


# Load at import time so a freshly-started server is immediately usable. Tests
# rebind MODEL / VOCABS / META themselves, so a failed autoloading here is fine.
_load_backend()


if __name__ == '__main__':
    app.run(debug=True)
