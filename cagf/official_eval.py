"""Official CoNLL-2018 / Universal Dependencies evaluation wrapper.

This is a thin adapter over the vendored ``third_party/conll18_ud_eval.py``
(the official UFAL evaluator, MPL-2.0). The evaluator is imported as a module
and called directly -- never via subprocess -- so we get Python exceptions
rather than parse errors when something is wrong.

Why a wrapper at all
--------------------
The official script returns ``Score`` objects with ``.f1`` / ``.precision`` /
``.recall`` / ``.aligned_accuracy`` attributes. Research code only needs the
four morphological F1 metrics that are comparable across the UD literature:

  * ``Lemmas``   -- lemma exact match (gold ``_`` is treated as "any" match)
  * ``UPOS``     -- UPOS tag exact match
  * ``UFeats``   -- the full FEATS bundle exact match (string comparison, so
                    feature ORDER matters -- this is why
                    cagf.predict_writer serialises FEATS alphabetically)
  * ``AllTags``  -- (UPOS, XPOS, FEATS) triple exact match

``UFeats`` is the metric that replaces our earlier "Grammeme macro-F1" when
comparing to KazRoBERTa / UDify / any UD-pipeline paper. The two are not the
same quantity and must be reported side by side, never conflated.

Acceptance guarantee
--------------------
``evaluate_conllu(gold_path, gold_path)`` returns exactly 100.0 on all four
metrics. This is tested in ``tests/test_official_eval.py`` and is the
guarantee that our CoNLL-U writer (cagf.predict_writer) produces files the
official scorer can align.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

# Make the vendored third_party/ directory importable without polluting
# sys.path globally. Only adds it once.
_TP_DIR = str(Path(__file__).resolve().parent.parent / "third_party")
if _TP_DIR not in sys.path:
    sys.path.insert(0, _TP_DIR)

import conll18_ud_eval  # noqa: E402  (vendored, path-injected above)

# The four morphological metrics we report in the paper.
REPORTED_METRICS = ("Lemmas", "UPOS", "UFeats", "AllTags")


def evaluate_conllu(gold_path: str, system_path: str) -> Dict[str, float]:
    """Score ``system_path`` against ``gold_path`` with the official UD script.

    Returns a dict mapping each of ``Lemmas``/``UPOS``/``UFeats``/``AllTags``
    to its F1 in the range [0.0, 1.0]. Both files must be valid CoNLL-U; the
    official script raises ``UDError`` on malformed input (propagated).
    """
    gold_ud = conll18_ud_eval.load_conllu_file(gold_path)
    system_ud = conll18_ud_eval.load_conllu_file(system_path)
    scores = conll18_ud_eval.evaluate(gold_ud, system_ud)
    return {metric: float(scores[metric].f1) for metric in REPORTED_METRICS}


def evaluate_conllu_full(gold_path: str, system_path: str) -> Dict[str, Dict[str, float]]:
    """Like :func:`evaluate_conllu` but returns precision/recall/f1/counts for
    diagnostics (e.g. when debugging why a metric is unexpectedly low)."""
    gold_ud = conll18_ud_eval.load_conllu_file(gold_path)
    system_ud = conll18_ud_eval.load_conllu_file(system_path)
    scores = conll18_ud_eval.evaluate(gold_ud, system_ud)
    out: Dict[str, Dict[str, float]] = {}
    for metric in REPORTED_METRICS:
        s = scores[metric]
        out[metric] = {
            "precision": float(s.precision),
            "recall": float(s.recall),
            "f1": float(s.f1),
            "correct": float(s.correct),
            "gold_total": float(s.gold_total),
            "system_total": float(s.system_total),
        }
    return out
