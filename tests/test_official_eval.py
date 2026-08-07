"""Tests for cagf.official_eval: the conll18_ud_eval wrapper.

The acceptance criterion from the spec is strict: gold-vs-gold must score
exactly 100.0 on every reported metric. This is the test that the CoNLL-U
writer (cagf.predict_writer) and the official scorer agree on alignment, and
it must pass before any model evaluation can be trusted.
"""
from __future__ import annotations

from pathlib import Path

from cagf.data import read_conllu
from cagf.official_eval import REPORTED_METRICS, evaluate_conllu, evaluate_conllu_full
from cagf.predict_writer import write_conllu


def test_gold_vs_identical_gold_is_100():
    """The canonical acceptance test: scoring a file against itself returns
    exactly 1.0 (100.0%) on Lemmas, UPOS, UFeats, AllTags."""
    gold_path = "data/gold_merged/gold_test.conllu"
    scores = evaluate_conllu(gold_path, gold_path)
    for metric in REPORTED_METRICS:
        assert scores[metric] == 1.0, \
            f'{metric} gold-vs-gold = {scores[metric]}, expected exactly 1.0'


def test_gold_vs_gold_written_through_predict_writer_is_100(tmp_path):
    """The real acceptance test: write gold values through OUR writer, then
    score the result against the original gold. If this is not 100.0, the
    writer is producing CoNLL-U the official scorer cannot align -- and every
    downstream model metric would be silently wrong."""
    gold_path = "data/gold_merged/gold_test.conllu"
    sents = read_conllu(gold_path)
    preds = [{
        "lemma": [t.lemma for t in s.tokens],
        "upos": [t.upos for t in s.tokens],
        "feats": [t.feats.split("|") if t.feats != "_" else [] for t in s.tokens],
    } for s in sents]
    out = tmp_path / "gold_as_pred.conllu"
    write_conllu(sents, preds, out)

    scores = evaluate_conllu(gold_path, str(out))
    for metric in REPORTED_METRICS:
        assert scores[metric] == 1.0, \
            f'{metric} gold-through-writer vs gold = {scores[metric]}, expected 1.0'


def test_degraded_predictions_score_below_100(tmp_path):
    """Sanity check the other direction: deliberately wrong predictions must
    score noticeably below 1.0. If this failed, the scorer would be a no-op."""
    gold_path = "data/gold_merged/gold_test.conllu"
    sents = read_conllu(gold_path)
    # corrupt: every lemma -> "X", every upos -> "X", drop all feats
    preds = [{
        "lemma": ["X"] * len(s.tokens),
        "upos": ["X"] * len(s.tokens),
        "feats": [[] for _ in s.tokens],
    } for s in sents]
    out = tmp_path / "degraded.conllu"
    write_conllu(sents, preds, out)

    scores = evaluate_conllu(gold_path, str(out))
    # UPOS should be near zero (everything is "X"); Lemmas near zero too.
    assert scores["UPOS"] < 0.05, f'UPOS on all-"X" predictions = {scores["UPOS"]}, expected near 0'
    assert scores["Lemmas"] < 0.05, f'Lemmas on all-"X" predictions = {scores["Lemmas"]}, expected near 0'


def test_full_eval_returns_precision_recall_f1(tmp_path):
    """evaluate_conllu_full returns the precision/recall/f1/counts dict used
    for diagnostics -- verify the schema and that f1 matches the simple call."""
    gold_path = "data/gold_merged/gold_test.conllu"
    full = evaluate_conllu_full(gold_path, gold_path)
    simple = evaluate_conllu(gold_path, gold_path)
    for metric in REPORTED_METRICS:
        assert set(full[metric].keys()) == {"precision", "recall", "f1", "correct", "gold_total", "system_total"}
        assert full[metric]["f1"] == simple[metric]
        # gold-vs-gold: precision == recall == f1 == 1.0
        assert full[metric]["precision"] == 1.0
        assert full[metric]["recall"] == 1.0
        assert full[metric]["correct"] == full[metric]["gold_total"]
