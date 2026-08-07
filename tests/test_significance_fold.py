"""Tests for scripts/significance_test.py fold-mode statistics.

These test the new ``--unit fold`` behaviour and the bootstrap CI, both
introduced to fix the n=3 statistics problem (Wilcoxon minimum p is 0.25 at
n=3; Cohen's d is unstable). The legacy seed-mode tests live in
test_new_scripts.py.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

import pytest

from scripts.significance_test import (
    bootstrap_ci_diff,
    cohens_d_paired,
    effect_label,
    holm_bonferroni,
    load_scores,
    paired_test,
)


# ---------------------------------------------------------------------------
# bootstrap_ci_diff
# ---------------------------------------------------------------------------
def test_bootstrap_ci_contains_observed_difference():
    # clear effect with real per-fold noise (so the bootstrap has variance to
    # work with; identical diffs would give a degenerate point CI).
    rng = random.Random(3)
    a = [0.85 + rng.gauss(0, 0.015) for _ in range(10)]
    b = [x - 0.05 + rng.gauss(0, 0.005) for x in a]
    lo, hi = bootstrap_ci_diff(a, b, n_boot=2000)
    mean_diff = sum(a) / len(a) - sum(b) / len(b)
    assert lo <= mean_diff <= hi, f'CI [{lo}, {hi}] must bracket the mean diff {mean_diff}'


def test_bootstrap_ci_excludes_zero_when_effect_is_clear():
    a = [0.85, 0.86, 0.84, 0.87, 0.85, 0.86, 0.84, 0.87, 0.85, 0.86]
    b = [x - 0.05 for x in a]
    lo, hi = bootstrap_ci_diff(a, b, n_boot=5000)
    assert lo > 0, f'clear positive effect: CI lower bound {lo} should exclude 0'


def test_bootstrap_ci_brackets_zero_when_no_effect():
    rng = random.Random(0)
    a = [0.80 + rng.gauss(0, 0.02) for _ in range(10)]
    b = [0.80 + rng.gauss(0, 0.02) for _ in range(10)]
    lo, hi = bootstrap_ci_diff(a, b, n_boot=5000)
    assert lo <= 0 <= hi, f'no effect: CI [{lo}, {hi}] should bracket 0'


def test_bootstrap_ci_too_few_returns_nan():
    lo, hi = bootstrap_ci_diff([0.8], [0.7])
    assert math.isnan(lo) and math.isnan(hi)


# ---------------------------------------------------------------------------
# load_scores fold mode
# ---------------------------------------------------------------------------
def test_load_scores_fold_mode_sorts_by_fold():
    # deliberately unsorted by fold to verify the pairing alignment
    rows = [
        {'config': 'full_model', 'fold': 2, 'official': {'UPOS': 0.90}},
        {'config': 'full_model', 'fold': 0, 'official': {'UPOS': 0.85}},
        {'config': 'full_model', 'fold': 1, 'official': {'UPOS': 0.88}},
    ]
    scores = load_scores(rows, 'full_model', 'official.UPOS', unit='fold')
    assert scores == [0.85, 0.88, 0.90], 'fold mode must return scores sorted by fold index'


def test_load_scores_fold_mode_dotted_official_path():
    rows = [{'config': 'full_model', 'fold': 0, 'official': {'UPOS': 0.85, 'UFeats': 0.60}}]
    assert load_scores(rows, 'full_model', 'official.UPOS', unit='fold') == [0.85]
    assert load_scores(rows, 'full_model', 'official.UFeats', unit='fold') == [0.60]


def test_load_scores_seed_mode_unchanged():
    # legacy behaviour: bare task name -> row[task][metric]
    rows = [{'config': 'full_model', 'seed': 13, 'upos': {'f1': 0.71}}]
    assert load_scores(rows, 'full_model', 'upos', metric='f1', unit='seed') == [0.71]


# ---------------------------------------------------------------------------
# paired_test with n=10 folds (the whole point of the CV work)
# ---------------------------------------------------------------------------
def test_paired_test_n10_wilcoxon_can_reach_significance():
    # At n=3 wilcoxon min p is 0.25; at n=10 it can reach ~0.002.
    a = [0.85, 0.86, 0.84, 0.87, 0.85, 0.86, 0.84, 0.87, 0.85, 0.86]
    b = [x - 0.05 for x in a]
    r = paired_test(a, b)
    assert r['n'] == 10
    assert r['wilcoxon_p'] < 0.05, f'n=10 Wilcoxon should be significant, got p={r["wilcoxon_p"]}'
    assert r['t_p'] < 0.001
    assert not math.isnan(r['bootstrap_ci_lo'])


def test_paired_test_no_effect_not_significant():
    rng = random.Random(42)
    a = [0.80 + rng.gauss(0, 0.02) for _ in range(10)]
    b = [0.80 + rng.gauss(0, 0.02) for _ in range(10)]
    r = paired_test(a, b)
    # no consistent effect -> t-test should not be strongly significant
    assert r['t_p'] > 0.05
    assert r['bootstrap_ci_lo'] <= 0 <= r['bootstrap_ci_hi']


# ---------------------------------------------------------------------------
# Holm monotonicity (regression guard -- already in test_new_scripts but
# re-asserted here for the fold-mode path)
# ---------------------------------------------------------------------------
def test_holm_never_decreases_adjusted_p():
    pvals = [0.001, 0.01, 0.04, 0.03, 0.005]
    adj = holm_bonferroni(pvals)
    # sort adjusted by the original sort order and verify monotonic non-decrease
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj_in_order = [adj[i] for i in order]
    for i in range(len(adj_in_order) - 1):
        assert adj_in_order[i] <= adj_in_order[i + 1] + 1e-12, \
            'Holm adjusted p must be monotone non-decreasing in raw-p rank'
