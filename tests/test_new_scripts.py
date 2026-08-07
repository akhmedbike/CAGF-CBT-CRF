"""Tests for the new reproducibility/analysis scripts.

Covers: pick_device (MPS priority), the Holm-Bonferroni step-down correction,
Cohen's d sign, and make_tables output shape from a synthetic raw-results file.
These are pure-logic tests (no GPU, no training) so they run in well under a
second and do not interfere with the overnight training run.
"""
from __future__ import annotations
import json
import math
import sys
from pathlib import Path

# make the repo root importable when pytest is run from anywhere
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cagf.device import pick_device


class TestPickDevice:
    def test_explicit_override_wins(self):
        assert pick_device("cpu") == "cpu"
        assert pick_device("mps") == "mps"

    def test_auto_returns_valid_backend(self):
        d = pick_device()
        assert d in ("mps", "cuda", "cpu")

    def test_empty_string_falls_through_to_auto(self):
        # empty prefer should not be treated as an override
        assert pick_device("") in ("mps", "cuda", "cpu")


class TestHolmBonferroni:
    def _holm(self, pvals, alpha=0.05):
        # import the module-level function from the script
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "sig", ROOT / "scripts" / "significance_test.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.holm_bonferroni(pvals, alpha)

    def test_all_significant_stay_significant(self):
        # tiny p-values survive Holm
        adj = self._holm([0.001, 0.002, 0.003])
        assert all(p < 0.05 for p in adj)

    def test_correction_never_decreases_p(self):
        # Holm is step-down: adjusted p-values are non-decreasing in rank order
        import random
        random.seed(0)
        pvals = [random.uniform(0, 0.5) for _ in range(8)]
        adj = self._holm(pvals)
        order = sorted(range(len(pvals)), key=lambda i: pvals[i])
        adj_in_rank_order = [adj[order[r]] for r in range(len(pvals))]
        for a, b in zip(adj_in_rank_order, adj_in_rank_order[1:]):
            assert a <= b + 1e-9

    def test_capped_at_one(self):
        adj = self._holm([0.9, 0.95, 0.99])
        assert all(p <= 1.0 for p in adj)

    def test_trivial_single(self):
        assert self._holm([0.03]) == [0.03]


class TestCohensD:
    def test_sign_and_magnitude(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "sig", ROOT / "scripts" / "significance_test.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        # a clearly larger than b -> positive d; b larger -> negative
        d_pos = mod.cohens_d_paired([0.9, 0.91, 0.92, 0.93, 0.94],
                                    [0.5, 0.51, 0.52, 0.53, 0.54])
        d_neg = mod.cohens_d_paired([0.5, 0.51, 0.52, 0.53, 0.54],
                                    [0.9, 0.91, 0.92, 0.93, 0.94])
        assert d_pos > 0
        assert d_neg < 0
        assert abs(d_pos - abs(d_neg)) < 1e-9
        # identical -> d = 0
        assert mod.cohens_d_paired([0.5] * 5, [0.5] * 5) == 0.0

    def test_effect_labels(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "sig", ROOT / "scripts" / "significance_test.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert mod.effect_label(0.1) == 'negligible'
        assert mod.effect_label(0.3) == 'small'
        assert mod.effect_label(0.6) == 'medium'
        assert mod.effect_label(1.0) == 'large'


class TestMakeTables:
    def test_generates_main_table_from_synthetic_data(self, tmp_path):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "mt", ROOT / "scripts" / "make_tables.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        raw = [
            {"config": "full_model", "seed": 13,
             "lemma": {"accuracy": 0.6, "precision": 0.4, "recall": 0.4, "f1": 0.4},
             "upos": {"accuracy": 0.8, "precision": 0.7, "recall": 0.7, "f1": 0.7},
             "grammeme": {"accuracy": 0.5, "precision": 0.5, "recall": 0.4, "f1": 0.44}},
            {"config": "full_model", "seed": 42,
             "lemma": {"accuracy": 0.62, "precision": 0.42, "recall": 0.41, "f1": 0.415},
             "upos": {"accuracy": 0.82, "precision": 0.71, "recall": 0.69, "f1": 0.70},
             "grammeme": {"accuracy": 0.51, "precision": 0.49, "recall": 0.39, "f1": 0.43}},
        ]
        raw_path = tmp_path / "raw.json"
        raw_path.write_text(json.dumps(raw))
        out_dir = tmp_path / "tables"
        # invoke main via argparse by constructing sys.argv
        import sys as _sys
        old = _sys.argv
        _sys.argv = ["make_tables.py", "--ablation", str(raw_path),
                     "--baselines", str(tmp_path / "none.json"),
                     "--significance", str(tmp_path / "none.json"),
                     "--out-dir", str(out_dir)]
        try:
            mod.main()
        finally:
            _sys.argv = old
        main_md = (out_dir / "main_results.md").read_text()
        assert "CAGF-CBT+CRF (full)" in main_md
        assert "Lemma" in main_md and "UPOS" in main_md and "Grammeme" in main_md
        # mean of 0.40 and 0.415 = 0.4075 -> 40.75
        assert "40.75" in main_md
