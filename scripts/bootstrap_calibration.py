#!/usr/bin/env python3
"""Sentence-level (cluster) bootstrap CIs for the calibration metrics of the
ICCIDC 2026 reliability paper (docs/bali.md).

Closes the Limitations gap "the reported Wilson intervals treat token outcomes
as independent and therefore do not capture within-sentence dependence": the
resampling unit here is the sentence, not the token. No model is retrained and
no temperature is refitted — the stored predictions/probabilities are fixed
and only the evaluation set is resampled.

Inputs (regenerated deterministically into results_calibration/dumps/):
  CRF  — the bundled kz-kalib CRF baseline (kzcalib.crf.run_crf) over the
         canonical 754/162/162 primary split's 162-sentence test partition
         (1,594 tokens). Confidence = marginal probability of the decoded tag.
  CAGF — models/interface_model.pt over its original 862/107/109 split
         (109-sentence, 1,094-token held-out test). Confidence = emission
         softmax at the fixed CRF-decoded UPOS tag; temperature is fixed at
         the archived values T = 2.55 (dev-NLL) and T = 2.60 (dev-ECE grid).

Before any resampling, point estimates computed from the dumps are checked
against the numbers published in docs/bali.md; a mismatch aborts the run
(checkpoint/split alignment gate).

Bootstrap design (per docs/bali_comments.md spec):
  * B = 1000 resamples, seed 42, percentile 95% CIs.
  * Each resample draws n_sentences sentence ids with replacement and takes
    ALL tokens of each drawn sentence (duplicates included).
  * For CAGF the same resampled sentence multiset is used in the raw, T=2.55
    and T=2.60 conditions of every iteration, so paired deltas
    (risk_T260 - risk_raw, coverage_T260 - coverage_raw) get their own CIs.
  * BCa intervals and bootstrap histograms are produced as diagnostics.

Usage:
    PYTHONHASHSEED=42 .venv/bin/python scripts/bootstrap_calibration.py \
        [--kzkalib-root ../kz-kalib-paper] [--n-boot 1000] [--seed 42]
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DUMPS = REPO / "results_calibration" / "dumps"
OUT_DIR = REPO / "results_calibration"

LOW, HIGH = 0.70, 0.85
N_BINS = 10
T_NLL, T_ECE = 2.55, 2.60  # archived dev-fitted temperatures, not refitted here

# ---------------------------------------------------------------------------
# Published point estimates (docs/bali.md, sections 4.2/4.3) — the sanity gate.
# Tolerance is half of the last published digit.
# ---------------------------------------------------------------------------
PUBLISHED_CRF = {
    "n_tokens": 1594, "n_sentences": 162, "n_correct": 1506,
    "accuracy_pct": 94.48, "ece_pct": 2.71, "brier_top1": 0.044,
    "coverage085_pct": 92.97, "risk085_pct": 3.44, "n_auto085": 1482,
    "review070_pct": 3.26, "intermediate_pct": 3.76,
}
PUBLISHED_CAGF = {
    "n_tokens": 1094, "n_sentences": 109, "n_correct": 919,
    "accuracy_pct": 84.00,
    "raw": {"ece_pct": 12.79, "brier_top1": 0.136, "coverage085_pct": 92.50,
            "risk085_pct": 13.04},
    "T255": {"ece_pct": 2.99},
    "T260": {"ece_pct": 2.79, "brier_top1": 0.103, "coverage085_pct": 67.18,
             "risk085_pct": 5.17, "review070_pct": 19.56,
             "intermediate_pct": 13.25},
}


class SanityError(AssertionError):
    pass


def check(name: str, value: float, target: float, tol: float) -> float:
    if not abs(value - target) <= tol:
        raise SanityError(
            f"sanity gate FAILED: {name} = {value!r}, published {target!r} "
            f"(tol {tol}). The dumps do not reproduce the paper's numbers — "
            "do not use these bootstrap results."
        )
    return value


# ---------------------------------------------------------------------------
# Metric implementations (same conventions as the paper / kz-kalib metrics.py)
# ---------------------------------------------------------------------------

def softmax(z: np.ndarray, T: float = 1.0) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64) / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = N_BINS) -> float:
    """Equal-width ECE (paper Eq. 2, M = 10); empty bins contribute nothing."""
    conf = np.asarray(conf, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(conf)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        cnt = int(in_bin.sum())
        if cnt == 0:
            continue
        total += (cnt / n) * abs(correct[in_bin].mean() - conf[in_bin].mean())
    return float(total)


def brier_top1(conf: np.ndarray, correct: np.ndarray) -> float:
    """Binary top-1 Brier: mean (conf - correct)^2.

    This is the quantity behind the paper's reported Brier values (CRF 0.044,
    CAGF 0.136 raw / 0.103 scaled) — verified against the published numbers
    to publication precision by the sanity gate.
    """
    conf = np.asarray(conf, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    return float(((conf - correct) ** 2).mean())


def brier_full(probs: np.ndarray, gold_ids: np.ndarray) -> float:
    """Full-vector multiclass Brier (paper Eq. 3), reported alongside."""
    probs = np.asarray(probs, dtype=np.float64)
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(gold_ids)), gold_ids] = 1.0
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def selective_stats(conf: np.ndarray, correct: np.ndarray,
                    low: float = LOW, high: float = HIGH) -> dict:
    conf = np.asarray(conf, dtype=np.float64)
    correct = np.asarray(correct, dtype=bool)
    keep = conf >= high
    return {
        "coverage085": float(keep.mean()),
        "risk085": float((~correct[keep]).mean()) if keep.any() else float("nan"),
        "n_auto085": int(keep.sum()),
        "review070": float((conf < low).mean()),
        "intermediate": float(((conf >= low) & (conf < high)).mean()),
    }


def crf_metrics(gold_ids: np.ndarray, pred_ids: np.ndarray,
                probs: np.ndarray) -> dict:
    """CRF condition: confidence = marginal probability of the decoded tag."""
    correct = pred_ids == gold_ids
    conf = probs[np.arange(len(pred_ids)), pred_ids]
    m = {"accuracy": float(correct.mean()), "ece": ece(conf, correct),
         "brier_top1": brier_top1(conf, correct), "brier_full": brier_full(probs, gold_ids)}
    m.update(selective_stats(conf, correct))
    return m


def cagf_metrics(gold_ids: np.ndarray, pred_ids: np.ndarray,
                 emissions: np.ndarray, T: float) -> dict:
    """CAGF condition: fixed CRF-decoded tags, emission softmax at that tag."""
    p = softmax(emissions, T)
    correct = pred_ids == gold_ids
    conf = p[np.arange(len(pred_ids)), pred_ids]
    m = {"ece": ece(conf, correct), "brier_top1": brier_top1(conf, correct),
         "brier_full": brier_full(p, gold_ids)}
    m.update(selective_stats(conf, correct))
    return m


# ---------------------------------------------------------------------------
# Stage A — deterministic per-token dumps
# ---------------------------------------------------------------------------

def dump_crf(kzkalib_root: Path) -> dict:
    sys.path.insert(0, str(kzkalib_root))
    from kzcalib.crf import run_crf            # noqa: E402
    from kzcalib.data import LABEL2ID, UPOS_LABELS, parse_conllu  # noqa: E402

    sentences = parse_conllu(kzkalib_root / "data" / "test.conllu")
    out = run_crf(sentences)
    records, probs = out["records"], out["probs"]
    gold_ids = np.array([LABEL2ID[r["gold"]] for r in records])
    pred_ids = np.array([LABEL2ID[r["pred"]] if r["pred"] in LABEL2ID else -1
                         for r in records])
    assert (pred_ids >= 0).all(), "CRF predicted a tag outside the 17-label space"
    sent_ids = np.array([r["sent_id"] for r in records])
    tok_ids = np.array([r["tok_idx"] for r in records])
    DUMPS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        DUMPS / "crf_test.npz", sent_ids=sent_ids, tok_ids=tok_ids,
        gold_ids=gold_ids, pred_ids=pred_ids, probs=probs,
        upos_labels=np.array(UPOS_LABELS),
    )
    return {"n_tokens": len(records), "n_sentences": len(sentences)}


def dump_cagf() -> dict:
    sys.path.insert(0, str(REPO))
    import torch
    from cagf.data import read_conllu
    from cagf.dataset import MAX_WORD_CHARS
    from cagf.inference import load_checkpoint

    model, vocabs, meta = load_checkpoint(
        str(REPO / "models" / "interface_model.pt"),
        str(REPO / "models" / "interface_vocabs.json"), device="cpu")
    assert meta["ablation_name"] == "full_model"
    BOS, EOS, CHAR_PAD = "<bos>", "<eos>", vocabs.char_vocab.stoi["<pad>"]

    stats = {}
    with torch.no_grad():
        for split in ("dev", "test"):
            sents = read_conllu(REPO / "data" / "gold_merged" / f"gold_{split}.conllu")
            rows_gold, rows_pred, rows_emit, rows_sent = [], [], [], []
            for si, sent in enumerate(sents):
                tokens = [t.form for t in sent.tokens]
                word_ids = [vocabs.word_vocab.encode(t.lower()) for t in tokens]
                char_ids = [[vocabs.char_vocab.encode(c)
                             for c in [BOS] + list(t)[:MAX_WORD_CHARS] + [EOS]]
                            for t in tokens]
                max_chars = max(len(c) for c in char_ids)
                char_tensor = torch.full((1, len(tokens), max_chars), CHAR_PAD,
                                         dtype=torch.long)
                for i, c in enumerate(char_ids):
                    char_tensor[0, i, :len(c)] = torch.tensor(c, dtype=torch.long)
                out = model(
                    torch.tensor([word_ids], dtype=torch.long), char_tensor,
                    torch.tensor([len(tokens)], dtype=torch.long),
                    torch.ones(1, len(tokens), dtype=torch.bool), upos_ids=None)
                upos_pred = out["upos_pred"]
                pred = upos_pred[0] if isinstance(upos_pred, list) else upos_pred[0].tolist()
                emissions = out["upos_emissions"][0].numpy().astype(np.float64)
                for i, tok in enumerate(sent.tokens):
                    gold = vocabs.upos_vocab.stoi.get(tok.upos, vocabs.upos_vocab.stoi["<unk>"])
                    rows_gold.append(gold)
                    rows_pred.append(pred[i])
                    rows_emit.append(emissions[i])
                    rows_sent.append(si)
            DUMPS.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                DUMPS / f"cagf_{split}.npz",
                sent_ids=np.array(rows_sent),
                gold_ids=np.array(rows_gold, dtype=np.int64),
                pred_ids=np.array(rows_pred, dtype=np.int64),
                emissions=np.array(rows_emit),
                upos_labels=np.array(vocabs.upos_vocab.itos))
            stats[split] = {"n_tokens": len(rows_gold), "n_sentences": len(sents)}
    return stats


# ---------------------------------------------------------------------------
# Stage B — sanity gate against the published point estimates
# ---------------------------------------------------------------------------

def run_checks(model: str, checks: dict) -> dict:
    report = {}
    for name, (value, target, tol) in checks.items():
        if tol == 0:
            if int(value) != int(target):
                raise SanityError(f"sanity gate FAILED [{model}] {name}: "
                                  f"{value} != {target}")
        else:
            check(f"[{model}] {name}", float(value), float(target), tol)
        report[name] = {"value": float(value), "published": target, "ok": True}
    print(f"sanity gate [{model}]: all {len(checks)} checks passed")
    return report


def sanity_crf(data: dict) -> dict:
    m = crf_metrics(data["gold_ids"], data["pred_ids"], data["probs"])
    p = PUBLISHED_CRF
    checks = {
        "n_tokens": (data["gold_ids"].size, p["n_tokens"], 0),
        "n_sentences": (len(np.unique(data["sent_ids"])), p["n_sentences"], 0),
        "n_correct": (round(m["accuracy"] * data["gold_ids"].size), p["n_correct"], 0),
        "accuracy_pct": (100 * m["accuracy"], p["accuracy_pct"], 0.005),
        "ece_pct": (100 * m["ece"], p["ece_pct"], 0.005),
        "brier_top1": (m["brier_top1"], p["brier_top1"], 0.0005),
        "coverage085_pct": (100 * m["coverage085"], p["coverage085_pct"], 0.005),
        "risk085_pct": (100 * m["risk085"], p["risk085_pct"], 0.005),
        "n_auto085": (m["n_auto085"], p["n_auto085"], 0),
        "review070_pct": (100 * m["review070"], p["review070_pct"], 0.005),
        "intermediate_pct": (100 * m["intermediate"], p["intermediate_pct"], 0.005),
    }
    return run_checks("CRF", checks)


def sanity_cagf(data: dict) -> dict:
    gold, pred, emit = data["gold_ids"], data["pred_ids"], data["emissions"]
    p = PUBLISHED_CAGF
    acc = float((pred == gold).mean())
    raw = cagf_metrics(gold, pred, emit, 1.0)
    t255 = cagf_metrics(gold, pred, emit, T_NLL)
    t260 = cagf_metrics(gold, pred, emit, T_ECE)
    checks = {
        "n_tokens": (gold.size, p["n_tokens"], 0),
        "n_sentences": (len(np.unique(data["sent_ids"])), p["n_sentences"], 0),
        "n_correct": (round(acc * gold.size), p["n_correct"], 0),
        "accuracy_pct": (100 * acc, p["accuracy_pct"], 0.005),
        "raw.ece_pct": (100 * raw["ece"], p["raw"]["ece_pct"], 0.005),
        "raw.brier_top1": (raw["brier_top1"], p["raw"]["brier_top1"], 0.0005),
        "raw.coverage085_pct": (100 * raw["coverage085"], p["raw"]["coverage085_pct"], 0.005),
        "raw.risk085_pct": (100 * raw["risk085"], p["raw"]["risk085_pct"], 0.005),
        "T255.ece_pct": (100 * t255["ece"], p["T255"]["ece_pct"], 0.005),
        "T260.ece_pct": (100 * t260["ece"], p["T260"]["ece_pct"], 0.005),
        "T260.brier_top1": (t260["brier_top1"], p["T260"]["brier_top1"], 0.0005),
        "T260.coverage085_pct": (100 * t260["coverage085"], p["T260"]["coverage085_pct"], 0.005),
        "T260.risk085_pct": (100 * t260["risk085"], p["T260"]["risk085_pct"], 0.005),
        "T260.review070_pct": (100 * t260["review070"], p["T260"]["review070_pct"], 0.005),
        "T260.intermediate_pct": (100 * t260["intermediate"], p["T260"]["intermediate_pct"], 0.005),
    }
    return run_checks("CAGF", checks)


# ---------------------------------------------------------------------------
# Stage C — sentence-level cluster bootstrap
# ---------------------------------------------------------------------------

def group_by_sentence(sent_ids: np.ndarray):
    unique = np.unique(sent_ids)
    idx_by_sent = {s: np.where(sent_ids == s)[0] for s in unique}
    return unique, idx_by_sent


def percentile_ci(values) -> dict:
    v = np.asarray(values, dtype=np.float64)
    n_nan = int(np.isnan(v).sum())
    v = v[~np.isnan(v)]
    return {"mean": float(v.mean()), "ci95_low": float(np.percentile(v, 2.5)),
            "ci95_high": float(np.percentile(v, 97.5)), "n_nan": n_nan,
            "n": int(v.size)}


def _ppf(q: float) -> float:
    """Normal quantile via bisection on the CDF (no scipy dependency)."""
    from math import erf
    lo, hi = -10.0, 10.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if 0.5 * (1.0 + erf(mid / 2 ** 0.5)) < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _cdf(z: float) -> float:
    from math import erf
    return 0.5 * (1.0 + erf(z / 2 ** 0.5))


def bca_ci(point: float, boot: np.ndarray, jack: np.ndarray) -> dict:
    """BCa interval from bootstrap replicates + leave-one-cluster-out
    jackknife estimates (bias-correction z0 from P(boot < point), acceleration
    from the jackknife)."""
    boot = np.asarray(boot, dtype=np.float64)
    boot = boot[~np.isnan(boot)]
    b = len(boot)
    z0 = _ppf(np.clip((boot < point).sum(), 0.5, b - 0.5) / b)
    jack = np.asarray(jack, dtype=np.float64)
    diffs = jack.mean() - jack
    denom = (diffs ** 2).sum() ** 1.5
    a = float((diffs ** 3).sum() / (6.0 * denom)) if denom > 0 else 0.0
    out = {"z0": float(z0), "accel": a}
    for tag, alpha in (("low", 0.025), ("high", 0.975)):
        z = _ppf(alpha)
        adj = z0 + (z0 + z) / (1.0 - a * (z0 + z))
        out[tag] = float(np.percentile(boot, 100.0 * _cdf(adj)))
    return {"ci95_low": out["low"], "ci95_high": out["high"],
            "z0": out["z0"], "accel": out["accel"]}


def bootstrap_crf(data: dict, n_boot: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    unique, idx_by_sent = group_by_sentence(data["sent_ids"])
    gold, pred, probs = data["gold_ids"], data["pred_ids"], data["probs"]
    keys = ["accuracy", "ece", "brier_top1", "brier_full", "coverage085",
            "risk085", "review070", "intermediate"]
    samples = {k: [] for k in keys}
    n_unique_seen = []
    for _ in range(n_boot):
        sampled = rng.choice(unique, size=unique.size, replace=True)
        idx = np.concatenate([idx_by_sent[s] for s in sampled])
        m = crf_metrics(gold[idx], pred[idx], probs[idx])
        for k in keys:
            samples[k].append(m[k])
        n_unique_seen.append(len(np.unique(sampled)))
    cis = {k: percentile_ci(v) for k, v in samples.items()}
    # BCa for the headline metrics (jackknife over sentences)
    point = crf_metrics(gold, pred, probs)
    for k in ("accuracy", "ece", "risk085"):
        jack = []
        for s in unique:
            idx = np.concatenate([idx_by_sent[t] for t in unique if t != s])
            jack.append(crf_metrics(gold[idx], pred[idx], probs[idx])[k])
        cis[k]["bca"] = bca_ci(point[k], np.array(samples[k]), np.array(jack))
    return {"ci": cis, "samples": samples, "unique_sentences": {
        "point": int(unique.size),
        "resample_min": int(min(n_unique_seen)),
        "resample_mean": float(np.mean(n_unique_seen)),
        "resample_max": int(max(n_unique_seen))}}


def bootstrap_cagf(data: dict, n_boot: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    unique, idx_by_sent = group_by_sentence(data["sent_ids"])
    gold, pred, emit = data["gold_ids"], data["pred_ids"], data["emissions"]
    conditions = {"raw": 1.0, "T255": T_NLL, "T260": T_ECE}
    keys = ["ece", "brier_top1", "brier_full", "coverage085", "risk085",
            "review070", "intermediate"]
    samples = {c: {k: [] for k in keys} for c in conditions}
    samples["accuracy"] = []  # condition-independent (predictions are fixed)
    delta_keys = [f"d_{k}_{c}" for c in ("T255", "T260")
                  for k in ("ece", "risk085", "coverage085")]
    deltas = {d: [] for d in delta_keys}
    n_unique_seen = []
    for _ in range(n_boot):
        sampled = rng.choice(unique, size=unique.size, replace=True)
        idx = np.concatenate([idx_by_sent[s] for s in sampled])
        samples["accuracy"].append(float((pred[idx] == gold[idx]).mean()))
        iter_m = {}
        for cond, T in conditions.items():
            m = cagf_metrics(gold[idx], pred[idx], emit[idx], T)
            iter_m[cond] = m
            for k in keys:
                samples[cond][k].append(m[k])
        for c in ("T255", "T260"):
            for k in ("ece", "risk085", "coverage085"):
                deltas[f"d_{k}_{c}"].append(iter_m[c][k] - iter_m["raw"][k])
        n_unique_seen.append(len(np.unique(sampled)))
    cis = {c: {k: percentile_ci(v) for k, v in samples[c].items()}
           for c in conditions}
    cis["accuracy"] = percentile_ci(samples["accuracy"])
    delta_cis = {d: percentile_ci(v) for d, v in deltas.items()}

    def point_of(cond, key):
        return cagf_metrics(gold, pred, emit, conditions[cond])[key]

    # BCa for the headline metrics + the paired risk delta
    for cond in conditions:
        for k in ("ece", "risk085"):
            jack = []
            for s in unique:
                idx = np.concatenate([idx_by_sent[t] for t in unique if t != s])
                jack.append(cagf_metrics(gold[idx], pred[idx], emit[idx],
                                         conditions[cond])[k])
            cis[cond][k]["bca"] = bca_ci(point_of(cond, k),
                                        np.array(samples[cond][k]), np.array(jack))
    jack = []
    for s in unique:
        idx = np.concatenate([idx_by_sent[t] for t in unique if t != s])
        jack.append(cagf_metrics(gold[idx], pred[idx], emit[idx], T_ECE)["risk085"]
                    - cagf_metrics(gold[idx], pred[idx], emit[idx], 1.0)["risk085"])
    delta_cis["d_risk085_T260"]["bca"] = bca_ci(
        point_of("T260", "risk085") - point_of("raw", "risk085"),
        np.array(deltas["d_risk085_T260"]), np.array(jack))
    return {"ci": cis, "samples": samples, "deltas": delta_cis,
            "deltas_samples": deltas, "unique_sentences": {
                "point": int(unique.size),
                "resample_min": int(min(n_unique_seen)),
                "resample_mean": float(np.mean(n_unique_seen)),
                "resample_max": int(max(n_unique_seen))}}


# ---------------------------------------------------------------------------
# Stage D — report
# ---------------------------------------------------------------------------

def fmt_ci(ci: dict, scale: float = 100.0, digits: int = 2) -> str:
    lo, hi = ci["ci95_low"] * scale, ci["ci95_high"] * scale
    return f"{lo:.{digits}f}\u2013{hi:.{digits}f}"


def write_markdown(crf_boot: dict, cagf_boot: dict, crf_point: dict,
                   cagf_point: dict, n_boot: int, seed: int) -> str:
    cr, cg = crf_boot["ci"], cagf_boot["ci"]
    d = cagf_boot["deltas"]
    rows = [
        ("CRF accuracy (%)", 100 * crf_point["accuracy"], fmt_ci(cr["accuracy"])),
        ("CRF ECE (%)", 100 * crf_point["ece"], fmt_ci(cr["ece"])),
        ("CRF Brier (top-1)", crf_point["brier_top1"], fmt_ci(cr["brier_top1"], 1.0, 3)),
        ("CRF coverage@0.85 (%)", 100 * crf_point["coverage085"], fmt_ci(cr["coverage085"])),
        ("CRF risk@0.85 (%)", 100 * crf_point["risk085"], fmt_ci(cr["risk085"])),
        ("CRF review@0.70 (%)", 100 * crf_point["review070"], fmt_ci(cr["review070"])),
        ("CAGF accuracy (%)", 100 * cagf_point["accuracy"],
         fmt_ci(cagf_boot["ci"]["accuracy"])),
        ("CAGF ECE raw (%)", 100 * cagf_point["raw"]["ece"], fmt_ci(cg["raw"]["ece"])),
        ("CAGF ECE T=2.55 (%)", 100 * cagf_point["T255"]["ece"], fmt_ci(cg["T255"]["ece"])),
        ("CAGF ECE T=2.60 (%)", 100 * cagf_point["T260"]["ece"], fmt_ci(cg["T260"]["ece"])),
        ("CAGF risk@0.85 raw (%)", 100 * cagf_point["raw"]["risk085"], fmt_ci(cg["raw"]["risk085"])),
        ("CAGF risk@0.85 T=2.60 (%)", 100 * cagf_point["T260"]["risk085"], fmt_ci(cg["T260"]["risk085"])),
        ("CAGF coverage@0.85 raw (%)", 100 * cagf_point["raw"]["coverage085"], fmt_ci(cg["raw"]["coverage085"])),
        ("CAGF coverage@0.85 T=2.60 (%)", 100 * cagf_point["T260"]["coverage085"], fmt_ci(cg["T260"]["coverage085"])),
        ("\u0394 risk@0.85 (T=2.60 \u2212 raw, pp)",
         100 * (cagf_point["T260"]["risk085"] - cagf_point["raw"]["risk085"]),
         fmt_ci(d["d_risk085_T260"])),
        ("\u0394 coverage@0.85 (T=2.60 \u2212 raw, pp)",
         100 * (cagf_point["T260"]["coverage085"] - cagf_point["raw"]["coverage085"]),
         fmt_ci(d["d_coverage085_T260"])),
    ]
    lines = [
        "# Sentence-level cluster bootstrap 95% CIs",
        "",
        f"B = {n_boot} resamples, seed = {seed}, resampling unit = sentence "
        "(CRF: 162 test sentences / 1,594 tokens; CAGF-CBT+CRF: 109 test "
        "sentences / 1,094 tokens). Temperatures are the archived dev-fitted "
        "values (T = 2.55 NLL, T = 2.60 ECE grid) and are not refitted. "
        "CAGF conditions share each iteration's resample, so the delta rows "
        "are paired. Point estimates are computed on the full test data and "
        "pass the sanity gate against docs/bali.md.",
        "",
        "| Metric | Point | Sentence-bootstrap 95% CI |",
        "|---|---|---|",
    ]
    for name, point, ci_str in rows:
        lines.append(f"| {name} | {point:.2f} | {ci_str} |")
    lines += [
        "",
        "BCa intervals for the headline metrics, full replicate arrays and "
        "the sanity-gate report are in bootstrap_cis.json; histograms in "
        "figs/.",
        "",
    ]
    return "\n".join(lines)


def make_histograms(crf_boot: dict, cagf_boot: dict) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = OUT_DIR / "figs"
    fig_dir.mkdir(parents=True, exist_ok=True)
    panels = [
        ("crf_ece", "CRF ECE (%)", crf_boot["samples"]["ece"]),
        ("crf_risk085", "CRF selective risk @0.85 (%)", crf_boot["samples"]["risk085"]),
        ("cagf_ece_raw", "CAGF ECE raw (%)", cagf_boot["samples"]["raw"]["ece"]),
        ("cagf_ece_T260", "CAGF ECE T=2.60 (%)", cagf_boot["samples"]["T260"]["ece"]),
        ("cagf_risk085_raw", "CAGF risk @0.85 raw (%)", cagf_boot["samples"]["raw"]["risk085"]),
        ("cagf_risk085_T260", "CAGF risk @0.85 T=2.60 (%)", cagf_boot["samples"]["T260"]["risk085"]),
        ("cagf_d_risk085", "\u0394 risk @0.85 (T=2.60 \u2212 raw, pp)",
         cagf_boot["deltas_samples"]["d_risk085_T260"]),
    ]
    written = []
    for name, title, values in panels:
        v = np.asarray(values, dtype=np.float64) * 100.0
        fig, ax = plt.subplots(figsize=(4.2, 2.8))
        ax.hist(v, bins=30, color="#1f4e79", edgecolor="white", linewidth=0.4)
        for q in (2.5, 97.5):
            ax.axvline(np.percentile(v, q), color="#a8431f", lw=1.0, ls="--")
        ax.set_title(title, fontsize=9)
        ax.set_ylabel("resamples", fontsize=8)
        ax.tick_params(labelsize=7)
        fig.tight_layout()
        path = fig_dir / f"bootstrap_hist_{name}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)
    return written


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def environment_info() -> dict:
    """Provenance of the machine that produced the artifacts (no timestamps,
    so repeat runs stay bit-identical apart from this static description)."""
    env = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
    }
    for name in ("torch", "scipy", "sklearn_crfsuite", "joblib"):
        try:
            if name == "torch":
                import torch
                env[name] = torch.__version__
            else:
                from importlib import metadata
                env[name] = metadata.version(name)
        except Exception:
            env[name] = None
    return env


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kzkalib-root", type=Path,
                    default=REPO.parent / "kz-kalib-paper",
                    help="checkout of the kz-kalib-paper repo (CRF baseline)")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reuse-dumps", action="store_true",
                    help="skip stage A if dumps already exist")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    crf_npz, cagf_test_npz = DUMPS / "crf_test.npz", DUMPS / "cagf_test.npz"
    if args.reuse_dumps and crf_npz.exists() and cagf_test_npz.exists():
        print("reusing existing dumps")
    else:
        print("[stage A] dumping per-token predictions/probabilities ...")
        info_crf = dump_crf(args.kzkalib_root)
        info_cagf = dump_cagf()
        print(f"  CRF : {info_crf['n_sentences']} sentences, {info_crf['n_tokens']} tokens")
        print(f"  CAGF: dev {info_cagf['dev']['n_sentences']}/{info_cagf['dev']['n_tokens']}, "
              f"test {info_cagf['test']['n_sentences']}/{info_cagf['test']['n_tokens']}")

    crf_data = dict(np.load(crf_npz, allow_pickle=False))
    cagf_data = dict(np.load(cagf_test_npz, allow_pickle=False))

    print("[stage B] sanity gate against published point estimates ...")
    sanity = {"crf": sanity_crf(crf_data), "cagf": sanity_cagf(cagf_data)}

    print(f"[stage C] sentence-level bootstrap (B={args.n_boot}, seed={args.seed}) ...")
    crf_boot = bootstrap_crf(crf_data, args.n_boot, args.seed)
    cagf_boot = bootstrap_cagf(cagf_data, args.n_boot, args.seed)

    crf_point = crf_metrics(crf_data["gold_ids"], crf_data["pred_ids"], crf_data["probs"])
    cagf_point = {
        "accuracy": float((cagf_data["pred_ids"] == cagf_data["gold_ids"]).mean()),
        "raw": cagf_metrics(cagf_data["gold_ids"], cagf_data["pred_ids"], cagf_data["emissions"], 1.0),
        "T255": cagf_metrics(cagf_data["gold_ids"], cagf_data["pred_ids"], cagf_data["emissions"], T_NLL),
        "T260": cagf_metrics(cagf_data["gold_ids"], cagf_data["pred_ids"], cagf_data["emissions"], T_ECE),
    }

    result = {
        "meta": {
            "script": "scripts/bootstrap_calibration.py",
            "n_boot": args.n_boot, "seed": args.seed,
            "resampling_unit": "sentence (cluster bootstrap)",
            "thresholds": {"automatic": HIGH, "review": LOW},
            "n_bins_ece": N_BINS,
            "temperatures": {"NLL_fitted": T_NLL, "ECE_grid_fitted": T_ECE,
                             "refit_per_resample": False},
            "ece_binning": "equal-width bins, (lo, hi] except first bin [0, 0.1]",
            "crf_confidence": "marginal probability of the decoded tag",
            "cagf_confidence": "emission softmax at the fixed CRF-decoded tag",
            "brier_note": ("the paper's reported Brier values are the binary "
                           "top-1 Brier; the full-vector multiclass Brier "
                           "(Eq. 3) is stored alongside as brier_full"),
            "env": environment_info(),
        },
        "sanity": sanity,
        "crf_point": crf_point,
        "cagf_point": cagf_point,
        "crf_bootstrap": crf_boot,
        "cagf_bootstrap": cagf_boot,
    }
    # replicate arrays are kept in the JSON for reproducible downstream use
    (OUT_DIR / "bootstrap_cis.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    md = write_markdown(crf_boot, cagf_boot, crf_point, cagf_point,
                        args.n_boot, args.seed)
    (OUT_DIR / "bootstrap_cis.md").write_text(md + "\n", encoding="utf-8")
    print(f"[stage D] wrote {OUT_DIR / 'bootstrap_cis.json'} and bootstrap_cis.md")

    figs = make_histograms(crf_boot, cagf_boot)
    print(f"[stage D] wrote {len(figs)} diagnostic histograms to {OUT_DIR / 'figs'}")
    print("done.")


if __name__ == "__main__":
    main()
