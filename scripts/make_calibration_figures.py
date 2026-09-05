#!/usr/bin/env python3
"""Regenerate the three figures of the ICCIDC reliability paper (docs/bali.md)
as code, so the manuscript no longer depends on hash-named PNGs exported from
a docx.

  Fig. 1  confidence-aware decision process (four-stage pipeline fanning out
          into the three decision regions)                  -- drawn
  Fig. 2  calibration-specific evaluation flow for CAGF-CBT+CRF
          (checkpoint -> emissions/tags -> dev fit / test apply -> analysis
          -> decision regions)                              -- drawn
  Fig. 3  reliability of CAGF-CBT+CRF before/after temperature scaling
          (raw vs T = 2.60, 10 equal-width bins)            -- computed from
          results_calibration/dumps/cagf_test.npz

Fig. 3 recomputes the binned (mean confidence, accuracy) points from the same
token-level artifact as the bootstrap experiment and verifies the published
ECE values (raw 12.79%, T = 2.60 -> 2.79%) before drawing, so the figure and
the text cannot drift apart.

Usage:
    .venv/bin/python scripts/make_calibration_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from bootstrap_calibration import T_ECE, ece, softmax  # noqa: E402

OUT = REPO / "results_calibration" / "figs" / "paper"
OUT.mkdir(parents=True, exist_ok=True)


def _save(fig, name):
    for ext in ("png", "svg"):
        path = OUT / f"{name}.{ext}"
        fig.savefig(path, format=ext, bbox_inches="tight",
                    dpi=300 if ext == "png" else None)
        print(f"  wrote {path}")
    plt.close(fig)


def _box(ax, x0, y0, w, h, text, fill, edge, fontsize=9, family="sans-serif",
         lw=1.2):
    ax.add_patch(FancyBboxPatch((x0, y0), w, h,
                                boxstyle="round,pad=0.6,rounding_size=1.2",
                                linewidth=lw, edgecolor=edge, facecolor=fill,
                                mutation_aspect=1.0))
    ax.text(x0 + w / 2, y0 + h / 2, text, ha="center", va="center",
            fontsize=fontsize, family=family, color="#1a1a2e", linespacing=1.35)


def _arrow(ax, x0, y0, x1, y1, color="#4A5A7A", lw=1.3):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=12, linewidth=lw, color=color,
                                 shrinkA=0, shrinkB=0))


# ---------------------------------------------------------------------------
# Fig. 1 — confidence-aware decision process
# ---------------------------------------------------------------------------

def figure1():
    plt.rcParams.update({"font.family": "sans-serif"})
    fig, ax = plt.subplots(figsize=(9.6, 3.6))
    fill, edge = "#F0F4FA", "#4A5A7A"
    top = [("UPOS\nprediction", 2), ("confidence\nscore", 32),
           ("post-hoc\ncalibration (neural logits,\nwhere available)", 62),
           ("decision layer", 92)]
    for text, x in top:
        _box(ax, x, 26, 26, 10, text, fill, edge, fontsize=8.5)
    for x in (28, 58, 88):
        _arrow(ax, x + 0.6, 31, x + 3.4, 31)
    regions = [("c < 0.70\nreview", 32), ("0.70 ≤ c < 0.85\nintermediate", 62),
               ("c ≥ 0.85\nautomatic action", 92)]
    for text, x in regions:
        _box(ax, x, 5, 26, 10, text, fill, edge, fontsize=8.5)
    # fan out from spread points along the decision layer's bottom edge to
    # the top centers of the region boxes, which share the pipeline's
    # column grid (columns 2-4)
    for ox, (_, x) in zip((99, 105, 111), regions):
        _arrow(ax, ox, 25.4, x + 13, 15.6)
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 40)
    ax.axis("off")
    fig.tight_layout()
    _save(fig, "paper_fig1_decision_process")


# ---------------------------------------------------------------------------
# Fig. 2 — calibration-specific evaluation flow for CAGF-CBT+CRF
# ---------------------------------------------------------------------------

def figure2():
    plt.rcParams.update({"font.family": "serif"})
    fig, ax = plt.subplots(figsize=(11.4, 4.0))
    edge = "#3a3a3a"
    _box(ax, 2, 17, 22, 11,
         "CAGF-CBT+CRF\ncheckpoint\n(predictions fixed)",
         "#E8ECF4", edge, family="serif")
    _arrow(ax, 24.4, 22.5, 29.6, 22.5, color=edge)
    _box(ax, 30, 17, 24, 11,
         "UPOS emission scores $z_i$\n+ predicted tags $\\hat{y}_i$",
         "#F5F5F5", edge, family="serif")
    # fork into the dev (fit) and test (apply) branches
    _arrow(ax, 54.4, 24.5, 59.6, 36.5, color=edge)
    _arrow(ax, 54.4, 20.5, 59.6, 8.5, color=edge)
    _box(ax, 60, 31, 26, 11,
         "Development split\nfit T by NLL\n($T = 2.55$)",
         "#DEEBF7", edge, family="serif")
    _box(ax, 60, 3, 26, 11,
         "Held-out test split\nraw $c_i$ and scaled $c_i^{(T)}$",
         "#E2F0D9", edge, family="serif")
    _arrow(ax, 73, 30.6, 73, 14.4, color=edge)
    ax.text(74.4, 22.5, "apply T", fontsize=9, family="serif",
            style="italic", va="center")
    _arrow(ax, 86.4, 8.5, 91.6, 8.5, color=edge)
    _box(ax, 92, 3, 20, 37,
         "Reliability analysis\n\nECE\nBrier score\nReliability diagram\n"
         "Per-tag gaps\n\nThreshold analysis\nCoverage / risk",
         "#F5F8FC", edge, family="serif", fontsize=8.5)
    _arrow(ax, 112.4, 21.5, 117.6, 21.5, color=edge)
    _box(ax, 118, 3, 22, 37,
         "Decision regions\n\n$c \\geq 0.85$\nautomatic action\n\n"
         "$0.70 \\leq c < 0.85$\nintermediate\n\n$c < 0.70$\nreview",
         "#FDE9D9", edge, family="serif", fontsize=8.5)
    ax.set_xlim(0, 142)
    ax.set_ylim(0, 46)
    ax.axis("off")
    fig.tight_layout()
    _save(fig, "paper_fig2_calibration_flow")


# ---------------------------------------------------------------------------
# Fig. 3 — reliability of CAGF-CBT+CRF, raw vs temperature-scaled
# ---------------------------------------------------------------------------

def binned_points(gold, pred, emissions, T, n_bins=10):
    """(mean confidence, accuracy) of every occupied equal-width bin."""
    p = softmax(emissions, T)
    conf = p[np.arange(len(pred)), pred]
    correct = (pred == gold).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    pts, counts = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if in_bin.sum() == 0:
            continue
        pts.append((float(conf[in_bin].mean()), float(correct[in_bin].mean())))
        counts.append(int(in_bin.sum()))
    return np.array(pts), counts


def figure3():
    data = np.load(REPO / "results_calibration" / "dumps" / "cagf_test.npz")
    gold, pred, emit = data["gold_ids"], data["pred_ids"], data["emissions"]

    def conf_at(T):
        p = softmax(emit, T)
        return p[np.arange(len(pred)), pred]

    correct = pred == gold
    ece_raw, ece_cal = ece(conf_at(1.0), correct), ece(conf_at(T_ECE), correct)
    assert round(100 * ece_raw, 2) == 12.79, f"raw ECE drifted: {100*ece_raw:.4f}"
    assert round(100 * ece_cal, 2) == 2.79, f"scaled ECE drifted: {100*ece_cal:.4f}"
    print(f"  verified: raw ECE {100*ece_raw:.2f}%, T={T_ECE} ECE {100*ece_cal:.2f}%")

    raw_pts, raw_n = binned_points(gold, pred, emit, 1.0)
    cal_pts, cal_n = binned_points(gold, pred, emit, T_ECE)
    print(f"  raw bins (conf, acc, n): {[tuple(np.round(p,3)) for p in raw_pts]}, {raw_n}")
    print(f"  cal bins (conf, acc, n): {[tuple(np.round(p,3)) for p in cal_pts]}, {cal_n}")

    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif"})
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.plot([0, 1], [0, 1], "--", color="0.45", lw=1.2, label="perfect")
    ax.plot(raw_pts[:, 0], raw_pts[:, 1], "-o", color="#d62728", ms=4.5,
            lw=1.4, label="raw")
    ax.plot(cal_pts[:, 0], cal_pts[:, 1], "-o", color="#1f77b4", ms=4.5,
            lw=1.4, label="calibrated")
    ax.set_xlabel("mean confidence")
    ax.set_ylabel("accuracy")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.grid(True, color="0.9", lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=True, edgecolor="0.8")
    fig.tight_layout()
    _save(fig, "paper_fig3_reliability_cagf")


if __name__ == "__main__":
    print("Fig. 1 ...")
    figure1()
    print("Fig. 2 ...")
    figure2()
    print("Fig. 3 ...")
    figure3()
    print("done.")
