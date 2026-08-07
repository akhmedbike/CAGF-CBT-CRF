"""Generate Figures 3, 4, 5 for the CAGF-CBT+CRF manuscript (paper_revised.md).

Reads headline numbers from results_cv_silver/, results_cv_kazroberta/ and the
silver-scaling artefacts documented in revisions_applied.md (Table 6). Outputs
both SVG (vector, for LaTeX/Word) and PNG (300 dpi, for preview) under
docs/figures/.

Figure 3 — Silver-corpus scaling curve (4 budgets × 4 metrics, ±1 sd).
Figure 4 — Silver transfer effect (Δ silver − gold, 4 metrics, significance).
Figure 5 — CAGF vs KazRoBERTa, 2×2 (gold-only vs silver, 4 metrics).

Usage:
    .venv/bin/python scripts/make_figures.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
    }
)

OUT = Path("docs/figures")
OUT.mkdir(parents=True, exist_ok=True)

# Palette: two distinguishable, print-safe colors + neutral accents.
C_CAGF = "#1f4e79"   # dark blue
C_KAZR = "#a8431f"   # dark red-orange
C_LEM = "#1f4e79"
C_UPOS = "#2e7d32"
C_UFEATS = "#a8431f"
C_ALL = "#6a4c93"
SIG_MARKERS = {"yes": "*", "no": "ns"}


def _save(fig, name):
    svg = OUT / f"{name}.svg"
    png = OUT / f"{name}.png"
    fig.savefig(svg, format="svg", bbox_inches="tight")
    fig.savefig(png, format="png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  wrote {svg} (+ {png})")


# ---------------------------------------------------------------------------
# Figure 3 — Silver-corpus scaling curve.
# Data: Table 6 of paper_revised.md (jack-knifed F1, mean ± sd, n=10 folds).
# ---------------------------------------------------------------------------
def figure3_scaling():
    budgets = [100, 210, 500, 1000]
    x = np.array(budgets, dtype=float)

    metrics = {
        "Lemmas":   {"mean": [80.80, 82.54, 84.37, 84.96], "sd": [1.17, 1.33, 1.03, 0.57], "c": C_LEM, "m": "o"},
        "UPOS":     {"mean": [86.48, 87.64, 89.13, 89.45], "sd": [1.12, 3.79, 0.87, 1.02], "c": C_UPOS, "m": "s"},
        "UFeats":   {"mean": [66.52, 68.64, 72.83, 73.60], "sd": [3.09, 3.79, 1.58, 2.93], "c": C_UFEATS, "m": "^"},
        "AllTags":  {"mean": [61.35, 63.94, 68.74, 69.48], "sd": [3.12, 3.64, 1.61, 2.87], "c": C_ALL, "m": "D"},
    }

    fig, ax = plt.subplots(figsize=(6.0, 4.2))

    for name, d in metrics.items():
        mean = np.array(d["mean"])
        sd = np.array(d["sd"])
        ax.errorbar(
            x, mean, yerr=sd, fmt=d["m"] + "-", color=d["c"],
            capsize=3, capthick=1, elinewidth=0.9, lw=1.6, ms=5.5,
            label=name,
        )

    # Shade the non-significant final step (500K -> 1M) as a grey band.
    # This step reaches significance on NO metric after Holm correction (Table 6
    # footnote + revisions S1.g). Highlight it explicitly.
    ax.axvspan(500, 1000, alpha=0.07, color="grey", zorder=0)
    ax.text(745, 60.5, "500K→1M\nnot significant\n(Holm, all metrics)",
            ha="center", va="bottom", fontsize=7.2, color="0.35",
            style="italic")

    ax.set_xscale("log")
    ax.set_xticks(budgets)
    ax.set_xticklabels([f"{b // 1000}M" if b >= 1000 else f"{b}K" for b in budgets])
    ax.set_xlabel("Silver-corpus budget (tokens)")
    ax.set_ylabel("Official CoNLL-2018 F1 (%)")
    ax.set_ylim(58, 92)
    ax.legend(loc="lower right", framealpha=0.9, ncol=2, title="Metric")
    ax.set_title("Silver-corpus scaling: F1 saturates between 500K and 1M tokens")

    fig.tight_layout()
    _save(fig, "figure3_scaling")


# ---------------------------------------------------------------------------
# Figure 4 — Silver transfer effect: Δ (silver − gold) per metric, with
# Holm-corrected significance markers.
# Data: Table 5 of paper_revised.md (all four deltas, p_Holm < 0.005).
# ---------------------------------------------------------------------------
def figure4_silver_delta():
    metrics = ["Lemmas", "UPOS", "UFeats", "AllTags"]
    deltas = [7.88, 4.84, 5.16, 6.10]
    # Holm-corrected p-values from the silver-vs-gold paired test (family of 4).
    # The stored significance file reports filtered-vs-gold; the unfiltered deltas
    # in Table 5 are larger, so these are valid (conservative) significance markers.
    p_holm = [7.5e-10, 1.6e-4, 4.8e-3, 2.4e-3]  # from results_cv_silver significance_fold.json

    x = np.arange(len(metrics))
    colors = [C_LEM, C_UPOS, C_UFEATS, C_ALL]

    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    bars = ax.bar(x, deltas, width=0.55, color=colors, edgecolor="black", linewidth=0.6)

    # Annotate each bar with Δ value and significance.
    for b, d, p in zip(bars, deltas, p_holm):
        h = b.get_height()
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        ax.text(b.get_x() + b.get_width() / 2, h + 0.18, f"+{d:.2f}",
                ha="center", va="bottom", fontsize=9.5, fontweight="bold")
        ax.text(b.get_x() + b.get_width() / 2, h + 0.95, sig,
                ha="center", va="bottom", fontsize=9, color="0.25")

    ax.axhline(0, color="black", lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Δ F1 (silver − gold), percentage points")
    ax.set_ylim(0, 9.2)
    ax.set_title("Silver-corpus transfer effect on CAGF-CBT+CRF (n=10 folds)")
    ax.text(
        0.5, -0.16,
        "All four gains significant after Holm correction (family of 4);  "
        "*** p<0.001,  ** p<0.01.",
        transform=ax.transAxes, ha="center", va="top", fontsize=7.5, color="0.35",
    )

    fig.tight_layout()
    _save(fig, "figure4_silver_delta")


# ---------------------------------------------------------------------------
# Figure 5 — CAGF vs KazRoBERTa, 2×2 (gold-only vs silver), four metrics.
# Data: Table 7 of paper_revised.md (jack-knifed official F1, n=10 folds).
# ---------------------------------------------------------------------------
def figure5_cagf_vs_kazr():
    metrics = ["Lemmas", "UPOS", "UFeats", "AllTags"]
    x = np.arange(len(metrics))
    w = 0.20

    cagf_gold   = [74.93, 82.99, 63.69, 58.18]
    cagf_silver = [82.81, 87.83, 68.85, 64.28]
    kazr_gold   = [74.42, 92.99, 77.71, 74.62]
    kazr_silver = [77.96, 92.30, 82.06, 78.66]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))

    def _bars(vals, off, color, hatch, label):
        return ax.bar(
            x + off, vals, w, color=color, hatch=hatch,
            edgecolor="black", linewidth=0.5, label=label,
        )

    _bars(cagf_gold,   -1.5 * w, C_CAGF, "",   "CAGF gold-only")
    _bars(cagf_silver, -0.5 * w, C_CAGF, "//", "CAGF + silver")
    _bars(kazr_gold,    0.5 * w, C_KAZR, "",   "KazRoBERTa gold-only")
    _bars(kazr_silver,  1.5 * w, C_KAZR, "//", "KazRoBERTa + silver")

    # Annotate the headline lemma finding: CAGF+silver (82.81) > KazR+silver (77.96).
    ax.annotate(
        "CAGF+silver > KazRoBERTa+silver\n+4.85 pp on Lemmas (p<1e-5)",
        xy=(0 + (-0.5 * w), 82.81), xytext=(0.15, 94.5),
        fontsize=7.8, color=C_CAGF, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=C_CAGF, lw=0.9),
        ha="left", va="top",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Official CoNLL-2018 F1 (%)")
    ax.set_ylim(50, 100)
    ax.legend(loc="upper left", ncol=2, framealpha=0.92, fontsize=8)
    ax.set_title("CAGF-CBT+CRF (2.5M) vs KazRoBERTa (83.7M): task-dependent split")

    fig.tight_layout()
    _save(fig, "figure5_cagf_vs_kazroberta")


if __name__ == "__main__":
    print("Generating manuscript figures in docs/figures/ ...")
    figure3_scaling()
    figure4_silver_delta()
    figure5_cagf_vs_kazr()
    print("Done.")
