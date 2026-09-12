#!/usr/bin/env python3
"""Plot the silver-scaling curve from results_cv_silver/scaling_perfold_stats.json.

Produces:
  results_cv_silver/figures/scaling_curve_main.png  (UPOS + AllTags, log-x)
  results_cv_silver/figures/scaling_curve_all4.png  (all 4 metrics)
  results_cv_silver/figures/scaling_curve_main.svg  (vector for LaTeX)
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STATS = "results_cv_silver/scaling_perfold_stats.json"
FIGDIR = "results_cv_silver/figures"
os.makedirs(FIGDIR, exist_ok=True)

d = json.load(open(STATS))
tokens = np.array([b["tokens"] for b in d["budgets"]], dtype=float)
labels = [b["label"] for b in d["budgets"]]

METRICS = ["UPOS", "UFeats", "Lemmas", "AllTags"]
COLORS  = {"UPOS": "#1f77b4", "UFeats": "#ff7f0e", "Lemmas": "#2ca02c", "AllTags": "#d62728"}

def series(m):
    mean = np.array([b["mean"][m] for b in d["budgets"]])
    std  = np.array([b["std"][m]  for b in d["budgets"]])
    return mean, std

# ---- Figure 1: main (UPOS + AllTags) on log-x, with error bands ----
fig, ax = plt.subplots(figsize=(5.2, 3.6))
for m in ["UPOS", "AllTags"]:
    mean, std = series(m)
    ax.errorbar(tokens, mean, yerr=std, marker="o", capsize=3,
                color=COLORS[m], label=m, linewidth=1.6)
ax.set_xscale("log")
ax.set_xticks(tokens); ax.set_xticklabels(labels)
ax.set_xlabel("Silver-corpus size (tokens)")
ax.set_ylabel("Jack-knife F$_1$ (10-fold CV)")
ax.grid(True, which="both", alpha=0.25)
ax.legend(frameon=False, loc="lower right")
ax.set_title("Quality vs. silver-corpus size", fontsize=10)
fig.tight_layout()
for ext in ("png", "svg"):
    fig.savefig(f"{FIGDIR}/scaling_curve_main.{ext}", dpi=200)
plt.close(fig)

# ---- Figure 2: all 4 metrics ----
fig, ax = plt.subplots(figsize=(5.6, 4.0))
for m in METRICS:
    mean, std = series(m)
    ax.plot(tokens, mean, marker="o", color=COLORS[m], label=m, linewidth=1.6)
    ax.fill_between(tokens, mean - std, mean + std, color=COLORS[m], alpha=0.12)
ax.set_xscale("log")
ax.set_xticks(tokens); ax.set_xticklabels(labels)
ax.set_xlabel("Silver-corpus size (tokens)")
ax.set_ylabel("Jack-knife F$_1$ (10-fold CV)")
ax.grid(True, which="both", alpha=0.25)
ax.legend(frameon=False, loc="lower right", ncol=2)
ax.set_title("Scaling curve, all metrics", fontsize=10)
fig.tight_layout()
fig.savefig(f"{FIGDIR}/scaling_curve_all4.png", dpi=200)
plt.close(fig)

# ---- Figure 3: marginal gain per doubling ----
fig, ax = plt.subplots(figsize=(5.2, 3.4))
mid = tokens[1:]            # report gain at each step
gains = {m: np.diff(series(m)[0]) for m in METRICS}
width = 0.2
x = np.arange(len(mid))
for i, m in enumerate(METRICS):
    ax.bar(x + (i - 1.5) * width, gains[m] * 100, width, label=m, color=COLORS[m])
ax.set_xticks(x)
ax.set_xticklabels([f"{labels[i]}→{labels[i+1]}" for i in range(len(mid))])
ax.set_ylabel("Δ F$_1$ (percentage points)")
ax.axhline(0, color="k", linewidth=0.6)
ax.legend(frameon=False, ncol=2, fontsize=8)
ax.set_title("Marginal gain per scaling step", fontsize=10)
fig.tight_layout()
fig.savefig(f"{FIGDIR}/scaling_marginal_gain.png", dpi=200)
plt.close(fig)

print("Figures written to", FIGDIR)
for f in sorted(os.listdir(FIGDIR)):
    print(" ", f)
