"""Generate the conference-paper figures (Figs. 3-7) from SAAL raw records.

Reads results_saal_full/<language>/saal_raw.json and writes PNGs to
results_saal_full/figures/:
  fig3_tsc_aulc.png      normalized transformed-TSC AULC bars (5-50%)
  fig4_lemma_kazakh.png  lemma learning curves, Kazakh
  fig5_lemma_turkish.png lemma learning curves, Turkish
  fig6_lemma_kyrgyz.png  lemma learning curves, Kyrgyz
  fig7_b95.png           median b95 per strategy
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

X = [5.0, 10.0, 20.0, 30.0, 50.0]
BUDGET_KEYS = ['initial', '0.1', '0.2', '0.3', '0.5']
MAIN = ['random', 'entropy', 'novelty', 'baseline']
STYLE = {'random': ('Random', '#999999'), 'entropy': ('Entropy', '#1f77b4'),
         'novelty': ('Novelty', '#2ca02c'), 'baseline': ('Support-Aware', '#d62728')}
LANGS = ['kazakh', 'turkish', 'kyrgyz']


def load(out_dir: str) -> dict:
    data: dict = {}
    for lang_dir in sorted(Path(out_dir).iterdir()):
        raw = lang_dir / 'saal_raw.json'
        if not raw.exists():
            continue
        for r in json.loads(raw.read_text(encoding='utf-8')):
            data.setdefault((lang_dir.name, r['strategy']), {})[r['seed']] = r['checkpoints']
    return data


def curves(data, lang, strat, metric):
    ys = []
    for cps in data[(lang, strat)].values():
        ys.append([cps[k][metric] for k in BUDGET_KEYS])
    ys = np.array(ys)
    return ys.mean(axis=0), ys.std(axis=0)


def aulc(data, lang, strat, metric):
    m, _ = curves(data, lang, strat, metric)
    return float(np.trapezoid(m, X) / (X[-1] - X[0]))


def b95(data, lang, strat):
    out = []
    full_by_seed = {s: cps['full']['lemma_acc'] for s, cps in data[(lang, 'fullpool')].items()}
    for seed, cps in data[(lang, strat)].items():
        ys = [cps[k]['lemma_acc'] for k in BUDGET_KEYS]
        target = 0.95 * full_by_seed[seed]
        if ys[-1] < target:
            out.append(X[-1])
            continue
        for i in range(1, len(X)):
            if ys[i] >= target:
                x0, x1, y0, y1 = X[i - 1], X[i], ys[i - 1], ys[i]
                out.append(x0 if y1 == y0 else x0 + (target - y0) * (x1 - x0) / (y1 - y0))
                break
    return statistics.median(out) if out else float('nan')


def main() -> None:
    out_dir = sys.argv[1] if len(sys.argv) > 1 else 'results_saal_full'
    fig_dir = Path(out_dir) / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)
    data = load(out_dir)

    # Fig. 3: transformed-TSC AULC bars.
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.2
    for si, strat in enumerate(MAIN):
        vals = [aulc(data, lang, strat, 'tsc') for lang in LANGS]
        ax.bar(np.arange(3) + (si - 1.5) * width, vals, width, label=STYLE[strat][0],
               color=STYLE[strat][1])
    ax.set_xticks(range(3))
    ax.set_xticklabels(['Kazakh KTB', 'Turkish IMST', 'Kyrgyz KTMU'])
    ax.set_ylabel('Normalized transformed-TSC AULC (%)')
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(fig_dir / 'fig3_tsc_aulc.png', dpi=200)
    plt.close(fig)

    # Figs. 4-6: lemma learning curves.
    for fig_no, lang, title in ((4, 'kazakh', 'UD_Kazakh-KTB'),
                                (5, 'turkish', 'UD_Turkish-IMST'),
                                (6, 'kyrgyz', 'UD_Kyrgyz-KTMU')):
        fig, ax = plt.subplots(figsize=(6, 4))
        for strat in MAIN:
            m, sd = curves(data, lang, strat, 'lemma_acc')
            label = STYLE[strat][0]
            ax.errorbar(X, m, yerr=sd, marker='o', ms=4, capsize=3,
                        label=label, color=STYLE[strat][1])
        ax.set_xlabel('Labeling budget (% of pool words)')
        ax.set_ylabel('Exact lemma accuracy (%)')
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(fig_dir / f'fig{fig_no}_lemma_{lang}.png', dpi=200)
        plt.close(fig)

    # Fig. 7: median b95.
    fig, ax = plt.subplots(figsize=(7, 4))
    for si, strat in enumerate(MAIN):
        vals = [b95(data, lang, strat) for lang in LANGS]
        ax.bar(np.arange(3) + (si - 1.5) * width, vals, width, label=STYLE[strat][0],
               color=STYLE[strat][1])
    ax.set_xticks(range(3))
    ax.set_xticklabels(['Kazakh KTB\n(censored at 50)', 'Turkish IMST', 'Kyrgyz KTMU'])
    ax.set_ylabel('Median b95 (% of pool words)')
    ax.set_ylim(0, 55)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(fig_dir / 'fig7_b95.png', dpi=200)
    plt.close(fig)

    print(f'Figures written to {fig_dir}/')


if __name__ == '__main__':
    main()
