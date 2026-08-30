"""Aggregate SAAL raw records into the paper's tables (branch siccis-malta).

Reads results_saal_full/<language>/saal_raw.json produced by
scripts/run_weight_sensitivity.py and emits:

* Table 2  — transformed TSC at 10/20/30/50% budgets (mean +/- SD, 4 strategies
             x 3 languages);
* Table 3  — normalized AULC over the 5-50% range (lemma / TSC / UPOS);
* Table 4  — paired Support-Aware vs Entropy TSC comparison at 20% with
             Wilcoxon signed-rank + Holm correction across languages;
* b95      — median interpolated budget to reach 95% of the full-pool
             surrogate lemma accuracy;
* the Kazakh Eq. (7)-(8) weight-sensitivity table.

Everything is printed and saved to results_saal_full/paper_numbers.md and
paper_numbers.json for the manuscript edit.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

X = [5.0, 10.0, 20.0, 30.0, 50.0]
BUDGET_KEYS = ['0.1', '0.2', '0.3', '0.5']
MAIN = ['random', 'entropy', 'novelty', 'baseline']
ABLATION = ['baseline', 'equal', 'support-heavy', 'novelty-heavy',
            'uncertainty-heavy', 'mean-only', 'balanced']
LANGS = ['kazakh', 'turkish', 'kyrgyz']
LABEL = {'random': 'Random', 'entropy': 'Entropy', 'novelty': 'Novelty',
         'baseline': 'Support-Aware'}


def load(out_dir: str) -> dict:
    data: dict = {}
    for lang_dir in sorted(Path(out_dir).iterdir()):
        raw = lang_dir / 'saal_raw.json'
        if not raw.exists():
            continue
        for r in json.loads(raw.read_text(encoding='utf-8')):
            data.setdefault((lang_dir.name, r['strategy']), {})[r['seed']] = r['checkpoints']
    return data


def series(checkpoints: dict, metric: str) -> list[float] | None:
    ys = [checkpoints.get(k, {}).get(metric) for k in ['initial'] + BUDGET_KEYS]
    return ys if all(v is not None for v in ys) else None


def aulc_values(data: dict, language: str, strategy: str, metric: str) -> list[float]:
    out = []
    for cps in data[(language, strategy)].values():
        ys = series(cps, metric)
        if ys is not None:
            out.append(float(np.trapezoid(ys, X) / (X[-1] - X[0])))
    return out


def mean_sd(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def holm(pvals: list[float]) -> list[float]:
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m = len(pvals)
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, pvals[i] * (m - rank)))
        adj[i] = running
    return adj


def b95(data: dict, language: str, strategy: str) -> list[float]:
    out = []
    full_by_seed = {s: cps['full']['lemma_acc'] for s, cps in data[(language, 'fullpool')].items()}
    for seed, cps in data[(language, strategy)].items():
        ys = series(cps, 'lemma_acc')
        if ys is None or seed not in full_by_seed:
            continue
        target = 0.95 * full_by_seed[seed]
        if ys[-1] < target:
            out.append(X[-1])  # censored at the top of the studied range
            continue
        for i in range(1, len(X)):
            if ys[i] >= target:
                x0, x1, y0, y1 = X[i - 1], X[i], ys[i - 1], ys[i]
                out.append(x0 if y1 == y0 else x0 + (target - y0) * (x1 - x0) / (y1 - y0))
                break
    return out


def fmt(values: list[float]) -> str:
    m, sd = mean_sd(values)
    return f'{m:.2f} ± {sd:.2f}'


def main() -> None:
    out_dir = sys.argv[1] if len(sys.argv) > 1 else 'results_saal_full'
    data = load(out_dir)
    lines: list[str] = []
    numbers: dict = {}

    def emit(text: str = '') -> None:
        lines.append(text)
        print(text)

    langs = [l for l in LANGS
             if all((l, s) in data for s in MAIN) and (l, 'fullpool') in data]
    skipped = [l for l in LANGS if l not in langs]
    if skipped:
        print(f'[incomplete, skipped: {", ".join(skipped)}]')

    # ---------------- Table 2: transformed TSC by budget ----------------
    emit('## Table 2. Transformed TSC across annotation budgets, mean ± SD (%)')
    emit('| Language | Budget | Random | Entropy | Novelty | Support-Aware |')
    emit('|---|---|---|---|---|---|')
    t2 = {}
    for lang in langs:
        for bkey, bx in zip(BUDGET_KEYS, [10, 20, 30, 50]):
            row = []
            for strat in MAIN:
                vals = [cps[bkey]['tsc'] for cps in data[(lang, strat)].values() if bkey in cps]
                t2[(lang, bx, strat)] = fmt(vals)
                row.append(fmt(vals))
            emit(f"| {lang.capitalize()} | {bx}% | " + ' | '.join(row) + ' |')
    numbers['table2'] = {f'{l}|{b}|{s}': v for (l, b, s), v in t2.items()}

    # ---------------- Table 3: normalized AULC ----------------
    emit('\n## Table 3. Normalized AULC over the 5–50% annotation range (%)')
    emit('| Language | Lemma: Random | Lemma: Entropy | Lemma: Support-aware | '
         'TSC: Random | TSC: Support-aware | UPOS: Entropy | UPOS: Support-aware |')
    emit('|---|---|---|---|---|---|---|---|')
    t3 = {}
    for lang in langs:
        cells = []
        for metric, strats in (('lemma_acc', ['random', 'entropy', 'baseline']),
                               ('tsc', ['random', 'baseline']),
                               ('upos_acc', ['entropy', 'baseline'])):
            for strat in strats:
                vals = aulc_values(data, lang, strat, metric)
                t3[f'{lang}|{metric}|{strat}'] = fmt(vals)
                cells.append(f'{mean_sd(vals)[0]:.2f}')
        emit(f"| {lang.capitalize()} | " + ' | '.join(cells) + ' |')
    numbers['table3'] = t3
    for lang in langs:
        for metric, strats in (('lemma_acc', ['random', 'entropy', 'novelty', 'baseline']),
                               ('tsc', ['random', 'entropy', 'novelty', 'baseline']),
                               ('upos_acc', ['random', 'entropy', 'novelty', 'baseline'])):
            for strat in strats:
                numbers.setdefault('aulc_all', {})[f'{lang}|{metric}|{strat}'] = \
                    fmt(aulc_values(data, lang, strat, metric))

    # ---------------- Table 4: paired SA vs Entropy @20% ----------------
    emit('\n## Table 4. Paired SA vs Entropy, transformed TSC @20% (Wilcoxon + Holm)')
    diffs, pvals, langs_done = [], [], []
    for lang in langs:
        seeds = sorted(set(data[(lang, 'baseline')]) & set(data[(lang, 'entropy')]))
        sa = [data[(lang, 'baseline')][s]['0.2']['tsc'] for s in seeds]
        ent = [data[(lang, 'entropy')][s]['0.2']['tsc'] for s in seeds]
        d = statistics.mean(sa) - statistics.mean(ent)
        diffs.append(d)
        pvals.append(float(wilcoxon(sa, ent).pvalue) if d != 0 else 1.0)
        langs_done.append(lang)
    adj = holm(pvals)
    emit('| Language | SA − Entropy (pp) | Holm-adjusted p | n seeds |')
    emit('|---|---|---|---|')
    for lang, d, p in zip(langs_done, diffs, adj):
        emit(f'| {lang.capitalize()} | {d:+.2f} | {p:.4f} | '
             f'{len(set(data[(lang, "baseline")]) & set(data[(lang, "entropy")]))} |')
    numbers['table4'] = {lang: {'diff': round(d, 2), 'p_holm': round(p, 4)}
                         for lang, d, p in zip(langs_done, diffs, adj)}

    # ---------------- b95 ----------------
    emit('\n## b95: median budget to reach 95% of full-pool lemma accuracy (%)')
    emit('| Language | Random | Entropy | Novelty | Support-Aware | full-pool lemma acc |')
    emit('|---|---|---|---|---|---|')
    for lang in langs:
        cells = []
        for strat in MAIN:
            vals = b95(data, lang, strat)
            cells.append(f'{statistics.median(vals):.1f}' if vals else '--')
        full = [cps['full']['lemma_acc'] for cps in data[(lang, 'fullpool')].values()]
        numbers.setdefault('b95', {})[lang] = {
            strat: (round(statistics.median(b95(data, lang, strat)), 1) if b95(data, lang, strat) else None)
            for strat in MAIN}
        emit(f"| {lang.capitalize()} | " + ' | '.join(cells) + f' | {statistics.mean(full):.2f} |')

    # ---------------- inline values used by the running text ----------------
    inline = {}
    for lang in langs:
        for strat in MAIN:
            for metric in ('lemma_acc', 'upos_acc', 'tsc'):
                for bkey in BUDGET_KEYS:
                    vals = [cps[bkey][metric] for cps in data[(lang, strat)].values() if bkey in cps]
                    inline[f'{lang}|{strat}|{metric}@{bkey}'] = fmt(vals)
        full = [cps['full']['lemma_acc'] for cps in data[(lang, 'fullpool')].values()]
        inline[f'{lang}|fullpool|lemma'] = fmt(full)
    numbers['inline'] = inline

    # ---------------- Kazakh weight sensitivity ----------------
    if all((('kazakh', s) in data) for s in ABLATION):
        emit('\n## Kazakh Eq. (7)-(8) weight sensitivity @20% (liblinear_ovr surrogate)')
        emit('| Config | Transformed TSC @20% | Δ | Lemma acc @20% | Δ |')
        emit('|---|---|---|---|---|')
        base_t = statistics.mean([cps['0.2']['tsc'] for cps in data[('kazakh', 'baseline')].values()])
        base_l = statistics.mean([cps['0.2']['lemma_acc'] for cps in data[('kazakh', 'baseline')].values()])
        sens = {}
        for strat in ABLATION:
            tvals = [cps['0.2']['tsc'] for cps in data[('kazakh', strat)].values()]
            lvals = [cps['0.2']['lemma_acc'] for cps in data[('kazakh', strat)].values()]
            dt = statistics.mean(tvals) - base_t
            dl = statistics.mean(lvals) - base_l
            sens[strat] = {'tsc': fmt(tvals), 'dt': round(dt, 2), 'lemma': fmt(lvals), 'dl': round(dl, 2)}
            emit(f"| {strat} | {fmt(tvals)} | {dt:+.2f} | {fmt(lvals)} | {dl:+.2f} |")
        numbers['sensitivity'] = sens

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path(out_dir, 'paper_numbers.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    Path(out_dir, 'paper_numbers.json').write_text(
        json.dumps(numbers, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"\nSaved: {out_dir}/paper_numbers.md, {out_dir}/paper_numbers.json")


if __name__ == '__main__':
    main()
