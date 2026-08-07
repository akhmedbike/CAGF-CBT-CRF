"""Paired significance testing for ablation / transfer experiments.

Supports two units of observation (``--unit``):

  * ``seed`` (default, legacy) -- each row is one training seed of a single
    train/dev/test split. n = number of seeds (typically 3-5). At n=3 Wilcoxon
    cannot reach p<0.05 (minimum 0.25) and Cohen's d is unstable; this mode is
    kept for backward compatibility with the single-split results in
    ``results/ablation_raw.json``.
  * ``fold`` -- each row is one cross-validation fold. n = k (typically 10).
    Paired observations are the *same fold* across two configurations, so the
    pairing removes fold difficulty as a confound. This is the recommended mode
    for ``results_cv/*/cv_raw.json``: n=10 makes Wilcoxon and Cohen's d
    well-defined, and the per-fold official CoNLL-2018 metrics (Lemmas/UPOS/
    UFeats/AllTags) become available for the first time.

For a single config-vs-config comparison it reports, per task/metric:
  * paired t-test  (parametric, mean difference)
  * Wilcoxon signed-rank (non-parametric; robust on small n)
  * Cohen's d  (standardized effect size; small>=0.2, medium>=0.5, large>=0.8)
  * bootstrap 95% CI of the mean difference (fold mode; 10 000 resamples,
    percentile method). In the paper this is more informative than the p-value.

For multiple comparisons (``--holm``), it runs the full pairwise matrix of
ablation configs x tasks and applies the Holm-Bonferroni step-down correction,
reporting p-values both raw and adjusted. This is what Reviewer 4 (comment 4)
asked for: "add formal hypothesis testing" with control for the family-wise
error rate across the 3 tasks x multiple ablations tested.

Output is plain text (stdout) plus an optional JSON dump for the tables script.

Usage
-----
    # single comparison (legacy single-split, by seed)
    python scripts/significance_test.py --config-a full_model --config-b transformer_only

    # full ablation matrix with Holm correction + JSON dump
    python scripts/significance_test.py --holm --json-out results/significance.json

    # cross-validation fold statistics (recommended for the paper)
    python scripts/significance_test.py --results results_cv/stratified/cv_raw.json \\
        --unit fold --holm --tasks official.UPOS official.UFeats \\
        --json-out results_cv/stratified/significance_fold.json
"""
from __future__ import annotations
import argparse
import json
import math
import random
from pathlib import Path
from typing import List, Sequence, Tuple
from scipy import stats


def _dotted_get(obj: dict, dotted_key: str):
    """Resolve a possibly-dotted key like 'official.UPOS' against a result row.

    Seed-mode rows store task metrics as dicts: row['upos']['f1']. Fold-mode
    rows additionally store row['official']['UPOS'] etc. To let --tasks address
    either uniformly, we treat a key with no dot as {task}.{metric} (legacy)
    and a key with a dot as a literal path into the row.
    """
    if '.' in dotted_key:
        cur = obj
        for part in dotted_key.split('.'):
            cur = cur[part]
        return cur
    # legacy: bare task name -> the row's {task} dict (caller adds metric)
    return obj[dotted_key]


def load_scores(results: list[dict], config_name: str, task: str,
                metric: str = 'f1', unit: str = 'seed') -> list[float]:
    """Pull the per-observation score list for one config/task/metric.

    In seed mode (legacy), each result row is one seed and we return
    ``row[task][metric]`` for rows of the requested config.

    In fold mode, each result row is one fold. ``task`` may be a bare task
    name (``'upos'`` -> row['upos'][metric]) or a dotted official-metric path
    (``'official.UPOS'`` -> row['official']['UPOS'], metric ignored). Rows are
    returned sorted by fold so the pairing across configs is fold-aligned.
    """
    rows = [r for r in results if r['config'] == config_name]
    if unit == 'fold':
        rows = sorted(rows, key=lambda r: r.get('fold', 0))
    if task.startswith('official.'):
        return [float(_dotted_get(r, task)) for r in rows]
    return [float(r[task][metric]) for r in rows]


def bootstrap_ci_diff(a: Sequence[float], b: Sequence[float],
                      n_boot: int = 10000, alpha: float = 0.05,
                      seed: int = 12345) -> Tuple[float, float]:
    """Percentile bootstrap 95% CI of the mean paired difference (a - b).

    Resamples the per-observation differences with replacement, takes the mean
    of each resample, and returns the (alpha/2, 1-alpha/2) percentiles. This
    is the metric to quote in the paper: "the difference is X pp, 95% CI [lo, hi]".
    """
    if len(a) != len(b) or len(a) < 2:
        return (float('nan'), float('nan'))
    diffs = [x - y for x, y in zip(a, b)]
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        sample = [diffs[rng.randrange(len(diffs))] for _ in range(len(diffs))]
        means.append(sum(sample) / len(sample))
    means.sort()
    lo_idx = int((alpha / 2) * n_boot)
    hi_idx = int((1 - alpha / 2) * n_boot)
    return (means[lo_idx], means[hi_idx])


def cohens_d_paired(a: Sequence[float], b: Sequence[float]) -> float:
    """Cohen's d for paired samples (standardized mean of per-seed differences)."""
    if len(a) != len(b) or len(a) < 2:
        return float('nan')
    diffs = [x - y for x, y in zip(a, b)]
    mean = sum(diffs) / len(diffs)
    # sample std of differences (ddof=1)
    var = sum((d - mean) ** 2 for d in diffs) / (len(diffs) - 1)
    sd = math.sqrt(var)
    if sd == 0:
        return 0.0 if mean == 0 else float('inf')
    return mean / sd


def effect_label(d: float) -> str:
    if math.isnan(d):
        return 'n/a'
    ad = abs(d)
    if ad < 0.2:
        return 'negligible'
    if ad < 0.5:
        return 'small'
    if ad < 0.8:
        return 'medium'
    return 'large'


def holm_bonferroni(pvals: List[float], alpha: float = 0.05) -> List[float]:
    """Step-down Holm-Bonferroni: returns adjusted p-values aligned to input order."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        corrected = min(1.0, pvals[idx] * (m - rank))
        # Holm is step-down: adjusted p cannot decrease as rank increases
        running_max = max(running_max, corrected)
        adj[idx] = running_max
    return adj


def paired_test(a: Sequence[float], b: Sequence[float]) -> dict:
    t_stat, t_p = stats.ttest_rel(a, b)
    try:
        w_stat, w_p = stats.wilcoxon(a, b)
    except ValueError:
        w_stat, w_p = (float('nan'), float('nan'))
    d = cohens_d_paired(a, b)
    ci_lo, ci_hi = bootstrap_ci_diff(a, b)
    return {
        'mean_a': sum(a) / len(a), 'mean_b': sum(b) / len(b),
        'mean_diff': sum(a) / len(a) - sum(b) / len(b),
        'n': len(a), 't_stat': float(t_stat), 't_p': float(t_p),
        'wilcoxon_w': float(w_stat), 'wilcoxon_p': float(w_p),
        'cohens_d': d, 'effect': effect_label(d),
        'bootstrap_ci_lo': ci_lo, 'bootstrap_ci_hi': ci_hi,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', default='results/ablation_raw.json')
    ap.add_argument('--config-a', default='full_model')
    ap.add_argument('--config-b', default='transformer_only')
    ap.add_argument('--metric', default='f1')
    ap.add_argument('--unit', choices=['seed', 'fold'], default='seed',
                    help='observation unit: seed (legacy single-split) or fold (CV)')
    ap.add_argument('--alpha', type=float, default=0.05)
    ap.add_argument('--holm', action='store_true',
                    help='run the full pairwise ablation matrix with Holm-Bonferroni correction')
    ap.add_argument('--tasks', nargs='+', default=['lemma', 'upos', 'grammeme'],
                    help='task names; in fold mode may be dotted official paths '
                         'like official.UPOS, official.UFeats, official.AllTags')
    ap.add_argument('--reference', default=None,
                    help='reference config for Holm comparisons; default: full_model '
                         'if present, else the alphabetically-first config. For the '
                         'silver-transfer CV the reference should be ktb_only so the '
                         'matrix answers "does silver help?" rather than "does the '
                         'baseline help?".')
    ap.add_argument('--json-out', default=None)
    args = ap.parse_args()

    results = json.loads(Path(args.results).read_text(encoding='utf-8'))
    configs = sorted({r['config'] for r in results})
    unit_label = 'seed' if args.unit == 'seed' else 'fold'

    if not args.holm:
        # ---- single comparison (legacy behaviour, now unit-aware) ----
        print(f'Paired significance test ({unit_label}-wise): '
              f'{args.config_a} vs {args.config_b} (metric={args.metric})\n')
        for task in args.tasks:
            a = load_scores(results, args.config_a, task, args.metric, unit=args.unit)
            b = load_scores(results, args.config_b, task, args.metric, unit=args.unit)
            if len(a) != len(b) or len(a) < 2:
                print(f'[{task}] SKIPPED: need >=2 matched {unit_label}s '
                      f'(a={len(a)}, b={len(b)}).\n')
                continue
            r = paired_test(a, b)
            sig = 'SIGNIFICANT' if r['t_p'] < args.alpha else 'not significant'
            print(f"[{task}] {args.config_a} mean={r['mean_a']:.4f}  "
                  f"{args.config_b} mean={r['mean_b']:.4f}  (n={r['n']} {unit_label}s)")
            print(f"          paired t-test:      t={r['t_stat']:.4f}, p={r['t_p']:.4f}  ({sig} at alpha={args.alpha})")
            print(f"          Wilcoxon signed-rank: W={r['wilcoxon_w']:.4f}, p={r['wilcoxon_p']:.4f}")
            print(f"          Cohen's d:           {r['cohens_d']:.3f}  ({r['effect']} effect)")
            print(f"          bootstrap 95% CI of diff: [{r['bootstrap_ci_lo']:.4f}, {r['bootstrap_ci_hi']:.4f}]\n")
        return

    # ---- full matrix with Holm correction ----
    if args.reference is not None:
        reference = args.reference
        if reference not in configs:
            raise SystemExit(f'--reference {reference!r} not found in results; '
                             f'available: {configs}')
    elif 'full_model' in configs:
        reference = 'full_model'
    else:
        print(f'NOTE: full_model not in results; using {configs[0]!r} as reference. '
              f'Pass --reference to override (e.g. ktb_only for the silver-transfer CV).')
        reference = configs[0]
    comparisons = [c for c in configs if c != reference]
    print(f'Holm-Bonferroni multiple-comparison correction ({unit_label}-wise, metric={args.metric})')
    print(f'Reference config: {reference}   family-size: {len(comparisons)} configs x {len(args.tasks)} tasks '
          f'= {len(comparisons) * len(args.tasks)} hypotheses\n')

    family = []  # (task, config, raw_p, test_dict)
    for task in args.tasks:
        ref_scores = load_scores(results, reference, task, args.metric, unit=args.unit)
        for cfg in comparisons:
            cfg_scores = load_scores(results, cfg, task, args.metric, unit=args.unit)
            if len(ref_scores) != len(cfg_scores) or len(ref_scores) < 2:
                continue
            t = paired_test(ref_scores, cfg_scores)
            family.append((task, cfg, t['t_p'], t))

    raw_ps = [x[2] for x in family]
    adj_ps = holm_bonferroni(raw_ps, alpha=args.alpha)

    print(f"{'task':<18} {'comparison':<32} {'raw p':>9} {'Holm p':>9} {'d':>7} {'effect':<11} {'95% CI diff':<22} {'sig'}")
    print('-' * 120)
    json_out = []
    for (task, cfg, raw_p, t), holm_p in zip(family, adj_ps):
        sig = '*' if holm_p < args.alpha else ''
        cmp_label = f'{reference} vs {cfg}'
        ci_str = f"[{t['bootstrap_ci_lo']:+.4f}, {t['bootstrap_ci_hi']:+.4f}]"
        print(f"{task:<18} {cmp_label:<32} {raw_p:>9.4f} {holm_p:>9.4f} "
              f"{t['cohens_d']:>7.3f} {t['effect']:<11} {ci_str:<22} {sig}")
        json_out.append({'task': task, 'reference': reference, 'config': cfg,
                         'unit': args.unit, 'n': t['n'],
                         'mean_ref': t['mean_a'], 'mean_cfg': t['mean_b'],
                         'mean_diff': t['mean_diff'],
                         't_stat': t['t_stat'], 'p_raw': raw_p, 'p_holm': holm_p,
                         'cohens_d': t['cohens_d'], 'effect': t['effect'],
                         'bootstrap_ci_lo': t['bootstrap_ci_lo'],
                         'bootstrap_ci_hi': t['bootstrap_ci_hi'],
                         'significant_holm': holm_p < args.alpha, 'alpha': args.alpha})

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(json_out, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f"\nJSON written to {args.json_out}")


if __name__ == '__main__':
    main()
