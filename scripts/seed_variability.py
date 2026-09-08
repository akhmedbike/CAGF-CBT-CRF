"""R3.6: training-seed variability for the major conditions (task B analysis).

Two bases, chosen so that the three seeds are always compared like-for-like:

  * CAGF (gold-only / silver-transfer): pooled out-of-fold jack-knife scores,
    full 10-fold stratified protocol. seed 42 from results_cv_silver/stratified
    (the paper's runs), seeds 13/2024 from results_cv_silver/seed{13,2024}
    (task B, --train-seed; folds identical to seed 42).

  * KazRoBERTa (gold_only / silver_finetune): mean over folds {0, 3, 7} only --
    the pre-fixed fold subset used by task B for the seed replicates. seed 42
    is subsetted to the same folds from the full 10-fold run so all three
    seeds average exactly the same test sentences.

Reports, per condition and metric: per-seed values, mean, std, and max-min
spread across seeds, next to the silver-vs-gold effect for the same model.
Writes results_cv/seed_variability.json.
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

METRICS = ('Lemmas', 'UPOS', 'UFeats', 'AllTags')
KAZR_FOLDS = (0, 3, 7)  # fixed in advance for task B; never changed after results

CAGF_DIRS = {
    'ktb_only': {
        42: 'results_cv_silver/stratified/ktb_only',
        13: 'results_cv_silver/seed13/ktb_only',
        2024: 'results_cv_silver/seed2024/ktb_only',
    },
    'silver_unfiltered_pretrain_ktb_finetune': {
        42: 'results_cv_silver/stratified/silver_unfiltered_pretrain_ktb_finetune',
        13: 'results_cv_silver/seed13/silver_unfiltered_pretrain_ktb_finetune',
        2024: 'results_cv_silver/seed2024/silver_unfiltered_pretrain_ktb_finetune',
    },
}

KAZR_DIRS = {
    'gold_only': {
        42: 'results_cv_kazroberta/gold_only',
        13: 'results_cv_kazroberta/seed13_gold/gold_only',
        2024: 'results_cv_kazroberta/seed2024_gold/gold_only',
    },
    'silver_finetune': {
        42: 'results_cv_kazroberta/silver_finetune',
        13: 'results_cv_kazroberta/seed13_silver/silver_finetune',
        2024: 'results_cv_kazroberta/seed2024_silver/silver_finetune',
    },
}


def cagf_pooled(path: str) -> dict:
    j = json.loads(Path(path, 'jackknifed.json').read_text(encoding='utf-8'))
    return {k: v * 100 for k, v in j['official_jackknifed'].items() if k in METRICS}


def kazr_folds_mean(path: str) -> dict:
    entries = json.loads(Path(path, 'per_fold.json').read_text(encoding='utf-8'))
    sel = [e for e in entries if e['fold'] in KAZR_FOLDS]
    assert len(sel) == len(KAZR_FOLDS), f'{path}: folds {[e["fold"] for e in sel]} != {KAZR_FOLDS}'
    return {k: st.mean(e['official'][k] * 100 for e in sel) for k in METRICS}


def condition_report(loader, dirs) -> dict:
    per_seed = {str(seed): loader(p) for seed, p in sorted(dirs.items())}
    out = {'per_seed': per_seed, 'spread': {}, 'mean': {}, 'std': {}}
    for m in METRICS:
        vals = [per_seed[s][m] for s in per_seed]
        out['mean'][m] = st.mean(vals)
        out['std'][m] = st.stdev(vals)
        out['spread'][m] = max(vals) - min(vals)
    return out


def main() -> None:
    report = {'cagf_basis': 'pooled out-of-fold jack-knife, 10-fold stratified',
              'kazroberta_basis': f'mean over folds {list(KAZR_FOLDS)} (pre-fixed task-B subset)',
              'cagf': {}, 'kazroberta': {}, 'silver_effect_stability': {}}

    for cfg, dirs in CAGF_DIRS.items():
        report['cagf'][cfg] = condition_report(cagf_pooled, dirs)
    for cfg, dirs in KAZR_DIRS.items():
        report['kazroberta'][cfg] = condition_report(kazr_folds_mean, dirs)

    # the effect R3.6 asks about: does the silver-vs-gold delta move with the seed?
    for model, key in (('cagf', 'cagf'), ('kazroberta', 'kazroberta')):
        gold_cfg = 'ktb_only' if key == 'cagf' else 'gold_only'
        sil_cfg = 'silver_unfiltered_pretrain_ktb_finetune' if key == 'cagf' else 'silver_finetune'
        seeds = sorted(CAGF_DIRS[gold_cfg] if key == 'cagf' else KAZR_DIRS[gold_cfg])
        per_seed_effect = {}
        for s in seeds:
            g = report[key][gold_cfg]['per_seed'][str(s)]
            v = report[key][sil_cfg]['per_seed'][str(s)]
            per_seed_effect[str(s)] = {m: v[m] - g[m] for m in METRICS}
        spread = {m: max(e[m] for e in per_seed_effect.values()) - min(e[m] for e in per_seed_effect.values())
                  for m in METRICS}
        report['silver_effect_stability'][model] = {
            'per_seed_delta': per_seed_effect,
            'spread_of_delta': spread,
        }

    out_path = Path('results_cv/seed_variability.json')
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Wrote {out_path}\n')

    for key, label in (('cagf', 'CAGF (pooled jack-knife, 10-fold)'),
                       ('kazroberta', f'KazRoBERTa (mean over folds {list(KAZR_FOLDS)})')):
        print(f'== {label} ==')
        for cfg in report[key]:
            r = report[key][cfg]
            seeds = ' '.join(f's{s}={r["per_seed"][s][m]:.2f}' for s, m in
                             [(s, 'Lemmas') for s in r['per_seed']])
            print(f'  {cfg}:')
            for m in METRICS:
                vals = ' '.join(f'{r["per_seed"][s][m]:6.2f}' for s in r['per_seed'])
                print(f'    {m:8s} {vals}   spread={r["spread"][m]:.2f} std={r["std"][m]:.2f}')
        eff = report['silver_effect_stability'][key]
        print('  silver-minus-gold per seed:')
        for m in METRICS:
            vals = ' '.join(f'{eff["per_seed_delta"][s][m]:+6.2f}' for s in eff['per_seed_delta'])
            print(f'    {m:8s} {vals}   spread={eff["spread_of_delta"][m]:.2f}')
        print()


if __name__ == '__main__':
    main()
