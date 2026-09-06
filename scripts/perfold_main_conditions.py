"""E.6 (revision plan, R2.5): explicit per-fold variability for the main conditions.

Figure 2 shows per-fold boxes only for the ablations and Table 6 only for the
scaling study; the reviewer asked for the same visibility for the four main
conditions, read against the 0.40-pp run-to-run floor. This script reads the
stored per_fold.json files and emits per-fold official metrics with mean, SD,
and range (max-min), plus a comparison of the two silver-vs-gold contrasts.

CPU-only, trains nothing.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/perfold_main_conditions.py
"""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

CONDITIONS = {
    'cagf_gold': Path('results_cv_silver/stratified/ktb_only'),
    'cagf_silver': Path('results_cv_silver/stratified/silver_unfiltered_pretrain_ktb_finetune'),
    'kazroberta_gold': Path('results_cv_kazroberta/gold_only'),
    'kazroberta_silver': Path('results_cv_kazroberta/silver_finetune'),
}
METRICS = ('Lemmas', 'UPOS', 'UFeats', 'AllTags')


def main() -> None:
    rows: list[dict] = []
    summary: dict = {}
    for name, cfg_dir in CONDITIONS.items():
        pf = json.loads((cfg_dir / 'per_fold.json').read_text())
        summary[name] = {}
        for m in METRICS:
            vals = [r['official'][m] * 100 for r in sorted(pf, key=lambda r: r['fold'])]
            mean = statistics.mean(vals)
            sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
            summary[name][m] = {
                'per_fold': [round(v, 2) for v in vals],
                'mean': round(mean, 2), 'sd': round(sd, 2),
                'min': round(min(vals), 2), 'max': round(max(vals), 2),
                'range': round(max(vals) - min(vals), 2),
            }
            for r, v in zip(sorted(pf, key=lambda x: x['fold']), vals):
                rows.append({'condition': name, 'fold': r['fold'], 'metric': m,
                             'official_f1': round(v, 2), 'best_epoch': r['best_epoch']})

    out_csv = Path('results_cv/perfold_main_conditions.csv')
    with open(out_csv, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=['condition', 'fold', 'metric',
                                           'official_f1', 'best_epoch'])
        w.writeheader()
        w.writerows(rows)
    out_json = Path('results_cv/perfold_main_conditions.json')
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                        encoding='utf-8')

    print(f'Wrote {out_csv} and {out_json}\n')
    for name in CONDITIONS:
        print(f'{name}:')
        for m in METRICS:
            s = summary[name][m]
            print(f"  {m:8s} mean={s['mean']:6.2f}  sd={s['sd']:5.2f}  "
                  f"range={s['min']:6.2f}..{s['max']:6.2f} ({s['range']:5.2f})")


if __name__ == '__main__':
    main()
