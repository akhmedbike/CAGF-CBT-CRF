"""Run the SAAL acquisition-weight sensitivity ablation (branch siccis-malta).

Executes the offline annotation simulation of cagf.saal for the reference
strategies (Random / Entropy / Novelty) and the Eq. (7)-(8) weight
configurations defined in configs/saal_kazakh_ablation.yaml, then writes:

  <out-dir>/weight_sensitivity_raw.json   one record per (strategy, seed)
  <out-dir>/weight_sensitivity_summary.csv  mean +/- SD per checkpoint/metric

Each (strategy, seed) pair is an independent task; completed pairs are kept
when the script is re-run, so the ablation can resume after interruption.
Mean +/- SD over seeds is the reported measure (no significance testing:
this is a sensitivity illustration, not a better/worse claim).

Usage:
    .venv/bin/python scripts/run_weight_sensitivity.py \
        --config configs/saal_kazakh_ablation.yaml \
        --out-dir results_weight_sensitivity
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Must precede Pool creation: spawn workers inherit the environment before
# numpy/BLAS initialise, preventing thread oversubscription across workers.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

from cagf.saal import (AcquisitionWeights, CharHashVectorizer, build_split,
                       load_corpus, run_simulation)

_WORKER_CTX: dict = {}


def _init_worker(cfg: dict) -> None:
    sur = cfg['surrogate']
    _WORKER_CTX['cfg'] = cfg
    _WORKER_CTX['sentences'] = load_corpus(cfg['data']['corpus_paths'])
    _WORKER_CTX['vectorizer'] = CharHashVectorizer(dim=sur['feature_dim'],
                                                   ngram_range=tuple(sur['ngram_range']))


def _run_task(task: tuple[str, int]) -> dict:
    name, seed = task
    cfg = _WORKER_CTX['cfg']
    sentences = _WORKER_CTX['sentences']
    vectorizer = _WORKER_CTX['vectorizer']
    sim = cfg['simulation']
    sur = cfg['surrogate']
    entry = cfg['strategies'][name]
    aw = entry.get('acquisition_weights', [0.25, 0.20, 0.30, 0.15, 0.10])
    sw = entry.get('sentence_weights', [0.8, 0.2])
    weights = AcquisitionWeights(hu=aw[0], hr=aw[1], qpT=aw[2], nsuf=aw[3], nlex=aw[4],
                                 sentence_mean=sw[0], sentence_max=sw[1])
    kind = name if name in ('random', 'entropy', 'novelty') else 'support_aware'
    surrogate_kwargs = {'solver': sur.get('solver', 'multinomial'),
                        'c': sur.get('c', 1.0), 'max_iter': sur.get('max_iter', 500)}
    split = build_split(sentences, seed, sim['pool_fraction'], sim['initial_fraction'])
    t0 = time.time()
    checkpoints = run_simulation(sentences, split, kind, list(sim['budgets']), weights,
                                 vectorizer, surrogate_kwargs)
    return {'strategy': name, 'seed': seed, 'checkpoints': checkpoints,
            'weights': weights.describe() if kind == 'support_aware' else kind,
            'seconds': round(time.time() - t0, 1)}


def summarize(records: list[dict], out_dir: Path) -> None:
    metrics = ('tsc', 'lemma_acc', 'upos_acc')
    strategies = []
    for r in records:
        if r['strategy'] not in strategies:
            strategies.append(r['strategy'])
    rows = []
    for strategy in strategies:
        cps: dict[str, list[dict]] = {}
        for r in records:
            if r['strategy'] != strategy:
                continue
            for cp, vals in r['checkpoints'].items():
                cps.setdefault(cp, []).append(vals)
        for cp in sorted(cps, key=lambda c: (c != 'initial', float(c) if c != 'initial' else -1.0)):
            for metric in metrics:
                values = [v[metric] for v in cps[cp]]
                rows.append({'strategy': strategy, 'checkpoint': cp, 'metric': metric,
                             'mean': round(statistics.mean(values), 2),
                             'sd': round(statistics.stdev(values), 2) if len(values) > 1 else 0.0,
                             'n_seeds': len(values)})
    csv_path = out_dir / 'weight_sensitivity_summary.csv'
    with open(csv_path, 'w', encoding='utf-8') as fh:
        fh.write('strategy,checkpoint,metric,mean,sd,n_seeds\n')
        for row in rows:
            fh.write(f"{row['strategy']},{row['checkpoint']},{row['metric']},"
                     f"{row['mean']},{row['sd']},{row['n_seeds']}\n")
    print(f'Summary CSV: {csv_path}')

    # Markdown view of the 20% checkpoint: the deliverable table for the paper.
    budgets = sorted({cp for row in rows if row['checkpoint'] != 'initial' for cp in [row['checkpoint']]})
    main_cp = budgets[-1] if budgets else 'initial'
    by_key = {(row['strategy'], row['metric']): row for row in rows if row['checkpoint'] == main_cp}
    print(f'\n### Checkpoint {main_cp} (mean +/- SD over seeds)\n')
    print('| Strategy | Transformed TSC | Lemma acc | UPOS acc |')
    print('|---|---|---|---|')
    for strategy in strategies:
        cells = []
        for metric in metrics:
            row = by_key.get((strategy, metric))
            cells.append(f"{row['mean']:.2f} ± {row['sd']:.2f}" if row else '--')
        print(f"| {strategy} | {cells[0]} | {cells[1]} | {cells[2]} |")
    if 'baseline' in strategies:
        for metric_name, label in (('tsc', 'TSC'), ('lemma_acc', 'lemma acc')):
            base = by_key.get(('baseline', metric_name))
            if not base:
                continue
            deltas = []
            for strategy in strategies:
                if strategy in ('baseline', 'random', 'entropy', 'novelty'):
                    continue
                row = by_key.get((strategy, metric_name))
                if row:
                    deltas.append((strategy, round(row['mean'] - base['mean'], 2)))
            worst = max(deltas, key=lambda d: abs(d[1])) if deltas else None
            if worst is not None:
                print(f"\nMax |delta| vs baseline, {label} @ {main_cp}: {worst[1]:+.2f} pp ({worst[0]})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', default='configs/saal_kazakh_ablation.yaml')
    ap.add_argument('--out-dir', default='results_weight_sensitivity')
    ap.add_argument('--strategies', nargs='+', default=None,
                    help='subset of strategy names from the config (default: all)')
    ap.add_argument('--seeds', nargs='+', type=int, default=None,
                    help='subset of seeds (default: all from the config)')
    ap.add_argument('--workers', type=int, default=min(8, os.cpu_count() or 1))
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    strategy_names = list(cfg['strategies'])
    if args.strategies:
        unknown = set(args.strategies) - set(strategy_names)
        if unknown:
            ap.error(f'unknown strategies {sorted(unknown)}; available: {strategy_names}')
        strategy_names = args.strategies
    seeds = args.seeds or list(cfg['simulation']['seeds'])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / 'weight_sensitivity_raw.json'
    records: list[dict] = []
    done: set[tuple[str, int]] = set()
    if raw_path.exists():
        records = json.loads(raw_path.read_text(encoding='utf-8'))
        done = {(r['strategy'], r['seed']) for r in records}
        print(f'Resuming: {len(done)} (strategy, seed) runs already complete.')

    tasks = [(name, seed) for name in strategy_names for seed in seeds
             if (name, seed) not in done]
    print(f'{len(tasks)} runs to do: {len(strategy_names)} strategies x {len(seeds)} seeds.')

    if tasks:
        import multiprocessing as mp
        ctx = mp.get_context('spawn')
        completed = 0
        with ctx.Pool(args.workers, initializer=_init_worker, initargs=(cfg,)) as pool:
            for record in pool.imap_unordered(_run_task, tasks):
                records.append(record)
                completed += 1
                raw_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
                print(f"[{len(done) + completed}/{len(done) + len(tasks)}] "
                      f"{record['strategy']} seed={record['seed']} "
                      f"@0.2 TSC={record['checkpoints'].get('0.2', {}).get('tsc', float('nan')):.2f} "
                      f"lemma={record['checkpoints'].get('0.2', {}).get('lemma_acc', float('nan')):.2f} "
                      f"({record['seconds']}s)", flush=True)

    summarize(records, out_dir)


if __name__ == '__main__':
    main()
