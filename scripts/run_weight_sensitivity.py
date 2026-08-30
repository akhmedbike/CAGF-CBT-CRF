"""Run the SAAL offline annotation simulation matrix (branch siccis-malta).

Two configs are supported:

* configs/saal_kazakh_ablation.yaml — the Eq. (7)-(8) weight-sensitivity
  ablation (single language).
* configs/saal_full_experiment.yaml — the full three-language matrix
  (Random / Entropy / Novelty / Support-Aware, budgets 10-50%), the
  full-pool b95 reference, and (for Kazakh) the weight-sensitivity
  configurations, all under one surrogate setting.

Tasks are (language, strategy, seed) triples computed independently and
resumable: completed triples are kept on re-run.  Outputs:

  <out-dir>/<language>/saal_raw.json          one record per task
  <out-dir>/<language>/saal_summary.csv       mean +/- SD per checkpoint

Use scripts/analyze_saal_paper.py to turn the raw records into the paper's
tables (TSC by budget, normalized AULC, paired Wilcoxon+Holm, b95).

Usage:
    .venv/bin/python scripts/run_weight_sensitivity.py \
        --config configs/saal_full_experiment.yaml \
        --out-dir results_saal_full [--language kazakh kyrgyz ...]
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

# Must precede Pool creation: spawn workers inherit the environment before
# numpy/BLAS initialise, preventing thread oversubscription across workers.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cagf.saal import (AcquisitionWeights, CharHashVectorizer, build_split,
                       load_corpus, run_fullpool, run_simulation)

_WORKER_CTX: dict = {}


def _init_worker(cfg: dict) -> None:
    sur = cfg['surrogate']
    _WORKER_CTX['cfg'] = cfg
    _WORKER_CTX['corpora'] = {}
    _WORKER_CTX['vectorizer'] = CharHashVectorizer(dim=sur['feature_dim'],
                                                   ngram_range=tuple(sur['ngram_range']))


def _run_task(task: tuple[str, str, int]) -> dict:
    language, name, seed = task
    cfg = _WORKER_CTX['cfg']
    sim = cfg['simulation']
    sur = cfg['surrogate']
    sentences = _WORKER_CTX['corpora'].setdefault(
        language, load_corpus(cfg['languages'][language]['corpus_paths']))
    vectorizer = _WORKER_CTX['vectorizer']
    surrogate_kwargs = {'solver': sur.get('solver', 'multinomial'),
                        'c': sur.get('c', 1.0), 'max_iter': sur.get('max_iter', 500)}
    split = build_split(sentences, seed, sim['pool_fraction'], sim['initial_fraction'])
    t0 = time.time()
    if name == 'fullpool':
        checkpoints = {'full': run_fullpool(sentences, split, vectorizer, surrogate_kwargs)}
        weights_desc = 'fullpool'
    else:
        entry = cfg.get('strategies', {}).get(name, {})
        aw = entry.get('acquisition_weights', [0.25, 0.20, 0.30, 0.15, 0.10])
        sw = entry.get('sentence_weights', [0.8, 0.2])
        weights = AcquisitionWeights(hu=aw[0], hr=aw[1], qpT=aw[2], nsuf=aw[3], nlex=aw[4],
                                     sentence_mean=sw[0], sentence_max=sw[1])
        kind = name if name in ('random', 'entropy', 'novelty') else 'support_aware'
        checkpoints = run_simulation(sentences, split, kind, list(sim['budgets']), weights,
                                     vectorizer, surrogate_kwargs)
        weights_desc = weights.describe() if kind == 'support_aware' else kind
    return {'language': language, 'strategy': name, 'seed': seed,
            'checkpoints': checkpoints, 'weights': weights_desc,
            'seconds': round(time.time() - t0, 1)}


def summarize(records: list[dict], out_dir: Path) -> None:
    metrics = ('tsc', 'lemma_acc', 'upos_acc')
    strategies = list(dict.fromkeys(r['strategy'] for r in records))
    rows = []
    for strategy in strategies:
        cps: dict[str, list[dict]] = {}
        for r in records:
            if r['strategy'] != strategy:
                continue
            for cp, vals in r['checkpoints'].items():
                cps.setdefault(cp, []).append(vals)
        for cp in sorted(cps, key=lambda c: (c != 'initial', c != 'full', float(c) if c not in ('initial', 'full') else -1.0)):
            for metric in metrics:
                values = [v[metric] for v in cps[cp]]
                rows.append({'strategy': strategy, 'checkpoint': cp, 'metric': metric,
                             'mean': round(statistics.mean(values), 2),
                             'sd': round(statistics.stdev(values), 2) if len(values) > 1 else 0.0,
                             'n_seeds': len(values)})
    csv_path = out_dir / 'saal_summary.csv'
    with open(csv_path, 'w', encoding='utf-8') as fh:
        fh.write('strategy,checkpoint,metric,mean,sd,n_seeds\n')
        for row in rows:
            fh.write(f"{row['strategy']},{row['checkpoint']},{row['metric']},"
                     f"{row['mean']},{row['sd']},{row['n_seeds']}\n")
    print(f'Summary CSV: {csv_path}')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', default='configs/saal_kazakh_ablation.yaml')
    ap.add_argument('--out-dir', default='results_weight_sensitivity')
    ap.add_argument('--language', nargs='+', default=None,
                    help='subset of languages from the config (default: all)')
    ap.add_argument('--strategies', nargs='+', default=None,
                    help='subset of strategy names (default: all for each language)')
    ap.add_argument('--seeds', nargs='+', type=int, default=None)
    ap.add_argument('--workers', type=int, default=min(10, os.cpu_count() or 1))
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    languages = list(cfg.get('languages', {})) or ['default']
    if args.language:
        unknown = set(args.language) - set(languages)
        if unknown:
            ap.error(f'unknown languages {sorted(unknown)}; available: {languages}')
        languages = args.language
    seeds = args.seeds or list(cfg['simulation']['seeds'])

    for language in languages:
        lang_cfg = cfg.get('languages', {}).get(language, {})
        strategy_names = lang_cfg.get('strategies') or list(cfg.get('strategies', {})) or \
            [s for s in ('random', 'entropy', 'novelty', 'baseline') ]
        if args.strategies:
            unknown = set(args.strategies) - set(strategy_names) - {'fullpool'}
            if unknown:
                ap.error(f'unknown strategies {sorted(unknown)}; available: {strategy_names}')
            strategy_names = args.strategies
        out_dir = Path(args.out_dir) / (language if language != 'default' else '')
        out_dir.mkdir(parents=True, exist_ok=True)
        raw_path = out_dir / 'saal_raw.json'
        records: list[dict] = []
        done: set[tuple[str, int]] = set()
        if raw_path.exists():
            records = json.loads(raw_path.read_text(encoding='utf-8'))
            done = {(r['strategy'], r['seed']) for r in records}
            print(f'[{language}] Resuming: {len(done)} runs already complete.')
        tasks = [(language, name, seed) for name in strategy_names for seed in seeds
                 if (name, seed) not in done]
        print(f'[{language}] {len(tasks)} runs to do: {len(strategy_names)} strategies x {len(seeds)} seeds.')
        if tasks:
            import multiprocessing as mp
            ctx = mp.get_context('spawn')
            completed = 0
            with ctx.Pool(args.workers, initializer=_init_worker, initargs=(cfg,)) as pool:
                for record in pool.imap_unordered(_run_task, tasks):
                    records.append(record)
                    completed += 1
                    raw_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
                    last_cp = list(record['checkpoints'])[-1]
                    print(f"[{language}][{len(done) + completed}/{len(done) + len(tasks)}] "
                          f"{record['strategy']} seed={record['seed']} @{last_cp} "
                          f"TSC={record['checkpoints'][last_cp]['tsc']:.2f} "
                          f"lemma={record['checkpoints'][last_cp]['lemma_acc']:.2f} "
                          f"({record['seconds']}s)", flush=True)
        summarize(records, out_dir)


if __name__ == '__main__':
    main()
