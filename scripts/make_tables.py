"""Generate publication tables from raw experiment JSON.

Every number cited in the paper's Results tables is produced here, so the text
and the artifacts cannot drift apart (Reviewer 1 #15 / Reviewer 4 #15:
"results are largely communicated through figures only -- provide a summary
results table with exact accuracy, F1, precision, recall numbers for all
models").

Inputs
------
  * results/ablation_raw.json         -- ablation runs (5 configs x 5 seeds)
  * results/significance.json         -- optional Holm-corrected p-values
                                         (from scripts/significance_test.py --holm)
  * results_baselines/baseline_raw.json -- optional baseline runs
  * results_transfer/transfer_summary_seed*.json -- optional silver transfer runs

Outputs (Markdown, ready to paste into the manuscript)
  * results/tables/main_results.md      -- models x task x {acc,prec,rec,f1} mean+-std
  * results/tables/ablation_stats.md    -- ablation + p-values (raw & Holm) + Cohen's d
  * results/tables/transfer.md          -- silver pretrain->finetune vs gold-only (if present)

Usage
-----
    PYTHONPATH=. python scripts/make_tables.py
    PYTHONPATH=. python scripts/make_tables.py --ablation results/ablation_raw.json \
        --baselines results/baselines/baseline_raw.json --out-dir results/tables
"""
from __future__ import annotations
import argparse
import json
import statistics
from pathlib import Path
from typing import Iterable

TASKS = ('lemma', 'upos', 'grammeme')
METRICS = ('accuracy', 'precision', 'recall', 'f1')
TASK_LABEL = {'lemma': 'Lemma', 'upos': 'UPOS', 'grammeme': 'Grammeme'}
# display order: baselines first, then ablation ladder ending on full_model
PREFERRED_ORDER = ['cnn', 'bilstm', 'cnn_bilstm', 'cnn_bilstm_transformer',
                   'subword_tagging', 'transformer_only',
                   'wo_character_encoder', 'wo_gated_fusion', 'wo_crf', 'full_model',
                   'ktb_only',
                   'silver_filtered_pretrain_ktb_finetune',
                   'silver_unfiltered_pretrain_ktb_finetune']
PRETTY = {'cnn': 'CNN', 'bilstm': 'BiLSTM', 'cnn_bilstm': 'CNN-BiLSTM',
          'cnn_bilstm_transformer': 'CNN-BiLSTM-Transformer',
          'subword_tagging': 'Subword-tagging',
          'transformer_only': 'Transformer-only', 'wo_character_encoder': 'w/o char encoder',
          'wo_gated_fusion': 'w/o gated fusion', 'wo_crf': 'w/o CRF', 'full_model': 'CAGF-CBT+CRF (full)',
          'ktb_only': 'Gold-only (baseline)',
          'silver_filtered_pretrain_ktb_finetune': 'Silver-pretrain -> gold-finetune (filtered)',
          'silver_unfiltered_pretrain_ktb_finetune': 'Silver-pretrain -> gold-finetune (unfiltered)'}


def _order(configs: Iterable[str]) -> list[str]:
    cfgs = set(configs)
    ordered = [c for c in PREFERRED_ORDER if c in cfgs]
    return ordered + sorted(cfgs - set(ordered))


def _agg(runs: list[dict], task: str, metric: str) -> tuple[float, float, int]:
    vals = [r[task][metric] for r in runs]
    n = len(vals)
    if n == 0:
        return (float('nan'), float('nan'), 0)
    mean = statistics.mean(vals)
    std = statistics.pstdev(vals) if n > 1 else 0.0
    return (mean, std, n)


def _cell(mean: float, std: float, n: int, pct: bool = True, show_std: bool = True) -> str:
    if n == 0:
        return '--'
    scale = 100 if pct else 1
    base = f'{mean * scale:.2f}'
    if show_std and n > 1:
        return f'{base} \u00b1 {std * scale:.2f}'
    return base


def load_runs(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding='utf-8'))


def by_config(runs: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in runs:
        # ablation runs use 'config', baseline runs use 'model'
        key = r.get('config') or r.get('model') or r.get('name', 'unknown')
        out.setdefault(key, []).append(r)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ablation', default='results/ablation_raw.json')
    ap.add_argument('--baselines', default='results_baselines/baseline_raw.json')
    ap.add_argument('--significance', default='results/significance.json')
    ap.add_argument('--transfer-glob', default='results_transfer/transfer_summary_seed*.json')
    ap.add_argument('--cv', default=None,
                    help='optional cv_raw.json from scripts/run_cv.py; generates a CV table '
                         'with official CoNLL-2018 metrics (Lemmas/UPOS/UFeats/AllTags)')
    ap.add_argument('--out-dir', default='results/tables')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ablation_runs = load_runs(args.ablation)
    baseline_runs = load_runs(args.baselines)
    all_by_cfg = by_config(baseline_runs + ablation_runs)

    # ---------- Table 1: main results ----------
    lines = ['# Main results (gold KTB test set, mean +/- std over N seeds)', '',
             'Numbers are percentages. Acc = accuracy, P/R/F1 = macro precision/recall/F1.', '']
    for task in TASKS:
        lines.append(f'## {TASK_LABEL[task]}')
        header = '| Model | ' + ' | '.join(m.upper() for m in METRICS) + ' | N |'
        sep = '|---' * (len(METRICS) + 2) + '|'
        lines += [header, sep]
        for cfg in _order(all_by_cfg):
            runs = all_by_cfg[cfg]
            cells = [_cell(*_agg(runs, task, m)) for m in METRICS]
            n = _agg(runs, task, METRICS[0])[2]
            name = PRETTY.get(cfg, cfg)
            lines.append(f'| {name} | ' + ' | '.join(cells) + f' | {n} |')
        lines.append('')
    (out_dir / 'main_results.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'Wrote {out_dir / "main_results.md"}')

    # ---------- Table 2: ablation + significance ----------
    sig = {}
    sp = Path(args.significance)
    if sp.exists():
        for row in json.loads(sp.read_text(encoding='utf-8')):
            sig[(row['task'], row['config'])] = row
    lines = ['# Ablation statistics vs full model (paired t-test, 5 seeds)',
             '',
             'p_raw = uncorrected paired t-test p-value; p_Holm = Holm-Bonferroni adjusted;',
             'd = Cohen\'s d (paired). * = significant after Holm correction at alpha=0.05.', '']
    for task in TASKS:
        lines.append(f'## {TASK_LABEL[task]} (macro F1)')
        header = '| Configuration | F1 (mean +/- std) | Delta vs full | p_raw | p_Holm | Cohen\'s d |'
        sep = '|---' * 7 + '|'
        lines += [header, sep]
        full_f1 = _agg(ablation_runs and by_config(ablation_runs).get('full_model', []) or [], task, 'f1')[0]
        for cfg in _order(by_config(ablation_runs)):
            runs = by_config(ablation_runs).get(cfg, [])
            if not runs:
                continue
            mean, std, n = _agg(runs, task, 'f1')
            delta = (mean - full_f1) * 100 if full_f1 == full_f1 else float('nan')  # nan check
            row = sig.get((task, cfg), {})
            p_raw = f"{row.get('p_raw', float('nan')):.4f}" if row else '--'
            p_holm = f"{row.get('p_holm', float('nan')):.4f}" if row else '--'
            d = f"{row.get('cohens_d', float('nan')):.3f}" if row else '--'
            star = ' *' if row.get('significant_holm') else ''
            lines.append(f'| {PRETTY.get(cfg, cfg)} | {mean*100:.2f} +/- {std*100:.2f} | '
                         f'{delta:+.2f} | {p_raw} | {p_holm}{star} | {d} |')
        lines.append('')
    (out_dir / 'ablation_stats.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'Wrote {out_dir / "ablation_stats.md"}')

    # ---------- Table 3: silver transfer (optional) ----------
    transfer_files = sorted(Path('.').glob(args.transfer_glob))
    if transfer_files:
        lines = ['# Silver pretraining -> gold fine-tuning (gold KTB test set)', '',
                 'Compares gold-only-from-scratch vs silver-pretrain-then-finetune.', '']
        header = '| Config | Lemma F1 | UPOS F1 | Grammeme F1 |'
        sep = '|---' * 4 + '|'
        lines += [header, sep]
        for tf in transfer_files:
            data = json.loads(tf.read_text(encoding='utf-8'))
            for key, label in (('gold_only_from_scratch', 'gold-only (scratch)'),
                               ('silver_pretrain_then_gold_finetune', 'silver->gold')):
                if key in data:
                    m = data[key]
                    lines.append(f"| {label} (seed {data.get('seed')}) | "
                                 f"{m['lemma']['f1']*100:.2f} | {m['upos']['f1']*100:.2f} | "
                                 f"{m['grammeme']['f1']*100:.2f} |")
        (out_dir / 'transfer.md').write_text('\n'.join(lines), encoding='utf-8')
        print(f'Wrote {out_dir / "transfer.md"}')
    else:
        print('(no transfer_summary files found -- skipping transfer table)')

    # ---------- Table 4: cross-validation with official CoNLL-2018 metrics ----------
    if args.cv:
        cv_runs = load_runs(args.cv)
        if cv_runs:
            _cv_table(cv_runs, out_dir, args.cv)
        else:
            print(f'(no runs in {args.cv} -- skipping CV table)')


def _cv_table(cv_runs: list[dict], out_dir: Path, cv_path: str) -> None:
    """CV table: per-config official CoNLL-2018 F1 (jack-knifed) + per-fold stats.

    This is the table that replaces the single-split main table when CV is the
    reported protocol. It reports the metrics that are actually comparable to
    the UD parsing literature (Lemmas/UPOS/UFeats/AllTags F1), alongside the
    internal accuracy/macro-F1 numbers for continuity with the older tables.
    """
    OFFICIAL = ('Lemmas', 'UPOS', 'UFeats', 'AllTags')
    by_cfg = by_config(cv_runs)
    lines = [
        '# Cross-validation results (k-fold, official CoNLL-2018 metrics)', '',
        f'Source: `{cv_path}`',
        'Jack-knifed F1: every sentence scored exactly once (by the fold whose '
        'test set held it).',
        'Internal metrics (Acc/macro-F1) are mean +/- std across folds; '
        'official F1 is computed on the concatenated out-of-fold predictions.', '',
    ]
    # per-config jack-knife official F1, read from jackknifed.json if present
    # (run_cv.py writes one per config dir). Fall back to per-fold mean if absent.
    cv_root = Path(cv_path).parent
    jackknife: dict[str, dict] = {}
    for cfg in by_cfg:
        jp = cv_root / cfg / 'jackknifed.json'
        if jp.exists():
            jackknife[cfg] = json.loads(jp.read_text(encoding='utf-8'))['official_jackknifed']

    lines.append('## Official CoNLL-2018 F1 (jack-knifed over all sentences)')
    header = '| Configuration | ' + ' | '.join(OFFICIAL) + ' | N folds |'
    sep = '|---' * (len(OFFICIAL) + 2) + '|'
    lines += [header, sep]
    for cfg in _order(by_cfg):
        n = len(by_cfg[cfg])
        if cfg in jackknife:
            jk = jackknife[cfg]
            cells = [f'{jk[m]*100:.2f}' for m in OFFICIAL]
        else:
            # fallback: mean of per-fold official (less exact, but still useful)
            cells = []
            for m in OFFICIAL:
                vals = [r.get('official', {}).get(m, float('nan')) for r in by_cfg[cfg]]
                vals = [v for v in vals if v == v]
                cells.append(f'{statistics.mean(vals)*100:.2f}' if vals else '--')
        lines.append(f'| {PRETTY.get(cfg, cfg)} | ' + ' | '.join(cells) + f' | {n} |')
    lines.append('')

    lines.append('## Per-fold internal metrics (mean +/- std across folds)')
    for task in TASKS:
        lines.append(f'### {TASK_LABEL[task]}')
        header = '| Configuration | ' + ' | '.join(m.upper() for m in METRICS) + ' | N folds |'
        sep = '|---' * (len(METRICS) + 2) + '|'
        lines += [header, sep]
        for cfg in _order(by_cfg):
            runs = by_cfg[cfg]
            cells = [_cell(*_agg(runs, task, m)) for m in METRICS]
            n = _agg(runs, task, METRICS[0])[2]
            lines.append(f'| {PRETTY.get(cfg, cfg)} | ' + ' | '.join(cells) + f' | {n} |')
        lines.append('')

    (out_dir / 'cv_results.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'Wrote {out_dir / "cv_results.md"}')


if __name__ == '__main__':
    main()
