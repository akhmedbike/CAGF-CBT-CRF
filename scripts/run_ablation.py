from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import yaml
from cagf.data import build_vocabs, read_conllu
from cagf.model import AblationConfig, ModelHParams
from cagf.train_loop import RunResult, train_one_run
ABLATION_CONFIGS = {'full_model': AblationConfig(), 'wo_character_encoder': AblationConfig(use_char_cnn=False, use_char_bilstm=False), 'wo_gated_fusion': AblationConfig(use_gated_fusion=False), 'wo_crf': AblationConfig(use_crf=False), 'transformer_only': AblationConfig(use_char_cnn=False, use_char_bilstm=False, use_word_bilstm=False, use_gated_fusion=False)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/default.yaml')
    ap.add_argument('--seeds', nargs='+', type=int, default=None)
    ap.add_argument('--max-epochs', type=int, default=None)
    ap.add_argument('--out-dir', default='results')
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    seeds = args.seeds or cfg['training']['seed_list']
    max_epochs = args.max_epochs or cfg['training']['max_epochs']
    train = read_conllu(cfg['data']['train_path'])
    dev = read_conllu(cfg['data']['dev_path'])
    test = read_conllu(cfg['data']['test_path'])
    vocabs = build_vocabs(train, min_word_freq=cfg['data']['min_word_freq'], max_word_vocab=cfg['data']['max_word_vocab'])
    hp = ModelHParams(**cfg['model'])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vocabs.save(out_dir / 'vocabs.json')
    raw_path = out_dir / 'ablation_raw.json'
    all_results: list[RunResult] = []
    already_done: set[tuple[str, int]] = set()
    if raw_path.exists():
        prior = json.loads(raw_path.read_text(encoding='utf-8'))
        for r in prior:
            already_done.add((r['config'], r['seed']))
        print(f'Found existing partial results: {len(prior)} runs already completed, will be kept.')

    def flush_results():
        raw = [{'config': r.config_name, 'seed': r.seed, 'lemma': r.lemma, 'upos': r.upos, 'grammeme': r.grammeme, 'best_epoch': r.best_epoch} for r in all_results]
        raw_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding='utf-8')
        write_summary_csv(all_results, out_dir / 'ablation_summary.csv')
    if raw_path.exists():
        prior = json.loads(raw_path.read_text(encoding='utf-8'))
        for r in prior:
            all_results.append(RunResult(config_name=r['config'], seed=r['seed'], lemma=r['lemma'], upos=r['upos'], grammeme=r['grammeme'], best_epoch=r['best_epoch'], history=[]))
    for config_name, ablation in ABLATION_CONFIGS.items():
        for seed in seeds:
            if (config_name, seed) in already_done:
                print(f'=== Skipping {config_name}, seed={seed} (already completed) ===')
                continue
            print(f'=== Running {config_name}, seed={seed} ===')
            result = train_one_run(train, dev, test, vocabs, ablation, hp, seed=seed, max_epochs=max_epochs, batch_size=cfg['training']['batch_size'], learning_rate=cfg['training']['learning_rate'], weight_decay=cfg['training']['weight_decay'], grad_clip_norm=cfg['training']['grad_clip_norm'], early_stopping_patience=cfg['training']['early_stopping_patience'], verbose=True)
            all_results.append(result)
            flush_results()
    summary_path = out_dir / 'ablation_summary.csv'
    print(f'\nDone. Raw results: {raw_path}')
    print(f'Summary table:     {summary_path}')

def write_summary_csv(results: list[RunResult], path: Path) -> None:
    import statistics
    by_config: dict[str, list[RunResult]] = {}
    for r in results:
        by_config.setdefault(r.config_name, []).append(r)
    rows = []
    for config_name, runs in by_config.items():
        for task in ('lemma', 'upos', 'grammeme'):
            for metric in ('accuracy', 'precision', 'recall', 'f1'):
                values = [getattr(r, task)[metric] for r in runs]
                mean = statistics.mean(values)
                std = statistics.pstdev(values) if len(values) > 1 else 0.0
                rows.append({'config': config_name, 'task': task, 'metric': metric, 'mean': round(mean, 4), 'std': round(std, 4), 'n_seeds': len(values)})
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=['config', 'task', 'metric', 'mean', 'std', 'n_seeds'])
        writer.writeheader()
        writer.writerows(rows)
if __name__ == '__main__':
    main()