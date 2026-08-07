"""Run the silver-pretraining transfer-learning experiment configurations.

Core architecture is unchanged (same CAGF-CBT+CRF full model, same
train_one_run protocol as scripts/run_ablation.py); only the *data*
preparation differs between configurations. Three configurations:

  * ``ktb_only``
        Train from scratch on gold UD_Kazakh-KTB only. Vocabularies are
        built from gold training data alone (no silver enrichment), so this
        is a clean lower-bound baseline.

  * ``silver_filtered_pretrain_ktb_finetune``  (requires --silver-filtered)
        Pretrain on the FILTERED silver corpus (scripts/filter_silver_corpus.py
        output), then fine-tune on gold KTB. This is the main configuration
        of interest.

  * ``silver_unfiltered_pretrain_ktb_finetune``  (requires --silver-unfiltered)
        Identical protocol, but pretrains on the raw/unfiltered silver
        corpus (scripts/build_silver_corpus.py output, before filtering).
        Comparing this against the filtered configuration is what tells you
        whether the filtering stage is actually useful, rather than assumed
        to be.

All configurations are evaluated on the same, unchanged gold KTB test set.
Output schema (``config``, ``seed``, ``lemma``, ``upos``, ``grammeme``,
``best_epoch``) matches ``results/ablation_raw.json``, so
``scripts/significance_test.py`` can be pointed at this script's raw-results
file directly to compare e.g. ``ktb_only`` vs
``silver_filtered_pretrain_ktb_finetune``.

Usage
-----
    # two configurations (minimum required)
    PYTHONPATH=. python scripts/run_silver_ablation.py \
        --gold-train data/gold_merged/gold_train.conllu \
        --gold-dev   data/gold_merged/gold_dev.conllu \
        --gold-test  data/gold_merged/gold_test.conllu \
        --silver-filtered data/silver/silver_filtered.conllu \
        --seeds 13 42 123 777 2026

    # add the third (filtering-ablation) configuration
    PYTHONPATH=. python scripts/run_silver_ablation.py \
        ... \
        --silver-filtered   data/silver/silver_filtered.conllu \
        --silver-unfiltered data/silver/silver.conllu \
        --seeds 13 42 123 777 2026
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import List, Optional

from cagf.data import CorpusVocabs, Sentence, build_vocabs, read_conllu
from cagf.model import AblationConfig, ModelHParams
from cagf.train_loop import RunResult, train_one_run
from scripts.pretrain_finetune import build_shared_vocabs

KTB_ONLY = "ktb_only"
SILVER_FILTERED = "silver_filtered_pretrain_ktb_finetune"
SILVER_UNFILTERED = "silver_unfiltered_pretrain_ktb_finetune"


def run_ktb_only(gold_train: List[Sentence], gold_dev: List[Sentence], gold_test: List[Sentence],
                 hp: ModelHParams, ablation: AblationConfig, seed: int, args) -> RunResult:
    vocabs = build_vocabs(gold_train, min_word_freq=args.min_word_freq, max_word_vocab=args.max_word_vocab)
    result = train_one_run(
        train_sentences=gold_train, dev_sentences=gold_dev, test_sentences=gold_test,
        vocabs=vocabs, ablation=ablation, hparams=hp, seed=seed,
        max_epochs=args.finetune_epochs, batch_size=args.batch_size,
        learning_rate=args.pretrain_lr, weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm, early_stopping_patience=args.finetune_patience,
        device=args.device, verbose=True)
    return replace(result, config_name=KTB_ONLY)


def run_silver_pretrain_then_finetune(config_name: str, silver: List[Sentence],
                                      gold_train: List[Sentence], gold_dev: List[Sentence],
                                      gold_test: List[Sentence], hp: ModelHParams,
                                      ablation: AblationConfig, seed: int, args) -> RunResult:
    vocabs: CorpusVocabs = build_shared_vocabs(gold_train, silver,
                                               min_word_freq=args.min_word_freq,
                                               max_word_vocab=args.max_word_vocab)
    print(f"  [{config_name}] silver={len(silver):,} sents, "
         f"vocab: chars={len(vocabs.char_vocab)} words={len(vocabs.word_vocab)}")

    import torch
    ckpt_path = Path(args.out_dir) / f"{config_name}_seed{seed}_pretrained.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    train_one_run(
        train_sentences=silver, dev_sentences=gold_dev, test_sentences=gold_test,
        vocabs=vocabs, ablation=ablation, hparams=hp, seed=seed,
        max_epochs=args.pretrain_epochs, batch_size=args.batch_size,
        learning_rate=args.pretrain_lr, weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm, early_stopping_patience=args.pretrain_patience,
        device=args.device, verbose=True, save_checkpoint_path=str(ckpt_path))
    pretrained_state = torch.load(ckpt_path, map_location="cpu", weights_only=False)["model_state_dict"]

    finetuned = train_one_run(
        train_sentences=gold_train, dev_sentences=gold_dev, test_sentences=gold_test,
        vocabs=vocabs, ablation=ablation, hparams=hp, seed=seed,
        max_epochs=args.finetune_epochs, batch_size=args.batch_size,
        learning_rate=args.finetune_lr, weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm, early_stopping_patience=args.finetune_patience,
        device=args.device, verbose=True, init_state=pretrained_state)
    return replace(finetuned, config_name=config_name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gold-train", default="data/gold_merged/gold_train.conllu")
    ap.add_argument("--gold-dev", default="data/gold_merged/gold_dev.conllu")
    ap.add_argument("--gold-test", default="data/gold_merged/gold_test.conllu")
    ap.add_argument("--silver-filtered", default=None,
                    help="filtered silver corpus (scripts/filter_silver_corpus.py output); "
                         "enables the silver_filtered_pretrain_ktb_finetune configuration")
    ap.add_argument("--silver-unfiltered", default=None,
                    help="raw silver corpus (scripts/build_silver_corpus.py output, before "
                         "filtering); enables the silver_unfiltered_pretrain_ktb_finetune "
                         "configuration, used to measure whether filtering helps")
    ap.add_argument("--seeds", nargs="+", type=int, default=[13, 42, 123, 777, 2026])
    ap.add_argument("--out-dir", default="results_silver_ablation")
    ap.add_argument("--min-word-freq", type=int, default=1)
    ap.add_argument("--max-word-vocab", type=int, default=50000)
    ap.add_argument("--pretrain-epochs", type=int, default=30)
    ap.add_argument("--pretrain-lr", type=float, default=3e-4)
    ap.add_argument("--finetune-epochs", type=int, default=200)
    ap.add_argument("--finetune-lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--grad-clip-norm", type=float, default=5.0)
    ap.add_argument("--pretrain-patience", type=int, default=5)
    ap.add_argument("--finetune-patience", type=int, default=15)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "silver_ablation_raw.json"

    gold_train = read_conllu(args.gold_train)
    gold_dev = read_conllu(args.gold_dev)
    gold_test = read_conllu(args.gold_test)
    silver_filtered = read_conllu(args.silver_filtered) if args.silver_filtered else None
    silver_unfiltered = read_conllu(args.silver_unfiltered) if args.silver_unfiltered else None

    configs = [KTB_ONLY]
    if silver_filtered is not None:
        configs.append(SILVER_FILTERED)
    if silver_unfiltered is not None:
        configs.append(SILVER_UNFILTERED)
    if len(configs) < 2:
        raise SystemExit("Need at least --silver-filtered (in addition to the always-on "
                         "ktb_only baseline) to have two configurations to compare.")

    hp = ModelHParams()
    ablation = AblationConfig()  # full model; architecture is not being ablated here

    all_results: List[RunResult] = []
    already_done: set[tuple[str, int]] = set()
    if raw_path.exists():
        prior = json.loads(raw_path.read_text(encoding="utf-8"))
        for r in prior:
            all_results.append(RunResult(config_name=r["config"], seed=r["seed"], lemma=r["lemma"],
                                         upos=r["upos"], grammeme=r["grammeme"],
                                         best_epoch=r["best_epoch"], history=[]))
            already_done.add((r["config"], r["seed"]))
        print(f"Found existing partial results: {len(prior)} runs already completed, will be kept.")

    def flush_results() -> None:
        raw = [{"config": r.config_name, "seed": r.seed, "lemma": r.lemma, "upos": r.upos,
               "grammeme": r.grammeme, "best_epoch": r.best_epoch} for r in all_results]
        raw_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")

    for config_name in configs:
        for seed in args.seeds:
            if (config_name, seed) in already_done:
                print(f"=== Skipping {config_name}, seed={seed} (already completed) ===")
                continue
            print(f"=== Running {config_name}, seed={seed} ===")
            if config_name == KTB_ONLY:
                result = run_ktb_only(gold_train, gold_dev, gold_test, hp, ablation, seed, args)
            elif config_name == SILVER_FILTERED:
                result = run_silver_pretrain_then_finetune(
                    SILVER_FILTERED, silver_filtered, gold_train, gold_dev, gold_test, hp, ablation, seed, args)
            else:
                result = run_silver_pretrain_then_finetune(
                    SILVER_UNFILTERED, silver_unfiltered, gold_train, gold_dev, gold_test, hp, ablation, seed, args)
            all_results.append(result)
            flush_results()

    print(f"\nDone. Raw results: {raw_path}")
    print("Compare configurations with, e.g.:")
    print(f"  PYTHONPATH=. python scripts/significance_test.py --results {raw_path} "
         f"--config-a {KTB_ONLY} --config-b {SILVER_FILTERED}")
    if SILVER_UNFILTERED in configs:
        print(f"  PYTHONPATH=. python scripts/significance_test.py --results {raw_path} "
             f"--config-a {SILVER_UNFILTERED} --config-b {SILVER_FILTERED}")


if __name__ == "__main__":
    main()
