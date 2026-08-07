"""Single entry point that runs the whole silver-corpus pipeline end to end.

    Raw auxiliary corpus
          |
    KazNLP pre-annotation                    scripts/build_silver_corpus.py
          |
    KazNLP-to-UD label mapping                cagf/kaznlp_ud_map.py
          |
    Filtering of incomplete and               scripts/filter_silver_corpus.py
    inconsistent annotations
          |
    Silver corpus (filtered / unfiltered)
          |
    Pretraining -> Fine-tuning on gold KTB -> Evaluation on gold KTB test
                                              scripts/run_silver_ablation.py
          |
    Paired significance test                  scripts/significance_test.py

This script does not reimplement any of the above -- it only calls the
existing, independently-testable scripts as subprocesses, in order, with
resumability at every stage (an already-produced file is reused unless
--force is given). This keeps each stage runnable and testable on its own,
exactly as before; this script just saves you from invoking five commands
by hand.

Runs two configurations by default (ktb_only, silver_filtered_pretrain_ktb_finetune).
Add --with-unfiltered to also run silver_unfiltered_pretrain_ktb_finetune, which
tells you whether the filtering stage is actually worth it.

Usage
-----
    # full run, KazNLP backend, all three configurations, five seeds
    PYTHONPATH=.:/path/to/kaznlp python scripts/run_full_pipeline.py \
        --backend kaznlp --kaznlp-model /path/to/kaznlp/kaznlp/morphology/mdl \
        --with-unfiltered --seeds 13 42 123 777 2026

    # quick offline smoke test on a handful of sentences, no KazNLP needed
    PYTHONPATH=. python scripts/run_full_pipeline.py \
        --backend rule --limit 200 --with-unfiltered \
        --seeds 1 --pretrain-epochs 2 --finetune-epochs 2
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], step: str) -> None:
    print(f"\n{'=' * 70}\n[run_full_pipeline] {step}\n{'=' * 70}")
    print(" ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(f"[run_full_pipeline] FAILED at step: {step} (exit code {result.returncode})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # stage 1: build silver corpus
    ap.add_argument("--raw-input", default="data/auxiliary/books_skeleton.conllu",
                    help="tokenised-but-unannotated auxiliary corpus")
    ap.add_argument("--backend", choices=["kaznlp", "rule"], default="kaznlp")
    ap.add_argument("--kaznlp-model", default="kaznlp/morphology/mdl")
    ap.add_argument("--limit", type=int, default=0, help="annotate only first N sentences (0 = all)")
    # stage 2: filter
    ap.add_argument("--max-unknown-pos-ratio", type=float, default=None)
    ap.add_argument("--max-noise-ratio", type=float, default=None)
    ap.add_argument("--min-tokens", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=None)
    # gold data
    ap.add_argument("--gold-train", default="data/gold_merged/gold_train.conllu")
    ap.add_argument("--gold-dev", default="data/gold_merged/gold_dev.conllu")
    ap.add_argument("--gold-test", default="data/gold_merged/gold_test.conllu")
    # stage 3: experiment configurations
    ap.add_argument("--with-unfiltered", action="store_true",
                    help="also run the silver_unfiltered_pretrain_ktb_finetune config")
    ap.add_argument("--seeds", nargs="+", type=int, default=[13, 42, 123, 777, 2026])
    ap.add_argument("--pretrain-epochs", type=int, default=30)
    ap.add_argument("--finetune-epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=32)
    # shared
    ap.add_argument("--data-dir", default="data/silver", help="where silver.conllu / silver_filtered.conllu go")
    ap.add_argument("--out-dir", default="results_silver_ablation")
    ap.add_argument("--force", action="store_true", help="rebuild every stage even if outputs already exist")
    ap.add_argument("--skip-significance", action="store_true",
                    help="skip the final significance_test.py calls (e.g. if scipy is unavailable)")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    silver_raw = data_dir / "silver.conllu"
    silver_filtered = data_dir / "silver_filtered.conllu"

    # ---- stage 1: KazNLP pre-annotation + KLC-to-UD mapping (unfiltered silver) ----
    if args.force or not silver_raw.exists():
        cmd = [sys.executable, "scripts/build_silver_corpus.py",
              "--input", args.raw_input, "--output", str(silver_raw), "--backend", args.backend]
        if args.backend == "kaznlp":
            cmd += ["--kaznlp-model", args.kaznlp_model]
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        run(cmd, "1/4 build unfiltered silver corpus (pre-annotation + mapping)")
    else:
        print(f"\n[run_full_pipeline] Reusing existing {silver_raw} (use --force to rebuild)")

    # ---- stage 2: filtering of incomplete/inconsistent annotations ----
    if args.force or not silver_filtered.exists():
        cmd = [sys.executable, "scripts/filter_silver_corpus.py",
              "--input", str(silver_raw), "--output", str(silver_filtered)]
        if args.max_unknown_pos_ratio is not None:
            cmd += ["--max-unknown-pos-ratio", str(args.max_unknown_pos_ratio)]
        if args.max_noise_ratio is not None:
            cmd += ["--max-noise-ratio", str(args.max_noise_ratio)]
        if args.min_tokens is not None:
            cmd += ["--min-tokens", str(args.min_tokens)]
        if args.max_tokens is not None:
            cmd += ["--max-tokens", str(args.max_tokens)]
        run(cmd, "2/4 filter incomplete/inconsistent annotations")
    else:
        print(f"\n[run_full_pipeline] Reusing existing {silver_filtered} (use --force to rebuild)")

    # ---- stage 3: pretrain -> fine-tune -> evaluate, for each configuration ----
    cmd = [sys.executable, "scripts/run_silver_ablation.py",
          "--gold-train", args.gold_train, "--gold-dev", args.gold_dev, "--gold-test", args.gold_test,
          "--silver-filtered", str(silver_filtered),
          "--seeds", *[str(s) for s in args.seeds],
          "--out-dir", args.out_dir,
          "--pretrain-epochs", str(args.pretrain_epochs),
          "--finetune-epochs", str(args.finetune_epochs),
          "--batch-size", str(args.batch_size)]
    if args.with_unfiltered:
        cmd += ["--silver-unfiltered", str(silver_raw)]
    run(cmd, "3/4 run experiment configurations (ktb_only / silver pretrain+finetune)")

    # ---- stage 4: paired significance test between configurations ----
    raw_results = Path(args.out_dir) / "silver_ablation_raw.json"
    if not args.skip_significance:
        run([sys.executable, "scripts/significance_test.py", "--results", str(raw_results),
            "--config-a", "ktb_only", "--config-b", "silver_filtered_pretrain_ktb_finetune"],
            "4/4 significance test: ktb_only vs silver_filtered_pretrain_ktb_finetune")
        if args.with_unfiltered:
            run([sys.executable, "scripts/significance_test.py", "--results", str(raw_results),
                "--config-a", "silver_unfiltered_pretrain_ktb_finetune",
                "--config-b", "silver_filtered_pretrain_ktb_finetune"],
                "4/4 significance test: unfiltered vs filtered silver pretraining")

    print(f"\n{'=' * 70}\n[run_full_pipeline] Done.")
    print(f"  unfiltered silver: {silver_raw}")
    print(f"  filtered silver:   {silver_filtered}")
    print(f"  raw results:       {raw_results}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
