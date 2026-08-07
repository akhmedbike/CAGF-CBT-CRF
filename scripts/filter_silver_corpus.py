"""Filter incomplete/inconsistent annotations out of the raw silver corpus.

Reads the (unfiltered) silver corpus written by ``scripts/build_silver_corpus.py``
-- which carries per-token diagnostics in MISC (``UnkPOS=``, ``Dropped=``,
``Unmapped=``) -- and applies the sentence-level quality filter in
``cagf.silver_filter``. This is a standalone stage: it only reads/writes
CoNLL-U and a JSON report, and does not re-run tagging or mapping, so the
"filtered" and "unfiltered" corpora used in the transfer-learning
experiments are guaranteed to originate from the exact same annotation run.

Usage
-----
    PYTHONPATH=. python scripts/filter_silver_corpus.py \
        --input data/silver/silver.conllu \
        --output data/silver/silver_filtered.conllu

Optionally keep the rejected sentences for inspection:

    PYTHONPATH=. python scripts/filter_silver_corpus.py \
        --input data/silver/silver.conllu \
        --output data/silver/silver_filtered.conllu \
        --rejected data/silver/silver_rejected.conllu

The unfiltered input file can be used as-is (no need to run this script) to
run the "unfiltered silver pretraining + KTB fine-tuning" ablation config
that tests whether this filtering stage is actually worth it -- see
``scripts/run_silver_ablation.py``.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from collections import Counter
from pathlib import Path
from typing import Iterator, List, Tuple

from cagf.silver_filter import FilterThresholds, assess_sentence, parse_misc_diagnostics


# --------------------------------------------------------------------------
# streaming reader: yield (comment_lines, [full 10-column rows]) per sentence
# --------------------------------------------------------------------------
def iter_conllu_rows(path: str | Path) -> Iterator[Tuple[List[str], List[List[str]]]]:
    comments: List[str] = []
    rows: List[List[str]] = []
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if not line.strip():
                if rows:
                    yield comments, rows
                comments, rows = [], []
                continue
            if line.startswith("#"):
                comments.append(line)
                continue
            cols = line.split("\t")
            if len(cols) < 10:
                continue
            tid = cols[0]
            if "-" in tid or "." in tid:      # skip multiword/empty nodes
                continue
            rows.append(cols)
    if rows:
        yield comments, rows


def write_sentence(fout, comments: List[str], rows: List[List[str]]) -> None:
    for c in comments:
        fout.write(c + "\n")
    for row in rows:
        fout.write("\t".join(row) + "\n")
    fout.write("\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="data/silver/silver.conllu",
                    help="raw/unfiltered silver corpus from scripts/build_silver_corpus.py")
    ap.add_argument("--output", default="data/silver/silver_filtered.conllu")
    ap.add_argument("--rejected", default=None,
                    help="optional path to also write out the rejected sentences, for inspection")
    ap.add_argument("--max-unknown-pos-ratio", type=float, default=FilterThresholds().max_unknown_pos_ratio)
    ap.add_argument("--max-noise-ratio", type=float, default=FilterThresholds().max_noise_ratio)
    ap.add_argument("--min-tokens", type=int, default=FilterThresholds().min_tokens)
    ap.add_argument("--max-tokens", type=int, default=FilterThresholds().max_tokens)
    args = ap.parse_args()

    thresholds = FilterThresholds(
        max_unknown_pos_ratio=args.max_unknown_pos_ratio,
        max_noise_ratio=args.max_noise_ratio,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rejected_path = Path(args.rejected) if args.rejected else None
    if rejected_path is not None:
        rejected_path.parent.mkdir(parents=True, exist_ok=True)

    n_seen = 0
    n_kept = 0
    tokens_seen = 0
    tokens_kept = 0
    reason_counts: Counter = Counter()

    fout = open(out_path, "w", encoding="utf-8")
    frej = open(rejected_path, "w", encoding="utf-8") if rejected_path is not None else None
    try:
        fout.write(f"# silver_corpus_filtered "
                  f"max_unknown_pos_ratio={thresholds.max_unknown_pos_ratio} "
                  f"max_noise_ratio={thresholds.max_noise_ratio} "
                  f"min_tokens={thresholds.min_tokens} max_tokens={thresholds.max_tokens}\n")
        for comments, rows in iter_conllu_rows(args.input):
            n_seen += 1
            tokens_seen += len(rows)
            diagnostics = [parse_misc_diagnostics(row[9]) for row in rows]
            assessment = assess_sentence(diagnostics, thresholds)
            if assessment.kept:
                write_sentence(fout, comments, rows)
                n_kept += 1
                tokens_kept += len(rows)
            else:
                reason_counts.update(assessment.reasons)
                if frej is not None:
                    write_sentence(frej, comments, rows)
    finally:
        fout.close()
        if frej is not None:
            frej.close()

    report = {
        "input": str(args.input),
        "output": str(out_path),
        "rejected_output": str(rejected_path) if rejected_path else None,
        "thresholds": dataclasses.asdict(thresholds),
        "sentences_seen": n_seen,
        "sentences_kept": n_kept,
        "sentences_rejected": n_seen - n_kept,
        "tokens_seen": tokens_seen,
        "tokens_kept": tokens_kept,
        "rejection_reason_counts": dict(reason_counts),
    }
    report_path = out_path.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    kept_pct = 100.0 * n_kept / n_seen if n_seen else 0.0
    print(f"Kept {n_kept:,}/{n_seen:,} sentences ({kept_pct:.1f}%) -> {out_path}")
    print(f"Rejection reasons: {dict(reason_counts)}")
    print(f"Filter report: {report_path}")
    if n_kept == 0:
        print("\nWARNING: every sentence was rejected. Check --input is the output of "
              "build_silver_corpus.py (MISC must contain UnkPOS=/Dropped=/Unmapped=), "
              "or relax the thresholds.")


if __name__ == "__main__":
    main()
