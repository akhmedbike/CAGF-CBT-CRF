"""Summarize the silver corpus for the paper's Dataset / Transfer-Learning section.

Produces the exact numbers cited in the manuscript:
  * sentence / token counts, before and after filtering
  * UPOS distribution and its overlap with the gold KTB label space
  * share of the ``X`` (foreign/other) category, which is large in book text
  * edit-script lemma coverage on silver (does the silver lemma vocabulary
    actually overlap with the gold edit-rule space?)

Reads only CoNLL-U files plus the build/filter JSON reports, so it is cheap
and deterministic. Output: a single JSON + a human-readable Markdown table.

Usage
-----
    PYTHONPATH=. python scripts/analyze_silver.py \
        --gold-train data/gold_merged/gold_train.conllu \
        --silver data/silver/silver_filtered.conllu
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from cagf.data import read_conllu, form_to_edit_script, parse_feats


def summarize_conllu(path: str) -> dict:
    sents = read_conllu(path)
    upos = Counter()
    feats_keys = Counter()
    n_tok = 0
    lengths = []
    for s in sents:
        lengths.append(len(s))
        for t in s.tokens:
            n_tok += 1
            upos[t.upos] += 1
            for f in parse_feats(t.feats):
                feats_keys[f.split("=")[0]] += 1
    return {
        "sentences": len(sents),
        "tokens": n_tok,
        "mean_sentence_length": round(sum(lengths) / max(len(lengths), 1), 2),
        "upos_distribution": dict(upos.most_common()),
        "feat_categories": dict(feats_keys.most_common()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gold-train", default="data/gold_merged/gold_train.conllu")
    ap.add_argument("--silver", default="data/silver/silver_filtered.conllu")
    ap.add_argument("--silver-report", default="data/silver/silver.report.json")
    ap.add_argument("--filter-report", default="data/silver/silver_filtered.report.json")
    ap.add_argument("--out", default="results_transfer/silver_analysis.json")
    args = ap.parse_args()

    print("Reading gold train (for label-space overlap)...")
    gold = read_conllu(args.gold_train)
    gold_upos = {t.upos for s in gold for t in s.tokens}
    gold_rules = {form_to_edit_script(t.form, t.lemma) for s in gold for t in s.tokens}

    print("Summarizing filtered silver corpus (this is the 2.89M-token resource)...")
    silver_summary = summarize_conllu(args.silver)

    # How many silver UPOS tokens fall inside the gold label space?
    silver_in_gold = sum(n for u, n in silver_summary["upos_distribution"].items() if u in gold_upos)
    silver_total = silver_summary["tokens"]

    # Edit-script overlap: of the silver edit-rules, how many are seen in gold?
    # (gold rule space is what the lemma head can actually predict)
    silver_rules_seen = 0
    silver_rules_in_gold = 0
    n_silver_tok_rule = 0
    for s in read_conllu(args.silver):
        for t in s.tokens:
            r = form_to_edit_script(t.form, t.lemma)
            silver_rules_seen += 1
            n_silver_tok_rule += 1
            if r in gold_rules:
                silver_rules_in_gold += 1

    x_share = silver_summary["upos_distribution"].get("X", 0) / max(silver_total, 1)

    report = {
        "silver_filtered": silver_summary,
        "gold_upos_categories": sorted(gold_upos),
        "silver_upos_in_gold_space": silver_in_gold,
        "silver_upos_in_gold_share": round(silver_in_gold / max(silver_total, 1), 4),
        "silver_X_foreign_share": round(x_share, 4),
        "silver_edit_rules_in_gold_space": silver_rules_in_gold,
        "silver_edit_rules_in_gold_share": round(silver_rules_in_gold / max(n_silver_tok_rule, 1), 4),
        "gold_edit_rule_vocab_size": len(gold_rules),
    }
    # attach provenance reports if present
    for key, path in (("build_report", args.silver_report), ("filter_report", args.filter_report)):
        p = Path(path)
        if p.exists():
            report[key] = json.loads(p.read_text(encoding="utf-8"))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # Human-readable summary
    print("\n=== Silver corpus analysis ===")
    print(f"Sentences: {silver_summary['sentences']:,}   Tokens: {silver_summary['tokens']:,}")
    print(f"Mean sentence length: {silver_summary['mean_sentence_length']}")
    print(f"UPOS categories: {len(silver_summary['upos_distribution'])} "
          f"({', '.join(list(silver_summary['upos_distribution'])[:8])}...)")
    print(f"Silver tokens whose UPOS is in the gold KTB label space: "
          f"{silver_in_gold:,} / {silver_total:,} ({100*silver_in_gold/silver_total:.1f}%)")
    print(f"'X' (foreign/other) share: {100*x_share:.1f}%  "
          f"[report in paper: book text carries many numerals/symbols tagged X]")
    print(f"Silver edit-rules overlapping gold rule vocab: "
          f"{silver_rules_in_gold:,} / {n_silver_tok_rule:,} "
          f"({100*silver_rules_in_gold/n_silver_tok_rule:.1f}%)  "
          f"[gold rule vocab = {len(gold_rules)}]")
    print(f"\nReport written to {out_path}")


if __name__ == "__main__":
    main()
