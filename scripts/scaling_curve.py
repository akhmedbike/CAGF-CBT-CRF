"""Aggregate the silver-scaling curve: quality vs. silver-corpus size.

Reads the per-budget CV-silver runs produced by run_cv_silver.py (one output
directory per budget, e.g. results_cv_silver/scaling_100k/...) and builds:
  * a flat table (budget -> jack-knifed F1 + per-fold mean/std),
  * a CSV/JSON artifact for the manuscript figure,
  * a quick text rendering of the curve for sanity-checking in the terminal.

This answers the reviewer question "how much silver do you need, and where is
saturation?" -- the most valuable additional result for a low-resource NLP
paper, because it generalises to other Turkic languages.

Usage
-----
    PYTHONPATH=. python scripts/scaling_curve.py \\
        --budgets 100000 210000 500000 1000000 \\
        --root results_cv_silver \\
        --reference-dir results_cv_silver/stratified/ktb_only
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Dict, List, Optional


def _suffix(b: int) -> str:
    return f"{b // 1000}k"


def _load_point(run_dir: Path) -> Optional[dict]:
    """Read one budget point: jack-knifed F1 + per-fold stats.

    Returns None if the point is incomplete (missing jackknifed.json).
    """
    cfg_dir = run_dir / "silver_filtered_pretrain_ktb_finetune"
    jk_path = cfg_dir / "jackknifed.json"
    pf_path = cfg_dir / "per_fold.json"
    if not jk_path.exists() or not pf_path.exists():
        return None
    jk = json.loads(jk_path.read_text(encoding="utf-8"))
    pf = json.loads(pf_path.read_text(encoding="utf-8"))
    official = jk["official_jackknifed"]
    out = {
        "n_folds": jk["n_folds"],
        "n_sentences": jk["n_sentences"],
        "official_jackknifed": official,
        "per_fold": {},
    }
    for metric in ("UPOS", "UFeats", "Lemmas", "AllTags"):
        vals = [r["official"][metric] for r in pf]
        out["per_fold"][metric] = {
            "mean": statistics.mean(vals),
            "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
            "n": len(vals),
        }
    return out


def _bar(value: float, lo: float, hi: float, width: int = 40) -> str:
    """Render a crude ASCII bar for terminal sanity view."""
    span = hi - lo or 1.0
    pos = int((value - lo) / span * width)
    pos = max(0, min(width, pos))
    return "[" + "#" * pos + "." * (width - pos) + "]"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--budgets", type=int, nargs="+",
                    default=[100000, 210000, 500000, 1000000])
    ap.add_argument("--root", default="results_cv_silver",
                    help="dir holding scaling_<budget>/ subdirs")
    ap.add_argument("--reference-dir", default=None,
                    help="optional ktb_only dir for the 0-silver reference point")
    ap.add_argument("--metrics", nargs="+",
                    default=["Lemmas", "UPOS", "UFeats", "AllTags"])
    ap.add_argument("--json-out", default="results_cv_silver/scaling_curve.json")
    ap.add_argument("--csv-out", default="results_cv_silver/scaling_curve.csv")
    args = ap.parse_args()

    points: List[dict] = []

    # Optional 0-silver reference (training from scratch on gold only).
    if args.reference_dir:
        ref = Path(args.reference_dir)
        # ktb_only has the SAME config sub-dir name; read it as budget=0.
        cfg_dir = ref
        jk_path = cfg_dir / "jackknifed.json"
        pf_path = cfg_dir / "per_fold.json"
        if jk_path.exists() and pf_path.exists():
            jk = json.loads(jk_path.read_text(encoding="utf-8"))
            pf = json.loads(pf_path.read_text(encoding="utf-8"))
            official = jk["official_jackknifed"]
            per_fold = {}
            for metric in args.metrics:
                vals = [r["official"][metric] for r in pf]
                per_fold[metric] = {"mean": statistics.mean(vals),
                                    "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
                                    "n": len(vals)}
            points.append({"budget_tokens": 0, "label": "0 (ktb_only)",
                           "n_folds": jk["n_folds"], "official_jackknifed": official,
                           "per_fold": per_fold, "dir": str(ref)})
            print(f"  reference 0 (ktb_only): loaded ({jk['n_folds']} folds)")
        else:
            print(f"  WARN: reference {ref} missing jackknifed.json/per_fold.json; skipping 0-point")

    for b in sorted(set(args.budgets)):
        run_dir = Path(args.root) / f"scaling_{_suffix(b)}"
        point = _load_point(run_dir)
        if point is None:
            print(f"  budget {b}: NOT FOUND/INCOMPLETE at {run_dir} -- skipping")
            continue
        point["budget_tokens"] = b
        point["label"] = _suffix(b)
        point["dir"] = str(run_dir)
        points.append(point)
        o = point["official_jackknifed"]
        print(f"  budget {b:>7} ({_suffix(b)}): {point['n_folds']} folds  "
              f"Lemmas={o['Lemmas']*100:5.2f}  UPOS={o['UPOS']*100:5.2f}  "
              f"UFeats={o['UFeats']*100:5.2f}  AllTags={o['AllTags']*100:5.2f}")

    if not points:
        print("\nNo complete points found. Run the scaling CV first.")
        return

    # ---- JSON artifact (for the manuscript figure) ----
    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_out).write_text(
        json.dumps({"points": points, "metrics": args.metrics},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nJSON: {args.json_out}")

    # ---- CSV (one row per point, columns per metric) ----
    with open(args.csv_out, "w", encoding="utf-8", newline="\n") as fh:
        w = csv.writer(fh)
        header = ["budget_tokens", "label", "n_folds"]
        for m in args.metrics:
            header += [f"{m}_jackknife", f"{m}_fold_mean", f"{m}_fold_std"]
        w.writerow(header)
        for p in points:
            row = [p["budget_tokens"], p["label"], p["n_folds"]]
            for m in args.metrics:
                jk = p["official_jackknifed"].get(m, float("nan"))
                pf = p["per_fold"].get(m, {"mean": float("nan"), "std": 0})
                row += [round(jk, 4), round(pf["mean"], 4), round(pf["std"], 4)]
            w.writerow(row)
    print(f"CSV:  {args.csv_out}")

    # ---- terminal sanity curve ----
    print("\nScaling curve (jack-knifed F1, %):")
    for m in args.metrics:
        lo = min(p["official_jackknifed"][m] for p in points)
        hi = max(p["official_jackknifed"][m] for p in points)
        span = hi - lo
        # pad the bar range so a single flat metric still shows something
        if span < 0.02:
            lo, hi = lo - 0.01, hi + 0.01
        print(f"\n  {m}  (range {lo*100:.2f}..{hi*100:.2f}):")
        for p in points:
            v = p["official_jackknifed"][m]
            print(f"    {p['label']:>14}: {v*100:6.2f}  {_bar(v, lo, hi)}")


if __name__ == "__main__":
    main()
