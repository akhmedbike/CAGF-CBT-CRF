"""Exhaustive decomposition of UPOS=X and heuristic defaults in the silver corpus.

Supports the revision claims (letter R2.2 / manuscript Section 2.5):

* the X-cut by origin — KazNLP analyzer OOV fallback (root tag ``R_X`` /
  ``R_BOS`` mapped to ``X`` through POS_MAP) vs. mapping-table misses
  (root tags absent from POS_MAP, the ``unknown_pos`` counter);
* the heuristic feature defaults inserted by ``cagf/kaznlp_ud_map.py``
  (Case=Nom / Number=Sing on nominals, VerbForm=Fin / Mood=Ind on finite
  verbs), including which values have NO source in MORPH_MAP and which do
  (subject-agreement P1/P2/P4/P8 also yield Number=Sing on verbs);
* the unmapped-morpheme accounting (one event per affected token, none
  dropped from the gold inventory);
* the X-prior shift against gold KTB.

Pass 1 reads the silver CoNLL-U directly (per-token MISC provenance:
``UnkPOS=|Dropped=|Unmapped=``). Pass 2 optionally re-runs the KazNLP
analyzer over every unique X surface form to verify the OOV-fallback claim;
it needs the kaznlp clone (see README, CC-BY-SA, not vendored) and is
skipped with a recorded note when unavailable. Deterministic: no sampling,
sorted iteration, PYTHONHASHSEED=0 recommended.

Usage:
    PYTHONHASHSEED=0 PYTHONPATH=.:kaznlp python scripts/silver_decomposition.py
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from cagf.kaznlp_ud_map import MORPH_MAP, POS_MAP, TRANSITION_MAP

NOMINAL_UPOS = {"NOUN", "PROPN", "PRON", "ADJ", "NUM"}
VERBAL_UPOS = {"VERB", "AUX"}
# values that CAN come out of a map (Number=Sing via agreement tags)
DEFAULT_VALUES = ("Case=Nom", "Number=Sing", "VerbForm=Fin", "Mood=Ind")
# values with no morpheme source at all -> every occurrence is a default
PURE_DEFAULTS = ("Case=Nom", "VerbForm=Fin", "Mood=Ind")

LATIN_RE = re.compile(r"[A-Za-z]")
CYRILLIC_RE = re.compile(r"[^\x00-\x7F]")


def parse_misc(misc: str) -> dict:
    out = {}
    for part in misc.split("|"):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k] = v
    return out


def scan_silver(path: Path) -> dict:
    upos = Counter()
    x_forms = Counter()          # forms of X tokens, UnkPOS=0 (analyzer path)
    x_unkpos1 = 0
    x_tokens = 0
    x_with_feats = 0
    val_tokens = Counter()       # per value: tokens carrying it
    ns_nominal = ns_verbal = 0   # Number=Sing split by UPOS
    any_of_four = 0
    explicit_ns_only = 0
    unmapped_sum = unmapped_tokens = unmapped_max = 0
    dropped_sum = 0
    total = 0
    mixed_forms = Counter()      # mixed-script forms among X (UnkPOS=0)
    t0 = time.time()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            cols = line.rstrip("\n").split("\t")
            if not cols[0].isdigit():
                continue
            total += 1
            u = cols[3]
            upos[u] += 1
            feats = cols[5] if cols[5] != "_" else ""
            fset = set(feats.split("|")) if feats else set()
            misc = parse_misc(cols[9]) if len(cols) > 9 else {}
            unm = int(misc.get("Unmapped", 0))
            if unm:
                unmapped_sum += unm
                unmapped_tokens += 1
                unmapped_max = max(unmapped_max, unm)
            dropped_sum += int(misc.get("Dropped", 0))

            has_pure_default = False
            for v in PURE_DEFAULTS:
                if v in fset:
                    val_tokens[v] += 1
                    has_pure_default = True
            if "Number=Sing" in fset:
                val_tokens["Number=Sing"] += 1
                if u in NOMINAL_UPOS:
                    ns_nominal += 1
                    has_pure_default = True   # nominal singular is always a default
                elif u in VERBAL_UPOS:
                    ns_verbal += 1            # agreement-derived via P1/P2/P4/P8
            if any(v in fset for v in DEFAULT_VALUES):
                any_of_four += 1
                # non-nominal Number=Sing is explicit (agreement morphemes);
                # it is the only one of the four values that can be mapped
                if ("Number=Sing" in fset and u not in NOMINAL_UPOS
                        and not has_pure_default):
                    explicit_ns_only += 1

            if u == "X":
                x_tokens += 1
                if fset:
                    x_with_feats += 1
                if misc.get("UnkPOS") == "1":
                    x_unkpos1 += 1
                else:
                    form = cols[1]
                    x_forms[form] += 1
                    if LATIN_RE.search(form) and CYRILLIC_RE.search(form):
                        mixed_forms[form] += 1
    return {
        "total_tokens": total,
        "upos_distribution": dict(upos),
        "x_tokens": x_tokens,
        "x_unkpos1_mapping_miss": x_unkpos1,
        "x_unkpos0_analyzer_path": x_tokens - x_unkpos1,
        "x_tokens_with_feats": x_with_feats,
        "x_unique_forms_analyzer_path": len(x_forms),
        "value_token_counts": {k: val_tokens.get(k, 0) for k in DEFAULT_VALUES},
        "number_sing_nominal_default": ns_nominal,
        "number_sing_verbal_agreement": ns_verbal,
        "tokens_with_any_of_four": any_of_four,
        "explicit_ns_only_no_default": explicit_ns_only,
        "unmapped_sum": unmapped_sum,
        "unmapped_tokens": unmapped_tokens,
        "unmapped_max_per_token": unmapped_max,
        "dropped_non_gold_sum": dropped_sum,
        "mixed_script_form_types": len(mixed_forms),
        "mixed_script_token_occurrences": sum(mixed_forms.values()),
        "top_x_forms": x_forms.most_common(20),
        "scan_seconds": round(time.time() - t0, 1),
        "_x_forms": x_forms,
    }


def analyzer_verification(forms, model_dir: str) -> dict:
    from kaznlp.morphology.analyzers import AnalyzerDD
    an = AnalyzerDD()
    an.load_model(model_dir)
    single_rx = r_bos = other = no_analysis = 0
    t0 = time.time()
    for i, form in enumerate(sorted(forms), start=1):
        res = an.analyze(form)
        anls = res[1] if isinstance(res, tuple) else (res or [])
        if not anls:
            no_analysis += 1
        elif len(anls) == 1 and anls[0] == f"{form}_R_X":
            single_rx += 1
        elif any("_R_BOS" in a for a in anls):
            r_bos += 1
        else:
            other += 1
        if i % 25000 == 0:
            print(f"  analyzer: {i}/{len(forms)} forms "
                  f"({time.time()-t0:.0f}s)", flush=True)
    return {
        "n_unique_forms": len(forms),
        "single_R_X_candidate": single_rx,
        "any_R_BOS": r_bos,
        "other_analyses": other,
        "no_analysis": no_analysis,
        "seconds": round(time.time() - t0, 1),
    }


def gold_x_share(gold_files: list) -> dict:
    total = x = 0
    for p in gold_files:
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                cols = line.split("\t")
                if cols[0].isdigit():
                    total += 1
                    x += cols[3] == "X"
    return {"gold_tokens": total, "gold_x": x,
            "gold_x_share_pct": round(100 * x / total, 2)}


def map_machine_checks() -> dict:
    mapped_values = {v for vals in MORPH_MAP.values() for v in vals}
    mapped_values |= {v for vals in TRANSITION_MAP.values() for v in vals}
    pure_ok = {v: v not in mapped_values for v in PURE_DEFAULTS}
    ns_sources = sorted(t for t, vals in MORPH_MAP.items()
                        if "Number=Sing" in vals)
    ns_sources += sorted(t for t, vals in TRANSITION_MAP.items()
                         if "Number=Sing" in vals)
    x_root_tags = sorted(t for t, u in POS_MAP.items() if u == "X")
    return {"pure_defaults_have_no_map_source": pure_ok,
            "number_sing_map_sources": ns_sources,
            "pos_map_keys_to_X": x_root_tags}


def git_info() -> dict:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"],
                                capture_output=True, text=True,
                                check=True).stdout.strip()[:12]
        dirty = bool(subprocess.run(["git", "status", "--porcelain"],
                                    capture_output=True, text=True,
                                    check=True).stdout.strip())
        return {"commit": commit, "dirty": dirty}
    except Exception:
        return {"commit": None, "dirty": None}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--silver", default="data/silver/silver.conllu")
    ap.add_argument("--report", default="data/silver/silver.report.json")
    ap.add_argument("--gold-glob", default="data/gold_merged/*.conllu")
    ap.add_argument("--kaznlp-model", default="kaznlp/kaznlp/morphology/mdl")
    ap.add_argument("--skip-analyzer", action="store_true")
    ap.add_argument("--out", default="results/silver_decomposition.json")
    args = ap.parse_args()

    print(f"scan {args.silver} ...", flush=True)
    scan = scan_silver(Path(args.silver))
    x_forms = scan.pop("_x_forms")
    print(f"  {scan['total_tokens']} tokens, X={scan['x_tokens']} "
          f"({scan['x_unkpos1_mapping_miss']} mapping-miss) "
          f"in {scan['scan_seconds']}s", flush=True)

    if args.skip_analyzer:
        analyzer = {"skipped": True,
                    "reason": "--skip-analyzer"}
    else:
        try:
            print(f"analyzer verification over "
                  f"{len(x_forms)} unique X forms ...", flush=True)
            analyzer = analyzer_verification(x_forms, args.kaznlp_model)
            print(f"  single _R_X: {analyzer['single_R_X_candidate']}"
                  f"/{analyzer['n_unique_forms']}, R_BOS: {analyzer['any_R_BOS']},"
                  f" other: {analyzer['other_analyses']}", flush=True)
        except ImportError as e:
            analyzer = {"skipped": True, "reason": f"kaznlp not importable: {e}"}

    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    totals = scan["total_tokens"]
    out = {
        "provenance": {
            "computed": datetime.now().isoformat(timespec="seconds"),
            "driver": "scripts/silver_decomposition.py",
            "git": git_info(),
            "python": sys.version.split()[0],
            "silver_path": args.silver,
            "silver_bytes": Path(args.silver).stat().st_size,
        },
        "totals": {"tokens": totals,
                   "sentences_reported": report.get("sentences")},
        "x_decomposition": {
            "x_tokens": scan["x_tokens"],
            "x_share_pct": round(100 * scan["x_tokens"] / totals, 3),
            "mapping_table_miss": scan["x_unkpos1_mapping_miss"],
            "analyzer_path": scan["x_unkpos0_analyzer_path"],
            "analyzer_path_share_pct": round(
                100 * scan["x_unkpos0_analyzer_path"] / totals, 3),
            "x_tokens_with_feats": scan["x_tokens_with_feats"],
            "verification": analyzer,
        },
        "heuristic_defaults": {
            "value_token_counts": scan["value_token_counts"],
            "number_sing_nominal_default": scan["number_sing_nominal_default"],
            "number_sing_verbal_agreement": scan["number_sing_verbal_agreement"],
            "tokens_with_any_default_value": scan["tokens_with_any_of_four"],
            "tokens_with_any_default_value_pct": round(
                100 * scan["tokens_with_any_of_four"] / totals, 2),
            "explicit_ns_only_no_default":
                scan["explicit_ns_only_no_default"],
            "machine_checks": map_machine_checks(),
        },
        "unmapped": {
            "events": scan["unmapped_sum"],
            "tokens": scan["unmapped_tokens"],
            "max_events_per_token": scan["unmapped_max_per_token"],
            "events_pct_of_corpus": round(
                100 * scan["unmapped_sum"] / totals, 3),
            "dropped_non_gold": scan["dropped_non_gold_sum"],
        },
        "gold_comparison": gold_x_share(sorted(
            __import__("glob").glob(args.gold_glob))),
        "report_consistency": {
            "tokens": report.get("tokens") == totals,
            "upos_X": report.get("upos_distribution", {}).get("X")
            == scan["x_tokens"],
            "unknown_pos": report.get("coverage_events", {}).get("unknown_pos")
            == scan["x_unkpos1_mapping_miss"],
            "unmapped_morpheme": report.get("coverage_events", {})
            .get("unmapped_morpheme") == scan["unmapped_sum"],
            "dropped_non_gold": report.get("coverage_events", {})
            .get("dropped_non_gold") == scan["dropped_non_gold_sum"],
        },
        "x_form_profile": {
            "mixed_script_form_types": scan["mixed_script_form_types"],
            "mixed_script_token_occurrences":
                scan["mixed_script_token_occurrences"],
            "top20_by_tokens": [{"form": f, "tokens": n}
                                for f, n in scan["top_x_forms"]],
        },
    }
    dest = Path(args.out)
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    print(f"wrote {dest}")
    ok = all(out["report_consistency"].values())
    print("report consistency:", "OK" if ok else "MISMATCH "
          + str(out["report_consistency"]))


if __name__ == "__main__":
    main()
