"""Build a reproducible silver corpus from the connected book skeleton.

The connected corpus (``data/auxiliary/books_skeleton.conllu``, 220,098
sentences / 2,886,405 tokens) is tokenised and sentence-segmented but
unannotated. This script fills the lemma (col 3), UPOS (col 4) and FEATS
(col 6) columns by running a morphological annotator over each sentence and
mapping its output into the gold UD label space (see cagf.kaznlp_ud_map).

Two annotator backends are provided:

  * ``kaznlp`` (default, intended for the published silver corpus): the
    KazNLP data-driven analyzer + trigram-HMM tagger. KazNLP is not a PyPI
    package; clone https://github.com/nlacslab/kaznlp and make it importable
    (e.g. ``PYTHONPATH=/path/to/kaznlp``). Point ``--kaznlp-model`` at the
    model directory shipped with the clone (``kaznlp/morphology/mdl``). The
    HMM tagger is deterministic (Viterbi), so a fixed KazNLP version plus the
    fixed mapping module yields a byte-reproducible silver corpus.

  * ``rule`` (offline fallback, no external models): the repository's own
    ``cagf.kk_morphrules`` nominal guesser. It requires nothing beyond this
    repo and lets the full pipeline run and be tested anywhere, but produces
    markedly lower-quality annotation and must never be presented as the
    KazNLP silver corpus.

Every token is written with a MISC provenance field
(``Silver=<backend>|Map=<version>|UnkPOS=<0|1>|Dropped=<n>|Unmapped=<n>``) so
silver tokens can never be mistaken for gold, and so the per-token mapping
diagnostics (unknown POS, gold-gated FEATS drops, unmapped morphemes) travel
with the corpus for the downstream filtering stage
(``scripts/filter_silver_corpus.py``) without having to re-run tagging. This
script itself does not drop or select sentences -- its output is the
*unfiltered* silver corpus (stage: KazNLP pre-annotation + KLC-to-UD mapping).
A JSON coverage report is written next to the output.

Usage
-----
    # intended, KazNLP-based silver corpus
    PYTHONPATH=.:/path/to/kaznlp python scripts/build_silver_corpus.py \
        --input data/auxiliary/books_skeleton.conllu \
        --output data/silver/silver.conllu \
        --backend kaznlp --kaznlp-model /path/to/kaznlp/kaznlp/morphology/mdl

    # offline smoke test on the first 500 sentences (no KazNLP needed)
    PYTHONPATH=. python scripts/build_silver_corpus.py \
        --input data/auxiliary/books_skeleton.conllu \
        --output data/silver/silver_sample.conllu \
        --backend rule --limit 500

The output of this script is the corpus BEFORE filtering. To obtain the
filtered silver corpus used in the main transfer-learning configuration,
run it through ``scripts/filter_silver_corpus.py``. The unfiltered output can
also be used directly, to run the "unfiltered silver pretraining" ablation
that measures whether the filtering stage is actually worth it.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Tuple

from cagf.kaznlp_ud_map import MAPPING_VERSION, map_analysis


# --------------------------------------------------------------------------
# streaming reader: yield (comment_lines, [forms]) per sentence
# --------------------------------------------------------------------------
def iter_skeleton(path: str | Path) -> Iterator[Tuple[List[str], List[str]]]:
    comments: List[str] = []
    forms: List[str] = []
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if not line.strip():
                if forms:
                    yield comments, forms
                comments, forms = [], []
                continue
            if line.startswith("#"):
                comments.append(line)
                continue
            cols = line.split("\t")
            if len(cols) < 2:
                continue
            tid = cols[0]
            if "-" in tid or "." in tid:      # skip multiword/empty nodes
                continue
            forms.append(cols[1])
    if forms:
        yield comments, forms


# --------------------------------------------------------------------------
# backends: each takes a list of surface forms and returns, per token, a
# (lemma, upos, feats) triple plus a merged coverage counter.
# --------------------------------------------------------------------------
TokenAnnotation = Tuple[str, str, str, Dict[str, int]]  # (lemma, upos, feats, per_token_counters)


def make_kaznlp_backend(model_dir: str) -> Callable[[List[str]], List[TokenAnnotation]]:
    # imported lazily so the ``rule`` backend needs no KazNLP install
    from kaznlp.morphology.analyzers import AnalyzerDD
    from kaznlp.morphology.taggers import TaggerHMM

    analyzer = AnalyzerDD()
    analyzer.load_model(model_dir)
    tagger = TaggerHMM(lyzer=analyzer)
    tagger.load_model(model_dir)

    def annotate(forms: List[str]) -> List[TokenAnnotation]:
        lowered = [f.lower() for f in forms]
        analyses = tagger.tag_sentence(lowered)   # deterministic Viterbi, one analysis per token
        out: List[TokenAnnotation] = []
        for form, analysis in zip(forms, analyses):
            lemma, upos, feats, counters = map_analysis(form, analysis)
            out.append((lemma, upos, feats, counters))
        return out

    return annotate


def make_rule_backend() -> Callable[[List[str]], List[TokenAnnotation]]:
    """Offline fallback. Emits a KazNLP-shaped analysis string from the
    repository's nominal guesser so it flows through the exact same mapping
    code path as the KazNLP backend."""
    from cagf.kk_morphrules import guess_nominal_morphology

    _CASE_TO_KLC = {"Gen": "C2", "Dat": "C3", "Acc": "C4", "Loc": "C5", "Abl": "C6", "Ins": "C7"}

    def annotate(forms: List[str]) -> List[TokenAnnotation]:
        out: List[TokenAnnotation] = []
        for form in forms:
            if not any(ch.isalpha() for ch in form):
                # treat as punctuation / symbol; map_analysis handles R_NKT etc.
                analysis = f"{form}_R_NKT"
            else:
                g = guess_nominal_morphology(form)
                segs = [f"{g.lemma_guess}_R_ZE"]
                if g.number == "Plur":
                    segs.append("дар_N1")
                if g.case is not None:
                    segs.append(f"x_{_CASE_TO_KLC[g.case]}")
                analysis = " ".join(segs)
            lemma, upos, feats, counters = map_analysis(form, analysis)
            out.append((lemma, upos, feats, counters))
        return out

    return annotate


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="data/auxiliary/books_skeleton.conllu")
    ap.add_argument("--output", default="data/silver/silver.conllu")
    ap.add_argument("--backend", choices=["kaznlp", "rule"], default="kaznlp")
    ap.add_argument("--kaznlp-model", default="kaznlp/morphology/mdl",
                    help="KazNLP model directory (only used by the kaznlp backend)")
    ap.add_argument("--limit", type=int, default=0, help="annotate only the first N sentences (0 = all)")
    ap.add_argument("--seed", type=int, default=42, help="recorded for provenance; tagging itself is deterministic")
    args = ap.parse_args()

    if args.backend == "kaznlp":
        annotate = make_kaznlp_backend(args.kaznlp_model)
    else:
        annotate = make_rule_backend()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    coverage: Counter = Counter()
    n_sent = 0
    n_tok = 0
    upos_counts: Counter = Counter()

    with open(out_path, "w", encoding="utf-8") as fout:
        fout.write(f"# silver_corpus backend={args.backend} map_version={MAPPING_VERSION} seed={args.seed}\n")
        for comments, forms in iter_skeleton(args.input):
            if args.limit and n_sent >= args.limit:
                break
            annotations = annotate(forms)
            for c in comments:
                fout.write(c + "\n")
            for i, (form, (lemma, upos, feats, counters)) in enumerate(zip(forms, annotations), start=1):
                upos_counts[upos] += 1
                coverage.update(counters)
                unk_pos = counters.get("unknown_pos", 0)
                dropped = counters.get("dropped_non_gold", 0)
                unmapped = counters.get("unmapped_morpheme", 0)
                misc = (f"Silver={args.backend}|Map={MAPPING_VERSION}"
                        f"|UnkPOS={unk_pos}|Dropped={dropped}|Unmapped={unmapped}")
                # CoNLL-U: ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS MISC
                fout.write("\t".join([str(i), form, lemma, upos, "_", feats, "_", "_", "_", misc]) + "\n")
                n_tok += 1
            fout.write("\n")
            n_sent += 1

    report = {
        "backend": args.backend,
        "map_version": MAPPING_VERSION,
        "input": str(args.input),
        "output": str(out_path),
        "sentences": n_sent,
        "tokens": n_tok,
        "upos_distribution": dict(sorted(upos_counts.items(), key=lambda kv: -kv[1])),
        "coverage_events": dict(coverage),
    }
    report_path = out_path.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {n_sent:,} sentences / {n_tok:,} tokens to {out_path}")
    print(f"Coverage report: {report_path}")
    print(f"UPOS distribution (top): {dict(list(report['upos_distribution'].items())[:8])}")
    print(f"Coverage events: {dict(coverage)}")
    if args.backend == "rule":
        print("\nNOTE: 'rule' is the OFFLINE FALLBACK backend. It is for pipeline")
        print("testing only and must NOT be reported as the KazNLP silver corpus.")


if __name__ == "__main__":
    main()
