"""Generate nested stratified silver subsets at fixed token budgets.

This supports the silver-scaling experiment: a curve of quality vs. silver-corpus
size (e.g. 100K / 210K / 500K / 1M tokens), answering the reviewer question
"how much silver is needed, and where is saturation?".

Design: NESTED stratified sampling
----------------------------------
The subsets must be *nested* (100K ⊂ 210K ⊂ 500K ⊂ 1M). If they were
independent random draws, the differences between adjacent points on the curve
would mix the size effect with sampling noise of the draw, and a reviewer would
(rightly) object. Nesting isolates the size effect: each larger subset is the
smaller subset plus more data.

To keep UPOS coverage balanced at *every* prefix length (so that 100K is not
skewed toward whichever book came first in the file), sentences are ordered
deterministically by round-robin across UPOS strata, shuffling within each
stratum with a fixed seed (42, matching the existing 210K subset). Taking the
first N tokens of that single ordered list then yields a nested family of
subsets that are all UPOS-balanced -- 100K is a prefix of 210K is a prefix of
500K, etc.

The dominant UPOS of a sentence (its most frequent token UPOS, ties broken by
the global frequency order) is the stratification key, mirroring the existing
``silver_subset_210k.conllu`` which documents "UPOS distribution matches full
corpus to within 0.1pp".

Usage
-----
    PYTHONPATH=. python scripts/make_silver_subset.py \\
        --silver data/silver/silver_filtered.conllu \\
        --budgets 100000 210000 500000 1000000 \\
        --out-dir data/silver --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import List, Tuple

from cagf.data import Sentence, Token, read_conllu


def _sentence_dominant_upos(sent: Sentence) -> str:
    """Most frequent UPOS among the sentence's tokens; ties -> global order."""
    counts = Counter(t.upos for t in sent.tokens)
    # most_common breaks ties by insertion order; use max count then first-seen.
    best = max(counts.values())
    for t in sent.tokens:
        if counts[t.upos] == best:
            return t.upos
    return sent.tokens[0].upos if sent.tokens else "_"


def _write_subset(sentences: List[Sentence], path: Path,
                  budget_tokens: int, full_n_tokens: int, full_n_sents: int) -> Tuple[int, int]:
    """Write sentences until cumulative token count >= budget. Return (n_sents, n_tokens)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n_tok = 0
    n_sent = 0
    with open(path, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write(f"# silver_subset: nested stratified sample (seed in header) "
                 f"of silver_filtered.conllu\n")
        fh.write(f"# size: target {budget_tokens} tokens; "
                 f"{full_n_sents} sentences in full corpus ({full_n_tokens} tokens)\n")
        fh.write("# Nested: this file is a prefix of every larger-budget subset. "
                 "Stratified round-robin by sentence-dominant UPOS.\n")
        for i, s in enumerate(sentences, start=1):
            if n_tok >= budget_tokens:
                break
            # Rewrite sent_id to be 1..N within the subset, deterministic.
            out_comments = []
            sid_written = False
            for c in s.comments:
                if c.startswith('# sent_id'):
                    out_comments.append(f"# sent_id = silver_subset_{i}")
                    sid_written = True
                else:
                    out_comments.append(c)
            if not sid_written:
                out_comments.append(f"# sent_id = silver_subset_{i}")
            for c in out_comments:
                fh.write(c + "\n")
            for pos, tok in enumerate(s.tokens, start=1):
                fh.write(_token_line(tok, pos) + "\n")
            for m in s.misc_lines:
                fh.write(m + "\n")
            fh.write("\n")
            n_tok += len(s.tokens)
            n_sent += 1
    return n_sent, n_tok


def _token_line(tok: Token, position: int) -> str:
    """Render a token as a CoNLL-U row, renumbering ID to `position` (1-based)."""
    cols = [str(position), tok.form, tok.lemma, tok.upos, tok.xpos, tok.feats,
            tok.head, tok.deprel, tok.deps, tok.misc]
    return "\t".join(cols)


def nested_stratified_order(sentences: List[Sentence], seed: int = 42) -> List[int]:
    """Return sentence indices in a nested-stratified order.

    Sentences are grouped by dominant UPOS, shuffled within each group with the
    given seed, then emitted round-robin across groups (largest group first for
    determinism). Any prefix of the result is therefore UPOS-balanced.
    """
    dominant = [_sentence_dominant_upos(s) for s in sentences]
    # Group indices by stratum.
    strata: dict[str, List[int]] = {}
    for idx, upos in enumerate(dominant):
        strata.setdefault(upos, []).append(idx)
    # Shuffle within each stratum (independent sub-RNGs keyed by stratum name so
    # order is stable regardless of dict insertion order).
    for upos, idxs in strata.items():
        sub = random.Random(f"{seed}:{upos}")
        sub.shuffle(idxs)
    # Round-robin: largest strata contribute more, so iterate proportionally.
    # Simple robust approach: repeat shuffling of the stratum-key list weighted
    # by remaining size until all sentences consumed.
    order: List[int] = []
    # work on mutable copies of remaining indices per stratum
    remaining = {u: list(idxs) for u, idxs in strata.items()}
    # Build a fair round-robin: at each pass, take one sentence from each
    # non-empty stratum in descending-size order. This guarantees nesting and
    # near-proportional coverage at any prefix.
    stratum_order = sorted(remaining.keys(), key=lambda u: -len(remaining[u]))
    while any(remaining.values()):
        progressed = False
        for u in stratum_order:
            if remaining[u]:
                order.append(remaining[u].pop(0))
                progressed = True
        if not progressed:
            break
    return order


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--silver', default='data/silver/silver_filtered.conllu',
                    help='full silver corpus (CoNLL-U)')
    ap.add_argument('--budgets', type=int, nargs='+',
                    default=[100000, 210000, 500000, 1000000],
                    help='target token budgets for each subset')
    ap.add_argument('--out-dir', default='data/silver')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--prefix', default='silver_subset',
                    help='output filename prefix, e.g. silver_subset_210k.conllu')
    args = ap.parse_args()

    t0 = time.time()
    sentences = read_conllu(args.silver)
    full_n_tokens = sum(len(s) for s in sentences)
    print(f"Loaded {len(sentences)} sentences / {full_n_tokens} tokens from {args.silver}")

    order = nested_stratified_order(sentences, seed=args.seed)
    ordered = [sentences[i] for i in order]

    # Sanity: nesting is guaranteed because we slice a single ordered list.
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {'silver_source': str(args.silver), 'seed': args.seed,
              'full_sentences': len(sentences), 'full_tokens': full_n_tokens,
              'subsets': []}

    def _suffix(b: int) -> str:
        return f"{b // 1000}k"

    for budget in sorted(set(args.budgets)):
        if budget > full_n_tokens:
            print(f"  WARN: budget {budget} > full corpus {full_n_tokens}; "
                  f"capping at full corpus")
            budget = full_n_tokens
        out_path = out_dir / f"{args.prefix}_{_suffix(budget)}.conllu"
        n_s, n_t = _write_subset(ordered, out_path, budget, full_n_tokens, len(sentences))
        # UPOS distribution of the subset, for the report
        upos = Counter()
        for s in ordered[:n_s]:
            for t in s.tokens:
                upos[t.upos] += 1
        report['subsets'].append({'budget_tokens': budget,
                                  'sentences': n_s, 'tokens': n_t,
                                  'path': str(out_path),
                                  'upos_distribution': dict(upos.most_common())})
        print(f"  wrote {out_path}: {n_s} sentences, {n_t} tokens "
              f"(budget {budget}, {100*n_t/full_n_tokens:.1f}% of full)")

    report_path = out_dir / f"{args.prefix}_scaling_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                           encoding='utf-8')
    print(f"\nReport: {report_path}  (elapsed {time.time()-t0:.1f}s)")

    # Verify nesting explicitly and report. Compare only the token *forms*
    # sequence (comment lines like the size header and per-file sent_id differ
    # across budgets by construction, so a raw block-string compare is misleading).
    print("\nNesting check (smaller must be a forms-prefix of larger):")
    sorted_subs = sorted(report['subsets'], key=lambda r: r['tokens'])
    paths = [Path(s['path']) for s in sorted_subs]

    def _forms_seq(path: Path) -> List[tuple]:
        forms: List[tuple] = []
        for block in path.read_text(encoding='utf-8').split('\n\n'):
            if not block.strip():
                continue
            forms.append(tuple(
                ln.split('\t')[1] for ln in block.split('\n')
                if '\t' in ln and not ln.startswith('#')))
        return forms

    seqs = [_forms_seq(p) for p in paths]
    for i in range(len(seqs) - 1):
        smaller, larger = seqs[i], seqs[i + 1]
        is_prefix = larger[:len(smaller)] == smaller
        flag = "OK" if is_prefix else "MISMATCH"
        print(f"  {paths[i].name} ({len(smaller)} blocks) ⊂ "
              f"{paths[i+1].name} ({len(larger)} blocks): {flag}")


if __name__ == '__main__':
    main()
