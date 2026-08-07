"""Sentence-level filtering of the raw (unfiltered) silver corpus.

Pipeline position
------------------
This module implements one discrete data-preparation stage, inserted
between KLC-to-UD mapping and the final silver corpus used for pretraining::

    Raw auxiliary corpus
          |
    KazNLP pre-annotation                  (scripts/build_silver_corpus.py)
          |
    KazNLP-to-UD label mapping             (cagf/kaznlp_ud_map.py)
          |
    Filtering of incomplete/inconsistent   (this module,
    annotations                             scripts/filter_silver_corpus.py)
          |
    Silver corpus
          |
    Pretraining -> Fine-tuning on gold KTB -> Evaluation on gold KTB test

It does not change the model architecture or the mapping tables; it only
decides, per silver sentence, whether the mapped annotation is trustworthy
enough to pretrain on. Two independent, additive quality signals are used,
both computed from information already written into MISC by
``scripts/build_silver_corpus.py`` (``UnkPOS``, ``Dropped``, ``Unmapped``
per token) plus basic sentence-length sanity checks -- nothing here depends
on gold labels, so it cannot leak gold information into the silver corpus.

1.  **Incompleteness**: the fraction of tokens KazNLP could not analyse at
    all (``UnkPOS=1``, mapped to UPOS=X for lack of any POS tag) is high.
    A sentence dominated by unanalysed tokens contributes mostly noise to
    pretraining.
2.  **Inconsistency**: the fraction of *morphological* signal that had to be
    discarded by the gold-grammeme gate (``Dropped``) or that KazNLP
    produced but the mapping table does not recognise at all
    (``Unmapped``) is high. This flags sentences where the tagger's output
    disagreed heavily with the label space fine-tuning will target.

Degenerate sentences (too short to carry usable context, or implausibly
long -- usually a sentence-segmentation failure on the raw text) are
rejected regardless of tagging quality.

All thresholds are explicit, documented defaults, not tuned on gold data.
They should be reported alongside any experiment that uses the filtered
silver corpus, and are written into the filter report for that reason.
"""
from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Tuple


@dataclasses.dataclass(frozen=True)
class FilterThresholds:
    """Explicit, documented cutoffs for the filtering stage.

    These are engineering defaults, analogous in spirit to the ones in
    ``configs/default.yaml``: reasonable starting points, not values tuned
    against gold-test performance (which would bias the "filtering helps"
    comparison). Report the thresholds actually used alongside any results.
    """
    max_unknown_pos_ratio: float = 0.3     # drop if >30% of tokens are UPOS=X (unanalysed)
    max_noise_ratio: float = 0.5           # drop if >0.5 dropped/unmapped morpheme events per token
    min_tokens: int = 2                    # drop degenerate one-token "sentences"
    max_tokens: int = 200                  # drop implausibly long sentences (segmentation failures)


@dataclasses.dataclass(frozen=True)
class TokenDiagnostics:
    unknown_pos: int
    dropped_non_gold: int
    unmapped_morpheme: int


@dataclasses.dataclass(frozen=True)
class SentenceAssessment:
    kept: bool
    reasons: Tuple[str, ...]          # empty iff kept
    n_tokens: int
    unknown_pos_ratio: float
    noise_ratio: float


def parse_misc_diagnostics(misc: str) -> TokenDiagnostics:
    """Parse the ``UnkPOS=/Dropped=/Unmapped=`` fields written by
    ``scripts/build_silver_corpus.py`` into a :class:`TokenDiagnostics`.

    Tokens without these fields (e.g. gold data, or a MISC written by an
    older version of the builder) are treated as fully clean (all zeros),
    so this function is safe to call on any CoNLL-U MISC field.
    """
    values: Dict[str, int] = {}
    if misc and misc != "_":
        for part in misc.split("|"):
            if "=" not in part:
                continue
            key, _, val = part.partition("=")
            if key in ("UnkPOS", "Dropped", "Unmapped"):
                try:
                    values[key] = int(val)
                except ValueError:
                    values[key] = 0
    return TokenDiagnostics(
        unknown_pos=values.get("UnkPOS", 0),
        dropped_non_gold=values.get("Dropped", 0),
        unmapped_morpheme=values.get("Unmapped", 0),
    )


def assess_sentence(token_diagnostics: List[TokenDiagnostics],
                    thresholds: Optional[FilterThresholds] = None) -> SentenceAssessment:
    """Decide whether one silver sentence passes the filter.

    A sentence can be rejected for more than one reason at once; all
    applicable reasons are reported (useful for the aggregate breakdown in
    the filter report), but rejection is a single kept=False decision.
    """
    thresholds = thresholds or FilterThresholds()
    n_tokens = len(token_diagnostics)
    reasons: List[str] = []

    if n_tokens < thresholds.min_tokens:
        reasons.append("too_short")
    if n_tokens > thresholds.max_tokens:
        reasons.append("too_long")

    if n_tokens > 0:
        unknown_pos_ratio = sum(d.unknown_pos for d in token_diagnostics) / n_tokens
        noise_ratio = sum(d.dropped_non_gold + d.unmapped_morpheme
                          for d in token_diagnostics) / n_tokens
    else:
        unknown_pos_ratio = 1.0
        noise_ratio = 0.0

    if unknown_pos_ratio > thresholds.max_unknown_pos_ratio:
        reasons.append("incomplete_annotation")
    if noise_ratio > thresholds.max_noise_ratio:
        reasons.append("inconsistent_annotation")

    return SentenceAssessment(
        kept=not reasons,
        reasons=tuple(reasons),
        n_tokens=n_tokens,
        unknown_pos_ratio=unknown_pos_ratio,
        noise_ratio=noise_ratio,
    )
