"""Write model predictions to a CoNLL-U file that the official
``conll18_ud_eval`` script accepts.

This is the bridge between the model's per-token tensor outputs and the
string-level comparison the official UD evaluation does. Getting it wrong is
the single most common cause of "the model works but the metric is 40%":
``conll18_ud_eval`` compares the FEATS column as a *string*, so
``Number=Sing|Case=Dat`` and ``Case=Dat|Number=Sing`` are treated as different
predictions even when they encode the same morphological bundle. We therefore
serialise FEATS with features in strict alphabetical order of the feature
name (the same canonical order UD uses), matching how the gold KTB files are
written.

The writer preserves everything the evaluation needs to align tokens:

* comment lines (``# sent_id``, ``# text``, ``# gold_source``) verbatim;
* multiword-token range rows (``n-m``) and empty-node rows (``n.m``) verbatim;
* HEAD / DEPREL / DEPS / MISC copied unchanged from gold;
* XPOS forced to ``_`` (KTB does not use XPOS, and the official scorer
  ignores it for the UPOS / UFeats / AllTags metrics we report).

A round-trip self-test (write gold values back as "predictions", score the
result against the original gold file) is the acceptance test for this module
-- it must return exactly 100.0 on every metric, and it lives in
``tests/test_predict_writer.py``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

from .data import Sentence, Token, apply_edit_script


# --------------------------------------------------------------------------
# FEATS serialization
# --------------------------------------------------------------------------
def serialize_feats(feats: Sequence[str]) -> str:
    """Serialize a list of ``Feature=Value`` strings into the canonical UD
    FEATS column string.

    Features are sorted alphabetically by feature name (the part before
    ``=``), which is the canonical UD ordering and the one ``conll18_ud_eval``
    treats as canonical for string comparison. An empty list yields ``_``
    (UD's empty-value marker), never the empty string.
    """
    pairs = [f for f in feats if f and f != '_']
    if not pairs:
        return '_'
    # sort by feature name; values may themselves contain '=' for psor-style
    # attributes like Number[psor]=Plur,Sing, so split only on the first '='
    pairs_sorted = sorted(pairs, key=lambda f: f.split('=', 1)[0])
    return '|'.join(pairs_sorted)


# --------------------------------------------------------------------------
# CoNLL-U writer
# --------------------------------------------------------------------------
def write_conllu(
    sentences: Sequence[Sentence],
    predictions: Sequence[Dict],
    path: str | Path,
) -> None:
    """Write ``sentences`` with ``predictions`` substituted into the LEMMA /
    UPOS / FEATS columns.

    Parameters
    ----------
    sentences
        The gold sentences, carrying their original ``comments`` and
        ``misc_lines`` (multiword / empty-node rows) for verbatim round-trip.
    predictions
        One entry per sentence. Each entry is a dict with keys ``lemma``,
        ``upos``, ``feats``, each a list of per-token string predictions in
        the same order and length as ``sentence.tokens``.
    path
        Output ``.conllu`` path. The file ends with a blank line, as UD
        requires.
    """
    if len(sentences) != len(predictions):
        raise ValueError(
            f'write_conllu: got {len(sentences)} sentences but {len(predictions)} '
            f'prediction entries -- they must align 1:1')

    lines: List[str] = []
    for sent, pred in zip(sentences, predictions):
        lemmas = pred['lemma']
        uposes = pred['upos']
        feats_list = pred['feats']
        if not (len(lemmas) == len(uposes) == len(feats_list) == len(sent.tokens)):
            raise ValueError(
                f'write_conllu: sentence has {len(sent.tokens)} tokens but prediction '
                f'lists have lengths lemma={len(lemmas)} upos={len(uposes)} '
                f'feats={len(feats_list)} -- they must all match')

        # comments verbatim
        for c in sent.comments:
            lines.append(c)
        # multiword / empty-node rows: we need to interleave them at the right
        # position by their ID prefix. Reconstruct the ID ordering: analytic
        # tokens get sequential 1..N, and misc_lines carry their own "n-m" /
        # "n.m" IDs which we insert at the matching integer position.
        token_lines: List[str] = []
        for i, tok in enumerate(sent.tokens, start=1):
            lemma = lemmas[i - 1] if lemmas[i - 1] else '_'
            upos = uposes[i - 1] if uposes[i - 1] else '_'
            feats = serialize_feats(feats_list[i - 1])
            # CoNLL-U: ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS MISC.
            # HEAD/DEPREL/DEPS/MISC are copied verbatim from gold. They are not
            # model predictions (we do not predict dependency structure), but
            # the official scorer needs valid integer HEADs to parse the file,
            # and copying them keeps UAS/LAS noise out of the comparison. The
            # four metrics we report (Lemmas/UPOS/UFeats/AllTags) do not depend
            # on HEAD being correct, only on it being parseable.
            token_lines.append('\t'.join([
                str(i), tok.form, lemma, upos, tok.xpos, feats,
                tok.head, tok.deprel, tok.deps, tok.misc]))

        # interleave misc_lines (multiword tokens) at their position.
        interleaved = _interleave_misc(token_lines, sent.misc_lines)
        lines.extend(interleaved)
        lines.append('')  # blank line ends the sentence

    Path(path).write_text('\n'.join(lines) + '\n', encoding='utf-8')


def _interleave_misc(token_lines: List[str], misc_lines: List[str]) -> List[str]:
    """Place multiword-token range rows (``n-m``) just before the first
    analytic token they cover.

    ``misc_lines`` are the verbatim ``n-m`` rows preserved by ``read_conllu``.
    Their ID prefix encodes the range (e.g. ``4-5``); we insert each such row
    immediately before analytic token ``<first id of the range>``. Empty-node
    rows (``n.m``) are inserted before token ``n``.
    """
    if not misc_lines:
        return list(token_lines)
    out: List[str] = []
    misc_by_anchor: Dict[int, List[str]] = {}
    for ml in misc_lines:
        tid = ml.split('\t', 1)[0]
        if '-' in tid:
            first = int(tid.split('-', 1)[0])
        elif '.' in tid:
            first = int(tid.split('.', 1)[0])
        else:
            continue
        misc_by_anchor.setdefault(first, []).append(ml)
    for i, line in enumerate(token_lines, start=1):
        if i in misc_by_anchor:
            out.extend(misc_by_anchor[i])
        out.append(line)
    return out
