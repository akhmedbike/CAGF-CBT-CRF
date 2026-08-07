"""POS-conditional morphological feature constraints (post-hoc, no retraining).

A cheap, retraining-free way to remove a class of morphological errors: a
token predicted as ``VERB`` should not carry ``Case=Dat`` (the
``шыға / VERB / Case=Dat`` kind of error), and a ``NOUN`` should not carry
``Tense=Past``. We learn, from the training set, the set of
``(upos, feature=value)`` combinations that were ever observed, and at
inference time we mask out (set to -inf) any grammeme-label whose combination
with the predicted UPOS was never seen in training.

Why post-hoc
------------
The model already predicts UPOS and grammemes jointly via independent heads.
Re-training with hard constraints would require coupling the heads, which is
out of scope for the revision timeline. Post-hoc masking takes the model's
UPOS prediction as given and only filters the grammeme head's output, so it
applies to any already-trained checkpoint (and to any CV fold prediction
file) without re-running training. The cost is that a wrong UPOS prediction
propagates -- but that is a property of the base model, not of this fix.

What counts as "impossible"
---------------------------
The constraint is **data-driven, not hand-written**. We never hard-code
"VERB cannot have Case" -- in fact KTB verbs DO take Case (converbs,
gerunds: ``келгенде`` "having come-Loc"). Instead, the allowed set for each
UPOS is exactly the ``Feature=Value`` pairs observed with that UPOS in the
training partition. This makes the method corpus-portable and avoids imposing
a linguist's prior that the data contradicts.

The value "absent" is ALWAYS allowed: if the model predicts no feature for a
slot, we never force one in. Masking can only remove a prediction, never add.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Set, Tuple

from .data import Sentence, parse_feats


# Type alias: (upos, feature_name) -> set of values seen in training.
# feature_name is e.g. "Case"; value is e.g. "Dat". We key on the pair because
# the same value can be legal for one UPOS and illegal for another, but the
# feature NAME is the more stable unit (Case means Case everywhere).
ConstraintMap = Dict[Tuple[str, str], Set[str]]


def build_constraints(train_sentences: Sequence[Sentence]) -> ConstraintMap:
    """Learn the observed ``(upos, feature) -> {values}`` map from training data.

    Every pair that occurs at least once is allowed; anything not seen is
    treated as impossible at inference time. The training partition alone is
    used -- never dev/test -- so the constraint set itself cannot leak.
    """
    allowed: Dict[Tuple[str, str], Set[str]] = {}
    for s in train_sentences:
        for tok in s.tokens:
            for feat in parse_feats(tok.feats):
                if '=' not in feat:
                    continue
                name, value = feat.split('=', 1)
                allowed.setdefault((tok.upos, name), set()).add(value)
    return allowed


def allowed_values_for(upos: str, feature_name: str,
                       constraints: ConstraintMap) -> Set[str] | None:
    """Return the set of allowed values, or None if the (upos, feature) pair
    itself was never seen (in which case we mask the whole feature out)."""
    return constraints.get((upos, feature_name))


def apply_constraints_to_feats(pred_feats: List[str], upos_pred: str,
                               constraints: ConstraintMap) -> List[str]:
    """Filter a token's predicted feature list, keeping only the
    ``Feature=Value`` pairs whose combination with ``upos_pred`` was observed
    in training.

    Returns the filtered list (possibly empty). The caller serialises empty ->
    ``_`` in the CoNLL-U writer. This is the core post-hoc operation: it never
    adds a feature, only removes impossible ones.
    """
    kept: List[str] = []
    for feat in pred_feats:
        if '=' not in feat:
            continue
        name, value = feat.split('=', 1)
        allowed = constraints.get((upos_pred, name))
        if allowed is None:
            # whole feature name unseen with this UPOS -> drop
            continue
        if value in allowed:
            kept.append(feat)
        # else: value unseen with this UPOS -> drop (the impossible combo)
    return kept


def count_impossible(pred_rows: Sequence[dict], constraints: ConstraintMap) -> dict:
    """Tally impossible (upos, feature=value) combos in a set of predictions.

    ``pred_rows`` is a list of dicts, one per token, each carrying ``upos``
    and ``feats`` (a list of ``Feature=Value`` strings). Returns counts that
    feed the before/after table in the paper.
    """
    total_feats = 0
    impossible_before = 0
    for row in pred_rows:
        upos = row['upos']
        for feat in row['feats']:
            if '=' not in feat:
                continue
            total_feats += 1
            name, value = feat.split('=', 1)
            allowed = constraints.get((upos, name))
            if allowed is None or value not in allowed:
                impossible_before += 1
    return {
        'total_predicted_feats': total_feats,
        'impossible_before': impossible_before,
        'impossible_rate_before': impossible_before / total_feats if total_feats else 0.0,
    }
