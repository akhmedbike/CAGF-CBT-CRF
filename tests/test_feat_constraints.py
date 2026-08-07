"""Tests for cagf.feat_constraints: data-driven POS-conditional masking."""
from __future__ import annotations

from cagf.data import Sentence, Token
from cagf.feat_constraints import (
    apply_constraints_to_feats,
    build_constraints,
    count_impossible,
)


def _sents(upos_feats_pairs):
    """Build single-token sentences from (upos, feats_string) pairs."""
    out = []
    for upos, feats in upos_feats_pairs:
        out.append(Sentence(tokens=[Token(form='x', lemma='x', upos=upos, feats=feats)]))
    return out


# ---------------------------------------------------------------------------
# build_constraints: learns ONLY from given data
# ---------------------------------------------------------------------------
def test_build_constraints_captures_observed_pairs():
    sents = _sents([('NOUN', 'Case=Nom|Number=Sing'),
                    ('NOUN', 'Case=Dat|Number=Plur'),
                    ('VERB', 'Tense=Past|VerbForm=Fin')])
    c = build_constraints(sents)
    assert c[('NOUN', 'Case')] == {'Nom', 'Dat'}
    assert c[('NOUN', 'Number')] == {'Sing', 'Plur'}
    assert c[('VERB', 'Tense')] == {'Past'}
    assert c[('VERB', 'VerbForm')] == {'Fin'}


def test_build_constraints_ignores_empty_feats():
    sents = _sents([('NOUN', '_'), ('NOUN', 'Case=Nom')])
    c = build_constraints(sents)
    # NOUN+Case observed once; NOUN with no feats contributes nothing
    assert c[('NOUN', 'Case')] == {'Nom'}


def test_build_constraints_verbs_can_have_case_if_data_says_so():
    # KTB reality: verbs take Case (converbs). The constraint is data-driven,
    # so if train has VERB+Case, it is allowed -- we do NOT impose a prior.
    sents = _sents([('VERB', 'Case=Loc')])
    c = build_constraints(sents)
    assert 'Loc' in c[('VERB', 'Case')]


# ---------------------------------------------------------------------------
# apply_constraints_to_feats: removes impossible, never adds
# ---------------------------------------------------------------------------
def test_apply_keeps_legal_combos():
    sents = _sents([('NOUN', 'Case=Nom|Number=Sing')])
    c = build_constraints(sents)
    out = apply_constraints_to_feats(['Case=Nom', 'Number=Sing'], 'NOUN', c)
    assert set(out) == {'Case=Nom', 'Number=Sing'}


def test_apply_removes_impossible_value():
    # NOUN never seen with Tense -> dropped
    sents = _sents([('NOUN', 'Case=Nom')])
    c = build_constraints(sents)
    out = apply_constraints_to_feats(['Case=Nom', 'Tense=Past'], 'NOUN', c)
    assert out == ['Case=Nom']


def test_apply_removes_unseen_value_for_known_feature():
    # NOUN has Case={Nom} in train; Case=Dat was never seen -> dropped
    sents = _sents([('NOUN', 'Case=Nom')])
    c = build_constraints(sents)
    out = apply_constraints_to_feats(['Case=Dat'], 'NOUN', c)
    assert out == []  # Dat not in allowed set for NOUN+Case


def test_apply_drops_whole_feature_when_pair_unseen():
    # NOUN never seen with any Tense -> the whole Tense feature is masked
    sents = _sents([('NOUN', 'Case=Nom')])
    c = build_constraints(sents)
    out = apply_constraints_to_feats(['Tense=Past', 'Tense=Pres'], 'NOUN', c)
    assert out == []


def test_apply_never_adds_features():
    sents = _sents([('NOUN', 'Case=Nom')])
    c = build_constraints(sents)
    out = apply_constraints_to_feats([], 'NOUN', c)
    assert out == []  # cannot add what wasn't predicted


def test_apply_value_legal_for_one_upos_illegal_for_another():
    # Case=Dat legal for NOUN but (in this toy data) illegal for VERB
    sents = _sents([('NOUN', 'Case=Dat'), ('VERB', 'Tense=Past')])
    c = build_constraints(sents)
    assert apply_constraints_to_feats(['Case=Dat'], 'NOUN', c) == ['Case=Dat']
    assert apply_constraints_to_feats(['Case=Dat'], 'VERB', c) == []


# ---------------------------------------------------------------------------
# count_impossible: the before/after metric for the paper
# ---------------------------------------------------------------------------
def test_count_impossible_flags_wrong_combos():
    train = _sents([('NOUN', 'Case=Nom')])  # only NOUN+Case=Nom allowed
    c = build_constraints(train)
    preds = [
        {'upos': 'NOUN', 'feats': ['Case=Nom']},        # legal
        {'upos': 'NOUN', 'feats': ['Case=Dat']},        # impossible value
        {'upos': 'VERB', 'feats': ['Tense=Past']},      # whole pair unseen
        {'upos': 'NOUN', 'feats': []},                  # nothing predicted
    ]
    r = count_impossible(preds, c)
    assert r['total_predicted_feats'] == 3
    assert r['impossible_before'] == 2
    assert abs(r['impossible_rate_before'] - 2/3) < 1e-9


def test_count_impossible_handles_empty():
    c = build_constraints(_sents([('NOUN', 'Case=Nom')]))
    r = count_impossible([], c)
    assert r['total_predicted_feats'] == 0
    assert r['impossible_before'] == 0


# ---------------------------------------------------------------------------
# end-to-end on real KTB: constraints built from train should mark few gold
# test combos impossible (sanity -- gold should be mostly self-consistent)
# ---------------------------------------------------------------------------
def test_real_gold_test_has_low_impossibility_vs_train():
    from cagf.data import read_conllu
    train = read_conllu('data/gold_merged/gold_train.conllu')
    test = read_conllu('data/gold_merged/gold_test.conllu')
    c = build_constraints(train)
    rows = [{'upos': t.upos,
             'feats': t.feats.split('|') if t.feats != '_' else []}
            for s in test for t in s.tokens]
    r = count_impossible(rows, c)
    # gold-vs-train-constraints: a small fraction may be unseen (test has
    # rare combos), but it should be well under 5% -- if it were large, the
    # constraint method would be removing legitimate predictions.
    assert r['impossible_rate_before'] < 0.05, \
        f'gold test impossibility vs train = {r["impossible_rate_before"]:.3f}, expected < 0.05'
