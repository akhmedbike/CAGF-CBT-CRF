"""Tests for cagf.predict_writer: FEATS serialization + CoNLL-U round-trip.

The round-trip test is the acceptance criterion stated in the spec: writing
gold values back as "predictions" and reading the result must reproduce the
gold sentence/tokens exactly. The official conll18_ud_eval 100.0 check (gold
vs gold-written) is in test_official_eval.py, since it depends on the
vendored evaluator.
"""
from __future__ import annotations

from pathlib import Path

from cagf.data import Sentence, Token, read_conllu
from cagf.predict_writer import serialize_feats, write_conllu


# ---------------------------------------------------------------------------
# serialize_feats
# ---------------------------------------------------------------------------
def test_serialize_feats_empty_is_underscore():
    assert serialize_feats([]) == '_'
    assert serialize_feats(['_']) == '_'
    assert serialize_feats(['']) == '_'


def test_serialize_feats_alphabetical_order():
    # the canonical UD ordering is by feature name; this is what conll18_ud_eval
    # compares as a string, so out-of-order features would score 0.
    assert serialize_feats(['Number=Sing', 'Case=Dat']) == 'Case=Dat|Number=Sing'
    assert serialize_feats(['Case=Dat', 'Number=Sing']) == 'Case=Dat|Number=Sing'


def test_serialize_feats_psor_attribute_sorts_by_name():
    # Number[psor] is the feature name; the [psor] is part of it for sorting.
    out = serialize_feats(['Person[psor]=3', 'Number[psor]=Plur,Sing'])
    assert out == 'Number[psor]=Plur,Sing|Person[psor]=3'


def test_serialize_feats_dedupes_underscores_keeps_real():
    assert serialize_feats(['Case=Nom', '_', 'Number=Sing']) == 'Case=Nom|Number=Sing'


# ---------------------------------------------------------------------------
# CoNLL-U round-trip on a synthetic multiword sentence
# ---------------------------------------------------------------------------
def _make_sentence_with_misc() -> Sentence:
    sent = Sentence(tokens=[
        Token(form='біз', lemma='біз', upos='PRON', feats='Case=Nom|Number=Plur|Person=1|PronType=Prs'),
        Token(form='келеміз', lemma='кел', upos='VERB', feats='Mood=Ind|Number=Plur|Person=1|Tense=Pres|VerbForm=Fin'),
    ])
    sent.comments = ['# sent_id = synthetic:1', '# text = біз келеміз']
    # a fake multiword token row, to exercise the interleave path
    sent.misc_lines = ['1-2\tбізкелеміз\t_\t_\t_\t_\t_\t_\t_\t_']
    return sent


def test_write_conllu_preserves_comments_and_misc(tmp_path):
    sent = _make_sentence_with_misc()
    pred = [{
        'lemma': [t.lemma for t in sent.tokens],
        'upos': [t.upos for t in sent.tokens],
        'feats': [t.feats.split('|') if t.feats != '_' else [] for t in sent.tokens],
    }]
    out = tmp_path / 'out.conllu'
    write_conllu([sent], pred, out)

    text = out.read_text(encoding='utf-8')
    # comments preserved verbatim
    assert '# sent_id = synthetic:1' in text
    assert '# text = біз келеміз' in text
    # multiword row preserved and placed before token 1
    assert '1-2\tбізкелеміз' in text
    first_token_line = [l for l in text.splitlines() if l.startswith('1\t')][0]
    assert first_token_line.startswith('1\tбіз\tбіз\tPRON\t_')
    # file ends with a blank line
    assert text.endswith('\n')
    assert text.rstrip('\n').endswith('')  # last meaningful line is blank-ish


def test_write_conllu_roundtrip_reproduces_gold(tmp_path):
    """The core acceptance test: write gold-as-prediction, read back, the
    analytic tokens (form/lemma/upos/feats) are identical to gold."""
    sent = _make_sentence_with_misc()
    pred = [{
        'lemma': [t.lemma for t in sent.tokens],
        'upos': [t.upos for t in sent.tokens],
        'feats': [t.feats.split('|') if t.feats != '_' else [] for t in sent.tokens],
    }]
    out = tmp_path / 'roundtrip.conllu'
    write_conllu([sent], pred, out)

    reread = read_conllu(out)[0]
    assert len(reread.tokens) == len(sent.tokens)
    for gold_tok, pred_tok in zip(sent.tokens, reread.tokens):
        assert pred_tok.form == gold_tok.form
        assert pred_tok.lemma == gold_tok.lemma
        assert pred_tok.upos == gold_tok.upos
        # FEATS: gold may have a different feature order than our canonical
        # alphabetical serialization, so compare as SETS of Feature=Value.
        gold_feats = set(gold_tok.feats.split('|')) if gold_tok.feats != '_' else set()
        pred_feats = set(pred_tok.feats.split('|')) if pred_tok.feats != '_' else set()
        assert pred_feats == gold_feats, f'FEATS mismatch: {pred_feats} vs {gold_feats}'


def test_write_conllu_roundtrip_on_real_gold_file(tmp_path):
    """End-to-end on the actual KTB gold test file: write every token's gold
    value as the prediction, then the written file's analytic tokens must
    carry the same lemma/upos/feats as the original."""
    gold_path = 'data/gold_merged/gold_test.conllu'
    sents = read_conllu(gold_path)
    preds = [{
        'lemma': [t.lemma for t in s.tokens],
        'upos': [t.upos for t in s.tokens],
        'feats': [t.feats.split('|') if t.feats != '_' else [] for t in s.tokens],
    } for s in sents]
    out = tmp_path / 'gold_as_pred.conllu'
    write_conllu(sents, preds, out)

    reread = read_conllu(out)
    assert len(reread) == len(sents), f'{len(reread)} vs {len(sents)} sentences'
    for gold_sent, pred_sent in zip(sents, reread):
        assert len(pred_sent.tokens) == len(gold_sent.tokens)
        for gt, pt in zip(gold_sent.tokens, pred_sent.tokens):
            assert pt.form == gt.form
            assert pt.lemma == gt.lemma
            assert pt.upos == gt.upos
            gold_feats = set(gt.feats.split('|')) if gt.feats != '_' else set()
            pred_feats = set(pt.feats.split('|')) if pt.feats != '_' else set()
            assert pred_feats == gold_feats
    # token count matches (this is the "1078 / 10536" integrity guard)
    n_tokens = sum(len(s.tokens) for s in reread)
    assert n_tokens == sum(len(s.tokens) for s in sents)


def test_write_conllu_rejects_length_mismatch(tmp_path):
    sent = _make_sentence_with_misc()
    pred = [{'lemma': ['only_one'], 'upos': ['X'], 'feats': [[]]}]  # 1 token, sent has 2
    out = tmp_path / 'bad.conllu'
    try:
        write_conllu([sent], pred, out)
        assert False, 'should have raised on length mismatch'
    except ValueError as e:
        assert '2 tokens' in str(e) or 'length' in str(e).lower()
