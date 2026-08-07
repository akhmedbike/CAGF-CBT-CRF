"""Tests for the KazNLP silver-corpus pipeline and gold fine-tuning path."""
from __future__ import annotations

import torch

from cagf.kaznlp_ud_map import GOLD_GRAMMEMES, POS_MAP, map_analysis, parse_analysis
from cagf.data import parse_feats
from cagf.silver_filter import FilterThresholds, TokenDiagnostics, assess_sentence, parse_misc_diagnostics


def test_parse_analysis_basic():
    lemma, pos, morphs = parse_analysis("алма_R_ZE сы_S3 н_C4")
    assert lemma == "алма"
    assert pos == "R_ZE"
    assert morphs == ["S3", "C4"]


def test_mapping_examples_from_kaznlp_docs():
    # examples taken verbatim from the KazNLP README morphology demo
    lemma, upos, feats, _ = map_analysis("алмасын", "алма_R_ZE сы_S3 н_C4")
    assert lemma == "алма" and upos == "NOUN"
    assert set(feats.split("|")) == {"Case=Acc", "Number=Sing", "Person[psor]=3"}

    lemma, upos, feats, _ = map_analysis("етсең", "ет_R_ET се_M4 ң_P2")
    assert upos == "VERB"
    assert "Mood=Cnd" in feats and "Person=2" in feats and "VerbForm=Fin" in feats

    _, upos, feats, _ = map_analysis("ерінбей", "ерін_R_ET бе_ET_ETB й_ETB_KSE")
    assert upos == "VERB" and "Polarity=Neg" in feats and "VerbForm=Conv" in feats


def test_unanalyzed_maps_to_X():
    _, upos, feats, cov = map_analysis("лмасын", "лмасын_R_X")
    assert upos == "X" and feats == "_"


def test_all_produced_feats_are_within_gold_inventory():
    # exhaustively map a spread of analyses and assert no FEATS value leaks
    samples = [
        "бала_R_ZE лар_N1 ды_C4",
        "кел_R_ET ді_T3 м_P1",
        "үлкен_R_SE",
        "ол_R_SIM",
        "үш_R_SN",
        "және_R_ZHL",
        "мен_R_SH",
        "._R_NKT",
    ]
    for a in samples:
        _, _, feats, _ = map_analysis(a.split("_")[0], a)
        for f in parse_feats(feats):
            assert f in GOLD_GRAMMEMES, f"leaked non-gold FEAT: {f}"


def test_mapping_is_deterministic():
    a = "алма_R_ZE сы_S3 н_C4"
    r1 = map_analysis("алмасын", a)
    r2 = map_analysis("алмасын", a)
    assert r1[:3] == r2[:3]


def test_every_pos_map_target_is_a_valid_upos():
    valid = {"NOUN", "PROPN", "VERB", "AUX", "ADJ", "PRON", "ADV", "CCONJ",
             "SCONJ", "NUM", "ADP", "PART", "INTJ", "SYM", "X", "PUNCT", "DET"}
    for tag, upos in POS_MAP.items():
        assert upos in valid, f"{tag} maps to invalid UPOS {upos}"


def test_parse_misc_diagnostics_reads_builder_fields():
    d = parse_misc_diagnostics("Silver=kaznlp|Map=1.0.0|UnkPOS=1|Dropped=2|Unmapped=3")
    assert d == TokenDiagnostics(unknown_pos=1, dropped_non_gold=2, unmapped_morpheme=3)


def test_parse_misc_diagnostics_defaults_to_clean_when_absent():
    # gold data, or MISC from an older builder version, has no diagnostic fields
    assert parse_misc_diagnostics("_") == TokenDiagnostics(0, 0, 0)
    assert parse_misc_diagnostics("SpaceAfter=No") == TokenDiagnostics(0, 0, 0)


def test_clean_sentence_is_kept():
    clean = [TokenDiagnostics(0, 0, 0)] * 8
    result = assess_sentence(clean)
    assert result.kept
    assert result.reasons == ()


def test_mostly_unanalysed_sentence_is_rejected_as_incomplete():
    # 4/5 tokens unanalysed (UnkPOS=1) far exceeds the default 0.3 ratio
    diags = [TokenDiagnostics(1, 0, 0)] * 4 + [TokenDiagnostics(0, 0, 0)]
    result = assess_sentence(diags)
    assert not result.kept
    assert "incomplete_annotation" in result.reasons


def test_high_noise_sentence_is_rejected_as_inconsistent():
    # heavy dropped/unmapped morpheme rate, but every token still got a POS
    diags = [TokenDiagnostics(0, 3, 2)] * 5
    result = assess_sentence(diags)
    assert not result.kept
    assert "inconsistent_annotation" in result.reasons


def test_too_short_and_too_long_sentences_are_rejected():
    thresholds = FilterThresholds(min_tokens=2, max_tokens=3)
    too_short = assess_sentence([TokenDiagnostics(0, 0, 0)], thresholds)
    assert not too_short.kept and "too_short" in too_short.reasons

    too_long = assess_sentence([TokenDiagnostics(0, 0, 0)] * 4, thresholds)
    assert not too_long.kept and "too_long" in too_long.reasons

    just_right = assess_sentence([TokenDiagnostics(0, 0, 0)] * 3, thresholds)
    assert just_right.kept


def test_custom_thresholds_are_respected():
    # 1 in 5 tokens has one dropped-morpheme event -> noise_ratio = 0.2
    diags = [TokenDiagnostics(0, 1, 0)] + [TokenDiagnostics(0, 0, 0)] * 4
    assert not assess_sentence(diags, FilterThresholds(max_noise_ratio=0.1)).kept
    assert assess_sentence(diags, FilterThresholds(max_noise_ratio=0.3)).kept


def test_finetune_init_state_loads_matching_tensors():
    # a small end-to-end check that pretrained weights transfer into a fresh
    # model via train_one_run(init_state=...), which is what fine-tuning uses.
    from cagf.data import build_vocabs, read_conllu
    from cagf.model import AblationConfig, CAGFCBTCRF, ModelHParams

    sents = read_conllu("data/toy_train.conllu")
    vocabs = build_vocabs(sents, min_word_freq=1)
    hp = ModelHParams()
    ab = AblationConfig()
    kw = dict(num_chars=len(vocabs.char_vocab), num_words=len(vocabs.word_vocab),
              num_upos=len(vocabs.upos_vocab), num_grammemes=len(vocabs.grammeme_vocab),
              num_lemma_rules=len(vocabs.lemma_rule_vocab), hparams=hp, ablation=ab,
              char_pad_id=vocabs.char_vocab.stoi["<pad>"], word_pad_id=vocabs.word_vocab.stoi["<pad>"])
    src = CAGFCBTCRF(**kw)
    dst = CAGFCBTCRF(**kw)
    src_state = src.state_dict()
    own = dst.state_dict()
    compatible = {k: v for k, v in src_state.items() if k in own and own[k].shape == v.shape}
    assert len(compatible) == len(own)  # identical architecture -> everything transfers
    own.update(compatible)
    dst.load_state_dict(own)
    for k in own:
        assert torch.equal(dst.state_dict()[k], src_state[k])
