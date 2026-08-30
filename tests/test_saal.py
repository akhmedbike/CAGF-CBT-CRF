"""Unit tests for the SAAL reimplementation (cagf.saal, branch siccis-malta)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cagf.saal import (AcquisitionWeights, CharHashVectorizer, LabeledPoolStats,
                       SaalSentence, SaalToken, SplitPlan, apply_compact_class,
                       build_split, compact_class, exact_class, load_corpus,
                       run_simulation, token_scores_novelty, token_scores_support_aware)


def sent(pairs: list[tuple[str, str, str]]) -> SaalSentence:
    return SaalSentence([SaalToken(f, l, u) for f, l, u in pairs])


def tiny_corpus() -> list[SaalSentence]:
    # Eight sentences of assorted Kazakh-like material; enough for the
    # simulation to train at least two heads per class.
    return [
        sent([('бала', 'бала', 'NOUN'), ('бар', 'бар', 'VERB')]),
        sent([('балалар', 'бала', 'NOUN'), ('келді', 'кел', 'VERB')]),
        sent([('баланың', 'бала', 'NOUN'), ('көрді', 'көр', 'VERB')]),
        sent([('қыз', 'қыз', 'NOUN'), ('оқиды', 'оқы', 'VERB')]),
        sent([('қыздар', 'қыз', 'NOUN'), ('ойнайды', 'ойна', 'VERB')]),
        sent([('мектеп', 'мектеп', 'NOUN'), ('ашты', 'аш', 'VERB')]),
        sent([('мектепте', 'мектеп', 'NOUN'), ('оқиды', 'оқы', 'VERB')]),
        sent([('жігіт', 'жігіт', 'NOUN'), ('жүрді', 'жүр', 'VERB')]),
    ]


# --- transformation representations -----------------------------------------

def test_compact_identity_and_roundtrip():
    assert compact_class('бала', 'бала') == 'ID'
    cls = compact_class('балалар', 'бала')
    assert cls != 'ID'
    assert apply_compact_class('балалар', cls) == 'бала'


def test_exact_class_matches_repo_edit_script():
    # exact_class must be the repo's P/S/D/I script: the representation whose
    # type count reproduces the paper's Table 1 fingerprint (1,475 on KTB).
    from cagf.data import form_to_edit_script
    assert exact_class('балалар', 'бала') == form_to_edit_script('балалар', 'бала')


# --- features ----------------------------------------------------------------

def test_char_hash_vectorizer_shape_and_cache():
    vec = CharHashVectorizer(dim=128, ngram_range=(2, 3))
    x = vec.transform(['абвг', 'абвг', 'а'])
    assert x.shape == (3, 128)
    # duplicate rows are identical; 1-char form yields an all-zero row
    assert (x[0] != x[1]).nnz == 0
    assert x[2].nnz == 0
    assert vec._cache['абвг'] is vec._cache['абвг']


# --- pool statistics -----------------------------------------------------------

def test_novelty_and_z_labels():
    pool = [sent([('бала', 'бала', 'NOUN'), ('баланың', 'бала', 'NOUN')])]
    stats = LabeledPoolStats(pool)
    assert stats.n_lex('бала') == 0.0
    assert stats.n_lex('балалар') == 1.0
    # none of the 2/3/4-char endings of 'балалар' occur in the pool
    assert stats.n_suf('балалар') == 1.0
    # 'баланың' itself shares endings with the pool -> all seen
    assert stats.n_suf('баланың') == 0.0
    # transformed token whose exact class occurs twice -> z = 1
    assert stats.z_label(SaalToken('баланың', 'бала', 'NOUN')) == 1
    assert stats.z_label(SaalToken('бала', 'бала', 'NOUN')) == 0


# --- acquisition scores --------------------------------------------------------

def test_support_aware_weights_match_paper_baseline():
    w = AcquisitionWeights()
    assert (w.hu, w.hr, w.qpT, w.nsuf, w.nlex) == (0.25, 0.20, 0.30, 0.15, 0.10)
    assert w.sentence_mean == 0.8 and w.sentence_max == 0.2


def test_token_scores_scale_with_weights():
    pred = {'H_u': np.array([1.0]), 'H_r': np.array([0.0]), 'p_T': np.array([0.5]),
            'q': np.array([1.0])}
    n_suf = np.array([0.0])
    n_lex = np.array([1.0])
    w = AcquisitionWeights()
    score = token_scores_support_aware(pred, n_suf, n_lex, w)[0]
    assert score == pytest.approx(0.25 * 1.0 + 0.30 * 0.5 + 0.10 * 1.0)
    assert token_scores_novelty(n_suf, n_lex)[0] == pytest.approx(0.45)


# --- simulation ----------------------------------------------------------------

def test_split_is_deterministic_and_shared():
    corpus = tiny_corpus()
    s1 = build_split(corpus, seed=13)
    s2 = build_split(corpus, seed=13)
    assert (s1.pool_idx, s1.eval_idx, s1.initial_idx) == (s2.pool_idx, s2.eval_idx, s2.initial_idx)
    assert set(s1.pool_idx) | set(s1.eval_idx) == set(range(len(corpus)))
    assert set(s1.initial_idx) <= set(s1.pool_idx)
    # eval split is the 20% held-out portion
    assert len(s1.eval_idx) == round(0.2 * len(corpus))


def test_simulation_budgets_and_metrics():
    corpus = tiny_corpus()
    split = build_split(corpus, seed=13, pool_frac=0.75, initial_frac=0.25)
    vec = CharHashVectorizer(dim=64, ngram_range=(2, 3))
    out = run_simulation(corpus, split, 'support_aware', [0.5], AcquisitionWeights(), vec)
    assert 'initial' in out and '0.5' in out
    for metrics in out.values():
        assert set(metrics) == {'lemma_acc', 'upos_acc', 'tsc'}
        assert 0.0 <= metrics['lemma_acc'] <= 100.0
        assert 0.0 <= metrics['upos_acc'] <= 100.0
        assert 0.0 <= metrics['tsc'] <= 100.0


def test_strategy_names_are_validated():
    corpus = tiny_corpus()
    split = SplitPlan(seed=1, pool_idx=[0, 1, 2], eval_idx=[3], initial_idx=[0])
    with pytest.raises(ValueError):
        run_simulation(corpus, split, 'unknown', [0.5], AcquisitionWeights(),
                       CharHashVectorizer(dim=64))


def test_surrogate_solver_variants_share_the_output_contract():
    from cagf.saal import Surrogate
    corpus = tiny_corpus()
    toks = [t for s in corpus for t in s.tokens]
    vec = CharHashVectorizer(dim=64, ngram_range=(2, 3))
    x = vec.transform([t.form for t in toks])
    stats = LabeledPoolStats(corpus)
    for solver in ('multinomial', 'liblinear_ovr'):
        surrogate = Surrogate(solver=solver).fit(
            x, [t.upos for t in toks],
            [compact_class(t.form, t.lemma) for t in toks],
            [stats.z_label(t) for t in toks])
        pred = surrogate.predict_all(x)
        for key in ('H_u', 'H_r', 'p_T', 'q'):
            assert pred[key].shape == (len(toks),)
            assert ((pred[key] >= 0) & (pred[key] <= 1)).all()
    with pytest.raises(ValueError):
        Surrogate(solver='nope')._make_lr()


def test_load_corpus_drops_punct_and_keeps_sentence_count():
    tmp = Path(__file__).parent / '_tmp_saal_load.conllu'
    tmp.write_text(
        '# sent_id = x:1\n# text = Бала келді.\n'
        '1\tБала\tбала\tNOUN\tn\t_\t2\tnsubj\t_\t_\n'
        '2\tкелді\tкел\tVERB\tv\t_\t0\troot\t_\t_\n'
        '3\t.\t.\tPUNCT\tsent\t_\t2\tpunct\t_\t_\n\n'
        '1-2\tсәлемде_\t_\t_\t_\t_\t_\t_\t_\t_\n'
        '1\tсәлемде\tсәлемде\tVERB\tv\t_\t0\troot\t_\t_\n\n',
        encoding='utf-8')
    try:
        corpus = load_corpus([tmp])
        assert len(corpus) == 2
        assert all(t.upos != 'PUNCT' for s in corpus for t in s.tokens)
        assert corpus[0].tokens[0].form == 'Бала'
    finally:
        tmp.unlink()
