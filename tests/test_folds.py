"""Tests for cagf.folds: partition, determinism, no-leak invariants.

These use synthetic Sentence lists (not the real KTB) so they stay fast and
self-contained; the real-corpus smoke test lives in the CV runner.
"""
from __future__ import annotations

import pytest

from cagf.data import Sentence, Token
from cagf.folds import Fold, make_folds, sent_id_of, source_of


def _make_sentences(n: int, sources: list[str]) -> list[Sentence]:
    """Build n fake sentences cycling through the given source names. Each
    sentence's comment carries a KTB-shaped sent_id so source_of can parse it."""
    out: list[Sentence] = []
    for i in range(n):
        src = sources[i % len(sources)]
        sent = Sentence(tokens=[Token(form=f'w{i}', lemma=f'w{i}', upos='NOUN', feats='_')])
        sent.comments = [f'# sent_id = {src}.tagged.txt:{i}:{i*10}']
        out.append(sent)
    return out


# ---------------------------------------------------------------------------
# source_of / sent_id_of parsing
# ---------------------------------------------------------------------------
def test_source_of_ktb_format():
    assert source_of('akorda-random.tagged.txt:59:1025') == 'akorda-random.tagged.txt'
    assert source_of('wikitravel.tagged.txt:34:362') == 'wikitravel.tagged.txt'


def test_source_of_plain_id_no_colon():
    # sent_id without a colon: whole string is the source key
    assert source_of('plain_id') == 'plain_id'


def test_sent_id_of_reads_from_comments():
    s = Sentence(tokens=[Token(form='x', lemma='x', upos='NOUN', feats='_')])
    s.comments = ['# sent_id = wikipedia.tagged.txt:1:10', '# text = ...']
    assert sent_id_of(s) == 'wikipedia.tagged.txt:1:10'


def test_sent_id_of_empty_when_no_comment():
    s = Sentence(tokens=[Token(form='x', lemma='x', upos='NOUN', feats='_')])
    assert sent_id_of(s) == ''


# ---------------------------------------------------------------------------
# partition invariants (both strategies)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('strategy', ['stratified', 'grouped'])
def test_test_sets_partition_whole_corpus(strategy):
    # 10 sources so grouped k=10 has one source per fold.
    sents = _make_sentences(100, [f'src{i}' for i in range(10)])
    folds = make_folds(sents, k=10, seed=42, strategy=strategy)
    all_test = sorted(idx for f in folds for idx in f.test_idx)
    assert all_test == list(range(100)), 'test sets must cover every sentence exactly once'


@pytest.mark.parametrize('strategy', ['stratified', 'grouped'])
def test_no_overlap_within_fold(strategy):
    sents = _make_sentences(100, [f'src{i}' for i in range(10)])
    folds = make_folds(sents, k=10, seed=42, strategy=strategy)
    for f in folds:
        tr, dv, te = set(f.train_idx), set(f.dev_idx), set(f.test_idx)
        assert not (tr & dv), f'fold {f.index} train/dev overlap'
        assert not (tr & te), f'fold {f.index} train/test overlap'
        assert not (dv & te), f'fold {f.index} dev/test overlap'
        assert te, f'fold {f.index} empty test'
        assert tr, f'fold {f.index} empty train'


@pytest.mark.parametrize('strategy', ['stratified', 'grouped'])
def test_reproducible_for_fixed_seed(strategy):
    sents = _make_sentences(80, [f'src{i}' for i in range(8)])
    f1 = make_folds(sents, k=8, seed=42, strategy=strategy)
    f2 = make_folds(sents, k=8, seed=42, strategy=strategy)
    assert f1 == f2, 'same seed must give identical folds'


def test_stratified_different_seeds_give_different_folds():
    # Stratified mode shuffles within each source, so seed changes the split.
    sents = _make_sentences(80, [f'src{i}' for i in range(8)])
    f1 = make_folds(sents, k=8, seed=42, strategy='stratified')
    f2 = make_folds(sents, k=8, seed=7, strategy='stratified')
    assert [f.test_idx for f in f1] != [f.test_idx for f in f2]


def test_grouped_seed_does_not_change_source_assignment():
    # Grouped mode assigns each whole source to one fold deterministically
    # (no within-source shuffle), so the seed must NOT change which sentences
    # land in which fold. This is a feature: grouped CV answers "what if no
    # source is split", and that question has one canonical answer per (k, corpus).
    sents = _make_sentences(80, [f'src{i}' for i in range(8)])
    f1 = make_folds(sents, k=8, seed=42, strategy='grouped')
    f2 = make_folds(sents, k=8, seed=7, strategy='grouped')
    assert [f.test_idx for f in f1] == [f.test_idx for f in f2]


# ---------------------------------------------------------------------------
# grouped-specific: no source split across folds
# ---------------------------------------------------------------------------
def test_grouped_no_source_split_across_folds():
    sents = _make_sentences(100, ['a', 'b', 'c', 'd', 'e'])
    folds = make_folds(sents, k=5, seed=42, strategy='grouped')
    fold_sources = [set(source_of(sent_id_of(sents[i])) for i in f.test_idx) for f in folds]
    for a in range(len(fold_sources)):
        for b in range(a + 1, len(fold_sources)):
            assert not (fold_sources[a] & fold_sources[b]), \
                f'source appears in folds {a} and {b}: leakage-by-duplication risk'


# ---------------------------------------------------------------------------
# stratified-specific: source distribution tracks the corpus
# ---------------------------------------------------------------------------
def test_stratified_source_distribution_close_to_corpus():
    # Use enough sentences per source that per-fold counts are not dominated
    # by discretisation noise. 8 sources, ~1000 sentences total.
    sents = _make_sentences(1000, ['major', 'major', 'major', 'mid', 'mid', 'rare'])
    from collections import Counter
    overall = Counter(source_of(sent_id_of(s)) for s in sents)
    folds = make_folds(sents, k=10, seed=42, strategy='stratified')
    # Each source's share in each fold's test set should track its corpus share.
    # We allow a generous 15pp band: small sources (a few dozen per fold) have
    # inherent counting noise; the point is no source is wildly over/under.
    n = len(sents)
    for f in folds:
        fold_src = Counter(source_of(sent_id_of(sents[i])) for i in f.test_idx)
        n_fold = sum(fold_src.values())
        for src, corpus_count in overall.items():
            corpus_share = corpus_count / n
            fold_share = fold_src.get(src, 0) / n_fold
            assert abs(fold_share - corpus_share) < 0.15, \
                f'source {src}: fold share {fold_share:.2f} vs corpus {corpus_share:.2f} in fold {f.index}'


# ---------------------------------------------------------------------------
# input validation
# ---------------------------------------------------------------------------
def test_rejects_k_too_small():
    with pytest.raises(ValueError):
        make_folds(_make_sentences(100, ['a', 'b']), k=2)


def test_rejects_corpus_too_small():
    with pytest.raises(ValueError):
        make_folds(_make_sentences(5, ['a', 'b']), k=10)


def test_grouped_rejects_k_above_source_count():
    # Only 4 sources, k=5: grouped CV cannot fill 5 non-empty folds.
    sents = _make_sentences(40, ['a', 'b', 'c', 'd'])
    with pytest.raises(ValueError, match='grouped CV needs at least'):
        make_folds(sents, k=5, seed=42, strategy='grouped')
