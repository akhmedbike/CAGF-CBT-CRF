"""Support-Aware Active Learning (SAAL) offline annotation simulation.

Reimplements the acquisition loop of the conference paper
(docs/paper_conference.md, Section 3): a hashed character-n-gram surrogate
with three linear heads (UPOS, compact lemma-rewrite, weak-support risk),
the token- and sentence-level acquisition scores of Eq. (7)-(8) with
configurable weights, and the Random / Entropy / Novelty baselines of
Eq. (9)-(10).

The experiment code that produced the published Table 2-4 numbers is not
part of this repository; this module is a paper-faithful reimplementation
written for the weight-sensitivity ablation on branch siccis-malta.  Its
exact-transformation representation reuses cagf.data.form_to_edit_script,
which reproduces the Table 1 corpus fingerprint (1,078 sentences / 10,536
syntactic words / 8,440 non-PUNCT words / 4,369 unique forms / 1,475 exact
transformation classes / 4,618 transformed words) on UD_Kazakh-KTB.

Design choices not pinned down by the paper text are marked inline with
"reimplementation choice" and applied identically to every strategy, so
within-run comparisons stay internally consistent.
"""
from __future__ import annotations

import random
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression

from cagf.data import form_to_edit_script, read_conllu, restore_lemma_case

SEEDS = (7, 13, 19, 31, 42, 55, 77, 101, 123, 202)
IDENTITY_COMPACT = 'ID'
RARE_CLASS_THRESHOLD = 2  # z = 1 when the exact class occurs <= 2 times in L_k


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class SaalToken:
    form: str
    lemma: str
    upos: str


@dataclass
class SaalSentence:
    tokens: list[SaalToken] = field(default_factory=list)

    @property
    def n_words(self) -> int:
        return len(self.tokens)


def load_corpus(paths: list[str | Path]) -> list[SaalSentence]:
    """Read CoNLL-U files and keep only non-PUNCT analytic tokens.

    Multiword-token range rows and empty nodes are already excluded by
    cagf.data.read_conllu; punctuation is dropped here, matching the paper's
    "excluded from the token-level learning statistics".
    """
    sentences: list[SaalSentence] = []
    for path in paths:
        for sent in read_conllu(path):
            tokens = [SaalToken(t.form, t.lemma, t.upos) for t in sent.tokens if t.upos != 'PUNCT']
            if tokens:
                sentences.append(SaalSentence(tokens))
    return sentences


# ---------------------------------------------------------------------------
# Transformation representations
# ---------------------------------------------------------------------------

def _longest_common_substring(a: str, b: str) -> tuple[int, int, int]:
    """Leftmost longest common substring of ``a`` and ``b`` as (i, j, length)."""
    best = (0, 0, 0)
    for i in range(len(a)):
        for j in range(len(b)):
            k = 0
            while i + k < len(a) and j + k < len(b) and a[i + k] == b[j + k]:
                k += 1
            if k > best[2]:
                best = (i, j, k)
    return best


def compact_class(form: str, lemma: str) -> str:
    """Compact rewrite class: LCS-aligned edits at the left/right boundaries.

    Mirrors the paper's surrogate representation: align the lower-cased form
    and lemma around their longest shared substring and record the material
    removed/inserted on each side, pipe-joined.  An unchanged mapping gets
    the identity class.  663 classes on full UD_Kazakh-KTB vs 1,475 exact
    classes, matching the paper's "relatively small" surrogate output space.
    """
    f, l = form.lower(), lemma.lower()
    if f == l:
        return IDENTITY_COMPACT
    i, j, n = _longest_common_substring(f, l)
    return '|'.join((f[:i], l[:j], f[i + n:], l[j + n:]))


def apply_compact_class(form: str, cls: str) -> str:
    """Reconstruct a (lower-cased) lemma by applying a compact rewrite class.

    The anchor is assumed to occupy the middle of the form, i.e. the left
    edit removes ``len(left_del)`` leading characters.  When the class does
    not fit the form (an impossible combination at prediction time), the
    lower-cased form itself is returned.
    """
    if cls == IDENTITY_COMPACT:
        return form.lower()
    parts = cls.split('|')
    if len(parts) != 4:
        return form.lower()
    left_del, left_ins, right_del, right_ins = parts
    if len(left_del) + len(right_del) > len(form):
        return form.lower()
    return left_ins + form[len(left_del):len(form) - len(right_del)] + right_ins


def exact_class(form: str, lemma: str) -> str:
    """Exact transformation class (paper Section 3.C): P/S/D/I edit script."""
    return form_to_edit_script(form, lemma)


def is_transformed(token: SaalToken) -> bool:
    return token.form.lower() != token.lemma.lower()


# ---------------------------------------------------------------------------
# Features: 2,048-dim nonnegative hashed character n-grams (length 2-5)
# ---------------------------------------------------------------------------

class CharHashVectorizer:
    def __init__(self, dim: int = 2048, ngram_range: tuple[int, int] = (2, 5)):
        self.dim = dim
        self.ngram_range = ngram_range
        self._cache: dict[str, list[int]] = {}

    def _indices(self, form: str) -> list[int]:
        f = form.lower()
        lo, hi = self.ngram_range
        if len(f) < lo:
            return []
        return sorted({zlib.crc32(f[i:i + n].encode('utf-8')) % self.dim
                       for n in range(lo, hi + 1)
                       for i in range(len(f) - n + 1)})

    def transform(self, forms: list[str]) -> csr_matrix:
        rows: list[int] = []
        cols: list[int] = []
        for r, form in enumerate(forms):
            for c in self._cache.setdefault(form, self._indices(form)):
                rows.append(r)
                cols.append(c)
        data = np.ones(len(rows), dtype=np.float64)
        return csr_matrix((data, (rows, cols)), shape=(len(forms), self.dim))


# ---------------------------------------------------------------------------
# Surrogate heads
# ---------------------------------------------------------------------------

class Surrogate:
    """UPOS + compact-rewrite + weak-support-risk logistic heads trained on L_k.

    ``solver`` selects the head family: ``'multinomial'`` (softmax lbfgs) or
    ``'liblinear_ovr'`` (one-vs-rest liblinear — the older sklearn behaviour;
    diagnostics in docs/weight_sensitivity_kazakh.md show it reproduces the
    published Entropy baseline almost exactly, so it is the likely setting of
    the original pipeline).
    """

    def __init__(self, c: float = 1.0, max_iter: int = 500, solver: str = 'multinomial'):
        self.c = c
        self.max_iter = max_iter
        self.solver = solver
        self.upos_head: LogisticRegression | None = None
        self.rewrite_head: LogisticRegression | None = None
        self.risk_head: LogisticRegression | None = None
        self._degenerate_risk = 0.0

    def _make_lr(self, **kwargs):
        if self.solver == 'liblinear_ovr':
            from sklearn.multiclass import OneVsRestClassifier
            return OneVsRestClassifier(LogisticRegression(solver='liblinear', **kwargs))
        if self.solver != 'multinomial':
            raise ValueError(f'unknown solver: {self.solver}')
        return LogisticRegression(**kwargs)

    def fit(self, features: csr_matrix, upos_y: list[str], rewrite_y: list[str],
            risk_z: list[int]) -> 'Surrogate':
        self.upos_head = self._make_lr(C=self.c, max_iter=self.max_iter)
        self.upos_head.fit(features, upos_y)
        self.rewrite_head = self._make_lr(C=self.c, max_iter=self.max_iter)
        self.rewrite_head.fit(features, rewrite_y)
        # Class-balanced risk model (paper: "class-balanced logistic regression").
        unique_z = set(risk_z)
        if len(unique_z) < 2:
            # Reimplementation choice: with a single class present in L_k the
            # balanced prior is that constant (no discrimination possible).
            self.risk_head = None
            self._degenerate_risk = 1.0 if unique_z == {1} else 0.0
        else:
            self.risk_head = self._make_lr(C=self.c, max_iter=self.max_iter, class_weight='balanced')
            self.risk_head.fit(features, risk_z)
        return self

    def predict_all(self, features: csr_matrix) -> dict[str, np.ndarray]:
        """Per-token predictive quantities for the candidate pool.

        Returns normalized entropies H_u, H_r (Eq. 1-2), the non-identity
        rewrite probability p_T (Eq. 6), the weak-support risk q (Eq. 5),
        and the arg-max UPOS / rewrite predictions used at evaluation time.
        """
        p_upos = self.upos_head.predict_proba(features)
        p_re = self.rewrite_head.predict_proba(features)
        h_u = _normalized_entropy(p_upos)
        h_r = _normalized_entropy(p_re)
        classes_re = self.rewrite_head.classes_
        id_col = np.where(classes_re == IDENTITY_COMPACT)[0]
        if id_col.size:
            p_t = 1.0 - p_re[:, id_col[0]]
        else:
            p_t = np.ones(p_re.shape[0])
        if self.risk_head is not None:
            q = self.risk_head.predict_proba(features)[:, 1]
        else:
            q = np.full(features.shape[0], self._degenerate_risk)
        return {
            'H_u': h_u, 'H_r': h_r, 'p_T': p_t, 'q': q,
            'upos_pred': self.upos_head.classes_[p_upos.argmax(axis=1)],
            'rewrite_pred': classes_re[p_re.argmax(axis=1)],
        }


def _normalized_entropy(probs: np.ndarray) -> np.ndarray:
    """Eq. (1)-(2): entropy divided by log of the class count; K=1 -> 0."""
    k = probs.shape[1]
    if k < 2:
        return np.zeros(probs.shape[0])
    plogp = np.where(probs > 0, probs * np.log(np.maximum(probs, 1e-300)), 0.0)
    return -plogp.sum(axis=1) / np.log(k)


# ---------------------------------------------------------------------------
# Labeled-pool statistics: novelty (Eq. 3-4) and z-labels for the risk model
# ---------------------------------------------------------------------------

class LabeledPoolStats:
    def __init__(self, labeled: list[SaalSentence]):
        self.forms: set[str] = set()
        self.suffixes: dict[int, set[str]] = {2: set(), 3: set(), 4: set()}
        self.exact_counts: dict[str, int] = {}
        self.exact_types: set[str] = set()
        for sent in labeled:
            for t in sent.tokens:
                f = t.form.lower()
                self.forms.add(f)
                for m in (2, 3, 4):
                    if len(f) >= m:
                        self.suffixes[m].add(f[-m:])
                ec = exact_class(t.form, t.lemma)
                self.exact_counts[ec] = self.exact_counts.get(ec, 0) + 1
                self.exact_types.add(ec)

    def n_lex(self, form: str) -> float:
        """Eq. (3): 1 when the case-folded form is unseen in L_k."""
        return 0.0 if form.lower() in self.forms else 1.0

    def n_suf(self, form: str) -> float:
        """Eq. (4): proportion of unseen 2/3/4-character endings."""
        f = form.lower()
        return float(np.mean([1.0 if (len(f) < m or f[-m:] not in self.suffixes[m]) else 0.0
                              for m in (2, 3, 4)]))

    def z_label(self, token: SaalToken) -> int:
        """z = 1 when the token is transformed and its exact class is rare in L_k."""
        if not is_transformed(token):
            return 0
        return 1 if self.exact_counts.get(exact_class(token.form, token.lemma), 0) <= RARE_CLASS_THRESHOLD else 0


# ---------------------------------------------------------------------------
# Acquisition strategies (Eq. 7-10)
# ---------------------------------------------------------------------------

@dataclass
class AcquisitionWeights:
    """Token-level weights of Eq. (7) and sentence-level weights of Eq. (8)."""
    hu: float = 0.25
    hr: float = 0.20
    qpT: float = 0.30
    nsuf: float = 0.15
    nlex: float = 0.10
    sentence_mean: float = 0.8
    sentence_max: float = 0.2

    def describe(self) -> str:
        return (f'Hu={self.hu:.2f}/Hr={self.hr:.2f}/qpT={self.qpT:.2f}/'
                f'Nsuf={self.nsuf:.2f}/Nlex={self.nlex:.2f}; '
                f'mean/max={self.sentence_mean:.1f}/{self.sentence_max:.1f}')


def token_scores_support_aware(pred: dict[str, np.ndarray], n_suf: np.ndarray, n_lex: np.ndarray,
                               w: AcquisitionWeights) -> np.ndarray:
    """Eq. (7): weighted combination of the five token-level signals."""
    return (w.hu * pred['H_u'] + w.hr * pred['H_r'] + w.qpT * pred['q'] * pred['p_T']
            + w.nsuf * n_suf + w.nlex * n_lex)


def token_scores_entropy(pred: dict[str, np.ndarray]) -> np.ndarray:
    """Eq. (9): Entropy baseline."""
    return 0.55 * pred['H_u'] + 0.45 * pred['H_r']


def token_scores_novelty(n_suf: np.ndarray, n_lex: np.ndarray) -> np.ndarray:
    """Eq. (10): Novelty baseline."""
    return 0.55 * n_suf + 0.45 * n_lex


def sentence_scores(per_sentence: list[np.ndarray], w: AcquisitionWeights) -> np.ndarray:
    """Eq. (8): combine per-sentence mean and max token scores."""
    means = np.array([s.mean() if len(s) else 0.0 for s in per_sentence])
    maxs = np.array([s.max() if len(s) else 0.0 for s in per_sentence])
    return w.sentence_mean * means + w.sentence_max * maxs


# ---------------------------------------------------------------------------
# Offline annotation simulation
# ---------------------------------------------------------------------------

@dataclass
class SplitPlan:
    seed: int
    pool_idx: list[int]
    eval_idx: list[int]
    initial_idx: list[int]


def build_split(sentences: list[SaalSentence], seed: int, pool_frac: float = 0.8,
                initial_frac: float = 0.05) -> SplitPlan:
    """80/20 sentence-level split + the shared initial labeled subset.

    The initial subset holds ~``initial_frac`` of the pool's non-PUNCT words
    and is shared by all strategies within a seed (paper: "All strategies use
    the same initial labeled subset within a split").
    """
    rng = random.Random(seed)
    order = list(range(len(sentences)))
    rng.shuffle(order)
    n_eval = round(len(sentences) * (1.0 - pool_frac))
    eval_idx = sorted(order[:n_eval])
    pool_idx = sorted(order[n_eval:])
    pool_words = sum(sentences[i].n_words for i in pool_idx)
    target = round(pool_words * initial_frac)
    candidates = list(pool_idx)
    random.Random(f'{seed}:init').shuffle(candidates)
    initial_idx: list[int] = []
    taken = 0
    for idx in candidates:
        if taken >= target:
            break
        initial_idx.append(idx)
        taken += sentences[idx].n_words
    return SplitPlan(seed, pool_idx, eval_idx, sorted(initial_idx))


def _fit_surrogate(labeled_sents: list[SaalSentence], vectorizer: CharHashVectorizer,
                   surrogate_kwargs: dict | None = None) -> Surrogate:
    lab_tokens = [t for s in labeled_sents for t in s.tokens]
    x_lab = vectorizer.transform([t.form for t in lab_tokens])
    stats = LabeledPoolStats(labeled_sents)
    return Surrogate(**(surrogate_kwargs or {})).fit(
        x_lab,
        [t.upos for t in lab_tokens],
        [compact_class(t.form, t.lemma) for t in lab_tokens],
        [stats.z_label(t) for t in lab_tokens],
    )


def evaluate(surrogate: Surrogate, eval_features: csr_matrix, eval_sentences: list[SaalSentence],
             pool_stats: LabeledPoolStats) -> dict[str, float]:
    """Exact lemma accuracy, UPOS accuracy and transformed TSC on the held-out set.

    Reimplementation choice: the reconstructed lemma (lower-cased compact
    rewrite applied to the form) gets UPOS-aware surface-case restoration via
    cagf.data.restore_lemma_case and is then compared byte-exactly with the
    gold lemma, following this repository's established CoNLL-style scoring.
    """
    pred = surrogate.predict_all(eval_features)
    lemma_ok = 0
    upos_ok = 0
    tsc_ok = 0
    n_tokens = 0
    n_transformed = 0
    row = 0
    for sent in eval_sentences:
        for t in sent.tokens:
            upos = pred['upos_pred'][row]
            lemma = restore_lemma_case(t.form, apply_compact_class(t.form, pred['rewrite_pred'][row]), upos)
            if lemma == t.lemma:
                lemma_ok += 1
            if upos == t.upos:
                upos_ok += 1
            if is_transformed(t):
                n_transformed += 1
                if exact_class(t.form, t.lemma) in pool_stats.exact_types:
                    tsc_ok += 1
            n_tokens += 1
            row += 1
    return {
        'lemma_acc': 100.0 * lemma_ok / n_tokens,
        'upos_acc': 100.0 * upos_ok / n_tokens,
        'tsc': 100.0 * tsc_ok / max(n_transformed, 1),
    }


def run_fullpool(sentences: list[SaalSentence], split: SplitPlan, vectorizer: CharHashVectorizer,
                 surrogate_kwargs: dict | None = None) -> dict[str, float]:
    """Train the surrogate on the entire acquisition pool and evaluate on the
    held-out set: the full-pool reference used for the b95 efficiency measure.

    Depends only on (language, seed) — the pool is identical for every
    strategy within a split — so it is run once per seed.
    """
    pool_sents = [sentences[i] for i in split.pool_idx]
    stats = LabeledPoolStats(pool_sents)
    surrogate = _fit_surrogate(pool_sents, vectorizer, surrogate_kwargs)
    eval_sents = [sentences[i] for i in split.eval_idx]
    eval_tokens = [t for s in eval_sents for t in s.tokens]
    x_eval = vectorizer.transform([t.form for t in eval_tokens])
    return evaluate(surrogate, x_eval, eval_sents, stats)


def run_simulation(sentences: list[SaalSentence], split: SplitPlan, strategy: str,
                   budgets: list[float], weights: AcquisitionWeights,
                   vectorizer: CharHashVectorizer,
                   surrogate_kwargs: dict | None = None) -> dict[str, dict[str, float]]:
    """One acquisition loop: seed set -> acquire to each budget -> evaluate.

    At every checkpoint the surrogate is retrained on the current L_k and the
    held-out set is evaluated; a batch of complete sentences is then added
    until the next budget word target is reached (reimplementation choice:
    the last sentence of a batch may overshoot the target by at most its own
    length, "added until the target budget is reached").  Returns
    {'initial': metrics, budget_fraction: metrics, ...}.
    """
    labeled = list(split.initial_idx)
    in_labeled = set(labeled)
    remaining = [i for i in split.pool_idx if i not in in_labeled]
    pool_words = sum(sentences[i].n_words for i in split.pool_idx)
    results: dict[str, dict[str, float]] = {}
    round_no = 0

    def train_and_eval(checkpoint: str) -> tuple[Surrogate, LabeledPoolStats]:
        labeled_sents = [sentences[i] for i in labeled]
        stats = LabeledPoolStats(labeled_sents)
        surrogate = _fit_surrogate(labeled_sents, vectorizer, surrogate_kwargs)
        eval_sents = [sentences[i] for i in split.eval_idx]
        eval_tokens = [t for s in eval_sents for t in s.tokens]
        x_eval = vectorizer.transform([t.form for t in eval_tokens])
        results[checkpoint] = evaluate(surrogate, x_eval, eval_sents, stats)
        return surrogate, stats

    surrogate, stats = train_and_eval('initial')
    for frac in budgets:
        target = round(pool_words * frac)
        if sum(sentences[i].n_words for i in labeled) >= target:
            continue
        cand_idx = list(remaining)
        cand_sents = [sentences[i] for i in cand_idx]
        cand_tokens = [t for s in cand_sents for t in s.tokens]
        if strategy == 'random':
            order = list(range(len(cand_idx)))
            random.Random(f'{split.seed}:{strategy}:{round_no}').shuffle(order)
            ranked = [cand_idx[i] for i in order]
        else:
            x_cand = vectorizer.transform([t.form for t in cand_tokens])
            pred = surrogate.predict_all(x_cand)
            n_suf = np.array([stats.n_suf(t.form) for t in cand_tokens])
            n_lex = np.array([stats.n_lex(t.form) for t in cand_tokens])
            if strategy == 'support_aware':
                tok = token_scores_support_aware(pred, n_suf, n_lex, weights)
            elif strategy == 'entropy':
                tok = token_scores_entropy(pred)
            elif strategy == 'novelty':
                tok = token_scores_novelty(n_suf, n_lex)
            else:
                raise ValueError(f'unknown strategy: {strategy}')
            offset = 0
            per_sentence: list[np.ndarray] = []
            for s in cand_sents:
                per_sentence.append(tok[offset:offset + s.n_words])
                offset += s.n_words
            sent_score = sentence_scores(per_sentence, weights)
            ranked = [cand_idx[i] for i in np.argsort(-sent_score, kind='stable')]
        taken = sum(sentences[i].n_words for i in labeled)
        newly: list[int] = []
        for idx in ranked:
            if taken >= target:
                break
            newly.append(idx)
            taken += sentences[idx].n_words
        newly_set = set(newly)
        labeled.extend(newly)
        remaining = [i for i in remaining if i not in newly_set]
        round_no += 1
        surrogate, stats = train_and_eval(f'{frac:g}')
    return results
