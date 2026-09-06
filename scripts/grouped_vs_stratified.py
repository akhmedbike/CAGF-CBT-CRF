"""Task A analysis (revision plan A.6, R2.3/R3.3): source-cluster inference for
the grouped-vs-stratified comparison and for contrasts within the grouped run.

Both protocols produce pooled out-of-fold predictions over the same 1,078
sentences, so per-sentence correctness differences are paired. Following the
paper's primary procedure (§3.3), uncertainty is quantified by resampling the
19 source documents with replacement (10,000 samples) and by a paired
randomization test that swaps systems within source clusters, with Holm
correction within each family of comparisons.

Token-level correctness per metric mirrors the official scorer semantics:
Lemmas (exact match), UPOS (exact match), UFeats (universal-filtered bundle
exact match), AllTags (UPOS+XPOS+UFEATS bundle match).

CPU-only, reads stored predictions, trains nothing.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/grouped_vs_stratified.py
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from cagf.official_eval import conll18_ud_eval  # vendored, path-injected

UNIVERSAL = conll18_ud_eval.UNIVERSAL_FEATURES
METRICS = ('Lemmas', 'UPOS', 'UFeats', 'AllTags')
N_BOOT = 10_000
RNG = np.random.default_rng(42)

PAIRS = {
    # grouped-vs-stratified degradation, per condition
    'grouped_minus_strat/cagf_gold': ('results_cv_silver/grouped5/ktb_only',
                                      'results_cv_silver/stratified/ktb_only'),
    'grouped_minus_strat/cagf_silver': ('results_cv_silver/grouped5/silver_unfiltered_pretrain_ktb_finetune',
                                        'results_cv_silver/stratified/silver_unfiltered_pretrain_ktb_finetune'),
    'grouped_minus_strat/kazr_gold': ('results_cv_kazroberta/grouped5/gold_only',
                                       'results_cv_kazroberta/gold_only'),
    'grouped_minus_strat/kazr_silver': ('results_cv_kazroberta/grouped5/silver_finetune',
                                        'results_cv_kazroberta/silver_finetune'),
    # within-grouped contrasts (is the ordering preserved?)
    'grouped_silver_minus_gold/cagf': ('results_cv_silver/grouped5/silver_unfiltered_pretrain_ktb_finetune',
                                       'results_cv_silver/grouped5/ktb_only'),
    'grouped_silver_minus_gold/kazr': ('results_cv_kazroberta/grouped5/silver_finetune',
                                       'results_cv_kazroberta/grouped5/gold_only'),
    'grouped_cagf_minus_kazr/silver': ('results_cv_silver/grouped5/silver_unfiltered_pretrain_ktb_finetune',
                                       'results_cv_kazroberta/grouped5/silver_finetune'),
    'grouped_cagf_minus_kazr/gold': ('results_cv_silver/grouped5/ktb_only',
                                     'results_cv_kazroberta/grouped5/gold_only'),
}


def _load(path: Path):
    """Return {sent_id: {metric: [correctness per token]}} for a pred file,
    scored against its sibling gold_all.conllu."""
    pred, gold = path / 'pred_all.conllu', path / 'gold_all.conllu'

    def parse(p):
        sents, sid, rows = {}, None, []
        sid_re = re.compile(r'sent_id\s*=\s*(.+)')
        for line in Path(p).read_text(encoding='utf-8').splitlines():
            if line.startswith('#'):
                m = sid_re.search(line)
                if m:
                    sid = m.group(1).strip()
                continue
            if not line.strip():
                if sid is not None and rows:
                    sents[sid] = rows
                sid, rows = None, []
                continue
            if '\t' not in line:
                continue
            c = line.split('\t')
            if '-' in c[0] or '.' in c[0]:
                continue
            feats = '|'.join(sorted(f for f in c[5].split('|')
                                    if f != '_' and f.split('=', 1)[0] in UNIVERSAL)) or '_'
            rows.append((c[2], c[3], c[4], feats))
        if sid is not None and rows:
            sents[sid] = rows
        return sents

    G, P = parse(gold), parse(pred)
    assert set(G) == set(P), f'{path}: sentence id mismatch gold vs pred'
    out = {}
    for sid in G:
        assert len(G[sid]) == len(P[sid]), f'{path}/{sid}: token count mismatch'
        rec = {m: [] for m in METRICS}
        for (gl, gu, gx, gf), (pl, pu, px, pf) in zip(G[sid], P[sid]):
            # gold lemma '_' means "any" per the scorer; treat as correct
            rec['Lemmas'].append(float(gl == '_' or gl == pl))
            rec['UPOS'].append(float(gu == pu))
            rec['UFeats'].append(float(gf == pf))
            rec['AllTags'].append(float(gu == pu and gx == px and gf == pf))
        out[sid] = rec
    return out


def _source_of(sid: str) -> str:
    return sid.split(':')[0]


def cluster_bootstrap(diff_per_sent: np.ndarray, sources: np.ndarray,
                      n_boot: int = N_BOOT) -> tuple[float, float]:
    """Percentile CI of the token-weighted mean difference, resampling sources."""
    uniq = np.unique(sources)
    idx_by_src = {s: np.where(sources == s)[0] for s in uniq}
    weights = np.where(np.isnan(diff_per_sent), 0.0, 1.0)
    vals = np.nan_to_num(diff_per_sent)
    means = np.empty(n_boot)
    for b in range(n_boot):
        pick = RNG.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_src[s] for s in pick])
        w = weights[idx]
        means[b] = (vals[idx] * w).sum() / w.sum() if w.sum() else 0.0
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_randomization(diff_per_sent: np.ndarray, sources: np.ndarray,
                         n_perm: int = 10_000) -> float:
    """Two-sided p: swap the two systems within each source cluster."""
    uniq = np.unique(sources)
    idx_by_src = {s: np.where(sources == s)[0] for s in uniq}
    valid = ~np.isnan(diff_per_sent)
    vals = np.nan_to_num(diff_per_sent)
    weights = valid.astype(float)

    def wmean(d):
        return (d * weights).sum() / weights.sum()

    obs = abs(wmean(vals))
    hits = 0
    for _ in range(n_perm):
        d = vals.copy()
        for s in uniq:
            idx = idx_by_src[s]
            if RNG.random() < 0.5:
                d[idx] = -d[idx]
        if abs(wmean(d)) >= obs - 1e-12:
            hits += 1
    return (hits + 1) / (n_perm + 1)


def compare(name: str, dir_a: Path, dir_b: Path) -> dict:
    """a minus b per sentence (token-weighted), clustered by source."""
    A, B = _load(dir_a), _load(dir_b)
    assert set(A) == set(B), f'{name}: sentence sets differ between protocols'
    sids = sorted(A)
    sources = np.array([_source_of(s) for s in sids])
    res = {}
    for m in METRICS:
        diff = np.array([
            (np.mean(A[s][m]) - np.mean(B[s][m])) if A[s][m] else np.nan
            for s in sids
        ])
        valid = ~np.isnan(diff)
        mean = float(np.mean(np.concatenate([A[s][m] for s in sids])) -
                     np.mean(np.concatenate([B[s][m] for s in sids])))
        lo, hi = cluster_bootstrap(diff, sources)
        p = paired_randomization(diff, sources)
        res[m] = {'mean_diff_pp': round(mean * 100, 2),
                  'ci95_pp': [round(lo * 100, 2), round(hi * 100, 2)],
                  'p_randomization': round(p, 5),
                  'n_sentences': int(valid.sum())}
        print(f'[{name}] {m:8s} diff={mean * 100:+6.2f} pp '
              f'CI[{lo * 100:+.2f},{hi * 100:+.2f}] p={p:.4f}')
    return res


def main() -> None:
    out = {}
    for name, (a, b) in PAIRS.items():
        print(f'--- {name}')
        out[name] = compare(name, Path(a), Path(b))

    # Holm correction within the full family of 8 comparisons x 4 metrics
    flat = [(name, m, out[name][m]['p_randomization'])
            for name in out for m in METRICS]
    flat.sort(key=lambda t: t[2])
    n = len(flat)
    holm = {}
    running = 0.0
    for i, (name, m, p) in enumerate(flat):
        running = max(running, (n - i) * p)
        holm[(name, m)] = min(1.0, running)
    for (name, m), ph in holm.items():
        out[name][m]['p_holm'] = round(ph, 5)

    dest = Path('results_cv/grouped5_analysis.json')
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\nWrote {dest} ({n} hypotheses, Holm-corrected)')


if __name__ == '__main__':
    main()
