from __future__ import annotations
import argparse
from pathlib import Path
from sklearn.metrics import cohen_kappa_score

def read_tokens(path: Path) -> list[tuple[str, str, str, str]]:
    tokens = []
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.rstrip('\n').split('\t')
            if len(parts) != 10:
                continue
            form, lemma, upos, feats = (parts[1], parts[2], parts[3], parts[5])
            tokens.append((form, lemma, upos, feats))
    return tokens

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--annotator-a', required=True)
    ap.add_argument('--annotator-b', required=True)
    args = ap.parse_args()
    tokens_a = read_tokens(Path(args.annotator_a))
    tokens_b = read_tokens(Path(args.annotator_b))
    if len(tokens_a) != len(tokens_b):
        raise SystemExit(f'Token count mismatch: annotator A has {len(tokens_a)} tokens, annotator B has {len(tokens_b)}. Both files must cover exactly the same sentences in the same order (e.g. both produced by reviewing the same active_learning_select.py batch) for IAA to be meaningful -- comparing different sentence sets is not a valid agreement measurement.')
    mismatched_forms = [i for i, (a, b) in enumerate(zip(tokens_a, tokens_b)) if a[0] != b[0]]
    if mismatched_forms:
        raise SystemExit(f"Surface forms differ at {len(mismatched_forms)} token position(s) (first mismatch at index {mismatched_forms[0]}: '{tokens_a[mismatched_forms[0]][0]}' vs '{tokens_b[mismatched_forms[0]][0]}'). The two files are not aligned to the same tokens -- check both were derived from the identical input batch.")
    lemmas_a = [t[1] for t in tokens_a]
    lemmas_b = [t[1] for t in tokens_b]
    upos_a = [t[2] for t in tokens_a]
    upos_b = [t[2] for t in tokens_b]
    feats_a = [t[3] for t in tokens_a]
    feats_b = [t[3] for t in tokens_b]
    n = len(tokens_a)
    lemma_exact_agree = sum((a == b for a, b in zip(lemmas_a, lemmas_b))) / n
    upos_exact_agree = sum((a == b for a, b in zip(upos_a, upos_b))) / n
    feats_exact_agree = sum((a == b for a, b in zip(feats_a, feats_b))) / n
    kappa_upos = cohen_kappa_score(upos_a, upos_b)
    kappa_feats = cohen_kappa_score(feats_a, feats_b)
    kappa_lemma = cohen_kappa_score(lemmas_a, lemmas_b)
    print(f'Tokens compared: {n}\n')
    print(f"{'Field':10s} {'% exact agreement':20s} {'Cohen kappa':12s}")
    print(f"{'LEMMA':10s} {lemma_exact_agree:20.4f} {kappa_lemma:12.4f}")
    print(f"{'UPOS':10s} {upos_exact_agree:20.4f} {kappa_upos:12.4f}")
    print(f"{'FEATS':10s} {feats_exact_agree:20.4f} {kappa_feats:12.4f}")
    print('\nStandard interpretation (Landis & Koch, 1977): <0 poor, 0.00-0.20 slight, 0.21-0.40 fair, 0.41-0.60 moderate, 0.61-0.80 substantial, 0.81-1.00 almost perfect.')
    print('Report these numbers (and the exact table above) directly in the')
    print("paper's annotation-methodology section as evidence of annotation")
    print('reliability -- do not round selectively or omit a low value.')
if __name__ == '__main__':
    main()