from __future__ import annotations
import argparse
from pathlib import Path
from cagf.kk_morphrules import guess_nominal_morphology
CASE_TO_FEATS = {'Gen': 'Case=Gen', 'Dat': 'Case=Dat', 'Acc': 'Case=Acc', 'Loc': 'Case=Loc', 'Abl': 'Case=Abl', 'Ins': 'Case=Ins'}

def annotate_line(line: str) -> str:
    if line.startswith('#') or not line.strip():
        return line
    parts = line.rstrip('\n').split('\t')
    if len(parts) != 10:
        return line
    form = parts[1]
    if not form.isalpha():
        return line
    guess = guess_nominal_morphology(form)
    feats_parts = []
    if guess.number == 'Plur':
        feats_parts.append('Number=Plur')
    if guess.case is not None:
        feats_parts.append(CASE_TO_FEATS[guess.case])
    feats = '|'.join(feats_parts) if feats_parts else '_'
    lemma = guess.lemma_guess if guess.number or guess.case else '_'
    misc = f'PreAnnot=rule|Confidence={guess.confidence}'
    parts[2] = lemma
    parts[5] = feats
    parts[9] = misc
    return '\t'.join(parts) + '\n'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_high, n_low, n_total = (0, 0, 0)
    with open(in_path, encoding='utf-8') as fin, open(out_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            annotated = annotate_line(line)
            fout.write(annotated)
            if '\t' in annotated and (not annotated.startswith('#')):
                n_total += 1
                if 'Confidence=high' in annotated:
                    n_high += 1
                elif 'Confidence=low' in annotated:
                    n_low += 1
    print(f'Wrote draft-annotated corpus to {out_path}')
    print(f'Tokens with a high-confidence rule match: {n_high}')
    print(f'Tokens with a low-confidence / no match:  {n_low}')
    print('\nREMINDER: this is DRAFT annotation (MISC=PreAnnot=rule). Known')
    print('false-positive patterns exist (non-Kazakh text, coincidental')
    print('suffix-like word endings) -- every tag must be reviewed by a')
    print('linguist before use as training gold data. Prioritize reviewing')
    print('Confidence=high tokens first (higher volume), but do not skip')
    print('Confidence=low tokens either -- they are simply unmatched, not')
    print('verified-absent.')
if __name__ == '__main__':
    main()