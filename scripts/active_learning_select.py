from __future__ import annotations
import argparse
from pathlib import Path

def read_blocks(path: Path) -> list[str]:
    text = path.read_text(encoding='utf-8')
    return [b for b in text.split('\n\n') if b.strip()]

def score_block(block: str) -> float:
    lines = [l for l in block.split('\n') if l and (not l.startswith('#'))]
    if not lines:
        return -1.0
    n = len(lines)
    if n < 4:
        return -1.0
    n_high = sum((1 for l in lines if 'Confidence=high' in l))
    frac_high = n_high / n
    mix_score = 1.0 - abs(frac_high - 0.5) * 2
    is_flagged = 'review_flag' in block
    penalty = 0.3 if is_flagged else 0.0
    length_bonus = min(n / 30.0, 1.0)
    return mix_score + 0.2 * length_bonus - penalty

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--top-n', type=int, default=200)
    args = ap.parse_args()
    blocks = read_blocks(Path(args.input))
    scored = sorted(((score_block(b), b) for b in blocks), key=lambda x: -x[0])
    top = [b for score, b in scored[:args.top_n] if score > -1.0]
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as fh:
        for b in top:
            fh.write(b.strip() + '\n\n')
    print(f'Selected {len(top)} of {len(blocks)} sentences for priority review.')
    print(f'Written to: {out_path}')
    print('Feed this file directly into scripts/review_annotations.py as --input')
    print('to review the highest-value sentences first.')
if __name__ == '__main__':
    main()