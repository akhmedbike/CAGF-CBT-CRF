from __future__ import annotations
import argparse
import random
from pathlib import Path

def read_blocks(path: Path) -> list[str]:
    if not path.exists():
        return []
    text = path.read_text(encoding='utf-8')
    blocks = [b for b in text.split('\n\n') if b.strip()]
    return blocks

def is_verified(block: str) -> bool:
    return 'Verified=yes' in block

def tag_source(block: str, source_tag: str) -> str:
    lines = block.split('\n')
    out = []
    inserted = False
    for line in lines:
        out.append(line)
        if line.startswith('# sent_id') and (not inserted):
            out.append(f'# gold_source = {source_tag}')
            inserted = True
    if not inserted:
        out.insert(0, f'# gold_source = {source_tag}')
    return '\n'.join(out)

def write_blocks(blocks: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as fh:
        for b in blocks:
            fh.write(b.strip() + '\n\n')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ud-train', default='data/gold/ud_kk_train.conllu')
    ap.add_argument('--ud-test', default='data/gold/ud_kk_test.conllu')
    ap.add_argument('--user-verified', default='data/gold/user_verified.conllu')
    ap.add_argument('--out-dir', default='data/gold_merged')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--train-ratio', type=float, default=0.8)
    ap.add_argument('--dev-ratio', type=float, default=0.1)
    ap.add_argument('--include-unverified', action='store_true', help='DANGEROUS: include draft (non-reviewed) sentences as if gold. Never use this for numbers going into the paper.')
    args = ap.parse_args()
    ud_train_blocks = [tag_source(b, 'UD_Kazakh-KTB') for b in read_blocks(Path(args.ud_train))]
    ud_test_blocks = [tag_source(b, 'UD_Kazakh-KTB') for b in read_blocks(Path(args.ud_test))]
    ud_all = ud_train_blocks + ud_test_blocks
    user_blocks_raw = read_blocks(Path(args.user_verified))
    if args.include_unverified:
        user_blocks = user_blocks_raw
        n_excluded = 0
        print('WARNING: --include-unverified is set. Unreviewed draft tags are being')
        print('         treated as gold. Do NOT use resulting numbers in the paper.')
    else:
        user_blocks = [b for b in user_blocks_raw if is_verified(b)]
        n_excluded = len(user_blocks_raw) - len(user_blocks)
    user_blocks = [tag_source(b, 'user_verified_books') for b in user_blocks]
    print(f"UD_Kazakh-KTB sentences (train+test, per UD's own merge recommendation): {len(ud_all)}")
    print(f'User-verified book sentences included: {len(user_blocks)}')
    if n_excluded:
        print(f'User sentences EXCLUDED (not yet verified, MISC lacks Verified=yes): {n_excluded}')
    all_blocks = ud_all + user_blocks
    if not all_blocks:
        raise SystemExit('No gold sentences available yet -- nothing to merge. Run review_annotations.py first, or check --ud-train/--ud-test paths.')
    rng = random.Random(args.seed)
    indices = list(range(len(all_blocks)))
    rng.shuffle(indices)
    n_train = int(len(indices) * args.train_ratio)
    n_dev = int(len(indices) * args.dev_ratio)
    train_idx = indices[:n_train]
    dev_idx = indices[n_train:n_train + n_dev]
    test_idx = indices[n_train + n_dev:]
    out_dir = Path(args.out_dir)
    train_path = out_dir / 'gold_train.conllu'
    dev_path = out_dir / 'gold_dev.conllu'
    test_path = out_dir / 'gold_test.conllu'
    write_blocks([all_blocks[i] for i in train_idx], train_path)
    write_blocks([all_blocks[i] for i in dev_idx], dev_path)
    write_blocks([all_blocks[i] for i in test_idx], test_path)
    print(f'\nTotal gold sentences merged: {len(all_blocks)}')
    print(f'  train: {len(train_idx)} -> {train_path}')
    print(f'  dev:   {len(dev_idx)} -> {dev_path}')
    print(f'  test:  {len(test_idx)} -> {test_path}')
    print(f'  seed:  {args.seed} (fixed, for reproducibility)')
    print('\nNOTE: total size is still far smaller than the ~48,750 sentences')
    print("claimed in the manuscript's Table 1. Table 1 must be updated to")
    print('reflect the actual merged corpus size once annotation is complete,')
    print('not left as-is.')
if __name__ == '__main__':
    main()