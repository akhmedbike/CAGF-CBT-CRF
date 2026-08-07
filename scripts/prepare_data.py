from __future__ import annotations
import argparse
import random
from pathlib import Path
from cagf.data import read_conllu, Sentence

def write_conllu(sentences: list[Sentence], path: Path) -> None:
    with open(path, 'w', encoding='utf-8') as fh:
        for sent in sentences:
            for i, tok in enumerate(sent.tokens, start=1):
                fh.write(f'{i}\t{tok.form}\t{tok.lemma}\t{tok.upos}\t_\t{tok.feats}\t0\t_\t_\t_\n')
            fh.write('\n')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--out-dir', default='data')
    ap.add_argument('--train-ratio', type=float, default=0.8)
    ap.add_argument('--dev-ratio', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    sentences = read_conllu(args.input)
    rng = random.Random(args.seed)
    indices = list(range(len(sentences)))
    rng.shuffle(indices)
    n_train = int(len(indices) * args.train_ratio)
    n_dev = int(len(indices) * args.dev_ratio)
    train_idx = indices[:n_train]
    dev_idx = indices[n_train:n_train + n_dev]
    test_idx = indices[n_train + n_dev:]
    train = [sentences[i] for i in train_idx]
    dev = [sentences[i] for i in dev_idx]
    test = [sentences[i] for i in test_idx]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_conllu(train, out_dir / 'split_train.conllu')
    write_conllu(dev, out_dir / 'split_dev.conllu')
    write_conllu(test, out_dir / 'split_test.conllu')
    train_path = out_dir / 'split_train.conllu'
    dev_path = out_dir / 'split_dev.conllu'
    test_path = out_dir / 'split_test.conllu'
    print(f'Total sentences: {len(sentences)}')
    print(f'Train: {len(train)} ({len(train) / len(sentences):.1%})  -> {train_path}')
    print(f'Dev:   {len(dev)} ({len(dev) / len(sentences):.1%})  -> {dev_path}')
    print(f'Test:  {len(test)} ({len(test) / len(sentences):.1%})  -> {test_path}')
    print(f'Seed:  {args.seed} (fixed, for reproducibility)')
if __name__ == '__main__':
    main()