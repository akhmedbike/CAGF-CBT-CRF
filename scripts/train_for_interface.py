from __future__ import annotations
import argparse
import yaml
from cagf.data import build_vocabs, read_conllu
from cagf.model import AblationConfig, ModelHParams
from cagf.train_loop import train_one_run

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/default.yaml')
    ap.add_argument('--seed', type=int, default=13)
    ap.add_argument('--max-epochs', type=int, default=None)
    ap.add_argument('--out', default='models/interface_model.pt')
    ap.add_argument('--vocabs-out', default='models/interface_vocabs.json')
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding='utf-8'))
    max_epochs = args.max_epochs or cfg['training']['max_epochs']
    train = read_conllu(cfg['data']['train_path'])
    dev = read_conllu(cfg['data']['dev_path'])
    test = read_conllu(cfg['data']['test_path'])
    vocabs = build_vocabs(train, min_word_freq=cfg['data']['min_word_freq'], max_word_vocab=cfg['data']['max_word_vocab'])
    from pathlib import Path
    Path(args.vocabs_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    vocabs.save(args.vocabs_out)
    hp = ModelHParams(**cfg['model'])
    ablation = AblationConfig()
    print(f'Training full_model, seed={args.seed}, max_epochs={max_epochs}, on {len(train)} train / {len(dev)} dev / {len(test)} test sentences.')
    result = train_one_run(train, dev, test, vocabs, ablation, hp, seed=args.seed, max_epochs=max_epochs, batch_size=cfg['training']['batch_size'], learning_rate=cfg['training']['learning_rate'], weight_decay=cfg['training']['weight_decay'], grad_clip_norm=cfg['training']['grad_clip_norm'], early_stopping_patience=cfg['training']['early_stopping_patience'], verbose=True, save_checkpoint_path=args.out)
    import torch
    ckpt = torch.load(args.out, weights_only=False)
    ckpt['train_sentences'] = len(train)
    ckpt['dev_sentences'] = len(dev)
    ckpt['test_sentences'] = len(test)
    torch.save(ckpt, args.out)
    print(f'\nTest metrics -- lemma: {result.lemma}')
    print(f'Test metrics -- upos:  {result.upos}')
    print(f'Test metrics -- grammeme: {result.grammeme}')
    print(f'\nCheckpoint saved to {args.out}')
    print(f'Vocabs saved to {args.vocabs_out}')
    print('\nIMPORTANT: this checkpoint quality directly reflects the size and')
    print('quality of the training corpus used. At the current corpus size')
    print('(see README), predictions from this model will be weak. Retrain')
    print('with this same script once the gold corpus has grown.')
if __name__ == '__main__':
    main()