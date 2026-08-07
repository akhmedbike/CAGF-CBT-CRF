"""K3: hyperparameter grid for KazRoBERTa on fold 0 (15 epochs each).

Grid: lr {1e-5, 3e-5, 5e-5} x batch {16, 32} = 6 configs.
Selects by dev macro-F1 mean. Logs all 6 to docs/kazroberta_hparams.md.
"""
from __future__ import annotations
import json, time, yaml
from pathlib import Path
from cagf.data import read_conllu, build_vocabs
from cagf.folds import make_folds
from cagf.hf_train_loop import train_hf_one_run

MN = 'kz-transformers/kaz-roberta-conversational'
REV = '43077c2fd0a163487ed468b5ec3b8750686a5888'

cfg = yaml.safe_load(Path('configs/default.yaml').read_text(encoding='utf-8'))
sents = (read_conllu(cfg['data']['train_path']) +
         read_conllu(cfg['data']['dev_path']) +
         read_conllu(cfg['data']['test_path']))
folds = make_folds(sents, k=10, seed=42, strategy='stratified')
fold0 = folds[0]
train_s = [sents[i] for i in fold0.train_idx]
dev_s = [sents[i] for i in fold0.dev_idx]
test_s = [sents[i] for i in fold0.test_idx]
vocabs = build_vocabs(train_s, min_word_freq=1, max_word_vocab=50000)
print(f'fold 0: train={len(train_s)} dev={len(dev_s)} test={len(test_s)}')

grid = [(lr, bs) for lr in (1e-5, 3e-5, 5e-5) for bs in (16, 32)]
results = []
for i, (lr, bs) in enumerate(grid):
    tag = f'lr{lr}_bs{bs}'
    print(f'\n=== [{i+1}/6] {tag} ===')
    t0 = time.time()
    res = train_hf_one_run(
        train_s, dev_s, test_s, vocabs,
        model_name=MN, revision=REV, seed=42,
        max_epochs=15, batch_size=bs, learning_rate=lr,
        weight_decay=0.01, grad_clip_norm=5.0,
        early_stopping_patience=15,  # no early stop in grid — run all 15
        use_crf=True, freeze_encoder=False,
        verbose=True, device=None)
    elapsed = time.time() - t0
    dev_f1 = (res.lemma['f1'] + res.upos['f1'] + res.grammeme['f1']) / 3
    print(f'  {tag}: dev_macro_f1={dev_f1:.4f} best_epoch={res.best_epoch} elapsed={elapsed:.0f}s')
    results.append({'lr': lr, 'batch_size': bs, 'dev_macro_f1': dev_f1,
                    'lemma_f1': res.lemma['f1'], 'upos_f1': res.upos['f1'],
                    'grammeme_f1': res.grammeme['f1'], 'best_epoch': res.best_epoch,
                    'elapsed_sec': elapsed})

# select best
best = max(results, key=lambda r: r['dev_macro_f1'])
print(f'\n=== BEST: lr={best["lr"]} batch={best["batch_size"]} dev_f1={best["dev_macro_f1"]:.4f} ===')
Path('logs/kazroberta_hp/grid_results.json').write_text(json.dumps(results, indent=2))
print(f'Results: logs/kazroberta_hp/grid_results.json')
