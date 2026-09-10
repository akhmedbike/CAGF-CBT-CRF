"""D (R3.8, extension task): Stanza baseline under the paper's exact folds.

Trains Stanza's POS tagger (UPOS+FEATS, one biaffine model) and lemmatizer
per fold of the standard 10-fold stratified protocol (seed 42) on
pre-tokenized gold CoNLL-U — no tokenizer (the corpus is pre-tokenized), no
pretrained vectors (per the revision spec: from-scratch embeddings), model
selection on the fold's DEV via the trainer's eval file, then prediction on
the fold's TEST with a local Pipeline and scoring with the official
conll18_ud_eval — identical to every other condition in the paper.

The trainer runs as a subprocess (stanza's own CLI); per-fold artifacts and
resume follow the repo conventions (result.json per condition dir, run_meta,
summary loso-style JSON).

Timeboxed per the revision plan; --max-steps-pos/--max-steps-lemma bound the
budget (defaults sized for this 862-sentence treebank on MPS).

Usage
-----
    PYTHONPATH=. .venv/bin/python scripts/run_stanza_cv.py --out-dir results_stanza
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

from cagf.data import read_conllu
from cagf.folds import make_folds
from cagf.official_eval import evaluate_conllu
from cagf.predict_writer import write_conllu
from cagf.run_meta import write_run_meta
from scripts.run_cv import _gold_for_fold

PY = sys.executable


def write_full_conllu(sentences, path: Path) -> None:
    """CoNLL-U with the original gold syntax columns — stanza's reader wants
    the full 10-column format."""
    with open(path, 'w', encoding='utf-8') as f:
        for s in sentences:
            for c in s.comments:
                f.write(c + '\n')
            for i, t in enumerate(s.tokens, start=1):
                cols = [str(i), t.form, t.lemma, t.upos, t.xpos or '_',
                        t.feats or '_', str(t.head) if t.head is not None else '_',
                        t.deprel or '_', t.deps or '_', t.misc or '_']
                f.write('\t'.join(cols) + '\n')
            f.write('\n')


def train_stanza(module: str, cond_dir: Path, max_budget: int, device: str, seed: int) -> None:
    """Train one module for the current fold.

    stanza's trainer builds input paths itself from DATA_ROOT/{module}/
    kk_cv.{train,dev}.in.conllu, so the fold files are written there (env
    DATA_ROOT isolates each fold) and model selection runs on the fold dev.
    --no_charlm --no_pretrain keep the setup free of pretrained components,
    per the revision spec. Budget flag differs per trainer: --max_steps (pos)
    vs --num_epoch (lemma seq2seq)."""
    import os
    data_dir = cond_dir / 'stanza_data' / module
    data_dir.mkdir(parents=True, exist_ok=True)
    for split in ('train', 'dev'):
        src = cond_dir / f'{split}.conllu'
        (data_dir / f'kk_cv.{split}.in.conllu').write_bytes(src.read_bytes())
    save_dir = cond_dir / 'stanza_models'
    save_name = f'kk_{module}.pt'
    budget_flag = '--max_steps' if module == 'pos' else '--num_epoch'
    cmd = [PY, '-m', f'stanza.utils.training.run_{module}', 'kk_cv',
           '--eval_file', str(data_dir / 'kk_cv.dev.in.conllu'),
           '--save_dir', str(save_dir), '--save_name', save_name,
           '--device', device, '--seed', str(seed),
           budget_flag, str(max_budget), '--force', '--save_output',
           '--no_charlm']
    if module == 'pos':
        # the POS tagger would otherwise latch onto kk's fasttext157
        # pretrained vectors; the revision spec forbids pretrained vectors
        # for this baseline (the lemma seq2seq is char-level and needs none)
        cmd.append('--no_pretrain')
    env = dict(os.environ, DATA_ROOT=str(cond_dir / 'stanza_data'))
    print(f'  $ run_{module} ({budget_flag}={max_budget}, device={device})')
    t0 = time.time()
    subprocess.run(cmd, check=True, env=env)
    out = save_dir / save_name
    assert out.exists(), f'trainer did not produce {out}'
    print(f'  [{module}] trained in {time.time()-t0:.0f}s -> {out}')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', default='configs/default.yaml')
    ap.add_argument('--out-dir', default='results_stanza/stratified')
    ap.add_argument('--k', type=int, default=10)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--folds', default=None, help='comma list of fold indices (default: all)')
    ap.add_argument('--device', default='mps')
    ap.add_argument('--max-steps-pos', type=int, default=4000)
    ap.add_argument('--num-epoch-lemma', type=int, default=60)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    sentences = (read_conllu(cfg['data']['train_path']) +
                 read_conllu(cfg['data']['dev_path']) +
                 read_conllu(cfg['data']['test_path']))
    folds = make_folds(sentences, k=args.k, seed=args.seed, strategy='stratified')
    if args.folds:
        wanted = {int(x) for x in args.folds.split(',') if x.strip()}
        folds = [f for f in folds if f.index in wanted]

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    write_run_meta(out_root, driver='run_stanza_cv.py', k=args.k,
                   strategy='stratified', seed=args.seed, folds=args.folds,
                   max_steps_pos=args.max_steps_pos,
                   num_epoch_lemma=args.num_epoch_lemma)

    results = []
    for fold in folds:
        cond_dir = out_root / f'fold_{fold.index}'
        done_path = cond_dir / 'result.json'
        if done_path.exists():
            results.append(json.loads(done_path.read_text(encoding='utf-8')))
            print(f'[fold {fold.index}] already done — skipped')
            continue
        cond_dir.mkdir(parents=True, exist_ok=True)
        train_s = [sentences[i] for i in fold.train_idx]
        dev_s = [sentences[i] for i in fold.dev_idx]
        test_s = [sentences[i] for i in fold.test_idx]
        train_f = cond_dir / 'train.conllu'
        dev_f = cond_dir / 'dev.conllu'
        write_full_conllu(train_s, train_f)
        write_full_conllu(dev_s, dev_f)
        gold_path = cond_dir / 'gold.conllu'
        if not gold_path.exists():
            _gold_for_fold(test_s, gold_path)

        t0 = time.time()
        # model selection happens on the fold DEV (the trainer's eval file)
        train_stanza('pos', cond_dir, args.max_steps_pos, args.device, args.seed)
        train_stanza('lemma', cond_dir, args.num_epoch_lemma, args.device, args.seed)

        import stanza
        from stanza.resources.common import DEFAULT_MODEL_DIR
        # lay the trained models out as the 'ktb_nocharlm' package (our models
        # have no charlm / pretrained vectors) and copy the pretokenized-mode
        # tokenizer from the downloaded cache; resources.json rides along
        kk_dir = cond_dir / 'kk'
        (kk_dir / 'pos').mkdir(parents=True, exist_ok=True)
        (kk_dir / 'lemma').mkdir(parents=True, exist_ok=True)
        (kk_dir / 'tokenize').mkdir(parents=True, exist_ok=True)
        (kk_dir / 'pos' / 'ktb_nocharlm.pt').write_bytes(
            (cond_dir / 'stanza_models' / 'kk_pos.pt').read_bytes())
        (kk_dir / 'lemma' / 'ktb_nocharlm.pt').write_bytes(
            (cond_dir / 'stanza_models' / 'kk_lemma.pt').read_bytes())
        tok_src = Path(DEFAULT_MODEL_DIR) / 'kk' / 'tokenize' / 'ktb.pt'
        tok_dst = kk_dir / 'tokenize' / 'ktb.pt'
        if not tok_dst.exists() and tok_src.exists():
            tok_dst.write_bytes(tok_src.read_bytes())
        res_src = Path(DEFAULT_MODEL_DIR) / 'resources.json'
        res_dst = cond_dir / 'resources.json'
        if not res_dst.exists() and res_src.exists():
            res_dst.write_bytes(res_src.read_bytes())
        nlp = stanza.Pipeline(lang='kk', dir=str(cond_dir), processors='tokenize,pos,lemma',
                              package={'tokenize': 'ktb', 'pos': 'ktb_nocharlm',
                                       'lemma': 'ktb_nocharlm'},
                              tokenize_pretokenized=True, verbose=False,
                              download_method=None, device=args.device)
        preds = []
        for batch_start in range(0, len(test_s), 32):
            chunk = test_s[batch_start:batch_start+32]
            # pretokenized input form: list of word lists, one per sentence
            doc = nlp([[t.form for t in s.tokens] for s in chunk])
            assert len(doc.sentences) == len(chunk), \
                f'sentence count mismatch: {len(doc.sentences)} != {len(chunk)}'
            for sent in doc.sentences:
                preds.append({
                    'lemma': [w.lemma or w.text for w in sent.words],
                    'upos': [w.upos or '_' for w in sent.words],
                    'feats': [w.feats.split('|') if w.feats not in (None, '_', '')
                              else [] for w in sent.words],
                })
        pred_path = cond_dir / 'pred.conllu'
        write_conllu(test_s, preds, pred_path)
        official = evaluate_conllu(str(gold_path), str(pred_path))
        entry = {'fold': fold.index, 'official': official,
                 'n_train': len(train_s), 'n_test': len(test_s),
                 'elapsed_sec': time.time() - t0}
        done_path.write_text(json.dumps(entry, ensure_ascii=False, indent=2), encoding='utf-8')
        results.append(entry)
        print(f"[fold {fold.index}] official: "
              + ' '.join(f'{k}={v:.4f}' for k, v in official.items()))

    if results:
        import statistics as st
        summary = {m: {'mean': st.mean(e['official'][m] for e in results) * 100,
                       'n_folds': len(results)}
                   for m in results[0]['official']}
        out = out_root / 'stanza_results.json'
        out.write_text(json.dumps({'folds': results, 'summary': summary},
                                  ensure_ascii=False, indent=2), encoding='utf-8')
        print('\nStanza mean over folds: '
              + ' '.join(f'{m}={v["mean"]:.2f}' for m, v in summary.items()))
        print(f'Wrote {out}')


if __name__ == '__main__':
    main()
