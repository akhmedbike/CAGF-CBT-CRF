"""Silver-pretraining followed by gold KTB fine-tuning.

Pipeline
--------
1. Build a *shared* vocabulary. The label spaces (UPOS, grammemes, lemma
   edit-rules) are anchored to the gold KTB training set, because every
   reported metric is computed on the gold test set and fine-tuning happens
   on gold. The character and word vocabularies are additionally enriched
   from the silver corpus, so silver pretraining benefits from the far larger
   surface-form coverage of the 2.89M-token connected corpus.
2. Pretrain the full CAGF-CBT+CRF model on the silver corpus, using the gold
   dev set for early stopping (so pretraining is stopped when transfer to the
   target domain peaks).
3. Fine-tune the pretrained model on the gold KTB training set and evaluate
   on the gold KTB test set.
4. Optionally also train a gold-only model from scratch with the identical
   protocol, to quantify what silver pretraining actually buys.

All results are reported on the gold KTB test set. The silver corpus is only
ever used for pretraining and is never evaluated as gold.

Usage
-----
    PYTHONPATH=. python scripts/pretrain_finetune.py \
        --silver data/silver/silver.conllu \
        --gold-train data/gold_merged/gold_train.conllu \
        --gold-dev   data/gold_merged/gold_dev.conllu \
        --gold-test  data/gold_merged/gold_test.conllu \
        --seed 42 --with-scratch-baseline
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import List

from cagf.data import (PAD, UNK, BOS, EOS, CorpusVocabs, Sentence, Vocab,
                       build_vocabs, form_to_edit_script, parse_feats, read_conllu)
from cagf.model import AblationConfig, ModelHParams
from cagf.train_loop import train_one_run


def build_shared_vocabs(gold_train: List[Sentence], silver: List[Sentence],
                        min_char_freq: int = 1, min_word_freq: int = 2,
                        max_word_vocab: int = 50000) -> CorpusVocabs:
    """Label vocabularies from gold only; char/word vocabularies from gold+silver."""
    base = build_vocabs(gold_train, min_char_freq=min_char_freq,
                        min_word_freq=min_word_freq, max_word_vocab=max_word_vocab)
    chars: List[str] = []
    words: List[str] = []
    for sent in gold_train + silver:
        for tok in sent.tokens:
            words.append(tok.form.lower())
            chars.extend(list(tok.form))
    char_vocab = Vocab.build(chars, specials=[PAD, UNK, BOS, EOS], min_freq=min_char_freq)
    word_vocab = Vocab.build(words, specials=[PAD, UNK], min_freq=min_word_freq, max_size=max_word_vocab)
    # labels stay anchored to gold (base); only char/word are widened
    return CorpusVocabs(char_vocab=char_vocab, word_vocab=word_vocab,
                        upos_vocab=base.upos_vocab, grammeme_vocab=base.grammeme_vocab,
                        lemma_rule_vocab=base.lemma_rule_vocab)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--silver", default="data/silver/silver.conllu")
    ap.add_argument("--gold-train", default="data/gold_merged/gold_train.conllu")
    ap.add_argument("--gold-dev", default="data/gold_merged/gold_dev.conllu")
    ap.add_argument("--gold-test", default="data/gold_merged/gold_test.conllu")
    ap.add_argument("--out-dir", default="results_transfer")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pretrain-epochs", type=int, default=30)
    ap.add_argument("--pretrain-lr", type=float, default=3e-4)
    ap.add_argument("--finetune-epochs", type=int, default=200)
    ap.add_argument("--finetune-lr", type=float, default=1e-4,
                    help="lower than pretraining LR: fine-tuning refines pretrained weights")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--grad-clip-norm", type=float, default=5.0)
    ap.add_argument("--pretrain-patience", type=int, default=5)
    ap.add_argument("--finetune-patience", type=int, default=15)
    ap.add_argument("--with-scratch-baseline", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    silver = read_conllu(args.silver)
    gold_train = read_conllu(args.gold_train)
    gold_dev = read_conllu(args.gold_dev)
    gold_test = read_conllu(args.gold_test)
    vocabs = build_shared_vocabs(gold_train, silver)
    vocabs.save(out_dir / "shared_vocabs.json")
    print(f"silver sentences={len(silver):,}  gold train/dev/test="
          f"{len(gold_train)}/{len(gold_dev)}/{len(gold_test)}")
    print(f"shared vocab: chars={len(vocabs.char_vocab)} words={len(vocabs.word_vocab)} "
          f"upos={len(vocabs.upos_vocab)} gram={len(vocabs.grammeme_vocab)} "
          f"lemma_rules={len(vocabs.lemma_rule_vocab)}")

    hparams = ModelHParams()
    ablation = AblationConfig()  # full model

    # ---- 1. pretrain on silver, early-stop on gold dev ----
    print("\n=== PRETRAIN on silver ===")
    pretrain = train_one_run(
        train_sentences=silver, dev_sentences=gold_dev, test_sentences=gold_test,
        vocabs=vocabs, ablation=ablation, hparams=hparams, seed=args.seed,
        max_epochs=args.pretrain_epochs, batch_size=args.batch_size,
        learning_rate=args.pretrain_lr, weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm, early_stopping_patience=args.pretrain_patience,
        device=args.device, verbose=True,
        save_checkpoint_path=str(out_dir / "pretrained.pt"))
    import torch
    pretrained_state = torch.load(out_dir / "pretrained.pt", map_location="cpu",
                                  weights_only=False)["model_state_dict"]

    # ---- 2. fine-tune on gold ----
    print("\n=== FINE-TUNE on gold KTB ===")
    finetuned = train_one_run(
        train_sentences=gold_train, dev_sentences=gold_dev, test_sentences=gold_test,
        vocabs=vocabs, ablation=ablation, hparams=hparams, seed=args.seed,
        max_epochs=args.finetune_epochs, batch_size=args.batch_size,
        learning_rate=args.finetune_lr, weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm, early_stopping_patience=args.finetune_patience,
        device=args.device, verbose=True,
        save_checkpoint_path=str(out_dir / "finetuned.pt"),
        init_state=pretrained_state)

    summary = {
        "seed": args.seed,
        "silver_sentences": len(silver),
        "silver_pretrain_then_gold_finetune": {
            "lemma": finetuned.lemma, "upos": finetuned.upos, "grammeme": finetuned.grammeme,
            "best_epoch": finetuned.best_epoch,
        },
    }

    # ---- 3. optional gold-only-from-scratch baseline (same protocol) ----
    if args.with_scratch_baseline:
        print("\n=== BASELINE: gold-only from scratch ===")
        scratch = train_one_run(
            train_sentences=gold_train, dev_sentences=gold_dev, test_sentences=gold_test,
            vocabs=vocabs, ablation=ablation, hparams=hparams, seed=args.seed,
            max_epochs=args.finetune_epochs, batch_size=args.batch_size,
            learning_rate=args.pretrain_lr, weight_decay=args.weight_decay,
            grad_clip_norm=args.grad_clip_norm, early_stopping_patience=args.finetune_patience,
            device=args.device, verbose=True)
        summary["gold_only_from_scratch"] = {
            "lemma": scratch.lemma, "upos": scratch.upos, "grammeme": scratch.grammeme,
            "best_epoch": scratch.best_epoch,
        }

    (out_dir / f"transfer_summary_seed{args.seed}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== RESULTS (gold KTB test set) ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
