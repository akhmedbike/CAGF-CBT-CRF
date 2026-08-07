from __future__ import annotations
import argparse
import json
from pathlib import Path
import yaml
import torch
from cagf.data import build_vocabs, read_conllu, PAD
from cagf.baselines import BASELINE_CLASSES, BaselineHParams
from cagf.device import pick_device
from cagf.losses import UncertaintyWeightedLoss, masked_lemma_ce, masked_multilabel_bce
from cagf.train_loop import set_seed, make_loader, evaluate


def train_one_baseline(train_sentences, dev_sentences, test_sentences, vocabs, model_name,
                        hparams, seed, max_epochs, batch_size, learning_rate, weight_decay,
                        grad_clip_norm, early_stopping_patience, device=None, verbose=True):
    device = device or pick_device()
    set_seed(seed)

    train_loader = make_loader(train_sentences, vocabs, batch_size, shuffle=True)
    dev_loader = make_loader(dev_sentences, vocabs, batch_size, shuffle=False)
    test_loader = make_loader(test_sentences, vocabs, batch_size, shuffle=False)

    model_cls = BASELINE_CLASSES[model_name]
    model = model_cls(
        num_chars=len(vocabs.char_vocab), num_words=len(vocabs.word_vocab),
        num_upos=len(vocabs.upos_vocab), num_grammemes=len(vocabs.grammeme_vocab),
        num_lemma_rules=len(vocabs.lemma_rule_vocab), hparams=hparams,
        char_pad_id=vocabs.char_vocab.stoi[PAD], word_pad_id=vocabs.word_vocab.stoi[PAD],
    ).to(device)

    loss_weighter = UncertaintyWeightedLoss(num_tasks=3).to(device)
    params = list(model.parameters()) + list(loss_weighter.parameters())
    optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    best_score, best_state, best_epoch, no_improve = -1.0, None, -1, 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for batch in train_loader:
            word_ids = batch["word_ids"].to(device)
            char_ids = batch["char_ids"].to(device)
            lengths = batch["lengths"].to(device)
            mask = batch["mask"].to(device)
            upos_ids = batch["upos_ids"].to(device)
            lemma_ids = batch["lemma_rule_ids"].to(device)
            gram_targets = batch["grammeme_targets"].to(device)

            optimizer.zero_grad()
            out = model(word_ids, char_ids, lengths, mask, upos_ids=upos_ids)
            lemma_loss = masked_lemma_ce(out["lemma_logits"], lemma_ids)
            gram_loss = masked_multilabel_bce(out["grammeme_logits"], gram_targets, mask)
            upos_loss = out["upos_nll"]
            total_loss, _ = loss_weighter([lemma_loss, upos_loss, gram_loss])
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(params, grad_clip_norm)
            optimizer.step()
            epoch_loss += float(total_loss.item())
            n_batches += 1

        dev_metrics = evaluate(model, dev_loader, device)
        dev_score = (dev_metrics["lemma"]["f1"] + dev_metrics["upos"]["f1"] + dev_metrics["grammeme"]["f1"]) / 3.0
        scheduler.step(dev_score)
        if verbose:
            print(f"  [{model_name} seed={seed}] epoch {epoch}: train_loss={epoch_loss/max(n_batches,1):.4f} dev_f1={dev_score:.4f}")

        if dev_score > best_score:
            best_score, best_epoch = dev_score, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= early_stopping_patience:
                if verbose:
                    print(f"  early stopping at epoch {epoch} (best epoch {best_epoch})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, device)
    return {"model": model_name, "seed": seed, "lemma": test_metrics["lemma"],
            "upos": test_metrics["upos"], "grammeme": test_metrics["grammeme"], "best_epoch": best_epoch}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seeds = args.seeds or cfg["training"]["seed_list"]
    max_epochs = args.max_epochs or cfg["training"]["max_epochs"]

    train = read_conllu(cfg["data"]["train_path"])
    dev = read_conllu(cfg["data"]["dev_path"])
    test = read_conllu(cfg["data"]["test_path"])
    vocabs = build_vocabs(train, min_word_freq=cfg["data"]["min_word_freq"], max_word_vocab=cfg["data"]["max_word_vocab"])

    hp = BaselineHParams(
        char_emb_dim=cfg["model"]["char_emb_dim"], char_cnn_filters=cfg["model"]["char_cnn_filters"],
        word_emb_dim=cfg["model"]["word_emb_dim"], hidden_dim=cfg["model"]["hidden_dim"],
        bilstm_layers=cfg["model"]["bilstm_layers"], transformer_layers=cfg["model"]["transformer_layers"],
        transformer_heads=cfg["model"]["transformer_heads"], transformer_ff_dim=cfg["model"]["transformer_ff_dim"],
        dropout=cfg["model"]["dropout"],
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "baseline_raw.json"

    all_results = []
    already_done = set()
    if raw_path.exists():
        prior = json.loads(raw_path.read_text(encoding="utf-8"))
        all_results = prior
        already_done = {(r["model"], r["seed"]) for r in prior}
        print(f"Found {len(prior)} existing runs, will skip already-completed ones.")

    def flush():
        raw_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")

    for model_name in BASELINE_CLASSES:
        for seed in seeds:
            if (model_name, seed) in already_done:
                print(f"=== Skipping {model_name}, seed={seed} (already completed) ===")
                continue
            print(f"=== Running {model_name}, seed={seed} ===")
            result = train_one_baseline(
                train, dev, test, vocabs, model_name, hp, seed=seed, max_epochs=max_epochs,
                batch_size=cfg["training"]["batch_size"], learning_rate=cfg["training"]["learning_rate"],
                weight_decay=cfg["training"]["weight_decay"], grad_clip_norm=cfg["training"]["grad_clip_norm"],
                early_stopping_patience=cfg["training"]["early_stopping_patience"], verbose=True,
            )
            all_results.append(result)
            flush()

    print(f"\nDone. Raw results: {raw_path}")


if __name__ == "__main__":
    main()
