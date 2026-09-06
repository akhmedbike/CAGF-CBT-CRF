from __future__ import annotations
import random
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Optional
import numpy as np
import torch
from torch.utils.data import DataLoader
from .data import CorpusVocabs, Sentence, PAD
from .dataset import MorphDataset, collate_batch
from .device import pick_device
from .losses import UncertaintyWeightedLoss, masked_multilabel_bce, masked_lemma_ce
from .metrics import compute_classification_metrics, compute_multilabel_metrics
from .model import AblationConfig, CAGFCBTCRF, ModelHParams

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # warn_only: MPS/CUDA lack deterministic kernels for some ops, so we ask
    # for determinism where available instead of crashing; the empirical
    # run-to-run floor (results_cv/noise_floor.json) quantifies the residual.
    torch.use_deterministic_algorithms(True, warn_only=True)

@dataclass
class RunResult:
    config_name: str
    seed: int
    lemma: dict
    upos: dict
    grammeme: dict
    best_epoch: int
    history: list

def make_loader(sentences: list[Sentence], vocabs: CorpusVocabs, batch_size: int, shuffle: bool) -> DataLoader:
    ds = MorphDataset(sentences, vocabs)
    collate = partial(collate_batch, pad_word_id=vocabs.word_vocab.stoi[PAD], pad_char_id=vocabs.char_vocab.stoi[PAD])
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)

@torch.no_grad()
def _collect_gram_probs(model: CAGFCBTCRF, loader: DataLoader, device: str):
    """Sigmoid probabilities of the grammeme head plus gold multi-hot targets
    for every unmasked token, flattened in loader order (callers pass
    shuffle=False loaders, so this is corpus order). Feeds the sigmoid-threshold
    sensitivity analysis (reviewer 2, comment 1)."""
    model.eval()
    probs, gold = [], []
    for batch in loader:
        mask = batch['mask'].to(device)
        gram_targets = batch['grammeme_targets'].to(device)
        out = model(batch['word_ids'].to(device), batch['char_ids'].to(device),
                    batch['lengths'].to(device), mask, upos_ids=None)
        p = torch.sigmoid(out['grammeme_logits'])
        m = mask.bool()
        probs.append(p[m].cpu().numpy().astype('float16'))
        gold.append(gram_targets[m].cpu().numpy().astype('uint8'))
    return (np.concatenate(probs, axis=0) if probs else np.zeros((0, 0), dtype='float16'),
            np.concatenate(gold, axis=0) if gold else np.zeros((0, 0), dtype='uint8'))


@torch.no_grad()
def evaluate(model: CAGFCBTCRF, loader: DataLoader, device: str) -> dict:
    model.eval()
    all_lemma_true, all_lemma_pred = ([], [])
    all_upos_true, all_upos_pred = ([], [])
    all_gram_true, all_gram_pred = ([], [])
    for batch in loader:
        word_ids = batch['word_ids'].to(device)
        char_ids = batch['char_ids'].to(device)
        lengths = batch['lengths'].to(device)
        mask = batch['mask'].to(device)
        upos_ids = batch['upos_ids'].to(device)
        lemma_ids = batch['lemma_rule_ids'].to(device)
        gram_targets = batch['grammeme_targets'].to(device)
        out = model(word_ids, char_ids, lengths, mask, upos_ids=None)
        lemma_pred = out['lemma_logits'].argmax(dim=-1)
        gram_pred = (torch.sigmoid(out['grammeme_logits']) > 0.5).float()
        upos_pred = out['upos_pred']
        mask_cpu = mask.cpu()
        for b in range(word_ids.size(0)):
            length = int(mask_cpu[b].sum().item())
            all_lemma_true.extend(lemma_ids[b, :length].cpu().tolist())
            all_lemma_pred.extend(lemma_pred[b, :length].cpu().tolist())
            if isinstance(upos_pred, list):
                all_upos_pred.extend(upos_pred[b])
            else:
                all_upos_pred.extend(upos_pred[b, :length].cpu().tolist())
            all_upos_true.extend(upos_ids[b, :length].cpu().tolist())
            all_gram_true.append(gram_targets[b, :length].cpu().numpy())
            all_gram_pred.append(gram_pred[b, :length].cpu().numpy())
    lemma_metrics = compute_classification_metrics(all_lemma_true, all_lemma_pred)
    upos_metrics = compute_classification_metrics(all_upos_true, all_upos_pred)
    gram_metrics = compute_multilabel_metrics(np.concatenate(all_gram_true, axis=0), np.concatenate(all_gram_pred, axis=0))
    return {'lemma': lemma_metrics.as_dict(), 'upos': upos_metrics.as_dict(), 'grammeme': gram_metrics.as_dict()}


@torch.no_grad()
def predict(model: CAGFCBTCRF, sentences: list[Sentence], vocabs: CorpusVocabs,
            batch_size: int, device: str) -> list[dict]:
    """Run a trained model over ``sentences`` and return, per sentence, the
    decoded LEMMA / UPOS / FEATS strings in the exact order of the input.

    This is the inference counterpart of :func:`evaluate`: ``evaluate``
    collapses predictions into scalar metrics, while ``predict`` keeps the
    per-token string predictions so they can be written to a CoNLL-U file
    (cagf.predict_writer) and scored by the official conll18_ud_eval script.
    The ``loader`` is rebuilt with ``shuffle=False`` so output order is
    guaranteed to match ``sentences`` regardless of batching.

    Output schema (one dict per input sentence)::

        {'lemma': [str, ...], 'upos': [str, ...], 'feats': [[str, ...], ...]}

    - ``lemma[i]``: the predicted surface lemma, produced by applying the
      predicted edit-script rule to ``tokens[i].form`` via ``apply_edit_script``
      (falls back to the surface form if the rule is invalid).
    - ``upos[i]``: the predicted UPOS tag string (CRF-decoded when the model
      uses CRF, argmax otherwise).
    - ``feats[i]``: the list of predicted ``Feature=Value`` grammeme labels
      (sigmoid > 0.5, decoded through ``grammeme_vocab``). Empty list means
      no grammeme predicted -- the writer serialises this to ``_``.
    """
    from .data import apply_edit_script, PAD
    model.eval()
    loader = make_loader(sentences, vocabs, batch_size, shuffle=False)
    # DataLoader with shuffle=False still iterates sentences in dataset order,
    # which is the input list order (MorphDataset does not reorder).
    per_sentence: list[dict] = []
    sent_idx = 0
    for batch in loader:
        word_ids = batch['word_ids'].to(device)
        char_ids = batch['char_ids'].to(device)
        lengths = batch['lengths'].to(device)
        mask = batch['mask'].to(device)
        out = model(word_ids, char_ids, lengths, mask, upos_ids=None)
        lemma_pred = out['lemma_logits'].argmax(dim=-1)
        gram_pred = (torch.sigmoid(out['grammeme_logits']) > 0.5).float()
        upos_pred = out['upos_pred']
        mask_cpu = mask.cpu()
        for b in range(word_ids.size(0)):
            length = int(mask_cpu[b].sum().item())
            sent = sentences[sent_idx]
            sent_idx += 1
            lemmas: list[str] = []
            uposes: list[str] = []
            feats_list: list[list[str]] = []
            for i in range(length):
                rule_id = int(lemma_pred[b, i].item())
                rule = vocabs.lemma_rule_vocab.decode(rule_id)
                # apply the edit script to the surface form to recover the lemma;
                # if the rule is <unk>/<pad> or fails to apply, keep the surface
                # form (a reasonable fallback, never an empty string).
                form = sent.tokens[i].form
                # UPOS for case-aware lemma decoding: the predicted tag governs
                # case restoration (PROPN/acronym → keep capitals, everything
                # else → lowercase). Resolved after we know upos_pred below.
                if isinstance(upos_pred, list):
                    upos_id = int(upos_pred[b][i])
                else:
                    upos_id = int(upos_pred[b, i].item())
                upos_str = vocabs.upos_vocab.decode(upos_id)
                decoded = apply_edit_script(form, rule, upos=upos_str)
                lemmas.append(decoded if decoded else form)
                uposes.append(upos_str)
                # grammeme head: multi-label, each active dim is a Feature=Value
                active = gram_pred[b, i].cpu().tolist()
                feats = [vocabs.grammeme_vocab.decode(g) for g, a in enumerate(active)
                         if a > 0.5 and g < len(vocabs.grammeme_vocab)]
                # drop <unk> if it sneaks through as an active label
                feats = [f for f in feats if f and f != '<unk>']
                feats_list.append(feats)
            per_sentence.append({'lemma': lemmas, 'upos': uposes, 'feats': feats_list})
    return per_sentence

def train_one_run(train_sentences: list[Sentence], dev_sentences: list[Sentence], test_sentences: list[Sentence], vocabs: CorpusVocabs, ablation: AblationConfig, hparams: ModelHParams, seed: int, max_epochs: int, batch_size: int, learning_rate: float, weight_decay: float, grad_clip_norm: float, early_stopping_patience: int, device: Optional[str]=None, verbose: bool=True, save_checkpoint_path: Optional[str]=None, init_state: Optional[dict]=None, return_model: bool=False, probs_dump_path: Optional[str]=None):
    """Train one model and evaluate on test.

        When ``return_model=True`` the trained model (with best-state loaded) is
        returned alongside the RunResult, so callers that need further inference
        -- e.g. cross-validation fold prediction via :func:`predict` -- can do so
        without re-loading from a checkpoint. Default ``False`` keeps every
        existing call site unchanged.

        When ``probs_dump_path`` is set, sigmoid probabilities of the grammeme
        head and the gold multi-hot targets are dumped for dev and test to a
        compressed .npz (keys: probs_dev/gold_dev/probs_test/gold_test,
        grammemes, seed, best_epoch, config). Dev rows feed threshold tuning,
        test rows the frozen-threshold evaluation (reviewer 2, comment 1).
    """
    device = device or pick_device()
    set_seed(seed)
    train_loader = make_loader(train_sentences, vocabs, batch_size, shuffle=True)
    dev_loader = make_loader(dev_sentences, vocabs, batch_size, shuffle=False)
    test_loader = make_loader(test_sentences, vocabs, batch_size, shuffle=False)
    model = CAGFCBTCRF(num_chars=len(vocabs.char_vocab), num_words=len(vocabs.word_vocab), num_upos=len(vocabs.upos_vocab), num_grammemes=len(vocabs.grammeme_vocab), num_lemma_rules=len(vocabs.lemma_rule_vocab), hparams=hparams, ablation=ablation, char_pad_id=vocabs.char_vocab.stoi[PAD], word_pad_id=vocabs.word_vocab.stoi[PAD]).to(device)
    if init_state is not None:
        # Fine-tuning: load every pretrained parameter whose name and shape
        # match the current model (strict=False so heads that changed size
        # between pretraining and fine-tuning are simply left randomly
        # initialised rather than raising).
        own = model.state_dict()
        compatible = {k: v for k, v in init_state.items() if k in own and own[k].shape == v.shape}
        own.update(compatible)
        model.load_state_dict(own)
        if verbose:
            print(f'  loaded {len(compatible)}/{len(own)} pretrained tensors for fine-tuning')
    loss_weighter = UncertaintyWeightedLoss(num_tasks=3).to(device)
    params = list(model.parameters()) + list(loss_weighter.parameters())
    optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    best_score = -1.0
    best_state = None
    best_epoch = -1
    epochs_without_improvement = 0
    history = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            word_ids = batch['word_ids'].to(device)
            char_ids = batch['char_ids'].to(device)
            lengths = batch['lengths'].to(device)
            mask = batch['mask'].to(device)
            upos_ids = batch['upos_ids'].to(device)
            lemma_ids = batch['lemma_rule_ids'].to(device)
            gram_targets = batch['grammeme_targets'].to(device)
            optimizer.zero_grad()
            out = model(word_ids, char_ids, lengths, mask, upos_ids=upos_ids)
            lemma_loss = masked_lemma_ce(out['lemma_logits'], lemma_ids)
            gram_loss = masked_multilabel_bce(out['grammeme_logits'], gram_targets, mask)
            upos_loss = out['upos_nll']
            total_loss, _ = loss_weighter([lemma_loss, upos_loss, gram_loss])
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(params, grad_clip_norm)
            optimizer.step()
            epoch_loss += float(total_loss.item())
            n_batches += 1
        dev_metrics = evaluate(model, dev_loader, device)
        dev_score = (dev_metrics['lemma']['f1'] + dev_metrics['upos']['f1'] + dev_metrics['grammeme']['f1']) / 3.0
        scheduler.step(dev_score)
        history.append({'epoch': epoch, 'train_loss': epoch_loss / max(n_batches, 1), 'dev_macro_f1_mean': dev_score})
        if verbose:
            print(f'  [{ablation.name()} seed={seed}] epoch {epoch}: train_loss={epoch_loss / max(n_batches, 1):.4f} dev_f1={dev_score:.4f}')
        if dev_score > best_score:
            best_score = dev_score
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                if verbose:
                    print(f'  early stopping at epoch {epoch} (best epoch {best_epoch})')
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, device)
    if probs_dump_path is not None:
        probs_dev, gold_dev = _collect_gram_probs(model, dev_loader, device)
        probs_test, gold_test = _collect_gram_probs(model, test_loader, device)
        Path(probs_dump_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            probs_dump_path,
            probs_dev=probs_dev, gold_dev=gold_dev,
            probs_test=probs_test, gold_test=gold_test,
            grammemes=np.array(vocabs.grammeme_vocab.itos, dtype=object),
            seed=seed, best_epoch=best_epoch, config=ablation.name())
        if verbose:
            print(f'  Dumped grammeme probabilities to {probs_dump_path}')
    if save_checkpoint_path is not None:
        import dataclasses
        Path(save_checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({'model_state_dict': model.state_dict(), 'hparams': dataclasses.asdict(hparams), 'ablation': dataclasses.asdict(ablation), 'num_chars': len(vocabs.char_vocab), 'num_words': len(vocabs.word_vocab), 'num_upos': len(vocabs.upos_vocab), 'num_grammemes': len(vocabs.grammeme_vocab), 'num_lemma_rules': len(vocabs.lemma_rule_vocab), 'char_pad_id': vocabs.char_vocab.stoi[PAD], 'word_pad_id': vocabs.word_vocab.stoi[PAD], 'test_metrics': test_metrics, 'seed': seed, 'best_epoch': best_epoch}, save_checkpoint_path)
        if verbose:
            print(f'  Saved checkpoint to {save_checkpoint_path}')
    result = RunResult(config_name=ablation.name(), seed=seed, lemma=test_metrics['lemma'], upos=test_metrics['upos'], grammeme=test_metrics['grammeme'], best_epoch=best_epoch, history=history)
    if return_model:
        return result, model
    return result