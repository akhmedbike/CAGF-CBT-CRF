from __future__ import annotations
from dataclasses import dataclass
import torch
from .data import CorpusVocabs, PAD, UNK, BOS, EOS
from .model import AblationConfig, CAGFCBTCRF, ModelHParams
from .dataset import MAX_WORD_CHARS

@dataclass
class TokenAnalysis:
    form: str
    lemma_rule_guess: str
    upos: str
    grammemes: list[str]
    upos_known: bool
    upos_emission_confidence: float
    grammeme_probs: dict[str, float]

def load_checkpoint(checkpoint_path: str, vocabs_path: str, device: str='cpu'):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    vocabs = CorpusVocabs.load(vocabs_path)
    hparams = ModelHParams(**ckpt['hparams'])
    ablation = AblationConfig(**ckpt['ablation'])
    model = CAGFCBTCRF(num_chars=ckpt['num_chars'], num_words=ckpt['num_words'], num_upos=ckpt['num_upos'], num_grammemes=ckpt['num_grammemes'], num_lemma_rules=ckpt['num_lemma_rules'], hparams=hparams, ablation=ablation, char_pad_id=ckpt['char_pad_id'], word_pad_id=ckpt['word_pad_id'])
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    meta = {'test_metrics': ckpt.get('test_metrics'), 'seed': ckpt.get('seed'), 'best_epoch': ckpt.get('best_epoch'), 'ablation_name': ablation.name(), 'train_sentences': ckpt.get('train_sentences'), 'dev_sentences': ckpt.get('dev_sentences'), 'test_sentences': ckpt.get('test_sentences')}
    return (model, vocabs, meta)

def _encode_sentence(tokens: list[str], vocabs: CorpusVocabs):
    word_ids = [vocabs.word_vocab.encode(t.lower()) for t in tokens]
    char_ids = []
    for t in tokens:
        chars = [BOS] + list(t)[:MAX_WORD_CHARS] + [EOS]
        char_ids.append([vocabs.char_vocab.encode(c) for c in chars])
    return (word_ids, char_ids)

def analyze_tokens(tokens: list[str], model: CAGFCBTCRF, vocabs: CorpusVocabs, grammeme_threshold: float=0.5) -> list[TokenAnalysis]:
    if not tokens:
        return []
    word_ids, char_ids = _encode_sentence(tokens, vocabs)
    max_chars = max((len(c) for c in char_ids))
    char_pad = vocabs.char_vocab.stoi[PAD]
    word_tensor = torch.tensor([word_ids], dtype=torch.long)
    char_tensor = torch.full((1, len(tokens), max_chars), char_pad, dtype=torch.long)
    for i, c in enumerate(char_ids):
        char_tensor[0, i, :len(c)] = torch.tensor(c, dtype=torch.long)
    lengths = torch.tensor([len(tokens)], dtype=torch.long)
    mask = torch.ones(1, len(tokens), dtype=torch.bool)
    with torch.no_grad():
        out = model(word_tensor, char_tensor, lengths, mask, upos_ids=None)
    lemma_pred = out['lemma_logits'].argmax(dim=-1)[0].tolist()
    gram_probs = torch.sigmoid(out['grammeme_logits'])[0]
    upos_pred = out['upos_pred']
    if isinstance(upos_pred, list):
        upos_ids = upos_pred[0]
    else:
        upos_ids = upos_pred[0].tolist()
    emission_softmax = torch.softmax(out['upos_emissions'], dim=-1)[0]

    # PAD/UNK occupy real class indices in upos_vocab / grammeme_vocab
    # because num_upos == len(vocabs.upos_vocab) includes the special
    # tokens. Nothing upstream prevents the CRF/argmax from selecting the
    # PAD index for a real, non-padding token, which would otherwise leak
    # the literal string "<pad>" into the UI as if it were a predicted
    # part of speech. Filtered out explicitly here rather than relying on
    # it never happening in practice.
    upos_pad_idx = vocabs.upos_vocab.stoi.get(PAD)
    grammeme_unk_idx = vocabs.grammeme_vocab.stoi.get(UNK)

    results = []
    for i, form in enumerate(tokens):
        lemma_rule = vocabs.lemma_rule_vocab.decode(lemma_pred[i]) if lemma_pred[i] < len(vocabs.lemma_rule_vocab) else '<unk>'
        raw_upos_id = upos_ids[i]
        upos_is_known = raw_upos_id < len(vocabs.upos_vocab) and raw_upos_id != upos_pad_idx
        upos = vocabs.upos_vocab.decode(raw_upos_id) if upos_is_known else '<unk>'
        grams = [vocabs.grammeme_vocab.decode(j) for j in range(len(vocabs.grammeme_vocab)) if j != grammeme_unk_idx and gram_probs[i, j].item() > grammeme_threshold]
        gram_prob_map = {g: round(gram_probs[i, vocabs.grammeme_vocab.stoi[g]].item(), 4) for g in grams}
        emission_conf = round(emission_softmax[i, raw_upos_id].item(), 4) if raw_upos_id < emission_softmax.shape[-1] else 0.0
        results.append(TokenAnalysis(form=form, lemma_rule_guess=lemma_rule, upos=upos, grammemes=grams, upos_known=upos_is_known, upos_emission_confidence=emission_conf, grammeme_probs=gram_prob_map))
    return results