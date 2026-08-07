from __future__ import annotations
from typing import Dict, List
import torch
from torch.utils.data import Dataset
from .data import CorpusVocabs, Sentence, form_to_edit_script, parse_feats, BOS, EOS
MAX_WORD_CHARS = 20

class MorphDataset(Dataset):

    def __init__(self, sentences: List[Sentence], vocabs: CorpusVocabs):
        self.sentences = sentences
        self.vocabs = vocabs

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx: int) -> Dict:
        sent = self.sentences[idx]
        v = self.vocabs
        word_ids = [v.word_vocab.encode(t.form.lower()) for t in sent.tokens]
        char_ids = []
        for t in sent.tokens:
            chars = [BOS] + list(t.form)[:MAX_WORD_CHARS] + [EOS]
            char_ids.append([v.char_vocab.encode(c) for c in chars])
        upos_ids = [v.upos_vocab.encode(t.upos) for t in sent.tokens]
        lemma_rule_ids = [v.lemma_rule_vocab.encode(form_to_edit_script(t.form, t.lemma)) for t in sent.tokens]
        n_grammemes = len(v.grammeme_vocab)
        grammeme_targets = torch.zeros(len(sent.tokens), n_grammemes)
        for i, t in enumerate(sent.tokens):
            for feat in parse_feats(t.feats):
                gid = v.grammeme_vocab.stoi.get(feat)
                if gid is not None:
                    grammeme_targets[i, gid] = 1.0
        # Surface word strings are needed by the HF encoder (KazRoBERTa), which
        # consumes raw forms rather than vocab ids -- the HF tokenizer is
        # responsible for subword segmentation. This is a plain Python list of
        # str, NOT a tensor, so it is passed through the collate function
        # without padding and consumed positionally by HFMorphModel.forward.
        word_strings = [t.form for t in sent.tokens]
        return {'word_ids': torch.tensor(word_ids, dtype=torch.long), 'char_ids': [torch.tensor(c, dtype=torch.long) for c in char_ids], 'upos_ids': torch.tensor(upos_ids, dtype=torch.long), 'lemma_rule_ids': torch.tensor(lemma_rule_ids, dtype=torch.long), 'grammeme_targets': grammeme_targets, 'length': len(sent.tokens), 'word_strings': word_strings}

def collate_batch(batch: List[Dict], pad_word_id: int, pad_char_id: int) -> Dict:
    batch_size = len(batch)
    max_len = max((item['length'] for item in batch))
    max_word_chars = max((max((c.size(0) for c in item['char_ids']), default=1) for item in batch))
    n_grammemes = batch[0]['grammeme_targets'].size(1)
    word_ids = torch.full((batch_size, max_len), pad_word_id, dtype=torch.long)
    char_ids = torch.full((batch_size, max_len, max_word_chars), pad_char_id, dtype=torch.long)
    upos_ids = torch.full((batch_size, max_len), -100, dtype=torch.long)
    lemma_rule_ids = torch.full((batch_size, max_len), -100, dtype=torch.long)
    grammeme_targets = torch.zeros(batch_size, max_len, n_grammemes)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    for b, item in enumerate(batch):
        length = item['length']
        word_ids[b, :length] = item['word_ids']
        upos_ids[b, :length] = item['upos_ids']
        lemma_rule_ids[b, :length] = item['lemma_rule_ids']
        grammeme_targets[b, :length] = item['grammeme_targets']
        mask[b, :length] = True
        for t, chars in enumerate(item['char_ids']):
            char_ids[b, t, :chars.size(0)] = chars
    # ``word_strings`` is intentionally NOT tensorised: the HF encoder needs the
    # raw surface forms so its tokenizer can do BPE segmentation. We pass the
    # list-of-lists through unchanged. ``.get(...) or []`` keeps the collate
    # backward-compatible with any caller whose __getitem__ does not yet emit
    # the key (e.g. serialised older data, custom datasets).
    word_strings = [item.get('word_strings', []) for item in batch]
    return {'word_ids': word_ids, 'char_ids': char_ids, 'upos_ids': upos_ids, 'lemma_rule_ids': lemma_rule_ids, 'grammeme_targets': grammeme_targets, 'mask': mask, 'lengths': torch.tensor([item['length'] for item in batch], dtype=torch.long), 'word_strings': word_strings}