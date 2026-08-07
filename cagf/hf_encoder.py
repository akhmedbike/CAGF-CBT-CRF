from __future__ import annotations
from pathlib import Path
from typing import List, Optional
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from .data import Sentence
from .device import pick_device

class HFWordEncoder(nn.Module):
    """Wraps a HuggingFace encoder to produce per-WORD representations.

    Subtoken aggregation: first-subword pooling. Tokenizes the list of UD
    syntactic words with ``is_split_into_words=True``, runs the encoder, then
    takes the hidden state at the FIRST subtoken of each word (the canonical
    BERT/RoBERTa convention -- it is the only subtoken position that "sees" the
    whole word via its right context inside the bidirectional transformer).

    Output shape: ``(batch, n_words, proj_dim)`` where ``proj_dim = hidden_dim``
    (256 by default), matching the ``CAGFCBTCRF.encode()`` contract so the same
    task heads (upos / lemma / grammeme / CRF) can be plugged on top unchanged.

    Notes on context window
    -----------------------
    Verified corpus statistics for ``kz-transformers/kaz-roberta-conversational``
    (52000-token BPE, 6 layers, hidden 768): mean fertility 1.22 subtokens/word,
    83.5% of words are a single subtoken, and no sentence in UD_Kazakh-KTB
    exceeds 510 subtokens (longest = 65). RoBERTa's 512-subtoken window is
    therefore never exceeded and NO windowing/slicing logic is needed here --
    a single forward pass per sentence is always sufficient.
    """

    def __init__(self, model_name: str, revision: str, proj_dim: int = 256, freeze: bool = False, cache_dir: Optional[str] = None):
        super().__init__()
        self.model_name = model_name
        self.revision = revision
        self.proj_dim = proj_dim
        self.freeze = freeze
        self.cache_dir = cache_dir
        self.encoder = AutoModel.from_pretrained(model_name, revision=revision, cache_dir=cache_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision, cache_dir=cache_dir, add_prefix_space=True, use_fast=True)
        hidden_size = int(self.encoder.config.hidden_size)
        self.proj = nn.Linear(hidden_size, proj_dim)
        self.dropout = nn.Dropout(0.1)
        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False

    @property
    def device(self) -> torch.device:
        return next(self.encoder.parameters()).device

    @torch.no_grad()
    def precompute_tokenization(self, sentences: List[Sentence]) -> dict:
        """Pre-tokenize a corpus once and return aligned, batchable lists.

        Running the HF tokenizer inside the training loop is expensive (it
        re-segments every word on every epoch even though the segmentation is
        deterministic). This method does it ONCE up front and stores three
        positionally-aligned lists that ``forward`` can consume directly with a
        plain ``torch.tensor`` + ``pad_sequence`` -- no per-step tokenizer call.

        Returns
        -------
        dict with keys
            - ``input_ids``: List[List[int]] -- per sentence, with special tokens
            - ``attention_mask``: List[List[int]] -- 1 for real token, 0 for pad
            - ``word_to_first_subtok``: List[List[int]] -- for each UD word, the
              subtoken index (into ``input_ids[s]``) of its first subtoken.
              Length = number of UD words in the sentence.
            - ``lengths``: List[int] -- number of UD words per sentence.

        Notes
        -----
        Wrapped in ``torch.no_grad`` because tokenization has no gradient; this
        also makes the cache safe to build while a model with ``requires_grad``
        context is on the autograd tape.
        """
        all_input_ids: List[List[int]] = []
        all_attn: List[List[int]] = []
        all_word_to_first: List[List[int]] = []
        all_lengths: List[int] = []
        for sent in sentences:
            words = [t.form for t in sent.tokens]
            enc = self.tokenizer(words, is_split_into_words=True, add_special_tokens=True)
            word_ids = enc.word_ids()
            first_subtok: List[int] = []
            seen: dict[int, int] = {}
            for subtok_idx, wid in enumerate(word_ids):
                if wid is None:
                    continue
                if wid not in seen:
                    seen[wid] = subtok_idx
                    first_subtok.append(subtok_idx)
            # Safety: if a word produced zero subtokens (degenerate empty form
            # collapsed by the tokenizer), fall back to the CLS index so the
            # downstream gather does not go out of range. This keeps the length
            # contract (= number of UD words) intact.
            n_words = len(words)
            while len(first_subtok) < n_words:
                first_subtok.append(0 if not word_ids else word_ids.index(0) if 0 in word_ids else 0)
            first_subtok = first_subtok[:n_words]
            all_input_ids.append(list(enc['input_ids']))
            all_attn.append(list(enc['attention_mask']))
            all_word_to_first.append(first_subtok)
            all_lengths.append(n_words)
        return {'input_ids': all_input_ids, 'attention_mask': all_attn, 'word_to_first_subtok': all_word_to_first, 'lengths': all_lengths}

    @classmethod
    def tokenize_and_cache(cls, sentences: List[Sentence], cache_path: str | Path, model_name: str = 'kz-transformers/kaz-roberta-conversational', revision: str = '43077c2fd0a163487ed468b5ec3b8750686a5888') -> dict:
        """Pre-tokenize ``sentences`` and persist the result to ``cache_path``.

        Convenience wrapper around :meth:`precompute_tokenization` (optimization
        O1 from the paper's reproducibility notes): build the cache once per
        corpus split, save it as a torch pickle, and load it cheaply on every
        subsequent training run instead of re-tokenizing.

        Returns the tokenization dict (same schema as
        :meth:`precompute_tokenization`) and writes the same dict to disk.
        """
        cache_path = Path(cache_path)
        if cache_path.exists():
            return torch.load(cache_path, weights_only=False)
        enc = cls(model_name=model_name, revision=revision)
        tok = enc.precompute_tokenization(sentences)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(tok, cache_path)
        return tok

    def _encode_one(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, word_to_first: List[int], n_words: int) -> torch.Tensor:
        """Run the encoder for a single sentence and gather per-word states.

        Returns ``(n_words, proj_dim)`` projected + dropout-ready embeddings.
        """
        input_ids = input_ids.to(self.device).unsqueeze(0)
        attention_mask = attention_mask.to(self.device).unsqueeze(0)
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state  # (1, n_subtok, 768)
        # Gather the first-subtoken state for each UD word.
        idx = torch.tensor(word_to_first, dtype=torch.long, device=self.device)
        word_states = hidden[0].index_select(0, idx)  # (n_words, 768)
        return self.dropout(self.proj(word_states))  # (n_words, proj_dim)

    def _pad_batch(self, per_sentence: List[torch.Tensor], max_len: int) -> torch.Tensor:
        batch_size = len(per_sentence)
        out = torch.zeros(batch_size, max_len, self.proj_dim, device=self.device, dtype=per_sentence[0].dtype)
        for b, h in enumerate(per_sentence):
            out[b, :h.size(0)] = out[b, :h.size(0)] + h.to(out)
        return out

    def forward(self, word_strings_batch: List[List[str]], mask: torch.Tensor, cached: Optional[dict] = None) -> torch.Tensor:
        """Encode a batch of sentences into per-WORD representations.

        Parameters
        ----------
        word_strings_batch:
            ``List[List[str]]`` of length B -- each inner list is the UD words
            of one sentence (surface forms, NOT lemmas).
        mask:
            ``(B, max_len)`` boolean tensor; True = real word, False = pad.
            ``max_len`` defines the output's second dimension.
        cached:
            Optional pre-tokenization (output of :meth:`precompute_tokenization`)
            indexed by sentence position. When supplied, the tokenizer is NOT
            called inside forward; instead the cached ``input_ids`` /
            ``attention_mask`` / ``word_to_first_subtok`` are reused. This is
            the fast path used during training. Set via ``self.set_cache`` or
            passed explicitly.

        Returns
        -------
        torch.Tensor of shape ``(B, max_len, proj_dim)`` with padded positions
        exactly zero (so downstream heads see a neutral input where ``mask``
        is False).
        """
        device = self.device
        max_len = int(mask.size(1))
        per_sentence: List[torch.Tensor] = []
        for b, words in enumerate(word_strings_batch):
            if cached is not None:
                input_ids = torch.tensor(cached['input_ids'][b], dtype=torch.long)
                attn = torch.tensor(cached['attention_mask'][b], dtype=torch.long)
                word_to_first = cached['word_to_first_subtok'][b]
            else:
                enc = self.tokenizer(words, is_split_into_words=True, add_special_tokens=True)
                word_ids = enc.word_ids()
                input_ids = torch.tensor(enc['input_ids'], dtype=torch.long)
                attn = torch.tensor(enc['attention_mask'], dtype=torch.long)
                word_to_first: List[int] = []
                seen: dict[int, int] = {}
                for subtok_idx, wid in enumerate(word_ids):
                    if wid is None:
                        continue
                    if wid not in seen:
                        seen[wid] = subtok_idx
                        word_to_first.append(subtok_idx)
                # Truncate / pad to the actual UD word count for consistency.
                word_to_first = word_to_first[:len(words)]
                while len(word_to_first) < len(words):
                    word_to_first.append(0)
            n_words = len(words)
            per_sentence.append(self._encode_one(input_ids, attn, word_to_first, n_words))
        out = self._pad_batch(per_sentence, max_len)
        # Defensive: zero padded positions explicitly (the _pad_batch already
        # initialises with zeros, but this also handles the case where a
        # sentence is shorter than max_len and we want a hard guarantee).
        m = mask.to(out.device)
        out = out * m.unsqueeze(-1).to(out.dtype)
        return out
