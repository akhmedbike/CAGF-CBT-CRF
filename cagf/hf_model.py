from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from .crf import LinearChainCRF
from .hf_encoder import HFWordEncoder

class HFMorphModel(nn.Module):
    """Kazakh morphological tagger whose encoder is a HuggingFace transformer.

    This is a parameter-efficiency baseline against :class:`CAGFCBTCRF`: instead
    of the CharCNN + WordBiLSTM + Transformer + GatedFusion stack (which has
    ~3-4M task-specific parameters on top of learned word/char embeddings), we
    plug a pretrained KazRoBERTa (83.5M parameters, frozen by default) followed
    by a single ``Linear(768, 256)`` projection into the SAME three task heads
    (upos / lemma / grammeme) and the same :class:`LinearChainCRF` decoder.

    The interface mirrors ``CAGFCBTCRF`` so the train loop and the inference
    code can be reused with minimal changes:

    - ``encode(word_strings_batch, mask) -> (B, max_len, hidden_dim)``
    - ``forward(word_strings_batch, mask, upos_ids=None) -> dict`` with the same
      output keys (``lemma_logits``, ``grammeme_logits``, ``upos_emissions``,
      ``upos_nll`` when ``upos_ids`` is given, ``upos_pred``).

    The differences from ``CAGFCBTCRF`` are:
    - No ``word_ids`` / ``char_ids`` inputs -- the encoder consumes the raw
      surface-form strings, since the HF tokenizer is responsible for
      subword segmentation.
    - ``encode`` returns ``hidden_dim = proj_dim`` (256) directly, so there is
      no separate fusion/bilstm stage.
    """

    def __init__(self, num_upos: int, num_grammemes: int, num_lemma_rules: int, model_name: str = 'kz-transformers/kaz-roberta-conversational', revision: str = '43077c2fd0a163487ed468b5ec3b8750686a5888', hidden_dim: int = 256, use_crf: bool = True, freeze_encoder: bool = False, encoder_cache_dir: Optional[str] = None):
        super().__init__()
        self.use_crf = use_crf
        self.hidden_dim = hidden_dim
        self.encoder = HFWordEncoder(model_name=model_name, revision=revision, proj_dim=hidden_dim, freeze=freeze_encoder, cache_dir=encoder_cache_dir)
        self.upos_head = nn.Linear(hidden_dim, num_upos)
        self.lemma_head = nn.Linear(hidden_dim, num_lemma_rules)
        self.grammeme_head = nn.Linear(hidden_dim, num_grammemes)
        if use_crf:
            self.crf = LinearChainCRF(num_upos)

    def encode(self, word_strings_batch, mask: torch.Tensor, cached: Optional[dict] = None) -> torch.Tensor:
        """Per-WORD representations from the HF encoder.

        Returns ``(B, max_len, hidden_dim)`` -- the same contract as
        ``CAGFCBTCRF.encode``, so any head trained on top of the CAGF encoder
        transfers here unchanged.
        """
        return self.encoder(word_strings_batch, mask, cached=cached)

    def forward(self, word_strings_batch, mask: torch.Tensor, upos_ids: Optional[torch.Tensor] = None, cached: Optional[dict] = None) -> dict:
        fused = self.encode(word_strings_batch, mask, cached=cached)
        lemma_logits = self.lemma_head(fused)
        grammeme_logits = self.grammeme_head(fused)
        upos_emissions = self.upos_head(fused)
        out = {'lemma_logits': lemma_logits, 'grammeme_logits': grammeme_logits, 'upos_emissions': upos_emissions}
        if self.use_crf:
            if upos_ids is not None:
                safe_tags = upos_ids.clone()
                safe_tags[~mask] = 0
                out['upos_nll'] = self.crf.neg_log_likelihood(upos_emissions, safe_tags, mask)
            out['upos_pred'] = self.crf.decode(upos_emissions, mask)
        else:
            if upos_ids is not None:
                out['upos_nll'] = F.cross_entropy(upos_emissions.transpose(1, 2), upos_ids, ignore_index=-100)
            out['upos_pred'] = upos_emissions.argmax(dim=-1)
        return out
