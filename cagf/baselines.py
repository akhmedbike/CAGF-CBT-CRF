from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import CharCNNEncoder, WordBiLSTMEncoder, TransformerEncoderBlock


@dataclass
class BaselineHParams:
    char_emb_dim: int = 32
    char_cnn_filters: int = 64
    word_emb_dim: int = 128
    hidden_dim: int = 256
    bilstm_layers: int = 1
    transformer_layers: int = 2
    transformer_heads: int = 4
    transformer_ff_dim: int = 512
    dropout: float = 0.1


class _BaselineBase(nn.Module):
    """Shared forward()/loss interface matching cagf.model.CAGFCBTCRF's
    dict output (lemma_logits, grammeme_logits, upos_emissions, upos_nll,
    upos_pred), so these models can be evaluated with the same
    cagf.metrics functions and the same downstream comparison pipeline.
    None of these baselines use CRF decoding -- softmax classification
    only -- matching how the manuscript's Figures 5/6 describe them
    (no baseline in the text is credited with structured decoding).
    """

    def _heads(self, fused_dim: int, num_upos: int, num_grammemes: int, num_lemma_rules: int) -> None:
        self.upos_head = nn.Linear(fused_dim, num_upos)
        self.lemma_head = nn.Linear(fused_dim, num_lemma_rules)
        self.grammeme_head = nn.Linear(fused_dim, num_grammemes)

    def _finish(self, fused: torch.Tensor, upos_ids: torch.Tensor | None) -> dict:
        lemma_logits = self.lemma_head(fused)
        grammeme_logits = self.grammeme_head(fused)
        upos_emissions = self.upos_head(fused)
        out = {"lemma_logits": lemma_logits, "grammeme_logits": grammeme_logits, "upos_emissions": upos_emissions}
        if upos_ids is not None:
            out["upos_nll"] = F.cross_entropy(upos_emissions.transpose(1, 2), upos_ids, ignore_index=-100)
        out["upos_pred"] = upos_emissions.argmax(dim=-1)
        return out


class CNNBaseline(_BaselineBase):
    """Word + character CNN features, classified per token with NO
    sequence-level context (no BiLSTM, no Transformer). This is a
    deliberately context-free baseline: it represents "CNN" as named in
    the manuscript's Figures 5/6, which describe it as having "limited
    ability to capture sequential dependencies and long-range contextual
    information" -- i.e. exactly the absence of any recurrent or
    attention-based context this class implements.
    """

    def __init__(self, num_chars, num_words, num_upos, num_grammemes, num_lemma_rules,
                 hparams: BaselineHParams, char_pad_id: int = 0, word_pad_id: int = 0):
        super().__init__()
        self.word_embed = nn.Embedding(num_words, hparams.word_emb_dim, padding_idx=word_pad_id)
        self.char_cnn = CharCNNEncoder(num_chars, hparams.char_emb_dim, hparams.char_cnn_filters, pad_id=char_pad_id)
        input_dim = hparams.word_emb_dim + self.char_cnn.output_dim
        self.proj = nn.Linear(input_dim, hparams.hidden_dim)
        self.dropout = nn.Dropout(hparams.dropout)
        self._heads(hparams.hidden_dim, num_upos, num_grammemes, num_lemma_rules)

    def forward(self, word_ids, char_ids, lengths, mask, upos_ids=None):
        x = torch.cat([self.word_embed(word_ids), self.char_cnn(char_ids)], dim=-1)
        fused = self.dropout(F.gelu(self.proj(x)))
        return self._finish(fused, upos_ids)


class CNNBiLSTMBaseline(_BaselineBase):
    """Adds a word-level BiLSTM on top of the CNN baseline's features,
    giving the model sequential context that CNNBaseline lacks --
    matching the manuscript's description of "the benefit of
    bidirectional contextual encoding" for this configuration."""

    def __init__(self, num_chars, num_words, num_upos, num_grammemes, num_lemma_rules,
                 hparams: BaselineHParams, char_pad_id: int = 0, word_pad_id: int = 0):
        super().__init__()
        self.word_embed = nn.Embedding(num_words, hparams.word_emb_dim, padding_idx=word_pad_id)
        self.char_cnn = CharCNNEncoder(num_chars, hparams.char_emb_dim, hparams.char_cnn_filters, pad_id=char_pad_id)
        input_dim = hparams.word_emb_dim + self.char_cnn.output_dim
        self.proj = nn.Linear(input_dim, hparams.hidden_dim)
        self.dropout = nn.Dropout(hparams.dropout)
        self.bilstm = WordBiLSTMEncoder(hparams.hidden_dim, hparams.hidden_dim // 2, hparams.bilstm_layers, hparams.dropout)
        self._heads(hparams.hidden_dim, num_upos, num_grammemes, num_lemma_rules)

    def forward(self, word_ids, char_ids, lengths, mask, upos_ids=None):
        x = torch.cat([self.word_embed(word_ids), self.char_cnn(char_ids)], dim=-1)
        x = self.dropout(F.gelu(self.proj(x)))
        fused = self.bilstm(x, lengths)
        return self._finish(fused, upos_ids)


class CNNBiLSTMTransformerBaseline(_BaselineBase):
    """Stacks a Transformer encoder ON TOP OF the BiLSTM output
    (sequential composition, not the parallel gated fusion used by
    CAGF-CBT+CRF). This distinguishes it architecturally from the
    proposed model's fusion mechanism while still combining recurrent
    and self-attention context, matching the manuscript's description
    that "global self-attention enhances contextual representation
    learning" for this configuration."""

    def __init__(self, num_chars, num_words, num_upos, num_grammemes, num_lemma_rules,
                 hparams: BaselineHParams, char_pad_id: int = 0, word_pad_id: int = 0):
        super().__init__()
        self.word_embed = nn.Embedding(num_words, hparams.word_emb_dim, padding_idx=word_pad_id)
        self.char_cnn = CharCNNEncoder(num_chars, hparams.char_emb_dim, hparams.char_cnn_filters, pad_id=char_pad_id)
        input_dim = hparams.word_emb_dim + self.char_cnn.output_dim
        self.proj = nn.Linear(input_dim, hparams.hidden_dim)
        self.dropout = nn.Dropout(hparams.dropout)
        self.bilstm = WordBiLSTMEncoder(hparams.hidden_dim, hparams.hidden_dim // 2, hparams.bilstm_layers, hparams.dropout)
        self.transformer = TransformerEncoderBlock(hparams.hidden_dim, hparams.transformer_layers, hparams.transformer_heads, hparams.transformer_ff_dim, hparams.dropout)
        self._heads(hparams.hidden_dim, num_upos, num_grammemes, num_lemma_rules)

    def forward(self, word_ids, char_ids, lengths, mask, upos_ids=None):
        x = torch.cat([self.word_embed(word_ids), self.char_cnn(char_ids)], dim=-1)
        x = self.dropout(F.gelu(self.proj(x)))
        h = self.bilstm(x, lengths)
        fused = self.transformer(h, key_padding_mask=~mask)
        return self._finish(fused, upos_ids)


class SubwordTaggingBaseline(_BaselineBase):
    """Character-CNN features ONLY -- no word embeddings -- followed by a
    BiLSTM, emphasizing explicit subword-level modeling in isolation
    from whole-word lexical identity. This matches the manuscript's own
    characterization of this baseline as "highlighting the importance of
    explicit subword-level modeling for morphologically rich languages,"
    i.e. its distinguishing feature relative to the other baselines is
    the absence of word-level embeddings, not their presence."""

    def __init__(self, num_chars, num_words, num_upos, num_grammemes, num_lemma_rules,
                 hparams: BaselineHParams, char_pad_id: int = 0, word_pad_id: int = 0):
        super().__init__()
        self.char_cnn = CharCNNEncoder(num_chars, hparams.char_emb_dim, hparams.char_cnn_filters, pad_id=char_pad_id)
        self.proj = nn.Linear(self.char_cnn.output_dim, hparams.hidden_dim)
        self.dropout = nn.Dropout(hparams.dropout)
        self.bilstm = WordBiLSTMEncoder(hparams.hidden_dim, hparams.hidden_dim // 2, hparams.bilstm_layers, hparams.dropout)
        self._heads(hparams.hidden_dim, num_upos, num_grammemes, num_lemma_rules)

    def forward(self, word_ids, char_ids, lengths, mask, upos_ids=None):
        x = self.dropout(F.gelu(self.proj(self.char_cnn(char_ids))))
        fused = self.bilstm(x, lengths)
        return self._finish(fused, upos_ids)


BASELINE_CLASSES = {
    "cnn": CNNBaseline,
    "cnn_bilstm": CNNBiLSTMBaseline,
    "cnn_bilstm_transformer": CNNBiLSTMTransformerBaseline,
    "subword_tagging": SubwordTaggingBaseline,
}
