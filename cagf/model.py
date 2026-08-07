from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from .crf import LinearChainCRF
from .modules import CharCNNEncoder, CharBiLSTMEncoder, WordBiLSTMEncoder, TransformerEncoderBlock, GatedFusion

@dataclass
class AblationConfig:
    use_char_cnn: bool = True
    use_char_bilstm: bool = False
    use_word_bilstm: bool = True
    use_transformer: bool = True
    use_gated_fusion: bool = True
    use_crf: bool = True

    def name(self) -> str:
        if self.use_char_cnn is False and self.use_char_bilstm is False and (self.use_word_bilstm is False) and (self.use_gated_fusion is False) and (self.use_transformer is True) and (self.use_crf is True):
            return 'transformer_only'
        if not self.use_char_cnn and (not self.use_char_bilstm):
            return 'wo_character_encoder'
        if not self.use_gated_fusion:
            return 'wo_gated_fusion'
        if not self.use_crf:
            return 'wo_crf'
        return 'full_model'

@dataclass
class ModelHParams:
    char_emb_dim: int = 32
    char_cnn_filters: int = 64
    char_bilstm_hidden: int = 64
    word_emb_dim: int = 128
    hidden_dim: int = 256
    bilstm_layers: int = 1
    transformer_layers: int = 2
    transformer_heads: int = 4
    transformer_ff_dim: int = 512
    dropout: float = 0.1

class CAGFCBTCRF(nn.Module):

    def __init__(self, num_chars: int, num_words: int, num_upos: int, num_grammemes: int, num_lemma_rules: int, hparams: ModelHParams, ablation: AblationConfig, char_pad_id: int=0, word_pad_id: int=0):
        super().__init__()
        self.hp = hparams
        self.ab = ablation
        self.word_embed = nn.Embedding(num_words, hparams.word_emb_dim, padding_idx=word_pad_id)
        char_out_dim = 0
        if ablation.use_char_cnn:
            self.char_cnn = CharCNNEncoder(num_chars, hparams.char_emb_dim, hparams.char_cnn_filters, pad_id=char_pad_id)
            char_out_dim += self.char_cnn.output_dim
        if ablation.use_char_bilstm:
            self.char_bilstm = CharBiLSTMEncoder(num_chars, hparams.char_emb_dim, hparams.char_bilstm_hidden, pad_id=char_pad_id)
            char_out_dim += self.char_bilstm.output_dim
        input_dim = hparams.word_emb_dim + char_out_dim
        self.input_proj = nn.Linear(input_dim, hparams.hidden_dim)
        self.input_dropout = nn.Dropout(hparams.dropout)
        if ablation.use_word_bilstm:
            self.bilstm = WordBiLSTMEncoder(hparams.hidden_dim, hparams.hidden_dim // 2, hparams.bilstm_layers, hparams.dropout)
        if ablation.use_transformer:
            self.transformer = TransformerEncoderBlock(hparams.hidden_dim, hparams.transformer_layers, hparams.transformer_heads, hparams.transformer_ff_dim, hparams.dropout)
        if ablation.use_word_bilstm and ablation.use_transformer and ablation.use_gated_fusion:
            self.fusion = GatedFusion(hparams.hidden_dim)
        fused_dim = hparams.hidden_dim
        self.upos_head = nn.Linear(fused_dim, num_upos)
        self.lemma_head = nn.Linear(fused_dim, num_lemma_rules)
        self.grammeme_head = nn.Linear(fused_dim, num_grammemes)
        if ablation.use_crf:
            self.crf = LinearChainCRF(num_upos)

    def encode(self, word_ids: torch.Tensor, char_ids: torch.Tensor, lengths: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        feats = [self.word_embed(word_ids)]
        if self.ab.use_char_cnn:
            feats.append(self.char_cnn(char_ids))
        if self.ab.use_char_bilstm:
            feats.append(self.char_bilstm(char_ids))
        x = torch.cat(feats, dim=-1)
        x = self.input_dropout(F.gelu(self.input_proj(x)))
        h_lstm = self.bilstm(x, lengths) if self.ab.use_word_bilstm else None
        h_trans = self.transformer(x, key_padding_mask=~mask) if self.ab.use_transformer else None
        if h_lstm is not None and h_trans is not None:
            if self.ab.use_gated_fusion:
                fused = self.fusion(h_lstm, h_trans)
            else:
                fused = h_lstm + h_trans
        elif h_lstm is not None:
            fused = h_lstm
        elif h_trans is not None:
            fused = h_trans
        else:
            raise ValueError('At least one of use_word_bilstm / use_transformer must be True')
        return fused

    def forward(self, word_ids: torch.Tensor, char_ids: torch.Tensor, lengths: torch.Tensor, mask: torch.Tensor, upos_ids: torch.Tensor | None=None) -> dict:
        fused = self.encode(word_ids, char_ids, lengths, mask)
        lemma_logits = self.lemma_head(fused)
        grammeme_logits = self.grammeme_head(fused)
        upos_emissions = self.upos_head(fused)
        out = {'lemma_logits': lemma_logits, 'grammeme_logits': grammeme_logits, 'upos_emissions': upos_emissions}
        if self.ab.use_crf:
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