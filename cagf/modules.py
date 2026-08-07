from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CharCNNEncoder(nn.Module):

    def __init__(self, num_chars: int, char_emb_dim: int=32, num_filters: int=64, kernel_sizes: tuple[int, ...]=(2, 3, 4), pad_id: int=0, dropout: float=0.1):
        super().__init__()
        self.char_embed = nn.Embedding(num_chars, char_emb_dim, padding_idx=pad_id)
        self.convs = nn.ModuleList([nn.Conv1d(char_emb_dim, num_filters, kernel_size=k, padding=k // 2) for k in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.output_dim = num_filters * len(kernel_sizes)

    def forward(self, char_ids: torch.Tensor) -> torch.Tensor:
        b, t, w = char_ids.shape
        x = char_ids.view(b * t, w)
        emb = self.char_embed(x)
        emb = self.dropout(emb).transpose(1, 2)
        pooled = []
        for conv in self.convs:
            c = F.relu(conv(emb))
            p = F.max_pool1d(c, kernel_size=c.size(-1)).squeeze(-1)
            pooled.append(p)
        out = torch.cat(pooled, dim=-1)
        return out.view(b, t, self.output_dim)

class CharBiLSTMEncoder(nn.Module):

    def __init__(self, num_chars: int, char_emb_dim: int=32, hidden_dim: int=64, pad_id: int=0, dropout: float=0.1):
        super().__init__()
        self.char_embed = nn.Embedding(num_chars, char_emb_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(char_emb_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.output_dim = hidden_dim * 2

    def forward(self, char_ids: torch.Tensor) -> torch.Tensor:
        b, t, w = char_ids.shape
        x = char_ids.view(b * t, w)
        lengths = (x != 0).sum(dim=1).clamp(min=1)
        emb = self.dropout(self.char_embed(x))
        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        out = torch.cat([h_n[0], h_n[1]], dim=-1)
        return out.view(b, t, self.output_dim)

class WordBiLSTMEncoder(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int=256, num_layers: int=1, dropout: float=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0.0)
        self.output_dim = hidden_dim * 2

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=x.size(1))
        return out

class PositionalEncoding(nn.Module):

    def __init__(self, dim: int, max_len: int=512):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

class TransformerEncoderBlock(nn.Module):

    def __init__(self, input_dim: int, num_layers: int=2, num_heads: int=4, ff_dim: int=512, dropout: float=0.1):
        super().__init__()
        self.pos_enc = PositionalEncoding(input_dim)
        layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout, batch_first=True, activation='gelu')
        # enable_nested_tensor=False disables the fast-path that routes padding
        # through aten::_nested_tensor_from_mask_left_aligned -- an op not yet
        # implemented on MPS (torch 2.13). With it off, src_key_padding_mask
        # still works correctly on every backend (MPS/CUDA/CPU).
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        self.output_dim = input_dim

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        x = self.pos_enc(x)
        return self.encoder(x, src_key_padding_mask=key_padding_mask)

class GatedFusion(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim * 2, dim)
        self.output_dim = dim

    def forward(self, h_lstm: torch.Tensor, h_trans: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate_proj(torch.cat([h_lstm, h_trans], dim=-1)))
        return gate * h_lstm + (1 - gate) * h_trans