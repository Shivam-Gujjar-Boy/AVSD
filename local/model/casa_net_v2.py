# -*- coding: utf-8 -*-
"""
CASA-Net v2: audio-visual fusion for target-speaker activity detection.

Operates on one (audio, single-speaker-video) stream at a time:
    audio : [M, T, D_a]   (same session audio, repeated for each present speaker)
    video : [M, T, D_v]   (lip embeddings of that speaker)
    -> logits [M, T]

Changes vs. casa_net.py (v1):
  * sinusoidal positional encoding, so attention knows which audio frame
    lines up with which video frame;
  * pre-LayerNorm cross-attention blocks WITH feed-forward sublayers, stacked;
  * a self-attention encoder over the fused stream;
  * a BiLSTM over time to model speaker-turn dynamics;
  * ONE shared output head (speaker slots carry no identity across sessions);
  * LayerNorm everywhere instead of BatchNorm, and boolean key-padding masks
    (no -1e9 constants, so fp16/bf16 safe).

casa_net.py is left untouched because avsd_net.py / avsd_net_spk4.py /
avsd_net_spk6.py still import its classes.
"""
import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # x: [M, T, D]
        T = x.shape[1]
        position = torch.arange(T, device=x.device, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.dim, 2, device=x.device, dtype=torch.float32)
            * (-math.log(10000.0) / self.dim)
        )
        pe = torch.zeros(T, self.dim, device=x.device, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return x + pe.unsqueeze(0).to(x.dtype)


class FeedForward(nn.Module):
    def __init__(self, dim, ff_mult=4, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class CrossModalBlock(nn.Module):
    """Bidirectional cross-attention (audio<->video) + FFN, pre-LayerNorm."""

    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.norm_a = nn.LayerNorm(dim)
        self.norm_v = nn.LayerNorm(dim)
        self.attn_v_from_a = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.attn_a_from_v = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)

        self.ff_norm_a = nn.LayerNorm(dim)
        self.ff_norm_v = nn.LayerNorm(dim)
        self.ff_a = FeedForward(dim, dropout=dropout)
        self.ff_v = FeedForward(dim, dropout=dropout)

    def forward(self, a, v, key_padding_mask=None):
        a_n, v_n = self.norm_a(a), self.norm_v(v)
        # Both directions read the block's inputs, so the update is symmetric.
        v_upd, _ = self.attn_v_from_a(v_n, a_n, a_n, key_padding_mask=key_padding_mask, need_weights=False)
        a_upd, _ = self.attn_a_from_v(a_n, v_n, v_n, key_padding_mask=key_padding_mask, need_weights=False)
        v = v + self.drop(v_upd)
        a = a + self.drop(a_upd)
        v = v + self.ff_v(self.ff_norm_v(v))
        a = a + self.ff_a(self.ff_norm_a(a))
        return a, v


class SharedSpeakerHead(nn.Module):
    """Same classifier for every speaker stream."""

    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class CASA_Net_AVSD_v2(nn.Module):
    def __init__(
        self,
        dim_audio,
        dim_video,
        d_model=256,
        num_heads=4,
        num_cross_layers=2,
        num_self_layers=2,
        hidden_dim=256,
        dropout=0.1,
    ):
        super().__init__()
        self.audio_in = nn.Linear(dim_audio, d_model) if dim_audio != d_model else nn.Identity()
        self.video_in = nn.Linear(dim_video, d_model) if dim_video != d_model else nn.Identity()
        self.pos_enc = SinusoidalPositionalEncoding(d_model)
        self.input_drop = nn.Dropout(dropout)

        self.cross_layers = nn.ModuleList(
            [CrossModalBlock(d_model, num_heads, dropout) for _ in range(num_cross_layers)]
        )

        self.fuse = nn.Sequential(nn.LayerNorm(2 * d_model), nn.Linear(2 * d_model, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.self_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_self_layers, enable_nested_tensor=False
        )

        assert d_model % 2 == 0
        self.blstm = nn.LSTM(d_model, d_model // 2, num_layers=1, batch_first=True, bidirectional=True)
        self.post_lstm_norm = nn.LayerNorm(d_model)

        self.head = SharedSpeakerHead(d_model, hidden_dim, dropout)

    def forward(self, audio, video, key_padding_mask=None, lengths=None):
        """
        audio            : [M, T, D_a]
        video            : [M, T, D_v]
        key_padding_mask : [M, T] bool, True = padded frame (ignored)
        lengths          : [M] valid frame counts (needed only when there is padding)
        returns logits   : [M, T]
        """
        a = self.input_drop(self.pos_enc(self.audio_in(audio)))
        v = self.input_drop(self.pos_enc(self.video_in(video)))

        if key_padding_mask is not None and not bool(key_padding_mask.any()):
            key_padding_mask = None  # no padding in this batch -> skip masking entirely

        for layer in self.cross_layers:
            a, v = layer(a, v, key_padding_mask)

        x = self.fuse(torch.cat([a, v], dim=-1))
        x = self.self_encoder(x, src_key_padding_mask=key_padding_mask)

        T = x.shape[1]
        if key_padding_mask is not None and lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.detach().cpu().long(), batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.blstm(packed)
            x_lstm, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=T)
        else:
            x_lstm, _ = self.blstm(x)
        x = self.post_lstm_norm(x + x_lstm)

        return self.head(x)