# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, dim_q, dim_kv, num_heads=4, dropout=0.1):
        super(MultiHeadCrossAttention, self).__init__()
        assert dim_q % num_heads == 0, "dim_q must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.dim_q = dim_q
        self.dim_kv = dim_kv
        self.head_dim = dim_q // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.W_q = nn.Linear(dim_q, dim_q)
        self.W_k = nn.Linear(dim_kv, dim_q)
        self.W_v = nn.Linear(dim_kv, dim_q)
        
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(dim_q, dim_q)
        
    def forward(self, query, key, value, mask=None):
        BxN, T, _ = query.shape
        
        Q = self.W_q(query).view(BxN, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(key).view(BxN, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(value).view(BxN, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(BxN, T, self.dim_q)
        output = self.out_proj(attn_output)
        
        return output

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv_proj = nn.Linear(dim, dim * 3)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(dim, dim)
        
    def forward(self, x, mask=None):
        BxN, T, _ = x.shape
        
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(BxN, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2]
        
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(BxN, T, self.dim)
        output = self.out_proj(attn_output)
        
        return output

class CASA_Module(nn.Module):
    def __init__(self, dim_audio, dim_video, dim_spk, num_heads=4, dropout=0.1):
        super(CASA_Module, self).__init__()
        
        self.dim_audio = dim_audio
        self.dim_video = dim_video
        self.dim_spk = dim_spk
        self.dim_audio_combined = dim_audio + dim_spk
        
        self.cross_attn_a2v = MultiHeadCrossAttention(
            dim_q=dim_video,
            dim_kv=self.dim_audio_combined,
            num_heads=num_heads,
            dropout=dropout
        )
        self.norm_a2v = nn.LayerNorm(dim_video)
        
        self.cross_attn_v2a = MultiHeadCrossAttention(
            dim_q=self.dim_audio_combined,
            dim_kv=dim_video,
            num_heads=num_heads,
            dropout=dropout
        )
        self.norm_v2a = nn.LayerNorm(self.dim_audio_combined)
        
        dim_concat = dim_video + self.dim_audio_combined
        self.self_attn = MultiHeadSelfAttention(
            dim=dim_concat,
            num_heads=num_heads,
            dropout=dropout
        )
        self.norm_sa = nn.LayerNorm(dim_concat)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, audio_features, video_features, speaker_embeddings):
        B, T, D_A, N = audio_features.shape
        _, _, D_V, _ = video_features.shape
        
        F_a = torch.cat([audio_features, speaker_embeddings], dim=2)
        
        F_a_reshaped = F_a.permute(0, 3, 1, 2).reshape(B*N, T, self.dim_audio_combined)
        F_v_reshaped = video_features.permute(0, 3, 1, 2).reshape(B*N, T, D_V)
        
        F_a2v = self.cross_attn_a2v(query=F_v_reshaped, key=F_a_reshaped, value=F_a_reshaped)
        F_a2v = self.norm_a2v(F_v_reshaped + self.dropout(F_a2v))
        
        F_v2a = self.cross_attn_v2a(query=F_a_reshaped, key=F_v_reshaped, value=F_v_reshaped)
        F_v2a = self.norm_v2a(F_a_reshaped + self.dropout(F_v2a))
        
        F_concat = torch.cat([F_a2v, F_v2a], dim=-1)
        F_sa = self.self_attn(F_concat)
        F_out = self.norm_sa(F_concat + self.dropout(F_sa))
        
        F_out = F_out.reshape(B, N, T, -1).permute(0, 2, 3, 1)
        
        return F_out

class CASA_Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_speakers, dropout=0.2):
        super(CASA_Decoder, self).__init__()
        
        self.num_speakers = num_speakers
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        
        self.speaker_heads = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(num_speakers)
        ])
        
    def forward(self, x, nframes):
        B, T, D, N = x.shape
        
        if isinstance(nframes, torch.Tensor):
            nframes = list(nframes.detach().cpu().numpy())
        
        x_flat = x.permute(0, 3, 1, 2).reshape(B*N*T, D)
        x_flat = self.fc1(x_flat)
        x_flat = self.bn1(x_flat)
        x_flat = self.relu(x_flat)
        x_flat = self.dropout(x_flat)
        
        valid_indices = []
        for batch_idx, m in enumerate(nframes):
            for speaker_idx in range(N):
                for frame_idx in range(m):
                    global_idx = batch_idx * N * T + speaker_idx * T + frame_idx
                    valid_indices.append(global_idx)
        
        x_valid = x_flat[valid_indices, :]
        
        outputs = []
        for speaker_idx in range(N):
            speaker_indices = [i for i, idx in enumerate(valid_indices) 
                             if (idx // T) % N == speaker_idx]
            speaker_features = x_valid[speaker_indices, :]
            speaker_out = self.speaker_heads[speaker_idx](speaker_features)
            outputs.append(speaker_out)
        
        return outputs

class CASA_Net_AVSD(nn.Module):
    def __init__(self, dim_audio, dim_video, dim_spk, num_speakers, 
                 num_heads=4, hidden_dim=256, dropout=0.1):
        super(CASA_Net_AVSD, self).__init__()
        
        fused_dim = dim_audio + dim_video + dim_spk
        
        self.casa_module = CASA_Module(
            dim_audio=dim_audio,
            dim_video=dim_video,
            dim_spk=dim_spk,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.decoder = CASA_Decoder(
            input_dim=fused_dim,
            hidden_dim=hidden_dim,
            num_speakers=num_speakers,
            dropout=dropout
        )
        
    def forward(self, audio_features, video_features, speaker_embeddings, nframes):
        fused_features = self.casa_module(audio_features, video_features, speaker_embeddings)
        outputs = self.decoder(fused_features, nframes)
        return outputs