# -*- coding: utf-8 -*-
"""
AVSD model (speaker cap 5), v2.

Key changes vs. the previous version:
  1. The visual encoder is TRAINED by default. Freezing is only allowed when
     pretrained visual weights are loaded (a frozen random encoder produces
     noise, which was the cause of the training-loss plateau).
  2. Only speakers that are actually present (speaker_mask == 1) go through the
     visual encoder and fusion. Dummy all-zero speakers no longer pollute the
     BatchNorm statistics or waste memory/compute.
  3. Video is standardized ((x - mean) / std) before the encoder.
  4. Audio front-end keeps frequency structure (strided over frequency, then
     flattened) instead of averaging all 40 mel bands into one value per channel.
  5. Fusion uses casa_net_v2: positional encoding, stacked cross-attention
     blocks with FFNs, self-attention, BiLSTM and one shared output head.

Interface is unchanged:
    model(audio_fea [B, 40, T], audio_embedding [B, N, 100] (unused),
          video_fea [B, N, T, 1, H, W], nframes, mask [B, T], speaker_mask [B, N])
    -> logits [B, T, N]
"""
import os
import sys

import torch
import torch.nn as nn
import torch.utils.checkpoint as torch_checkpoint

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

try:
    from vsd_net import ASD_Wrapper_Visual_Encoder
    from casa_net_v2 import CASA_Net_AVSD_v2
except ImportError:
    from .vsd_net import ASD_Wrapper_Visual_Encoder
    from .casa_net_v2 import CASA_Net_AVSD_v2

# Logit written for absent speaker slots (sigmoid(-10) ~ 0, i.e. "not speaking").
ABSENT_SPEAKER_LOGIT = -10.0

# Grayscale lip-ROI statistics commonly used for LRS-style crops (inputs in [0, 1]).
VIDEO_MEAN = 0.421
VIDEO_STD = 0.165


class CNN2D_BN_Relu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=(1, 1)):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Audio_Feature_Extractor(nn.Module):
    """[B, F, T] fbank -> [B, T, audio_output_dim]. Time resolution is preserved."""

    def __init__(self, input_dim=40, audio_output_dim=256):
        super().__init__()
        self.input_norm = nn.BatchNorm1d(input_dim)
        # stride (2, 1): halve frequency, keep time. 40 -> 20 -> 10 -> 5 -> 3
        self.cnn_layers = nn.Sequential(
            CNN2D_BN_Relu(1, 32, 3, (2, 1)),
            CNN2D_BN_Relu(32, 64, 3, (2, 1)),
            CNN2D_BN_Relu(64, 128, 3, (2, 1)),
            CNN2D_BN_Relu(128, 128, 3, (2, 1)),
        )
        freq_out = input_dim
        for _ in range(4):
            freq_out = (freq_out + 1) // 2
        self.output_proj = nn.Linear(128 * freq_out, audio_output_dim)

    def forward(self, x):
        x = self.input_norm(x).unsqueeze(1)          # [B, 1, F, T]
        x = self.cnn_layers(x)                       # [B, C, F', T]
        B, C, Fq, T = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B, T, C * Fq)
        return self.output_proj(x)                   # [B, T, D_a]


class AIVECTOR_CASA_SD(nn.Module):
    """Audio front-end + audio-visual fusion/decoder."""

    def __init__(self, configs):
        super().__init__()
        self.input_size = configs["input_dim"]
        self.output_speaker = configs["output_speaker"]
        self.audio_dim = configs.get("audio_output_dim", 256)
        self.video_dim = configs.get("video_embedding_dim", 256)

        self.audio_extractor = Audio_Feature_Extractor(self.input_size, self.audio_dim)
        self.casa_net = CASA_Net_AVSD_v2(
            dim_audio=self.audio_dim,
            dim_video=self.video_dim,
            d_model=configs.get("fusion_dim", 256),
            num_heads=configs.get("num_attention_heads", 4),
            num_cross_layers=configs.get("num_cross_layers", 2),
            num_self_layers=configs.get("num_self_layers", 2),
            hidden_dim=configs.get("decoder_hidden_dim", 256),
            dropout=configs.get("dropout", 0.1),
        )

    def encode_audio(self, audio_fea):
        return self.audio_extractor(audio_fea)


class AIVECTOR_ConformerVEmbedding_SD_JOINT(nn.Module):
    def __init__(self, configs, freeze_visual_encoder=False):
        super().__init__()
        print("Initializing CASA-Net AVSD Model (v2)...")

        self.num_speakers = configs["output_speaker"]
        self.visual_grad_checkpoint = configs.get("visual_grad_checkpoint", False)

        self.v_embedding = ASD_Wrapper_Visual_Encoder(num_speakers=self.num_speakers)
        self.av_sd = AIVECTOR_CASA_SD(configs)

        pretrained_path = configs.get("visual_pretrained_path")
        if pretrained_path:
            self.load_visual_weights(pretrained_path)

        if freeze_visual_encoder and not pretrained_path:
            raise ValueError(
                "freeze_visual_encoder=True but no 'visual_pretrained_path' was given. "
                "A frozen, randomly initialized visual encoder outputs noise and the model "
                "cannot tell speakers apart. Train it (pass --train-visual-encoder) or "
                "provide pretrained visual weights."
            )

        self.freeze_visual_encoder = freeze_visual_encoder
        if self.freeze_visual_encoder:
            for p in self.v_embedding.parameters():
                p.requires_grad = False
            self.v_embedding.eval()

        print("Model initialization complete!")

    def load_visual_weights(self, path):
        """Load visual-encoder weights from a checkpoint containing 'v_embedding.*' keys
        (e.g. an earlier AVSD run where the encoder was trained) or a plain
        ASD_Wrapper_Visual_Encoder state dict."""
        ckpt = torch.load(path, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        prefix = "v_embedding."
        if any(k.startswith(prefix) for k in state):
            state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
        missing, unexpected = self.v_embedding.load_state_dict(state, strict=False)
        if missing:
            raise RuntimeError(f"Visual weights at {path} are missing keys: {missing[:5]}...")
        print(f"Loaded visual encoder weights from {path} (unexpected keys ignored: {len(unexpected)})")

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_visual_encoder:
            self.v_embedding.eval()  # keep frozen BatchNorm statistics fixed
        return self

    def _encode_video(self, frames):
        """frames: [M, T, H, W] in [0, 1] -> [M, T, D_v]"""
        x = ((frames - VIDEO_MEAN) / VIDEO_STD).unsqueeze(1)   # [M, 1, T, H, W]
        enc = self.v_embedding.asd_encoder
        if self.visual_grad_checkpoint and self.training and not self.freeze_visual_encoder:
            x = torch_checkpoint.checkpoint(lambda t: enc.pool1(enc.block1(t)), x, use_reentrant=False)
            x = torch_checkpoint.checkpoint(lambda t: enc.pool2(enc.block2(t)), x, use_reentrant=False)
            x = torch_checkpoint.checkpoint(enc.block3, x, use_reentrant=False)
            x = x.transpose(1, 2)
            M, T, C, Hh, Ww = x.shape
            x = enc.maxpool(x.reshape(M * T, C, Hh, Ww)).view(M, T, C)
        else:
            x = enc(x)                                           # [M, T, 128]
        return self.v_embedding.proj(x)                          # [M, T, 256]

    def forward(self, audio_fea, audio_embedding, video_fea, nframes, mask=None, speaker_mask=None):
        # video_fea: [B, N, T, C, H, W] with C == 1
        B, N, T = video_fea.shape[:3]
        device = video_fea.device

        if speaker_mask is None:
            speaker_mask = torch.ones(B, N, device=device)
        if mask is None:
            lengths = torch.as_tensor(
                [int(n) for n in (nframes.tolist() if torch.is_tensor(nframes) else nframes)], device=device
            )
            mask = (torch.arange(T, device=device)[None, :] < lengths[:, None]).float()

        present = speaker_mask > 0.5                              # [B, N]
        b_idx, n_idx = present.nonzero(as_tuple=True)             # M present (session, speaker) pairs

        audio_feats = self.av_sd.encode_audio(audio_fea)          # [B, T, D_a]
        out = torch.full((B, T, N), ABSENT_SPEAKER_LOGIT, device=device, dtype=audio_feats.dtype)
        if b_idx.numel() == 0:
            return out

        frames = video_fea[b_idx, n_idx]                          # [M, T, C, H, W]
        frames = frames[:, :, 0] if frames.shape[2] == 1 else frames.mean(dim=2)

        if self.freeze_visual_encoder:
            with torch.no_grad():
                v = self._encode_video(frames)
        else:
            v = self._encode_video(frames)

        a = audio_feats[b_idx]                                    # [M, T, D_a]
        stream_mask = mask[b_idx]                                 # [M, T]
        key_padding_mask = stream_mask < 0.5
        lengths = stream_mask.sum(dim=1)

        logits = self.av_sd.casa_net(a, v, key_padding_mask, lengths)   # [M, T]
        out = out.to(logits.dtype)
        out[b_idx, :, n_idx] = logits
        return out