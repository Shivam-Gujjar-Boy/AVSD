# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

try:
    from vsd_net import ASD_Wrapper_Visual_Encoder
    from casa_net import CASA_Module, CASA_Decoder, CASA_Net_AVSD
except ImportError:
    from .vsd_net import ASD_Wrapper_Visual_Encoder
    from .casa_net import CASA_Module, CASA_Decoder, CASA_Net_AVSD

class LSTM_Projection(nn.Module):
    def __init__(self, input_size, hidden_size, linear_dim, num_layers=1, bidirectional=True, dropout=0):
        super(LSTM_Projection, self).__init__()
        self.hidden_size = hidden_size
        self.LSTM = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=bidirectional, dropout=dropout)
        self.forward_projection = nn.Linear(hidden_size, linear_dim)
        self.backward_projection = nn.Linear(hidden_size, linear_dim)
        self.relu = nn.ReLU(True)

    def forward(self, x, nframes):
        packed_x = nn.utils.rnn.pack_padded_sequence(x, nframes, batch_first=True)
        packed_x_1, hidden = self.LSTM(packed_x)
        x_1, l = nn.utils.rnn.pad_packed_sequence(packed_x_1, batch_first=True)
        forward_projection = self.relu(self.forward_projection(x_1[..., :self.hidden_size]))
        backward_projection = self.relu(self.backward_projection(x_1[..., self.hidden_size:]))
        x_2 = torch.cat((forward_projection, backward_projection), dim=2)
        return x_2

class CNN2D_BN_Relu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(CNN2D_BN_Relu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out

class Audio_Feature_Extractor(nn.Module):
    def __init__(self, input_dim=40, audio_output_dim=256):
        super(Audio_Feature_Extractor, self).__init__()
        
        self.cnn_layers = nn.Sequential(
            CNN2D_BN_Relu(2, 32, 3, 1),
            CNN2D_BN_Relu(32, 64, 3, 1),
            CNN2D_BN_Relu(64, 128, 3, 1),
            CNN2D_BN_Relu(128, 256, 3, 1),
        )
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, None))
        self.output_proj = nn.Conv1d(256, audio_output_dim, 1)
        
    def forward(self, x):
        x = self.cnn_layers(x)
        x = self.adaptive_pool(x).squeeze(2)
        audio_features = self.output_proj(x).transpose(1, 2)
        return audio_features

class AIVECTOR_CASA_SD(nn.Module):
    def __init__(self, configs):
        super(AIVECTOR_CASA_SD, self).__init__()
        self.input_size = configs["input_dim"]
        self.speaker_embedding_size = configs["speaker_embedding_dim"]
        self.output_speaker = configs["output_speaker"]
        
        self.audio_dim = configs.get("audio_output_dim", 256)
        self.audio_extractor = Audio_Feature_Extractor(
            input_dim=self.input_size,
            audio_output_dim=self.audio_dim
        )
        
        self.batchnorm = nn.BatchNorm2d(1)
        self.average_pooling = nn.AvgPool1d(configs["average_pooling"], stride=1, 
                                          padding=configs["average_pooling"]//2)
        
        self.video_dim = configs.get("video_embedding_dim", 256)
        
        self.casa_net = CASA_Net_AVSD(
            dim_audio=self.audio_dim,
            dim_video=self.video_dim, 
            # dim_spk=self.speaker_embedding_size, // For later when considering speaker embeddings
            dim_spk=0,
            num_speakers=self.output_speaker,
            num_heads=configs.get("num_attention_heads", 4),
            hidden_dim=configs.get("decoder_hidden_dim", 256),
            dropout=configs.get("dropout", 0.1)
        )

    def forward(self, x, audio_embedding, video_embedding, nframes):
        batchsize, Freq, Time = x.shape
        N = audio_embedding.shape[1]
        
        if isinstance(nframes, torch.Tensor):
            nframes = list(nframes.detach().cpu().numpy())

        x_3 = self.batchnorm(x.reshape(batchsize, 1, Freq, Time)).squeeze(dim=1)
        x_3_mean = self.average_pooling(x_3)
        x_4 = torch.cat((x_3, x_3_mean), dim=1).reshape(batchsize, 2, Freq, Time)
        
        audio_features = self.audio_extractor(x_4)
        audio_features = audio_features.unsqueeze(-1).expand(-1, -1, -1, N)
        
        video_features = video_embedding.permute(0, 2, 3, 1)
        
        # speaker_embeddings = audio_embedding.unsqueeze(1).expand(-1, Time, -1, -1)
        # speaker_embeddings = speaker_embeddings.permute(0, 1, 3, 2)

        speaker_embeddings = torch.zeros_like(audio_embedding.unsqueeze(1).expand(-1, Time, -1, -1))
        speaker_embeddings = speaker_embeddings.permute(0, 1, 3, 2)
        
        outputs = self.casa_net(audio_features, video_features, speaker_embeddings, nframes)
        
        return outputs

class AIVECTOR_ConformerVEmbedding_SD_JOINT(nn.Module):
    def __init__(self, configs):
        super(AIVECTOR_ConformerVEmbedding_SD_JOINT, self).__init__()
        print("Initializing CASA-Net AVSD Model...")

        self.v_embedding = ASD_Wrapper_Visual_Encoder()
        self.av_sd = AIVECTOR_CASA_SD(configs)

        print("Model initialization complete!")

    def forward(self, audio_fea, audio_embedding, video_fea, nframes):
        B, num_speaker, T, H, W = video_fea.shape
        
        if isinstance(nframes, torch.Tensor):
            nframes = list(nframes.detach().cpu().numpy())
        
        video_fea_reshaped = video_fea.reshape(B*num_speaker, T, H, W)
        _, v_embedding = self.v_embedding(video_fea_reshaped, return_embedding=True)
        v_embedding = v_embedding.reshape(B, num_speaker, T, -1)
        
        av_outputs = self.av_sd(audio_fea, audio_embedding, v_embedding, nframes)
        
        return av_outputs

if __name__=='__main__':
    dummy_configs = {
        "input_dim": 40,
        "average_pooling": 3,
        "speaker_embedding_dim": 100,
        "output_speaker": 4,
        "audio_output_dim": 256,
        "video_embedding_dim": 256,
        "num_attention_heads": 4,
        "decoder_hidden_dim": 256,
        "dropout": 0.1,
    }

    print("Testing CASA-Net AVSD Model...")
    model = AIVECTOR_ConformerVEmbedding_SD_JOINT(dummy_configs)

    B, N, T, Freq = 2, 4, 100, 40
    
    audio_input = torch.randn(B, Freq, T)
    audio_embedding = torch.randn(B, N, 100)
    video_input = torch.randn(B, N, T, 96, 96)
    nframes = [T, T-20]
    
    outputs = model(audio_input, audio_embedding, video_input, nframes)
    
    print(f"Number of speaker outputs: {len(outputs)}")
    for i, out in enumerate(outputs):
        print(f"Speaker {i} output shape: {out.shape}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')