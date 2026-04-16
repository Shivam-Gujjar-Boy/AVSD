import torch
import torch.nn as nn


class VisualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_1, kernel_2, is_down=False):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)
        self.padding_1 = (kernel_1 - 1) // 2
        self.padding_2 = (kernel_2 - 1) // 2

        if is_down:
            self.s_1 = nn.Conv3d(
                in_channels,
                out_channels // 2,
                kernel_size=(1, kernel_1, kernel_1),
                stride=(1, 2, 2),
                padding=(0, self.padding_1, self.padding_1),
                bias=False,
            )
        else:
            self.s_1 = nn.Conv3d(
                in_channels,
                out_channels // 2,
                kernel_size=(1, kernel_1, kernel_1),
                padding=(0, self.padding_1, self.padding_1),
                bias=False,
            )

        self.s_norm_1 = nn.BatchNorm3d(out_channels // 2, momentum=0.01, eps=0.001)

        self.s_2 = nn.Conv3d(
            out_channels // 2,
            out_channels,
            kernel_size=(1, kernel_2, kernel_2),
            padding=(0, self.padding_2, self.padding_2),
            bias=False,
        )
        self.s_norm_2 = nn.BatchNorm3d(out_channels, momentum=0.01, eps=0.001)

        self.t_1 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=(kernel_1, 1, 1),
            padding=(self.padding_1, 0, 0),
            bias=False,
        )
        self.t_norm_1 = nn.BatchNorm3d(out_channels, momentum=0.01, eps=0.001)

        self.t_2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=(kernel_2, 1, 1),
            padding=(self.padding_2, 0, 0),
            bias=False,
        )
        self.t_norm_2 = nn.BatchNorm3d(out_channels, momentum=0.01, eps=0.001)

    def forward(self, x):
        x = self.relu(self.s_norm_1(self.s_1(x)))
        x = self.relu(self.s_norm_2(self.s_2(x)))
        x = self.relu(self.t_norm_1(self.t_1(x)))
        x = self.relu(self.t_norm_2(self.t_2(x)))
        return x


class VisualEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.block1 = VisualBlock(1, 32, 5, 3, is_down=True)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        self.block2 = VisualBlock(32, 64, 5, 3)
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        self.block3 = VisualBlock(64, 128, 5, 3)
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))

        self._init_weight()

    def forward(self, x):
        x = self.block1(x)
        x = self.pool1(x)

        x = self.block2(x)
        x = self.pool2(x)

        x = self.block3(x)

        x = x.transpose(1, 2)
        batch, time, channels, width, height = x.shape
        x = x.reshape(batch * time, channels, width, height)

        x = self.maxpool(x)
        x = x.view(batch, time, channels)
        return x

    def _init_weight(self):
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight)
            elif isinstance(module, nn.BatchNorm3d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


class VSDNet(nn.Module):
    """Visual-only diarization model for per-frame speaking activity."""

    def __init__(self, embedding_dim=256):
        super().__init__()
        self.encoder = VisualEncoder()
        self.proj = nn.Linear(128, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, 1)

    def forward(self, video_inputs, return_embedding=False):
        # video_inputs: [B, T, H, W]
        if video_inputs.ndim != 4:
            raise ValueError(f"Expected [B, T, H, W], got shape {tuple(video_inputs.shape)}")

        x = video_inputs.unsqueeze(1)  # [B, 1, T, H, W]
        embeddings = self.encoder(x)  # [B, T, 128]
        embeddings = self.proj(embeddings)  # [B, T, embedding_dim]
        logits = self.classifier(embeddings).squeeze(-1)  # [B, T]

        if return_embedding:
            return logits, embeddings
        return logits


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
