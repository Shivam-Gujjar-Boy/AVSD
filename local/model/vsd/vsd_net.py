import torch
import torch.nn as nn


class VisualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_1, kernel_2, is_down=False):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)
        self.padding_1 = (kernel_1 - 1) // 2
        self.padding_2 = (kernel_2 - 1) // 2

        # Residual shortcut matching
        if is_down or in_channels != out_channels:
            stride = (1, 2, 2) if is_down else (1, 1, 1)
            self.shortcut = nn.Sequential(
                nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm3d(out_channels, momentum=0.01, eps=0.001),
            )
        else:
            self.shortcut = nn.Identity()

        # Spatial 2D Conv 1
        stride_1 = (1, 2, 2) if is_down else (1, 1, 1)
        self.s_1 = nn.Conv3d(
            in_channels,
            out_channels // 2,
            kernel_size=(1, kernel_1, kernel_1),
            stride=stride_1,
            padding=(0, self.padding_1, self.padding_1),
            bias=False,
        )
        self.s_norm_1 = nn.BatchNorm3d(out_channels // 2, momentum=0.01, eps=0.001)

        # Spatial 2D Conv 2
        self.s_2 = nn.Conv3d(
            out_channels // 2,
            out_channels,
            kernel_size=(1, kernel_2, kernel_2),
            padding=(0, self.padding_2, self.padding_2),
            bias=False,
        )
        self.s_norm_2 = nn.BatchNorm3d(out_channels, momentum=0.01, eps=0.001)

        # Temporal 1D Conv 1
        self.t_1 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=(kernel_1, 1, 1),
            padding=(self.padding_1, 0, 0),
            bias=False,
        )
        self.t_norm_1 = nn.BatchNorm3d(out_channels, momentum=0.01, eps=0.001)

        # Temporal 1D Conv 2
        self.t_2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=(kernel_2, 1, 1),
            padding=(self.padding_2, 0, 0),
            bias=False,
        )
        self.t_norm_2 = nn.BatchNorm3d(out_channels, momentum=0.01, eps=0.001)

    def forward(self, x):
        res = self.shortcut(x)

        x = self.relu(self.s_norm_1(self.s_1(x)))
        x = self.relu(self.s_norm_2(self.s_2(x)))
        x = self.relu(self.t_norm_1(self.t_1(x)))
        x = self.t_norm_2(self.t_2(x))  # No activation right before addition

        # Residual connection
        x = self.relu(x + res)
        return x


class VisualEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.block1 = VisualBlock(1, 32, 5, 3, is_down=True)
        self.pool1 = nn.MaxPool3d(
            kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)
        )

        self.block2 = VisualBlock(32, 64, 5, 3)
        self.pool2 = nn.MaxPool3d(
            kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)
        )

        self.block3 = VisualBlock(64, 128, 5, 3)
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))

        self._init_weight()

    def forward(self, x):
        # x input expected shape: [B, C=1, T, H, W]
        x = self.block1(x)
        x = self.pool1(x)

        x = self.block2(x)
        x = self.pool2(x)

        x = self.block3(x)

        # x is currently [B, C=128, T, H', W']
        # Permute to [B, T, C, H', W'] to bundle time & batch dimensions for spatial maxpool
        x = x.permute(0, 2, 1, 3, 4)
        batch, time, channels, height, width = x.shape
        x = x.reshape(batch * time, channels, height, width)

        x = self.maxpool(x)  # [B * T, 128, 1, 1]
        x = x.view(batch, time, channels)  # [B, T, 128]
        return x

    def _init_weight(self):
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
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
        # Flexible dimension parsing:
        # 4D: [B, T, H, W] -> unsqueeze C -> [B, 1, T, H, W]
        # 5D: [B, T, C, H, W] -> permute -> [B, C, T, H, W]
        # 5D: [B, C, T, H, W] -> pass directly
        if video_inputs.ndim == 4:
            x = video_inputs.unsqueeze(1)  # [B, 1, T, H, W]
        elif video_inputs.ndim == 5:
            # Check if Time is in dimension 1 (e.g. from vsd_collate_fn: [B, T, C, H, W])
            if video_inputs.shape[2] in (1, 3):  # Channel is at dim 2
                x = video_inputs.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
            else:
                x = video_inputs  # Already [B, C, T, H, W]
        else:
            raise ValueError(
                f"Expected 4D or 5D input tensor, got shape {tuple(video_inputs.shape)}"
            )

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