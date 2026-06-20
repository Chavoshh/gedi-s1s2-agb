"""ResNet-18 backbone adapted for 25x25 patches.

Differences from standard ImageNet ResNet-18:
  - Stem: single 3x3 conv stride 1, no maxpool (preserves 25x25 spatial dim).
  - Three residual stages (channels 64 -> 128 -> 256) instead of four.
    The fourth stage (channels 512) is dropped because it would reduce the
    spatial dimension to 1x1, throwing away all spatial information.
  - Output: a feature vector after global average pooling, not a classification
    head. Task-specific heads are defined separately in heads.py.

Output spatial sizes through the network on a 25x25 input:
    Input:   25 x 25 x in_channels
    Stem:    25 x 25 x 64    (conv 3x3 s=1)
    Stage1:  25 x 25 x 64    (2 BasicBlocks, no downsample)
    Stage2:  13 x 13 x 128   (2 BasicBlocks, first strides 2)
    Stage3:  7  x 7  x 256   (2 BasicBlocks, first strides 2)
    Global avg pool: 256-dim feature vector
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding 1, no bias (BN follows)."""
    return nn.Conv2d(
        in_channels, out_channels,
        kernel_size=3, stride=stride, padding=1, bias=False,
    )


def _conv1x1(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution, no bias (BN follows). Used only in downsample shortcut."""
    return nn.Conv2d(
        in_channels, out_channels,
        kernel_size=1, stride=stride, bias=False,
    )


class BasicBlock(nn.Module):
    """Standard ResNet BasicBlock: two 3x3 convs + residual connection.

    If stride > 1 or in_channels != out_channels, the shortcut uses a 1x1 conv
    to match dimensions. Otherwise the shortcut is identity.
    """

    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ):
        super().__init__()
        self.conv1 = _conv3x3(in_channels, out_channels, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(out_channels, out_channels, stride=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.downsample: nn.Module = nn.Sequential(
                _conv1x1(in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + identity)
        return out


class ResNet18Adapted(nn.Module):
    """ResNet-18 adapted for 25x25 input patches.

    Produces a 256-dim feature vector. No task head.

    Args:
        in_channels: number of input channels (5 / 12 / 15 depending on variant).
    """

    feature_dim = 256
    blocks_per_stage = (2, 2, 2)
    stage_channels = (64, 128, 256)

    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels

        # Stem: 3x3 conv, stride 1, no maxpool
        self.stem = nn.Sequential(
            _conv3x3(in_channels, self.stage_channels[0], stride=1),
            nn.BatchNorm2d(self.stage_channels[0]),
            nn.ReLU(inplace=True),
        )

        # Three residual stages
        self.stage1 = self._make_stage(
            in_channels=self.stage_channels[0],
            out_channels=self.stage_channels[0],
            n_blocks=self.blocks_per_stage[0],
            first_stride=1,
        )
        self.stage2 = self._make_stage(
            in_channels=self.stage_channels[0],
            out_channels=self.stage_channels[1],
            n_blocks=self.blocks_per_stage[1],
            first_stride=2,
        )
        self.stage3 = self._make_stage(
            in_channels=self.stage_channels[1],
            out_channels=self.stage_channels[2],
            n_blocks=self.blocks_per_stage[2],
            first_stride=2,
        )

        # Global average pool produces (B, 256, 1, 1); flatten to (B, 256)
        self.global_pool = nn.AdaptiveAvgPool2d(output_size=1)

        self._init_weights()

    @staticmethod
    def _make_stage(
        in_channels: int,
        out_channels: int,
        n_blocks: int,
        first_stride: int,
    ) -> nn.Sequential:
        """Build a stage of n_blocks BasicBlocks.

        First block uses first_stride (may be 2 for downsampling); subsequent
        blocks have stride 1.
        """
        blocks: list[nn.Module] = [
            BasicBlock(in_channels, out_channels, stride=first_stride)
        ]
        for _ in range(1, n_blocks):
            blocks.append(BasicBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*blocks)

    def _init_weights(self) -> None:
        """Kaiming initialization for convs, ones/zeros for BN.

        Standard ResNet initialization. Helps training convergence.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input patches.

        Args:
            x: (B, in_channels, 25, 25) input tensor

        Returns:
            (B, 256) feature vector
        """
        x = self.stem(x)        # (B, 64, 25, 25)
        x = self.stage1(x)      # (B, 64, 25, 25)
        x = self.stage2(x)      # (B, 128, 13, 13)
        x = self.stage3(x)      # (B, 256, 7, 7)
        x = self.global_pool(x) # (B, 256, 1, 1)
        x = torch.flatten(x, 1) # (B, 256)
        return x


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)