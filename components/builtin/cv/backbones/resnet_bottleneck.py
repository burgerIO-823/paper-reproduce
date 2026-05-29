"""
ResNet Bottleneck Block

Standard bottleneck residual block from "Deep Residual Learning for Image Recognition"
(He et al., 2015). Used as a building block for ResNet-50/101/152 architectures.

Architecture:
    input → 1x1 conv (reduce) → 3x3 conv → 1x1 conv (expand) → + shortcut → output

The block supports stride > 1 for downsampling and an optional shortcut projection
when input/output channel dimensions differ.
"""

import torch
import torch.nn as nn


class ResNetBottleneck(nn.Module):
    """
    ResNet bottleneck residual block.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        stride: Stride for the 3x3 convolution (default: 1)
        expansion: Channel expansion factor (default: 4, matches ResNet-50/101/152)
        downsample: Optional downsample layer for shortcut projection
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 expansion: int = 4, downsample: nn.Module = None):
        super().__init__()
        hidden_channels = out_channels // expansion

        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.conv3 = nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)
