import torch
import torch.nn as nn
import math


class ChannelAttention(nn.Module):

    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        self.channels = channels
        self.inter_channels = self.channels // reduction_ratio

        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.mlp = nn.Sequential(
            nn.Conv2d(self.channels, self.inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.inter_channels),
            nn.ReLU(),
            nn.Conv2d(self.inter_channels, self.channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.channels)
        )

        self.sigmoid = nn.Sigmoid()

        # Initialize weights
        self.reset_parameters()

    def reset_parameters(self):

        for m in self.mlp.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):

        # Spatial dimension compression
        maxout = self.maxpool(x)
        avgout = self.avgpool(x)

        # Channel-wise feature transformation
        maxout = self.mlp(maxout)
        avgout = self.mlp(avgout)

        # Attention weight calculation
        attention = self.sigmoid(maxout + avgout)

        return attention