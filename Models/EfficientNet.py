import torch

from torch import nn

basic_mb_params = [
    # k, channels(c), repeats(t), stride(s), kernel_size(k)
    [1, 16, 1, 1, 3],
    [6, 24, 2, 2, 3],
    [6, 40, 2, 2, 5],
    [6, 80, 3, 2, 3],
    [6, 112, 3, 1, 5],
    [6, 192, 4, 2, 5],
    [6, 320, 1, 1, 3],
]

alpha, beta = 1.2, 1.1

scale_values = {
    # (phi, resolution, dropout)
    "b0": (0, 224, 0.2),
    "b1": (0.5, 240, 0.2),
    "b2": (1, 260, 0.3),
    "b3": (2, 300, 0.3),
    "b4": (3, 380, 0.4),
    "b5": (4, 456, 0.4),
    "b6": (5, 528, 0.5),
    "b7": (6, 600, 0.5),
}

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1):
        super(ConvBlock, self).__init__()
        self.cnnblock = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,kernel_size, stride, padding, groups=groups),
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
        )
    def forward(self, x):
        return self.cnnblock(x)

class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduced_dim):
        super(SqueezeExcitation, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, reduced_dim, 1).
            nn.SiLU(),
            nn.Conv2d(reduced_dim, in_channels,1),
            nn.Sigmoid(),
        )
class MBBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, ratio, reduction=2):
        super(MBBlock, self).__init__()

        hidden_dim = in_channels * ratio
        self.exapnd = in_channels != hidden_dim

        reduced_dim = int((in_channels / reduction))

        if self.expand:
            self.exoand.conv = ConvBlock(in_channels, hidden_dim, kernel_Size = 3, sride=1, padding = 1)
        
        self.conv = nn.Sequential(
            ConvBlock(hidden_dim,hidden_dim, kernel_size, stride, padding, groupps=hidden_dim),
            SqueezeExcitation(hidden_dim, reduced_dim),
            nn.Conv2d(hidden_dim, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )
