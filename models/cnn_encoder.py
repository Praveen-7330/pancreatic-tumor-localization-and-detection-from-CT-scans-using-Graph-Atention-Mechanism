import torch
import torch.nn as nn

class ConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.PReLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.PReLU()
        )
        self.residual = (
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x):
        return self.conv(x) + self.residual(x)

class CNNEncoder3D(nn.Module):
    def __init__(self, in_channels=1, channels=(32, 64, 128, 256)):
        super().__init__()
        self.channels = channels
        self.stage0 = ConvBlock3D(in_channels, channels[0])
        self.down1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.stage1 = ConvBlock3D(channels[0], channels[1])
        self.down2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.stage2 = ConvBlock3D(channels[1], channels[2])
        self.down3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.stage3 = ConvBlock3D(channels[2], channels[3])
        self.down4 = nn.MaxPool3d(kernel_size=2, stride=2)
        
    def forward(self, x):
        x0 = self.stage0(x)
        x1 = self.stage1(self.down1(x0))
        x2 = self.stage2(self.down2(x1))
        x3 = self.stage3(self.down3(x2))
        bottleneck = self.down4(x3)
        return bottleneck, [x0, x1, x2, x3]
