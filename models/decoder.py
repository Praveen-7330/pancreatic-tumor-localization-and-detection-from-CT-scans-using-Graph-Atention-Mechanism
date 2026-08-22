import torch
import torch.nn as nn
import torch.nn.functional as F

class UpBlock3D(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv3d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.PReLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.PReLU()
        )

    def forward(self, x, skip):
        x_up = self.up(x)
        if x_up.shape[2:] != skip.shape[2:]:
            x_up = F.interpolate(x_up, size=skip.shape[2:], mode="trilinear", align_corners=False)
        concat = torch.cat([x_up, skip], dim=1)
        return self.conv(concat)

class Decoder3D(nn.Module):
    def __init__(self, bottleneck_channels=256, skip_channels=(256, 128, 64, 32), num_classes=3):
        super().__init__()
        self.up4 = UpBlock3D(bottleneck_channels, skip_channels[0], 256)
        self.up3 = UpBlock3D(256, skip_channels[1], 128)
        self.up2 = UpBlock3D(128, skip_channels[2], 64)
        self.up1 = UpBlock3D(64, skip_channels[3], 32)
        self.seg_head = nn.Conv3d(32, num_classes, kernel_size=1)
        self.loc_head = nn.Sequential(nn.Conv3d(32, 1, kernel_size=1), nn.Sigmoid())

    def forward(self, bottleneck, skips):
        x0, x1, x2, x3 = skips
        d4 = self.up4(bottleneck, x3)
        d3 = self.up3(d4, x2)
        d2 = self.up2(d3, x1)
        d1 = self.up1(d2, x0)
        seg_logits = self.seg_head(d1)
        loc_heatmap = self.loc_head(d1)
        return seg_logits, loc_heatmap
