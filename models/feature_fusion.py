import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphToVolumeProjection(nn.Module):
    def __init__(self, in_features=256, grid_size=(6, 6, 6)):
        super().__init__()
        self.grid_size = grid_size
        self.in_features = in_features
        
    def forward(self, node_features, target_spatial_shape):
        B, N, C = node_features.shape
        Dg, Hg, Wg = self.grid_size
        vol = node_features.permute(0, 2, 1).view(B, C, Dg, Hg, Wg)
        vol_upsampled = F.interpolate(vol, size=target_spatial_shape, mode="trilinear", align_corners=False)
        return vol_upsampled

class FeatureFusionModule(nn.Module):
    def __init__(self, cnn_channels=256, graph_channels=256, out_channels=256):
        super().__init__()
        self.conv_fuse = nn.Sequential(
            nn.Conv3d(cnn_channels + graph_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.PReLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.PReLU()
        )
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(out_channels, out_channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels // 4, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, cnn_feat, graph_feat):
        concat = torch.cat([cnn_feat, graph_feat], dim=1)
        fused = self.conv_fuse(concat)
        gate = self.channel_gate(fused)
        return fused * gate + cnn_feat
