import torch
import torch.nn as nn
from models.cnn_encoder import CNNEncoder3D
from models.graph_builder import GraphBuilder3D
from models.gat_module import GATModule
from models.feature_fusion import GraphToVolumeProjection, FeatureFusionModule
from models.decoder import Decoder3D

class PancreasGATUNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        num_classes=3,
        feature_channels=(32, 64, 128, 256),
        bottleneck_dim=256,
        grid_size=(6, 6, 6),
        gat_hidden_dim=128,
        gat_heads=4,
        gat_layers=2,
        k_neighbors=8,
        dropout=0.1
    ):
        super().__init__()
        self.encoder = CNNEncoder3D(in_channels=in_channels, channels=feature_channels)
        self.graph_builder = GraphBuilder3D(in_channels=bottleneck_dim, grid_size=grid_size, k_neighbors=k_neighbors)
        self.gat = GATModule(in_features=bottleneck_dim, hidden_features=gat_hidden_dim, out_features=bottleneck_dim, num_heads=gat_heads, num_layers=gat_layers, dropout=dropout)
        self.projection = GraphToVolumeProjection(in_features=bottleneck_dim, grid_size=grid_size)
        self.fusion = FeatureFusionModule(cnn_channels=bottleneck_dim, graph_channels=bottleneck_dim, out_channels=bottleneck_dim)
        self.decoder = Decoder3D(bottleneck_channels=bottleneck_dim, skip_channels=feature_channels[::-1], num_classes=num_classes)

    def forward(self, x):
        bottleneck, skips = self.encoder(x)
        node_feats, edge_index, orig_shape = self.graph_builder(bottleneck)
        updated_nodes, attn_weights = self.gat(node_feats, edge_index)
        graph_vol = self.projection(updated_nodes, orig_shape)
        fused_bottleneck = self.fusion(bottleneck, graph_vol)
        seg_logits, loc_heatmap = self.decoder(fused_bottleneck, skips)
        return seg_logits, loc_heatmap, attn_weights
