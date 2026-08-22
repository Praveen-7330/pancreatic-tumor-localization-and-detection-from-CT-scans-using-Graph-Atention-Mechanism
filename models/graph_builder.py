import torch
import torch.nn as nn

class GraphBuilder3D(nn.Module):
    def __init__(self, in_channels=256, grid_size=(6, 6, 6), k_neighbors=8):
        super().__init__()
        self.grid_size = grid_size
        self.k_neighbors = k_neighbors
        self.in_channels = in_channels
        self.adaptive_pool = nn.AdaptiveAvgPool3d(grid_size)
        self.pos_proj = nn.Linear(3, in_channels)

    def _generate_3d_coords(self, device):
        Dg, Hg, Wg = self.grid_size
        z = torch.linspace(-1.0, 1.0, Dg, device=device)
        y = torch.linspace(-1.0, 1.0, Hg, device=device)
        x = torch.linspace(-1.0, 1.0, Wg, device=device)
        grid_z, grid_y, grid_x = torch.meshgrid(z, y, x, indexing="ij")
        coords = torch.stack([grid_x, grid_y, grid_z], dim=-1).view(-1, 3)
        return coords

    def forward(self, feature_map):
        B, C, D, H, W = feature_map.shape
        pooled = self.adaptive_pool(feature_map)
        Dg, Hg, Wg = self.grid_size
        N = Dg * Hg * Wg
        node_feats = pooled.view(B, C, N).permute(0, 2, 1)
        coords = self._generate_3d_coords(feature_map.device)
        pos_emb = self.pos_proj(coords).unsqueeze(0)
        node_feats = node_feats + pos_emb
        dist_matrix = torch.cdist(coords, coords)
        _, topk_indices = torch.topk(dist_matrix, k=self.k_neighbors + 1, largest=False, dim=-1)
        src = torch.arange(N, device=feature_map.device).unsqueeze(1).repeat(1, self.k_neighbors).view(-1)
        dst = topk_indices[:, 1:].contiguous().view(-1)
        edge_index = torch.stack([src, dst], dim=0)
        return node_feats, edge_index, (D, H, W)
