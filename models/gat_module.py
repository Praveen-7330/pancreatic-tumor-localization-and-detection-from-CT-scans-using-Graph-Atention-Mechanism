import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseGATLayer(nn.Module):
    def __init__(self, in_features, out_features, heads=4, dropout=0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads
        self.head_dim = out_features // heads
        self.linear_src = nn.Linear(in_features, out_features, bias=False)
        self.linear_dst = nn.Linear(in_features, out_features, bias=False)
        self.attn_vec = nn.Parameter(torch.Tensor(1, heads, 1, self.head_dim))
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        self.res_proj = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()
        self.norm = nn.LayerNorm(out_features)
        nn.init.xavier_uniform_(self.attn_vec)

    def forward(self, h, edge_index):
        B, N, C = h.shape
        src_nodes, dst_nodes = edge_index[0], edge_index[1]
        h_src = self.linear_src(h).view(B, N, self.heads, self.head_dim).transpose(1, 2)
        h_dst = self.linear_dst(h).view(B, N, self.heads, self.head_dim).transpose(1, 2)
        h_src_edge = h_src[:, :, src_nodes, :]
        h_dst_edge = h_dst[:, :, dst_nodes, :]
        edge_attn = self.leaky_relu(h_src_edge + h_dst_edge)
        attn_scores = (edge_attn * self.attn_vec).sum(dim=-1)
        attn_matrix = torch.full((B, self.heads, N, N), -1e9, device=h.device)
        attn_matrix[:, :, src_nodes, dst_nodes] = attn_scores
        attn_weights = F.softmax(attn_matrix, dim=-1)
        attn_weights = self.dropout(attn_weights)
        out = torch.matmul(attn_weights, h_src)
        out = out.transpose(1, 2).contiguous().view(B, N, self.out_features)
        out = self.norm(out + self.res_proj(h))
        return out, attn_weights

class GATModule(nn.Module):
    def __init__(self, in_features=256, hidden_features=128, out_features=256, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        if num_layers == 1:
            self.layers.append(DenseGATLayer(in_features, out_features, heads=num_heads, dropout=dropout))
        else:
            self.layers.append(DenseGATLayer(in_features, hidden_features, heads=num_heads, dropout=dropout))
            for i in range(num_layers - 1):
                layer_out = out_features if i == num_layers - 2 else hidden_features
                self.layers.append(DenseGATLayer(hidden_features, layer_out, heads=num_heads, dropout=dropout))
            
    def forward(self, node_feats, edge_index):
        h = node_feats
        last_attn = None
        for layer in self.layers:
            h, last_attn = layer(h, edge_index)
        return h, last_attn
