import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, SAGEConv, GINConv, global_mean_pool, global_max_pool
from torch_scatter import scatter_add, scatter_max, scatter_mean


class EdgeProjection(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU()
        )

    def forward(self, edge_attr):
        return self.proj(edge_attr)


class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.att = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x, batch, node_weights=None):
        att_scores = self.att(x)

        if node_weights is not None:
            att_scores = att_scores + torch.log(node_weights + 1e-8)

        max_scores, _ = scatter_max(att_scores, batch, dim=0)
        att_scores = att_scores - max_scores[batch]
        att_weights = torch.exp(att_scores)

        sum_weights = scatter_add(att_weights, batch, dim=0)[batch]
        att_weights = att_weights / (sum_weights + 1e-8)

        weighted_x = x * att_weights
        out = scatter_add(weighted_x, batch, dim=0)

        return out


class GATv2Block(nn.Module):
    def __init__(self, in_dim, out_dim, heads, edge_dim, dropout):
        super().__init__()
        assert out_dim % heads == 0

        self.conv = GATv2Conv(
            in_channels=in_dim,
            out_channels=out_dim // heads,
            heads=heads,
            edge_dim=edge_dim,
            dropout=dropout,
            concat=True
        )

        self.norm = nn.LayerNorm(out_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr):
        x = self.conv(x, edge_index, edge_attr)
        x = self.norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class SAGEBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout):
        super().__init__()
        self.conv = SAGEConv(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr=None):
        x = self.conv(x, edge_index)
        x = self.norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class SAGEEdgeBlock(nn.Module):
    def __init__(self, in_dim, edge_dim, out_dim, dropout):
        super().__init__()
        self.conv = SAGEConv(in_dim + edge_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr):
        target_nodes = edge_index[1]
        agg_edge = scatter_mean(edge_attr, target_nodes, dim=0, dim_size=x.size(0))
        x_cat = torch.cat([x, agg_edge], dim=-1)

        x_out = self.conv(x_cat, edge_index)
        x_out = self.norm(x_out)
        x_out = self.relu(x_out)
        x_out = self.dropout(x_out)
        return x_out


class GINBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout):
        super().__init__()
        mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )
        self.conv = GINConv(mlp, train_eps=True)
        self.norm = nn.LayerNorm(out_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr=None):
        x = self.conv(x, edge_index)
        x = self.norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class EllipticGNN(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, hidden_dim, num_layers,
                 heads, edge_proj_dim, num_classes, dropout, conv_type="gatv2",
                 expansion_node_weight=1.0):
        super().__init__()
        assert num_layers >= 1

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.conv_type = conv_type
        self.expansion_node_weight = expansion_node_weight

        self.uses_edge_features = conv_type in ("gatv2", "sage_edge")
        assert conv_type in ("gatv2", "sage", "sage_edge", "gin"), f"Unknown conv_type: {conv_type}"
        if self.uses_edge_features:
            self.edge_proj = EdgeProjection(edge_feat_dim, edge_proj_dim)
        else:
            self.edge_proj = None

        self.input_proj = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        self.conv_layers = nn.ModuleList()
        for _ in range(num_layers):
            if conv_type == "gatv2":
                assert hidden_dim % heads == 0
                self.conv_layers.append(
                    GATv2Block(
                        in_dim=hidden_dim,
                        out_dim=hidden_dim,
                        heads=heads,
                        edge_dim=edge_proj_dim,
                        dropout=dropout
                    )
                )
            elif conv_type == "sage":
                self.conv_layers.append(
                    SAGEBlock(
                        in_dim=hidden_dim,
                        out_dim=hidden_dim,
                        dropout=dropout
                    )
                )
            elif conv_type == "sage_edge":
                self.conv_layers.append(
                    SAGEEdgeBlock(
                        in_dim=hidden_dim,
                        edge_dim=edge_proj_dim,
                        out_dim=hidden_dim,
                        dropout=dropout
                    )
                )
            elif conv_type == "gin":
                self.conv_layers.append(
                    GINBlock(
                        in_dim=hidden_dim,
                        out_dim=hidden_dim,
                        dropout=dropout
                    )
                )

        self.jk_proj = nn.Sequential(
            nn.Linear(num_layers * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        self.att_pool = AttentionPooling(hidden_dim)
        self.readout_dim = hidden_dim * 3

        self.classifier = nn.Sequential(
            nn.Linear(self.readout_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),

            nn.Linear(hidden_dim // 4, num_classes),
        )

    def compute_node_weights(self, data):
        if self.expansion_node_weight >= 1.0 or not hasattr(data, "is_original"):
            return None
        is_orig = data.is_original
        weights = torch.where(is_orig > 0.5,
                              torch.ones_like(is_orig),
                              torch.full_like(is_orig, self.expansion_node_weight))
        return weights.unsqueeze(-1)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch

        node_weights = self.compute_node_weights(data)

        if self.uses_edge_features:
            edge_attr = self.edge_proj(edge_attr)

        x = self.input_proj(x)

        layer_outputs = []
        for conv_block in self.conv_layers:
            x_new = conv_block(x, edge_index, edge_attr)
            x = x + x_new
            layer_outputs.append(x)

        x = self.jk_proj(torch.cat(layer_outputs, dim=-1))

        x_att = self.att_pool(x, batch, node_weights)
        if node_weights is not None:
            wx = x * node_weights
            x_mean = scatter_add(wx, batch, dim=0) / (scatter_add(node_weights, batch, dim=0) + 1e-8)
        else:
            x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_pooled = torch.cat((x_att, x_mean, x_max), dim=1)

        logits = self.classifier(x_pooled)

        return logits

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
