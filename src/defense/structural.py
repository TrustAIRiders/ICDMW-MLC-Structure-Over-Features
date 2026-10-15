import os

import torch
from torch_geometric.data import Data
from torch_scatter import scatter_max

from src.attacks.node_injection import LicitDistribution
from src.train.utils import atomic_save


STRATEGIES = ("random", "degree", "mimicry")


class LicitStats:

    KEYS = ("node_mean", "node_std", "node_min", "node_max",
            "edge_mean", "edge_std", "edge_min", "edge_max")

    def __init__(self, tensors, device=None):
        for k in self.KEYS:
            t = tensors[k]
            setattr(self, k, t.to(device) if device is not None else t)
        self.node_centroid = self.node_mean.clone()

    @classmethod
    def from_dataset(cls, train_dataset, cache_path=None, device=None):
        if cache_path and os.path.exists(cache_path):
            tensors = torch.load(cache_path, map_location="cpu", weights_only=True)
            return cls(tensors, device)

        dist = LicitDistribution(train_dataset)
        tensors = {k: getattr(dist, k).cpu() for k in cls.KEYS}
        if cache_path:
            atomic_save(tensors, cache_path)
        return cls(tensors, device)

    def sample_nodes(self, n):
        noise = torch.randn(n, self.node_mean.numel(), device=self.node_mean.device)
        return (self.node_mean + noise * self.node_std).clamp(self.node_min, self.node_max)

    def sample_edges(self, n):
        noise = torch.randn(n, self.edge_mean.numel(), device=self.edge_mean.device)
        return (self.edge_mean + noise * self.edge_std).clamp(self.edge_min, self.edge_max)

    def centroid_nodes(self, n):
        return self.node_centroid.unsqueeze(0).expand(n, -1).clone()


def n_graphs(data):
    if hasattr(data, "num_graphs"):
        try:
            return int(data.num_graphs)
        except (TypeError, ValueError):
            pass
    return int(data.batch.max().item()) + 1


def _node_degrees(edge_index, num_nodes):
    if edge_index.numel() == 0:
        return torch.zeros(num_nodes, device=edge_index.device)
    return torch.bincount(edge_index.reshape(-1), minlength=num_nodes).float()


def _uniform_targets(ptr, counts, new_graph, connections):
    u = torch.rand(new_graph.numel(), connections, device=new_graph.device)
    offset = (u * counts[new_graph].unsqueeze(1).float()).long()
    targets = ptr[new_graph].unsqueeze(1) + offset
    return torch.minimum(targets, (ptr[new_graph + 1] - 1).unsqueeze(1))


def _degree_targets(edge_index, batch_vec, new_graph, connections, num_nodes, num_graphs):
    weights = torch.log(_node_degrees(edge_index, num_nodes) + 1.0)
    picks = []
    for _ in range(connections):
        u = torch.rand(num_nodes, device=weights.device).clamp_min(1e-12)
        gumbel = -torch.log(-torch.log(u))
        _, argmax = scatter_max(weights + gumbel, batch_vec, dim=0, dim_size=num_graphs)
        argmax = argmax.clamp(max=num_nodes - 1)
        picks.append(argmax[new_graph])
    return torch.stack(picks, dim=1)


def inject_nodes_batch(batch, stats, k_nodes, connections, strategy="random"):
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}, got '{strategy}'")
    if k_nodes < 1 or connections < 1:
        raise ValueError(f"k_nodes and connections must be >= 1, got {k_nodes}, {connections}")

    device = batch.x.device
    batch_vec = batch.batch
    num_nodes = batch.x.size(0)
    num_graphs = n_graphs(batch)

    counts = torch.bincount(batch_vec, minlength=num_graphs)
    ptr = torch.cat([counts.new_zeros(1), counts.cumsum(0)])

    n_new = num_graphs * k_nodes
    new_graph = torch.arange(num_graphs, device=device).repeat_interleave(k_nodes)

    new_x = stats.centroid_nodes(n_new) if strategy == "mimicry" else stats.sample_nodes(n_new)

    if strategy == "degree":
        targets = _degree_targets(batch.edge_index, batch_vec, new_graph,
                                  connections, num_nodes, num_graphs)
    else:
        targets = _uniform_targets(ptr, counts, new_graph, connections)

    new_ids = num_nodes + torch.arange(n_new, device=device)
    src = new_ids.unsqueeze(1).expand(-1, connections).reshape(-1)
    dst = targets.reshape(-1)

    new_edge_index = torch.cat([torch.stack([src, dst]), torch.stack([dst, src])], dim=1)
    new_edge_attr = stats.sample_edges(new_edge_index.size(1))

    is_original = getattr(batch, "is_original", None)
    if is_original is None:
        is_original = torch.ones(num_nodes, device=device)
    is_original_aug = torch.cat([is_original, torch.zeros(n_new, device=device)])

    out = Data(
        x=torch.cat([batch.x, new_x], dim=0),
        edge_index=torch.cat([batch.edge_index, new_edge_index], dim=1),
        edge_attr=torch.cat([batch.edge_attr, new_edge_attr], dim=0),
        y=batch.y,
        is_original=is_original_aug,
        batch=torch.cat([batch_vec, new_graph]),
    )
    out.num_graphs = num_graphs
    return out


def sample_structural_params(rng, k_choices, conn_choices, strategies):
    return {
        "k_nodes": int(rng.choice(k_choices)),
        "connections": int(rng.choice(conn_choices)),
        "strategy": str(rng.choice(strategies)),
    }


def perturb_structural(batch, stats, params):
    return inject_nodes_batch(batch, stats, **params)
