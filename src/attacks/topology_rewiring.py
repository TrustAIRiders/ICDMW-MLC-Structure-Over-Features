from __future__ import annotations

from collections import deque

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data

from src.attacks.node_injection import LicitDistribution


def _pyg_to_nx(data) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(range(data.num_nodes))
    ei = data.edge_index.cpu().numpy()
    for u, v in zip(ei[0].tolist(), ei[1].tolist()):
        if u != v:
            G.add_edge(u, v)
    return G


def _diameter_path(G: nx.Graph) -> list[int]:
    if G.number_of_nodes() == 0:
        return []

    lcc_nodes = max(nx.connected_components(G), key=len)
    sub = G.subgraph(lcc_nodes)

    if sub.number_of_edges() == 0:
        return list(sub.nodes())[:1]

    def _bfs_farthest(source: int) -> int:
        dist: dict[int, int] = {source: 0}
        queue: deque[int] = deque([source])
        farthest = source
        while queue:
            node = queue.popleft()
            for nb in sub.neighbors(node):
                if nb not in dist:
                    dist[nb] = dist[node] + 1
                    queue.append(nb)
                    if dist[nb] > dist[farthest]:
                        farthest = nb
        return farthest

    start = next(iter(sub.nodes()))
    u = _bfs_farthest(start)
    v = _bfs_farthest(u)

    if u == v:
        return [u]

    return nx.shortest_path(sub, u, v)


class TopologyAnalyzer:

    @staticmethod
    def compute_metrics(data) -> dict:
        G = _pyg_to_nx(data)
        n = G.number_of_nodes()
        e = G.number_of_edges()

        if n == 0:
            return {
                "num_nodes":      0,
                "num_edges":      0,
                "diameter":       0,
                "avg_clustering": 0.0,
                "avg_degree":     0.0,
            }

        diameter = 0
        if e > 0:
            lcc = G.subgraph(max(nx.connected_components(G), key=len))
            if lcc.number_of_nodes() > 1:
                diameter = nx.diameter(lcc)

        avg_clustering = nx.average_clustering(G) if e > 0 else 0.0
        avg_degree     = 2.0 * e / n

        return {
            "num_nodes":      n,
            "num_edges":      e,
            "diameter":       diameter,
            "avg_clustering": round(avg_clustering, 6),
            "avg_degree":     round(avg_degree, 6),
        }

    @staticmethod
    def compare(clean: dict, perturbed: dict) -> dict:

        return {
            "delta_nodes":      perturbed["num_nodes"]      - clean["num_nodes"],
            "delta_edges":      perturbed["num_edges"]      - clean["num_edges"],
            "delta_diameter":   perturbed["diameter"]       - clean["diameter"],
            "delta_clustering": round(
                perturbed["avg_clustering"] - clean["avg_clustering"], 6
            ),
            "delta_avg_degree": round(
                perturbed["avg_degree"] - clean["avg_degree"], 6
            ),
        }


class TopologyAugmentationAttack:

    TECHNIQUES = ("chain_injection", "star", "parallel")

    def __init__(
        self,
        licit_dist: LicitDistribution,
        technique: str,
        k_intermediates: int,
        injected_is_original: float = 0.0,
    ) -> None:
        if technique not in self.TECHNIQUES:
            raise ValueError(
                f"technique must be one of {self.TECHNIQUES}, got '{technique}'"
            )
        if k_intermediates < 1:
            raise ValueError(f"k_intermediates must be ≥ 1, got {k_intermediates}")

        self.dist      = licit_dist
        self.technique = technique
        self.k         = k_intermediates
        self.injected_is_original = float(injected_is_original)


    def attack(self, data) -> Data:

        if self.technique == "chain_injection":
            return self._chain_injection(data)
        elif self.technique == "star":
            return self._star_injection(data)
        else:
            return self._parallel_paths(data)


    def _chain_injection(self, data) -> Data:
        G    = _pyg_to_nx(data)
        path = _diameter_path(G)

        if len(path) < 2:
            return self._star_injection(data)

        u, v = path[0], path[-1]
        N    = data.num_nodes
        k    = self.k

        inherited = self._get_edge_features(data, u, path[1])

        new_x = self.dist.sample_node_features(k)
        chain = [u] + [N + i for i in range(k)] + [v]

        src, dst, ea = [], [], []
        for a, b in zip(chain[:-1], chain[1:]):
            src += [a, b]
            dst += [b, a]
            ea  += [inherited, inherited]

        return self._build_augmented_data(
            data, new_x, src, dst,
            new_edge_attr=torch.stack(ea),
        )


    def _star_injection(self, data) -> Data:

        N = data.num_nodes
        k = min(self.k, N)

        ei     = data.edge_index.cpu().numpy()
        degree = np.zeros(N, dtype=np.int64)
        if ei.shape[1] > 0:
            np.add.at(degree, ei[0], 1)
            np.add.at(degree, ei[1], 1)

        prob = (degree + 1.0)
        prob = prob / prob.sum()
        targets = np.random.choice(N, size=k, replace=False, p=prob)

        hub_idx = N
        new_x   = self.dist.sample_node_features(1)

        src, dst = [], []
        for t in targets.tolist():
            src += [hub_idx, t]
            dst += [t, hub_idx]

        return self._build_augmented_data(data, new_x, src, dst)


    def _parallel_paths(self, data) -> Data:
        G = _pyg_to_nx(data)

        if G.number_of_edges() == 0:
            return self._star_injection(data)

        betweenness = nx.edge_betweenness_centrality(G)
        u, v        = max(betweenness, key=betweenness.get)
        inherited   = self._get_edge_features(data, u, v)

        N    = data.num_nodes
        k    = self.k
        new_x = self.dist.sample_node_features(k)

        src, dst, ea = [], [], []
        for i in range(k):
            ci   = N + i
            src += [u,  ci, ci, v ]
            dst += [ci, u,  v,  ci]
            ea  += [inherited] * 4

        return self._build_augmented_data(
            data, new_x, src, dst,
            new_edge_attr=torch.stack(ea),
        )


    def _get_edge_features(self, data, u: int, v: int) -> torch.Tensor:

        if data.edge_attr is None or data.edge_index.shape[1] == 0:
            return self.dist.sample_edge_features(1).squeeze(0)

        ei   = data.edge_index.cpu()
        ea   = data.edge_attr.cpu()
        mask = ((ei[0] == u) & (ei[1] == v)) | ((ei[0] == v) & (ei[1] == u))
        hits = mask.nonzero(as_tuple=False).view(-1)

        if hits.numel() > 0:
            return ea[hits[0]].clone()

        return self.dist.sample_edge_features(1).squeeze(0)

    def _build_augmented_data(
        self,
        data,
        new_x: torch.Tensor,
        src_list: list[int],
        dst_list: list[int],
        new_edge_attr: torch.Tensor | None = None,
    ) -> Data:
        N           = data.num_nodes
        k_new       = new_x.shape[0]
        n_new_edges = len(src_list)

        x_aug = torch.cat([data.x.cpu(), new_x], dim=0)

        if n_new_edges > 0:
            new_ei         = torch.tensor([src_list, dst_list], dtype=torch.long)
            edge_index_aug = torch.cat([data.edge_index.cpu(), new_ei], dim=1)
        else:
            edge_index_aug = data.edge_index.cpu().clone()

        if n_new_edges > 0:
            ea_new = (
                new_edge_attr.cpu()
                if new_edge_attr is not None
                else self.dist.sample_edge_features(n_new_edges)
            )
            edge_attr_aug = torch.cat([data.edge_attr.cpu(), ea_new], dim=0)
        else:
            edge_attr_aug = data.edge_attr.cpu().clone()

        injected_flag = torch.full((k_new,), self.injected_is_original,
                                   dtype=torch.float32)
        if hasattr(data, "is_original") and data.is_original is not None:
            is_orig_aug = torch.cat([data.is_original.cpu(), injected_flag])
        else:
            is_orig_aug = torch.cat([
                torch.ones(N, dtype=torch.float32),
                injected_flag,
            ])

        return Data(
            x          =x_aug,
            edge_index =edge_index_aug,
            edge_attr  =edge_attr_aug,
            y          =data.y,
            is_original=is_orig_aug,
            batch      =torch.zeros(N + k_new, dtype=torch.long),
        )
