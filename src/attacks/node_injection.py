import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm


class LicitDistribution:

    def __init__(self, train_dataset):
        node_feats: list[torch.Tensor] = []
        edge_feats: list[torch.Tensor] = []

        for data in tqdm(train_dataset, desc="  Computing licit distribution", leave=False):
            if data.y.item() != 0:
                continue

            node_feats.append(data.x.cpu())

            ei   = data.edge_index
            ea   = data.edge_attr.cpu()
            mask = ei[0] != ei[1]
            if mask.any():
                edge_feats.append(ea[mask])

        if not node_feats:
            raise RuntimeError(
                "No licit (y=0) subgraphs found in the training dataset. "
                "Cannot compute LicitDistribution."
            )

        all_nodes = torch.cat(node_feats, dim=0)
        self.node_mean     = all_nodes.mean(dim=0)
        self.node_std      = all_nodes.std(dim=0).clamp(min=1e-6)
        self.node_min      = all_nodes.min(dim=0).values
        self.node_max      = all_nodes.max(dim=0).values
        self.node_centroid = self.node_mean.clone()

        if edge_feats:
            all_edges = torch.cat(edge_feats, dim=0)
            self.edge_mean = all_edges.mean(dim=0)
            self.edge_std  = all_edges.std(dim=0).clamp(min=1e-6)
            self.edge_min  = all_edges.min(dim=0).values
            self.edge_max  = all_edges.max(dim=0).values
            n_licit_edges  = all_edges.shape[0]
        else:
            F_edge = train_dataset[0].edge_attr.shape[1]
            self.edge_mean = torch.zeros(F_edge)
            self.edge_std  = torch.ones(F_edge)
            self.edge_min  = torch.zeros(F_edge)
            self.edge_max  = torch.ones(F_edge)
            n_licit_edges  = 0

        print(
            f"  LicitDistribution: "
            f"{all_nodes.shape[0]:,} licit nodes | "
            f"{n_licit_edges:,} licit edges | "
            f"node_dim={all_nodes.shape[1]} edge_dim={self.edge_mean.shape[0]}"
        )

    def sample_node_features(self, n: int) -> torch.Tensor:

        noise   = torch.randn(n, self.node_mean.shape[0])
        samples = self.node_mean + noise * self.node_std
        return samples.clamp(self.node_min, self.node_max)

    def sample_edge_features(self, n: int) -> torch.Tensor:

        noise   = torch.randn(n, self.edge_mean.shape[0])
        samples = self.edge_mean + noise * self.edge_std
        return samples.clamp(self.edge_min, self.edge_max)


class NodeInjectionAttack:

    STRATEGIES = ("random", "degree", "mimicry")

    def __init__(
        self,
        licit_dist: LicitDistribution,
        strategy: str,
        k_nodes: int,
        connections_per_node: int,
        injected_is_original: float = 0.0,
    ):
        if strategy not in self.STRATEGIES:
            raise ValueError(
                f"strategy must be one of {self.STRATEGIES}, got '{strategy}'"
            )
        if k_nodes < 1:
            raise ValueError(f"k_nodes must be >= 1, got {k_nodes}")
        if connections_per_node < 1:
            raise ValueError(f"connections_per_node must be >= 1, got {connections_per_node}")

        self.dist = licit_dist
        self.strategy = strategy
        self.k    = k_nodes
        self.conn = connections_per_node
        self.injected_is_original = float(injected_is_original)


    def _random_targets(self, N: int) -> np.ndarray:

        n = min(self.conn, N)
        return np.random.choice(N, size=n, replace=False)

    def _degree_targets(self, data) -> np.ndarray:

        N  = data.num_nodes
        ei = data.edge_index.cpu().numpy()

        degree = np.zeros(N, dtype=np.int64)
        if ei.shape[1] > 0:
            np.add.at(degree, ei[0], 1)
            np.add.at(degree, ei[1], 1)

        n = min(self.conn, N)
        return np.argsort(-degree)[:n]


    def _bidirectional_edges(
        self, inj_idx: int, targets: np.ndarray
    ) -> tuple[list[int], list[int]]:

        src, dst = [], []
        for t in targets.tolist():
            src.extend([inj_idx, t])
            dst.extend([t,       inj_idx])
        return src, dst


    def _random_strategy(
        self, data
    ) -> tuple[torch.Tensor, list[int], list[int]]:
        N     = data.num_nodes
        new_x = self.dist.sample_node_features(self.k)

        src_all, dst_all = [], []
        for i in range(self.k):
            targets      = self._random_targets(N)
            s, d         = self._bidirectional_edges(N + i, targets)
            src_all.extend(s)
            dst_all.extend(d)

        return new_x, src_all, dst_all

    def _degree_strategy(
        self, data
    ) -> tuple[torch.Tensor, list[int], list[int]]:
        N       = data.num_nodes
        new_x   = self.dist.sample_node_features(self.k)
        targets = self._degree_targets(data)

        src_all, dst_all = [], []
        for i in range(self.k):
            s, d = self._bidirectional_edges(N + i, targets)
            src_all.extend(s)
            dst_all.extend(d)

        return new_x, src_all, dst_all

    def _mimicry_strategy(
        self, data
    ) -> tuple[torch.Tensor, list[int], list[int]]:

        N     = data.num_nodes
        new_x = self.dist.node_centroid.unsqueeze(0).expand(self.k, -1).clone()

        src_all, dst_all = [], []
        for i in range(self.k):
            targets = self._random_targets(N)
            s, d    = self._bidirectional_edges(N + i, targets)
            src_all.extend(s)
            dst_all.extend(d)

        return new_x, src_all, dst_all


    def _build_augmented_data(
        self,
        data,
        new_x: torch.Tensor,
        src_list: list[int],
        dst_list: list[int],
    ) -> Data:

        N          = data.num_nodes
        k          = new_x.shape[0]
        n_new_edges = len(src_list)

        x_aug = torch.cat([data.x.cpu(), new_x], dim=0)

        if n_new_edges > 0:
            new_ei         = torch.tensor([src_list, dst_list], dtype=torch.long)
            edge_index_aug = torch.cat([data.edge_index.cpu(), new_ei], dim=1)
            new_ea         = self.dist.sample_edge_features(n_new_edges)
            edge_attr_aug  = torch.cat([data.edge_attr.cpu(), new_ea], dim=0)
        else:
            edge_index_aug = data.edge_index.cpu()
            edge_attr_aug  = data.edge_attr.cpu()

        injected_flag = torch.full((k,), self.injected_is_original, dtype=torch.float32)
        if hasattr(data, "is_original") and data.is_original is not None:
            is_orig_aug = torch.cat([data.is_original.cpu(), injected_flag], dim=0)
        else:
            is_orig_aug = torch.cat([
                torch.ones(N, dtype=torch.float32),
                injected_flag,
            ], dim=0)

        batch_aug = torch.zeros(N + k, dtype=torch.long)

        return Data(
            x=x_aug,
            edge_index=edge_index_aug,
            edge_attr=edge_attr_aug,
            y=data.y,
            is_original=is_orig_aug,
            batch=batch_aug,
        )


    def attack(self, data) -> Data:

        if self.strategy == "random":
            new_x, src_list, dst_list = self._random_strategy(data)
        elif self.strategy == "degree":
            new_x, src_list, dst_list = self._degree_strategy(data)
        else:
            new_x, src_list, dst_list = self._mimicry_strategy(data)

        return self._build_augmented_data(data, new_x, src_list, dst_list)
