import os
import json
import torch
import numpy as np
import polars as pl
from torch.utils.data import Dataset
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split


class Elliptic2Dataset(Dataset):
    
    _cache = {}

    def __init__(self, processed_dir, split="train", val_ratio=0.15, test_ratio=0.10):
        self.processed_dir = processed_dir
        self.split = split
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        all_data = self.get_all_data()

        self.data_list = self.stratified_split(all_data)

        labels = [d.y.item() for d in self.data_list]
        counts = np.bincount(labels, minlength=2)
        print(f"  [{split.upper()}] {len(self.data_list)} subgraphs "
              f"(class 0: {counts[0]}, class 1: {counts[1]})")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

    def get_labels(self):
        return [d.y.item() for d in self.data_list]

    def nodes_have_is_original(self):
        nodes_path = os.path.join(self.processed_dir, "nodes.parquet")
        if not os.path.exists(nodes_path):
            return None
        return "is_original" in pl.read_parquet(nodes_path, n_rows=1).columns

    def get_all_data(self):
        if self.processed_dir in Elliptic2Dataset._cache:
            return Elliptic2Dataset._cache[self.processed_dir]

        cache_path = os.path.join(self.processed_dir, "all_graphs.pt")
        summary_path = os.path.join(self.processed_dir, "summary.json")
        cache_meta_path = os.path.join(self.processed_dir, "cache_meta.json")

        cache_valid = os.path.exists(cache_path)
        if cache_valid and os.path.exists(summary_path):
            with open(summary_path) as f:
                summary = json.load(f)
            cached_meta = {}
            if os.path.exists(cache_meta_path):
                with open(cache_meta_path) as f:
                    cached_meta = json.load(f)

            stale_reason = None
            cached_k_hop = cached_meta.get("k_hop", None)
            if cached_k_hop is not None and cached_k_hop != summary.get("k_hop", 0):
                stale_reason = f"k_hop changed ({cached_k_hop} -> {summary.get('k_hop', 0)})"
                
            if stale_reason is None:
                cached_flag = cached_meta.get("has_is_original", None)
                current_flag = self.nodes_have_is_original()
                if cached_flag is not None and cached_flag != current_flag:
                    stale_reason = f"nodes.parquet is_original changed ({cached_flag} -> {current_flag})"

            if stale_reason is not None:
                print(f"  Cache stale: {stale_reason}, rebuilding...")
                os.remove(cache_path)
                cache_valid = False

        if cache_valid:
            print("  Loading cached graphs from all_graphs.pt...")
            data_list = torch.load(cache_path, weights_only=False)
            print(f"  Loaded {len(data_list)} graphs from cache")
            Elliptic2Dataset._cache[self.processed_dir] = data_list
            return data_list

        data_list = self.load_and_build()

        print(f"  Saving graph cache to {cache_path}...")
        torch.save(data_list, cache_path)

        if os.path.exists(summary_path):
            with open(summary_path) as f:
                summary = json.load(f)
            with open(cache_meta_path, "w") as f:
                json.dump({
                    "k_hop": summary.get("k_hop", 0),
                    "has_is_original": self.nodes_have_is_original(),
                }, f)

        Elliptic2Dataset._cache[self.processed_dir] = data_list
        return data_list

    def load_and_build(self):
        print("  Loading preprocessed data...")

        with open(f"{self.processed_dir}/summary.json") as f:
            summary = json.load(f)
        node_feat_dim = summary["node_feature_dims"]
        edge_feat_dim = summary["edge_feature_dims"]
        print(f"    Node features: {node_feat_dim} dims")
        print(f"    Edge features: {edge_feat_dim} dims")

        nodes_df = pl.read_parquet(f"{self.processed_dir}/nodes.parquet")
        edges_df = pl.read_parquet(f"{self.processed_dir}/edges.parquet")
        components_df = pl.read_parquet(f"{self.processed_dir}/components.parquet")
        node_features_df = pl.read_parquet(f"{self.processed_dir}/node_features.parquet")
        edge_features_df = pl.read_parquet(f"{self.processed_dir}/edge_features.parquet")

        print(f"    Nodes: {len(nodes_df):,} | Edges: {len(edges_df):,} | Subgraphs: {len(components_df):,}")

        node_feat_cols = [c for c in node_features_df.columns if c != "node_id"]
        node_id_to_features = {}
        node_features_np = node_features_df.select(node_feat_cols).to_numpy().astype(np.float32)
        node_ids_arr = node_features_df["node_id"].to_numpy()
        for i, nid in enumerate(node_ids_arr):
            node_id_to_features[nid] = node_features_np[i]

        edge_feat_cols = [c for c in edge_features_df.columns if c not in ("source", "target", "txId")]
        edge_to_features = {}
        if len(edge_feat_cols) > 0:
            edge_features_np = edge_features_df.select(edge_feat_cols).to_numpy().astype(np.float32)
            txid_arr = edge_features_df["txId"].to_numpy()
            for i, tid in enumerate(txid_arr):
                edge_to_features[tid] = edge_features_np[i]

        subgraph_labels = dict(zip(
            components_df["subgraph_id"].to_list(),
            [int(x) for x in components_df["label"].to_list()]
        ))

        has_is_original = "is_original" in nodes_df.columns
        subgraph_to_nodes = {}
        for row in nodes_df.iter_rows(named=True):
            sg_id = row["subgraph_id"]
            is_orig = row["is_original"] if has_is_original else True
            if sg_id not in subgraph_to_nodes:
                subgraph_to_nodes[sg_id] = []
            subgraph_to_nodes[sg_id].append((row["node_id"], is_orig))

        subgraph_to_edges = {}
        has_sg_col = "subgraph_id" in edges_df.columns
        for row in edges_df.iter_rows(named=True):
            if has_sg_col:
                sg_id = row["subgraph_id"]
            else:
                src = row["source"]
                sg_id = None
                for sg, nodes in subgraph_to_nodes.items():
                    if src in nodes:
                        sg_id = sg
                        break
            if sg_id is not None:
                if sg_id not in subgraph_to_edges:
                    subgraph_to_edges[sg_id] = []
                subgraph_to_edges[sg_id].append((row["source"], row["target"], row["txId"]))


        print("    Building PyG Data objects...")
        zero_node_feat = np.zeros(node_feat_dim, dtype=np.float32)
        zero_edge_feat = np.zeros(edge_feat_dim, dtype=np.float32)

        data_list = []
        skipped = 0

        for sg_id in components_df["subgraph_id"].to_list():
            if sg_id not in subgraph_to_nodes:
                skipped += 1
                continue

            node_entries = subgraph_to_nodes[sg_id]
            node_ids_in_sg = [entry[0] for entry in node_entries]
            is_original_flags = [entry[1] for entry in node_entries]
            edges_in_sg = subgraph_to_edges.get(sg_id, [])
            label = subgraph_labels[sg_id]

            global_to_local = {nid: i for i, nid in enumerate(node_ids_in_sg)}

            x_list = []
            for nid in node_ids_in_sg:
                x_list.append(node_id_to_features.get(nid, zero_node_feat))
            x = torch.tensor(np.stack(x_list), dtype=torch.float32)

            is_original = torch.tensor(
                [1.0 if orig else 0.0 for orig in is_original_flags],
                dtype=torch.float32
            )

            edge_index_list = []
            edge_attr_list = []
            for src, dst, txid in edges_in_sg:
                if src in global_to_local and dst in global_to_local:
                    edge_index_list.append([global_to_local[src], global_to_local[dst]])
                    edge_attr_list.append(
                        edge_to_features.get(txid, zero_edge_feat)
                    )

            if len(edge_index_list) == 0:
                for i in range(len(node_ids_in_sg)):
                    edge_index_list.append([i, i])
                    edge_attr_list.append(zero_edge_feat)

            edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(np.stack(edge_attr_list), dtype=torch.float32)

            y = torch.tensor([label], dtype=torch.long)

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y,
                        is_original=is_original)
            data_list.append(data)

        if skipped > 0:
            print(f"    Skipped {skipped} subgraphs (no nodes found)")

        print(f"    Built {len(data_list)} Data objects")

        sizes = [d.num_nodes for d in data_list]
        labels = [d.y.item() for d in data_list]
        counts = np.bincount(labels, minlength=2)
        print(f"    Avg nodes/subgraph: {np.mean(sizes):.2f} | Max: {max(sizes)} | Min: {min(sizes)}")
        print(f"    Label distribution: class 0={counts[0]:,}, class 1={counts[1]:,} "
              f"(ratio {counts[0]/max(counts[1],1):.1f}:1)")

        return data_list

    def stratified_split(self, data_list):
        labels = np.array([d.y.item() for d in data_list])
        indices = np.arange(len(data_list))

        train_val_idx, test_idx = train_test_split(
            indices, test_size=self.test_ratio,
            stratify=labels, random_state=42
        )

        val_ratio_adjusted = self.val_ratio / (1.0 - self.test_ratio)
        train_idx, val_idx = train_test_split(
            train_val_idx, test_size=val_ratio_adjusted,
            stratify=labels[train_val_idx], random_state=42
        )

        if self.split == "train":
            selected = train_idx
        elif self.split == "val":
            selected = val_idx
        elif self.split == "test":
            selected = test_idx
        else:
            raise ValueError(f"Unknown split: {self.split}")

        return [data_list[i] for i in selected]
