import os
import sys
import time
import json
import argparse
import polars as pl
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocess.utils import load_config, resolve_path, atomic_write_parquet, atomic_write_json, khop_expand


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess Elliptic2 into k-hop subgraphs")
    parser.add_argument("--k-hop", type=int, default=None,
                        help="k-hop expansion depth (default: data.k_hop from config.yaml)")
    parser.add_argument("--raw-dir", default=None,
                        help="Directory holding the five Elliptic2 CSVs (default: data.raw_dir)")
    parser.add_argument("--processed-dir", default=None,
                        help="Output directory (default: data.processed_dir)")
    return parser.parse_args()


def main():
    args = parse_args()

    config = load_config()
    raw_dir, out_dir = resolve_path(config)
    col_cfg = config["data"]["columns"]
    k_hop = config["data"].get("k_hop", 0)

    if args.raw_dir is not None:
        raw_dir = args.raw_dir
    if args.k_hop is not None:
        k_hop = args.k_hop
    if args.processed_dir is not None:
        out_dir = args.processed_dir
    elif args.k_hop is not None:
        out_dir = f"data/processed_k_hop_{k_hop}"
    print(f"  raw_dir={raw_dir} | processed_dir={out_dir} | k_hop={k_hop}")

    os.makedirs(out_dir, exist_ok=True)

    total_start = time.time()

    print("\n" + "=" * 60)
    print("[1/6] Loading labeled subgraph files")
    print("=" * 60)

    nodes_df = pl.read_csv(f"{raw_dir}/nodes.csv")
    edges_df = pl.read_csv(f"{raw_dir}/edges.csv")
    components_df = pl.read_csv(f"{raw_dir}/connected_components.csv")

    node_id_col = col_cfg["nodes"]["node_id"]
    node_subgraph_col = col_cfg["nodes"]["subgraph_id"]
    edge_src_col = col_cfg["edges"]["source"]
    edge_dst_col = col_cfg["edges"]["target"]
    edge_txid_col = col_cfg["edges"]["txId"]
    comp_id_col = col_cfg["components"]["subgraph_id"]
    comp_label_col = col_cfg["components"]["label"]

    sample_ratio = config["data"].get("sample_ratio", 1.0)

    if sample_ratio < 1.0:
        n_sample = max(1, int(len(components_df) * sample_ratio))
        components_df = components_df.sample(n=n_sample, seed=42)
        sampled_subgraphs = components_df[comp_id_col].to_list()
        nodes_df = nodes_df.filter(pl.col(node_subgraph_col).is_in(sampled_subgraphs))
        sampled_node_ids = nodes_df[node_id_col].to_list()
        edges_df = edges_df.filter(pl.col(edge_src_col).is_in(sampled_node_ids))
        print(f"  SAMPLED {sample_ratio*100:.1f}%: {n_sample} subgraphs")

    print(f"  nodes.csv:                {len(nodes_df):>10,} rows")
    print(f"  edges.csv:                {len(edges_df):>10,} rows")
    print(f"  connected_components.csv: {len(components_df):>10,} rows")

    nodes_df = nodes_df.rename({node_id_col: "node_id", node_subgraph_col: "subgraph_id"})
    nodes_df = nodes_df.with_columns(
        pl.lit(True).alias("is_original"),
        pl.lit(0).cast(pl.Int32).alias("hop"),
    )
    edges_df = edges_df.rename({edge_src_col: "source", edge_dst_col: "target", edge_txid_col: "txId"})

    label_counts = components_df[comp_label_col].value_counts().sort("count", descending=True)
    print(f"\n  Label distribution:")
    for row in label_counts.iter_rows():
        print(f"    Label {row[0]}: {row[1]:,} ({row[1] / len(components_df) * 100:.1f}%)")

    original_num_nodes = len(nodes_df)
    original_num_edges = len(edges_df)

    print("\n" + "=" * 60)
    print(f"[2/6] k-hop expansion (k={k_hop})")
    print("=" * 60)

    if k_hop > 0:
        node_to_subgraphs = defaultdict(set)
        for row in nodes_df.iter_rows(named=True):
            node_to_subgraphs[row["node_id"]].add(row["subgraph_id"])

        node_id_set = set(nodes_df["node_id"].to_list())

        expansion_nodes, expansion_edges, expanded_node_set = khop_expand(
            raw_dir, col_cfg, node_to_subgraphs, node_id_set, k_hop
        )

        if expansion_nodes:
            exp_nodes_df = pl.DataFrame(
                {
                    "node_id": [n[0] for n in expansion_nodes],
                    "subgraph_id": [n[1] for n in expansion_nodes],
                    "is_original": [False] * len(expansion_nodes),
                    "hop": [n[2] for n in expansion_nodes],
                },
                schema_overrides={"hop": pl.Int32},
            ).unique(subset=["node_id", "subgraph_id"])
            nodes_df = pl.concat([nodes_df, exp_nodes_df])
            print(f"\n  Expanded nodes: {original_num_nodes:,} -> {len(nodes_df):,} (+{len(nodes_df) - original_num_nodes:,})")


        orig_edge_rows = []
        for row in edges_df.iter_rows(named=True):
            src, dst, txid = row["source"], row["target"], row["txId"]
            src_sgs = node_to_subgraphs.get(src, set())
            dst_sgs = node_to_subgraphs.get(dst, set())
            shared = src_sgs & dst_sgs
            if shared:
                for sg_id in shared:
                    orig_edge_rows.append((src, dst, txid, sg_id))
            else:
                for sg_id in src_sgs:
                    orig_edge_rows.append((src, dst, txid, sg_id))
        edges_df = pl.DataFrame({
            "source": [r[0] for r in orig_edge_rows],
            "target": [r[1] for r in orig_edge_rows],
            "txId": [r[2] for r in orig_edge_rows],
            "subgraph_id": [r[3] for r in orig_edge_rows],
        })

        if expansion_edges:
            exp_edges_df = pl.DataFrame({
                "source": [e[0] for e in expansion_edges],
                "target": [e[1] for e in expansion_edges],
                "txId": [e[2] for e in expansion_edges],
                "subgraph_id": [e[3] for e in expansion_edges],
            }).unique()
            edges_df = pl.concat([edges_df, exp_edges_df])
            print(f"  Expanded edges: {original_num_edges:,} -> {len(edges_df):,} (+{len(edges_df) - original_num_edges:,})")

        node_id_set = expanded_node_set
    else:
        print("  k_hop=0 — no expansion")
        node_id_set = set(nodes_df["node_id"].to_list())

        node_to_subgraphs = defaultdict(set)
        for row in nodes_df.iter_rows(named=True):
            node_to_subgraphs[row["node_id"]].add(row["subgraph_id"])
        edge_rows = []
        for row in edges_df.iter_rows(named=True):
            src, dst, txid = row["source"], row["target"], row["txId"]
            src_sgs = node_to_subgraphs.get(src, set())
            dst_sgs = node_to_subgraphs.get(dst, set())
            shared = src_sgs & dst_sgs
            if shared:
                for sg_id in shared:
                    edge_rows.append((src, dst, txid, sg_id))
            else:
                for sg_id in src_sgs:
                    edge_rows.append((src, dst, txid, sg_id))
        edges_df = pl.DataFrame({
            "source": [r[0] for r in edge_rows],
            "target": [r[1] for r in edge_rows],
            "txId": [r[2] for r in edge_rows],
            "subgraph_id": [r[3] for r in edge_rows],
        })

    node_id_list = list(node_id_set)
    txid_set = set(edges_df["txId"].to_list())
    txid_list = list(txid_set)

    print(f"\n  Final unique nodes to extract: {len(node_id_set):,}")
    print(f"  Final unique txIds to extract: {len(txid_set):,}")

    print("\n" + "=" * 60)
    print("[3/6] Saving parquet files")
    print("=" * 60)

    nodes_parquet_path = f"{out_dir}/nodes.parquet"
    edges_parquet_path = f"{out_dir}/edges.parquet"
    components_parquet_path = f"{out_dir}/components.parquet"

    summary_path = f"{out_dir}/summary.json"
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            old_summary = json.load(f)
        if old_summary.get("k_hop", 0) != k_hop:
            print(f"  k_hop changed ({old_summary.get('k_hop', 0)} -> {k_hop}) — forcing rewrite of all files")
            for p in [nodes_parquet_path, edges_parquet_path,
                       f"{out_dir}/node_features.parquet", f"{out_dir}/edge_features.parquet",
                       f"{out_dir}/all_graphs.pt"]:
                if os.path.exists(p):
                    os.remove(p)
                    print(f"    removed {p}")

    if os.path.exists(nodes_parquet_path):
        _check = pl.read_parquet(nodes_parquet_path, n_rows=1)
        missing = [c for c in ("is_original", "hop") if c not in _check.columns]
        if missing:
            print(f"  Old nodes.parquet without {', '.join(missing)} — removing, rewriting")
            os.remove(nodes_parquet_path)
            graphs_cache = f"{out_dir}/all_graphs.pt"
            if os.path.exists(graphs_cache):
                os.remove(graphs_cache)
                print(f"    removed {graphs_cache}")

    if not os.path.exists(nodes_parquet_path):
        print("\n  Writing nodes.parquet")
        atomic_write_parquet(nodes_df, nodes_parquet_path)
    else:
        print("\n  nodes.parquet exists — skipped")

    if os.path.exists(edges_parquet_path):
        _check = pl.read_parquet(edges_parquet_path, n_rows=1)
        if "subgraph_id" not in _check.columns:
            print("  Old edges.parquet without subgraph_id — removing, rewriting")
            os.remove(edges_parquet_path)

    if not os.path.exists(edges_parquet_path):
        print("  Writing edges.parquet")
        atomic_write_parquet(edges_df, edges_parquet_path)
    else:
        print("  edges.parquet exists — skipped")

    if os.path.exists(components_parquet_path):
        _check = pl.read_parquet(components_parquet_path, n_rows=1)
        if _check["label"].dtype not in [pl.Int32, pl.Int64]:
            print(f"  Old components.parquet with non-int labels ({_check['label'].dtype}) — removing, rewriting")
            os.remove(components_parquet_path)
        else:
            print("  components.parquet exists — skipped")

    if not os.path.exists(components_parquet_path):
        print("  Writing components.parquet")
        components_to_save = components_df.rename({comp_id_col: "subgraph_id", comp_label_col: "label"})

        components_to_save = components_to_save.with_columns(
            pl.col("label").replace({"licit": 0, "suspicious": 1})
        )
        atomic_write_parquet(components_to_save, components_parquet_path)

    node_features_path = f"{out_dir}/node_features.parquet"

    print("\n" + "=" * 60)
    print("[4/6] Node features from background_nodes.csv")
    print("=" * 60)

    if os.path.exists(node_features_path):
        print("  node_features.parquet exists — skipped (resume)")
        filtered_node_features = pl.read_parquet(node_features_path)
        node_feat_dim = len(filtered_node_features.columns) - 1
    else:
        t0 = time.time()
        bg_nodes_lazy = pl.scan_csv(f"{raw_dir}/background_nodes.csv")
        bg_node_id_col_name = bg_nodes_lazy.collect_schema().names()[0]
        print(f"  background_nodes: {len(bg_nodes_lazy.collect_schema().names())} columns, ID col='{bg_node_id_col_name}'")

        filtered_node_features = (
            bg_nodes_lazy
            .filter(pl.col(bg_node_id_col_name).is_in(node_id_list))
            .collect(engine="streaming")
        ).rename({bg_node_id_col_name: "node_id"})

        node_feat_dim = len(filtered_node_features.columns) - 1
        print(f"  Extracted: {len(filtered_node_features):,} rows | {node_feat_dim} feature dims | {time.time() - t0:.1f}s")

        atomic_write_parquet(filtered_node_features, node_features_path)
        print(f"  Saved node_features.parquet")

    edge_features_path = f"{out_dir}/edge_features.parquet"

    print("\n" + "=" * 60)
    print("[5/6] Edge features from background_edges.csv")
    print("=" * 60)

    if os.path.exists(edge_features_path):
        _check = pl.read_parquet(edge_features_path, n_rows=1)
        if "txId" not in _check.columns:
            print("  Old edge_features.parquet without txId — removing, re-extracting")
            os.remove(edge_features_path)
        else:
            print("  edge_features.parquet exists — skipped (resume)")
            filtered_edge_features = pl.read_parquet(edge_features_path)
            edge_feat_dim = len(filtered_edge_features.columns) - 3

    if not os.path.exists(edge_features_path):
        print("  Scanning 77GB file (2-10 min)")
        t0 = time.time()

        bg_edges_lazy = pl.scan_csv(f"{raw_dir}/background_edges.csv")
        bg_txid_col = col_cfg["background_edges"]["txId"]
        bg_src_col = col_cfg["background_edges"]["source"]
        bg_dst_col = col_cfg["background_edges"]["target"]
        print(f"  background_edges: {len(bg_edges_lazy.collect_schema().names())} columns | "
              f"txId='{bg_txid_col}', src='{bg_src_col}', dst='{bg_dst_col}'")

        filtered_edge_features = (
            bg_edges_lazy
            .filter(pl.col(bg_txid_col).is_in(txid_list))
            .collect(engine="streaming")
        )

        filtered_edge_features = filtered_edge_features.rename({
            bg_txid_col: "txId",
            bg_src_col: "source",
            bg_dst_col: "target",
        })

        edge_feat_dim = len(filtered_edge_features.columns) - 3
        print(f"  Extracted: {len(filtered_edge_features):,} edges | {edge_feat_dim} feature dims | {time.time() - t0:.1f}s")

        atomic_write_parquet(filtered_edge_features, edge_features_path)
        print(f"  Saved edge_features.parquet")


    print("\n" + "=" * 60)
    print("[6/6] Validation & Summary")
    print("=" * 60)

    found_nodes = set(filtered_node_features["node_id"].to_list())
    missing_nodes = node_id_set - found_nodes
    if missing_nodes:
        print(f"  {len(missing_nodes):,} nodes not found in background — will use zero vectors")

    found_txids = set(filtered_edge_features["txId"].to_list())
    missing_edges = txid_set - found_txids
    if missing_edges:
        print(f"  {len(missing_edges):,} edges (txIds) not found in background — will use zero vectors")

    if not missing_nodes and not missing_edges:
        print("  All nodes and edges matched")

    summary = {
        "num_subgraphs": len(components_df),
        "num_nodes": len(nodes_df),
        "num_edges": len(edges_df),
        "original_nodes": original_num_nodes,
        "original_edges": original_num_edges,
        "k_hop": k_hop,
        "node_feature_dims": node_feat_dim,
        "edge_feature_dims": edge_feat_dim,
        "labels": label_counts.to_dicts(),
        "missing_nodes": len(missing_nodes),
        "missing_edges": len(missing_edges),
    }
    atomic_write_json(summary, f"{out_dir}/summary.json")

    total_time = time.time() - total_start
    print("\n" + "=" * 60)
    print(f"PREPROCESSING DONE — {total_time:.1f}s")
    print("=" * 60)
    print(f"\n  Output: {out_dir}/")
    print(f"  Files:  {os.listdir(out_dir)}")
    print(f"  Node features: {node_feat_dim} dims | Edge features: {edge_feat_dim} dims")
    if k_hop > 0:
        print(f"  k-hop expansion: {original_num_nodes:,} -> {len(nodes_df):,} nodes | "
              f"{original_num_edges:,} -> {len(edges_df):,} edges")


if __name__ == "__main__":
    main()
