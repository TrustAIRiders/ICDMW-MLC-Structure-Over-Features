import os
import sys
import time
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import torch
from sklearn.metrics import f1_score

from scripts.utils import (
    ALL_CONV_TYPES, ARCH_NAMES,
    load_yaml_config, resolve_device,
    load_model, load_datasets, get_feat_dims,
    infer, save_results,
)
from src.attacks.node_injection import LicitDistribution
from src.attacks.topology_rewiring import TopologyAnalyzer, TopologyAugmentationAttack
from src.attacks.transfer_attack import get_clean_predictions, get_tp_indices


TECHNIQUE_LABELS = {
    "chain_injection": "Chain Injection",
    "star":            "Star",
    "parallel":        "Parallel",
}


def run_topology_for_combo(
    model,
    test_dataset,
    attacker: TopologyAugmentationAttack,
    tp_indices: np.ndarray,
    labels: np.ndarray,
    preds_clean: np.ndarray,
    probs_clean: np.ndarray,
    device,
) -> dict:
    n_suspicious = int((labels == 1).sum())
    n_tp         = len(tp_indices)
    clean_f1     = float(f1_score(labels, preds_clean, average="macro", zero_division=0))

    preds_post     = preds_clean.copy()
    probs_post     = probs_clean.copy()
    n_evaded       = 0
    conf_pre_list  = []
    conf_post_list = []
    topo_deltas    = []
    pg_evaded      = []

    for idx in tp_indices:
        data = test_dataset[int(idx)]

        metrics_pre  = TopologyAnalyzer.compute_metrics(data)
        aug_data     = attacker.attack(data)
        metrics_post = TopologyAnalyzer.compute_metrics(aug_data)
        topo_deltas.append(TopologyAnalyzer.compare(metrics_pre, metrics_post))

        pred_post, prob_post = infer(model, aug_data, device)

        preds_post[int(idx)] = pred_post
        probs_post[int(idx)] = prob_post

        evaded = 1 if pred_post == 0 else 0
        n_evaded += evaded
        pg_evaded.append(evaded)

        conf_pre_list.append(float(probs_clean[int(idx)]))
        conf_post_list.append(prob_post)

    post_f1   = float(f1_score(labels, preds_post, average="macro", zero_division=0))
    conf_pre  = np.array(conf_pre_list)
    conf_post = np.array(conf_post_list)

    def _mean(key: str) -> float:
        vals = [d[key] for d in topo_deltas]
        return round(float(np.mean(vals)), 4) if vals else 0.0

    return {
        "n_suspicious":            n_suspicious,
        "n_attacked":              n_tp,
        "n_evaded":                n_evaded,
        "asr":                     round(n_evaded / max(n_tp, 1), 6),
        "clean_f1_macro":          round(clean_f1, 6),
        "post_f1_macro":           round(post_f1, 6),
        "f1_drop":                 round(post_f1 - clean_f1, 6),
        "clean_recall_suspicious": round(n_tp / max(n_suspicious, 1), 6),
        "post_recall_suspicious":  round((n_tp - n_evaded) / max(n_suspicious, 1), 6),
        "mean_confidence_pre":     round(float(conf_pre.mean()), 6),
        "mean_confidence_post":    round(float(conf_post.mean()), 6),
        "mean_confidence_drop":    round(float((conf_pre - conf_post).mean()), 6),
        "mean_delta_nodes":        _mean("delta_nodes"),
        "mean_delta_edges":        _mean("delta_edges"),
        "mean_delta_diameter":     _mean("delta_diameter"),
        "mean_delta_clustering":   _mean("delta_clustering"),
        "mean_delta_avg_degree":   _mean("delta_avg_degree"),
        "per_graph": {
            "subgraph_indices": tp_indices.tolist(),
            "evaded":           pg_evaded,
            "confidence_pre":   conf_pre_list,
            "confidence_post":  conf_post_list,
        },
    }


def _print_summary(results: dict, args) -> None:
    max_k = max(args.k_intermediates)

    print(f"\n\n{'='*72}")
    print("  TOPOLOGY REWIRING — ASR  [all k values]")
    print(f"{'='*72}")
    k_hdr = "".join(f"  k={k:<5}" for k in args.k_intermediates)
    hdr   = f"{'Model':<14}{'Technique':<12}" + k_hdr
    print(hdr)
    print("-" * len(hdr))
    for ct in args.conv_types:
        if ct not in results["models"]:
            continue
        combos = results["models"][ct]["combinations"]
        for tech in args.techniques:
            row = f"{ARCH_NAMES[ct]:<14}{tech:<12}"
            for k in args.k_intermediates:
                key = f"{tech}_k{k}"
                asr = combos.get(key, {}).get("asr", float("nan"))
                row += f"  {asr * 100:>6.2f}%"
            print(row)
        print()

    print(f"\n{'='*72}")
    print("  TOPOLOGY REWIRING — Mean Δdiameter  [all k values]")
    print(f"{'='*72}")
    print(hdr)
    print("-" * len(hdr))
    for ct in args.conv_types:
        if ct not in results["models"]:
            continue
        combos = results["models"][ct]["combinations"]
        for tech in args.techniques:
            row = f"{ARCH_NAMES[ct]:<14}{tech:<12}"
            for k in args.k_intermediates:
                key  = f"{tech}_k{k}"
                diam = combos.get(key, {}).get("mean_delta_diameter", float("nan"))
                row += f"  {diam:>+7.2f} "
            print(row)
        print()

    print(f"\n{'='*72}")
    print("  TOPOLOGY REWIRING — Mean confidence drop  [all k values]")
    print(f"{'='*72}")
    print(hdr)
    print("-" * len(hdr))
    for ct in args.conv_types:
        if ct not in results["models"]:
            continue
        combos = results["models"][ct]["combinations"]
        for tech in args.techniques:
            row = f"{ARCH_NAMES[ct]:<14}{tech:<12}"
            for k in args.k_intermediates:
                key  = f"{tech}_k{k}"
                drop = combos.get(key, {}).get("mean_confidence_drop", float("nan"))
                row += f"  {drop:>+7.4f}"
            print(row)
        print()

    print(f"\n{'='*72}")
    print(f"  TOPOLOGY REWIRING — Cross-technique  [k={max_k}]")
    print(f"{'='*72}")
    h4 = (
        f"{'Model':<14}{'Technique':<12}"
        f"  {'ASR':>7}  {'ΔDiam':>7}  {'ΔClust':>8}  {'ΔConf':>8}  {'ΔF1':>8}"
    )
    print(h4)
    print("-" * len(h4))
    for ct in args.conv_types:
        if ct not in results["models"]:
            continue
        combos = results["models"][ct]["combinations"]
        for tech in args.techniques:
            key   = f"{tech}_k{max_k}"
            combo = combos.get(key, {})
            asr   = combo.get("asr",                   float("nan"))
            diam  = combo.get("mean_delta_diameter",   float("nan"))
            clust = combo.get("mean_delta_clustering", float("nan"))
            drop  = combo.get("mean_confidence_drop",  float("nan"))
            f1d   = combo.get("f1_drop",               float("nan"))
            print(
                f"{ARCH_NAMES[ct]:<14}{tech:<12}"
                f"  {asr*100:>6.2f}%  {diam:>+7.2f}  {clust:>+8.4f}"
                f"  {drop:>+8.4f}  {f1d:>+8.4f}"
            )
        print()


def _save_scatter_plot(results: dict, args, output_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [skip plot] matplotlib not available.")
        return

    technique_colors = {
        "chain_injection": "#2196F3",
        "star":            "#FF9800",
        "parallel":        "#4CAF50",
    }
    arch_markers = {
        "gatv2":     "o",
        "sage":      "s",
        "sage_edge": "^",
        "gin":       "D",
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle(
        "Topology Rewiring Attack — Topology Change vs ASR",
        fontsize=13, fontweight="bold", y=1.01,
    )

    x_keys    = ["mean_delta_nodes",    "mean_delta_diameter"]
    x_labels  = ["Mean Δnodes (added)", "Mean Δdiameter"]
    subtitles = [
        "Structural budget vs effectiveness",
        "Topological disruption vs effectiveness",
    ]

    tech_handles  = {}
    model_handles = {}

    for ax, x_key, x_lbl, subtitle in zip(axes, x_keys, x_labels, subtitles):
        for ct in args.conv_types:
            if ct not in results["models"]:
                continue
            combos = results["models"][ct]["combinations"]
            for tech in args.techniques:
                for k in args.k_intermediates:
                    key   = f"{tech}_k{k}"
                    combo = combos.get(key)
                    if combo is None:
                        continue
                    x_val  = combo.get(x_key, 0.0)
                    y_val  = combo.get("asr",   0.0) * 100
                    color  = technique_colors.get(tech, "#888888")
                    marker = arch_markers.get(ct, "x")

                    ax.scatter(
                        x_val, y_val,
                        c=color, marker=marker,
                        s=90, alpha=0.85, linewidths=0.6,
                        edgecolors="white", zorder=3,
                    )
                    ax.annotate(
                        f"k={k}",
                        (x_val, y_val),
                        textcoords="offset points", xytext=(4, 4),
                        fontsize=7, color=color, alpha=0.8,
                    )

                    if tech not in tech_handles:
                        tech_handles[tech] = matplotlib.lines.Line2D(
                            [], [], color=color, marker="o",
                            linestyle="None", markersize=8,
                            label=TECHNIQUE_LABELS.get(tech, tech),
                        )
                    if ct not in model_handles:
                        model_handles[ct] = matplotlib.lines.Line2D(
                            [], [], color="gray", marker=marker,
                            linestyle="None", markersize=8,
                            label=ARCH_NAMES.get(ct, ct),
                        )

        ax.set_xlabel(x_lbl,    fontsize=10)
        ax.set_ylabel("ASR (%)", fontsize=10)
        ax.set_title(subtitle,   fontsize=10, style="italic")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_ylim(bottom=0)

    legend_handles = (
        list(tech_handles.values()) +
        [matplotlib.patches.Patch(color="none")] +
        list(model_handles.values())
    )
    legend_labels = (
        [h.get_label() for h in tech_handles.values()] +
        [""] +
        [h.get_label() for h in model_handles.values()]
    )
    fig.legend(
        legend_handles, legend_labels,
        loc="lower center", ncol=len(tech_handles) + 1 + len(model_handles),
        frameon=True, fontsize=9, bbox_to_anchor=(0.5, -0.06),
    )

    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "topology_scatter.png")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Scatter plot saved → {plot_path}")


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default="config.yaml")
    pre_args, _ = pre.parse_known_args()
    config  = load_yaml_config(pre_args.config)
    twr_cfg = config.get("topology_rewiring_attack", {})

    parser = argparse.ArgumentParser(
        description="Topology Rewiring Attack on Elliptic2 GNN Models"
    )
    parser.add_argument("--config",          default="config.yaml")
    parser.add_argument("--seed",            type=int,
                        default=twr_cfg.get("seed", 42))
    parser.add_argument("--techniques",      nargs="+",
                        default=twr_cfg.get("techniques", ["chain_injection", "star", "parallel"]),
                        choices=["chain_injection", "star", "parallel"])
    parser.add_argument("--k_intermediates", nargs="+", type=int,
                        default=twr_cfg.get("k_intermediates", [2, 5, 10]))
    parser.add_argument("--conv_types",      nargs="+",
                        default=twr_cfg.get("conv_types", ALL_CONV_TYPES),
                        choices=ALL_CONV_TYPES)
    parser.add_argument("--injected_is_original", type=float, default=0.0,
                        help="Provenance flag carried by injected nodes. 0.0 (default) "
                             "makes them look like k-hop expansion context, which the "
                             "model discounts at readout via expansion_node_weight; "
                             "1.0 marks them as original cluster nodes. "
                             "Values <= 0.5 behave as 0.0.")
    parser.add_argument("--output",
                        default=twr_cfg.get("output", "data/results/topology_rewiring_results.json"))
    parser.add_argument("--device",          default=None)
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"Device: {device}")

    checkpoint_dir = config["training"]["checkpoint_dir"]

    print("Loading datasets ")
    datasets      = load_datasets(config)
    train_dataset = datasets["train"]
    test_dataset  = datasets["test"]

    node_feat_dim, edge_feat_dim = get_feat_dims(train_dataset)
    print(f"Node features: {node_feat_dim} | Edge features: {edge_feat_dim}")

    print("Computing licit distribution from training set ")
    licit_dist = LicitDistribution(train_dataset)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    results = {
        "experiment":      "topology_rewiring_attack",
        "seed":            args.seed,
        "techniques":      args.techniques,
        "k_intermediates": args.k_intermediates,
        "injected_is_original": args.injected_is_original,
        "checkpoint_dir":  checkpoint_dir,
        "node_feat_dim":   node_feat_dim,
        "edge_feat_dim":   edge_feat_dim,
        "models":          {},
    }

    for conv_type in args.conv_types:
        arch_name = ARCH_NAMES.get(conv_type, conv_type)
        print(f"\n{'='*60}")
        print(f"  Architecture: {arch_name}")
        print(f"{'='*60}")

        try:
            model = load_model(
                conv_type, checkpoint_dir, args.seed,
                node_feat_dim, edge_feat_dim, config, device,
            )
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue

        labels, preds_clean, probs_clean = get_clean_predictions(model, test_dataset, device)
        tp_indices = get_tp_indices(labels, preds_clean)
        n_sus      = int((labels == 1).sum())
        clean_f1   = float(f1_score(labels, preds_clean, average="macro", zero_division=0))
        print(
            f"  Clean F1={clean_f1:.4f}  "
            f"Recall(susp)={len(tp_indices)}/{n_sus}="
            f"{len(tp_indices)/max(n_sus, 1):.3f}  "
            f"Attacking {len(tp_indices)} TP subgraphs"
        )

        model_results = {"arch": arch_name, "combinations": {}}
        total_combos  = len(args.techniques) * len(args.k_intermediates)
        combo_num     = 0

        for technique in args.techniques:
            for k in args.k_intermediates:
                combo_num += 1
                combo_key  = f"{technique}_k{k}"
                print(
                    f"\n  [{combo_num}/{total_combos}] [{arch_name}] "
                    f"technique={technique}  k={k}"
                )
                t0 = time.time()

                attacker = TopologyAugmentationAttack(
                    licit_dist=licit_dist,
                    technique=technique,
                    k_intermediates=k,
                    injected_is_original=args.injected_is_original,
                )

                stats = run_topology_for_combo(
                    model=model,
                    test_dataset=test_dataset,
                    attacker=attacker,
                    tp_indices=tp_indices,
                    labels=labels,
                    preds_clean=preds_clean,
                    probs_clean=probs_clean,
                    device=device,
                )
                stats["technique"]        = technique
                stats["k_intermediates"]  = k
                stats["time_seconds"]     = round(time.time() - t0, 2)

                model_results["combinations"][combo_key] = stats

                print(
                    f"    ASR={stats['asr']:.2%} "
                    f"({stats['n_evaded']}/{stats['n_attacked']})  "
                    f"F1: {stats['clean_f1_macro']:.4f}→{stats['post_f1_macro']:.4f} "
                    f"({stats['f1_drop']:+.4f})  "
                    f"ΔConf: {stats['mean_confidence_drop']:+.4f}  "
                    f"Δdiam: {stats['mean_delta_diameter']:+.2f}  "
                    f"Δnodes: {stats['mean_delta_nodes']:+.1f}  "
                    f"Time: {stats['time_seconds']:.1f}s"
                )

        results["models"][conv_type] = model_results

    _print_summary(results, args)

    plots_dir = os.path.join(os.path.dirname(os.path.abspath(args.output)), "plots")
    _save_scatter_plot(results, args, plots_dir)

    save_results(results, args.output)


if __name__ == "__main__":
    main()
