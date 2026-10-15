import os
import sys
import time
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import numpy as np
from sklearn.metrics import f1_score

from scripts.utils import (
    ALL_CONV_TYPES, ARCH_NAMES,
    load_yaml_config, resolve_device,
    load_model, load_datasets, get_feat_dims,
    infer, save_results,
)
from src.attacks.node_injection import LicitDistribution, NodeInjectionAttack
from src.attacks.transfer_attack import get_clean_predictions, get_tp_indices


def run_injection_for_combo(
    model,
    test_dataset,
    attacker: NodeInjectionAttack,
    tp_indices: np.ndarray,
    labels: np.ndarray,
    preds_clean: np.ndarray,
    probs_clean: np.ndarray,
    device,
) -> dict:
    n_suspicious   = int((labels == 1).sum())
    n_tp           = len(tp_indices)
    clean_f1       = float(f1_score(labels, preds_clean, average="macro", zero_division=0))

    preds_post     = preds_clean.copy()
    probs_post     = probs_clean.copy()
    n_evaded       = 0
    conf_pre_list  = []
    conf_post_list = []
    new_edge_counts: list[int] = []
    pg_evaded      = []

    for idx in tp_indices:
        data     = test_dataset[int(idx)]
        aug_data = attacker.attack(data)

        pred_post, prob_post = infer(model, aug_data, device)

        preds_post[int(idx)] = pred_post
        probs_post[int(idx)] = prob_post

        evaded = 1 if pred_post == 0 else 0
        n_evaded += evaded
        pg_evaded.append(evaded)

        conf_pre_list.append(float(probs_clean[int(idx)]))
        conf_post_list.append(prob_post)
        new_edge_counts.append(
            aug_data.edge_index.shape[1] - data.edge_index.shape[1]
        )

    post_f1   = float(f1_score(labels, preds_post, average="macro", zero_division=0))
    conf_pre  = np.array(conf_pre_list)
    conf_post = np.array(conf_post_list)

    return {
        "n_suspicious":              n_suspicious,
        "n_attacked":                n_tp,
        "n_evaded":                  n_evaded,
        "asr":                       round(n_evaded / max(n_tp, 1), 6),
        "clean_f1_macro":            round(clean_f1, 6),
        "post_f1_macro":             round(post_f1, 6),
        "f1_drop":                   round(post_f1 - clean_f1, 6),
        "clean_recall_suspicious":   round(n_tp / max(n_suspicious, 1), 6),
        "post_recall_suspicious":    round((n_tp - n_evaded) / max(n_suspicious, 1), 6),
        "mean_confidence_pre":       round(float(conf_pre.mean()), 6),
        "mean_confidence_post":      round(float(conf_post.mean()), 6),
        "mean_confidence_drop":      round(float((conf_pre - conf_post).mean()), 6),
        "mean_new_edges_per_graph":  round(float(np.mean(new_edge_counts)), 2),
        "per_graph": {
            "subgraph_indices": tp_indices.tolist(),
            "evaded":           pg_evaded,
            "confidence_pre":   conf_pre_list,
            "confidence_post":  conf_post_list,
        },
    }


def _print_summary(results: dict, args) -> None:
    max_conn = max(args.connections_per_node)

    print(f"\n\n{'='*70}")
    print(f"  NODE INJECTION — ASR  [conn={max_conn}, full test set]")
    print(f"{'='*70}")

    k_header = "".join(f"  k={k:<5}" for k in args.k_nodes)
    header   = f"{'Model':<14}{'Strategy':<10}" + k_header
    print(header)
    print("-" * len(header))

    for ct in args.conv_types:
        if ct not in results["models"]:
            continue
        combos = results["models"][ct]["combinations"]
        for strategy in args.strategies:
            row = f"{ARCH_NAMES[ct]:<14}{strategy:<10}"
            for k in args.k_nodes:
                key = f"{strategy}_k{k}_c{max_conn}"
                asr = combos.get(key, {}).get("asr", float("nan"))
                row += f"  {asr * 100:>6.2f}%"
            print(row)
        print()

    print(f"\n{'='*70}")
    print(f"  NODE INJECTION — Mean confidence drop  [conn={max_conn}]")
    print(f"{'='*70}")
    print(header)
    print("-" * len(header))

    for ct in args.conv_types:
        if ct not in results["models"]:
            continue
        combos = results["models"][ct]["combinations"]
        for strategy in args.strategies:
            row = f"{ARCH_NAMES[ct]:<14}{strategy:<10}"
            for k in args.k_nodes:
                key  = f"{strategy}_k{k}_c{max_conn}"
                drop = combos.get(key, {}).get("mean_confidence_drop", float("nan"))
                row += f"  {drop:>+7.4f}"
            print(row)
        print()

    k_mid = 5 if 5 in args.k_nodes else args.k_nodes[len(args.k_nodes) // 2]
    print(f"\n{'='*70}")
    print(f"  NODE INJECTION — Strategy comparison  [k={k_mid}, conn={max_conn}]")
    print(f"{'='*70}")
    h3 = f"{'Model':<14}{'Strategy':<10}  {'ASR':>7}  {'ΔConf':>8}  {'ΔF1':>8}"
    print(h3)
    print("-" * len(h3))

    for ct in args.conv_types:
        if ct not in results["models"]:
            continue
        combos = results["models"][ct]["combinations"]
        for strategy in args.strategies:
            key   = f"{strategy}_k{k_mid}_c{max_conn}"
            combo = combos.get(key, {})
            asr   = combo.get("asr", float("nan"))
            drop  = combo.get("mean_confidence_drop", float("nan"))
            f1d   = combo.get("f1_drop", float("nan"))
            print(
                f"{ARCH_NAMES[ct]:<14}{strategy:<10}  "
                f"{asr*100:>6.2f}%  {drop:>+8.4f}  {f1d:>+8.4f}"
            )
        print()


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default="config.yaml")
    pre_args, _ = pre.parse_known_args()
    config  = load_yaml_config(pre_args.config)
    inj_cfg = config.get("node_injection_attack", {})

    parser = argparse.ArgumentParser(
        description="Domain-Specific Node Injection Attack on Elliptic2 GNN Models"
    )
    parser.add_argument("--config",               default="config.yaml")
    parser.add_argument("--seed",                 type=int,
                        default=inj_cfg.get("seed", 42))
    parser.add_argument("--k_nodes",              nargs="+", type=int,
                        default=inj_cfg.get("k_nodes", [1, 3, 5, 10]))
    parser.add_argument("--connections_per_node", nargs="+", type=int,
                        default=inj_cfg.get("connections_per_node", [1, 2, 3]))
    parser.add_argument("--strategies",           nargs="+",
                        default=inj_cfg.get("strategies", ["random", "degree", "mimicry"]),
                        choices=["random", "degree", "mimicry"])
    parser.add_argument("--conv_types",           nargs="+",
                        default=inj_cfg.get("conv_types", ALL_CONV_TYPES),
                        choices=ALL_CONV_TYPES)
    parser.add_argument("--injected_is_original", type=float, default=0.0,
                        help="Provenance flag carried by injected nodes. 0.0 (default) "
                             "makes them look like k-hop expansion context, which the "
                             "model discounts at readout via expansion_node_weight; "
                             "1.0 marks them as original cluster nodes. "
                             "Values <= 0.5 behave as 0.0.")
    parser.add_argument("--output",
                        default=inj_cfg.get("output", "data/results/node_injection_results.json"))
    parser.add_argument("--device", default=None)
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
        "experiment":           "node_injection_attack",
        "seed":                 args.seed,
        "k_nodes":              args.k_nodes,
        "connections_per_node": args.connections_per_node,
        "strategies":           args.strategies,
        "injected_is_original": args.injected_is_original,
        "checkpoint_dir":       checkpoint_dir,
        "node_feat_dim":        node_feat_dim,
        "edge_feat_dim":        edge_feat_dim,
        "models":               {},
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
            f"{len(tp_indices)/max(n_sus,1):.3f}  "
            f"Attacking {len(tp_indices)} TP subgraphs"
        )

        model_results = {"arch": arch_name, "combinations": {}}
        total_combos  = len(args.strategies) * len(args.k_nodes) * len(args.connections_per_node)
        combo_num     = 0

        for strategy in args.strategies:
            for k in args.k_nodes:
                for conn in args.connections_per_node:
                    combo_num += 1
                    combo_key  = f"{strategy}_k{k}_c{conn}"
                    print(
                        f"\n  [{combo_num}/{total_combos}] [{arch_name}] "
                        f"strategy={strategy}  k={k}  conn={conn}"
                    )
                    t0 = time.time()

                    attacker = NodeInjectionAttack(
                        licit_dist=licit_dist,
                        strategy=strategy,
                        k_nodes=k,
                        connections_per_node=conn,
                        injected_is_original=args.injected_is_original,
                    )

                    stats = run_injection_for_combo(
                        model=model,
                        test_dataset=test_dataset,
                        attacker=attacker,
                        tp_indices=tp_indices,
                        labels=labels,
                        preds_clean=preds_clean,
                        probs_clean=probs_clean,
                        device=device,
                    )
                    stats["strategy"]             = strategy
                    stats["k_nodes"]              = k
                    stats["connections_per_node"] = conn
                    stats["time_seconds"]         = round(time.time() - t0, 2)

                    model_results["combinations"][combo_key] = stats

                    print(
                        f"    ASR={stats['asr']:.2%} "
                        f"({stats['n_evaded']}/{stats['n_attacked']})  "
                        f"F1: {stats['clean_f1_macro']:.4f}→{stats['post_f1_macro']:.4f} "
                        f"({stats['f1_drop']:+.4f})  "
                        f"Conf drop: {stats['mean_confidence_drop']:+.4f}  "
                        f"Avg new edges: {stats['mean_new_edges_per_graph']:.1f}  "
                        f"Time: {stats['time_seconds']:.1f}s"
                    )

        results["models"][conv_type] = model_results

    _print_summary(results, args)
    save_results(results, args.output)


if __name__ == "__main__":
    main()
