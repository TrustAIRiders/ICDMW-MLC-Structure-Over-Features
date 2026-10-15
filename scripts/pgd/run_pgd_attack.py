import os
import sys
import time
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm

from scripts.utils import (
    ALL_CONV_TYPES, ARCH_NAMES,
    load_yaml_config, resolve_device,
    load_model, load_datasets, get_feat_dims,
    compute_feat_bounds, save_results,
)
from src.models.calibration import decide
from src.attacks.pgd_feature import PGDFeatureAttack
from src.attacks.transfer_attack import get_clean_predictions, get_tp_indices


def run_attack_for_model(model, test_dataset, attacker, device):
    labels, preds_clean, probs_clean = get_clean_predictions(model, test_dataset, device)
    tp_indices = get_tp_indices(labels, preds_clean)

    n_suspicious  = int((labels == 1).sum())
    n_tp          = len(tp_indices)
    clean_f1_macro = float(f1_score(labels, preds_clean, average="macro", zero_division=0))

    print(f"    Suspicious in test set : {n_suspicious}")
    print(f"    Clean recall (susp.)   : {n_tp}/{n_suspicious} = {n_tp/max(n_suspicious,1):.3f}")
    print(f"    Clean F1-macro (full)  : {clean_f1_macro:.4f}")
    print(f"    Attacking {n_tp} TP graphs ")

    if n_tp == 0:
        print("    No TPs found — skipping attack for this epsilon.")
        return {
            "n_suspicious":            n_suspicious,
            "n_attacked":              0,
            "n_evaded":                0,
            "asr":                     0.0,
            "clean_accuracy_attacked": 1.0,
            "post_accuracy_attacked":  1.0,
            "clean_f1_macro":          round(clean_f1_macro, 6),
            "post_f1_macro":           round(clean_f1_macro, 6),
            "clean_recall_suspicious": round(n_tp / max(n_suspicious, 1), 6),
            "post_recall_suspicious":  round(n_tp / max(n_suspicious, 1), 6),
            "mean_confidence_pre":     0.0,
            "mean_confidence_post":    0.0,
            "mean_confidence_drop":    0.0,
            "mean_l2_perturbation":    0.0,
        }

    preds_post = preds_clean.copy()
    probs_post = probs_clean.copy()

    n_evaded     = 0
    total_l2     = 0.0
    conf_drops   = []
    pg_conf_pre  = []
    pg_conf_post = []
    pg_l2        = []
    pg_evaded    = []

    for idx in tqdm(tp_indices, desc="Attacking", leave=False):
        data      = test_dataset[int(idx)]
        perturbed = attacker.attack(data)

        with torch.no_grad():
            pertubed_dev = perturbed.to(device)
            logits_post  = model(pertubed_dev)
            probs_post_t, preds_post_t = decide(model, logits_post)
            prob_post    = float(probs_post_t[0].item())
            pred_post    = int(preds_post_t[0].item())

        preds_post[idx] = pred_post
        probs_post[idx] = prob_post

        evaded = 1 if pred_post == 0 else 0
        n_evaded += evaded
        pg_evaded.append(evaded)

        delta    = (perturbed.x.cpu() - data.x).abs()
        graph_l2 = float(delta.pow(2).sum().sqrt().item())
        total_l2 += graph_l2

        prob_pre = float(probs_clean[idx])
        conf_drops.append(prob_pre - prob_post)
        pg_conf_pre.append(prob_pre)
        pg_conf_post.append(prob_post)
        pg_l2.append(graph_l2)

    n_attacked           = n_tp
    attacked_probs_clean = probs_clean[tp_indices]
    attacked_probs_post  = probs_post[tp_indices]
    post_f1_macro        = float(f1_score(labels, preds_post, average="macro", zero_division=0))

    return {
        "n_suspicious":            n_suspicious,
        "n_attacked":              n_attacked,
        "n_evaded":                n_evaded,
        "asr":                     round(n_evaded / n_attacked, 6),
        "clean_accuracy_attacked": 1.0,
        "post_accuracy_attacked":  round((n_attacked - n_evaded) / n_attacked, 6),
        "clean_f1_macro":          round(clean_f1_macro, 6),
        "post_f1_macro":           round(post_f1_macro, 6),
        "clean_recall_suspicious": round(n_tp / n_suspicious, 6),
        "post_recall_suspicious":  round((n_tp - n_evaded) / n_suspicious, 6),
        "mean_confidence_pre":     round(float(attacked_probs_clean.mean()), 6),
        "mean_confidence_post":    round(float(attacked_probs_post.mean()), 6),
        "mean_confidence_drop":    round(float(np.mean(conf_drops)), 6),
        "mean_l2_perturbation":    round(total_l2 / n_attacked, 6),
        "per_graph": {
            "subgraph_indices": tp_indices.tolist(),
            "evaded":           pg_evaded,
            "confidence_pre":   pg_conf_pre,
            "confidence_post":  pg_conf_post,
            "l2_perturbation":  pg_l2,
        },
    }


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default="config.yaml")
    pre_args, _ = pre.parse_known_args()

    config = load_yaml_config(pre_args.config)
    pgd    = config.get("pgd_attack", {})

    parser = argparse.ArgumentParser(
        description="PGD Node Feature Attack on Elliptic2 GNN Models"
    )
    parser.add_argument("--config",     default="config.yaml")
    parser.add_argument("--seed",       type=int,              default=pgd.get("seed", 42))
    parser.add_argument("--epsilons",   nargs="+", type=float, default=pgd.get("epsilons", [0.01, 0.05, 0.1, 0.2, 0.5]))
    parser.add_argument("--steps",      type=int,              default=pgd.get("steps", 40))
    parser.add_argument("--restarts",   type=int,              default=pgd.get("restarts", 5))
    parser.add_argument("--step_size",  type=float,            default=pgd.get("step_size", None))
    parser.add_argument("--conv_types", nargs="+",             default=pgd.get("conv_types", ALL_CONV_TYPES), choices=ALL_CONV_TYPES)
    parser.add_argument("--output",                            default=pgd.get("output", "data/results/pgd_attack_results.json"))
    parser.add_argument("--device",     default=None)
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"Device: {device}")

    checkpoint_dir = config["training"]["checkpoint_dir"]

    print("Loading dataset")
    datasets      = load_datasets(config)
    train_dataset = datasets["train"]
    test_dataset  = datasets["test"]

    node_feat_dim, edge_feat_dim = get_feat_dims(train_dataset)
    print(f"Node features: {node_feat_dim} | Edge features: {edge_feat_dim}")

    print("Computing feat bounds from training set")
    feat_min, feat_max = compute_feat_bounds(train_dataset)
    print(f"Feat range: min: {feat_min}, max: {feat_max}")

    results = {
        "experiment":    "pgd_feature_attack",
        "seed":          args.seed,
        "pgd_steps":     args.steps,
        "num_restarts":  args.restarts,
        "epsilons":      args.epsilons,
        "node_feat_dim": node_feat_dim,
        "models":        {},
    }

    print("PGD attack")

    for conv_type in args.conv_types:
        arch_name = ARCH_NAMES.get(conv_type, conv_type)
        print(f"Architecture: {arch_name}")

        try:
            model = load_model(
                conv_type, checkpoint_dir, args.seed,
                node_feat_dim, edge_feat_dim, config, device,
            )
            print("Loaded model")
        except FileNotFoundError as e:
            print(e)
            continue

        model_results = {"arch": arch_name, "epsilon_results": {}}

        for epsilon in args.epsilons:
            step_size = args.step_size if args.step_size is not None else 2.5 * epsilon / args.steps
            t0 = time.time()

            attacker = PGDFeatureAttack(
                model=model,
                epsilon=epsilon,
                steps=args.steps,
                step_size=step_size,
                num_restarts=args.restarts,
                feat_min=feat_min,
                feat_max=feat_max,
            )

            stats   = run_attack_for_model(model, test_dataset, attacker, device)
            elapsed = time.time() - t0
            stats["time_seconds"] = round(elapsed, 2)

            model_results["epsilon_results"][str(epsilon)] = stats
            print(f"    ASR            : {stats['asr']:.2%} "
                  f"({stats['n_evaded']}/{stats['n_attacked']})")
            print(f"    F1-macro drop  : {stats['clean_f1_macro']:.4f} → {stats['post_f1_macro']:.4f} "
                  f"(Δ={stats['post_f1_macro']-stats['clean_f1_macro']:+.4f})")
            print(f"    Recall drop    : {stats['clean_recall_suspicious']:.4f} → "
                  f"{stats['post_recall_suspicious']:.4f}")
            print(f"    Conf drop      : {stats['mean_confidence_pre']:.4f} → "
                  f"{stats['mean_confidence_post']:.4f} "
                  f"(Δ={-stats['mean_confidence_drop']:+.4f})")
            print(f"    Mean L2        : {stats['mean_l2_perturbation']:.4f} | "
                  f"Time: {elapsed:.1f}s")

        results["models"][conv_type] = model_results

    print("\n\n=== Summary: Attack Success Rate (ASR) per Model x Epsilon ===")
    col_w  = 9
    header = f"{'Model':<14}" + "".join(f"  ε={e:<{col_w-3}}" for e in args.epsilons)
    print(header)
    print("-" * len(header))
    for conv_type in args.conv_types:
        if conv_type not in results["models"]:
            continue
        row = f"{ARCH_NAMES[conv_type]:<14}"
        for epsilon in args.epsilons:
            asr = results["models"][conv_type]["epsilon_results"].get(
                str(epsilon), {}
            ).get("asr", float("nan"))
            row += f"  {asr * 100:>{col_w-1}.2f}%"
        print(row)

    print("\n=== Summary: F1-macro Drop per Model x Epsilon ===")
    print(header)
    print("-" * len(header))
    for conv_type in args.conv_types:
        if conv_type not in results["models"]:
            continue
        row = f"{ARCH_NAMES[conv_type]:<14}"
        for epsilon in args.epsilons:
            eps_res = results["models"][conv_type]["epsilon_results"].get(str(epsilon), {})
            drop    = eps_res.get("post_f1_macro", float("nan")) - eps_res.get("clean_f1_macro", float("nan"))
            row    += f"  {drop:>+{col_w-1}.4f}"
        print(row)

    save_results(results, args.output)


if __name__ == "__main__":
    main()
