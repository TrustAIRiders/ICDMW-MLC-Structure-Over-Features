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
from src.analysis.feature_importance import FeatureImportanceAnalyzer
from src.models.calibration import decide
from src.attacks.saliency_attack import SaliencyGuidedAttack
from src.attacks.transfer_attack import get_clean_predictions, get_tp_indices


def run_attack_for_model(model, test_dataset, attacker, device) -> dict:
    labels, preds_clean, probs_clean = get_clean_predictions(model, test_dataset, device)
    tp_indices   = get_tp_indices(labels, preds_clean)
    n_suspicious = int((labels == 1).sum())
    n_tp         = len(tp_indices)
    clean_f1     = float(f1_score(labels, preds_clean, average="macro", zero_division=0))

    if n_tp == 0:
        return _empty_result(n_suspicious, clean_f1)

    preds_post = preds_clean.copy()
    probs_post = probs_clean.copy()

    n_evaded             = 0
    total_l2             = 0.0
    pg_conf_pre, pg_conf_post, pg_l2 = [], [], []
    pg_evaded            = []

    for idx in tqdm(tp_indices, desc="    Attacking", leave=False):
        data      = test_dataset[int(idx)]
        perturbed = attacker.attack(data)

        with torch.no_grad():
            logits_post = model(perturbed.to(device))
            probs_post_t, preds_post_t = decide(model, logits_post)
            prob_post   = float(probs_post_t[0].item())
            pred_post   = int(preds_post_t[0].item())

        preds_post[idx] = pred_post
        probs_post[idx] = prob_post

        evaded = 1 if pred_post == 0 else 0
        n_evaded += evaded
        pg_evaded.append(evaded)

        delta    = (perturbed.x.cpu() - data.x).abs()
        graph_l2 = float(delta.pow(2).sum().sqrt().item())
        total_l2 += graph_l2

        pg_conf_pre.append(float(probs_clean[idx]))
        pg_conf_post.append(prob_post)
        pg_l2.append(graph_l2)

    post_f1              = float(f1_score(labels, preds_post, average="macro", zero_division=0))
    attacked_probs_clean = probs_clean[tp_indices]
    attacked_probs_post  = probs_post[tp_indices]

    return {
        "n_suspicious":            n_suspicious,
        "n_attacked":              n_tp,
        "n_evaded":                n_evaded,
        "asr":                     round(n_evaded / n_tp, 6),
        "clean_accuracy_attacked": 1.0,
        "post_accuracy_attacked":  round((n_tp - n_evaded) / n_tp, 6),
        "clean_f1_macro":          round(clean_f1, 6),
        "post_f1_macro":           round(post_f1, 6),
        "clean_recall_suspicious": round(n_tp / n_suspicious, 6),
        "post_recall_suspicious":  round((n_tp - n_evaded) / n_suspicious, 6),
        "mean_confidence_pre":     round(float(attacked_probs_clean.mean()), 6),
        "mean_confidence_post":    round(float(attacked_probs_post.mean()), 6),
        "mean_confidence_drop":    round(
            float(np.mean([pre - post for pre, post in zip(pg_conf_pre, pg_conf_post)])), 6
        ),
        "mean_l2_perturbation":    round(total_l2 / n_tp, 6),
        "per_graph": {
            "subgraph_indices": tp_indices.tolist(),
            "evaded":           pg_evaded,
            "confidence_pre":   pg_conf_pre,
            "confidence_post":  pg_conf_post,
            "l2_perturbation":  pg_l2,
        },
    }


def _empty_result(n_suspicious: int, clean_f1: float) -> dict:
    return {
        "n_suspicious":            n_suspicious,
        "n_attacked":              0,
        "n_evaded":                0,
        "asr":                     0.0,
        "clean_accuracy_attacked": 1.0,
        "post_accuracy_attacked":  1.0,
        "clean_f1_macro":          round(clean_f1, 6),
        "post_f1_macro":           round(clean_f1, 6),
        "clean_recall_suspicious": 0.0,
        "post_recall_suspicious":  0.0,
        "mean_confidence_pre":     0.0,
        "mean_confidence_post":    0.0,
        "mean_confidence_drop":    0.0,
        "mean_l2_perturbation":    0.0,
    }


def print_asr_table(results: dict, conv_types: list, epsilons: list,
                    top_k_values: list) -> None:
    print("\n\n=== ASR: Model x top_k x ε ===")
    for conv_type in conv_types:
        if conv_type not in results["models"]:
            continue
        arch     = ARCH_NAMES.get(conv_type, conv_type)
        topk_res = results["models"][conv_type]["topk_results"]
        print(f"\n  {arch}:")
        header = f"  {'top_k':>6}" + "".join(f"  ε={e:<6}" for e in epsilons)
        print(header)
        print("  " + "-" * (len(header) - 2))
        for top_k in sorted(top_k_values):
            row = f"  {top_k:>6}"
            for epsilon in epsilons:
                asr = (topk_res
                       .get(str(top_k), {})
                       .get("epsilon_results", {})
                       .get(str(epsilon), {})
                       .get("asr", float("nan")))
                row += f"  {asr * 100:>6.2f}%"
            print(row)


def print_efficiency_table(results: dict, conv_types: list, epsilons: list,
                           top_k_values: list, node_feat_dim: int) -> None:
    baseline_k = max(top_k_values)
    print(f"\n=== Efficiency: ASR(k) / ASR(k={baseline_k}) ===")
    for conv_type in conv_types:
        if conv_type not in results["models"]:
            continue
        arch     = ARCH_NAMES.get(conv_type, conv_type)
        topk_res = results["models"][conv_type]["topk_results"]
        print(f"\n  {arch}:")
        for epsilon in epsilons:
            base_asr = (topk_res
                        .get(str(baseline_k), {})
                        .get("epsilon_results", {})
                        .get(str(epsilon), {})
                        .get("asr", float("nan")))
            print(f"    ε={epsilon:.2f}  baseline (k={baseline_k}): {base_asr:.2%}")
            for top_k in sorted(top_k_values):
                if top_k >= node_feat_dim:
                    continue
                asr = (topk_res
                       .get(str(top_k), {})
                       .get("epsilon_results", {})
                       .get(str(epsilon), {})
                       .get("asr", float("nan")))
                eff = asr / base_asr if (base_asr > 0 and not np.isnan(base_asr)) \
                    else float("nan")
                print(f"      k={top_k:>2}: {asr:.2%}  efficiency={eff:.2f}×")


def print_feature_agreement(results: dict, conv_types: list) -> None:
    print("\n=== Feature Agreement (top-10 gradient, 1-based indices) ===")
    all_top10 = {}
    for conv_type in conv_types:
        if conv_type not in results["models"]:
            continue
        arch   = ARCH_NAMES.get(conv_type, conv_type)
        top10  = results["models"][conv_type]["feature_importance"][
            "top_features_gradient"
        ][:10]
        all_top10[conv_type] = set(top10)
        print(f"  {arch:12s}: {sorted(top10)}")

    if len(all_top10) >= 2:
        shared = set.intersection(*all_top10.values())
        print(f"\n  Shared by ALL models: {sorted(shared)}")

        models_list = list(all_top10.keys())
        if len(models_list) >= 2:
            print("\n  Pairwise Jaccard similarity (top-10):")
            for i in range(len(models_list)):
                for j in range(i + 1, len(models_list)):
                    a, b     = all_top10[models_list[i]], all_top10[models_list[j]]
                    jaccard  = len(a & b) / len(a | b)
                    print(f"    {ARCH_NAMES[models_list[i]]:12s} ∩ "
                          f"{ARCH_NAMES[models_list[j]]:12s} = "
                          f"{len(a & b)}/10  Jaccard={jaccard:.2f}")


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default="config.yaml")
    pre_args, _ = pre.parse_known_args()
    config  = load_yaml_config(pre_args.config)
    sal_cfg = config.get("saliency_attack", {})

    parser = argparse.ArgumentParser(
        description="Saliency-Guided Feature Attack on Elliptic2 GNN Models"
    )
    parser.add_argument("--config",    default="config.yaml")
    parser.add_argument("--seed",      type=int,
                        default=sal_cfg.get("seed", 42))
    parser.add_argument("--epsilons",  nargs="+", type=float,
                        default=sal_cfg.get("epsilons", [0.1, 0.2]))
    parser.add_argument("--steps",     type=int,
                        default=sal_cfg.get("steps", 40))
    parser.add_argument("--restarts",  type=int,
                        default=sal_cfg.get("restarts", 5))
    parser.add_argument("--top_k_values", nargs="+", type=int,
                        default=sal_cfg.get("top_k_values", [3, 5, 10, 20, 43]))
    parser.add_argument("--saliency_n_samples", type=int,
                        default=sal_cfg.get("saliency_n_samples", 500))
    parser.add_argument("--ig_n_steps", type=int,
                        default=sal_cfg.get("ig_n_steps", 50))
    parser.add_argument("--ig_n_samples", type=int,
                        default=sal_cfg.get("ig_n_samples", 200))
    parser.add_argument("--ablation_n_samples", type=int,
                        default=sal_cfg.get("ablation_n_samples", 200))
    parser.add_argument("--skip_ablation", action="store_true",
                        default=sal_cfg.get("skip_ablation", False))
    parser.add_argument("--importance_method",
                        default=sal_cfg.get("importance_method", "gradient"),
                        choices=["gradient", "ig", "ablation"])
    parser.add_argument("--conv_types", nargs="+",
                        default=sal_cfg.get("conv_types", ALL_CONV_TYPES),
                        choices=ALL_CONV_TYPES)
    parser.add_argument("--output",
                        default=sal_cfg.get("output", "data/results/saliency_attack_results.json"))
    parser.add_argument("--plots_dir",
                        default=sal_cfg.get("plots_dir", "data/results/plots/saliency"))
    parser.add_argument("--device", default=None)
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

    args.top_k_values = sorted(set(min(k, node_feat_dim) for k in args.top_k_values))

    print("Computing feature bounds from training set")
    feat_min, feat_max = compute_feat_bounds(train_dataset)

    results = {
        "experiment":        "saliency_guided_attack",
        "importance_method": args.importance_method,
        "seed":              args.seed,
        "pgd_steps":         args.steps,
        "num_restarts":      args.restarts,
        "epsilons":          args.epsilons,
        "top_k_values":      args.top_k_values,
        "node_feat_dim":     node_feat_dim,
        "models":            {},
    }

    os.makedirs(args.plots_dir, exist_ok=True)

    for conv_type in args.conv_types:
        arch_name = ARCH_NAMES.get(conv_type, conv_type)
        print(f"\n{'=' * 60}")
        print(f"  Architecture: {arch_name}")
        print(f"{'=' * 60}")

        try:
            model = load_model(
                conv_type, checkpoint_dir, args.seed,
                node_feat_dim, edge_feat_dim, config, device,
            )
        except FileNotFoundError as exc:
            print(f"  [SKIP] {exc}")
            continue

        print(f"\n  [Phase 1] Feature importance analysis")
        analyzer = FeatureImportanceAnalyzer(
            model=model, dataset=train_dataset, device=device
        )
        print(f"  Suspicious training samples: {len(analyzer.suspicious_data)}")

        print(f"  Gradient saliency  (max {args.saliency_n_samples} samples)")
        t0          = time.time()
        grad_scores = analyzer.gradient_saliency(max_samples=args.saliency_n_samples)
        print(f"  Done in {time.time() - t0:.1f}s")

        print(f"  Integrated gradients "
              f"(max {args.ig_n_samples} samples x {args.ig_n_steps} steps)")
        t0        = time.time()
        ig_scores = analyzer.integrated_gradients(
            n_steps=args.ig_n_steps, max_samples=args.ig_n_samples
        )
        print(f"  Done in {time.time() - t0:.1f}s")

        if not args.skip_ablation:
            print(f"  Ablation (max {args.ablation_n_samples} samples x "
                  f"{node_feat_dim} features)")
            t0              = time.time()
            ablation_scores = analyzer.ablation_importance(
                max_samples=args.ablation_n_samples
            )
            print(f"  Done in {time.time() - t0:.1f}s")
        else:
            print("  Skipping ablation (--skip_ablation).")
            ablation_scores = np.zeros(node_feat_dim)

        top_gradient = (np.argsort(grad_scores)[::-1] + 1).tolist()
        top_ig       = (np.argsort(ig_scores)[::-1]   + 1).tolist()
        top_ablation = (np.argsort(ablation_scores)[::-1] + 1).tolist()

        print(f"  Top-5 (gradient):  {top_gradient[:5]}")
        print(f"  Top-5 (IG):        {top_ig[:5]}")
        if not args.skip_ablation:
            print(f"  Top-5 (ablation):  {top_ablation[:5]}")

        scores_dict = {"gradient": grad_scores, "ig": ig_scores}
        if not args.skip_ablation:
            scores_dict["ablation"] = ablation_scores

        plot_path = os.path.join(args.plots_dir, f"importance_{conv_type}.png")
        analyzer.plot_importance(
            scores_dict=scores_dict,
            save_path=plot_path,
            arch_name=arch_name,
        )

        feature_importance = {
            "gradient_scores":       grad_scores.tolist(),
            "ig_scores":             ig_scores.tolist(),
            "ablation_scores":       ablation_scores.tolist(),
            "top_features_gradient": top_gradient,
            "top_features_ig":       top_ig,
            "top_features_ablation": top_ablation,
        }

        _score_map    = {"gradient": grad_scores, "ig": ig_scores, "ablation": ablation_scores}
        attack_scores = _score_map[args.importance_method]
        print(f"\n  [Phase 2] Saliency-guided PGD attack "
              f"(mask method: {args.importance_method})")
        topk_results: dict = {}

        for top_k in sorted(args.top_k_values):
            label = "all → PGD baseline" if top_k >= node_feat_dim else f"top-{top_k}"
            print(f"\n  top_k = {top_k}  ({label})")
            topk_results[str(top_k)] = {"epsilon_results": {}}

            for epsilon in args.epsilons:
                step_size = 2.5 * epsilon / args.steps

                attacker = SaliencyGuidedAttack(
                    model=model,
                    importance_scores=attack_scores,
                    top_k=top_k,
                    epsilon=epsilon,
                    steps=args.steps,
                    feat_min=feat_min,
                    feat_max=feat_max,
                    step_size=step_size,
                    num_restarts=args.restarts,
                )

                t0      = time.time()
                stats   = run_attack_for_model(model, test_dataset, attacker, device)
                elapsed = time.time() - t0
                stats["time_seconds"] = round(elapsed, 2)

                topk_results[str(top_k)]["epsilon_results"][str(epsilon)] = stats

                f1_delta     = stats["post_f1_macro"] - stats["clean_f1_macro"]
                recall_delta = stats["post_recall_suspicious"] - stats["clean_recall_suspicious"]
                print(
                    f"    ε={epsilon:.2f} | "
                    f"ASR={stats['asr']:.2%} ({stats['n_evaded']}/{stats['n_attacked']}) | "
                    f"F1Δ={f1_delta:+.4f} | "
                    f"RecallΔ={recall_delta:+.4f} | "
                    f"ConfΔ={-stats['mean_confidence_drop']:+.4f} | "
                    f"L2={stats['mean_l2_perturbation']:.4f} | "
                    f"{elapsed:.1f}s"
                )

        results["models"][conv_type] = {
            "arch":               arch_name,
            "feature_importance": feature_importance,
            "topk_results":       topk_results,
        }

    print_asr_table(results, args.conv_types, args.epsilons, args.top_k_values)
    print_efficiency_table(
        results, args.conv_types, args.epsilons,
        args.top_k_values, node_feat_dim,
    )
    print_feature_agreement(results, args.conv_types)

    save_results(results, args.output)


if __name__ == "__main__":
    main()
