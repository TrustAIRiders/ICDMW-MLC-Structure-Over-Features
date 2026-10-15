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
from src.attacks.zoo_attack import ModelOracle, ZOOAttack
from src.attacks.transfer_attack import get_clean_predictions, get_tp_indices


def asr_at_budget(success_at_query_list: list, n_attacked: int, budget: int) -> float:
    if n_attacked == 0:
        return 0.0
    n_success = sum(1 for q in success_at_query_list if 0 < q <= budget)
    return round(n_success / n_attacked, 6)


def run_zoo_for_model(
    model,
    test_dataset,
    attacker: ZOOAttack,
    tp_indices: np.ndarray,
    labels: np.ndarray,
    preds_clean: np.ndarray,
    probs_clean: np.ndarray,
    query_budgets: list,
    device,
) -> dict:
    n_suspicious = int((labels == 1).sum())
    n_tp         = len(tp_indices)
    clean_f1     = float(f1_score(labels, preds_clean, average="macro", zero_division=0))

    preds_post = preds_clean.copy()
    probs_post = probs_clean.copy()

    n_evaded      = 0
    pg_conf_pre   = []
    pg_conf_post  = []
    pg_l2         = []
    pg_queries    = []
    pg_success_at = []
    pg_evaded     = []

    for idx in tqdm(tp_indices, desc="  Attacking", leave=False):
        data = test_dataset[int(idx)]

        perturbed, _success, queries_used, success_at_q = attacker.attack(data)

        with torch.no_grad():
            adv = perturbed.to(device)
            if not hasattr(adv, "batch") or adv.batch is None:
                adv.batch = torch.zeros(adv.num_nodes, dtype=torch.long, device=device)
            logits_post = model(adv)
            probs_post_t, preds_post_t = decide(model, logits_post)
            prob_post   = float(probs_post_t[0].item())
            pred_post   = int(preds_post_t[0].item())

        preds_post[int(idx)] = pred_post
        probs_post[int(idx)] = prob_post

        evaded = 1 if pred_post == 0 else 0
        n_evaded += evaded
        pg_evaded.append(evaded)

        delta    = (perturbed.x.cpu() - data.x).abs()
        graph_l2 = float(delta.pow(2).sum().sqrt().item())

        pg_conf_pre.append(float(probs_clean[int(idx)]))
        pg_conf_post.append(prob_post)
        pg_l2.append(graph_l2)
        pg_queries.append(queries_used)
        pg_success_at.append(success_at_q)

    post_f1 = float(f1_score(labels, preds_post, average="macro", zero_division=0))

    successful_q     = [q for q in pg_success_at if q > 0]
    mean_q_success   = float(np.mean(successful_q))   if successful_q else 0.0
    median_q_success = float(np.median(successful_q)) if successful_q else 0.0

    conf_pre_arr  = np.array(pg_conf_pre)
    conf_post_arr = np.array(pg_conf_post)

    return {
        "n_suspicious":              n_suspicious,
        "n_attacked":                n_tp,
        "n_evaded":                  n_evaded,
        "asr":                       round(n_evaded / max(n_tp, 1), 6),
        "asr_at_queries":            {str(b): asr_at_budget(pg_success_at, n_tp, b) for b in query_budgets},
        "mean_queries_used":         round(float(np.mean(pg_queries)), 2),
        "median_queries_used":       round(float(np.median(pg_queries)), 2),
        "mean_queries_to_success":   round(mean_q_success, 2),
        "median_queries_to_success": round(median_q_success, 2),
        "clean_f1_macro":            round(clean_f1, 6),
        "post_f1_macro":             round(post_f1, 6),
        "f1_drop":                   round(post_f1 - clean_f1, 6),
        "clean_recall_suspicious":   round(n_tp / max(n_suspicious, 1), 6),
        "post_recall_suspicious":    round((n_tp - n_evaded) / max(n_suspicious, 1), 6),
        "mean_confidence_pre":       round(float(conf_pre_arr.mean()), 6),
        "mean_confidence_post":      round(float(conf_post_arr.mean()), 6),
        "mean_confidence_drop":      round(float((conf_pre_arr - conf_post_arr).mean()), 6),
        "mean_l2_perturbation":      round(float(np.mean(pg_l2)), 6),
        "per_graph": {
            "subgraph_indices":  tp_indices.tolist(),
            "evaded":            pg_evaded,
            "queries_used":      pg_queries,
            "success_at_query":  pg_success_at,
            "confidence_pre":    pg_conf_pre,
            "confidence_post":   pg_conf_post,
            "l2_perturbation":   pg_l2,
        },
    }


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default="config.yaml")
    pre_args, _ = pre.parse_known_args()
    config = load_yaml_config(pre_args.config)
    zoo    = config.get("zoo_attack", {})

    parser = argparse.ArgumentParser(
        description="ZOO Black-Box Feature Attack on Elliptic2 GNN Models"
    )
    parser.add_argument("--config",        default="config.yaml")
    parser.add_argument("--seed",          type=int,              default=zoo.get("seed", 42))
    parser.add_argument("--epsilons",      nargs="+", type=float, default=zoo.get("epsilons", [0.05, 0.1, 0.2]))
    parser.add_argument("--delta",         type=float,            default=zoo.get("delta", 0.01))
    parser.add_argument("--subspace_dim",  type=int,              default=zoo.get("subspace_dim", 10))
    parser.add_argument("--step_size",     type=float,            default=zoo.get("step_size", 0.01))
    parser.add_argument("--max_queries",   type=int,              default=zoo.get("max_queries", 2000))
    parser.add_argument("--query_budgets", nargs="+", type=int,   default=zoo.get("query_budgets", [100, 250, 500, 1000, 2000]))
    parser.add_argument("--conv_types",    nargs="+",             default=zoo.get("conv_types", ALL_CONV_TYPES), choices=ALL_CONV_TYPES)
    parser.add_argument("--output",                               default=zoo.get("output", "data/results/zoo_attack_results.json"))
    parser.add_argument("--device",        default=None)
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

    print("Computing feature bounds ")
    feat_min, feat_max = compute_feat_bounds(train_dataset)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    results = {
        "experiment":    "zoo_feature_attack",
        "seed":          args.seed,
        "delta":         args.delta,
        "subspace_dim":  args.subspace_dim,
        "step_size":     args.step_size,
        "max_queries":   args.max_queries,
        "query_budgets": args.query_budgets,
        "epsilons":      args.epsilons,
        "node_feat_dim": node_feat_dim,
        "models":        {},
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
        print(f"  Clean F1={clean_f1:.4f}  "
              f"Recall(susp)={len(tp_indices)}/{n_sus}={len(tp_indices)/max(n_sus,1):.3f}  "
              f"Attacking {len(tp_indices)} TP subgraphs")

        oracle        = ModelOracle(model, device, threshold=model.threshold)
        model_results = {"arch": arch_name, "epsilon_results": {}}

        for epsilon in args.epsilons:
            print(f"\n  ε={epsilon}  δ={args.delta}  k={args.subspace_dim}  "
                  f"step={args.step_size}  budget={args.max_queries}")
            t0 = time.time()

            attacker = ZOOAttack(
                oracle=oracle,
                epsilon=epsilon,
                delta=args.delta,
                subspace_dim=args.subspace_dim,
                max_queries=args.max_queries,
                step_size=args.step_size,
                feat_min=feat_min,
                feat_max=feat_max,
            )

            stats = run_zoo_for_model(
                model=model,
                test_dataset=test_dataset,
                attacker=attacker,
                tp_indices=tp_indices,
                labels=labels,
                preds_clean=preds_clean,
                probs_clean=probs_clean,
                query_budgets=args.query_budgets,
                device=device,
            )
            stats["time_seconds"] = round(time.time() - t0, 2)
            model_results["epsilon_results"][str(epsilon)] = stats

            print(f"    ASR (full budget) : {stats['asr']:.2%} "
                  f"({stats['n_evaded']}/{stats['n_attacked']})")
            budget_str = "  ".join(
                f"@{b}:{stats['asr_at_queries'][str(b)]:.2%}" for b in args.query_budgets
            )
            print(f"    ASR curve         : {budget_str}")
            print(f"    Mean Q used       : {stats['mean_queries_used']:.0f}  "
                  f"Median: {stats['median_queries_used']:.0f}")
            print(f"    Mean Q to success : {stats['mean_queries_to_success']:.0f}  "
                  f"Median: {stats['median_queries_to_success']:.0f}")
            print(f"    F1 drop           : {stats['clean_f1_macro']:.4f} → "
                  f"{stats['post_f1_macro']:.4f} ({stats['f1_drop']:+.4f})")
            print(f"    Recall drop       : {stats['clean_recall_suspicious']:.4f} → "
                  f"{stats['post_recall_suspicious']:.4f}")
            print(f"    Conf drop         : {stats['mean_confidence_pre']:.4f} → "
                  f"{stats['mean_confidence_post']:.4f}")
            print(f"    Mean L2           : {stats['mean_l2_perturbation']:.4f}  "
                  f"Time: {stats['time_seconds']:.1f}s")

        results["models"][conv_type] = model_results

    print(f"\n\n{'='*60}")
    print("  ZOO ATTACK — ASR per (Model × ε)  [full query budget]")
    print(f"{'='*60}")
    col_w  = 9
    header = f"{'Model':<14}" + "".join(f"  ε={e:<{col_w-3}}" for e in args.epsilons)
    print(header)
    print("-" * len(header))
    for ct in args.conv_types:
        if ct not in results["models"]:
            continue
        row = f"{ARCH_NAMES[ct]:<14}"
        for eps in args.epsilons:
            asr = results["models"][ct]["epsilon_results"].get(str(eps), {}).get("asr", float("nan"))
            row += f"  {asr * 100:>{col_w-1}.2f}%"
        print(row)

    mid_eps = str(args.epsilons[len(args.epsilons) // 2])
    print(f"\n  Query efficiency at ε={mid_eps}")
    bw      = 7
    header2 = f"  {'Model':<14}" + "".join(f"  {'@'+str(b):>{bw}}" for b in args.query_budgets)
    print(header2)
    print("  " + "-" * (len(header2) - 2))
    for ct in args.conv_types:
        if ct not in results["models"]:
            continue
        eps_res = results["models"][ct]["epsilon_results"].get(mid_eps, {})
        if not eps_res:
            continue
        row = f"  {ARCH_NAMES[ct]:<14}"
        row += "".join(
            f"  {eps_res['asr_at_queries'].get(str(b), 0.0) * 100:>{bw}.1f}%"
            for b in args.query_budgets
        )
        print(row)

    save_results(results, args.output)


if __name__ == "__main__":
    main()
