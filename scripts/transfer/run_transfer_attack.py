import os
import sys
import time
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm

from scripts.utils import (
    ALL_CONV_TYPES, ARCH_NAMES,
    load_yaml_config, resolve_device,
    load_model, load_datasets, get_feat_dims,
    compute_feat_bounds, save_results,
)
from src.models.surrogate import SurrogateGCN, train_surrogate
from src.attacks.transfer_attack import TransferAttack, get_clean_predictions, get_tp_indices


def run_epsilon(
        surrogate_model,
        target_models: dict,
        test_dataset,
        epsilon,
        steps,
        restarts,
        step_size,
        feat_min,
        feat_max,
        device,
        target_clean_preds,
):
    t0 = time.time()

    labels, surr_preds, _ = get_clean_predictions(surrogate_model, test_dataset, device)
    tp_indices = get_tp_indices(labels, surr_preds)

    n_tp  = len(tp_indices)
    n_sus = int((labels == 1).sum())
    print(f"    Surrogate recall: {n_tp}/{n_sus}={n_tp/max(n_sus,1):.3f}  "
          f"attacking {n_tp} TP graphs (ε={epsilon})")

    if n_tp == 0:
        empty = {
            "n_suspicious": n_sus, "n_tp_surrogate": 0,
            "n_attacked": 0, "n_evaded": 0, "transfer_asr": 0.0,
            "clean_f1_macro": 0.0, "post_f1_macro": 0.0, "f1_drop": 0.0,
            "clean_recall_suspicious": 0.0, "post_recall_suspicious": 0.0,
            "mean_confidence_pre": 0.0, "mean_confidence_post": 0.0,
            "mean_confidence_drop": 0.0, "mean_l2_perturbation": 0.0,
        }
        return {ct: empty for ct in target_models}

    attacker = TransferAttack(
        surrogate=surrogate_model,
        epsilon=epsilon,
        steps=steps,
        step_size=step_size,
        num_restarts=restarts,
        feat_min=feat_min,
        feat_max=feat_max,
    )

    perturbed = attacker.generate_perturbations(test_dataset, tp_indices)
    print(f"    Perturbations generated in {time.time() - t0:.1f}s")

    target_results = {}
    for conv_type, target_model in target_models.items():
        t1 = time.time()
        lbl, preds_clean, probs_clean = target_clean_preds[conv_type]
        metrics = attacker.evaluate_transfer(
            target_model=target_model,
            test_dataset=test_dataset,
            tp_indices=tp_indices,
            perturbed_graphs=perturbed,
            labels=lbl,
            preds_clean=preds_clean,
            probs_clean=probs_clean,
        )
        metrics["time_seconds"]          = round(time.time() - t1, 2)
        metrics["tp_indices_surrogate"]  = tp_indices.tolist()
        target_results[conv_type]        = metrics

        print(f"      → {ARCH_NAMES.get(conv_type, conv_type):<12} "
              f"Transfer ASR={metrics['transfer_asr']:.2%}  "
              f"F1 drop={metrics['f1_drop']:+.4f}  "
              f"Conf drop={metrics['mean_confidence_drop']:.4f}")

    return target_results


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default="config.yaml")
    pre_args, _ = pre.parse_known_args()

    config = load_yaml_config(pre_args.config)
    ta     = config.get("transfer_attack", {})

    parser = argparse.ArgumentParser(
        description="Transfer (Gray-Box) Attack on Elliptic2 GNN Models"
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--seed",   type=int, default=ta.get("seed", 42))
    parser.add_argument("--output", default=ta.get("output", "data/results/transfer_attack_results.json"))
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    seed                 = args.seed
    epsilons             = ta.get("epsilons", [0.05, 0.1, 0.2])
    steps                = ta.get("steps", 40)
    restarts             = ta.get("restarts", 5)
    step_size_cfg        = ta.get("step_size", None)
    surrogate_hidden_dim = ta.get("surrogate_hidden_dim", 128)
    surrogate_epochs     = ta.get("surrogate_epochs", 50)
    surrogate_lr         = ta.get("surrogate_lr", 0.001)
    surrogate_batch_size = ta.get("surrogate_batch_size", 256)
    surrogate_checkpoint = ta.get("surrogate_checkpoint", "checkpoints/surrogate_gcn.pt")
    cross_arch           = ta.get("cross_arch", True)
    conv_types           = ta.get("conv_types", ALL_CONV_TYPES)
    output               = args.output

    device = resolve_device(args.device if args.device else config.get("device"))
    print(f"Device: {device}")

    checkpoint_dir = config["training"]["checkpoint_dir"]

    print("Loading datasets ")
    datasets      = load_datasets(config, splits=("train", "val", "test"))
    train_dataset = datasets["train"]
    val_dataset   = datasets["val"]
    test_dataset  = datasets["test"]

    node_feat_dim, edge_feat_dim = get_feat_dims(train_dataset)
    print(f"Node features: {node_feat_dim} | Edge features: {edge_feat_dim}")

    print("Computing feature bounds")
    feat_min, feat_max = compute_feat_bounds(train_dataset)

    surrogate = SurrogateGCN(
        node_feat_dim=node_feat_dim,
        hidden_dim=surrogate_hidden_dim,
        num_classes=config["model"]["num_classes"],
    )

    if os.path.exists(surrogate_checkpoint):
        print(f"Loading cached surrogate from {surrogate_checkpoint}")
        state = torch.load(surrogate_checkpoint, map_location=device, weights_only=True)
        surrogate.load_state_dict(state)
        surrogate.to(device).eval()
    else:
        print(f"Training SurrogateGCN ({surrogate_epochs} epochs)")
        surrogate = train_surrogate(
            surrogate=surrogate,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            epochs=surrogate_epochs,
            lr=surrogate_lr,
            batch_size=surrogate_batch_size,
            device=str(device),
            verbose=True,
        )
        os.makedirs(os.path.dirname(os.path.abspath(surrogate_checkpoint)), exist_ok=True)
        torch.save(surrogate.state_dict(), surrogate_checkpoint)
        print(f"Surrogate saved → {surrogate_checkpoint}")

    print(f"Surrogate params: {surrogate.count_params():,}")

    print("Loading target models ")
    target_models = {}
    for conv_type in conv_types:
        try:
            target_models[conv_type] = load_model(
                conv_type, checkpoint_dir, seed,
                node_feat_dim, edge_feat_dim, config, device,
            )
            print(f"  Loaded {ARCH_NAMES[conv_type]}")
        except FileNotFoundError as e:
            print(f"  SKIP {conv_type}: {e}")

    if not target_models:
        raise RuntimeError("No target models loaded — check checkpoint_dir and seed.")

    print("Computing clean predictions for all targets")
    target_clean_preds = {}
    for conv_type, model in target_models.items():
        lbl, preds, probs = get_clean_predictions(model, test_dataset, device)
        target_clean_preds[conv_type] = (lbl, preds, probs)
        n_sus = int((lbl == 1).sum())
        n_tp  = int(((lbl == 1) & (preds == 1)).sum())
        f1    = float(f1_score(lbl, preds, average="macro", zero_division=0))
        print(f"  {ARCH_NAMES[conv_type]:<12}  F1={f1:.4f}  "
              f"Recall(susp)={n_tp}/{n_sus}={n_tp/max(n_sus,1):.3f}")

    results = {
        "experiment":        "transfer_feature_attack",
        "seed":              seed,
        "surrogate":         f"GCN-2layer-h{surrogate_hidden_dim}",
        "pgd_steps":         steps,
        "num_restarts":      restarts,
        "epsilons":          epsilons,
        "node_feat_dim":     node_feat_dim,
        "gcn_to_targets":    {},
        "cross_arch_matrix": {},
    }

    print("\n=== A) GCN Surrogate → All Targets ===")
    for epsilon in epsilons:
        step_size = step_size_cfg if step_size_cfg is not None else 2.5 * epsilon / steps
        print(f"\n  ε = {epsilon}")

        results["gcn_to_targets"][str(epsilon)] = run_epsilon(
            surrogate_model=surrogate,
            target_models=target_models,
            test_dataset=test_dataset,
            epsilon=epsilon,
            steps=steps,
            restarts=restarts,
            step_size=step_size,
            feat_min=feat_min,
            feat_max=feat_max,
            device=device,
            target_clean_preds=target_clean_preds,
        )

    if cross_arch:
        print("\n=== B) Cross-Architecture Transfer Matrix ===")
        for surr_conv in conv_types:
            if surr_conv not in target_models:
                continue

            surr_model    = target_models[surr_conv]
            other_targets = {ct: m for ct, m in target_models.items() if ct != surr_conv}
            other_preds   = {ct: v for ct, v in target_clean_preds.items() if ct != surr_conv}

            print(f"\n  Surrogate: {ARCH_NAMES[surr_conv]}")
            results["cross_arch_matrix"][surr_conv] = {}

            for epsilon in epsilons:
                step_size = step_size_cfg if step_size_cfg is not None else 2.5 * epsilon / steps
                print(f"  ε = {epsilon}")
                results["cross_arch_matrix"][surr_conv][str(epsilon)] = run_epsilon(
                    surrogate_model=surr_model,
                    target_models=other_targets,
                    test_dataset=test_dataset,
                    epsilon=epsilon,
                    steps=steps,
                    restarts=restarts,
                    step_size=step_size,
                    feat_min=feat_min,
                    feat_max=feat_max,
                    device=device,
                    target_clean_preds=other_preds,
                )

    col_w  = 9
    header = f"{'Target':<14}" + "".join(f"  ε={e:<{col_w-3}}" for e in epsilons)
    print("\n\n=== Transfer ASR: GCN Surrogate → Targets ===")
    print(header)
    print("-" * len(header))
    for conv_type in conv_types:
        if conv_type not in target_models:
            continue
        row = f"{ARCH_NAMES[conv_type]:<14}"
        for epsilon in epsilons:
            asr = (results["gcn_to_targets"]
                   .get(str(epsilon), {})
                   .get(conv_type, {})
                   .get("transfer_asr", float("nan")))
            row += f"  {asr * 100:>{col_w-1}.2f}%"
        print(row)

    if cross_arch and results["cross_arch_matrix"]:
        mid_eps      = str(epsilons[len(epsilons) // 2])
        header_label = "Surrogate \\ Target"
        print(f"\n=== Cross-Arch Transfer ASR Matrix (ε={mid_eps}) ===")
        print(f"{header_label:<16}", end="")
        for ct in conv_types:
            if ct in target_models:
                print(f"  {ARCH_NAMES[ct]:<12}", end="")
        print()
        print("-" * (16 + 14 * len(target_models)))
        for surr_conv in conv_types:
            if surr_conv not in results["cross_arch_matrix"]:
                continue
            print(f"  {ARCH_NAMES[surr_conv]:<14}", end="")
            eps_data = results["cross_arch_matrix"][surr_conv].get(mid_eps, {})
            for tgt_conv in conv_types:
                if tgt_conv == surr_conv or tgt_conv not in target_models:
                    print(f"  {'—':<12}", end="")
                else:
                    asr = eps_data.get(tgt_conv, {}).get("transfer_asr", float("nan"))
                    print(f"  {asr*100:>6.2f}%      ", end="")
            print()

    save_results(results, output)


if __name__ == "__main__":
    main()
