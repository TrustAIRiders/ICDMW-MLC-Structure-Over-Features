import os
import sys
import json
import time
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score, accuracy_score

from scripts.utils import (
    ALL_CONV_TYPES, ARCH_NAMES, load_yaml_config, resolve_device, save_results, load_model,
)
from scripts.defense.run_adv_training import (
    parse_struct_grid, assert_safe_checkpoint_dir, assert_safe_results_dir,
)
from src.dataset.Elliptic2Dataset import Elliptic2Dataset
from src.models.losses import FocalLoss
from src.train.sampler import BalancedBatchSampler, make_weighted_sampler
from src.train.threshold import cost_aware_threshold_search
from src.train.utils import set_all_seeds
from src.defense.adv_train import (
    install_signal_handlers, stop_requested, load_or_compute_feat_bounds,
    pgd_perturb_batch, format_eta, time_left,
)
from src.defense.structural import LicitStats, perturb_structural


@torch.no_grad()
def _forward_probs(model, data):
    return torch.softmax(model(data), dim=-1)[:, 1].cpu().numpy()


def collect_probs(model, loader, device, perturb=None, max_graphs=0):
    model.eval()
    probs, labels, seen = [], [], 0

    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        data = batch if perturb is None else perturb(model, batch)
        probs.append(_forward_probs(model, data))
        labels.append(batch.y.view(-1).cpu().numpy())
        seen += int(batch.num_graphs)
        if max_graphs and seen >= max_graphs:
            break

    return np.concatenate(probs), np.concatenate(labels)


def pgd_perturbation(epsilon, steps, feat_min, feat_max, criterion):
    def perturb(model, batch):
        x_adv = pgd_perturb_batch(
            model, batch, epsilon=epsilon, steps=steps,
            step_size=2.5 * epsilon / max(steps, 1),
            feat_min=feat_min, feat_max=feat_max,
            random_start=True, inner_loss="ce", criterion=criterion,
        )
        clone = batch.clone()
        clone.x = x_adv
        return clone
    return perturb


def structural_perturbation(stats, strategy, k_nodes, connections):
    def perturb(_model, batch):
        return perturb_structural(batch, stats, {
            "strategy": strategy, "k_nodes": k_nodes, "connections": connections,
        })
    return perturb


def metrics_at_threshold(probs, labels, threshold, clean_preds=None):
    preds = (probs >= threshold).astype(int)
    out = {
        "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "acc": float(accuracy_score(labels, preds)),
        "recall_suspicious": float((preds[labels == 1] == 1).mean()) if (labels == 1).any() else 0.0,
    }
    if clean_preds is not None:
        tp_mask = (labels == 1) & (clean_preds == 1)
        n_tp = int(tp_mask.sum())
        out["n_attacked"] = n_tp
        out["asr"] = float((preds[tp_mask] == 0).mean()) if n_tp else 0.0
    return out


def evaluate_att_setting(model, att, loaders, device, criterion, cost, perturbations, max_graphs):
    model.expansion_node_weight = att

    val_probs, val_labels = collect_probs(model, loaders["val"], device)
    if cost["fn"] is not None:
        threshold, _, _ = cost_aware_threshold_search(val_probs, val_labels, cost["fn"], cost["fp"])
    else:
        threshold = 0.5

    clean_probs, clean_labels = collect_probs(model, loaders["test"], device, max_graphs=max_graphs)
    clean_preds = (clean_probs >= threshold).astype(int)

    result = {
        "att": att,
        "threshold": float(threshold),
        "clean": metrics_at_threshold(clean_probs, clean_labels, threshold, clean_preds),
        "attacks": {},
    }

    for name, perturb in perturbations.items():
        probs, labels = collect_probs(model, loaders["test"], device, perturb=perturb, max_graphs=max_graphs)
        result["attacks"][name] = metrics_at_threshold(probs, labels, threshold, clean_preds)

    return result


def train_matched_arm(att, seed, conv_type, config, device, data, study_dir, epochs=None):
    from src.train.train import train_and_evaluate

    arm_config = json.loads(json.dumps(config))
    arm_config["model"]["expansion_node_weight"] = att
    if epochs is not None:
        arm_config["training"]["epochs"] = epochs

    ckpt_dir = os.path.join(study_dir, f"att_{att}", f"seed_{seed}")
    os.makedirs(ckpt_dir, exist_ok=True)

    set_all_seeds(seed)
    return train_and_evaluate(
        conv_type=conv_type,
        config=arm_config,
        device=device,
        train_dataset=data["train_dataset"],
        train_loader=data["train_loader"],
        val_loader=data["val_loader"],
        test_loader=data["test_loader"],
        node_feat_dim=data["node_feat_dim"],
        edge_feat_dim=data["edge_feat_dim"],
        checkpoint_dir=ckpt_dir,
        fresh=False,
    ), ckpt_dir


def main():
    parser = argparse.ArgumentParser(description="Attention (expansion_node_weight) ablation")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--mode", choices=["swap", "matched", "both"], default="both")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--arch", nargs="+", default=ALL_CONV_TYPES, choices=ALL_CONV_TYPES)
    parser.add_argument("--att_values", nargs="+", type=float, default=[1.0, 0.5])

    parser.add_argument("--eval_epsilons", nargs="+", type=float, default=[0.05, 0.1, 0.2])
    parser.add_argument("--pgd_steps", type=int, default=10)
    parser.add_argument("--struct_grid", nargs="+",
                        default=["random:1:1", "degree:5:2", "mimicry:10:3", "degree:1:10"],
                        help="strategy:k:connections — degree:1:10 is a star rewiring (one hub, 10 spokes)")
    parser.add_argument("--max_graphs", type=int, default=0, help="Cap test graphs (0 = all)")

    parser.add_argument("--baseline_dir", default=None, help="Defaults to config training.checkpoint_dir")
    parser.add_argument("--study_dir", default="checkpoints_att_study")
    parser.add_argument("--output", default="data/results/defense/attention_study.json")
    parser.add_argument("--train_epochs", type=int, default=None,
                        help="Override training.epochs for the matched arm")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_hours", type=float, default=0.0)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    struct_grid = parse_struct_grid(args.struct_grid)
    install_signal_handlers()

    config = load_yaml_config(args.config)
    if args.baseline_dir is None:
        args.baseline_dir = config["training"]["checkpoint_dir"]
    assert_safe_checkpoint_dir(args.study_dir, config)
    assert_safe_results_dir(os.path.dirname(os.path.abspath(args.output)), config)

    device = resolve_device(args.device)
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    deadline = time.time() + args.max_hours * 3600 if args.max_hours > 0 else None
    config_att = config["model"].get("expansion_node_weight", 1.0)

    print("=" * 100)
    print("  ATTENTION ABLATION — expansion_node_weight")
    print("=" * 100)
    print(f"  Device        : {device}")
    print(f"  Mode          : {args.mode}")
    print(f"  att values    : {args.att_values}   (config.yaml currently: {config_att})")
    print(f"  Seeds / archs : {seeds} / {[ARCH_NAMES[a] for a in args.arch]}")
    print(f"  Baseline      : {args.baseline_dir} (read-only)")
    print(f"  Study ckpts   : {args.study_dir}/")

    train_cfg = config["training"]
    processed_dir = config["data"]["processed_dir"]
    batch_size = train_cfg["batch_size"]
    val_ratio, test_ratio = train_cfg["val_ratio"], train_cfg["test_ratio"]

    print("\nLoading datasets (read-only)")
    train_dataset = Elliptic2Dataset(processed_dir, split="train", val_ratio=val_ratio, test_ratio=test_ratio)
    val_dataset = Elliptic2Dataset(processed_dir, split="val", val_ratio=val_ratio, test_ratio=test_ratio)
    test_dataset = Elliptic2Dataset(processed_dir, split="test", val_ratio=val_ratio, test_ratio=test_ratio)

    sample = train_dataset[0]
    node_feat_dim, edge_feat_dim = sample.x.shape[1], sample.edge_attr.shape[1]

    loaders = {
        "val": DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True),
        "test": DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=args.num_workers, pin_memory=True),
    }

    os.makedirs(args.study_dir, exist_ok=True)
    feat_min, feat_max = load_or_compute_feat_bounds(
        train_dataset, os.path.join(args.study_dir, "feat_bounds.pt"))
    feat_min, feat_max = feat_min.to(device), feat_max.to(device)
    licit_stats = LicitStats.from_dataset(
        train_dataset, cache_path=os.path.join(args.study_dir, "licit_stats.pt"), device=device)

    criterion = FocalLoss(gamma=2.0, label_smoothing=train_cfg.get("label_smoothing", 0.0))
    cost_matrix = train_cfg.get("cost_matrix", None) or {}
    cost = {"fn": cost_matrix.get("fn_cost", None), "fp": cost_matrix.get("fp_cost", None)}

    perturbations = {}
    for eps in args.eval_epsilons:
        perturbations[f"pgd_eps{eps}"] = pgd_perturbation(eps, args.pgd_steps, feat_min, feat_max, criterion)
    for strategy, k, conn in struct_grid:
        perturbations[f"{strategy}_k{k}_c{conn}"] = structural_perturbation(licit_stats, strategy, k, conn)

    print(f"  Perturbations : {list(perturbations)}")

    records = []
    if os.path.exists(args.output):
        with open(args.output) as f:
            records = json.load(f).get("records", [])
        print(f"  Resuming — {len(records)} records already on disk")

    eval_fingerprint = {
        "pgd_steps": args.pgd_steps,
        "eval_epsilons": sorted(args.eval_epsilons),
        "struct_grid": sorted(f"{s}:{k}:{c}" for s, k, c in struct_grid),
        "max_graphs": args.max_graphs,
        "baseline_dir": os.path.abspath(args.baseline_dir),
        "train_epochs": args.train_epochs,
    }

    def already_done(arm, seed, conv_type, att):
        for r in records:
            if (r["arm"] == arm and r["seed"] == seed and r["conv_type"] == conv_type
                    and r["att"] == att):
                if r.get("eval_fingerprint") == eval_fingerprint:
                    return True
                print(f"  ! existing {arm} record for {conv_type}/seed {seed}/att {att} used a "
                      f"different evaluation protocol — recomputing and replacing it")
                records.remove(r)
                return False
        return False

    def flush():
        save_results({
            "experiment": "attention_ablation_expansion_node_weight",
            "config_att": config_att,
            "att_values": args.att_values,
            "pgd_steps": args.pgd_steps,
            "eval_epsilons": args.eval_epsilons,
            "struct_grid": [f"{s}:{k}:{c}" for s, k, c in struct_grid],
            "records": records,
        }, args.output)

    if args.mode in ("swap", "both"):
        print("\n" + "#" * 100)
        print("#  ARM: swap — one baseline model, evaluated at each att")
        print("#" * 100)

        for seed in seeds:
            for conv_type in args.arch:
                try:
                    model = load_model(conv_type, args.baseline_dir, seed,
                                       node_feat_dim, edge_feat_dim, config, device)
                except FileNotFoundError as e:
                    print(f"  {e} — skipping")
                    continue

                for att in args.att_values:
                    if already_done("swap", seed, conv_type, att):
                        continue
                    t0 = time.time()
                    res = evaluate_att_setting(model, att, loaders, device, criterion,
                                               cost, perturbations, args.max_graphs)
                    res.update({"arm": "swap", "seed": seed, "conv_type": conv_type,
                                "arch": ARCH_NAMES[conv_type], "trained_att": config_att,
                                "eval_fingerprint": eval_fingerprint,
                                "checkpoint": os.path.join(args.baseline_dir, f"seed_{seed}",
                                                           conv_type, "best_model.pt")})
                    records.append(res)
                    flush()
                    print(f"  {ARCH_NAMES[conv_type]:<10} seed {seed} att={att:<4} "
                          f"clean F1={res['clean']['f1_macro']:.4f} "
                          f"({format_eta(time.time() - t0)})")

                if stop_requested() or time_left(deadline) <= 0:
                    print("  Stopping (signal or time budget).")
                    flush()
                    return

    if args.mode in ("matched", "both"):
        print("\n" + "#" * 100)
        print("#  ARM: matched — one model trained per att value, evaluated at its own att")
        print("#" * 100)

        for seed in seeds:
            set_all_seeds(seed)
            if train_cfg.get("balanced_sampling", False):
                train_loader = DataLoader(
                    train_dataset,
                    batch_sampler=BalancedBatchSampler(labels=train_dataset.get_labels(), batch_size=batch_size),
                    num_workers=args.num_workers, pin_memory=True)
            elif train_cfg.get("oversampling", False):
                train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                          sampler=make_weighted_sampler(train_dataset),
                                          num_workers=args.num_workers, pin_memory=True)
            else:
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                          num_workers=args.num_workers, pin_memory=True)

            data = {
                "train_dataset": train_dataset, "train_loader": train_loader,
                "val_loader": loaders["val"], "test_loader": loaders["test"],
                "node_feat_dim": node_feat_dim, "edge_feat_dim": edge_feat_dim,
            }

            for att in args.att_values:
                for conv_type in args.arch:
                    if already_done("matched", seed, conv_type, att):
                        continue

                    remaining = time_left(deadline)
                    if stop_requested() or remaining <= 0:
                        print("  Stopping (signal or time budget) — re-run to continue.")
                        flush()
                        return
                    print(f"\n  Training {ARCH_NAMES[conv_type]} seed {seed} at att={att} "
                          f"({format_eta(remaining)} of budget left)")

                    train_result, ckpt_dir = train_matched_arm(
                        att, seed, conv_type, config, device, data, args.study_dir, args.train_epochs)
                    if train_result is None:
                        print(f"  Training failed for {conv_type} att={att} — skipping")
                        continue

                    arm_config = json.loads(json.dumps(config))
                    arm_config["model"]["expansion_node_weight"] = att
                    model = load_model(conv_type, os.path.join(args.study_dir, f"att_{att}"), seed,
                                       node_feat_dim, edge_feat_dim, arm_config, device)

                    res = evaluate_att_setting(model, att, loaders, device, criterion,
                                               cost, perturbations, args.max_graphs)
                    res.update({"arm": "matched", "seed": seed, "conv_type": conv_type,
                                "arch": ARCH_NAMES[conv_type], "trained_att": att,
                                "eval_fingerprint": eval_fingerprint,
                                "checkpoint": os.path.join(ckpt_dir, conv_type, "best_model.pt"),
                                "train_test_f1_macro": train_result["f1_macro"]})
                    records.append(res)
                    flush()

    print_report(records, args.att_values, list(perturbations))
    flush()
    print()


def print_report(records, att_values, perturbation_names):
    for arm in ("swap", "matched"):
        arm_records = [r for r in records if r["arm"] == arm]
        if not arm_records:
            continue

        print("\n" + "#" * 100)
        print(f"#  {arm.upper()} — ASR (lower = more robust), threshold fixed from clean validation")
        print("#" * 100)
        header = f"{'Arch':<12} | {'Seed':>5} | {'att':>4} | {'F1 clean':>8} | {'TP':>4} | " + \
                 " | ".join(f"{n:>16}" for n in perturbation_names)
        print(header)
        print("-" * len(header))
        for r in sorted(arm_records, key=lambda x: (x["conv_type"], x["seed"], -x["att"])):
            row = (f"{r['arch']:<12} | {r['seed']:>5} | {r['att']:>4} | "
                   f"{r['clean']['f1_macro']:>8.4f} | {r['clean'].get('n_attacked', 0):>4} | ")
            row += " | ".join(f"{r['attacks'][n]['asr']:>15.2%}" for n in perturbation_names)
            print(row)
        print("  TP = clean true positives, i.e. the ASR denominator — compare ASR across att "
              "only with this column in view.")

        if len(att_values) == 2:
            hi, lo = max(att_values), min(att_values)
            print(f"\n  Δ ASR ({lo} minus {hi}) — negative means att={lo} blocks more attacks")
            print("-" * len(header))
            keys = sorted({(r["conv_type"], r["seed"]) for r in arm_records})
            for conv_type, seed in keys:
                pair = {r["att"]: r for r in arm_records
                        if r["conv_type"] == conv_type and r["seed"] == seed}
                if hi not in pair or lo not in pair:
                    continue
                d_clean = pair[lo]["clean"]["f1_macro"] - pair[hi]["clean"]["f1_macro"]
                d_tp = pair[lo]["clean"].get("n_attacked", 0) - pair[hi]["clean"].get("n_attacked", 0)
                row = (f"{ARCH_NAMES[conv_type]:<12} | {seed:>5} | {'Δ':>4} | {d_clean:>+8.4f} | "
                       f"{d_tp:>+4} | ")
                row += " | ".join(
                    f"{pair[lo]['attacks'][n]['asr'] - pair[hi]['attacks'][n]['asr']:>+15.2%}"
                    for n in perturbation_names)
                print(row)
        print("#" * 100)


if __name__ == "__main__":
    main()
