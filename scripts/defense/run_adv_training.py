import os
import sys
import glob
import json
import time
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

from scripts.utils import (
    ALL_CONV_TYPES, ARCH_NAMES,
    load_yaml_config, resolve_device, save_results,
)
from src.dataset.Elliptic2Dataset import Elliptic2Dataset
from src.models.model import EllipticGNN
from src.models.losses import FocalLoss
from src.models.calibration import fit_temperature, save_decision_rule
from src.train.sampler import BalancedBatchSampler, make_weighted_sampler
from src.train.threshold import evaluate_with_threshold_search, evaluate_with_fixed_threshold
from src.train.utils import atomic_save, compute_class_weights, set_all_seeds, print_comprehensive_metrics
from src.defense.adv_train import (
    install_signal_handlers, stop_requested,
    load_or_compute_feat_bounds,
    adv_train_epoch, evaluate_robust, evaluate_structural,
    empty_history, build_state, save_history, format_eta,
    epsilon_for_epoch, time_left,
)
from src.defense.structural import LicitStats, STRATEGIES


def assert_safe_checkpoint_dir(ckpt_dir, config):
    baseline = os.path.abspath(config["training"]["checkpoint_dir"])
    target = os.path.abspath(ckpt_dir)
    if target == baseline or target.startswith(baseline + os.sep):
        print(f"\nREFUSING TO RUN: --checkpoint_dir ({target}) is inside the baseline "
              f"checkpoint tree ({baseline}).")
        print("Adversarial training must write to a separate directory, otherwise it "
              "overwrites the clean models the attack scripts are evaluating.")
        sys.exit(2)


def assert_safe_results_dir(results_dir, config):
    attack_root = os.path.abspath(config["data"].get("results", "data/results"))
    target = os.path.abspath(results_dir)
    if target == attack_root:
        print(f"\nREFUSING TO RUN: --results_dir ({target}) is the attack results root.")
        print("Use a subdirectory (e.g. data/results/defense) so defence output cannot be "
              "confused with attack output.")
        sys.exit(2)


def defense_weights(args):
    adv = args.adv_weight
    if args.defense == "feature":
        return {"clean": 1.0 - adv, "feat": adv, "struct": 0.0}
    if args.defense == "structural":
        return {"clean": 1.0 - adv, "feat": 0.0, "struct": adv}
    share = args.struct_share
    return {"clean": 1.0 - adv, "feat": adv * (1.0 - share), "struct": adv * share}


def run_fingerprint(args, config):
    return {
        "defense": args.defense,
        "weights": defense_weights(args),
        "epsilon": args.epsilon,
        "pgd_steps": args.pgd_steps,
        "step_size": args.step_size,
        "inner_loss": args.inner_loss,
        "eps_warmup_epochs": args.eps_warmup_epochs,
        "struct_strategies": sorted(args.struct_strategies),
        "struct_k": sorted(args.struct_k),
        "struct_conn": sorted(args.struct_conn),
        "epochs": args.epochs,
        "lr": args.lr,
        "init": args.init,
        "select_on": args.select_on,
        "robust_eval_steps": args.robust_eval_steps,
        "robust_eval_n": args.robust_eval_n,
        "limit_train_batches": args.limit_train_batches,
        "eval_epsilons": sorted(args.eval_epsilons),
        "struct_eval_grid": sorted(f"{s}:{k}:{c}" for s, k, c in args.struct_eval_grid),
        "expansion_node_weight": config["model"].get("expansion_node_weight", 1.0),
    }


def fingerprint_diff(expected, found):
    if not isinstance(found, dict):
        return ["<no fingerprint recorded>"]
    keys = set(expected) | set(found)
    return sorted(k for k in keys if expected.get(k) != found.get(k))


def parse_struct_grid(items):
    grid = []
    for item in items:
        if isinstance(item, (tuple, list)):
            grid.append(tuple(item))
            continue
        parts = item.split(":")
        if len(parts) != 3 or parts[0] not in STRATEGIES:
            raise argparse.ArgumentTypeError(
                f"--struct_eval_grid entry must be strategy:k:connections with strategy in "
                f"{list(STRATEGIES)}, got '{item}'"
            )
        grid.append((parts[0], int(parts[1]), int(parts[2])))
    return grid


def selection_metric(select_on, clean_f1, pgd_f1, struct_f1):
    if select_on == "clean":
        return clean_f1
    if select_on == "robust":
        return pgd_f1
    if select_on == "struct":
        return struct_f1
    available = [v for v in (pgd_f1, struct_f1) if np.isfinite(v)]
    return float(np.mean(available)) if available else clean_f1


def build_model(conv_type, node_feat_dim, edge_feat_dim, config, device):
    mcfg = config["model"]
    return EllipticGNN(
        node_feat_dim=node_feat_dim,
        edge_feat_dim=edge_feat_dim,
        hidden_dim=mcfg["hidden_dim"],
        num_layers=mcfg["num_layers"],
        heads=mcfg["heads"],
        edge_proj_dim=mcfg["edge_proj_dim"],
        num_classes=mcfg["num_classes"],
        dropout=mcfg["dropout"],
        conv_type=conv_type,
        expansion_node_weight=mcfg.get("expansion_node_weight", 1.0),
    ).to(device)


def load_clean_init(model, conv_type, seed, baseline_dir, device):
    path = os.path.join(baseline_dir, f"seed_{seed}", conv_type, "best_model.pt")
    if not os.path.exists(path):
        return False, path
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    return True, path


def train_one_arch(conv_type, seed, config, args, device, ctx, deadline):
    arch_name = ARCH_NAMES.get(conv_type, conv_type)
    train_cfg = config["training"]
    num_classes = config["model"]["num_classes"]

    arch_dir = os.path.join(args.checkpoint_dir, f"seed_{seed}", conv_type)
    os.makedirs(arch_dir, exist_ok=True)
    best_ckpt_path = os.path.join(arch_dir, "best_model.pt")
    last_ckpt_path = os.path.join(arch_dir, "last_checkpoint.pt")
    history_path = os.path.join(arch_dir, "history.json")

    print(f"\n{'#' * 100}")
    print(f"#  ADVERSARIAL TRAINING — {arch_name} ({conv_type}) | seed {seed}")
    print(f"{'#' * 100}")

    model = build_model(conv_type, ctx["node_feat_dim"], ctx["edge_feat_dim"], config, device)
    print(f"    Parameters: {model.count_params():,}")

    if train_cfg.get("class_weighting", False):
        class_weights = compute_class_weights(ctx["train_dataset"], num_classes, device)
        criterion = FocalLoss(alpha=class_weights, gamma=2.0,
                              label_smoothing=train_cfg.get("label_smoothing", 0.0))
    else:
        criterion = FocalLoss(gamma=2.0, label_smoothing=train_cfg.get("label_smoothing", 0.0))

    init_source = "scratch"
    if args.init == "clean":
        ok, path = load_clean_init(model, conv_type, seed, args.baseline_dir, device)
        if ok:
            init_source = path
            print(f"    Warm start from clean baseline: {path}")
        elif args.require_clean_init:
            print(f"    Clean checkpoint missing: {path} — skipping (use --init scratch to train anyway)")
            return "failed", None
        else:
            print(f"    Clean checkpoint missing ({path}) — falling back to random init")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=train_cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1), eta_min=1e-6
    )

    start_epoch, best_metric, patience_counter = 0, -1.0, 0
    history = empty_history()

    fingerprint = run_fingerprint(args, config)

    if os.path.exists(last_ckpt_path) and not args.fresh:
        state = torch.load(last_ckpt_path, map_location="cpu", weights_only=False)

        drift = fingerprint_diff(fingerprint, (state.get("meta") or {}).get("fingerprint"))
        if drift:
            print(f"\n  REFUSING TO RESUME {last_ckpt_path}")
            print(f"  It was written by a run with a different configuration: {', '.join(drift)}")
            print("  Continuing would silently blend two experiments. Either re-run with --fresh")
            print("  to discard it, or point --checkpoint_dir at a new directory.")
            return "failed", None

        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        if state.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(state["scheduler_state_dict"])
        start_epoch = state["epoch"] + 1
        best_metric = state.get("best_metric", -1.0)
        history = state.get("history", history)
        patience_counter = state.get("patience_counter", 0)
        print(f"    Auto-resume from epoch {start_epoch} | best {args.select_on} = {best_metric:.4f}")
    elif args.fresh:
        for p in (last_ckpt_path, best_ckpt_path):
            if os.path.exists(p):
                os.remove(p)
                print(f"    removed {p}")

    cost_matrix = train_cfg.get("cost_matrix", None) or {}
    fn_cost = cost_matrix.get("fn_cost", None)
    fp_cost = cost_matrix.get("fp_cost", None)

    feat_min = ctx["feat_min"].to(device)
    feat_max = ctx["feat_max"].to(device)
    weights = defense_weights(args)

    struct_atk = None
    if weights["struct"] > 0.0 or args.defense in ("structural", "both"):
        struct_atk = {
            "stats": ctx["licit_stats"],
            "k_choices": args.struct_k,
            "conn_choices": args.struct_conn,
            "strategies": args.struct_strategies,
            "rng": np.random.default_rng(seed),
        }
    struct_eval_params = {
        "k_nodes": args.struct_eval_k,
        "connections": args.struct_eval_conn,
        "strategy": args.struct_eval_strategy,
    }

    def attack_kwargs(epsilon, steps=None):
        steps = args.pgd_steps if steps is None else steps
        step_size = args.step_size if args.step_size is not None else 2.5 * epsilon / max(steps, 1)
        return {
            "epsilon": epsilon,
            "steps": steps,
            "step_size": step_size,
            "feat_min": feat_min,
            "feat_max": feat_max,
            "random_start": True,
            "inner_loss": args.inner_loss,
        }

    if start_epoch >= args.epochs:
        print(f"\n    Already at epoch {start_epoch}/{args.epochs} — going straight to evaluation.")
    else:
        print(f"\n    defense={args.defense} | weights: clean={weights['clean']:.2f} "
              f"feat={weights['feat']:.2f} struct={weights['struct']:.2f}")
        print(f"    feature : epsilon={args.epsilon} | pgd_steps={args.pgd_steps} | inner_loss={args.inner_loss}")
        if struct_atk is not None:
            print(f"    structural: strategies={args.struct_strategies} | k={args.struct_k} "
                  f"| connections={args.struct_conn}")
            print(f"    structural val: {struct_eval_params}")
        print(f"    Epochs {start_epoch + 1} -> {args.epochs} | batches/epoch: {len(ctx['train_loader'])} | lr={args.lr}")
        print("=" * 120)
        print(f"{'Epoch':>6} | {'eps':>5} | {'TrainLoss':>9} | {'AdvLoss':>8} | {'StrLoss':>8} | "
              f"{'CleanF1':>8} | {'PgdF1':>8} | {'StructF1':>8} | {'LR':>9} | {'Time':>7}")
        print("-" * 120)

        for epoch in range(start_epoch, args.epochs):
            t0 = time.time()
            eps_now = epsilon_for_epoch(epoch, args.epsilon, args.eps_warmup_epochs)

            train_loss, adv_loss, struct_loss = adv_train_epoch(
                model, ctx["train_loader"], optimizer, criterion, device,
                weights=weights,
                feat_atk=attack_kwargs(eps_now) if weights["feat"] > 0 else None,
                struct_atk=struct_atk if weights["struct"] > 0 else None,
                desc=f"[{arch_name}] adv epoch {epoch + 1}/{args.epochs}",
                limit_batches=args.limit_train_batches,
            )

            if not np.isfinite(train_loss):
                print(f"\n  FATAL: train loss is {train_loss} at epoch {epoch + 1} [{arch_name}] — aborting arch.")
                return "failed", None

            val_loss, val_acc, val_f1_macro, _, _, _, _, _ = evaluate_with_threshold_search(
                model, ctx["val_loader"], criterion, device, num_classes, fn_cost, fp_cost
            )

            nan_metrics = {"f1_macro": float("nan"), "acc": float("nan"), "recall": float("nan")}
            do_robust = ((epoch + 1) % args.robust_eval_every == 0) or (epoch + 1 == args.epochs)

            rob = nan_metrics
            if do_robust and args.defense in ("feature", "both"):
                rob = evaluate_robust(
                    model, ctx["val_loader"], criterion, device,
                    atk=attack_kwargs(args.epsilon, steps=args.robust_eval_steps),
                    fn_cost=fn_cost, fp_cost=fp_cost, max_graphs=args.robust_eval_n,
                )

            struct = nan_metrics
            if do_robust and args.defense in ("structural", "both"):
                struct = evaluate_structural(
                    model, ctx["val_loader"], criterion, device,
                    stats=ctx["licit_stats"], params=struct_eval_params,
                    fn_cost=fn_cost, fp_cost=fp_cost, max_graphs=args.robust_eval_n,
                )

            scheduler.step()
            lr_now = optimizer.param_groups[0]["lr"]
            epoch_seconds = time.time() - t0

            history["epsilon"].append(float(eps_now))
            history["lr"].append(float(lr_now))
            history["epoch_seconds"].append(round(epoch_seconds, 1))
            history["train_loss"].append(float(train_loss))
            history["train_adv_loss"].append(float(adv_loss))
            history["train_struct_loss"].append(float(struct_loss))
            history["val_clean_loss"].append(float(val_loss))
            history["val_clean_f1_macro"].append(float(val_f1_macro))
            history["val_clean_acc"].append(float(val_acc))
            history["val_robust_f1_macro"].append(float(rob["f1_macro"]))
            history["val_robust_acc"].append(float(rob["acc"]))
            history["val_robust_recall"].append(float(rob["recall"]))
            history["val_struct_f1_macro"].append(float(struct["f1_macro"]))
            history["val_struct_acc"].append(float(struct["acc"]))
            history["val_struct_recall"].append(float(struct["recall"]))

            print(f"{epoch + 1:>6} | {eps_now:>5.3f} | {train_loss:>9.4f} | {adv_loss:>8.4f} | "
                  f"{struct_loss:>8.4f} | {val_f1_macro:>8.4f} | {rob['f1_macro']:>8.4f} | "
                  f"{struct['f1_macro']:>8.4f} | {lr_now:>9.6f} | {format_eta(epoch_seconds):>7}")

            metric = selection_metric(args.select_on, val_f1_macro, rob["f1_macro"], struct["f1_macro"])

            if np.isfinite(metric) and metric > best_metric:
                best_metric = metric
                patience_counter = 0
                atomic_save(model.state_dict(), best_ckpt_path)
                print(f"  >> new best ({args.select_on}) = {metric:.4f} — saved best_model.pt")
            elif do_robust or args.select_on == "clean":
                patience_counter += 1

            meta = {
                "fingerprint": fingerprint,
                "adv_weight": args.adv_weight, "struct_share": args.struct_share,
                "init_source": init_source, "seed": seed, "conv_type": conv_type,
            }
            atomic_save(build_state(epoch, model, optimizer, scheduler,
                                    best_metric, history, patience_counter, meta),
                        last_ckpt_path)
            save_history(history, history_path)

            if patience_counter >= args.early_stop_patience:
                print(f"\n  Early stopping: no improvement for {args.early_stop_patience} evaluations")
                break

            remaining = time_left(deadline)
            if stop_requested():
                print(f"\n  Stopping on signal after epoch {epoch + 1} — re-run to continue.")
                return "paused", None
            if remaining < epoch_seconds * 1.15:
                print(f"\n  Time budget nearly exhausted ({format_eta(remaining)} left, "
                      f"epoch takes {format_eta(epoch_seconds)}) — pausing after epoch {epoch + 1}.")
                print("  Re-run the same command to continue from here.")
                return "paused", None

    if not os.path.exists(best_ckpt_path):
        print(f"\n  [{arch_name}] no best_model.pt — nothing to evaluate")
        return "failed", None

    print(f"\n  [{arch_name}] Final evaluation (best adversarially-trained checkpoint)")
    model.load_state_dict(torch.load(best_ckpt_path, map_location=device, weights_only=True))

    learned_temp = 1.0
    if train_cfg.get("calibration", False):
        print(f"  [{arch_name}] Fitting temperature scaling on validation set")
        learned_temp = fit_temperature(model, ctx["val_loader"], device)

    _, val_acc_final, val_f1_final, _, val_preds, val_labels, val_threshold, val_probs = \
        evaluate_with_threshold_search(model, ctx["val_loader"], criterion, device,
                                       num_classes, fn_cost, fp_cost, temperature=learned_temp)
    save_decision_rule(arch_dir, learned_temp, val_threshold)
    print(f"    Val threshold: {val_threshold:.4f} | Val F1_macro: {val_f1_final:.4f}")

    test_loss, test_acc, test_f1_macro, test_f1_weighted, test_preds, test_labels, test_probs = \
        evaluate_with_fixed_threshold(model, ctx["test_loader"], criterion, device,
                                      threshold=val_threshold, temperature=learned_temp)

    print("\n" + "=" * 100)
    print(f"CLEAN TEST REPORT — {arch_name} (adversarially trained)")
    print("=" * 100)
    print(classification_report(test_labels, test_preds,
                                target_names=["Licit", "Suspicious"],
                                labels=list(range(num_classes))))
    extra = print_comprehensive_metrics(test_labels, test_preds, test_probs,
                                        target_names=["Licit", "Suspicious"])

    try:
        roc_auc = roc_auc_score(test_labels, test_probs)
        pr_auc = average_precision_score(test_labels, test_probs)
    except ValueError:
        roc_auc, pr_auc = 0.0, 0.0

    robust_test = {}
    for eps in args.eval_epsilons:
        r = evaluate_robust(model, ctx["test_loader"], criterion, device,
                            atk=attack_kwargs(eps, steps=args.robust_eval_steps),
                            fn_cost=fn_cost, fp_cost=fp_cost, max_graphs=args.robust_eval_n)
        robust_test[str(eps)] = {k: round(float(v), 6) for k, v in r.items()}
        print(f"    PGD eps={eps}: robust F1_macro={r['f1_macro']:.4f} | "
              f"recall(susp)={r['recall']:.4f} | acc={r['acc']:.4f}")

    structural_test = {}
    for strategy, k, conn in args.struct_eval_grid:
        params = {"strategy": strategy, "k_nodes": k, "connections": conn}
        r = evaluate_structural(model, ctx["test_loader"], criterion, device,
                                stats=ctx["licit_stats"], params=params,
                                fn_cost=fn_cost, fp_cost=fp_cost, max_graphs=args.robust_eval_n)
        structural_test[f"{strategy}_k{k}_c{conn}"] = {kk: round(float(vv), 6) for kk, vv in r.items()}
        print(f"    {strategy} k={k} c={conn}: F1_macro={r['f1_macro']:.4f} | "
              f"recall(susp)={r['recall']:.4f} | evasion={r['evasion_rate']:.4f}")

    return "done", {
        "arch": arch_name,
        "conv_type": conv_type,
        "seed": seed,
        "defense": f"adversarial_training_{args.defense}",
        "fingerprint": fingerprint,
        "defense_config": {
            "struct_share": args.struct_share,
            "mode": args.defense,
            "loss_weights": weights,
            "epsilon": args.epsilon,
            "pgd_steps": args.pgd_steps,
            "step_size": args.step_size,
            "adv_weight": args.adv_weight,
            "inner_loss": args.inner_loss,
            "struct_strategies": args.struct_strategies,
            "struct_k": args.struct_k,
            "struct_conn": args.struct_conn,
            "expansion_node_weight": config["model"].get("expansion_node_weight", 1.0),
            "robust_eval_steps": args.robust_eval_steps,
            "eps_warmup_epochs": args.eps_warmup_epochs,
            "epochs_configured": args.epochs,
            "epochs_ran": len(history["train_loss"]),
            "lr": args.lr,
            "init": init_source,
            "select_on": args.select_on,
        },
        "params": model.count_params(),
        "test_loss": test_loss,
        "test_acc": test_acc,
        "f1_macro": test_f1_macro,
        "f1_weighted": test_f1_weighted,
        "threshold": val_threshold,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "recall": extra["recall"],
        "specificity": extra["specificity"],
        "tp": extra["tp"], "fp": extra["fp"], "fn": extra["fn"], "tn": extra["tn"],
        "temperature": learned_temp,
        "val_f1_macro": val_f1_final,
        "val_acc": val_acc_final,
        "robust_test_pgd": robust_test,
        "robust_test_structural": structural_test,
        "test_probs": test_probs.tolist(),
        "test_preds": test_preds.tolist(),
        "test_labels": test_labels.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="PGD adversarial training (defence) for Elliptic2 GNNs")
    parser.add_argument("--config", default="config.yaml")

    parser.add_argument("--seeds", default="42", help="Comma-separated seeds (default: 42)")
    parser.add_argument("--arch", nargs="+", default=ALL_CONV_TYPES, choices=ALL_CONV_TYPES)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Lower than the baseline 1e-3 because we fine-tune a trained model")
    parser.add_argument("--init", choices=["clean", "scratch"], default="clean",
                        help="clean = warm-start from checkpoints/seed_<S>/<arch>/best_model.pt")
    parser.add_argument("--require_clean_init", action="store_true",
                        help="Fail the architecture instead of falling back to random init")
    parser.add_argument("--early_stop_patience", type=int, default=10)

    parser.add_argument("--defense", choices=["feature", "structural", "both"], default="both",
                        help="feature = PGD on node features; structural = node injection / star "
                             "rewiring; both = split the adversarial weight between them")
    parser.add_argument("--epsilon", type=float, default=0.1, help="L_inf radius used during training")
    parser.add_argument("--pgd_steps", type=int, default=5, help="Inner-maximisation steps per batch")
    parser.add_argument("--step_size", type=float, default=None, help="null = 2.5 * epsilon / steps")
    parser.add_argument("--adv_weight", type=float, default=0.5,
                        help="Weight of the adversarial loss; 1.0 = pure Madry, 0.5 = clean/adv mix")
    parser.add_argument("--inner_loss", choices=["ce", "focal"], default="ce",
                        help="ce matches what the evaluation attack maximises")
    parser.add_argument("--eps_warmup_epochs", type=int, default=3,
                        help="Linearly ramp epsilon over the first N epochs (0 = off)")

    parser.add_argument("--struct_share", type=float, default=0.6,
                        help="Fraction of the adversarial weight given to the structural term "
                             "(only used by --defense both)")
    parser.add_argument("--struct_k", nargs="+", type=int, default=[1, 3, 5, 10],
                        help="Injected nodes per graph, sampled per batch — must cover k=10, "
                             "the strongest setting in the baseline injection results")
    parser.add_argument("--struct_conn", nargs="+", type=int, default=[1, 2, 3],
                        help="Edges per injected node, sampled per batch")
    parser.add_argument("--struct_strategies", nargs="+", default=list(STRATEGIES),
                        choices=list(STRATEGIES))
    parser.add_argument("--struct_eval_strategy", default="random", choices=list(STRATEGIES),
                        help="Fixed setting for the per-epoch structural validation — defaults to "
                             "the measured worst case")
    parser.add_argument("--struct_eval_k", type=int, default=10)
    parser.add_argument("--struct_eval_conn", type=int, default=3)
    parser.add_argument("--struct_eval_grid", nargs="+",
                        default=["random:1:1", "random:10:3", "degree:5:2", "mimicry:10:3", "degree:1:10"],
                        help="strategy:k:connections triples for the final test-set report; "
                             "degree:1:10 is a star rewiring (one hub, 10 spokes)")

    parser.add_argument("--select_on", choices=["robust", "clean", "struct", "mean"], default="mean")
    parser.add_argument("--robust_eval_every", type=int, default=1)
    parser.add_argument("--robust_eval_steps", type=int, default=10,
                        help="PGD steps for the validation attack — stronger than training, "
                             "so model selection is not fooled by the weak inner attack")
    parser.add_argument("--robust_eval_n", type=int, default=0,
                        help="Cap graphs used for robust eval (0 = whole split)")
    parser.add_argument("--eval_epsilons", nargs="+", type=float, default=[0.05, 0.1, 0.2])

    parser.add_argument("--checkpoint_dir", default="checkpoints_adv")
    parser.add_argument("--baseline_dir", default=None, help="Defaults to config training.checkpoint_dir")
    parser.add_argument("--results_dir", default="data/results/defense")
    parser.add_argument("--batch_size", type=int, default=None, help="Defaults to config training.batch_size")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_hours", type=float, default=0.0, help="Wall-clock budget, 0 = unlimited")
    parser.add_argument("--fresh", action="store_true", help="Discard this defence run's own checkpoints")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip an architecture whose result JSON already exists")
    parser.add_argument("--limit_train_batches", type=int, default=0,
                        help="Cap batches per epoch (0 = all) — for smoke tests")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    args.struct_eval_grid = parse_struct_grid(args.struct_eval_grid)

    install_signal_handlers()

    config = load_yaml_config(args.config)
    if args.baseline_dir is None:
        args.baseline_dir = config["training"]["checkpoint_dir"]
    assert_safe_checkpoint_dir(args.checkpoint_dir, config)
    assert_safe_results_dir(args.results_dir, config)

    device = resolve_device(args.device)
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    deadline = time.time() + args.max_hours * 3600 if args.max_hours > 0 else None

    print("=" * 100)
    print("  PGD ADVERSARIAL TRAINING (defence)")
    print("=" * 100)
    print(f"  Device          : {device}")
    if device.type == "cuda":
        print(f"  GPU             : {torch.cuda.get_device_name(0)}")
    print(f"  Seeds           : {seeds}")
    print(f"  Architectures   : {[ARCH_NAMES[a] for a in args.arch]}")
    print(f"  Defence         : {args.defense}  {defense_weights(args)}")
    print(f"  Attention (att) : expansion_node_weight = {config['model'].get('expansion_node_weight', 1.0)}")
    print(f"  Warm start from : {args.baseline_dir} (read-only)")
    print(f"  Writes to       : {args.checkpoint_dir}/ and {args.results_dir}/")
    print(f"  Time budget     : {format_eta(args.max_hours * 3600) if deadline else 'unlimited'}")

    train_cfg = config["training"]
    processed_dir = config["data"]["processed_dir"]
    batch_size = args.batch_size or train_cfg["batch_size"]
    val_ratio, test_ratio = train_cfg["val_ratio"], train_cfg["test_ratio"]

    print("\nLoading datasets (read-only)")
    train_dataset = Elliptic2Dataset(processed_dir, split="train", val_ratio=val_ratio, test_ratio=test_ratio)
    val_dataset = Elliptic2Dataset(processed_dir, split="val", val_ratio=val_ratio, test_ratio=test_ratio)
    test_dataset = Elliptic2Dataset(processed_dir, split="test", val_ratio=val_ratio, test_ratio=test_ratio)

    sample = train_dataset[0]
    node_feat_dim, edge_feat_dim = sample.x.shape[1], sample.edge_attr.shape[1]

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    print("\nFeature bounds (L_inf box, from the training split)")
    feat_min, feat_max = load_or_compute_feat_bounds(
        train_dataset, os.path.join(args.checkpoint_dir, "feat_bounds.pt")
    )

    print("Licit feature distribution (source of injected nodes/edges)")
    licit_stats = LicitStats.from_dataset(
        train_dataset, cache_path=os.path.join(args.checkpoint_dir, "licit_stats.pt"), device=device
    )

    ctx = {
        "licit_stats": licit_stats,
        "train_dataset": train_dataset,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "node_feat_dim": node_feat_dim,
        "edge_feat_dim": edge_feat_dim,
        "feat_min": feat_min,
        "feat_max": feat_max,
    }

    statuses = {}
    finished_results = []
    paused = False

    for seed in seeds:
        if paused:
            break
        set_all_seeds(seed)

        if train_cfg.get("balanced_sampling", False):
            ctx["train_loader"] = DataLoader(
                train_dataset,
                batch_sampler=BalancedBatchSampler(labels=train_dataset.get_labels(), batch_size=batch_size),
                num_workers=args.num_workers, pin_memory=True,
            )
        elif train_cfg.get("oversampling", False):
            ctx["train_loader"] = DataLoader(
                train_dataset, batch_size=batch_size, sampler=make_weighted_sampler(train_dataset),
                num_workers=args.num_workers, pin_memory=True,
            )
        else:
            ctx["train_loader"] = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True,
                num_workers=args.num_workers, pin_memory=True,
            )

        seed_results_dir = os.path.join(args.results_dir, f"seed_{seed}")
        os.makedirs(seed_results_dir, exist_ok=True)

        for conv_type in args.arch:
            key = f"{conv_type}/{seed}"
            part_path = os.path.join(seed_results_dir, f"adv_training_{conv_type}.json")

            if args.skip_existing and os.path.exists(part_path):
                with open(part_path) as f:
                    existing = json.load(f)
                drift = fingerprint_diff(run_fingerprint(args, config), existing.get("fingerprint"))
                if not drift:
                    print(f"\n  – {ARCH_NAMES[conv_type]} (seed {seed}) already has {part_path} — skipping")
                    statuses[key] = "skipped"
                    continue
                print(f"\n  ! {part_path} exists but was produced by a different configuration "
                      f"({', '.join(drift)}) — recomputing instead of skipping")

            status, result = train_one_arch(conv_type, seed, config, args, device, ctx, deadline)
            statuses[key] = status

            if status == "done" and result is not None:
                save_results(result, part_path)
                finished_results.append(result)
            elif status == "paused":
                paused = True
                break

    merged = []
    for p in sorted(glob.glob(os.path.join(args.results_dir, "seed_*",
                                           "adv_training_*.json"))):
        with open(p) as f:
            merged.append(json.load(f))

    merged.sort(key=lambda r: (r.get("seed", 0), r.get("conv_type", "")))

    if merged:
        compact = [{k: v for k, v in r.items()
                    if k not in ("test_probs", "test_preds", "test_labels")} for r in merged]
        save_results(compact, os.path.join(args.results_dir, "adv_training_summary.json"))

        print("\n" + "#" * 100)
        print("#  ADVERSARIALLY TRAINED MODELS — clean test metrics")
        print("#" * 100)
        struct_keys = [f"{s}_k{k}_c{c}" for s, k, c in args.struct_eval_grid]
        header = (f"{'Arch':<12} | {'Seed':>5} | {'F1 clean':>8} | " +
                  " | ".join(f"pgd@{e:<6}" for e in args.eval_epsilons) + " | " +
                  " | ".join(f"{k:>16}" for k in struct_keys))
        print(header)
        print("-" * len(header))
        for r in compact:
            row = f"{r['arch']:<12} | {r['seed']:>5} | {r['f1_macro']:>8.4f} | "
            row += " | ".join(
                f"{r['robust_test_pgd'].get(str(e), {}).get('f1_macro', float('nan')):>10.4f}"
                for e in args.eval_epsilons
            )
            row += " | " + " | ".join(
                f"{r.get('robust_test_structural', {}).get(k, {}).get('f1_macro', float('nan')):>16.4f}"
                for k in struct_keys
            )
            print(row)
        print("#" * 100)
        print("  pgd@e / <strategy>_k<n>_c<m>: F1-macro under that perturbation (higher = more robust)")

    print("\n" + "=" * 100)
    print("  STATUS")
    print("=" * 100)
    for key, st in statuses.items():
        print(f"    {st.upper():<8} {key}")
    if paused or stop_requested():
        print("\n  Paused — re-run the same command to continue from the last saved epoch.")
    print()


if __name__ == "__main__":
    main()
