import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, '../..')))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, '..')))

from train.sampler import BalancedBatchSampler, make_weighted_sampler
from train.threshold import evaluate_with_threshold_search, evaluate_with_fixed_threshold
from train.utils import compute_class_weights, build_save_state, atomic_save, print_comprehensive_metrics, load_config, \
    resolve_paths, set_all_seeds
import json
import time
import argparse
import torch
import numpy as np
from torch_geometric.loader import DataLoader
from sklearn.metrics import (
    classification_report, f1_score, accuracy_score,
    roc_auc_score, average_precision_score
)
from tqdm import tqdm

from src.dataset.Elliptic2Dataset import Elliptic2Dataset
from src.models.model import EllipticGNN
from src.models.losses import FocalLoss
from src.models.calibration import fit_temperature, save_decision_rule


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    all_preds, all_labels = [], []
    total_loss, n_batches = 0.0, 0

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        loss = criterion(logits, batch.y.view(-1))
        total_loss += loss.item()
        n_batches += 1
        all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
        all_labels.extend(batch.y.view(-1).cpu().numpy())

    avg_loss = total_loss / max(n_batches, 1)
    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    return avg_loss, acc, f1_macro, f1_weighted, np.array(all_preds), np.array(all_labels)


ARCH_NAMES = {
    "gatv2": "GATv2",
    "sage": "SAGE",
    "sage_edge": "SAGE+Edge",
    "gin": "GIN",
}


def train_and_evaluate(conv_type, config, device, train_dataset,
                       train_loader, val_loader, test_loader, node_feat_dim, edge_feat_dim,
                       checkpoint_dir, fresh=False):
    arch_name = ARCH_NAMES.get(conv_type, conv_type)
    train_cfg = config["training"]
    num_classes = config["model"]["num_classes"]

    arch_ckpt_dir = os.path.join(checkpoint_dir, conv_type)
    os.makedirs(arch_ckpt_dir, exist_ok=True)

    last_ckpt_path = os.path.join(arch_ckpt_dir, "last_checkpoint.pt")
    best_ckpt_path = os.path.join(arch_ckpt_dir, "best_model.pt")

    print(f"\n{'#' * 100}")
    print(f"#  ARCHITECTURE: {arch_name} ({conv_type})")
    print(f"{'#' * 100}")

    model = EllipticGNN(
        node_feat_dim=node_feat_dim,
        edge_feat_dim=edge_feat_dim,
        hidden_dim=config["model"]["hidden_dim"],
        num_layers=config["model"]["num_layers"],
        heads=config["model"]["heads"],
        edge_proj_dim=config["model"]["edge_proj_dim"],
        num_classes=num_classes,
        dropout=config["model"]["dropout"],
        conv_type=conv_type,
        expansion_node_weight=config["model"].get("expansion_node_weight", 1.0),
    ).to(device)
    print(f"    Parameters: {model.count_params():,}")

    use_class_weights = train_cfg.get("class_weighting", False)
    label_smoothing = train_cfg.get("label_smoothing", 0.0)
    if use_class_weights:
        print("\n  Focal Loss with class weighting:")
        class_weights = compute_class_weights(train_dataset, num_classes, device)
        criterion = FocalLoss(alpha=class_weights, gamma=2.0, label_smoothing=label_smoothing)
    else:
        criterion = FocalLoss(gamma=2.0, label_smoothing=label_smoothing)
    if label_smoothing > 0:
        print(f"    Label smoothing: {label_smoothing}")

    warmup_epochs = train_cfg.get("warmup_epochs", 5)
    base_lr = train_cfg["learning_rate"]
    lr_schedule = train_cfg.get("lr_schedule", "plateau")
    epochs = train_cfg["epochs"]

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=base_lr,
        weight_decay=train_cfg["weight_decay"],
    )

    if lr_schedule == "cosine":
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(epochs - warmup_epochs, 1), eta_min=1e-6
        )
        plateau_scheduler = None
        print(f"    LR schedule: cosine annealing (T_max={epochs - warmup_epochs})")
    else:
        cosine_scheduler = None
        plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max",
            patience=train_cfg["lr_patience"],
            factor=train_cfg["lr_factor"],
        )
        print(f"    LR schedule: ReduceLROnPlateau (patience={train_cfg['lr_patience']})")

    def apply_warmup_lr(epoch):
        if epoch < warmup_epochs:
            warmup_factor = 0.1 + 0.9 * (epoch / warmup_epochs)
            for pg in optimizer.param_groups:
                pg["lr"] = base_lr * warmup_factor

    def step_scheduler(epoch, val_f1_macro):
        if epoch < warmup_epochs:
            pass
        elif cosine_scheduler is not None:
            cosine_scheduler.step()
        else:
            plateau_scheduler.step(val_f1_macro)

    active_scheduler = cosine_scheduler if cosine_scheduler is not None else plateau_scheduler

    start_epoch = 0
    best_val_f1 = 0.0
    patience_counter = 0
    history = {
        "train_loss": [], "val_loss": [],
        "val_acc": [], "val_f1_macro": [], "val_f1_weighted": [],
    }

    if fresh:
        for p in [last_ckpt_path, best_ckpt_path]:
            if os.path.exists(p):
                os.remove(p)
                print(f"    removed {p}")
    else:
        if os.path.exists(last_ckpt_path):
            print(f"\nAuto-resume: checkpoint found at {last_ckpt_path}")
            state = torch.load(last_ckpt_path, map_location="cpu")
            model.load_state_dict(state["model_state_dict"])
            optimizer.load_state_dict(state["optimizer_state_dict"])
            if "scheduler_state_dict" in state and active_scheduler is not None:
                active_scheduler.load_state_dict(state["scheduler_state_dict"])
            start_epoch     = state["epoch"] + 1
            best_val_f1     = state.get("best_val_f1", 0.0)
            history         = state.get("history", history)
            patience_counter = state.get("patience_counter", 0)
            print(f"    Resuming from epoch {start_epoch} | best_val_f1_macro: {best_val_f1:.4f}")

    cost_matrix = train_cfg.get("cost_matrix", None)
    if cost_matrix:
        fn_cost = cost_matrix.get("fn_cost", 10.0)
        fp_cost = cost_matrix.get("fp_cost", 1.0)
    else:
        fn_cost, fp_cost = None, None

    early_stop_patience = train_cfg["early_stop_patience"]
    save_every = train_cfg["save_every"]

    if start_epoch >= epochs:
        print(f"\nAlready at epoch {start_epoch} >= {epochs}. Skipping training.")
    else:
        print(f"\nTraining epochs {start_epoch+1} -> {epochs}")
        print("=" * 100)
        print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Val Loss':>9} | "
              f"{'Val Acc':>8} | {'F1 macro':>8} | {'F1 wght':>7} | {'LR':>10}")
        print("-" * 100)

        training_start = time.time()
        final_epoch = start_epoch

        for epoch in range(start_epoch, epochs):
            final_epoch = epoch

            apply_warmup_lr(epoch)

            model.train()
            train_loss_sum, n_batches = 0.0, 0

            for batch in tqdm(train_loader, desc=f"[{arch_name}] Epoch {epoch+1}/{epochs}", leave=False):

                batch = batch.to(device)
                optimizer.zero_grad()
                logits = model(batch)
                loss = criterion(logits, batch.y.view(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss_sum += loss.item()
                n_batches += 1

            avg_train_loss = train_loss_sum / max(n_batches, 1)

            if np.isnan(avg_train_loss) or np.isinf(avg_train_loss):
                print(f"\n  FATAL: Train loss is {avg_train_loss} at epoch {epoch+1} [{arch_name}].")
                print("  Model weights are corrupted. Skipping this architecture.")
                return None

            val_loss, val_acc, val_f1_macro, val_f1_weighted, _, _, _, _ = evaluate_with_threshold_search(
                model, val_loader, criterion, device, num_classes, fn_cost, fp_cost
            )
            step_scheduler(epoch, val_f1_macro)

            current_lr = optimizer.param_groups[0]["lr"]
            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["val_f1_macro"].append(val_f1_macro)
            history["val_f1_weighted"].append(val_f1_weighted)

            print(f"{epoch+1:>6} | {avg_train_loss:>10.4f} | {val_loss:>9.4f} | "
                  f"{val_acc:>7.2%} | {val_f1_macro:>8.4f} | {val_f1_weighted:>7.4f} | {current_lr:>10.6f}")

            if val_f1_macro > best_val_f1:
                best_val_f1 = val_f1_macro
                patience_counter = 0
                atomic_save(model.state_dict(), best_ckpt_path)
                print(f"  >> New best F1_macro: {val_f1_macro:.4f} — saved best_model.pt")
            else:
                patience_counter += 1

            if (epoch + 1) % save_every == 0:
                state = build_save_state(epoch, model, optimizer, active_scheduler, best_val_f1, history)
                state["patience_counter"] = patience_counter
                atomic_save(state, last_ckpt_path)

            if patience_counter >= early_stop_patience:
                print(f"\n  Early stopping: no improvement for {early_stop_patience} epochs")
                break

        state = build_save_state(final_epoch, model, optimizer, active_scheduler, best_val_f1, history)
        state["patience_counter"] = patience_counter
        atomic_save(state, last_ckpt_path)

        total_time = time.time() - training_start
        print("=" * 100)
        print(f"[{arch_name}] Training done in {total_time/60:.1f} min | epochs: {final_epoch+1}/{epochs}")

    history_path = os.path.join(arch_ckpt_dir, "history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    if not os.path.exists(best_ckpt_path):
        print(f"\n[{arch_name}] No best_model.pt found — skipping test eval")
        return None

    print(f"\n[{arch_name}] Final TEST evaluation (best model):")
    model.load_state_dict(torch.load(best_ckpt_path, map_location=device))

    use_calibration = train_cfg.get("calibration", False)
    learned_temp = 1.0
    if use_calibration:
        print(f"\n  [{arch_name}] Fitting temperature scaling on validation set")
        learned_temp = fit_temperature(model, val_loader, device)

    print(f"\n  [{arch_name}] Finding optimal threshold on validation set")
    _, val_acc_final, val_f1_final, _, val_preds, val_labels, val_threshold, val_probs = evaluate_with_threshold_search(
        model, val_loader, criterion, device, num_classes, fn_cost, fp_cost, temperature=learned_temp
    )
    save_decision_rule(arch_ckpt_dir, learned_temp, val_threshold)
    print(f"    Val threshold: {val_threshold:.4f} | Val F1_macro: {val_f1_final:.4f}")

    try:
        val_roc_auc = roc_auc_score(val_labels, val_probs)
        val_pr_auc = average_precision_score(val_labels, val_probs)
    except ValueError:
        val_roc_auc, val_pr_auc = 0.0, 0.0

    val_extra = print_comprehensive_metrics(val_labels, val_preds, val_probs,
                                            target_names=["Legitimate", "Illicit"])

    print(f"\n  [{arch_name}] Evaluating test set with fixed threshold={val_threshold:.4f} (from val)")
    test_loss, test_acc, test_f1_macro, test_f1_weighted, test_preds, test_labels, test_probs = evaluate_with_fixed_threshold(
        model, test_loader, criterion, device, threshold=val_threshold, temperature=learned_temp
    )

    target_names = ["Licit", "Suspicious"]

    print("\n" + "=" * 100)
    print(f"STANDARD CLASSIFICATION REPORT — {arch_name}")
    print("=" * 100)
    print(classification_report(test_labels, test_preds, target_names=target_names, labels=list(range(num_classes))))
    print(f"Threshold (from val): {val_threshold:.4f}")
    print(f"Test Loss: {test_loss:.4f} | Accuracy: {test_acc:.2%} | "
          f"F1_macro: {test_f1_macro:.4f} | F1_weighted: {test_f1_weighted:.4f}")

    extra = print_comprehensive_metrics(test_labels, test_preds, test_probs, target_names=target_names)

    try:
        roc_auc = roc_auc_score(test_labels, test_probs)
        pr_auc = average_precision_score(test_labels, test_probs)
    except ValueError:
        roc_auc, pr_auc = 0.0, 0.0

    return {
        "arch": arch_name,
        "conv_type": conv_type,
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
        "tp": extra["tp"],
        "fp": extra["fp"],
        "fn": extra["fn"],
        "tn": extra["tn"],
        "temperature": learned_temp,
        "val_f1_macro": val_f1_final,
        "val_acc": val_acc_final,
        "val_roc_auc": val_roc_auc,
        "val_pr_auc": val_pr_auc,
        "val_recall": val_extra["recall"],
        "val_specificity": val_extra["specificity"],
        "test_probs": test_probs.tolist(),
        "test_preds": test_preds.tolist(),
        "test_labels": test_labels.tolist(),
    }


def print_comparison_table(results):
    print("\n" + "#" * 100)
    print("#  ARCHITECTURE COMPARISON")
    print("#" * 100)

    header = (f"{'Architecture':<14} | {'Params':>10} | {'Test Acc':>8} | {'F1 macro':>8} | "
              f"{'F1 wght':>7} | {'Sensitivity':>11} | {'Specificity':>11} | "
              f"{'TP':>5} | {'FP':>5} | {'FN':>5} | {'TN':>6} | {'Threshold':>9}")
    print(header)
    print("-" * len(header))

    for r in results:
        print(f"{r['arch']:<14} | {r['params']:>10,} | {r['test_acc']:>7.2%} | {r['f1_macro']:>8.4f} | "
              f"{r['f1_weighted']:>7.4f} | {r['recall']:>11.4f} | {r['specificity']:>11.4f} | "
              f"{r['tp']:>5} | {r['fp']:>5} | {r['fn']:>5} | {r['tn']:>6} | {r['threshold']:>9.4f}")

    print()

    if results:
        best_f1 = max(results, key=lambda x: x["f1_macro"])
        best_sens = max(results, key=lambda x: x["recall"])
        print(f"  Best F1 macro:    {best_f1['arch']} ({best_f1['f1_macro']:.4f})")
        print(f"  Best Sensitivity: {best_sens['arch']} ({best_sens['recall']:.4f})")

    print("#" * 100 + "\n")


def print_multi_seed_summary(multi_seed_results):
    print("\n" + "#" * 100)
    print("#  MULTI-SEED RESULTS (mean ± std)")
    print("#" * 100)

    metrics = ["f1_macro", "test_acc", "roc_auc", "pr_auc", "recall", "specificity", "threshold"]

    header = f"{'Architecture':<14} | {'Seeds':>5}"
    for m in metrics:
        header += f" | {m:>16}"
    print(header)
    print("-" * len(header))

    for arch, runs in multi_seed_results.items():
        n = len(runs)
        row = f"{arch:<14} | {n:>5}"
        for m in metrics:
            vals = [r[m] for r in runs]
            mean_v = np.mean(vals)
            std_v = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
            row += f" | {mean_v:>7.4f}±{std_v:<6.4f}"
        print(row)

    print("#" * 100 + "\n")


def main():

    parser = argparse.ArgumentParser(description="Train Elliptic2 GNN — multi-architecture comparison")
    parser.add_argument("--fresh", action="store_true", help="Force fresh start — removes all checkpoints")
    parser.add_argument("--arch", type=str, default=None,
                        help="Run single architecture: gatv2, sage, sage_edge, gin (default: run all)")
    parser.add_argument("--seeds", type=str, default=None,
                        help="Comma-separated seeds for multi-seed training (e.g., 42,123,456)")
    args = parser.parse_args()

    config = load_config()
    device = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    spot_mode = config.get("spot_mode", False)
    train_cfg = config["training"]

    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"    GPU: {torch.cuda.get_device_name(0)}")
        print(f"    VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Spot mode: {'ON' if spot_mode else 'OFF'}")

    processed_dir, checkpoint_dir = resolve_paths(config)
    os.makedirs(checkpoint_dir, exist_ok=True)

    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",")]
    else:
        seeds = train_cfg.get("seeds", [42])
    multi_seed = len(seeds) > 1
    print(f"  Seeds: {seeds} ({'multi-seed' if multi_seed else 'single-seed'})")

    if not os.path.exists(f"{processed_dir}/summary.json"):
        print("\nERROR: Preprocessed data not found!")
        print(f"   Expected: {processed_dir}/summary.json")
        print("   Run: python preprocess.py")
        sys.exit(1)

    with open(f"{processed_dir}/summary.json") as f:
        summary = json.load(f)
    node_feat_dim = summary["node_feature_dims"]
    edge_feat_dim = summary["edge_feature_dims"]

    print(f"\nDataset: {summary['num_subgraphs']:,} subgraphs | "
          f"{summary['num_nodes']:,} nodes | "
          f"{summary['num_edges']:,} edges | "
          f"node_feat={node_feat_dim} | edge_feat={edge_feat_dim}")

    print("\nLoading datasets")
    val_ratio = train_cfg["val_ratio"]
    test_ratio = train_cfg["test_ratio"]

    train_dataset = Elliptic2Dataset(processed_dir, split="train", val_ratio=val_ratio, test_ratio=test_ratio)
    val_dataset   = Elliptic2Dataset(processed_dir, split="val",   val_ratio=val_ratio, test_ratio=test_ratio)
    test_dataset  = Elliptic2Dataset(processed_dir, split="test",  val_ratio=val_ratio, test_ratio=test_ratio)

    batch_size = train_cfg["batch_size"]

    val_loader  = DataLoader(val_dataset,  batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print(f"    Train: {len(train_dataset):,} | Val: {len(val_dataset):,} | Test: {len(test_dataset):,}")

    if args.fresh:
        cache_path = os.path.join(processed_dir, "all_graphs.pt")
        if os.path.exists(cache_path):
            os.remove(cache_path)
            print(f"    removed graph cache: {cache_path}")

    all_archs = ["gatv2", "sage", "sage_edge", "gin"]
    if args.arch:
        if args.arch not in all_archs:
            print(f"\nERROR: Unknown architecture '{args.arch}'. Choose from: {all_archs}")
            sys.exit(1)
        archs_to_run = [args.arch]
    else:
        archs_to_run = all_archs

    print(f"\n  Architectures to train: {[ARCH_NAMES[a] for a in archs_to_run]}")

    multi_seed_results = {ARCH_NAMES[a]: [] for a in archs_to_run}
    all_results = []
    total_start = time.time()

    results_dir = config["data"].get("results", "data/results")
    os.makedirs(results_dir, exist_ok=True)

    for seed_idx, seed in enumerate(seeds):
        print(f"\n{'=' * 100}")
        print(f"  SEED {seed_idx+1}/{len(seeds)}: {seed}")
        print(f"{'=' * 100}")

        set_all_seeds(seed)

        use_balanced_sampling = train_cfg.get("balanced_sampling", False)
        use_oversampling = train_cfg.get("oversampling", False)

        if use_balanced_sampling:
            batch_sampler = BalancedBatchSampler(
                labels=train_dataset.get_labels(),
                batch_size=batch_size
            )
            train_loader = DataLoader(
                train_dataset, batch_sampler=batch_sampler,
                num_workers=4, pin_memory=True
            )
        elif use_oversampling:
            sampler = make_weighted_sampler(train_dataset)
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, sampler=sampler,
                num_workers=4, pin_memory=True
            )
        else:
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True,
                num_workers=4, pin_memory=True
            )

        print(f"    Batch size: {batch_size} | Batches/epoch: {len(train_loader)}")

        seed_ckpt_dir = os.path.join(checkpoint_dir, f"seed_{seed}") if multi_seed else checkpoint_dir

        for conv_type in archs_to_run:

            result = train_and_evaluate(
                conv_type=conv_type,
                config=config,
                device=device,
                train_dataset=train_dataset,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                node_feat_dim=node_feat_dim,
                edge_feat_dim=edge_feat_dim,
                checkpoint_dir=seed_ckpt_dir,
                fresh=args.fresh,
            )

            if result is not None:
                result["seed"] = seed
                all_results.append(result)
                multi_seed_results[result["arch"]].append(result)
            else:
                print(f"\n  [{ARCH_NAMES[conv_type]}] Training failed — excluded from comparison")


    total_time = time.time() - total_start
    print(f"\nTotal time: {total_time/60:.1f} min")

    if len(all_results) > 1:
        print_comparison_table(all_results)

    if multi_seed:
        print_multi_seed_summary(multi_seed_results)

    if all_results:
        compact_results = []
        for r in all_results:
            compact = {k: v for k, v in r.items() if k not in ("test_probs", "test_preds", "test_labels")}
            compact_results.append(compact)

        comparison_path = os.path.join(results_dir, "comparison.json")
        with open(comparison_path, "w") as f:
            json.dump(compact_results, f, indent=2)
        print(f"Comparison saved to: {comparison_path}")

        full_results_path = os.path.join(results_dir, "full_results.json")
        with open(full_results_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Full results (with predictions) saved to: {full_results_path}")

        if multi_seed:
            summary_data = {}
            for arch, runs in multi_seed_results.items():
                if not runs:
                    continue
                metrics_summary = {}
                for m in ["f1_macro", "test_acc", "roc_auc", "pr_auc", "recall", "specificity",
                          "val_f1_macro", "val_acc", "val_roc_auc", "val_pr_auc", "val_recall", "val_specificity"]:
                    vals = [r[m] for r in runs]
                    metrics_summary[m] = {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0), "values": vals}
                summary_data[arch] = metrics_summary

            summary_path = os.path.join(results_dir, "multi_seed_summary.json")
            with open(summary_path, "w") as f:
                json.dump(summary_data, f, indent=2)
            print(f"Multi-seed summary saved to: {summary_path}")

    print("Done!")


if __name__ == "__main__":
    main()
