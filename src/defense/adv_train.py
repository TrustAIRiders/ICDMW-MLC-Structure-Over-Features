import os
import json
import time
import signal

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm

from src.train.threshold import cost_aware_threshold_search
from src.train.utils import atomic_save
from src.defense.structural import (
    perturb_structural, sample_structural_params, n_graphs,
)


_STOP = {"requested": False}


def stop_requested():
    return _STOP["requested"]


def install_signal_handlers():

    def handler(signum, _frame):
        if _STOP["requested"]:
            print("\n  Second signal — aborting now (last completed epoch is on disk).")
            raise KeyboardInterrupt
        _STOP["requested"] = True
        print(f"\n  Signal {signum} received — finishing this epoch, then saving and exiting.")

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)


def load_or_compute_feat_bounds(train_dataset, cache_path):
    if cache_path and os.path.exists(cache_path):
        bounds = torch.load(cache_path, map_location="cpu", weights_only=True)
        return bounds["feat_min"], bounds["feat_max"]

    feat_min = feat_max = None
    for data in tqdm(train_dataset, desc="  Scanning feat bounds", leave=False):
        bmin = data.x.min(dim=0).values
        bmax = data.x.max(dim=0).values
        if feat_min is None:
            feat_min, feat_max = bmin.clone(), bmax.clone()
        else:
            feat_min = torch.minimum(feat_min, bmin)
            feat_max = torch.maximum(feat_max, bmax)

    if cache_path:
        atomic_save({"feat_min": feat_min, "feat_max": feat_max}, cache_path)
    return feat_min, feat_max


def _project(x_orig, x_adv, epsilon, feat_min, feat_max):
    delta = (x_adv - x_orig).clamp(-epsilon, epsilon)
    return (x_orig + delta).clamp(feat_min, feat_max)


def pgd_perturb_batch(model, batch, epsilon, steps, step_size,
                      feat_min, feat_max, random_start=True, inner_loss="ce",
                      criterion=None):
    if epsilon <= 0 or steps <= 0:
        return batch.x.detach()

    was_training = model.training
    model.eval()
    saved_rg = [p.requires_grad for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)

    x_orig = batch.x.detach()
    y = batch.y.view(-1)

    if random_start:
        noise = torch.empty_like(x_orig).uniform_(-epsilon, epsilon)
        x_adv = _project(x_orig, x_orig + noise, epsilon, feat_min, feat_max)
    else:
        x_adv = x_orig.clone()

    try:
        for _ in range(steps):
            x_adv = x_adv.detach().requires_grad_(True)
            batch.x = x_adv
            logits = model(batch)
            if inner_loss == "focal" and criterion is not None:
                loss = criterion(logits, y)
            else:
                loss = F.cross_entropy(logits, y)
            grad = torch.autograd.grad(loss, x_adv)[0]
            with torch.no_grad():
                x_adv = _project(x_orig, x_adv + step_size * grad.sign(),
                                 epsilon, feat_min, feat_max)
    finally:
        batch.x = x_orig
        for p, rg in zip(model.parameters(), saved_rg):
            p.requires_grad_(rg)
        if was_training:
            model.train()

    return x_adv.detach()


def adv_train_epoch(model, loader, optimizer, criterion, device, weights,
                    feat_atk=None, struct_atk=None, desc="", limit_batches=0):
    model.train()
    sums = {"total": 0.0, "feat": 0.0, "struct": 0.0}
    n_batches = 0

    use_feat = feat_atk is not None and weights.get("feat", 0.0) > 0.0
    use_struct = struct_atk is not None and weights.get("struct", 0.0) > 0.0

    for batch in tqdm(loader, desc=desc, leave=False):
        if limit_batches and n_batches >= limit_batches:
            break
        batch = batch.to(device, non_blocking=True)
        y = batch.y.view(-1)

        x_adv = pgd_perturb_batch(model, batch, criterion=criterion, **feat_atk) if use_feat else None
        struct_batch = None
        if use_struct:
            params = sample_structural_params(
                struct_atk["rng"], struct_atk["k_choices"],
                struct_atk["conn_choices"], struct_atk["strategies"],
            )
            struct_batch = perturb_structural(batch, struct_atk["stats"], params)

        optimizer.zero_grad(set_to_none=True)
        loss = 0.0
        feat_loss = struct_loss = float("nan")

        if weights.get("clean", 0.0) > 0.0:
            loss = loss + weights["clean"] * criterion(model(batch), y)

        if use_feat:
            x_clean = batch.x
            batch.x = x_adv
            raw = criterion(model(batch), y)
            batch.x = x_clean
            feat_loss = raw.item()
            loss = loss + weights["feat"] * raw

        if use_struct:
            raw = criterion(model(struct_batch), y)
            struct_loss = raw.item()
            loss = loss + weights["struct"] * raw

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        sums["total"] += loss.item()
        if use_feat:
            sums["feat"] += feat_loss
        if use_struct:
            sums["struct"] += struct_loss
        n_batches += 1

    denom = max(n_batches, 1)
    return (sums["total"] / denom,
            sums["feat"] / denom if use_feat else float("nan"),
            sums["struct"] / denom if use_struct else float("nan"))


def _evaluate_perturbed(model, loader, criterion, device, perturb, desc,
                        fn_cost=None, fp_cost=None, max_graphs=0):
    model.eval()
    all_probs, all_labels = [], []
    loss_sum, n_batches, seen = 0.0, 0, 0

    for batch in tqdm(loader, desc=desc, leave=False):
        batch = batch.to(device, non_blocking=True)
        y = batch.y.view(-1)

        perturbed = perturb(batch)

        with torch.no_grad():
            logits = model(perturbed)
            loss_sum += criterion(logits, y).item()
            n_batches += 1
            all_probs.extend(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
            all_labels.extend(y.cpu().numpy())

        seen += n_graphs(batch)
        if max_graphs and seen >= max_graphs:
            break

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    if fn_cost is not None and fp_cost is not None:
        threshold, _, _ = cost_aware_threshold_search(all_probs, all_labels, fn_cost, fp_cost)
    else:
        threshold = 0.5

    preds = (all_probs >= threshold).astype(int)
    return {
        "loss": loss_sum / max(n_batches, 1),
        "acc": float(accuracy_score(all_labels, preds)),
        "f1_macro": float(f1_score(all_labels, preds, average="macro", zero_division=0)),
        "threshold": float(threshold),
        "n_graphs": int(len(all_labels)),
        "recall": float((preds[all_labels == 1] == 1).mean()) if (all_labels == 1).any() else 0.0,
        "evasion_rate": float((preds[all_labels == 1] == 0).mean()) if (all_labels == 1).any() else 0.0,
    }


def evaluate_robust(model, loader, criterion, device, atk,
                    fn_cost=None, fp_cost=None, max_graphs=0):

    def perturb(batch):
        x_adv = pgd_perturb_batch(model, batch, criterion=criterion, **atk)
        clone = batch.clone()
        clone.x = x_adv
        return clone

    return _evaluate_perturbed(model, loader, criterion, device, perturb,
                               "  robust val (pgd)", fn_cost, fp_cost, max_graphs)


def evaluate_structural(model, loader, criterion, device, stats, params,
                        fn_cost=None, fp_cost=None, max_graphs=0):

    def perturb(batch):
        return perturb_structural(batch, stats, params)

    label = f"  robust val ({params['strategy']} k={params['k_nodes']})"
    return _evaluate_perturbed(model, loader, criterion, device, perturb,
                               label, fn_cost, fp_cost, max_graphs)


def evaluate_clean(model, loader, criterion, device, fn_cost=None, fp_cost=None, max_graphs=0):
    return _evaluate_perturbed(model, loader, criterion, device, lambda b: b,
                               "  clean val", fn_cost, fp_cost, max_graphs)


def empty_history():
    return {
        "epsilon": [], "lr": [], "epoch_seconds": [],
        "train_loss": [], "train_adv_loss": [], "train_struct_loss": [],
        "val_clean_loss": [], "val_clean_f1_macro": [], "val_clean_acc": [],
        "val_robust_f1_macro": [], "val_robust_acc": [], "val_robust_recall": [],
        "val_struct_f1_macro": [], "val_struct_acc": [], "val_struct_recall": [],
    }


def build_state(epoch, model, optimizer, scheduler, best_metric, history, patience_counter, meta):
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "best_metric": best_metric,
        "history": history,
        "patience_counter": patience_counter,
        "meta": meta,
    }


def save_history(history, path):
    with open(path, "w") as f:
        json.dump(history, f, indent=2)


def format_eta(seconds):
    if not np.isfinite(seconds):
        return "unlimited"
    if seconds <= 0:
        return "—"
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def epsilon_for_epoch(epoch, epsilon, warmup_epochs):
    if warmup_epochs <= 0:
        return epsilon
    return epsilon * min(1.0, (epoch + 1) / warmup_epochs)


def time_left(deadline):
    return float("inf") if deadline is None else deadline - time.time()
