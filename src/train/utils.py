import random
import tempfile

import numpy as np
import torch
import os
import yaml
from sklearn.metrics import (
    f1_score, accuracy_score,
    confusion_matrix, roc_auc_score, average_precision_score,
    precision_score, recall_score
)

def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_paths(config):
    spot_mode = config.get("spot_mode", False)
    persist = config.get("persistent_storage_path", "/persistent")

    processed_dir = config["data"]["processed_dir"]
    checkpoint_dir = config["training"]["checkpoint_dir"]

    if spot_mode:
        processed_dir = os.path.join(persist, "processed")
        checkpoint_dir = os.path.join(persist, "checkpoints")
        print(f"Spot mode ON — using persistent storage paths:")
        print(f"processed:   {processed_dir}")
        print(f"checkpoints: {checkpoint_dir}")
    else:
        print(f"Standard mode — using local paths:")
        print(f"processed:   {processed_dir}")
        print(f"checkpoints: {checkpoint_dir}")

    return processed_dir, checkpoint_dir

def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def compute_class_weights(dataset, num_classes, device):
    labels = dataset.get_labels()
    class_counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    total = len(labels)

    weights = total / (num_classes * class_counts)
    weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)

    print(f"    Class counts: {dict(enumerate(class_counts.astype(int)))}")
    print(f"    Class weights: {dict(enumerate(weights.round(2)))}")
    return weights_tensor

def print_comprehensive_metrics(y_true, y_pred, y_probs, target_names=None):
    print("\n" + "=" * 80)
    print("COMPREHENSIVE EVALUATION METRICS")
    print("=" * 80)

    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"                 Neg    Pos")
    print(f"  Actual  Neg  [{cm[0,0]:6d}  {cm[0,1]:6d}]")
    print(f"          Pos  [{cm[1,0]:6d}  {cm[1,1]:6d}]")

    tn, fp, fn, tp = cm.ravel()
    print(f"\n  True Negatives:  {tn:6d}")
    print(f"  False Positives: {fp:6d}")
    print(f"  False Negatives: {fn:6d}")
    print(f"  True Positives:  {tp:6d}")

    print("\nPer-Class Metrics:")
    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_arr = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    if target_names is None:
        target_names = ["Class 0", "Class 1"]

    for i, name in enumerate(target_names):
        print(f"  {name:12s}: Precision={precision[i]:.4f}, Recall={recall_arr[i]:.4f}, F1={f1[i]:.4f}")

    print("\nAveraged Metrics:")
    print(f"  Precision (macro):   {precision_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print(f"  Recall (macro):      {recall_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print(f"  F1 (macro):          {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print(f"  F1 (weighted):       {f1_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")

    try:
        roc_auc = roc_auc_score(y_true, y_probs)
        pr_auc = average_precision_score(y_true, y_probs)
        print("\nArea Under Curve Metrics:")
        print(f"  ROC-AUC:  {roc_auc:.4f}")
        print(f"  PR-AUC:   {pr_auc:.4f}")
    except ValueError as e:
        print(f"\nWarning: Could not compute AUC metrics: {e}")

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    Recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    print("\nAdditional Metrics:")
    print(f"  Specificity (TNR): {specificity:.4f}")
    print(f"  Recall (TPR): {Recall:.4f}")
    print(f"  Accuracy:          {accuracy_score(y_true, y_pred):.4f}")

    print("=" * 80 + "\n")

    return {
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "specificity": specificity, "recall": Recall,
    }

def build_save_state(epoch, model, optimizer, scheduler, best_val_f1, history):
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_f1": best_val_f1,
        "history": history,
    }

def atomic_save(state, filepath):
    dirpath = os.path.dirname(filepath) or "."
    os.makedirs(dirpath, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(dir=dirpath, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            torch.save(state, f)
        os.replace(tmp_path, filepath)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise