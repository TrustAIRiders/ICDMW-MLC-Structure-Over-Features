import os
import json
import yaml
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from src.models.model import EllipticGNN
from src.models.calibration import load_decision_rule, decide
from src.dataset.Elliptic2Dataset import Elliptic2Dataset


ALL_CONV_TYPES = ["gatv2", "sage", "sage_edge", "gin"]

ARCH_NAMES = {
    "gatv2":     "GATv2",
    "sage":      "SAGE",
    "sage_edge": "SAGE+Edge",
    "gin":       "GIN",
}


def load_yaml_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_device(device_arg=None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(conv_type, checkpoint_dir, seed, node_feat_dim, edge_feat_dim, config, device):
    mcfg = config["model"]
    model = EllipticGNN(
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
    )
    ckpt_path = os.path.join(checkpoint_dir, f"seed_{seed}", conv_type, "best_model.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    load_decision_rule(model, os.path.dirname(ckpt_path))
    return model.to(device).eval()


def load_datasets(config, splits=("train", "test")) -> dict:
    processed_dir = config["data"]["processed_dir"]
    val_ratio     = config["training"]["val_ratio"]
    test_ratio    = config["training"]["test_ratio"]
    return {
        split: Elliptic2Dataset(
            processed_dir, split=split, val_ratio=val_ratio, test_ratio=test_ratio
        )
        for split in splits
    }


def get_feat_dims(dataset) -> tuple[int, int]:
    sample = dataset[0]
    return sample.x.shape[1], sample.edge_attr.shape[1]


def compute_feat_bounds(train_dataset) -> tuple:
    feat_min = feat_max = None
    for data in tqdm(train_dataset, desc="  Scanning feat bounds", leave=False):
        bmin = data.x.min(dim=0).values
        bmax = data.x.max(dim=0).values
        if feat_min is None:
            feat_min, feat_max = bmin.clone(), bmax.clone()
        else:
            feat_min = torch.minimum(feat_min, bmin)
            feat_max = torch.maximum(feat_max, bmax)
    return feat_min, feat_max


@torch.no_grad()
def infer(model, data, device) -> tuple[int, float]:
    d = data.clone().to(device)
    if not hasattr(d, "batch") or d.batch is None:
        d.batch = torch.zeros(d.num_nodes, dtype=torch.long, device=device)
    probs, preds = decide(model, model(d))
    prob   = float(probs[0].item())
    pred   = int(preds[0].item())
    return pred, prob


def save_results(results: dict, output_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {output_path}")
