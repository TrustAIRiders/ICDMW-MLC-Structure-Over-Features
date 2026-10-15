import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TemperatureScaling(nn.Module):
    def __init__(self, init_temp=1.5):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * init_temp)

    def forward(self, logits):
        return logits / self.temperature


@torch.no_grad()
def collect_logits_labels(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        all_logits.append(logits.cpu())
        all_labels.append(batch.y.squeeze().cpu())

    return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0)


def fit_temperature(model, val_loader, device, max_iter=50, lr=0.01):
    logits, labels = collect_logits_labels(model, val_loader, device)

    temp_model = TemperatureScaling(init_temp=1.5)
    optimizer = torch.optim.LBFGS([temp_model.temperature], lr=lr, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        scaled_logits = temp_model(logits)
        loss = F.cross_entropy(scaled_logits, labels)
        loss.backward()
        return loss

    optimizer.step(closure)

    learned_temp = temp_model.temperature.item()
    print(f"    Learned temperature: {learned_temp:.4f}")

    with torch.no_grad():
        nll_before = F.cross_entropy(logits, labels).item()
        nll_after = F.cross_entropy(logits / learned_temp, labels).item()
        print(f"    NLL before: {nll_before:.4f} | after: {nll_after:.4f}")

    return learned_temp


def calibrate_probabilities(logits, temperature):
    if isinstance(logits, np.ndarray):
        logits = torch.tensor(logits, dtype=torch.float32)

    scaled = logits / temperature
    probs = torch.softmax(scaled, dim=-1)[:, 1].numpy()
    return probs


DECISION_RULE_FILE = "calibration.json"


def save_decision_rule(ckpt_dir, temperature, threshold):
    import json, os
    with open(os.path.join(ckpt_dir, DECISION_RULE_FILE), "w") as f:
        json.dump({"temperature": float(temperature), "threshold": float(threshold)}, f, indent=2)


def load_decision_rule(model, ckpt_dir):
    import json, os
    with open(os.path.join(ckpt_dir, DECISION_RULE_FILE)) as f:
        rule = json.load(f)
    model.temperature = rule["temperature"]
    model.threshold = rule["threshold"]
    return model


def decide(model, logits):
    probs = torch.softmax(logits / getattr(model, "temperature", 1.0), dim=-1)[:, 1]
    threshold = getattr(model, "threshold", None)
    preds = (probs >= threshold).long() if threshold is not None else logits.argmax(dim=-1)
    return probs, preds
