from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


class FeatureImportanceAnalyzer:

    def __init__(self, model, dataset, device):
        self.model = model
        self.device = device
        self.n_features = dataset[0].x.shape[1]

        self.suspicious_data = [d for d in dataset if int(d.y.item()) == 1]

        if not self.suspicious_data:
            raise RuntimeError("No suspicious (label=1) samples found in dataset.")

    def _prepare(self, data):
        d = data.clone().to(self.device)
        if not hasattr(d, "batch") or d.batch is None:
            d.batch = torch.zeros(d.num_nodes, dtype=torch.long, device=self.device)
        return d

    def _freeze_model(self) -> dict:
        saved = {n: p.requires_grad for n, p in self.model.named_parameters()}
        for p in self.model.parameters():
            p.requires_grad_(False)
        return saved

    def _unfreeze_model(self, saved: dict) -> None:
        for n, p in self.model.named_parameters():
            p.requires_grad_(saved[n])

    def _subsample(self, max_samples: Optional[int]) -> list:
        if max_samples is None or max_samples >= len(self.suspicious_data):
            return self.suspicious_data
        return self.suspicious_data[:max_samples]


    def gradient_saliency(self, max_samples: Optional[int] = 500) -> np.ndarray:
        self.model.eval()
        saved = self._freeze_model()

        samples = self._subsample(max_samples)
        importance = np.zeros(self.n_features, dtype=np.float64)

        for data in tqdm(samples, desc="  Gradient saliency", leave=False):
            d = self._prepare(data)

            x = d.x.detach().float().requires_grad_(True)
            d.x = x

            logits = self.model(d)
            loss = F.cross_entropy(logits, d.y.view(-1).to(self.device))
            loss.backward()

            importance += x.grad.abs().mean(dim=0).detach().cpu().numpy()

        self._unfreeze_model(saved)
        return importance / len(samples)


    def integrated_gradients(
        self,
        n_steps: int = 50,
        max_samples: Optional[int] = 200,
    ) -> np.ndarray:
        self.model.eval()
        saved = self._freeze_model()

        samples = self._subsample(max_samples)
        importance = np.zeros(self.n_features, dtype=np.float64)

        for data in tqdm(samples, desc="  Integrated gradients", leave=False):
            d = self._prepare(data)
            x_orig = d.x.detach().float()

            accumulated_grad = torch.zeros_like(x_orig)

            for step in range(1, n_steps + 1):
                alpha = step / n_steps
                x_interp = (alpha * x_orig).requires_grad_(True)
                d.x = x_interp

                logits = self.model(d)
                loss = F.cross_entropy(logits, d.y.view(-1).to(self.device))
                loss.backward()

                accumulated_grad += x_interp.grad.detach()

            ig = x_orig * (accumulated_grad / n_steps)
            importance += ig.abs().mean(dim=0).cpu().numpy()

        self._unfreeze_model(saved)
        return importance / len(samples)


    def ablation_importance(self, max_samples: Optional[int] = 200) -> np.ndarray:
        self.model.eval()

        samples = self._subsample(max_samples)

        feat_sum = torch.zeros(self.n_features, dtype=torch.float64)
        n_total_nodes = 0
        for data in samples:
            feat_sum += data.x.double().sum(dim=0).cpu()
            n_total_nodes += data.num_nodes
        feat_mean = (feat_sum / n_total_nodes).float()

        baseline_correct = 0
        for data in samples:
            d = self._prepare(data)
            with torch.no_grad():
                logits = self.model(d)
                if logits.argmax(dim=-1).item() == 1:
                    baseline_correct += 1
        baseline_recall = baseline_correct / len(samples)

        importance = np.zeros(self.n_features, dtype=np.float64)
        mean_val = feat_mean.to(self.device)

        for feat_idx in tqdm(range(self.n_features), desc="  Ablation", leave=False):
            correct = 0
            for data in samples:
                d = self._prepare(data)
                d.x = d.x.clone()
                d.x[:, feat_idx] = mean_val[feat_idx]
                with torch.no_grad():
                    logits = self.model(d)
                    if logits.argmax(dim=-1).item() == 1:
                        correct += 1
            ablated_recall = correct / len(samples)
            importance[feat_idx] = baseline_recall - ablated_recall

        return importance


    def plot_importance(
        self,
        scores_dict: dict,
        save_path: str,
        top_k_highlight: int = 10,
        arch_name: str = "",
    ) -> None:
        METHOD_LABELS = {
            "gradient": "Gradient Saliency  |∂L/∂x|",
            "ig":       "Integrated Gradients",
            "ablation": "Ablation  (recall drop)",
        }

        methods = list(scores_dict.keys())
        n_methods = len(methods)
        x_ticks_pos = np.arange(1, self.n_features + 1)

        sns.set_theme(style="whitegrid", font_scale=1.0)
        fig, axes = plt.subplots(n_methods, 1, figsize=(14, 3.5 * n_methods))
        if n_methods == 1:
            axes = [axes]

        for ax, method in zip(axes, methods):
            scores = scores_dict[method]
            top_idx = set(np.argsort(scores)[-top_k_highlight:].tolist())
            colors = [
                "#E53935" if i in top_idx else "#90CAF9"
                for i in range(self.n_features)
            ]

            ax.bar(x_ticks_pos, scores, color=colors, width=0.7, edgecolor="none")
            ax.set_xlabel("Feature index", fontsize=10)
            ax.set_ylabel("Importance score", fontsize=10)
            ax.set_title(METHOD_LABELS.get(method, method), fontsize=11,
                         fontweight="bold")
            ax.set_xlim(0.5, self.n_features + 0.5)

            major_ticks = [1] + list(range(5, self.n_features + 1, 5))
            if self.n_features not in major_ticks:
                major_ticks.append(self.n_features)
            ax.set_xticks(major_ticks)

            top5 = np.argsort(scores)[::-1][:5]
            for i in top5:
                ax.text(
                    i + 1, scores[i], f"f{i + 1}",
                    ha="center", va="bottom", fontsize=8,
                    color="#B71C1C", fontweight="bold",
                )

        title = f"Feature Importance — {arch_name}" if arch_name else "Feature Importance"
        fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
        plt.tight_layout()

        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved importance plot → {save_path}")
