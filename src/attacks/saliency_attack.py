from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from src.attacks.pgd_feature import PGDFeatureAttack


class SaliencyGuidedAttack(PGDFeatureAttack):
    def __init__(
        self,
        model,
        importance_scores: np.ndarray,
        top_k: int,
        epsilon: float,
        steps: int,
        feat_min: torch.Tensor,
        feat_max: torch.Tensor,
        step_size: float | None = None,
        num_restarts: int = 5,
    ):
        n_features = len(importance_scores)
        if not (1 <= top_k <= n_features):
            raise ValueError(
                f"top_k must be in [1, {n_features}], got {top_k}"
            )
        if step_size is None:
            step_size = 2.5 * epsilon / steps

        super().__init__(
            model=model,
            epsilon=epsilon,
            steps=steps,
            step_size=step_size,
            num_restarts=num_restarts,
            feat_min=feat_min,
            feat_max=feat_max,
        )

        top_indices = np.argsort(importance_scores)[-top_k:]
        mask = torch.zeros(n_features, dtype=torch.float32)
        mask[top_indices] = 1.0

        self.feature_mask: torch.Tensor = mask
        self.top_k = top_k
        self.top_indices: list[int] = sorted(top_indices.tolist())

    def get_attack_mask(self) -> torch.Tensor:
        return self.feature_mask.clone()


    def pgd_step(self, data, x_adv: torch.Tensor) -> torch.Tensor:
        device = x_adv.device
        x_orig = data.x
        mask = self.feature_mask.to(device)  

        x_adv = x_adv.detach().requires_grad_(True)

        perturbed = data.clone()
        perturbed.x = x_adv

        logits = self.model(perturbed)
        loss = F.cross_entropy(logits, data.y.view(-1).to(device))
        loss.backward()

        with torch.no_grad():
            masked_grad = x_adv.grad * mask
            x_adv_new = x_adv + self.step_size * masked_grad.sign()
            x_adv_new = self.project(
                x_orig=x_orig, x_adv=x_adv_new, device=device
            )

        return x_adv_new

    def single_restart(self, data, device: torch.device):
        x_orig = data.x.detach()
        mask = self.feature_mask.to(device)

        noise = torch.zeros_like(x_orig).uniform_(-self.epsilon, self.epsilon)
        noise = noise * mask
        x_adv = self.project(
            x_orig=x_orig, x_adv=x_orig + noise, device=device
        )

        for _ in range(self.steps):
            x_adv = self.pgd_step(data=data, x_adv=x_adv)

        with torch.no_grad():
            perturbed = data.clone()
            perturbed.x = x_adv
            logits = self.model(perturbed)
            final_loss = F.cross_entropy(
                logits, data.y.view(-1).to(device)
            ).item()

        return x_adv, final_loss
