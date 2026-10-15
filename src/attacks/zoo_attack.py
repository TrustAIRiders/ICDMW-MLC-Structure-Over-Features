import numpy as np
import torch
import torch.nn.functional as F

from src.models.calibration import decide

class ModelOracle:
    def __init__(self, model, device, threshold: float = 0.5):
        self.model = model
        self.device = device
        self.threshold = threshold
        self._query_count = 0

    @property
    def query_count(self) -> int:
        return self._query_count

    def reset(self):
        self._query_count = 0

    @torch.no_grad()
    def query(self, data) -> float:
        self._query_count += 1
        d = data.clone().to(self.device)
        if not hasattr(d, "batch") or d.batch is None:
            d.batch = torch.zeros(d.num_nodes, dtype=torch.long, device=self.device)
        probs, _ = decide(self.model, self.model(d))
        return float(probs[0].item())
    
    def query_with_x(self, data, x_override: torch.Tensor) -> float:
        d = data.clone()
        d.x = x_override
        return self.query(d)


class ZOOAttack:
    def __init__(
        self,
        oracle: ModelOracle,
        epsilon: float,
        delta: float,
        subspace_dim: int,
        max_queries: int,
        step_size: float,
        feat_min: torch.Tensor,
        feat_max: torch.Tensor,
    ):
        self.oracle = oracle
        self.epsilon = epsilon
        self.delta = delta
        self.subspace_dim = subspace_dim
        self.max_queries = max_queries
        self.step_size = step_size
        self.feat_min = feat_min.cpu()
        self.feat_max = feat_max.cpu()

    def _random_subspace(self, total_dims: int, k: int) -> np.ndarray:
        return np.random.choice(total_dims, size=min(k, total_dims), replace=False)

    def _estimate_gradient(
        self,
        data,
        x_current: torch.Tensor,
        subspace_indices: np.ndarray,
    ) -> torch.Tensor:
        
        N = x_current.shape[0]
        grad = torch.zeros(N, len(subspace_indices))

        for j, f in enumerate(subspace_indices):
            x_plus = x_current.clone()
            x_plus[:, f] = (x_current[:, f] + self.delta).clamp(
                self.feat_min[f].item(), self.feat_max[f].item()
            )
            x_minus = x_current.clone()
            x_minus[:, f] = (x_current[:, f] - self.delta).clamp(
                self.feat_min[f].item(), self.feat_max[f].item()
            )

            p_plus  = self.oracle.query_with_x(data, x_plus)
            p_minus = self.oracle.query_with_x(data, x_minus)

            scalar_grad = (p_plus - p_minus) / (2.0 * self.delta)
            grad[:, j] = scalar_grad

        return grad

    def _project(self, x_orig: torch.Tensor, x_adv: torch.Tensor) -> torch.Tensor:
        delta = (x_adv - x_orig).clamp(-self.epsilon, self.epsilon)
        return (x_orig + delta).clamp(self.feat_min, self.feat_max)

    def attack(self, data):
        self.oracle.reset()

        x_orig      = data.x.clone().cpu()
        total_feats = x_orig.shape[1]
        x_adv       = x_orig.clone()

        p_current = self.oracle.query_with_x(data, x_adv)

        if p_current < self.oracle.threshold:
            result = data.clone()
            result.x = x_adv
            return result, True, self.oracle.query_count, self.oracle.query_count

        best_x = x_adv.clone()
        best_prob = p_current
        success_at_query = -1

        while self.oracle.query_count < self.max_queries:
            remaining = self.max_queries - self.oracle.query_count
            if remaining < 3:
                break
            k = min(self.subspace_dim, (remaining - 1) // 2)
            if k == 0:
                break

            subspace_idx = self._random_subspace(total_feats, k)
            grad = self._estimate_gradient(data, x_adv, subspace_idx)

            x_new = x_adv.clone()
            for j, f in enumerate(subspace_idx):
                x_new[:, f] -= self.step_size * torch.sign(grad[:, j])

            x_new = self._project(x_orig, x_new)

            if self.oracle.query_count >= self.max_queries:
                break
            p_new = self.oracle.query_with_x(data, x_new)

            x_adv = x_new
            if p_new < best_prob:
                best_prob = p_new
                best_x    = x_adv.clone()

            if p_new < self.oracle.threshold:
                success_at_query = self.oracle.query_count
                break

        result   = data.clone()
        result.x = best_x
        return result, success_at_query != -1, self.oracle.query_count, success_at_query
