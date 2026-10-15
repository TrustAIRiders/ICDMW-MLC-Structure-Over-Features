import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm

from src.models.calibration import decide
from src.attacks.pgd_feature import PGDFeatureAttack

class TransferAttack:

    def __init__(
            self,
            surrogate,
            epsilon: float,
            steps: int = 40,
            step_size: float = None,
            num_restarts: int = 5,
            feat_min=None,
            feat_max=None
            ):
        self.surrogate = surrogate
        self.epsilon = epsilon
        self.steps = steps
        self.step_size = step_size
        self.num_restarts = num_restarts
        self.feat_min = feat_min
        self.feat_max = feat_max

        self.pgd = PGDFeatureAttack(
            model=self.surrogate,
            epsilon=self.epsilon,
            steps=self.steps,
            step_size=self.step_size,
            num_restarts=self.num_restarts,
            feat_min=self.feat_min,
            feat_max=self.feat_max
        )

    def generate_perturbations(self, test_dataset, tp_indices) -> list:

        perturbed_graphs = []
        for idx in tqdm(tp_indices, desc="Generating perturbed graphs", leave=False):
            data = test_dataset[int(idx)]
            adv = self.pgd.attack(data)
            perturbed_graphs.append(adv.cpu())
        return perturbed_graphs
    

    @torch.no_grad()
    def evaluate_transfer(
        self,
        target_model,
        test_dataset,
        tp_indices,
        perturbed_graphs: list,
        labels: np.ndarray, 
        preds_clean: np.ndarray,
        probs_clean: np.ndarray 
        ) -> dict:
        device = next(target_model.parameters()).device

        preds_post = preds_clean.copy()
        probs_post = probs_clean.copy()

        n_evaded = 0
        pg_conf_pre, pg_conf_post, pg_l2 = [], [], []
        pg_evaded = []

        for i, idx in enumerate(tp_indices):
            adv = perturbed_graphs[i].to(device)

            logits_post = target_model(adv)
            probs_post_t, preds_post_t = decide(target_model, logits_post)
            prob_post = float(probs_post_t[0].item())
            pred_post = int(preds_post_t[0].item())

            preds_post[int(idx)] = pred_post
            probs_post[int(idx)] = prob_post

            evaded = 1 if pred_post == 0 else 0
            n_evaded += evaded
            pg_evaded.append(evaded)

            orig_x = test_dataset[int(idx)].x
            delta = (adv.x.cpu() - orig_x).abs()
            graph_l2 = float(delta.pow(2).sum().sqrt().item())

            pg_conf_pre.append(float(probs_clean[int(idx)]))
            pg_conf_post.append(prob_post)
            pg_l2.append(graph_l2)

        n_attacked = len(tp_indices)
        n_suspicious = int((labels == 1).sum())
        n_tp_target = int((preds_clean[labels == 1] == 1).sum())

        post_f1 = float(f1_score(labels, preds_post, average="macro", zero_division=0))
        clean_f1 = float(f1_score(labels, preds_clean, average="macro", zero_division=0))

        conf_pre_arr = np.array(pg_conf_pre)
        conf_post_arr = np.array(pg_conf_post)

        return {
            "n_suspicious":            n_suspicious,
            "n_tp_target":             n_tp_target,
            "n_attacked":              n_attacked,
            "n_evaded":                n_evaded,
            "transfer_asr":            round(n_evaded / max(n_attacked, 1), 6),
            "clean_f1_macro":          round(clean_f1, 6),
            "post_f1_macro":           round(post_f1, 6),
            "f1_drop":                 round(post_f1 - clean_f1, 6),
            "clean_recall_suspicious": round(n_tp_target / max(n_suspicious, 1), 6),
            "post_recall_suspicious":  round(
                (n_tp_target - n_evaded) / max(n_suspicious, 1), 6
            ),
            "mean_confidence_pre":     round(float(conf_pre_arr.mean()), 6),
            "mean_confidence_post":    round(float(conf_post_arr.mean()), 6),
            "mean_confidence_drop":    round(float((conf_pre_arr - conf_post_arr).mean()), 6),
            "mean_l2_perturbation":    round(float(np.mean(pg_l2)), 6),
            "per_graph": {
            "subgraph_indices": tp_indices.tolist(),
            "evaded":           pg_evaded,
            "confidence_pre":   pg_conf_pre,
            "confidence_post":  pg_conf_post,
            "l2_perturbation":  pg_l2,
        },
        }


@torch.no_grad()
def get_clean_predictions(model, dataset, device, batch_size: int = 512):

    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    all_probs, all_labels, all_preds = [], [], []    
    
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        probs, preds = decide(model, logits)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch.y.view(-1).cpu().numpy())
        all_probs.extend(probs.cpu().numpy())    
    return (
        np.array(all_labels, dtype=int),
        np.array(all_preds, dtype=int),
        np.array(all_probs, dtype=float)
    )
    

def get_tp_indices(labels: np.ndarray, preds: np.ndarray) -> np.ndarray:

    return np.where((labels == 1) & (preds == 1))[0]