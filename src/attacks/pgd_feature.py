import torch
import torch.nn.functional as F


class PGDFeatureAttack:

    def __init__(self, model, epsilon, steps=40, step_size=None,
                 num_restarts=5, feat_min=None, feat_max=None):
        if feat_min is None or feat_max is None:
            raise ValueError("feat_min and feat_max must be provided (compute from training set).")

        self.model = model
        self.epsilon = epsilon
        self.steps = steps
        self.step_size = step_size
        self.num_restarts = num_restarts
        self.feat_min = feat_min.cpu()
        self.feat_max = feat_max.cpu()

    
    def prepare(self, data, device):

        d = data.clone().to(device)
        if not hasattr(d, "batch") or d.batch is None:
            d.batch = torch.zeros(d.num_nodes, dtype=torch.long, device=device)
        return d

    def project(self, x_orig, x_adv, device):
        feat_min = self.feat_min.to(device)
        feat_max = self.feat_max.to(device)

        delta = (x_adv - x_orig).clamp(-self.epsilon, self.epsilon) 
        return (x_orig + delta).clamp(feat_min, feat_max).detach()
    
    def pgd_step(self, data, x_adv):

        device = x_adv.device 
        x_orig = data.x

        x_adv = x_adv.detach().requires_grad_(True)

        perturbed = data.clone()
        perturbed.x = x_adv

        logits = self.model(perturbed)
        loss = F.cross_entropy(logits, data.y.view(-1).to(device))
        loss.backward()

        with torch.no_grad():
            x_adv_new = x_adv + self.step_size * x_adv.grad.sign()
            x_adv_new = self.project(x_orig=x_orig, x_adv=x_adv_new, device=device)
        
        return x_adv_new
    
    def single_restart(self, data, device):

        x_orig = data.x.detach()

        noise = torch.zeros_like(x_orig).uniform_(-self.epsilon, self.epsilon)
        x_adv = self.project(x_orig=x_orig, x_adv=x_orig + noise, device=device)

        for _ in range(self.steps):
            x_adv = self.pgd_step(data=data, x_adv=x_adv)
        
        with torch.no_grad():
            perturbed = data.clone()
            perturbed.x = x_adv
            logits = self.model(perturbed)
            final_loss = F.cross_entropy(logits, data.y.view(-1).to(device)).item()
        
        return x_adv, final_loss
    
    def attack(self, data):

        device = next(self.model.parameters()).device
        data = self.prepare(data=data, device=device)

        saved_grad = {n: p.requires_grad for n, p in self.model.named_parameters()}
        for p in self.model.parameters():
            p.requires_grad_(False)

        best_loss = -float("inf")
        best_x = None

        try:
            for _ in range(self.num_restarts):
                x_adv, loss = self.single_restart(data=data, device=device)
                if loss > best_loss:
                    best_loss = loss
                    best_x = x_adv
        finally:
            for n, p in self.model.named_parameters():
                p.requires_grad_(saved_grad[n])
        
        result = data.clone()
        result.x = best_x
        return result