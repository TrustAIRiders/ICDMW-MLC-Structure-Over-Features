import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        inputs = torch.clamp(inputs, min=-30.0, max=30.0)

        ce_loss = F.cross_entropy(
            inputs, targets, reduction='none', weight=self.alpha,
            label_smoothing=self.label_smoothing,
        )
        ce_loss = torch.clamp(ce_loss, min=0.0, max=50.0)
        p_t = torch.exp(-ce_loss)
        focal_loss = (1 - p_t) ** self.gamma * ce_loss

        focal_loss = torch.nan_to_num(focal_loss, nan=0.0)

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
