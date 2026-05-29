"""
Focal Loss

From "Focal Loss for Dense Object Detection" (Lin et al., 2017).
Addresses class imbalance by down-weighting well-classified examples.

Formula: FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.

    Args:
        alpha: Class weight factor (scalar or tensor of shape [num_classes]).
               Default: 0.25 (as used in RetinaNet).
        gamma: Focusing parameter. Higher values down-weight easy examples more.
               Default: 2.0.
        reduction: 'mean', 'sum', or 'none'.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0,
                 reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Logits of shape (B, num_classes)
            targets: Class indices of shape (B,)
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss
