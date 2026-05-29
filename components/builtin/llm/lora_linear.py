"""
LoRA Linear Layer

Low-Rank Adaptation (LoRA) wrapper for nn.Linear, from
"LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2022).

Wraps a frozen linear layer with trainable low-rank decomposition:
    output = W @ x + (B @ A) @ x * (alpha / rank)

where A ∈ R^(rank × in_features) and B ∈ R^(out_features × rank).
"""

import math
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """
    LoRA-adapted linear layer.

    Args:
        in_features: Input dimension
        out_features: Output dimension
        rank: Rank of the low-rank decomposition (default: 8)
        alpha: Scaling factor (default: 16). Output is scaled by alpha/rank.
        dropout: Dropout applied to the LoRA path (default: 0.0)
        bias: Whether the base linear layer has bias (default: True)
    """

    def __init__(self, in_features: int, out_features: int, rank: int = 8,
                 alpha: float = 16.0, dropout: float = 0.0, bias: bool = True):
        super().__init__()

        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Frozen base weights
        self.base = nn.Linear(in_features, out_features, bias=bias)
        self.base.weight.requires_grad = False
        if bias and self.base.bias is not None:
            self.base.bias.requires_grad = False

        # Trainable LoRA weights
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.lora_dropout = nn.Dropout(dropout)

        self.reset_lora_parameters()

    def reset_lora_parameters(self):
        """Initialize LoRA weights: A ~ Kaiming uniform, B = 0."""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = (
            self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
        ) * self.scaling
        return base_out + lora_out

    def merge_weights(self):
        """Merge LoRA weights into the base layer for inference (in-place)."""
        merged = self.lora_B @ self.lora_A
        self.base.weight.data += merged * self.scaling
        self.lora_A.requires_grad = False
        self.lora_B.requires_grad = False
