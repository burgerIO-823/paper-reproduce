"""
Transformer Encoder Block

Standard transformer encoder layer with multi-head self-attention and
feed-forward network. From "Attention Is All You Need" (Vaswani et al., 2017).

Architecture:
    input → LayerNorm → MultiHeadAttention → + residual
          → LayerNorm → FFN (Linear→GELU→Linear) → + residual → output
"""

import torch
import torch.nn as nn


class TransformerEncoderBlock(nn.Module):
    """
    A single transformer encoder block.

    Args:
        hidden_size: Dimension of the hidden representations
        num_heads: Number of attention heads
        ffn_size: Dimension of the feed-forward intermediate layer
                  (default: 4 * hidden_size)
        dropout: Dropout probability applied after attention and FFN
        activation: Activation function for FFN ('gelu' or 'relu')
    """

    def __init__(self, hidden_size: int, num_heads: int = 8,
                 ffn_size: int = None, dropout: float = 0.1,
                 activation: str = "gelu"):
        super().__init__()
        ffn_size = ffn_size or 4 * hidden_size

        self.self_attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_size),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_size, hidden_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, L, hidden_size)
            attention_mask: Optional attention mask of shape (L, L) or (B, L, L)
        """
        # Self-attention with residual
        attn_out, _ = self.self_attn(x, x, x, attn_mask=attention_mask)
        x = self.norm1(x + self.dropout(attn_out))

        # FFN with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))

        return x
