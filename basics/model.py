"""Provided Transformer building blocks.

These classes are shared between the ViT (encoder) and the language decoder
used in §5. The `is_decoder` flag on Head / MultiHeadAttention / Block toggles
between bidirectional and causal attention.

DO NOT MODIFY THIS FILE.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
    """A single self-attention head.

    If `is_decoder=True`, applies a causal mask so that token i can only
    attend to tokens <= i. If `is_decoder=False`, attention is bidirectional
    (used inside the ViT and inside image blocks of the VLM).
    """

    def __init__(
        self,
        d_model: int,
        head_dim: int,
        block_size: int,
        is_decoder: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.is_decoder = is_decoder

        self.q_proj = nn.Linear(d_model, head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, head_dim, bias=False)

        if is_decoder:
            self.register_buffer(
                "tril",
                torch.tril(torch.ones(block_size, block_size, dtype=torch.bool)),
                persistent=False,
            )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        B, T, _ = x.shape
        q = self.q_proj(x)  # (B, T, head_dim)
        k = self.k_proj(x)
        v = self.v_proj(x)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, T, T)
        if self.is_decoder:
            attn = attn.masked_fill(~self.tril[:T, :T], float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        return attn @ v  # (B, T, head_dim)


class MultiHeadAttention(nn.Module):
    """Standard multi-head self-attention by concatenating Head outputs."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        block_size: int,
        is_decoder: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        head_dim = d_model // num_heads
        self.heads = nn.ModuleList(
            [
                Head(d_model, head_dim, block_size, is_decoder=is_decoder, dropout=dropout)
                for _ in range(num_heads)
            ]
        )
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # (B, T, d_model)
        return self.dropout(self.out_proj(out))


class MLP(nn.Module):
    """Two-layer MLP with GELU, used inside each Block."""

    def __init__(self, d_model: int, d_ff: int | None = None, dropout: float = 0.0) -> None:
        super().__init__()
        d_ff = d_ff if d_ff is not None else 4 * d_model
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


class Block(nn.Module):
    """Pre-LayerNorm Transformer block.

    Used both for the ViT (is_decoder=False) and the language decoder
    (is_decoder=True).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        block_size: int,
        is_decoder: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            block_size=block_size,
            is_decoder=is_decoder,
            dropout=dropout,
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model=d_model, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
