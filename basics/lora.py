"""LoRA adapters — §4.

You implement: LoRALinear, apply_lora_to_attention.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Low-rank adapter wrapping an existing nn.Linear layer.

    Computes:  W' x = base_layer(x) + (alpha / rank) * (B A x)

    where:
      - base_layer is the frozen pretrained linear (its weights are not trained).
      - A in R^{rank x d_in}  is initialized with kaiming_uniform_.
      - B in R^{d_out x rank} is initialized to zero (so the adapted layer
        starts equal to the base layer).

    Only A and B receive gradients; base_layer's parameters are frozen.

    Args:
        base_layer: Existing nn.Linear to wrap.
        rank:       Adapter rank `r` (typically 4..32).
        alpha:      Scaling factor; effective scale is `alpha / rank`.
    """

    def __init__(self, base_layer: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.base_layer = base_layer

        # TODO: freeze base_layer's parameters.
        # TODO: create self.A (nn.Parameter, shape (rank, d_in), kaiming-uniform init).
        # TODO: create self.B (nn.Parameter, shape (d_out, rank), zero init).
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: return base_layer(x) + scaling * (x @ A.T @ B.T)
        raise NotImplementedError


def apply_lora_to_attention(model: nn.Module, rank: int, alpha: float) -> nn.Module:
    """Replace `q_proj` and `v_proj` linear layers in every attention head
    with LoRA-wrapped versions.

    The HW writeup recommends adapting only Q and V projections (per the
    original LoRA paper). Walk the module tree and wherever you find an
    nn.Linear named `q_proj` or `v_proj` inside a Head, swap it for a
    LoRALinear.

    The function modifies `model` in place AND returns it for convenience.

    Args:
        model: A module containing one or more `basics.model.Head` instances
               (e.g., a ViT).
        rank, alpha: Forwarded to LoRALinear.
    """
    # TODO: implement.
    # Hint: iterate model.named_modules(), check isinstance(m, Head), and
    # set m.q_proj = LoRALinear(m.q_proj, rank, alpha) (and same for v_proj).
    raise NotImplementedError
