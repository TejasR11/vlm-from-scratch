"""CLIP-style contrastive learning — §3.

You implement: clip_loss, ProjectionHeads.

The frozen text encoder is provided in `basics.text_encoder.FrozenTextEncoder`.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class ProjectionHeads(nn.Module):
    """Two unbiased linear heads that project image and text embeddings into
    a shared d_proj-dimensional space, followed by L2 normalization.

    Args:
        d_image: Dim of image embeddings (your ViT's d_model).
        d_text:  Dim of text embeddings (FrozenTextEncoder.embedding_dim).
        d_proj:  Shared projection dim (256 in the writeup).

    Forward:
        image_embeds: (B, d_image) — typically the ViT's CLS embedding.
        text_embeds:  (B, d_text)  — from FrozenTextEncoder(captions).
        returns:      tuple (image_proj, text_proj), each (B, d_proj),
                      both L2-normalized along the last dim.
    """

    def __init__(self, d_image: int, d_text: int, d_proj: int = 256) -> None:
        super().__init__()
        self.image_proj = nn.Linear(d_image, d_proj, bias=False)
        self.text_proj = nn.Linear(d_text, d_proj, bias=False)


    def forward(
        self, image_embeds: torch.Tensor, text_embeds: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image_proj = nn.functional.normalize(self.image_proj(image_embeds), p=2, dim=-1)
        text_proj = nn.functional.normalize(self.text_proj(text_embeds), p=2, dim=-1)
        return image_proj, text_proj


def init_logit_scale() -> nn.Parameter:
    """CLIP-style learnable temperature, initialized to ln(1/0.07)."""
    return nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))


def clip_loss(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    logit_scale: torch.Tensor,
) -> torch.Tensor:
    """Symmetric InfoNCE loss.

    Computes
        L = 0.5 * ( CE(S, y) + CE(S^T, y) )
    where
        S = image_embeds @ text_embeds.T * exp(logit_scale)
        y = arange(B).

    `logit_scale` should be clamped to a maximum of ln(100) to prevent runaway
    growth (do this OUTSIDE this function, e.g. in your training loop, with
    `logit_scale.data.clamp_(max=math.log(100.0))`).

    Args:
        image_embeds: (B, d), L2-normalized.
        text_embeds:  (B, d), L2-normalized.
        logit_scale:  Scalar tensor (learnable).

    Returns:
        Scalar loss tensor.
    """
    logits = image_embeds @ text_embeds.T * logit_scale.exp()
    targets = torch.arange(image_embeds.shape[0], device=image_embeds.device)

    image_loss = nn.functional.cross_entropy(logits, targets)
    text_loss = nn.functional.cross_entropy(logits.T, targets)
    return 0.5 * (image_loss + text_loss)
