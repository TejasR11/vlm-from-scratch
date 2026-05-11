"""Provided frozen text encoder for §3 (CLIP-style training).

Wraps a pretrained sentence-transformer or HuggingFace text model and exposes
a simple callable interface returning per-caption embeddings.

DO NOT MODIFY THIS FILE.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FrozenTextEncoder(nn.Module):
    """Frozen pretrained text encoder.

    Default backbone: sentence-transformers/all-MiniLM-L6-v2 (384-dim, fast).
    Alternative: openai/clip-vit-base-patch32 (text tower only).

    All parameters are frozen and the module is permanently in eval mode.
    Use `.embedding_dim` to get the output dimensionality at runtime.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        super().__init__()
        self.model_name = model_name

        # Lazy import so importing `basics` doesn't pull in sentence_transformers
        # for students who only work on §2/§4.
        from sentence_transformers import SentenceTransformer

        self.backbone = SentenceTransformer(model_name)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        # Probe the embedding dimension once.
        with torch.no_grad():
            sample = self.backbone.encode(["test"], convert_to_tensor=True)
        self.embedding_dim: int = sample.shape[-1]

    def train(self, mode: bool = True) -> "FrozenTextEncoder":
        # Always keep the backbone in eval mode regardless of train()/eval() calls.
        super().train(mode)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def forward(self, captions: list[str]) -> torch.Tensor:
        """Encode a list of captions into a (B, embedding_dim) float tensor."""
        return self.backbone.encode(captions, convert_to_tensor=True, show_progress_bar=False)
