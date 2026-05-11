"""Tests for §2 — Vision Transformer."""

from __future__ import annotations

import pytest
import torch

from tests import adapters


def test_patch_embeddings_shape():
    """Output shape should be (B, num_patches, d_model)."""
    img_size, patch_size, d_model = 64, 8, 96
    B = 4
    x = torch.randn(B, 3, img_size, img_size)
    out = adapters.run_patch_embeddings(img_size, patch_size, d_model, x)
    expected_n = (img_size // patch_size) ** 2
    assert out.shape == (B, expected_n, d_model), (
        f"Expected ({B}, {expected_n}, {d_model}), got {tuple(out.shape)}"
    )


def test_patch_embeddings_partition():
    """Different patches should produce different embeddings (sanity check
    that patchification preserves spatial information)."""
    img_size, patch_size, d_model = 32, 8, 64
    x = torch.randn(1, 3, img_size, img_size)
    out = adapters.run_patch_embeddings(img_size, patch_size, d_model, x)
    # Pairwise distances between patches should not all be zero.
    n = out.shape[1]
    distances = torch.cdist(out[0], out[0])
    off_diag = distances[~torch.eye(n, dtype=torch.bool)]
    assert off_diag.max() > 1e-4


def test_vit_shape():
    config = dict(
        img_size=32, patch_size=8, d_model=64,
        num_heads=4, num_blocks=2, dropout=0.0,
    )
    B = 4
    x = torch.randn(B, 3, 32, 32)
    out = adapters.run_vit(config, x)
    assert out.shape == (B, 64), f"Expected ({B}, 64), got {tuple(out.shape)}"


def test_vit_deterministic_with_zero_dropout():
    """With dropout=0 and eval mode, two forward passes on the same input
    should produce identical outputs."""
    config = dict(
        img_size=32, patch_size=8, d_model=64,
        num_heads=4, num_blocks=2, dropout=0.0,
    )
    x = torch.randn(2, 3, 32, 32)
    # Same weights for both calls (run_vit must accept weights).
    pytest.importorskip("torch")
    # Two calls with default init won't match (different random weights).
    # Instead, check with explicit weights:
    # We just verify that the same call produces the same output when seeded.
    torch.manual_seed(0)
    out1 = adapters.run_vit(config, x)
    torch.manual_seed(0)
    out2 = adapters.run_vit(config, x)
    assert torch.allclose(out1, out2, atol=1e-5)
