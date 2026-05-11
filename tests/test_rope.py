"""Tests for §6 — RoPE."""

from __future__ import annotations

import torch

from tests import adapters


def test_rope_1d_shape():
    head_dim, max_seq = 64, 32
    B, H, T = 2, 4, 16
    x = torch.randn(B, H, T, head_dim)
    positions = torch.arange(T)
    out = adapters.run_rope_1d(head_dim, max_seq, 10_000.0, x, positions)
    assert out.shape == x.shape


def test_rope_1d_preserves_norm():
    """RoPE is a rotation, so it must preserve the L2 norm of every vector."""
    head_dim, max_seq = 64, 32
    B, H, T = 2, 4, 16
    x = torch.randn(B, H, T, head_dim)
    positions = torch.arange(T)
    out = adapters.run_rope_1d(head_dim, max_seq, 10_000.0, x, positions)
    in_norms = x.norm(dim=-1)
    out_norms = out.norm(dim=-1)
    assert torch.allclose(in_norms, out_norms, atol=1e-4), (
        f"RoPE should preserve norms. Max diff: {(in_norms - out_norms).abs().max()}"
    )


def test_rope_1d_at_zero_is_identity():
    """At position 0, all rotation angles are 0, so RoPE should be the
    identity."""
    head_dim, max_seq = 32, 16
    x = torch.randn(1, 2, 4, head_dim)
    positions = torch.zeros(4, dtype=torch.long)
    out = adapters.run_rope_1d(head_dim, max_seq, 10_000.0, x, positions)
    assert torch.allclose(out, x, atol=1e-5)


def test_rope_1d_relative_property():
    """Dot products of two RoPE'd vectors at offsets m1, m2 should depend
    only on the difference m1 - m2 (this is the whole point of RoPE)."""
    head_dim, max_seq = 32, 32
    q = torch.randn(1, 1, 1, head_dim)
    k = torch.randn(1, 1, 1, head_dim)

    def dot_at(p1: int, p2: int) -> float:
        q_ = adapters.run_rope_1d(head_dim, max_seq, 10_000.0, q, torch.tensor([p1]))
        k_ = adapters.run_rope_1d(head_dim, max_seq, 10_000.0, k, torch.tensor([p2]))
        return (q_ * k_).sum().item()

    # Offset = 5: positions (3, 8) and (10, 15) should produce the same dot.
    d1 = dot_at(3, 8)
    d2 = dot_at(10, 15)
    assert abs(d1 - d2) < 1e-3, (
        f"RoPE relative-position property failed: {d1:.4f} vs {d2:.4f}"
    )


def test_rope_2d_shape():
    head_dim, grid = 64, 8
    B, H, T = 1, 2, 16
    x = torch.randn(B, H, T, head_dim)
    x_coords = torch.randint(0, grid, (T,))
    y_coords = torch.randint(0, grid, (T,))
    out = adapters.run_rope_2d(head_dim, grid, 10_000.0, x, x_coords, y_coords)
    assert out.shape == x.shape


def test_rope_2d_preserves_norm():
    head_dim, grid = 64, 8
    x = torch.randn(1, 2, 16, head_dim)
    x_coords = torch.randint(0, grid, (16,))
    y_coords = torch.randint(0, grid, (16,))
    out = adapters.run_rope_2d(head_dim, grid, 10_000.0, x, x_coords, y_coords)
    assert torch.allclose(x.norm(dim=-1), out.norm(dim=-1), atol=1e-4)
