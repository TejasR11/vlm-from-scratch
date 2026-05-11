"""Tests for §4 — LoRA."""

from __future__ import annotations

import torch

from tests import adapters


def test_lora_linear_zero_init_recovers_base():
    """With B initialized to zero, LoRALinear(x) should equal base_layer(x)."""
    d_in, d_out = 16, 32
    rank, alpha = 4, 8.0
    base_w = torch.randn(d_out, d_in)
    A = torch.randn(rank, d_in)
    B = torch.zeros(d_out, rank)  # zero init
    x = torch.randn(2, d_in)
    out = adapters.run_lora_linear(base_w, rank, alpha, A, B, x)
    expected = x @ base_w.T
    assert torch.allclose(out, expected, atol=1e-5), (
        "With B=0, LoRALinear should match the base linear layer exactly."
    )


def test_lora_linear_update_formula():
    """Check the math: out = x @ W^T + (alpha/rank) * x @ A^T @ B^T."""
    d_in, d_out = 8, 16
    rank, alpha = 4, 8.0
    base_w = torch.randn(d_out, d_in)
    A = torch.randn(rank, d_in)
    B = torch.randn(d_out, rank)
    x = torch.randn(3, d_in)
    out = adapters.run_lora_linear(base_w, rank, alpha, A, B, x)
    expected = x @ base_w.T + (alpha / rank) * (x @ A.T) @ B.T
    assert torch.allclose(out, expected, atol=1e-4), (
        f"LoRA update formula mismatch. Max diff: {(out - expected).abs().max()}"
    )


def test_apply_lora_freezes_base_params():
    """After apply_lora_to_attention, only A and B should require grad."""
    config = dict(
        img_size=32, patch_size=8, d_model=64,
        num_heads=4, num_blocks=2, dropout=0.0,
    )
    info = adapters.run_apply_lora(config, rank=4, alpha=8.0)
    assert info["trainable_params"] < info["total_params"]
    assert info["trainable_params"] > 0
    # Trainable should be a small fraction.
    ratio = info["trainable_params"] / info["total_params"]
    assert ratio < 0.2, f"LoRA should make <20% of params trainable, got {ratio:.2%}"


def test_apply_lora_targets_q_and_v():
    """LoRA should be applied to q_proj and v_proj of every attention head,
    and not to k_proj."""
    config = dict(
        img_size=32, patch_size=8, d_model=64,
        num_heads=4, num_blocks=2, dropout=0.0,
    )
    info = adapters.run_apply_lora(config, rank=4, alpha=8.0)
    names = info["lora_param_names"]
    assert any("q_proj" in n for n in names), "Expected q_proj LoRA params"
    assert any("v_proj" in n for n in names), "Expected v_proj LoRA params"
    assert not any("k_proj" in n for n in names), "k_proj should NOT have LoRA"
