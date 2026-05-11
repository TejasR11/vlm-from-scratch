"""Tests for §3 — CLIP loss."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from tests import adapters


def test_clip_loss_minimum_when_aligned():
    """If image_i and text_i are identical and all other pairs orthogonal,
    the loss should be near zero (minimum)."""
    B, d = 8, 16
    embeds = F.normalize(torch.randn(B, d), dim=-1)
    logit_scale = torch.tensor(math.log(1 / 0.07))
    loss = adapters.run_clip_loss(embeds, embeds, logit_scale)
    assert loss.item() < 0.1, f"Expected near-zero loss, got {loss.item()}"


def test_clip_loss_symmetric():
    """clip_loss should be symmetric in (image_embeds, text_embeds)."""
    B, d = 8, 16
    img = F.normalize(torch.randn(B, d), dim=-1)
    txt = F.normalize(torch.randn(B, d), dim=-1)
    logit_scale = torch.tensor(math.log(1 / 0.07))
    loss_a = adapters.run_clip_loss(img, txt, logit_scale)
    loss_b = adapters.run_clip_loss(txt, img, logit_scale)
    assert torch.allclose(loss_a, loss_b, atol=1e-5)


def test_clip_loss_decreases_with_higher_temperature_when_aligned():
    """When image and text are aligned, increasing logit_scale (higher
    inverse temperature) should DECREASE the loss further."""
    B, d = 8, 16
    embeds = F.normalize(torch.randn(B, d), dim=-1)
    low = torch.tensor(math.log(1.0))
    high = torch.tensor(math.log(50.0))
    loss_low = adapters.run_clip_loss(embeds, embeds, low)
    loss_high = adapters.run_clip_loss(embeds, embeds, high)
    assert loss_high.item() <= loss_low.item() + 1e-4


def test_clip_loss_returns_scalar():
    B, d = 4, 8
    img = F.normalize(torch.randn(B, d), dim=-1)
    txt = F.normalize(torch.randn(B, d), dim=-1)
    loss = adapters.run_clip_loss(img, txt, torch.tensor(math.log(1 / 0.07)))
    assert loss.dim() == 0
