"""Provided attention-mask utilities for §5.

For Problem (masking), you need a way to construct an attention mask that is
*bidirectional inside the image block* and *causal everywhere else*. HF causal
LMs accept a custom 4D additive mask of shape (B, 1, T, T) where 0 means
"attend" and a large negative value (e.g., -inf or torch.finfo(dtype).min)
means "don't attend".

DO NOT MODIFY THIS FILE.
"""

from __future__ import annotations

import torch


def build_causal_mask(seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Standard causal mask. Shape (1, 1, T, T)."""
    mask = torch.full((seq_len, seq_len), torch.finfo(dtype).min, device=device, dtype=dtype)
    mask = torch.triu(mask, diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)


def build_image_bidir_mask(
    n_visual: int,
    n_text: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Mask that is bidirectional inside the visual block, causal across the
    image-to-text boundary, and causal among text tokens.

    Layout of the sequence: [v_1, ..., v_{n_visual}, t_1, ..., t_{n_text}].

    Allowed attention:
      - Visual i can attend to all visual j (bidirectional inside image block).
      - Visual i CANNOT attend to text (text tokens haven't been "seen" yet).
      - Text i can attend to all visual tokens AND to text tokens j <= i.

    Returns: (1, 1, T, T) additive mask, T = n_visual + n_text.
    """
    T = n_visual + n_text
    # Start "all allowed".
    mask = torch.zeros((T, T), device=device, dtype=dtype)
    # Disallow visual -> text.
    mask[:n_visual, n_visual:] = torch.finfo(dtype).min
    # Disallow text -> future text (upper-triangular within the text block).
    if n_text > 0:
        text_block = torch.full(
            (n_text, n_text), torch.finfo(dtype).min, device=device, dtype=dtype
        )
        text_block = torch.triu(text_block, diagonal=1)
        mask[n_visual:, n_visual:] = text_block
    return mask.unsqueeze(0).unsqueeze(0)
