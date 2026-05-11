"""Adapter functions for the test suite.

For each piece of work, you bind your implementation to the corresponding
`run_*` function below. The staff tests in `tests/test_*.py` import these
adapters and call them with controlled inputs to verify your code's behavior.

You should ONLY edit the bodies of the `run_*` functions — do not change
their signatures.
"""

from __future__ import annotations

import torch


# =============================================================================
# §2 — Vision Transformer
# =============================================================================


def run_patch_embeddings(
    img_size: int,
    patch_size: int,
    d_model: int,
    images: torch.Tensor,
    weights: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Construct a PatchEmbeddings module, optionally load weights, and run a
    forward pass.

    Args:
        img_size, patch_size, d_model: Constructor args.
        images: (B, 3, img_size, img_size).
        weights: Optional state dict to load_state_dict() into the module.

    Returns:
        (B, num_patches, d_model)
    """
    from basics.vit import PatchEmbeddings

    module = PatchEmbeddings(img_size, patch_size, d_model)
    if weights is not None:
        module.load_state_dict(weights)
    return module(images)


def run_vit(
    config: dict,
    images: torch.Tensor,
    weights: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Construct a ViT, optionally load weights, and run a forward pass.

    Args:
        config: Dict with keys img_size, patch_size, d_model, num_heads,
                num_blocks, dropout.
        images: (B, 3, img_size, img_size).
        weights: Optional state dict.

    Returns:
        (B, d_model) CLS embedding.
    """
    from basics.vit import ViT

    module = ViT(**config)
    if weights is not None:
        module.load_state_dict(weights)
    module.eval()
    return module(images)


# =============================================================================
# §3 — CLIP
# =============================================================================


def run_clip_loss(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    logit_scale: torch.Tensor,
) -> torch.Tensor:
    """Run your symmetric InfoNCE loss on the given inputs."""
    from vlm.clip import clip_loss

    return clip_loss(image_embeds, text_embeds, logit_scale)


# =============================================================================
# §4 — LoRA
# =============================================================================


def run_lora_linear(
    base_weight: torch.Tensor,
    rank: int,
    alpha: float,
    A: torch.Tensor,
    B: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    """Construct a LoRALinear, set its A and B parameters, and run forward.

    Args:
        base_weight: (d_out, d_in) weight matrix for the wrapped nn.Linear.
        rank, alpha: LoRA hyperparameters.
        A: (rank, d_in) — overrides Kaiming init.
        B: (d_out, rank) — overrides zero init.
        x: (B, d_in) input.

    Returns:
        (B, d_out)
    """
    import torch.nn as nn

    from basics.lora import LoRALinear

    base_layer = nn.Linear(base_weight.shape[1], base_weight.shape[0], bias=False)
    with torch.no_grad():
        base_layer.weight.copy_(base_weight)
    module = LoRALinear(base_layer, rank, alpha)
    with torch.no_grad():
        module.A.copy_(A)
        module.B.copy_(B)
    return module(x)


def run_apply_lora(
    vit_config: dict,
    rank: int,
    alpha: float,
) -> dict:
    """Build a ViT, apply LoRA to its attention q_proj/v_proj, and return:
        {
            "total_params": int,
            "trainable_params": int,
            "lora_param_names": list[str],   # names of A and B parameters
        }
    """
    from basics.lora import apply_lora_to_attention
    from basics.vit import ViT

    model = ViT(**vit_config)
    model = apply_lora_to_attention(model, rank, alpha)
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    lora_param_names = [
        name
        for name, param in model.named_parameters()
        if param.requires_grad and name.endswith(("A", "B"))
    ]
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "lora_param_names": lora_param_names,
    }


# =============================================================================
# §6 — RoPE
# =============================================================================


def run_rope_1d(
    head_dim: int,
    max_seq_len: int,
    base: float,
    x: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """Construct a RoPE1D module and apply it to x at the given positions."""
    from basics.rope import RoPE1D

    return RoPE1D(head_dim, max_seq_len, base)(x, positions)


def run_rope_2d(
    head_dim: int,
    grid_size: int,
    base: float,
    x: torch.Tensor,
    x_coords: torch.Tensor,
    y_coords: torch.Tensor,
) -> torch.Tensor:
    """Construct a RoPE2D module and apply it."""
    from basics.rope import RoPE2D

    return RoPE2D(head_dim, grid_size, base)(x, x_coords, y_coords)
