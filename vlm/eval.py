"""Provided evaluation utilities.

DO NOT MODIFY THIS FILE.
"""

from __future__ import annotations

import re

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# CLIP zero-shot accuracy — §3
# ---------------------------------------------------------------------------


@torch.no_grad()
def zeroshot_classification_accuracy(
    vit: torch.nn.Module,
    projection_heads: torch.nn.Module,
    text_encoder: torch.nn.Module,
    val_loader,
    class_prompts: list[str],
    class_indices: list[int],
    device: torch.device,
) -> float:
    """Standard CLIP zero-shot accuracy.

    Args:
        vit:               CLIP-trained image encoder. Returns (B, d_image).
        projection_heads:  ProjectionHeads instance (image_proj, text_proj, L2-norm).
        text_encoder:      FrozenTextEncoder.
        val_loader:        Yields (image_batch, list_of_captions). Captions are
                           ignored here — we only need the images and the true
                           class index, which we recover by matching the
                           caption to `class_prompts`.
        class_prompts:     One prompt per class (same template as training).
        class_indices:     Integer class IDs aligned with class_prompts.

    Returns:
        Accuracy as a float in [0, 1].
    """
    # Encode class prompts once.
    text_embeds = text_encoder(class_prompts)
    _, class_proj = projection_heads(
        torch.zeros(len(class_prompts), vit.d_model if hasattr(vit, "d_model") else 0,
                    device=text_embeds.device),
        text_embeds,
    )
    class_proj = F.normalize(class_proj, dim=-1)

    correct = 0
    total = 0
    vit_was_training = vit.training
    projection_heads_was_training = projection_heads.training
    vit.eval()
    projection_heads.eval()
    for images, captions in val_loader:
        images = images.to(device)
        # Recover true labels from captions.
        labels = torch.tensor(
            [class_prompts.index(c) for c in captions], device=device
        )
        feats = vit(images)
        img_proj, _ = projection_heads(feats, torch.zeros_like(text_embeds[:1]))
        img_proj = F.normalize(img_proj, dim=-1)
        sims = img_proj @ class_proj.T
        preds = sims.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    accuracy = correct / max(total, 1)
    vit.train(vit_was_training)
    projection_heads.train(projection_heads_was_training)
    return accuracy


# ---------------------------------------------------------------------------
# CLEVR exact-match grading — §5
# ---------------------------------------------------------------------------


def _normalize_clevr_answer(s: str) -> str:
    s = s.strip().lower()
    # Strip surrounding punctuation.
    s = re.sub(r"^[\s\.\!\?\"\']+|[\s\.\!\?\"\']+$", "", s)
    # Numeric word -> digit.
    word_to_digit = {
        "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
        "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    }
    if s in word_to_digit:
        return word_to_digit[s]
    return s


def clevr_exact_match(prediction: str, gold: str) -> bool:
    """CLEVR-friendly exact-match grader.

    Normalizes whitespace, case, surrounding punctuation, and numeric words
    so that 'three' matches '3'. Strict otherwise.
    """
    return _normalize_clevr_answer(prediction) == _normalize_clevr_answer(gold)


def batch_clevr_accuracy(
    predictions: list[str],
    golds: list[str],
    q_types: list[str] | None = None,
) -> dict[str, float]:
    """Returns overall accuracy and (optionally) per-q_type accuracy."""
    overall = sum(clevr_exact_match(p, g) for p, g in zip(predictions, golds)) / max(
        len(golds), 1
    )
    out = {"overall": overall}
    if q_types is not None:
        for qt in set(q_types):
            mask = [t == qt for t in q_types]
            if any(mask):
                out[qt] = sum(
                    clevr_exact_match(p, g)
                    for p, g, m in zip(predictions, golds, mask) if m
                ) / sum(mask)
    return out
