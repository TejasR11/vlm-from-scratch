"""§3 — CLIP-style pretraining on EuroSAT.

You implement the training loop. This script provides the CLI scaffolding,
config loading, and logging hooks.

Usage:
    uv run python scripts/pretrain_clip.py --config configs/clip_eurosat.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("runs/clip_eurosat"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--wandb", action="store_true", help="Log to W&B")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # TODO: students fill in the training loop.
    # Sketch:
    #   1. Build train/val/test loaders via vlm.data.build_eurosat_loaders.
    #   2. Build the ViT (basics.vit.ViT) and FrozenTextEncoder.
    #   3. Build ProjectionHeads + logit_scale.
    #   4. AdamW optimizer, cosine LR schedule.
    #   5. For each epoch:
    #         - Train one epoch with vlm.clip.clip_loss.
    #         - Clamp logit_scale.data to <= ln(100).
    #         - Compute zero-shot val accuracy via vlm.eval.zeroshot_classification_accuracy.
    #         - Log to stdout (and W&B if args.wandb).
    #   6. Save the best checkpoint to args.output_dir / "best.pt".
    raise NotImplementedError(
        "Implement the CLIP pretraining loop in scripts/pretrain_clip.py."
    )


if __name__ == "__main__":
    main()
