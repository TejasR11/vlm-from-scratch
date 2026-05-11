"""§4 — Compare full FT, LoRA, and linear probe on RESISC45.

Usage:
    uv run python scripts/finetune_resisc.py --config configs/lora_resisc.yaml \\
        --method lora --rank 8 --pretrained runs/clip_eurosat/best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--method", choices=["linear_probe", "lora", "full_ft"], required=True)
    p.add_argument("--rank", type=int, default=8, help="LoRA rank (only for --method lora)")
    p.add_argument("--alpha", type=float, default=16.0, help="LoRA alpha (only for --method lora)")
    p.add_argument("--pretrained", type=Path, required=True,
                   help="Path to CLIP-pretrained ViT checkpoint from §3")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.output_dir is None:
        args.output_dir = Path("runs") / f"resisc_{args.method}_rank{args.rank}"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # TODO: students fill in.
    # Sketch:
    #   1. Build RESISC45 loaders.
    #   2. Load the CLIP-pretrained ViT.
    #   3. Apply the chosen adaptation:
    #        - linear_probe: freeze ViT, attach 45-way head.
    #        - lora: apply_lora_to_attention(vit, rank, alpha), attach head.
    #        - full_ft: leave everything trainable, attach head.
    #   4. Train for cfg["num_epochs"], track:
    #        - test accuracy
    #        - num trainable params
    #        - peak memory: torch.cuda.max_memory_allocated()
    #        - wall clock
    #   5. Save the metrics dict to args.output_dir / "metrics.json".
    raise NotImplementedError(
        "Implement the RESISC45 fine-tuning loop in scripts/finetune_resisc.py."
    )


if __name__ == "__main__":
    main()
