"""§5 — VLM training on CLEVR.

Usage:
    uv run python scripts/train_vlm.py --config configs/vlm_clevr.yaml \\
        --injection all_patches --mask-mode image_bidir \\
        --freeze-config A
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--pretrained-vit", type=Path, required=True,
                   help="Path to CLIP-pretrained ViT checkpoint from §3")
    p.add_argument(
        "--injection",
        choices=["cls", "all_patches", "interleaved"],
        default="all_patches",
    )
    p.add_argument(
        "--mask-mode",
        choices=["causal", "image_bidir"],
        default="causal",
    )
    p.add_argument(
        "--freeze-config",
        choices=["A", "B", "C", "D"],
        default="A",
        help="Per writeup §5.6: A=projector only, B=+decoder LoRA, "
             "C=+full decoder, D=all three.",
    )
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.output_dir is None:
        args.output_dir = (
            Path("runs") / f"vlm_{args.injection}_{args.mask_mode}_{args.freeze_config}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # TODO: students fill in.
    # Sketch:
    #   1. Build CLEVR loaders via vlm.data.build_clevr_loaders.
    #   2. Load CLIP-pretrained ViT (args.pretrained_vit).
    #   3. Load SmolLM2-360M-Instruct decoder + tokenizer in bf16 with FA2.
    #      - Add the special <image> token to the tokenizer if injection ==
    #        "interleaved", and resize_token_embeddings on the decoder.
    #   4. Build VisionLanguageProjector and VisionLanguageModel.
    #   5. Apply the chosen freeze configuration:
    #        A: vit frozen, projector trained, decoder frozen.
    #        B: vit frozen, projector trained, decoder LoRA.
    #        C: vit frozen, projector trained, decoder full FT.
    #        D: everything full FT.
    #   6. Train for cfg["num_steps"] with bf16 gradient accumulation.
    #   7. Periodically run vlm.eval.batch_clevr_accuracy on the val set,
    #      log: val accuracy, peak memory, train loss, gradient norm.
    #   8. Save best checkpoint.
    raise NotImplementedError(
        "Implement the VLM training loop in scripts/train_vlm.py."
    )


if __name__ == "__main__":
    main()
