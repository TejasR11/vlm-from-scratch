"""§5 — Qualitative evaluation of a trained VLM.

Generates predictions on a held-out CLEVR sample and reports per-q_type
accuracy. Useful for both Problem (vlm_qualitative) and Problem (mrope_impl).

Usage:
    uv run python scripts/eval_vlm.py \\
        --checkpoint runs/vlm_all_patches_image_bidir_A/best.pt \\
        --num-examples 10 --save-images
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--num-examples", type=int, default=10,
                   help="Number of examples to dump for qualitative inspection")
    p.add_argument("--max-eval", type=int, default=500,
                   help="Number of examples to use for accuracy computation")
    p.add_argument("--save-images", action="store_true",
                   help="Save the example images alongside the JSON output")
    p.add_argument("--output-dir", type=Path, default=Path("runs/vlm_qualitative"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # TODO: students fill in.
    # Sketch:
    #   1. Load the checkpoint and reconstruct the VLM.
    #   2. Run model.generate on args.max_eval val examples.
    #   3. Compute vlm.eval.batch_clevr_accuracy with q_types -> per-type breakdown.
    #   4. Sample args.num_examples for qualitative dump:
    #        For each:
    #          - Save image (if --save-images).
    #          - Append {image_file, question, gold, prediction, correct} to
    #            args.output_dir / "examples.jsonl".
    #   5. Print summary table.
    raise NotImplementedError(
        "Implement VLM evaluation in scripts/eval_vlm.py."
    )


if __name__ == "__main__":
    main()
