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
import re
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from basics import vit
from basics.lora import LoRALinear
from vlm import data
from vlm.eval import batch_clevr_accuracy
from vlm.model import VisionLanguageModel
from vlm.projector import VisionLanguageProjector


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


def make_prompts(questions: list[str], injection: str) -> list[str]:
    prefix = "<image>\n" if injection == "interleaved" else ""
    return [f"{prefix}Question: {q}\nAnswer:" for q in questions]


def clean_prediction(text: str) -> str:
    text = text.strip().lower()
    if "answer:" in text:
        text = text.split("answer:")[-1].strip()
    lines = text.splitlines()
    if not lines:
        return ""
    text = lines[0].strip().strip(".")
    answer_vocab = [
        "yes", "no",
        "gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow",
        "cube", "sphere", "cylinder",
        "rubber", "metal",
        "small", "large",
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
        "zero", "one", "two", "three", "four", "five",
        "six", "seven", "eight", "nine", "ten",
    ]
    tokens = re.findall(r"[a-z]+|\d+", text)
    for token in tokens:
        if token in answer_vocab:
            return token
    return text


def apply_lora_to_decoder_qv(decoder: torch.nn.Module, rank: int = 8, alpha: float = 16.0) -> None:
    for module in decoder.modules():
        if hasattr(module, "q_proj") and isinstance(module.q_proj, torch.nn.Linear):
            module.q_proj = LoRALinear(module.q_proj, rank, alpha)
        if hasattr(module, "v_proj") and isinstance(module.v_proj, torch.nn.Linear):
            module.v_proj = LoRALinear(module.v_proj, rank, alpha)


def build_loader(split: str, img_size: int, batch_size: int = 32):
    ds = data.CLEVRMiniDataset(split=split, img_size=img_size)

    def collate(batch):
        return {
            "image": torch.stack([b["image"] for b in batch]),
            "question": [b["question"] for b in batch],
            "answer": [b["answer"] for b in batch],
            "q_type": [b["q_type"] for b in batch],
        }

    return DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate)


def denorm_image(tensor: torch.Tensor) -> Image.Image:
    mean = torch.tensor(data.IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(data.IMAGENET_STD).view(3, 1, 1)
    image = (tensor.cpu() * std + mean).clamp(0, 1)
    image = (image.permute(1, 2, 0).numpy() * 255).astype("uint8")
    return Image.fromarray(image)


def reconstruct_model(checkpoint: dict, device: torch.device) -> VisionLanguageModel:
    cfg = checkpoint["config"]
    vit_cfg = checkpoint["vit_config"]
    injection = checkpoint["injection"]
    freeze_config = checkpoint.get("freeze_config", "A")

    vit_ckpt = torch.load(checkpoint["pretrained_vit"], map_location="cpu")
    image_encoder = vit.ViT(**vit_cfg)
    image_encoder.load_state_dict(vit_ckpt["vit"] if "vit" in vit_ckpt else vit_ckpt)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint["decoder_model_name"])
    added_tokens = 0
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            added_tokens += tokenizer.add_special_tokens({"pad_token": "<pad>"})
    image_token_id = None
    if injection == "interleaved":
        added_tokens += tokenizer.add_special_tokens(
            {"additional_special_tokens": ["<image>"]}
        )
        image_token_id = tokenizer.convert_tokens_to_ids("<image>")

    dtype = torch.bfloat16 if cfg["decoder"].get("torch_dtype") == "bfloat16" else torch.float32
    decoder = AutoModelForCausalLM.from_pretrained(
        checkpoint["decoder_model_name"],
        torch_dtype=dtype,
    )
    if added_tokens > 0:
        decoder.resize_token_embeddings(len(tokenizer))
    decoder.config.use_cache = False

    if freeze_config == "B":
        apply_lora_to_decoder_qv(decoder, rank=8, alpha=16.0)

    d_decoder = decoder.get_input_embeddings().embedding_dim
    projector = VisionLanguageProjector(
        d_image=vit_cfg["d_model"],
        d_decoder=d_decoder,
        expansion=cfg["projector"].get("expansion", 4),
    )
    model = VisionLanguageModel(
        image_encoder,
        projector,
        decoder,
        tokenizer,
        image_token_id=image_token_id,
    )
    if "trainable_state" in checkpoint:
        model.load_state_dict(checkpoint["trainable_state"], strict=False)
    elif "projector" in checkpoint:
        model.projector.load_state_dict(checkpoint["projector"])
    model.to(device)
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = reconstruct_model(checkpoint, device)
    tokenizer = model.tokenizer
    cfg = checkpoint["config"]
    injection = checkpoint["injection"]

    loader = build_loader(
        args.split,
        img_size=checkpoint["vit_config"]["img_size"],
        batch_size=cfg["train"]["batch_size"],
    )

    gen_kwargs = {
        "do_sample": cfg["generation"].get("do_sample", False),
        "temperature": cfg["generation"].get("temperature", 1.0),
        "top_p": cfg["generation"].get("top_p", 1.0),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    records = []
    predictions: list[str] = []
    golds: list[str] = []
    q_types: list[str] = []

    with torch.no_grad():
        for batch in loader:
            remaining = args.max_eval - len(golds)
            if remaining <= 0:
                break
            images = batch["image"][:remaining].to(device, non_blocking=True)
            questions = batch["question"][:remaining]
            answers = batch["answer"][:remaining]
            batch_q_types = batch["q_type"][:remaining]
            prompts = make_prompts(questions, injection)
            raw_outputs = model.generate(
                images,
                prompts,
                injection=injection,
                max_new_tokens=cfg["generation"]["max_new_tokens"],
                **gen_kwargs,
            )
            cleaned = [clean_prediction(text) for text in raw_outputs]
            predictions.extend(cleaned)
            golds.extend(answers)
            q_types.extend(batch_q_types)

            for i, (question, answer, raw, pred, q_type) in enumerate(
                zip(questions, answers, raw_outputs, cleaned, batch_q_types)
            ):
                correct = batch_clevr_accuracy([pred], [answer])["overall"] == 1.0
                image_file = None
                if args.save_images:
                    image_file = f"example_{len(records):03d}.png"
                    denorm_image(batch["image"][i]).save(args.output_dir / image_file)
                records.append(
                    {
                        "image_file": image_file,
                        "question": question,
                        "gold": answer,
                        "raw_generation": raw,
                        "prediction": pred,
                        "q_type": q_type,
                        "correct": correct,
                    }
                )

    metrics = batch_clevr_accuracy(predictions, golds, q_types)

    correct_examples = [r for r in records if r["correct"]]
    incorrect_examples = [r for r in records if not r["correct"]]
    chosen = correct_examples[: args.num_examples // 2]
    chosen += incorrect_examples[: args.num_examples - len(chosen)]
    if len(chosen) < args.num_examples:
        chosen += records[: args.num_examples - len(chosen)]

    with open(args.output_dir / "examples.jsonl", "w") as f:
        for record in chosen[: args.num_examples]:
            f.write(json.dumps(record) + "\n")
    with open(args.output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"Wrote examples to {args.output_dir / 'examples.jsonl'}")


if __name__ == "__main__":
    main()
