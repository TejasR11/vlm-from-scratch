"""§5 — VLM training on CLEVR.

Usage:
    uv run python scripts/train_vlm.py --config configs/vlm_clevr.yaml \\
        --injection all_patches --mask-mode image_bidir \\
        --freeze-config A
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import yaml
from basics import vit
from basics.lora import LoRALinear
from transformers import AutoModelForCausalLM, AutoTokenizer
from vlm import data
from vlm.eval import batch_clevr_accuracy
from vlm.model import VisionLanguageModel
from vlm.projector import VisionLanguageProjector


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


def make_prompts(questions: list[str], injection: str) -> list[str]:
    prefix = "<image>\n" if injection == "interleaved" else ""
    return [f"{prefix}Question: {q}\nAnswer:" for q in questions]


def tokenize_vqa_batch(
    tokenizer,
    questions: list[str],
    answers: list[str],
    injection: str,
    device: torch.device,
    max_length: int = 128,
) -> dict[str, torch.Tensor]:
    prompts = make_prompts(questions, injection)
    eos = tokenizer.eos_token or ""
    texts = [f"{prompt} {answer}{eos}" for prompt, answer in zip(prompts, answers)]

    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    prompt_encoded = tokenizer(
        prompts,
        padding=False,
        truncation=True,
        max_length=max_length,
    )

    labels = encoded["input_ids"].clone()
    for i, prompt_ids in enumerate(prompt_encoded["input_ids"]):
        labels[i, : min(len(prompt_ids), labels.shape[1])] = -100
    labels[encoded["attention_mask"] == 0] = -100

    return {
        "input_ids": encoded["input_ids"].to(device),
        "attention_mask": encoded["attention_mask"].to(device),
        "labels": labels.to(device),
    }


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


def configure_trainable_parameters(model: VisionLanguageModel, freeze_config: str) -> list[torch.nn.Parameter]:
    for param in model.parameters():
        param.requires_grad_(False)

    for param in model.projector.parameters():
        param.requires_grad_(True)

    if freeze_config == "A":
        pass
    elif freeze_config == "B":
        apply_lora_to_decoder_qv(model.decoder, rank=8, alpha=16.0)
    elif freeze_config == "C":
        for param in model.decoder.parameters():
            param.requires_grad_(True)
    elif freeze_config == "D":
        for param in model.vit.parameters():
            param.requires_grad_(True)
        for param in model.decoder.parameters():
            param.requires_grad_(True)
    else:
        raise ValueError(f"Unknown freeze_config: {freeze_config}")

    return [param for param in model.parameters() if param.requires_grad]


def set_training_modes(model: VisionLanguageModel, freeze_config: str) -> None:
    model.projector.train()
    model.vit.train(freeze_config == "D")
    model.decoder.train(freeze_config in ("C", "D"))
    if freeze_config in ("A", "B"):
        model.decoder.eval()


@torch.no_grad()
def evaluate_exact_match(
    model: VisionLanguageModel,
    tokenizer,
    val_loader,
    injection: str,
    device: torch.device,
    max_examples: int,
    max_new_tokens: int,
    gen_kwargs: dict,
) -> float:
    model.eval()
    predictions: list[str] = []
    golds: list[str] = []

    for batch in val_loader:
        remaining = max_examples - len(golds)
        if remaining <= 0:
            break

        images = batch["image"][:remaining].to(device, non_blocking=True)
        questions = batch["question"][:remaining]
        answers = batch["answer"][:remaining]
        prompts = make_prompts(questions, injection)
        decoded = model.generate(
            images,
            prompts,
            injection=injection,
            max_new_tokens=max_new_tokens,
            **gen_kwargs,
        )
        predictions.extend(clean_prediction(text) for text in decoded)
        golds.extend(answers)

    return batch_clevr_accuracy(predictions, golds)["overall"]


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.output_dir is None:
        args.output_dir = (
            Path("runs") / f"vlm_{args.injection}_{args.mask_mode}_{args.freeze_config}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.cuda.reset_peak_memory_stats(device)

    ckpt = torch.load(args.pretrained_vit, map_location="cpu")
    vit_cfg = ckpt["config"]["vit"]
    train_loader, val_loader = data.build_clevr_loaders(
        img_size=vit_cfg["img_size"],
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["train"]["num_workers"],
    )

    image_encoder = vit.ViT(**vit_cfg)
    image_encoder.load_state_dict(ckpt["vit"] if "vit" in ckpt else ckpt)

    tokenizer = AutoTokenizer.from_pretrained(cfg["decoder"]["model_name"])
    added_tokens = 0
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            added_tokens += tokenizer.add_special_tokens({"pad_token": "<pad>"})
    image_token_id = None
    if args.injection == "interleaved":
        added_tokens += tokenizer.add_special_tokens(
            {"additional_special_tokens": ["<image>"]}
        )
        image_token_id = tokenizer.convert_tokens_to_ids("<image>")

    dtype = torch.bfloat16 if cfg["decoder"].get("torch_dtype") == "bfloat16" else torch.float32
    model_kwargs = {"torch_dtype": dtype}
    attn_impl = cfg["decoder"].get("attn_implementation")
    if attn_impl:
        model_kwargs["attn_implementation"] = attn_impl
    try:
        decoder = AutoModelForCausalLM.from_pretrained(
            cfg["decoder"]["model_name"], **model_kwargs
        )
    except Exception:
        model_kwargs.pop("attn_implementation", None)
        decoder = AutoModelForCausalLM.from_pretrained(
            cfg["decoder"]["model_name"], **model_kwargs
        )
    if added_tokens > 0:
        decoder.resize_token_embeddings(len(tokenizer))
    decoder.config.use_cache = False
    if args.freeze_config in ("C", "D") and hasattr(decoder, "gradient_checkpointing_enable"):
        decoder.gradient_checkpointing_enable()

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
    ).to(device)

    trainable_parameters = configure_trainable_parameters(model, args.freeze_config)

    trainable_params = sum(p.numel() for p in trainable_parameters)
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=cfg["optim"]["lr"],
        betas=tuple(cfg["optim"]["betas"]),
        weight_decay=cfg["optim"]["weight_decay"],
    )

    num_steps = cfg["train"]["num_steps"]
    grad_accum = cfg["train"]["gradient_accumulation_steps"]
    warmup_steps = cfg["optim"].get("warmup_steps", 0)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, num_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    use_amp = device.type == "cuda"
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_amp
        else nullcontext()
    )

    visual_tokens = 1 if args.injection == "cls" else image_encoder.patch_embedding.num_patches + 1
    max_new_tokens = cfg["generation"]["max_new_tokens"]
    gen_kwargs = {
        "do_sample": cfg["generation"].get("do_sample", False),
        "temperature": cfg["generation"].get("temperature", 1.0),
        "top_p": cfg["generation"].get("top_p", 1.0),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    history = []
    best_acc = -1.0
    total_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    set_training_modes(model, args.freeze_config)
    start_time = time.perf_counter()
    train_iter = itertools.cycle(train_loader)

    for step in range(1, num_steps + 1):
        set_training_modes(model, args.freeze_config)
        step_loss = 0.0
        for _ in range(grad_accum):
            batch = next(train_iter)
            images = batch["image"].to(device, non_blocking=True)
            tokens = tokenize_vqa_batch(
                tokenizer,
                batch["question"],
                batch["answer"],
                args.injection,
                device,
            )
            with autocast_ctx:
                outputs = model(
                    images=images,
                    input_ids=tokens["input_ids"],
                    attention_mask=tokens["attention_mask"],
                    labels=tokens["labels"],
                    injection=args.injection,
                    mask_mode=args.mask_mode,
                )
                loss = outputs["loss"] / grad_accum
            loss.backward()
            step_loss += loss.item()

        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        total_loss += step_loss
        if step % cfg["train"]["log_every"] == 0:
            print(
                f"step {step:04d}/{num_steps} "
                f"loss={total_loss / cfg['train']['log_every']:.4f} "
                f"lr={scheduler.get_last_lr()[0]:.2e} "
                f"grad_norm={float(grad_norm):.3f}"
            )
            total_loss = 0.0

        if step % cfg["train"]["eval_every_steps"] == 0 or step == num_steps:
            val_acc = evaluate_exact_match(
                model,
                tokenizer,
                val_loader,
                args.injection,
                device,
                cfg["train"]["eval_max_examples"],
                max_new_tokens,
                gen_kwargs,
            )
            row = {
                "step": step,
                "val_exact_match": val_acc,
                "lr": scheduler.get_last_lr()[0],
            }
            history.append(row)
            print(f"eval step {step:04d}: val_exact_match={val_acc:.4f}")
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(
                    {
                        "config": cfg,
                        "vit_config": vit_cfg,
                        "decoder_model_name": cfg["decoder"]["model_name"],
                        "injection": args.injection,
                        "mask_mode": args.mask_mode,
                        "freeze_config": args.freeze_config,
                        "pretrained_vit": str(args.pretrained_vit),
                        "trainable_state": {
                            name: param.detach().cpu()
                            for name, param in model.named_parameters()
                            if param.requires_grad
                        },
                        "projector": model.projector.state_dict(),
                        "image_token_id": image_token_id,
                        "step": step,
                        "val_exact_match": val_acc,
                    },
                    args.output_dir / "best.pt",
                )

    wall_clock_seconds = time.perf_counter() - start_time
    peak_memory_bytes = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    metrics = {
        "injection": args.injection,
        "mask_mode": args.mask_mode,
        "freeze_config": args.freeze_config,
        "val_exact_match_500": history[-1]["val_exact_match"],
        "best_val_exact_match_500": best_acc,
        "visual_tokens_per_example": visual_tokens,
        "trainable_parameters": trainable_params,
        "peak_gpu_memory_bytes": peak_memory_bytes,
        "peak_gpu_memory_mb": peak_memory_bytes / (1024**2),
        "wall_clock_seconds": wall_clock_seconds,
        "wall_clock_time_per_step": wall_clock_seconds / max(1, num_steps),
    }

    with open(args.output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(args.output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
