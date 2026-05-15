"""§4 — Compare full FT, LoRA, and linear probe on RESISC45.

Usage:
    uv run python scripts/finetune_resisc.py --config configs/lora_resisc.yaml \\
        --method lora --rank 8 --pretrained runs/clip_eurosat/best.pt
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from basics import vit
from basics.lora import apply_lora_to_attention
from vlm import data


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


class RESISCClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, d_model: int, num_classes: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(images))


@torch.no_grad()
def accuracy(model: nn.Module, loader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        preds = model(images).argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.numel()
    return correct / max(total, 1)


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.output_dir is None:
        args.output_dir = Path("runs") / f"resisc_{args.method}_rank{args.rank}"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.cuda.reset_peak_memory_stats(device)

    ckpt = torch.load(args.pretrained, map_location="cpu")
    pretrain_cfg = ckpt["config"]
    vit_cfg = pretrain_cfg["vit"]

    train_loader, test_loader = data.build_resisc45_loaders(
        img_size=vit_cfg["img_size"],
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["train"]["num_workers"],
    )
    encoder = vit.ViT(**vit_cfg)
    encoder.d_model = vit_cfg["d_model"]
    encoder.load_state_dict(ckpt["vit"] if "vit" in ckpt else ckpt)

    if args.method == "linear_probe":
        for param in encoder.parameters():
            param.requires_grad_(False)
    elif args.method == "lora":
        encoder = apply_lora_to_attention(encoder, args.rank, args.alpha)
    elif args.method == "full_ft":
        for param in encoder.parameters():
            param.requires_grad_(True)

    model = RESISCClassifier(
        encoder=encoder,
        d_model=vit_cfg["d_model"],
        num_classes=cfg["num_classes"],
    ).to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    method_lr = cfg.get("methods", {}).get(args.method, {}).get("lr")
    lr = method_lr if method_lr is not None else cfg["optim"]["lr"]
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        betas=tuple(cfg["optim"]["betas"]),
        weight_decay=cfg["optim"]["weight_decay"],
    )

    num_epochs = cfg["train"]["num_epochs"]
    total_steps = max(1, len(train_loader) * num_epochs)
    warmup_steps = cfg["optim"].get("warmup_steps", 0)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)
    loss_fn = nn.CrossEntropyLoss()
    history = []
    best_acc = -1.0
    best_path = args.output_dir / "best.pt"
    start_time = time.perf_counter()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            loss = loss_fn(logits, labels)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            scheduler.step()

            total_loss += loss.item()

        train_loss = total_loss / max(1, len(train_loader))
        test_acc = accuracy(model, test_loader, device)
        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "test_acc": test_acc,
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(row)
        print(
            f"epoch {epoch + 1:03d}/{num_epochs} "
            f"train_loss={train_loss:.4f} "
            f"test_acc={test_acc:.4f} "
            f"lr={row['lr']:.2e}"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(
                {
                    "epoch": epoch + 1,
                    "method": args.method,
                    "rank": args.rank,
                    "alpha": args.alpha,
                    "config": cfg,
                    "pretrained": str(args.pretrained),
                    "model": model.state_dict(),
                    "test_acc": test_acc,
                },
                best_path,
            )

    wall_clock_seconds = time.perf_counter() - start_time
    peak_memory_bytes = (
        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    )
    metrics = {
        "method": args.method,
        "rank": args.rank if args.method == "lora" else None,
        "alpha": args.alpha if args.method == "lora" else None,
        "pretrained": str(args.pretrained),
        "final_test_accuracy": history[-1]["test_acc"],
        "best_test_accuracy": best_acc,
        "trainable_parameters": trainable_params,
        "total_parameters": total_params,
        "trainable_ratio": trainable_params / max(1, total_params),
        "peak_gpu_memory_bytes": peak_memory_bytes,
        "peak_gpu_memory_mb": peak_memory_bytes / (1024**2),
        "wall_clock_seconds": wall_clock_seconds,
    }

    with open(args.output_dir / "history.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    with open(args.output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
