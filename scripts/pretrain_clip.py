"""§3 — CLIP-style pretraining on EuroSAT.

You implement the training loop. This script provides the CLI scaffolding,
config loading, and logging hooks.

Usage:
    uv run python scripts/pretrain_clip.py --config configs/clip_eurosat.yaml
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from vlm import data
from vlm import clip
from vlm import eval as vlm_eval
from basics import vit
from basics.text_encoder import FrozenTextEncoder
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

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    train, val, _test = data.build_eurosat_loaders(
        img_size=cfg["vit"]["img_size"],
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["train"]["num_workers"],
    )
    vis_transformer = vit.ViT(**cfg["vit"]).to(device)
    vis_transformer.d_model = cfg["vit"]["d_model"]
    text_encoder = FrozenTextEncoder(cfg["text_encoder"]["model_name"]).to(device)
    text_encoder.eval() 
    
    proj_heads = clip.ProjectionHeads(
        cfg["vit"]["d_model"],
        d_text=text_encoder.embedding_dim,
        d_proj=cfg["projection"]["d_proj"]).to(device)
    
    logit_scale = torch.nn.Parameter(clip.init_logit_scale().detach().to(device))
    optim = torch.optim.AdamW(
        list(vis_transformer.parameters())
        + list(proj_heads.parameters())
        + [logit_scale],
        lr=cfg["optim"]["lr"],
        betas=tuple(cfg["optim"]["betas"]),
        weight_decay = cfg["optim"]["weight_decay"]
    )

    num_epochs = cfg["train"]["num_epochs"]
    total_steps = max(1, len(train) * num_epochs)
    warmup_steps = cfg["optim"].get("warmup_steps", 0)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    wandb_run = None
    if args.wandb:
        import wandb

        wandb_run = wandb.init(project="clip-eurosat", config=cfg, dir=str(args.output_dir))

    class_prompts = [f"a satellite image of {name}" for name in data.EUROSAT_CLASSES]
    class_indices = list(range(len(class_prompts)))
    prompt_to_idx = {prompt: idx for idx, prompt in enumerate(class_prompts)}
    with torch.no_grad():
        cached_text_embeds = text_encoder(class_prompts).to(device)

    best_val_acc = -1.0
    best_path = args.output_dir / "best.pt"
    global_step = 0
    eval_every_epoch = cfg["train"].get("eval_every_epoch", 1)

    for epoch in range(num_epochs):
        vis_transformer.train()
        proj_heads.train()
        total_loss = 0.0

        for images, captions in train:
            images = images.to(device, non_blocking=True)
            caption_ids = torch.tensor(
                [prompt_to_idx[caption] for caption in captions],
                device=device,
                dtype=torch.long,
            )
            text_embeds = cached_text_embeds.index_select(0, caption_ids)

            image_embeds = vis_transformer(images)
            image_proj, text_proj = proj_heads(image_embeds, text_embeds)
            loss = clip.clip_loss(image_proj, text_proj, logit_scale)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            scheduler.step()
            logit_scale.data.clamp_(max=math.log(100.0))

            total_loss += loss.item()
            global_step += 1

        train_loss = total_loss / max(1, len(train))
        should_eval = (epoch + 1) % eval_every_epoch == 0 or epoch + 1 == num_epochs
        val_acc = None
        if should_eval:
            val_acc = vlm_eval.zeroshot_classification_accuracy(
                vis_transformer,
                proj_heads,
                text_encoder,
                val,
                class_prompts,
                class_indices,
                device,
            )
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "config": cfg,
                        "train_loss": train_loss,
                        "val_acc": val_acc,
                        "vit": vis_transformer.state_dict(),
                        "projection_heads": proj_heads.state_dict(),
                        "logit_scale": logit_scale.detach().cpu(),
                    },
                    best_path,
                )

        lr = scheduler.get_last_lr()[0]
        log_items = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "lr": lr,
            "logit_scale": logit_scale.exp().item(),
        }
        if val_acc is not None:
            log_items["val_acc"] = val_acc

        print(
            f"epoch {epoch + 1:03d}/{num_epochs} "
            f"train_loss={train_loss:.4f} "
            f"val_acc={val_acc:.4f} "
            f"lr={lr:.2e}"
            if val_acc is not None
            else f"epoch {epoch + 1:03d}/{num_epochs} "
                 f"train_loss={train_loss:.4f} lr={lr:.2e}"
        )
        if wandb_run is not None:
            wandb_run.log(log_items, step=global_step)

    if wandb_run is not None:
        wandb_run.finish()
    

if __name__ == "__main__":
    main()
