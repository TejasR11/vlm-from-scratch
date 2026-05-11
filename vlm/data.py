"""Dataset loaders for EuroSAT (§3), RESISC45 (§4), and CLEVR (§5).

All loaders return torch DataLoaders. Images are resized to 64x64 and
normalized to ImageNet stats unless otherwise specified.

DO NOT MODIFY THIS FILE (you may extend it, but the staff tests rely on the
provided functions).
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Callable, Sequence

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ImageNet normalization used by most pretrained vision encoders.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def default_image_transform(img_size: int = 64) -> Callable:
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


# ---------------------------------------------------------------------------
# EuroSAT — §3 (CLIP pretraining)
# ---------------------------------------------------------------------------


EUROSAT_CLASSES = [
    "Annual Crop", "Forest", "Herbaceous Vegetation", "Highway",
    "Industrial Buildings", "Pasture", "Permanent Crop", "Residential Buildings",
    "River", "Sea or Lake",
]


class EuroSATCLIPDataset(Dataset):
    """EuroSAT with synthetic captions of the form
    'a satellite image of {class_name}'.

    Yields (image_tensor, caption_string) tuples.
    """

    def __init__(self, split: str = "train", img_size: int = 64, ds=None) -> None:
        if ds is None:
            from datasets import load_dataset

            ds = load_dataset("blanchon/EuroSAT_RGB", split=split)
        self.ds = ds
        self.transform = default_image_transform(img_size)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        ex = self.ds[idx]
        img = ex["image"].convert("RGB")
        label_idx = ex["label"]
        class_name = EUROSAT_CLASSES[label_idx]
        caption = f"a satellite image of {class_name}"
        return self.transform(img), caption


def _stratified_split_indices(
    labels: Sequence[int],
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    rng = random.Random(seed)
    label_to_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        label_to_indices[int(label)].append(idx)

    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []
    for indices in label_to_indices.values():
        rng.shuffle(indices)
        train_end = int(len(indices) * train_frac)
        val_end = train_end + int(len(indices) * val_frac)
        train_indices.extend(indices[:train_end])
        val_indices.extend(indices[train_end:val_end])
        test_indices.extend(indices[val_end:])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(test_indices)
    return train_indices, val_indices, test_indices


def build_eurosat_loaders(
    img_size: int = 64,
    batch_size: int = 256,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    from datasets import load_dataset

    full_ds = load_dataset("blanchon/EuroSAT_RGB", split="train")
    train_indices, val_indices, test_indices = _stratified_split_indices(full_ds["label"])

    train = EuroSATCLIPDataset(img_size=img_size, ds=full_ds.select(train_indices))
    val = EuroSATCLIPDataset(img_size=img_size, ds=full_ds.select(val_indices))
    test = EuroSATCLIPDataset(img_size=img_size, ds=full_ds.select(test_indices))

    def _collate(batch):
        imgs = torch.stack([b[0] for b in batch])
        caps = [b[1] for b in batch]
        return imgs, caps

    train_dl = DataLoader(
        train, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        collate_fn=_collate, pin_memory=True, drop_last=True,
    )
    val_dl = DataLoader(
        val, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        collate_fn=_collate, pin_memory=True,
    )
    test_dl = DataLoader(
        test, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        collate_fn=_collate, pin_memory=True,
    )
    return train_dl, val_dl, test_dl


# ---------------------------------------------------------------------------
# RESISC45 — §4 (LoRA / full FT downstream task)
# ---------------------------------------------------------------------------


class RESISC45Dataset(Dataset):
    """Remote-sensing scene classification with 45 categories.

    Yields (image_tensor, label_int) tuples.
    """

    def __init__(self, split: str = "train", img_size: int = 64) -> None:
        from datasets import load_dataset

        self.ds = load_dataset("timm/resisc45", split=split)
        self.transform = default_image_transform(img_size)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        ex = self.ds[idx]
        img = ex["image"].convert("RGB")
        return self.transform(img), int(ex["label"])


def build_resisc45_loaders(
    img_size: int = 64,
    batch_size: int = 128,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader]:
    train = RESISC45Dataset("train", img_size=img_size)
    test = RESISC45Dataset("validation", img_size=img_size)
    train_dl = DataLoader(
        train, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True, drop_last=True,
    )
    test_dl = DataLoader(
        test, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True,
    )
    return train_dl, test_dl


# ---------------------------------------------------------------------------
# CLEVR — §5 (VLM training and evaluation)
# ---------------------------------------------------------------------------


class CLEVRMiniDataset(Dataset):
    """Preprocessed 10k-example CLEVR subset.

    Expects on disk:
        data/clevr_mini/{split}.jsonl   (one JSON per line: image_file, question, answer, q_type)
        data/clevr_mini/images/         (PNG files referenced by image_file)
    """

    def __init__(
        self,
        split: str = "train",
        root: str = "data/clevr_mini",
        img_size: int = 64,
    ) -> None:
        self.root = Path(root)
        self.img_size = img_size
        self.transform = default_image_transform(img_size)
        with open(self.root / f"{split}.jsonl") as f:
            self.examples = [json.loads(line) for line in f]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        img = Image.open(self.root / "images" / ex["image_file"]).convert("RGB")
        return {
            "image": self.transform(img),
            "question": ex["question"],
            "answer": ex["answer"],
            "q_type": ex.get("q_type", "other"),  # "spatial" or "other" for §6
        }


def build_clevr_loaders(
    img_size: int = 64,
    batch_size: int = 32,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader]:
    train = CLEVRMiniDataset("train", img_size=img_size)
    val = CLEVRMiniDataset("val", img_size=img_size)

    def _collate(batch):
        return {
            "image": torch.stack([b["image"] for b in batch]),
            "question": [b["question"] for b in batch],
            "answer": [b["answer"] for b in batch],
            "q_type": [b["q_type"] for b in batch],
        }

    train_dl = DataLoader(
        train, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        collate_fn=_collate, pin_memory=True, drop_last=True,
    )
    val_dl = DataLoader(
        val, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        collate_fn=_collate, pin_memory=True,
    )
    return train_dl, val_dl
