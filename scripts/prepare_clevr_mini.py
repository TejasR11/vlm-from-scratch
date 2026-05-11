"""Prepare the preprocessed CLEVR-mini subset used by the homework.

This script converts the official CLEVR v1.0 release into the layout consumed
by ``vlm.data.CLEVRMiniDataset``:

    data/clevr_mini/
    ├── train.jsonl
    ├── val.jsonl
    └── images/

By default it creates 9,500 train examples from the official train split and
500 validation examples from the official val split, for a 10k-example subset.
Images are copied at their original CLEVR resolution unless ``--image-size`` is
provided.

Usage:
    uv run python scripts/prepare_clevr_mini.py --source CLEVR_v1.0
    uv run python scripts/prepare_clevr_mini.py --source CLEVR_v1.0 --make-archive
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import zipfile
from collections import Counter
from pathlib import Path
from typing import Iterable

from PIL import Image


DEFAULT_SOURCE = Path("CLEVR_v1.0")
DEFAULT_DEST = Path("data/clevr_mini")

SPATIAL_WORDS = ("left", "right", "front", "behind")
QUERY_ATTR_FUNCS = {
    "query_color",
    "query_shape",
    "query_size",
    "query_material",
}
COMPARE_ATTR_FUNCS = {
    "equal_color",
    "equal_shape",
    "equal_size",
    "equal_material",
}
COUNT_COMPARE_FUNCS = {
    "equal_integer",
    "greater_than",
    "less_than",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Path to the unpacked official CLEVR_v1.0 directory.",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help="Output directory for the processed subset.",
    )
    parser.add_argument(
        "--train-count",
        type=int,
        default=9_500,
        help="Number of examples to sample from official CLEVR train.",
    )
    parser.add_argument(
        "--val-count",
        type=int,
        default=500,
        help="Number of examples to sample from official CLEVR val.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Sampling seed.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Optional square output image size. Omit to preserve original CLEVR images.",
    )
    parser.add_argument(
        "--jpeg",
        action="store_true",
        help="Write JPEG images instead of PNG images for a smaller archive.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=90,
        help="JPEG quality used with --jpeg.",
    )
    parser.add_argument(
        "--include-program",
        action="store_true",
        help="Include CLEVR functional programs in the JSONL rows.",
    )
    parser.add_argument(
        "--make-archive",
        action="store_true",
        help="Also write data/clevr_mini.zip and print its SHA256.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete an existing output directory before writing.",
    )
    return parser.parse_args()


def load_questions(source: Path, split: str) -> list[dict]:
    path = source / "questions" / f"CLEVR_{split}_questions.json"
    with open(path) as f:
        return json.load(f)["questions"]


def classify_q_type(question: dict) -> str:
    """Map CLEVR programs to the small set of labels used by the homework."""
    program = question.get("program") or []
    funcs = [step.get("function", "") for step in program]
    text = question.get("question", "").lower()

    if "relate" in funcs or any(word in text for word in SPATIAL_WORDS):
        return "spatial"

    last_func = funcs[-1] if funcs else ""
    if last_func == "count" or last_func in COUNT_COMPARE_FUNCS:
        return "count"
    if last_func == "exist":
        return "exist"
    if last_func in QUERY_ATTR_FUNCS:
        return "query_attr"
    if last_func in COMPARE_ATTR_FUNCS:
        return "compare_attr"
    return "other"


def sample_questions(questions: list[dict], count: int, seed: int) -> list[dict]:
    if count > len(questions):
        raise ValueError(f"Requested {count} examples from only {len(questions)} questions.")
    rng = random.Random(seed)
    indices = rng.sample(range(len(questions)), count)
    indices.sort()
    return [questions[i] for i in indices]


def output_image_name(source_name: str, *, use_jpeg: bool) -> str:
    suffix = ".jpg" if use_jpeg else ".png"
    return Path(source_name).with_suffix(suffix).name


def write_image(
    src: Path,
    dest: Path,
    *,
    image_size: int | None,
    use_jpeg: bool,
    jpeg_quality: int,
) -> None:
    if dest.exists():
        return

    if image_size is None and not use_jpeg:
        shutil.copy2(src, dest)
        return

    with Image.open(src) as img:
        img = img.convert("RGB")
        if image_size is not None:
            img = img.resize((image_size, image_size), Image.Resampling.BICUBIC)
        if use_jpeg:
            img.save(dest, format="JPEG", quality=jpeg_quality, optimize=True)
        else:
            img.save(dest, format="PNG", optimize=True)


def jsonl_row(question: dict, *, image_file: str, include_program: bool) -> dict:
    row = {
        "image_file": image_file,
        "question": question["question"],
        "answer": question["answer"],
        "q_type": classify_q_type(question),
        "source_split": question["split"],
        "source_image_file": question["image_filename"],
        "question_index": question.get("question_index"),
        "question_family_index": question.get("question_family_index"),
    }
    if include_program:
        row["program"] = question.get("program", [])
    return row


def write_split(
    *,
    source: Path,
    dest: Path,
    split: str,
    questions: Iterable[dict],
    image_size: int,
    use_jpeg: bool,
    jpeg_quality: int,
    include_program: bool,
) -> Counter:
    q_type_counts: Counter = Counter()
    image_dir = dest / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    with open(dest / f"{split}.jsonl", "w") as f:
        for question in questions:
            image_name = output_image_name(question["image_filename"], use_jpeg=use_jpeg)
            src_image = source / "images" / question["split"] / question["image_filename"]
            write_image(
                src_image,
                image_dir / image_name,
                image_size=image_size,
                use_jpeg=use_jpeg,
                jpeg_quality=jpeg_quality,
            )
            row = jsonl_row(question, image_file=image_name, include_program=include_program)
            q_type_counts[row["q_type"]] += 1
            f.write(json.dumps(row, sort_keys=True) + "\n")

    return q_type_counts


def write_attribution(source: Path, dest: Path, args: argparse.Namespace, counts: dict) -> None:
    for filename in ("LICENSE.txt", "COPYRIGHT.txt"):
        src = source / filename
        if src.exists():
            shutil.copy2(src, dest / filename)

    manifest = {
        "name": "clevr_mini",
        "source": "CLEVR v1.0",
        "source_url": "https://cs.stanford.edu/people/jcjohns/clevr/",
        "license": "Creative Commons Attribution 4.0 International",
        "train_count": args.train_count,
        "val_count": args.val_count,
        "seed": args.seed,
        "image_size": args.image_size if args.image_size is not None else "original",
        "image_format": "jpeg" if args.jpeg else "png",
        "q_type_counts": counts,
    }
    with open(dest / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")

    if args.image_size is None:
        image_note = "Images are copied from CLEVR v1.0 at their original resolution."
    else:
        image_note = f"Images are resized to {args.image_size}x{args.image_size}."

    with open(dest / "README.md", "w") as f:
        f.write(
            "# CLEVR-mini\n\n"
            "This is a deterministic subset of CLEVR v1.0 prepared for the HW3 "
            "vision-language modeling exercises. "
            f"{image_note}\n\n"
            "Source: https://cs.stanford.edu/people/jcjohns/clevr/\n\n"
            "License: Creative Commons Attribution 4.0 International. See "
            "`LICENSE.txt` and `COPYRIGHT.txt`.\n\n"
            "Citation: Johnson, Justin, et al. \"CLEVR: A diagnostic dataset for "
            "compositional language and elementary visual reasoning.\" CVPR 2017.\n"
        )


def make_archive(dest: Path) -> tuple[Path, str]:
    archive = dest.parent / f"{dest.name}.zip"
    if archive.exists():
        archive.unlink()
    with zipfile.ZipFile(
        archive,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as zip_file:
        for path in sorted(dest.rglob("*")):
            zip_file.write(path, arcname=path.relative_to(dest.parent))
    digest = sha256(archive)
    return archive, digest


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    args = parse_args()
    source = args.source
    dest = args.dest

    if not (source / "questions").exists() or not (source / "images").exists():
        raise FileNotFoundError(f"{source} does not look like an unpacked CLEVR_v1.0 directory.")

    if dest.exists():
        if not args.overwrite:
            raise FileExistsError(f"{dest} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    train = sample_questions(load_questions(source, "train"), args.train_count, args.seed)
    val = sample_questions(load_questions(source, "val"), args.val_count, args.seed + 1)

    counts = {
        "train": dict(
            write_split(
                source=source,
                dest=dest,
                split="train",
                questions=train,
                image_size=args.image_size,
                use_jpeg=args.jpeg,
                jpeg_quality=args.jpeg_quality,
                include_program=args.include_program,
            )
        ),
        "val": dict(
            write_split(
                source=source,
                dest=dest,
                split="val",
                questions=val,
                image_size=args.image_size,
                use_jpeg=args.jpeg,
                jpeg_quality=args.jpeg_quality,
                include_program=args.include_program,
            )
        ),
    }
    write_attribution(source, dest, args, counts)

    num_images = sum(1 for _ in (dest / "images").iterdir())
    print(f"Wrote {dest}")
    print(f"Examples: train={args.train_count}, val={args.val_count}, images={num_images}")
    print(f"q_type counts: {counts}")

    if args.make_archive:
        archive, digest = make_archive(dest)
        print(f"Archive: {archive}")
        print(f"SHA256:  {digest}")


if __name__ == "__main__":
    main()
