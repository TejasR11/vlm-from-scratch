# Data

Datasets are downloaded on demand.

- **EuroSAT** — auto-downloaded by `vlm.data.EuroSATCLIPDataset` from
  `blanchon/EuroSAT_RGB` on HuggingFace (~90 MB). Cached in
  `~/.cache/huggingface/`.
- **RESISC45** — auto-downloaded by `vlm.data.RESISC45Dataset` from
  `timm/resisc45` on HuggingFace.
- **CLEVR-mini** — preprocessed 10k-example subset with the original CLEVR
  image resolution. Run `uv run python scripts/download_clevr.py` before VLM
  training. The script downloads `data/clevr_mini.zip` from Google Drive:
  https://drive.google.com/file/d/1KsswLqfYLl1d91pg5kGUgwtPslo8njTB/view?usp=sharing
  and extracts it to `data/clevr_mini/`.
  Course staff can rebuild the hosted archive from the official CLEVR v1.0
  release with `uv run python scripts/prepare_clevr_mini.py --source CLEVR_v1.0
  --overwrite --make-archive`.

Once extracted, CLEVR-mini lives at `data/clevr_mini/` with this layout:

```
data/clevr_mini/
├── train.jsonl       # one JSON per example
├── val.jsonl
└── images/           # PNG files referenced by the JSONL
```

Each line in `train.jsonl` / `val.jsonl` has the shape:
```json
{"image_file": "scene_00012.png", "question": "How many red cubes are there?", "answer": "3", "q_type": "count"}
```

`q_type` is one of `count`, `exist`, `compare_attr`, `query_attr`, `spatial`,
or `other`. The `spatial` tag is used by the M-RoPE bonus problem in §6.

The generated archive includes CLEVR's `LICENSE.txt`, `COPYRIGHT.txt`, and a
small manifest describing the sampling seed, counts, and image handling.
