#!/usr/bin/env python3
"""Build MetaShift metadata CSV for the SPUME framework.

MetaShift (https://github.com/Weixin-Liang/MetaShift) organises images as::

    metashift/
    ├── cat/
    │   ├── couch/          # context = couch
    │   │   ├── xxx.jpg
    │   │   └── ...
    │   ├── bed/            # context = bed
    │   │   └── ...
    │   └── ...
    ├── dog/
    │   ├── grass/          # context = grass
    │   │   └── ...
    │   ├── beach/          # context = beach
    │   │   └── ...
    │   └── ...

This script:

1. Scans the directory tree.
2. Assigns train / val / test splits with a configurable distribution.
3. Writes ``metadata.csv`` consumable by ``MetaShiftDataset``.

Split strategy (Cat vs Dog example)
------------------------------------
- **Contexts unique to one class in train** (e.g. cat→{couch,bed},
  dog→{grass,beach}) → 80% train, 20% val (stratified by class).
- **Contexts shared or rare** (e.g. cat images on grass, dog images on
  couch) → all go to test.  This simulates the real-world shift where
  spurious correlations break at test time.
"""

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

import numpy as np


def collect_images(root: Path) -> list[dict]:
    """Walk *root* and collect every image file with its class and context.

    Expected layout:  root / class_name / context_name / *.jpg
    """
    records = []
    exts = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        for ctx_dir in sorted(class_dir.iterdir()):
            if not ctx_dir.is_dir():
                continue
            context_name = ctx_dir.name
            for img_file in sorted(ctx_dir.iterdir()):
                if img_file.suffix.lower() not in exts:
                    continue
                rel_path = img_file.relative_to(root)
                records.append(
                    {
                        "img_path": str(rel_path).replace("\\", "/"),
                        "class_name": class_name,
                        "context": context_name,
                        "filename": img_file.name,
                    }
                )
    return records


def assign_splits(
    records: list[dict],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> list[dict]:
    """Assign train / val / test split labels.

    Strategy
    --------
    1. Group records by (class_name, context).
    2. For each group, determine whether the context is "exclusive" to
       that class (i.e. appears only for one class).  Exclusive contexts
       get the standard train/val split.  Shared contexts go entirely to
       test since they represent the distribution-shift challenge.
    3. Within the train portion, do an 80/20 train/val stratified split.
    """
    rng = np.random.default_rng(seed)

    # build class → set of contexts
    class_contexts: dict[str, set[str]] = defaultdict(set)
    for r in records:
        class_contexts[r["class_name"]].add(r["context"])

    # determine exclusive vs shared contexts
    all_classes = list(class_contexts.keys())
    context_classes: dict[str, set[str]] = defaultdict(set)
    for cls, ctxs in class_contexts.items():
        for ctx in ctxs:
            context_classes[ctx].add(cls)

    exclusive: dict[str, set[str]] = {}
    shared: dict[str, set[str]] = {}
    for cls in all_classes:
        exclusive[cls] = {c for c in class_contexts[cls] if len(context_classes[c]) == 1}
        shared[cls] = {c for c in class_contexts[cls] if len(context_classes[c]) > 1}

    assigned = []
    for r in records:
        cls = r["class_name"]
        ctx = r["context"]

        if ctx in exclusive.get(cls, set()):
            # exclusive context → train or val
            if rng.random() < train_ratio:
                r["split"] = "train"
            else:
                r["split"] = "val"
        else:
            # shared or unknown context → test (distribution-shift samples)
            r["split"] = "test"

        # env = "context" directly (text label, same pattern as Spawrious)
        r["env"] = ctx
        assigned.append(r)

    return assigned


def build_y_mapping(records: list[dict]) -> dict[str, int]:
    """Map sorted unique class names to integer labels."""
    classes = sorted({r["class_name"] for r in records})
    return {c: i for i, c in enumerate(classes)}


def main():
    parser = argparse.ArgumentParser(
        description="Build MetaShift metadata CSV for SPUME."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Root directory of MetaShift images (class/context/*.jpg layout).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path for output metadata CSV. Default: <data-root>/../metadata_<subset>.csv",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of exclusive-context images assigned to train (rest→val).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for split assignment.",
    )
    parser.add_argument(
        "--subset-name",
        default="metashift",
        help="Name used in the default output filename.",
    )
    args = parser.parse_args()

    root = args.data_root.resolve()
    if not root.exists():
        raise FileNotFoundError(f"data-root not found: {root}")

    # --- collect images ---
    records = collect_images(root)
    if not records:
        raise RuntimeError(
            f"No images found under {root}. "
            f"Expected layout: root/class_name/context_name/*.jpg"
        )
    print(f"Found {len(records)} images across {root}")

    # --- assign splits ---
    records = assign_splits(records, train_ratio=args.train_ratio, seed=args.seed)

    # --- build y mapping ---
    class_to_y = build_y_mapping(records)
    for r in records:
        r["y"] = class_to_y[r["class_name"]]

    # --- summary ---
    split_counts = defaultdict(int)
    for r in records:
        split_counts[r["split"]] += 1
    print(f"Class mapping: {class_to_y}")
    print(f"Split counts: {dict(split_counts)}")

    # context distribution per split
    for split_name in ("train", "val", "test"):
        ctx_per_class: dict[str, set[str]] = defaultdict(set)
        for r in records:
            if r["split"] == split_name:
                ctx_per_class[r["class_name"]].add(r["context"])
        print(f"\n{split_name} contexts per class:")
        for cls in sorted(ctx_per_class):
            print(f"  {cls}: {sorted(ctx_per_class[cls])}")

    # --- write CSV ---
    output_path = args.output
    if output_path is None:
        output_path = root.parent / f"metadata_{args.subset_name}.csv"
    output_path = Path(output_path)

    fieldnames = ["img_path", "class_name", "y", "split", "env", "filename"]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow({k: r[k] for k in fieldnames})

    print(f"\nMetadata CSV written to: {output_path}")
    print(f"  {len(records)} rows, {len(class_to_y)} classes, "
          f"{len({r['env'] for r in records})} unique contexts")


if __name__ == "__main__":
    main()
