#!/usr/bin/env python3
"""Download MetaShift Cat vs Dog subset from HuggingFace and build SPUME metadata.

MetaShift (Liang & Zou, ICLR 2022) provides images with (class, context) pairs.
This script:

1. Downloads the cat/dog subset via ``datasets.load_dataset("metashift", ...)``
2. Analyses per-class context distributions
3. Creates train / val / test splits with **context shift**:
   - Train: images from contexts that are *exclusive* to each class (80/20 train/val)
   - Test: ALL contexts, so minority (shifted) groups are present
4. Saves images to ``datasets/metashift/images/``
5. Writes ``metadata.csv`` with columns:
   img_path, class_name, y, split, env, filename, group_id
6. Outputs dataset statistics to ``analysis/metashift/dataset_statistics.csv``

group_id = class_label * n_contexts + context_label
"""

import argparse
import csv
import hashlib
import io
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # d:\SPUME
DATASET_DIR = PROJECT_ROOT / "datasets" / "metashift"
IMAGE_DIR = DATASET_DIR / "images"
ANALYSIS_DIR = PROJECT_ROOT / "analysis" / "metashift"

SELECTED_CLASSES = ["cat", "dog"]

# Contexts that are typically "indoor" (cat-biased) vs "outdoor" (dog-biased)
# These are heuristics; the script will auto-discover them from the data.
CAT_LIKE_CONTEXTS = {
    "sink", "bathroom", "bed", "couch", "chair", "table", "counter",
    "bookshelf", "laptop", "keyboard", "remote", "toilet",
}
DOG_LIKE_CONTEXTS = {
    "grass", "beach", "fence", "car", "truck", "fire hydrant",
    "frisbee", "bench", "sidewalk", "street", "field", "snow",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def sanitise_filename(class_name: str, context: str, image_id: str) -> str:
    """Produce a unique, safe filename."""
    safe_class = class_name.replace(" ", "_")
    safe_context = context.replace(" ", "_").replace("/", "_")
    return f"{safe_class}_{safe_context}_{image_id}.jpg"


def compute_group_id(y: int, context_idx: int, n_contexts: int) -> int:
    """group_id = class_label * n_contexts + context_label"""
    return y * n_contexts + context_idx


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def download_dataset(cache_dir: Path) -> list[dict]:
    """Download MetaShift cat/dog subset from HuggingFace.

    Returns a list of dicts with keys: image_id, label (int), class_name,
    context, image (PIL Image).
    """
    print("=" * 72)
    print("Step 1: Downloading MetaShift from HuggingFace ...")
    print("=" * 72)

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' library not installed.")
        print("  Install with: pip install datasets")
        sys.exit(1)

    ds = load_dataset(
        "jameszou707/metashift",
        selected_classes=SELECTED_CLASSES,
        cache_dir=str(cache_dir),
    )
    print(f"Loaded splits: {list(ds.keys())}")

    records = []
    for split_name, split_ds in ds.items():
        print(f"  Processing split '{split_name}' ({len(split_ds)} samples) ...")
        for row in tqdm(split_ds, desc=f"  {split_name}"):
            records.append(
                {
                    "image_id": str(row["image_id"]),
                    "label": int(row["label"]),
                    "class_name": SELECTED_CLASSES[int(row["label"])],
                    "context": str(row["context"]),
                    "image": row["image"],
                    "orig_split": split_name,
                }
            )

    print(f"Total records: {len(records)}")
    return records


def analyse_contexts(records: list[dict]) -> dict:
    """Compute per-class context frequencies and categorize contexts."""
    print("\n" + "=" * 72)
    print("Step 2: Analysing context distributions ...")
    print("=" * 72)

    class_context_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    for r in records:
        class_context_counts[r["class_name"]][r["context"]] += 1

    # Determine which contexts belong predominantly to which class
    context_totals: dict[str, dict[str, int]] = defaultdict(dict)
    for ctx in {r["context"] for r in records}:
        for cls in SELECTED_CLASSES:
            context_totals[ctx][cls] = class_context_counts[cls].get(ctx, 0)

    # Classify contexts
    exclusive: dict[str, set[str]] = {c: set() for c in SELECTED_CLASSES}
    shared: set[str] = set()

    for ctx, cls_counts in context_totals.items():
        total = sum(cls_counts.values())
        if total < 5:  # skip tiny contexts
            continue
        # A context is "exclusive" to a class if > 80% of its images belong to that class
        for cls in SELECTED_CLASSES:
            if cls_counts.get(cls, 0) / total > 0.8:
                exclusive[cls].add(ctx)
                break
        else:
            shared.add(ctx)

    for cls in SELECTED_CLASSES:
        print(f"\n{cls} exclusive contexts ({len(exclusive[cls])}): "
              f"{sorted(exclusive[cls])[:15]}...")
    print(f"\nShared contexts ({len(shared)}): {sorted(shared)[:15]}...")

    # Top contexts per class
    for cls in SELECTED_CLASSES:
        top = sorted(class_context_counts[cls].items(), key=lambda x: -x[1])[:10]
        print(f"\nTop-10 contexts for {cls}:")
        for ctx, count in top:
            tag = "EXCL" if ctx in exclusive[cls] else ("SHRD" if ctx in shared else "RARE")
            print(f"  {ctx:25s} {count:6d}  [{tag}]")

    return {
        "class_context_counts": dict(class_context_counts),
        "exclusive": exclusive,
        "shared": shared,
    }


def assign_splits(records: list[dict], ctx_info: dict, seed: int = 42) -> list[dict]:
    """Assign train / val / test splits with context shift.

    Strategy
    --------
    - Train (80%) + Val (20%) from *exclusive* contexts → model learns spurious correlation
    - Test = ALL images (exclusive + shared + minority) → measures robustness to shift
    - This creates a genuine context-shift benchmark: cat mostly seen indoors in train,
      but must recognise cat outdoors at test time.
    """
    print("\n" + "=" * 72)
    print("Step 3: Assigning train / val / test splits ...")
    print("=" * 72)

    rng = np.random.default_rng(seed)
    exclusive = ctx_info["exclusive"]

    for r in records:
        cls = r["class_name"]
        ctx = r["context"]

        if ctx in exclusive.get(cls, set()):
            # Exclusive context → randomly assign 80% train, 20% val
            r["split"] = "train" if rng.random() < 0.8 else "val"
        else:
            # Shared or rare context → test (the distribution-shift samples)
            r["split"] = "test"

    # Ensure at least some samples of each exclusive context appear in test
    # (so we can measure worst-group accuracy on all groups)
    for r in records:
        if r["split"] != "test":
            # 10% chance of moving an exclusive-context image to test
            if rng.random() < 0.10:
                r["split"] = "test"

    return records


def save_images_and_build_metadata(records: list[dict]) -> list[dict]:
    """Save PIL images to disk and prepare metadata rows."""
    print("\n" + "=" * 72)
    print("Step 4: Saving images and building metadata ...")
    print("=" * 72)

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    metadata_rows = []

    # Build context → index mapping first
    all_contexts = sorted({r["context"] for r in records})
    context_to_idx = {ctx: i for i, ctx in enumerate(all_contexts)}
    n_contexts = len(all_contexts)

    for r in tqdm(records, desc="Saving images"):
        filename = sanitise_filename(r["class_name"], r["context"], r["image_id"])
        img_path_rel = f"images/{filename}"
        img_path_abs = DATASET_DIR / img_path_rel

        # Save image if not already on disk
        if not img_path_abs.exists():
            img = r["image"]
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.save(img_path_abs, quality=95)

        ctx_idx = context_to_idx[r["context"]]
        y = r["label"]
        group_id = compute_group_id(y, ctx_idx, n_contexts)

        metadata_rows.append(
            {
                "img_path": img_path_rel,
                "class_name": r["class_name"],
                "y": y,
                "split": r["split"],
                "env": r["context"],  # env = context name (same convention as Spawrious)
                "filename": filename,
                "group_id": group_id,
                "context_name": r["context"],
                "context_label": ctx_idx,
                "image_id": r["image_id"],
            }
        )

    return metadata_rows


def write_metadata(rows: list[dict]) -> Path:
    """Write metadata CSV."""
    csv_path = DATASET_DIR / "metadata_metashift_catdog.csv"
    fieldnames = [
        "img_path", "class_name", "y", "split", "env", "filename",
        "group_id", "context_name", "context_label", "image_id",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nMetadata CSV: {csv_path}  ({len(rows)} rows)")
    return csv_path


def compute_statistics(rows: list[dict], n_contexts: int) -> None:
    """Compute and save dataset statistics."""
    print("\n" + "=" * 72)
    print("Step 5: Computing dataset statistics ...")
    print("=" * 72)

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    # --- per-split counts ---
    split_counts = defaultdict(int)
    for r in rows:
        split_counts[r["split"]] += 1
    print(f"\nSplit sizes: {dict(split_counts)}")

    # --- per-group counts ---
    group_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    for r in rows:
        key = f"{r['class_name']}×{r['context_name']}"
        group_counts[key][r["split"]] += 1

    # --- per-class context distribution (train) ---
    print("\n--- Train split: class × context matrix ---")
    classes = sorted({r["class_name"] for r in rows})
    contexts = sorted({r["context_name"] for r in rows})

    train_matrix = defaultdict(lambda: defaultdict(int))
    test_matrix = defaultdict(lambda: defaultdict(int))
    for r in rows:
        if r["split"] == "train":
            train_matrix[r["class_name"]][r["context_name"]] += 1
        elif r["split"] == "test":
            test_matrix[r["class_name"]][r["context_name"]] += 1

    # Print train matrix
    print(f"{'class':>12s}", end="")
    for ctx in contexts[:8]:
        print(f"  {ctx:>20s}", end="")
    print()
    for cls in classes:
        print(f"{cls:>12s}", end="")
        for ctx in contexts[:8]:
            count = train_matrix[cls].get(ctx, 0)
            print(f"  {count:>20d}", end="")
        print()

    # --- class-context correlation matrix (train) ---
    print("\n--- Train class-context correlation (normalised) ---")
    corr_rows = []
    for cls in classes:
        total = sum(train_matrix[cls].values())
        row_data = {"class": cls}
        print(f"{cls:>12s}", end="")
        for ctx in contexts[:8]:
            count = train_matrix[cls].get(ctx, 0)
            prop = count / total if total > 0 else 0.0
            row_data[ctx] = round(prop, 4)
            print(f"  {prop:>20.3f}", end="")
        print()
        corr_rows.append(row_data)

    # --- Save statistics CSV ---
    stats_path = ANALYSIS_DIR / "dataset_statistics.csv"
    all_groups = sorted(group_counts.keys())
    with stats_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["group", "class", "context", "train_count", "val_count",
             "test_count", "total_count", "train_proportion"]
        )
        for group_key in all_groups:
            cls_name, ctx_name = group_key.split("×")
            t = group_counts[group_key].get("train", 0)
            v = group_counts[group_key].get("val", 0)
            te = group_counts[group_key].get("test", 0)
            total = t + v + te
            train_prop = t / total if total > 0 else 0.0
            writer.writerow(
                [group_key, cls_name, ctx_name, t, v, te, total,
                 round(train_prop, 4)]
            )
    print(f"\nStatistics CSV: {stats_path}")

    # --- Save summary JSON ---
    summary = {
        "total_images": len(rows),
        "classes": classes,
        "n_contexts": n_contexts,
        "contexts": contexts,
        "split_counts": dict(split_counts),
        "unique_groups": len(all_groups),
    }
    summary_path = ANALYSIS_DIR / "dataset_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Summary JSON: {summary_path}")

    # --- Print key finding ---
    print("\n" + "=" * 72)
    print("KEY FINDING: Context-Shift Structure")
    print("=" * 72)
    for cls in classes:
        train_contexts = {
            ctx for ctx, cnt in train_matrix[cls].items() if cnt > 0
        }
        test_contexts = {
            ctx for ctx, cnt in test_matrix[cls].items() if cnt > 0
        }
        new_in_test = test_contexts - train_contexts
        print(f"\n  {cls}:")
        print(f"    Train contexts: {len(train_contexts)}")
        print(f"    Test contexts:  {len(test_contexts)}")
        print(f"    New in test (shift): {len(new_in_test)}")
        if new_in_test:
            sample = sorted(new_in_test)[:10]
            print(f"    Examples: {sample}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Download MetaShift Cat vs Dog and build SPUME metadata."
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DATASET_DIR / "hf_cache",
        help="HuggingFace cache directory.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for splits."
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip HF download if images already exist.",
    )
    args = parser.parse_args()

    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Download
    if args.skip_download and IMAGE_DIR.exists() and any(IMAGE_DIR.iterdir()):
        print("Images already exist, skipping download.")
        print("(Loading from disk for metadata rebuild is not implemented;")
        print(" use this flag only when re-running the same script.)")
        return
    else:
        records = download_dataset(args.cache_dir)

    # Step 2: Analyse contexts
    ctx_info = analyse_contexts(records)

    # Step 3: Assign splits with context shift
    records = assign_splits(records, ctx_info, seed=args.seed)

    # Step 4: Save images & build metadata
    rows = save_images_and_build_metadata(records)

    # Step 5: Write metadata CSV
    write_metadata(rows)

    # Step 6: Statistics
    all_contexts = sorted({r["context"] for r in records})
    compute_statistics(rows, len(all_contexts))

    print("\n" + "=" * 72)
    print("DONE — MetaShift Cat vs Dog is ready for SPUME.")
    print(f"  Metadata: {DATASET_DIR / 'metadata_metashift_catdog.csv'}")
    print(f"  Images:   {IMAGE_DIR}")
    print(f"  Stats:    {ANALYSIS_DIR / 'dataset_statistics.csv'}")
    print("=" * 72)


if __name__ == "__main__":
    main()
