#!/usr/bin/env python3
"""Build MetaShift Cat vs Dog dataset from locally downloaded ZIP files.

Requires
--------
- datasets/metashift/downloads/images.zip    (Visual Genome GQA images, ~20 GB)
- datasets/metashift/downloads/sceneGraphs.zip
- datasets/metashift/full-candidate-subsets.pkl  (metadata, already downloaded)

Output
------
- datasets/metashift/images/               cat/dog images only (~19K files)
- datasets/metashift/metadata_metashift_catdog.csv   SPUME-compatible metadata
- analysis/metashift/dataset_statistics.csv
"""

import csv
import json
import os
import pickle
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "datasets" / "metashift"
DOWNLOADS_DIR = DATASET_DIR / "downloads"
IMAGES_ZIP = DOWNLOADS_DIR / "images.zip"
SCENE_GRAPHS_ZIP = DOWNLOADS_DIR / "sceneGraphs.zip"
PICKLE_PATH = DATASET_DIR / "full-candidate-subsets.pkl"
IMAGE_DIR = DATASET_DIR / "images"
ANALYSIS_DIR = PROJECT_ROOT / "analysis" / "metashift"

SELECTED_CLASSES = ["cat", "dog"]


# ---------------------------------------------------------------------------
# Step 1: Collect needed image IDs
# ---------------------------------------------------------------------------
def collect_image_ids() -> dict[str, set[str]]:
    """Parse pickle to get cat/dog image IDs grouped by (class, context)."""
    with PICKLE_PATH.open("rb") as f:
        data = pickle.load(f)

    result: dict[str, dict[str, set[str]]] = {c: {} for c in SELECTED_CLASSES}
    for key, img_ids in data.items():
        for cls in SELECTED_CLASSES:
            if key.startswith(f"{cls}(") and key.endswith(")"):
                context = key[len(cls) + 1 : -1]
                result[cls][context] = img_ids
                break

    for cls in SELECTED_CLASSES:
        n_ctx = len(result[cls])
        n_img = sum(len(v) for v in result[cls].values())
        print(f"  {cls}: {n_ctx} contexts, {n_img:,} unique image IDs")
    return result


# ---------------------------------------------------------------------------
# Step 2: Extract images
# ---------------------------------------------------------------------------
def extract_images(class_data: dict[str, dict[str, set[str]]]) -> None:
    """Extract only cat/dog images from images.zip in a single pass."""
    needed_ids: set[str] = set()
    for cls_data in class_data.values():
        for img_ids in cls_data.values():
            needed_ids.update(img_ids)
    print(f"\n  Total unique image IDs needed: {len(needed_ids):,}")

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    existing = set(f.stem for f in IMAGE_DIR.glob("*.jpg"))
    missing_ids = needed_ids - existing
    print(f"  Already extracted: {len(existing):,}")
    print(f"  Remaining to extract: {len(missing_ids):,}")

    if not missing_ids:
        print("  All images already extracted.")
        return

    # Single pass through the zip: iterate entries, extract if needed
    found = 0
    with zipfile.ZipFile(IMAGES_ZIP, "r") as zf:
        for entry in tqdm(zf.infolist(), desc="  Scanning zip", total=len(zf.infolist())):
            # entry.filename = "images/2345671.jpg"
            name = Path(entry.filename).stem  # "2345671"
            if name in missing_ids:
                zf.extract(entry, IMAGE_DIR)
                src = IMAGE_DIR / entry.filename  # images/2345671.jpg
                dst = IMAGE_DIR / f"{name}.jpg"
                if src.exists():
                    src.rename(dst)
                found += 1
                if found >= len(missing_ids):
                    break  # all done

    # Clean up empty images/ subdirectory
    subdir = IMAGE_DIR / "images"
    if subdir.exists():
        try:
            subdir.rmdir()
        except OSError:
            pass

    print(f"  Extracted {found:,} new images.")
    not_found = len(missing_ids) - found
    if not_found > 0:
        print(f"  WARNING: {not_found:,} image IDs not found in zip.")


# ---------------------------------------------------------------------------
# Step 3: Build metadata
# ---------------------------------------------------------------------------
def build_metadata(class_data: dict[str, dict[str, set[str]]], seed: int = 42) -> list[dict]:
    """Build flat records and assign train/val/test splits with context shift."""
    rng = np.random.default_rng(seed)

    # Build records for images that exist on disk
    records = []
    missing = 0
    for cls, ctx_data in class_data.items():
        for ctx, img_ids in ctx_data.items():
            for img_id in img_ids:
                img_path = IMAGE_DIR / f"{img_id}.jpg"
                if img_path.exists():
                    records.append({
                        "class_name": cls,
                        "context_name": ctx,
                        "image_id": str(img_id),
                    })
                else:
                    missing += 1

    print(f"\n  Images on disk: {len(records):,}")
    if missing:
        print(f"  Missing (skipped): {missing:,}")

    # Per-class per-context counts
    class_ctx_counts = defaultdict(lambda: defaultdict(int))
    for r in records:
        class_ctx_counts[r["class_name"]][r["context_name"]] += 1

    all_contexts = sorted({r["context_name"] for r in records})
    print(f"  Unique contexts: {len(all_contexts)}")

    # Exclusive contexts (>80% belong to one class)
    exclusive = {c: set() for c in SELECTED_CLASSES}
    for ctx in all_contexts:
        total = sum(class_ctx_counts[cls].get(ctx, 0) for cls in SELECTED_CLASSES)
        if total < 5:
            continue
        for cls in SELECTED_CLASSES:
            if class_ctx_counts[cls].get(ctx, 0) / total > 0.8:
                exclusive[cls].add(ctx)
                break

    # Assign splits
    for r in records:
        cls = r["class_name"]
        ctx = r["context_name"]
        if ctx in exclusive.get(cls, set()):
            r["split"] = "train" if rng.random() < 0.8 else "val"
        else:
            r["split"] = "test"
        if r["split"] != "test" and rng.random() < 0.10:
            r["split"] = "test"

    # Context index mapping
    context_to_idx = {ctx: i for i, ctx in enumerate(all_contexts)}
    n_contexts = len(all_contexts)

    # Build final rows
    rows = []
    for r in records:
        filename = f"{r['image_id']}.jpg"
        img_path = f"images/{filename}"
        ctx_idx = context_to_idx[r["context_name"]]
        y = 0 if r["class_name"] == "cat" else 1
        group_id = y * n_contexts + ctx_idx
        rows.append({
            "img_path": img_path,
            "class_name": r["class_name"],
            "y": y,
            "split": r["split"],
            "env": r["context_name"],
            "filename": filename,
            "group_id": group_id,
            "context_name": r["context_name"],
            "context_label": ctx_idx,
            "image_id": r["image_id"],
        })

    # Stats
    split_counts = defaultdict(int)
    for r in rows:
        split_counts[r["split"]] += 1
    print(f"\n  Split sizes:")
    for sn in ["train", "val", "test"]:
        print(f"    {sn}: {split_counts[sn]:,}")

    return rows


# ---------------------------------------------------------------------------
# Step 4: Write outputs
# ---------------------------------------------------------------------------
def write_outputs(rows: list[dict]) -> None:
    """Write metadata CSV and dataset statistics."""
    # Metadata CSV
    csv_path = DATASET_DIR / "metadata_metashift_catdog.csv"
    fieldnames = [
        "img_path", "class_name", "y", "split", "env", "filename",
        "group_id", "context_name", "context_label", "image_id",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  Metadata CSV: {csv_path}  ({len(rows):,} rows)")

    # Statistics
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    SEP = "||"
    group_counts = defaultdict(lambda: defaultdict(int))
    for r in rows:
        group_counts[f"{r['class_name']}{SEP}{r['context_name']}"][r["split"]] += 1

    stats_path = ANALYSIS_DIR / "dataset_statistics.csv"
    all_groups = sorted(group_counts.keys())
    with stats_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["group", "class", "context", "train_count", "val_count",
                          "test_count", "total_count"])
        for g in all_groups:
            cls_name, ctx_name = g.split(SEP)
            t = group_counts[g].get("train", 0)
            v = group_counts[g].get("val", 0)
            te = group_counts[g].get("test", 0)
            writer.writerow([g, cls_name, ctx_name, t, v, te, t + v + te])
    print(f"  Statistics CSV: {stats_path}")

    # Summary JSON
    classes = sorted({r["class_name"] for r in rows})
    contexts = sorted({r["context_name"] for r in rows})
    split_counts = defaultdict(int)
    for r in rows:
        split_counts[r["split"]] += 1
    summary = {
        "total_images": len(rows),
        "classes": classes,
        "n_contexts": len(contexts),
        "split_counts": dict(split_counts),
        "unique_groups": len(all_groups),
    }
    summary_path = ANALYSIS_DIR / "dataset_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  Summary JSON: {summary_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 72)
    print("MetaShift: Build Dataset from Local ZIP Files")
    print("=" * 72)

    # Verify inputs
    for p in [IMAGES_ZIP, SCENE_GRAPHS_ZIP, PICKLE_PATH]:
        if not p.exists():
            print(f"ERROR: Missing {p}")
            sys.exit(1)

    # Step 1: Collect image IDs
    print("\nStep 1: Collecting cat/dog image IDs from pickle ...")
    class_data = collect_image_ids()

    # Step 2: Extract images
    print("\nStep 2: Extracting cat/dog images from images.zip ...")
    print("  (This scans the 21GB zip and extracts ~19K files)")
    extract_images(class_data)

    # Step 3: Build metadata
    print("\nStep 3: Building metadata with train/val/test splits ...")
    rows = build_metadata(class_data)

    # Step 4: Write outputs
    print("\nStep 4: Writing outputs ...")
    write_outputs(rows)

    print("\n" + "=" * 72)
    print("DONE — MetaShift dataset is ready for SPUME.")
    print(f"  Metadata: {DATASET_DIR / 'metadata_metashift_catdog.csv'}")
    print(f"  Images:   {IMAGE_DIR}")
    print("=" * 72)


if __name__ == "__main__":
    main()
