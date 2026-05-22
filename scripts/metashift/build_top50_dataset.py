"""Build MetaShift dataset filtered to top-50 most frequent train contexts."""
import csv, json, pickle
from collections import Counter
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
META_SRC = ROOT / "datasets/metashift/metadata_metashift_catdog.csv"
META_OUT = ROOT / "datasets/metashift/metadata_metashift_catdog_top50.csv"

# 1. Find top-50 contexts from train
rows = list(csv.DictReader(open(META_SRC)))
train_rows = [r for r in rows if r["split"] == "train"]
ctx_freq = Counter(r["context_name"] for r in train_rows)
top50 = {c for c, _ in ctx_freq.most_common(50)}
print(f"Top-50 contexts (from {len(ctx_freq)} total)")

# 2. Filter rows
filtered = [r for r in rows if r["context_name"] in top50]
split_counts = Counter(r["split"] for r in filtered)
print(f"Filtered samples: {len(filtered)}/{len(rows)} ({len(filtered)/len(rows):.1%})")
for sn in ["train", "val", "test"]:
    print(f"  {sn}: {split_counts[sn]}")

# 3. Remap context labels and group IDs
all_ctx = sorted(top50)
ctx_to_idx = {ctx: i for i, ctx in enumerate(all_ctx)}
n_ctx = len(all_ctx)

for r in filtered:
    r["context_label"] = str(ctx_to_idx[r["context_name"]])
    y = int(r["y"])
    r["group_id"] = str(y * n_ctx + ctx_to_idx[r["context_name"]])

# 4. Save filtered metadata
fieldnames = list(filtered[0].keys())
with open(META_OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(filtered)
print(f"\nSaved: {META_OUT}")
print(f"  {len(filtered)} rows, {n_ctx} contexts, {len(set(r['group_id'] for r in filtered))} groups")

# 5. Summary
print("\nContexts:")
for i, ctx in enumerate(all_ctx):
    cat_n = sum(1 for r in filtered if r["class_name"]=="cat" and r["context_name"]==ctx)
    dog_n = sum(1 for r in filtered if r["class_name"]=="dog" and r["context_name"]==ctx)
    print(f"  {i:2d}. {ctx:25s} total={cat_n+dog_n:5d} cat={cat_n:4d} dog={dog_n:4d}")
