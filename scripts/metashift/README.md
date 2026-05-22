# MetaShift Scripts

## Files

| Script | Purpose |
|--------|---------|
| `download_and_build_metashift.py` | **Main entry point.** Downloads cat/dog subset from HuggingFace, builds context-shift train/val/test splits, saves images, writes metadata.csv + statistics |
| `build_metashift_metadata.py` | Scan an existing MetaShift directory tree (class/context/*.jpg) and generate metadata CSV. Use if you already have the images from another source |
| `analyse_metashift_distribution.py` | (to add) class-context distribution analysis for an existing metadata CSV |

## Quick Start (recommended)

```bash
# One command: download + split + build metadata
python scripts/metashift/download_and_build_metashift.py

# Then extract concepts
python extract_concepts.py --dataset metashift_catdog --model blip

# Train ERM baseline
python pretrain.py --config configs/metashift/metashift_catdog_erm_baseline.yaml

# Train Mixup baseline
python pretrain.py --config configs/metashift/metashift_catdog_mixup_baseline.yaml

# Train SPUME
python train_meta_spurious.py --config configs/metashift/metashift_catdog_blip_spume_full.yaml
```

## Context-Shift Design

Train/val/test splits are constructed to create a **genuine spurious correlation shift**:

```
Train:  cat → {sink, bed, couch, ...} (indoor/exclusive contexts)
        dog → {grass, fence, car, ...} (outdoor/exclusive contexts)
        → Model learns: cat↔indoor, dog↔outdoor

Test:   cat → ALL contexts (including grass, fence, car)
        dog → ALL contexts (including sink, bed, couch)
        → Spurious correlation is BROKEN
```

This is exactly the scenario SPUME is designed to handle and is fundamentally different from Spawrious O2O (where train/test distributions are identical).

## Metadata CSV Columns

| Column | Description |
|--------|-------------|
| `img_path` | Relative path to image file |
| `class_name` | "cat" or "dog" |
| `y` | Integer label (0=cat, 1=dog) |
| `split` | train / val / test |
| `env` | Context name (e.g. "grass", "couch") |
| `filename` | Image filename on disk |
| `group_id` | class × context (y * n_contexts + context_label) |
| `context_name` | Same as env, explicit |
| `context_label` | Integer context index |
| `image_id` | Original Visual Genome / COCO image ID |
