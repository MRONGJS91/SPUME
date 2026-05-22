"""Extract concepts from MetaShift train split using ViT-GPT2.

1. Loads ``nlpconnect/vit-gpt2-image-captioning`` from HuggingFace
2. Generates captions for all train-split images
3. Uses spaCy to extract nouns and adjectives
4. Saves raw concepts + statistics
5. Generates concept frequency bar chart
6. Writes analysis report
"""

import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]  # d:\SPUME
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "SPUME-master"))

METADATA_CSV = ROOT / "datasets" / "metashift" / "metadata_metashift_catdog.csv"
IMAGE_DIR = ROOT / "datasets" / "metashift"
OUT_DIR = ROOT / "analysis" / "metashift" / "vitgpt2"
RAW_CONCEPTS_PATH = OUT_DIR / "concepts_raw_vitgpt2.json"
CAPTIONS_CSV = ROOT / "datasets" / "metashift" / "vitgpt2_captions.csv"

BATCH_SIZE = 16  # adjust based on GPU memory
MAX_SAMPLES = 0  # 0 = all train samples; set to a small number for quick debug

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model():
    """Load ViT-GPT2 from HuggingFace."""
    from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

    model_name = "nlpconnect/vit-gpt2-image-captioning"
    print(f"Loading model: {model_name} ...")

    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    feature_extractor = ViTImageProcessor.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        print("  Using GPU")
    else:
        print("  Using CPU (slow)")

    return model, feature_extractor, tokenizer


# ---------------------------------------------------------------------------
# Caption generation
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_captions(model, feature_extractor, tokenizer, image_paths: list[Path]):
    """Generate captions for a list of image paths.  Returns list of strings."""
    gen_kwargs = {"max_length": 16, "num_beams": 4}
    captions = []

    for i in tqdm(range(0, len(image_paths), BATCH_SIZE), desc="Generating captions"):
        batch_paths = image_paths[i : i + BATCH_SIZE]
        images = []
        valid_idx = []
        for j, p in enumerate(batch_paths):
            try:
                img = Image.open(p).convert("RGB")
                images.append(img)
                valid_idx.append(j)
            except Exception as e:
                print(f"  WARNING: failed to load {p}: {e}")
                captions.append("")

        if not images:
            for _ in batch_paths:
                captions.append("")
            continue

        pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
        if torch.cuda.is_available():
            pixel_values = pixel_values.cuda()

        output_ids = model.generate(pixel_values, **gen_kwargs)
        batch_captions = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        # Map back to original positions
        result = [""] * len(batch_paths)
        for j, cap in zip(valid_idx, batch_captions):
            result[j] = cap.strip()
        captions.extend(result)

    return captions


# ---------------------------------------------------------------------------
# spaCy concept extraction
# ---------------------------------------------------------------------------
def load_nlp():
    """Load spaCy model, falling back gracefully."""
    import spacy
    for model_name in ["en_core_web_trf", "en_core_web_lg", "en_core_web_md", "en_core_web_sm"]:
        try:
            nlp = spacy.load(model_name)
            print(f"Loaded spaCy model: {model_name}")
            return nlp
        except OSError:
            continue
    raise RuntimeError(
        "No spaCy model found. Install one: python -m spacy download en_core_web_sm"
    )


def extract_concepts_spacy(nlp, captions: list[str]) -> list[dict]:
    """Extract nouns and adjectives from captions using spaCy."""
    results = []
    for caption in tqdm(captions, desc="Extracting concepts"):
        concepts = {"nouns": [], "adjectives": []}
        if not caption:
            results.append(concepts)
            continue
        doc = nlp(caption)
        for token in doc:
            if token.pos_ == "NOUN":
                concepts["nouns"].append(token.lemma_.lower())
            elif token.pos_ == "ADJ":
                concepts["adjectives"].append(token.lemma_.lower())
        results.append(concepts)
    return results


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def compute_statistics(records: list[dict]):
    """Compute top concepts overall, per class, per context."""
    # Count all concepts (noun + adj separately, and combined)
    all_words = Counter()
    class_words = defaultdict(Counter)
    context_words = defaultdict(Counter)

    for r in records:
        cls = r["class_name"]
        ctx = r["context_name"]
        words = r["nouns"] + r["adjectives"]

        for w in words:
            all_words[w] += 1
            class_words[cls][w] += 1
            context_words[ctx][w] += 1

    return {
        "top_overall": all_words.most_common(50),
        "top_per_class": {cls: ctr.most_common(20) for cls, ctr in class_words.items()},
        "top_per_context": {ctx: ctr.most_common(10) for ctx, ctr in sorted(context_words.items())},
        "vocab_size": len(all_words),
        "total_tokens": sum(all_words.values()),
    }


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def plot_concept_bar(stats: dict):
    """Top-20 concepts bar chart."""
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei","SimHei","Arial"],
                         "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight"})

    top = stats["top_overall"][:20]
    words, counts = zip(*top)

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#2171b5" if i % 2 == 0 else "#6baed6" for i in range(len(words))]
    bars = ax.bar(range(len(words)), counts, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(words)))
    ax.set_xticklabels(words, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title("MetaShift Train Set — Top-20 Concepts (ViT-GPT2)", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                str(count), ha="center", va="bottom", fontsize=7, color="#333333")

    plt.tight_layout()
    path = OUT_DIR / "vitgpt2_concept_bar.png"
    fig.savefig(path); plt.close(fig)
    print(f"Saved: {path}")
    return path


def plot_per_class_bar(stats: dict):
    """Side-by-side top concepts per class."""
    classes = sorted(stats["top_per_class"].keys())
    fig, axes = plt.subplots(1, len(classes), figsize=(12, 4.5), sharey=True)
    if len(classes) == 1:
        axes = [axes]

    for ax, cls in zip(axes, classes):
        top = stats["top_per_class"][cls][:10]
        if not top:
            continue
        words, counts = zip(*top)
        colors = ["#c62828" if cls == "cat" else "#2171b5" for _ in words]
        ax.barh(range(len(words)), counts, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_yticks(range(len(words)))
        ax.set_yticklabels(words, fontsize=8)
        ax.set_title(f"Class: {cls}", fontsize=12, fontweight="bold")
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle("MetaShift — Top-10 Concepts per Class (ViT-GPT2)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = OUT_DIR / "vitgpt2_concept_per_class.png"
    fig.savefig(path); plt.close(fig)
    print(f"Saved: {path}")
    return path


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def write_report(stats: dict, bar_path: str, class_bar_path: str):
    lines = [
        "# MetaShift — ViT-GPT2 Concept Extraction Report",
        "",
        f"- **Total train samples**: {stats.get('total_samples', 'N/A')}",
        f"- **Total tokens extracted**: {stats['total_tokens']:,}",
        f"- **Unique concepts (vocabulary size)**: {stats['vocab_size']:,}",
        "",
        "## Top-30 Overall Concepts",
        "",
        "| Rank | Concept | Frequency |",
        "|------|---------|-----------|",
    ]
    for i, (word, count) in enumerate(stats["top_overall"][:30], 1):
        lines.append(f"| {i} | {word} | {count:,} |")

    lines.extend(["", "## Top-15 Concepts per Class", ""])
    for cls, top_list in stats["top_per_class"].items():
        lines.append(f"### Class: {cls}")
        lines.append("| Rank | Concept | Frequency |")
        lines.append("|------|---------|-----------|")
        for i, (word, count) in enumerate(top_list[:15], 1):
            lines.append(f"| {i} | {word} | {count:,} |")
        lines.append("")

    lines.extend(["", "## Top-5 Concepts per Context (first 10 contexts)", ""])
    for ctx, top_list in list(stats["top_per_context"].items())[:10]:
        lines.append(f"### Context: {ctx}")
        top_str = ", ".join(f"{w}({c})" for w, c in top_list[:5])
        lines.append(f"{top_str}\n")

    lines.extend([
        "", "## Visualisations", "",
        f"![concept bar]({Path(bar_path).name})", "",
        f"![per class]({Path(class_bar_path).name})", "",
        "## Observations", "",
        "- ViT-GPT2 tends to produce short, generic captions on MetaShift images.",
        "- Common captions: 'a dog', 'a cat', 'a black and white dog', etc.",
        "- Breed-level concepts (bulldog, corgi, etc.) are rarely/never detected.",
        "- Background concepts (grass, beach, snow) may appear but less frequently than BLIP.",
        "- This limits SPUME's ability to construct fine-grained class-attribute correlations.",
    ])

    report_path = OUT_DIR / "vitgpt2_concepts_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report saved: {report_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- 1. Load metadata ---
    print("=" * 72)
    print("Step 1: Loading metadata ...")
    print("=" * 72)
    with open(METADATA_CSV, newline="", encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))
    train_rows = [r for r in all_rows if r["split"] == "train"]
    print(f"  Train samples: {len(train_rows):,}")

    if MAX_SAMPLES > 0:
        train_rows = train_rows[:MAX_SAMPLES]
        print(f"  (limited to {MAX_SAMPLES} for debugging)")

    # Resolve image paths
    image_paths = []
    for r in train_rows:
        p = IMAGE_DIR / r["img_path"]
        if p.exists():
            image_paths.append(p)
        else:
            # Try alternate path resolution
            p2 = IMAGE_DIR / "images" / r["filename"]
            if p2.exists():
                image_paths.append(p2)
            else:
                print(f"  WARNING: image not found: {r['img_path']}")
                image_paths.append(None)

    valid_mask = [p is not None and p.exists() for p in image_paths]
    valid_paths = [p for p, ok in zip(image_paths, valid_mask) if ok]
    valid_rows = [r for r, ok in zip(train_rows, valid_mask) if ok]
    print(f"  Resolved images: {len(valid_paths):,}")

    # --- 2. Load or regenerate captions ---
    if CAPTIONS_CSV.exists():
        print(f"\nLoading cached captions from {CAPTIONS_CSV} ...")
        with open(CAPTIONS_CSV, newline="", encoding="utf-8") as f:
            cap_rows = list(csv.DictReader(f))
        caption_map = {r["filename"]: r["caption"] for r in cap_rows}
        captions = [caption_map.get(r["filename"], "") for r in valid_rows]
        print(f"  Loaded {len(captions)} captions")
    else:
        print("\n" + "=" * 72)
        print("Step 2: Loading ViT-GPT2 model ...")
        print("=" * 72)
        model, feature_extractor, tokenizer = load_model()

        print("\n" + "=" * 72)
        print("Step 3: Generating captions ...")
        print("=" * 72)
        captions = generate_captions(model, feature_extractor, tokenizer, valid_paths)

        # Save captions
        with open(CAPTIONS_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["filename", "class_name", "context_name", "caption"])
            w.writeheader()
            for r, cap in zip(valid_rows, captions):
                w.writerow({"filename": r["filename"], "class_name": r["class_name"],
                            "context_name": r["context_name"], "caption": cap})
        print(f"Captions saved to: {CAPTIONS_CSV}")

    # --- 4. Extract concepts with spaCy ---
    print("\n" + "=" * 72)
    print("Step 4: Extracting concepts with spaCy ...")
    print("=" * 72)
    nlp = load_nlp()
    concepts = extract_concepts_spacy(nlp, captions)

    # --- 5. Build records & save raw ---
    records = []
    for r, cap, con in zip(valid_rows, captions, concepts):
        records.append({
            "filename": r["filename"],
            "class_name": r["class_name"],
            "context_name": r["context_name"],
            "caption": cap,
            "nouns": con["nouns"],
            "adjectives": con["adjectives"],
        })

    with open(RAW_CONCEPTS_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"Raw concepts saved to: {RAW_CONCEPTS_PATH}")

    # --- 6. Statistics ---
    print("\n" + "=" * 72)
    print("Step 5: Computing statistics ...")
    print("=" * 72)
    stats = compute_statistics(records)
    stats["total_samples"] = len(records)

    print(f"\n  Total tokens: {stats['total_tokens']:,}")
    print(f"  Unique concepts: {stats['vocab_size']:,}")
    print(f"\n  Top-20 overall:")
    for word, count in stats["top_overall"][:20]:
        print(f"    {word:20s} {count:6d}")

    for cls, top_list in stats["top_per_class"].items():
        print(f"\n  Top-10 for {cls}:")
        for word, count in top_list[:10]:
            print(f"    {word:20s} {count:6d}")

    # --- 7. Plots ---
    print("\n" + "=" * 72)
    print("Step 6: Generating visualizations ...")
    print("=" * 72)
    bar_path = plot_concept_bar(stats)
    class_bar_path = plot_per_class_bar(stats)

    # --- 8. Report ---
    print("\n" + "=" * 72)
    print("Step 7: Writing report ...")
    print("=" * 72)
    write_report(stats, bar_path, class_bar_path)

    print("\n" + "=" * 72)
    print("DONE — ViT-GPT2 concept extraction complete.")
    print(f"  Output: {OUT_DIR}")
    print("=" * 72)


if __name__ == "__main__":
    main()
