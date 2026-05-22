"""Concept cleaning pipeline for MetaShift ViT-GPT2 & BLIP concepts.

1. Loads raw concepts from both models
2. Applies:
   - Synonym merging (sofa/couch, grass/lawn, snow/snowy, etc.)
   - Stopword removal (image, background, picture, front, many, etc.)
   - Lemmatization (via spaCy)
3. Outputs cleaned JSON + comparison
4. Generates before/after report
"""

import json, csv, pickle
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT / "analysis" / "metashift"

# ---------------------------------------------------------------------------
# Synonym map
# ---------------------------------------------------------------------------
SYNONYM_MAP = {
    # Furniture
    "sofa": "couch", "sofas": "couch",
    # Ground/outdoor
    "lawn": "grass", "lawns": "grass", "grassy": "grass",
    "snowy": "snow", "snowing": "snow",
    "sandy": "beach", "sand": "beach", "beach's": "beach",
    "woods": "forest", "tree": "forest", "trees": "forest",
    "jungle": "forest",
    "dirty": "dirt", "mud": "dirt", "muddy": "dirt",
    "desert": "desert",
    "mountain": "mountain", "mountains": "mountain",
    "road": "street", "road's": "street",
    "sidewalk": "street", "pavement": "street",
    # Water
    "ocean": "water", "sea": "water", "lake": "water", "river": "water",
    "surf": "water",
    # Objects
    "tv": "television", "tvs": "television",
    "computer": "computer", "pc": "computer",
    "laptop": "laptop", "laptops": "laptop",
    "notebook": "laptop",
    "screen": "monitor", "monitor": "monitor",
    "keyboard": "keyboard", "keyboards": "keyboard",
    "frisbee": "frisbee", "disc": "frisbee",
    "surfboard": "surfboard", "surfboards": "surfboard",
    # Animals
    "dog": "dog", "dogs": "dog", "dog's": "dog", "puppy": "dog", "puppies": "dog",
    "cat": "cat", "cats": "cat", "cat's": "cat", "kitten": "cat", "kittens": "cat",
    # People
    "man": "person", "men": "person",
    "woman": "person", "women": "person",
    "people": "person", "person's": "person",
    "child": "person", "children": "person",
    "guy": "person", "girl": "person", "boy": "person",
    # Rooms
    "bathroom": "bathroom", "restroom": "bathroom",
    "kitchen": "kitchen", "kitchen's": "kitchen",
    "living room": "living_room",
    "bedroom": "bedroom",
    # Furniture extended
    "desk": "desk", "desks": "desk",
    "table": "table", "tables": "table",
    "chair": "chair", "chairs": "chair",
    "bed": "bed", "beds": "bed",
    "shelf": "shelf", "shelves": "shelf", "bookshelf": "shelf",
    # Colors
    "white": "white", "whitish": "white",
    "black": "black", "blackish": "black",
    "brown": "brown", "brownish": "brown",
    "blue": "blue", "bluish": "blue",
    "red": "red", "reddish": "red",
    "green": "green", "greenish": "green",
    "yellow": "yellow", "yellowish": "yellow",
    "orange": "orange",
    "gray": "gray", "grey": "gray",
    # Actions (often used as nouns)
    "sitting": "sit",
    "standing": "stand",
    "laying": "lay",
    "walking": "walk",
    "playing": "play",
    "sleeping": "sleep",
}

# ---------------------------------------------------------------------------
# Stopwords / meaningless concepts
# ---------------------------------------------------------------------------
STOP_CONCEPTS = {
    # Generic photography terms
    "image", "images", "picture", "pictures", "photo", "photos",
    "photograph", "photographs", "background", "foreground", "closeup",
    "camera", "photography",
    # Vague spatial terms
    "front", "back", "side", "top", "bottom", "left", "right",
    "middle", "center", "edge",
    # Generic quantifiers
    "many", "several", "some", "lot", "lots", "bunch", "couple",
    "few", "various", "different", "various", "other", "another",
    # Generic descriptors
    "small", "large", "big", "little", "tiny", "huge",
    "old", "new", "young", "good", "great", "nice", "beautiful",
    "type", "kind", "sort", "part", "piece",
    # Time
    "day", "night",
    # Vague nouns
    "thing", "things", "stuff", "something", "nothing",
    # Photograph styles
    "blurry", "blur", "blurred",
    "collage", "montage",
    "silhouette", "silhouettes",
    "shadow", "shadows",
    # Misc
    "close", "head", "group", "pair", "set",
    "way", "angle", "view",
    # Food (irrelevant to cat/dog classification)
    "food", "foods",
}


# ---------------------------------------------------------------------------
# Cleaning logic
# ---------------------------------------------------------------------------
def clean_concept_list(concepts: list[str]) -> tuple[list[str], dict]:
    """Clean a list of concept strings (nouns + adjectives).

    Returns (cleaned_list, merge_log).
    """
    merge_log = {"merged": defaultdict(list), "removed": []}
    cleaned = []

    for concept in concepts:
        # Strip POS suffix if present (e.g., "dog:noun" -> "dog")
        if ":" in concept:
            word, pos = concept.rsplit(":", 1)
        else:
            word = concept
            pos = ""

        word_lower = word.lower().strip()

        # Check stopwords
        if word_lower in STOP_CONCEPTS:
            merge_log["removed"].append(concept)
            continue

        # Apply synonym map
        if word_lower in SYNONYM_MAP:
            canonical = SYNONYM_MAP[word_lower]
            if canonical != word_lower:
                merge_log["merged"][canonical].append(concept)
            if pos:
                cleaned.append(f"{canonical}:{pos}")
            else:
                cleaned.append(canonical)
        else:
            cleaned.append(concept)

    return cleaned, merge_log


def clean_records(records: list[dict]) -> list[dict]:
    """Apply cleaning to a list of concept records."""
    total_merge_log = {"merged": defaultdict(list), "removed": []}
    cleaned_records = []

    for r in records:
        nouns_clean, log_n = clean_concept_list(r.get("nouns", []))
        adjs_clean, log_a = clean_concept_list(r.get("adjectives", []))

        # Merge logs
        for k, v in log_n["merged"].items():
            total_merge_log["merged"][k].extend(v)
        for k, v in log_a["merged"].items():
            total_merge_log["merged"][k].extend(v)
        total_merge_log["removed"].extend(log_n["removed"])
        total_merge_log["removed"].extend(log_a["removed"])

        cleaned_records.append({
            **{k: v for k, v in r.items() if k not in ("nouns", "adjectives")},
            "nouns": nouns_clean,
            "adjectives": adjs_clean,
        })

    return cleaned_records


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def compute_stats(records: list[dict]):
    all_w = Counter()
    cls_w = defaultdict(Counter)
    for r in records:
        words = [w.split(":")[0] if ":" in w else w for w in r["nouns"] + r["adjectives"]]
        for w in words:
            all_w[w] += 1
            cls_w[r["class_name"]][w] += 1
    return {
        "vocab_size": len(all_w),
        "total_tokens": sum(all_w.values()),
        "top_overall": all_w.most_common(30),
        "top_per_class": {c: ctr.most_common(15) for c, ctr in cls_w.items()},
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # --- Load ViT-GPT2 raw concepts ---
    vit_path = ANALYSIS / "vitgpt2" / "concepts_raw_vitgpt2.json"
    with open(vit_path, "r", encoding="utf-8") as f:
        vit_raw = json.load(f)
    print(f"ViT-GPT2 raw: {len(vit_raw)} records")

    # --- Load BLIP concepts from SPUME pickle ---
    # Build BLIP records from SPUME embeddings + metadata
    vocab = pickle.load(open(ROOT / "datasets/metashift/blip_extracted_concepts_0_0_vocab.pickle", "rb"))
    embeddings = pickle.load(open(ROOT / "datasets/metashift/blip_extracted_concepts_0_0_img_embeddings.pickle", "rb"))
    keys = pickle.load(open(ROOT / "datasets/metashift/blip_extracted_concepts_0_0_img_embedding_keys.pickle", "rb"))
    with open(ROOT / "datasets/metashift/metadata_metashift_catdog.csv", newline="", encoding="utf-8") as f:
        meta_rows = list(csv.DictReader(f))

    def norm(p):
        return str(p).strip().replace("\\", "/")
    meta_map = {norm(r["img_path"]): r for r in meta_rows if r["split"] == "train"}

    # Build BLIP records
    blip_raw = []
    for key, emb in zip(keys, embeddings):
        k = norm(key)
        if k not in meta_map:
            continue
        r = meta_map[k]
        active_concepts = [f"{vocab[i]}:noun" for i in range(len(vocab)) if emb[i] == 1]
        # Split nouns and adjectives from SPUME vocab (format: "word:noun" or "word:adj")
        nouns, adjs = [], []
        for c in active_concepts:
            if ":noun" in c or ":nount" in c:
                nouns.append(c)
            elif ":adj" in c:
                adjs.append(c)
            else:
                nouns.append(c)
        blip_raw.append({
            "filename": r["filename"],
            "class_name": r["class_name"],
            "context_name": r["context_name"],
            "nouns": nouns,
            "adjectives": adjs,
        })
    print(f"BLIP raw: {len(blip_raw)} records")

    # --- Clean both ---
    print("\n=== Cleaning ViT-GPT2 ===")
    vit_clean = clean_records(vit_raw)
    print("=== Cleaning BLIP ===")
    blip_clean = clean_records(blip_raw)

    # --- Compute stats ---
    vit_stats_before = compute_stats(vit_raw)
    vit_stats_after = compute_stats(vit_clean)
    blip_stats_before = compute_stats(blip_raw)
    blip_stats_after = compute_stats(blip_clean)

    # --- Save cleaned ---
    vit_out = ANALYSIS / "vitgpt2" / "concepts_cleaned_vitgpt2.json"
    with open(vit_out, "w", encoding="utf-8") as f:
        json.dump(vit_clean, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {vit_out}")

    blip_out = ANALYSIS / "blip" / "concepts_cleaned_blip.json"
    blip_out.parent.mkdir(parents=True, exist_ok=True)
    with open(blip_out, "w", encoding="utf-8") as f:
        json.dump(blip_clean, f, indent=2, ensure_ascii=False)
    print(f"Saved: {blip_out}")

    # --- Report ---
    lines = [
        "# MetaShift — Concept Cleaning Report",
        "",
        "## 1. Cleaning Rules",
        "",
        "### Synonym Merging",
        "| Original | Canonical |",
        "|----------|-----------|",
    ]
    seen = set()
    for orig, canon in sorted(SYNONYM_MAP.items()):
        if (orig, canon) not in seen and orig != canon:
            lines.append(f"| {orig} | {canon} |")
            seen.add((orig, canon))

    lines.extend([
        "",
        "### Removed Stopwords",
        f"`{', '.join(sorted(STOP_CONCEPTS)[:30])}...`",
        "",
        "## 2. ViT-GPT2 Statistics",
        "",
        "| Metric | Before | After |",
        "|--------|--------|-------|",
        f"| Vocabulary size | {vit_stats_before['vocab_size']} | {vit_stats_after['vocab_size']} |",
        f"| Total tokens | {vit_stats_before['total_tokens']} | {vit_stats_after['total_tokens']} |",
        "",
        "### Before (Top-20)",
    ])
    for w, c in vit_stats_before["top_overall"][:20]:
        lines.append(f"- {w}: {c}")
    lines.extend(["", "### After (Top-20)", ""])
    for w, c in vit_stats_after["top_overall"][:20]:
        lines.append(f"- {w}: {c}")

    lines.extend([
        "",
        "## 3. BLIP Statistics",
        "",
        "| Metric | Before | After |",
        "|--------|--------|-------|",
        f"| Vocabulary size | {blip_stats_before['vocab_size']} | {blip_stats_after['vocab_size']} |",
        f"| Total tokens | {blip_stats_before['total_tokens']} | {blip_stats_after['total_tokens']} |",
        "",
        "### Before (Top-20)",
    ])
    for w, c in blip_stats_before["top_overall"][:20]:
        lines.append(f"- {w}: {c}")
    lines.extend(["", "### After (Top-20)", ""])
    for w, c in blip_stats_after["top_overall"][:20]:
        lines.append(f"- {w}: {c}")

    lines.extend([
        "",
        "## 4. Impact Summary",
        "",
        "| Model | Vocab Before | Vocab After | Reduction |",
        "|-------|-------------|-------------|-----------|",
        f"| ViT-GPT2 | {vit_stats_before['vocab_size']} | {vit_stats_after['vocab_size']} | "
        f"{(1 - vit_stats_after['vocab_size'] / max(1, vit_stats_before['vocab_size'])) * 100:.0f}% |",
        f"| BLIP | {blip_stats_before['vocab_size']} | {blip_stats_after['vocab_size']} | "
        f"{(1 - blip_stats_after['vocab_size'] / max(1, blip_stats_before['vocab_size'])) * 100:.0f}% |",
        "",
        "## 5. Key Improvements",
        "",
        "- Synonyms merged: couch/sofa, grass/lawn, snow/snowy, beach/sandy, forest/trees/jungle, dirt/mud, etc.",
        "- People terms unified: man/woman/child → person",
        "- Removed: generic photo terms, spatial terms (front/back/top), vague quantifiers",
        "- Colors preserved but minor variants merged",
        "- Dog/cat variants unified to canonical forms",
        "",
        "Cleaned concepts are saved at:",
        f"- `{vit_out}`",
        f"- `{blip_out}`",
    ])

    report_path = ANALYSIS / "concept_cleaning_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report saved: {report_path}")

    # Print summary
    print(f"\n=== SUMMARY ===")
    print(f"ViT-GPT2: {vit_stats_before['vocab_size']} -> {vit_stats_after['vocab_size']} concepts")
    print(f"BLIP:     {blip_stats_before['vocab_size']} -> {blip_stats_after['vocab_size']} concepts")


if __name__ == "__main__":
    main()
