"""Build SPUME embeddings for train+val from cleaned ViT-GPT2 concepts + val extraction."""
import json, pickle, csv
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "datasets" / "metashift"
ANALYSIS = ROOT / "analysis" / "metashift"

# --- 1. Load existing cleaned train concepts ---
with open(ANALYSIS / "vitgpt2" / "concepts_cleaned_vitgpt2.json", encoding="utf-8") as f:
    train_records = json.load(f)
print(f"Train records: {len(train_records)}")

# --- 2. Generate val concepts with quick spaCy extraction ---
# Load existing ViT-GPT2 captions for val
captions_path = DATASET / "vitgpt2_captions.csv"
train_val_captions = {}
with open(captions_path, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        train_val_captions[row["filename"]] = row["caption"]
print(f"Cached captions: {len(train_val_captions)}")

# Load metadata for train+val
with open(DATASET / "metadata_metashift_catdog.csv", newline="", encoding="utf-8") as f:
    meta_rows = list(csv.DictReader(f))
tv_meta = [r for r in meta_rows if r["split"] in ("train", "val")]
print(f"Train+val metadata: {len(tv_meta)}")

# Check which val images don't have captions
missing_val = [r for r in tv_meta if r["split"] == "val" and r["filename"] not in train_val_captions]
print(f"Val images without captions: {len(missing_val)}/{len([r for r in tv_meta if r['split']=='val'])}")

if missing_val and len(missing_val) > 10:
    print("Need to generate val captions. Running ViT-GPT2 on val images...")
    import torch
    from PIL import Image
    from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
    from tqdm import tqdm

    model_name = "nlpconnect/vit-gpt2-image-captioning"
    model = VisionEncoderDecoderModel.from_pretrained(model_name).eval().cuda()
    fe = ViTImageProcessor.from_pretrained(model_name)
    tok = AutoTokenizer.from_pretrained(model_name)

    val_only = [r for r in tv_meta if r["split"] == "val" and r["filename"] not in train_val_captions]
    for r in tqdm(val_only, desc="Val captions"):
        p = DATASET / r["img_path"]
        if not p.exists():
            p = DATASET / "images" / r["filename"]
        try:
            img = Image.open(p).convert("RGB")
            pixel = fe(images=img, return_tensors="pt").pixel_values.cuda()
            out = model.generate(pixel, max_length=16, num_beams=4)
            cap = tok.decode(out[0], skip_special_tokens=True).strip()
            train_val_captions[r["filename"]] = cap
        except Exception as e:
            train_val_captions[r["filename"]] = ""

    # Update captions CSV
    with open(captions_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "class_name", "context_name", "caption"])
        w.writeheader()
        for r in tv_meta:
            if r["filename"] in train_val_captions:
                w.writerow({"filename": r["filename"], "class_name": r["class_name"],
                           "context_name": r["context_name"], "caption": train_val_captions[r["filename"]]})
    print(f"Updated captions: {len(train_val_captions)}")

# --- 3. Extract concepts from all train+val captions using spaCy ---
import spacy
nlp = None
for m in ["en_core_web_sm", "en_core_web_md", "en_core_web_lg"]:
    try: nlp = spacy.load(m); break
    except OSError: continue

# Load existing train concepts first (already cleaned)
rec_map = {}
for rec in train_records:
    rec_map[rec["filename"]] = rec

# Extract val concepts using the SAME cleaning as train
sys.path.insert(0, str(ROOT / "scripts" / "metashift"))
from clean_concepts import clean_concept_list

all_records = []
for r in tv_meta:
    fn = r["filename"]
    if fn in rec_map:
        all_records.append(rec_map[fn])
    else:
        cap = train_val_captions.get(fn, "")
        nouns, adjs = [], []
        if cap and nlp:
            doc = nlp(cap)
            for tok in doc:
                if tok.pos_ == "NOUN":
                    nouns.append(f"{tok.lemma_.lower()}:noun")
                elif tok.pos_ == "ADJ":
                    adjs.append(f"{tok.lemma_.lower()}:adj")
        # Apply cleaning
        nouns_clean, _ = clean_concept_list(nouns)
        adjs_clean, _ = clean_concept_list(adjs)
        all_records.append({
            "filename": fn,
            "class_name": r["class_name"],
            "context_name": r.get("context_name", r["env"]),
            "nouns": nouns_clean,
            "adjectives": adjs_clean,
        })

print(f"All records (train+val): {len(all_records)}")

# --- 4. Build vocab and embeddings ---
counter = Counter()
for r in all_records:
    for w in r["nouns"] + r["adjectives"]:
        word = w.split(":")[0] if ":" in w else w
        counter[word] += 1

vocab = sorted([w for w, c in counter.items() if c >= 5])
concept_to_idx = {w: i for i, w in enumerate(vocab)}
print(f"Vocab: {len(vocab)}")

embeddings = []
keys = []
img_key_map = {r["filename"]: r["img_path"] for r in tv_meta}

for r in all_records:
    fn = r["filename"]
    if fn in img_key_map:
        emb = [0] * len(vocab)
        for w in r["nouns"] + r["adjectives"]:
            word = w.split(":")[0] if ":" in w else w
            if word in concept_to_idx:
                emb[concept_to_idx[word]] = 1
        embeddings.append(emb)
        keys.append(img_key_map[fn])

import numpy as np
embeddings = np.array(embeddings, dtype=np.uint8)
print(f"Embeddings: {embeddings.shape}")

# Save
vocab_path = DATASET / "vitgpt2_cleaned_vocab.pickle"
emb_path = DATASET / "vitgpt2_cleaned_img_embeddings.pickle"
key_path = DATASET / "vitgpt2_cleaned_img_embedding_keys.pickle"

with open(vocab_path, "wb") as f: pickle.dump(vocab, f)
with open(emb_path, "wb") as f: pickle.dump(embeddings, f)
with open(key_path, "wb") as f: pickle.dump(keys, f)

print(f"Saved: {vocab_path} ({len(vocab)} concepts)")
print(f"Saved: {emb_path} ({embeddings.shape})")
print(f"Saved: {key_path} ({len(keys)} keys)")
print("Done. SPUME embeddings for train+val ready.")
