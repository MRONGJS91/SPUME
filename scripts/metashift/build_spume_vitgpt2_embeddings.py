"""Build SPUME-compatible concept embeddings from cleaned ViT-GPT2 concepts."""
import json, pickle, csv
from collections import Counter
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "datasets" / "metashift"
ANALYSIS = ROOT / "analysis" / "metashift"

# --- Load cleaned concepts ---
with open(ANALYSIS / "vitgpt2" / "concepts_cleaned_vitgpt2.json", encoding="utf-8") as f:
    records = json.load(f)
print(f"Loaded {len(records)} cleaned concept records")

# --- Build concept vocabulary (freq >= 5) ---
counter = Counter()
for r in records:
    for w in r["nouns"] + r["adjectives"]:
        word = w.split(":")[0] if ":" in w else w
        counter[word] += 1

vocab = sorted([w for w, c in counter.items() if c >= 5])
vocab_path = DATASET / "vitgpt2_cleaned_vocab.pickle"
with open(vocab_path, "wb") as f:
    pickle.dump(vocab, f)
print(f"Vocab: {len(vocab)} concepts -> {vocab_path}")

# --- Build concept embeddings ---
concept_to_idx = {w: i for i, w in enumerate(vocab)}
embeddings = np.zeros((len(records), len(vocab)), dtype=np.uint8)
for i, r in enumerate(records):
    for w in r["nouns"] + r["adjectives"]:
        word = w.split(":")[0] if ":" in w else w
        if word in concept_to_idx:
            embeddings[i, concept_to_idx[word]] = 1

emb_path = DATASET / "vitgpt2_cleaned_img_embeddings.pickle"
with open(emb_path, "wb") as f:
    pickle.dump(embeddings, f)
print(f"Embeddings: {embeddings.shape} -> {emb_path}")

# --- Build key file (match metadata img_path format) ---
with open(DATASET / "metadata_metashift_catdog.csv", newline="", encoding="utf-8") as f:
    meta_rows = list(csv.DictReader(f))
train_meta = {r["filename"]: r["img_path"] for r in meta_rows if r["split"] == "train"}

keys = []
valid_embeddings = []
for r in records:
    fn = r["filename"]
    if fn in train_meta:
        keys.append(train_meta[fn])
        valid_embeddings.append(embeddings[len(keys)-1])
    else:
        print(f"  WARNING: {fn} not found in train metadata")

embeddings = np.array(valid_embeddings, dtype=np.uint8)

key_path = DATASET / "vitgpt2_cleaned_img_embedding_keys.pickle"
with open(key_path, "wb") as f:
    pickle.dump(keys, f)
print(f"Keys: {len(keys)} -> {key_path}")

print("\nDone. SPUME ViT-GPT2 embedding files ready.")
