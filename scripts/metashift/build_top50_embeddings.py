"""Build SPUME embeddings for top-50 filtered MetaShift dataset."""
import csv, json, pickle, sys
from collections import Counter
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
META_TOP50 = ROOT / "datasets/metashift/metadata_metashift_catdog_top50.csv"
DATASET = ROOT / "datasets" / "metashift"

# Load filtered metadata
meta_rows = list(csv.DictReader(open(META_TOP50)))
tv_rows = [r for r in meta_rows if r["split"] in ("train", "val")]
print(f"Train+val in top-50: {len(tv_rows)}")

# Load cleaned ViT-GPT2 concepts
# Build from existing cleaned concepts + val captions
with open(ROOT / "analysis/metashift/vitgpt2/concepts_cleaned_vitgpt2.json", encoding="utf-8") as f:
    vt_recs = json.load(f)
vt_by_fn = {r["filename"]: r for r in vt_recs}
print(f"Cleaned ViT-GPT2 records: {len(vt_by_fn)}")

# For val images not in cleaned concepts, extract from captions
captions_csv = DATASET / "vitgpt2_captions.csv"
captions = {}
if captions_csv.exists():
    for r in csv.DictReader(open(captions_csv)):
        captions[r["filename"]] = r["caption"]

# Use spaCy for missing
missing_val = [r for r in tv_rows if r["filename"] not in vt_by_fn]
if missing_val:
    print(f"Extracting concepts for {len(missing_val)} missing val images...")
    import spacy
    nlp = spacy.load("en_core_web_sm")
    sys.path.insert(0, str(ROOT / "scripts" / "metashift"))  # noqa
    from clean_concepts import clean_concept_list

    for r in missing_val:
        cap = captions.get(r["filename"], "")
        nouns, adjs = [], []
        if cap:
            for tok in nlp(cap):
                if tok.pos_ == "NOUN": nouns.append(f"{tok.lemma_.lower()}:noun")
                elif tok.pos_ == "ADJ": adjs.append(f"{tok.lemma_.lower()}:adj")
        nouns_c, _ = clean_concept_list(nouns)
        adjs_c, _ = clean_concept_list(adjs)
        vt_by_fn[r["filename"]] = {"filename": r["filename"], "class_name": r["class_name"],
                                    "context_name": r["context_name"], "nouns": nouns_c, "adjectives": adjs_c}

# Build vocab from all tv_rows
all_recs = []
for r in tv_rows:
    fn = r["filename"]
    if fn in vt_by_fn:
        all_recs.append(vt_by_fn[fn])
    else:
        all_recs.append({"filename": fn, "nouns": [], "adjectives": []})

counter = Counter()
for r in all_recs:
    for w in r["nouns"] + r["adjectives"]:
        word = w.split(":")[0] if ":" in w else w
        counter[word] += 1

vocab = sorted([w for w, c in counter.items() if c >= 5])
concept_to_idx = {w: i for i, w in enumerate(vocab)}
print(f"Vocab: {len(vocab)}")

# Build embeddings (aligned with metadata order: deduplicate by filename)
import sys
seen_fns = set()
embeddings = []
keys = []
for r in all_recs:
    fn = r["filename"]
    if fn in seen_fns:
        continue
    seen_fns.add(fn)
    emb = np.zeros(len(vocab), dtype=np.uint8)
    for w in r["nouns"] + r["adjectives"]:
        word = w.split(":")[0] if ":" in w else w
        if word in concept_to_idx:
            emb[concept_to_idx[word]] = 1
    embeddings.append(emb)
    # Find img_path from metadata
    for mr in tv_rows:
        if mr["filename"] == fn:
            keys.append(mr["img_path"])
            break

embeddings = np.array(embeddings, dtype=np.uint8)
print(f"Embeddings: {embeddings.shape}, Keys: {len(keys)}")

# Save
for name, data in [("vocab", vocab), ("img_embeddings", embeddings), ("img_embedding_keys", keys)]:
    p = DATASET / f"vitgpt2_cleaned_top50_{name}.pickle"
    with open(p, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved: {p}")

# Verify key overlap with metadata
def norm(p): return str(p).strip().replace("\\", "/")
meta_key_set = {norm(r["img_path"]) for r in tv_rows}
embed_key_set = {norm(k) for k in keys}
print(f"Overlap: {len(meta_key_set & embed_key_set)}/{len(meta_key_set)}")
