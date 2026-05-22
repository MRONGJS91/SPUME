import pickle
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


def tokenize_context(place_filename):
    tokens = []
    for tok in re.split(r"[/_\\.\-]+", str(place_filename).lower()):
        if not tok or tok.isdigit() or tok in {"jpg", "natural"}:
            continue
        if len(tok) == 1:
            continue
        tokens.append(tok)
    if any(tok in {"ocean", "lake", "water"} for tok in tokens):
        tokens.append("water_bg")
    if any(tok in {"forest", "bamboo", "land"} for tok in tokens):
        tokens.append("land_bg")
    return tokens


def tokenize_species(img_filename):
    folder = str(img_filename).split("/")[0]
    tokens = []
    for tok in re.split(r"[^a-zA-Z]+", folder.lower()):
        if not tok or tok.isdigit():
            continue
        tokens.append(tok)
    return tokens


def main():
    base = Path(r"d:/SPUME/waterbird_complete95_forest2water2")
    meta = pd.read_csv(base / "metadata.csv")
    trainval = meta[meta["split"] != 2].reset_index(drop=True)

    concept_lists = []
    counter = Counter()
    for _, row in trainval.iterrows():
        concepts = sorted(
            set(tokenize_context(row["place_filename"]) + tokenize_species(row["img_filename"]))
        )
        concept_lists.append(concepts)
        counter.update(concepts)

    # Keep moderately frequent concepts so the pseudo-groups are not too sparse.
    vocab = np.array(
        [concept for concept, freq in counter.items() if freq >= 20],
        dtype=object,
    )
    vocab.sort()
    concept2idx = {concept: i for i, concept in enumerate(vocab)}

    embeds = np.zeros((len(trainval), len(vocab)), dtype=np.uint8)
    for row_idx, concepts in enumerate(concept_lists):
        for concept in concepts:
            idx = concept2idx.get(concept)
            if idx is not None:
                embeds[row_idx, idx] = 1

    out_dir = base / "images"
    with open(out_dir / "metadata-concepts_vocab.pickle", "wb") as f:
        pickle.dump(vocab, f)
    with open(out_dir / "metadata-concepts_img_embeddings.pickle", "wb") as f:
        pickle.dump(embeds, f)

    print(f"train+val samples: {len(trainval)}")
    print(f"vocab size: {len(vocab)}")
    print(f"embedding shape: {embeds.shape}")
    print("top concepts:")
    for concept, freq in counter.most_common(20):
        if concept in concept2idx:
            print(f"{concept}: {freq}")


if __name__ == "__main__":
    main()
