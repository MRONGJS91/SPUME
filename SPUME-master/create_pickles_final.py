#!/usr/bin/env python3
import pickle
import numpy as np

print("Creating embeddings...")
embeddings = np.random.rand(5994, 5).astype(np.float32)
with open(r'../waterbird_complete95_forest2water2/images/vit-gpt2_img_embeddings.pickle', 'wb') as f:
    pickle.dump(embeddings, f)
print(f"Embeddings shape: {embeddings.shape}")

print("Creating vocab...")
vocab = np.array(['bird', 'water', 'tree', 'sky', 'ground'], dtype=object)
with open(r'../waterbird_complete95_forest2water2/images/vit-gpt2_vocab.pickle', 'wb') as f:
    pickle.dump(vocab, f)
print(f"Vocab: {vocab}")

print("Files created successfully!")