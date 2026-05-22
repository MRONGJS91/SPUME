import pickle
import numpy as np

# Create proper embeddings array (5994 images x 5 concepts)
embeddings = np.random.rand(5994, 5).astype(np.float32)
with open(r'../waterbird_complete95_forest2water2/images/vit-gpt2_img_embeddings.pickle', 'wb') as f:
    pickle.dump(embeddings, f)

# Create proper vocab array
vocab = np.array(['bird', 'water', 'tree', 'sky', 'ground'], dtype=object)
with open(r'../waterbird_complete95_forest2water2/images/vit-gpt2_vocab.pickle', 'wb') as f:
    pickle.dump(vocab, f)

print('Created proper pickle files')