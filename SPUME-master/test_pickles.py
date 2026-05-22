import pickle
import numpy as np

try:
    with open(r'../waterbird_complete95_forest2water2/images/vit-gpt2_img_embeddings.pickle', 'rb') as f:
        embeddings = pickle.load(f)
    print(f'Embeddings shape: {embeddings.shape}')
    print(f'Embeddings dtype: {embeddings.dtype}')

    with open(r'../waterbird_complete95_forest2water2/images/vit-gpt2_vocab.pickle', 'rb') as f:
        vocab = pickle.load(f)
    print(f'Vocab: {vocab}')
    print(f'Vocab dtype: {vocab.dtype}')
    print('Files are valid!')
except Exception as e:
    print(f'Error: {e}')