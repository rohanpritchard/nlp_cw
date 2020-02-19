import pickle
import numpy as np

from utils.get_data import get_word_to_index, tokenize, get_data, normalize, Embedder, normalize_embeddings

with open("similarities.txt", 'rb') as f:
    similarities = pickle.load(f)
with open("en-de/train.ende.scores", 'r') as f:
    data = f.read().splitlines()

sim = np.zeros(len(data))
d = np.zeros(len(data))

for i, _ in enumerate(similarities):
    sim[i] = float(similarities[i])
    d[i] = float(data[i])
    print(i)

print(np.corrcoef(sim, d, rowvar=True, bias=True))
