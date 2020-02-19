import pickle
import numpy as np

with open("similarities.txt", 'rb') as f:
    similarities = pickle.load(f)
with open("embedded_data.txt", 'rb') as f:
    data = pickle.load(f)

sim = np.zeros(len(data))
d = np.zeros(len(data))

for i, (_, _, score) in enumerate(data):
    sim[i] = similarities[i]
    d[i] = score

print(np.cov(sim, d, bias=True))
