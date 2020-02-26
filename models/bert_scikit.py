import pickle

import numpy as np
import sklearn.neural_network as nn
from scipy.stats.stats import pearsonr

print("Getting data...")
with open("bert_encoded_train", "rb") as f:
    data = pickle.load(f)
with open("bert_encoded_dev", "rb") as f:
    data1 = pickle.load(f)
print("Tokenized data")

e, c, y = data
x = np.concatenate((e, c), axis=1)

e, c, y_val = data1
x_val = np.concatenate((e, c), axis=1)

print("Train!")
model = nn.MLPRegressor(hidden_layer_sizes=(600,), verbose=True)

model.fit(x, y)
my_y = model.predict(x_val)

print("PEARSON:", pearsonr(y_val, my_y))