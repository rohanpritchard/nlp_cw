import pickle

import numpy as np
import sklearn.neural_network as nn
from scipy.stats.stats import pearsonr

print("Getting data...")
with open("embedded_bert.train", "rb") as f:
    data = pickle.load(f)
print("Tokenized data")

data_averaged = []
for e, c, l in data:
    data_averaged.append((np.array(list(e)+list(c)), l))

model = nn.MLPRegressor(hidden_layer_sizes=(5,))
train = data_averaged[:-1000]
test = data_averaged[-1000:]
X = [x[0] for x in train]
y = [x[1] for x in train]
X_test = [x[0] for x in test]
y_test = [x[1] for x in test]
model.fit(X, y)
my_y = model.predict(X_test)

print("PEARSON:", pearsonr(y_test, my_y))