"""
This is experiment 2

pretrained word embbedings, averaged to sentence, concat, scikit mlp

"""

import numpy as np
import sklearn.neural_network as nn
from scipy.stats.stats import pearsonr

from utils.resourceManager import getEmbeddedResource


def getAverage(data):
    data_averaged = []
    for e, c, l in data:
        e_avg = np.mean(e, axis=0)
        c_avg = np.mean(c, axis=0)
        data_averaged.append((np.array(list(e_avg) + list(c_avg)), l))
    return data_averaged

print("Getting data...")

data = getEmbeddedResource("exp2", "FastText", "zh", "train")
val_data = getEmbeddedResource("exp2", "FastText", "zh", "dev")
print("Tokenized data")

train = getAverage(data)
model = nn.MLPRegressor(hidden_layer_sizes=(5,))
X = [x[0] for x in train]
y = [x[1] for x in train]
model.fit(X, y)

test = getAverage(val_data)
X_test = [x[0] for x in test]
y_test = [x[1] for x in test]
my_y = model.predict(X_test)

print("PEARSON:", pearsonr(y_test, my_y))