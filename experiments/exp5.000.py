"""
This is experiment 5

bert as a service, concat, scikit mlp

"""

import numpy as np
import sklearn.neural_network as nn
from scipy.stats.stats import pearsonr

from utils.resourceManager import getEmbeddedResource

print("Getting data...")
data = getEmbeddedResource("exp5", "BertAsService", "zh", "train")
val_data = getEmbeddedResource("exp5", "BertAsService", "zh", "dev")
print("Tokenized data")

e, c, y = data
x = np.concatenate((e, c), axis=1)

e, c, y_val = val_data
x_val = np.concatenate((e, c), axis=1)

print("Train!")
model = nn.MLPRegressor(hidden_layer_sizes=(600,), verbose=True)

model.fit(x, y)
my_y = model.predict(x_val)

print("PEARSON:", pearsonr(y_val, my_y))