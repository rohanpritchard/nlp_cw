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

es,cs,y =[],[],[]
for e, c, y_ in data:
  es.append(e)
  cs.append(c)
  y.append(y_)
x = np.concatenate((es, cs), axis=1)

es,cs,val_y =[],[],[]
for e, c, y_ in val_data:
  es.append(e)
  cs.append(c)
  val_y.append(y_)
val_x = np.concatenate((es, cs), axis=1)


print("Train!")
model = nn.MLPRegressor(hidden_layer_sizes=(600,), verbose=True)

model.fit(x, y)
my_y = model.predict(val_x)

print("PEARSON:", pearsonr(val_y, my_y))