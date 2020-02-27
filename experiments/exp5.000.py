"""
This is experiment 5

bert as a service, concat, scikit mlp

"""

import numpy as np
import sklearn.neural_network as nn
from scipy.stats.stats import pearsonr

from utils.resourceManager import getEmbeddedResource

print("Getting data...")
data = getEmbeddedResource("exp5", "BertAsService", "de", "train")
val_data = getEmbeddedResource("exp5", "BertAsService", "de", "dev")
print("Tokenized data")

es,cs,y =[],[],[]
for e, c, y_ in data:
  es.append(e)
  cs.append(c)
  y.append(y_)
x = np.concatenate((es, cs), axis=1)
y = np.asarray(y)

es,cs,val_y =[],[],[]
for e, c, y_ in val_data:
  es.append(e)
  cs.append(c)
  val_y.append(y_)
val_x = np.concatenate((es, cs), axis=1)
val_y = np.asarray(val_y)


print("Train!")
model = nn.MLPRegressor(max_iter=4, hidden_layer_sizes=(600, 600), verbose=True)

model.fit(x, y)
my_y = model.predict(val_x)

print("PEARSON:", pearsonr(val_y, my_y))
print("MSE", np.mean(np.power(my_y-val_y, 2)))
print("MAE", np.mean(np.abs(my_y-val_y)))