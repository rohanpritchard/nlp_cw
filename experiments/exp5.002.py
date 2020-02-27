"""
This is experiment 5

bert as a service, concat, scikit mlp

"""

import numpy as np
import sklearn.neural_network as nn
from scipy.stats.stats import pearsonr

from utils.resourceManager import getEmbeddedResource

print("Getting data...")
data = getEmbeddedResource("exp5", "BertAsService", "zh", "train", subname="en-ch", MultiServerBert=("./bert/uncased_L-12_H-768_A-12","./bert/chinese_L-12_H-768_A-12"))
val_data = getEmbeddedResource("exp5", "BertAsService", "zh", "dev", subname="en-ch", MultiServerBert=("./bert/uncased_L-12_H-768_A-12","./bert/chinese_L-12_H-768_A-12"))
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
model = nn.MLPRegressor(max_iter= 4, hidden_layer_sizes=(500,500,50), verbose=True, alpha=0.0005, learning_rate_init=0.001, batch_size=100)

model.fit(x, y)
my_y = model.predict(val_x)

print("PEARSON:", pearsonr(val_y, my_y))
print("MSE", np.mean(np.power(my_y-val_y, 2)))
print("MAE", np.mean(np.abs(my_y-val_y)))