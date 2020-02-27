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
test_data = getEmbeddedResource("exp5", "BertAsService", "zh", "test", subname="en-ch", MultiServerBert=("./bert/uncased_L-12_H-768_A-12","./bert/chinese_L-12_H-768_A-12"))
print("Tokenized data")
data.extend(val_data)

es,cs,y =[],[],[]
for e, c, y_ in data:
  es.append(e)
  cs.append(c)
  y.append(y_)
x = np.concatenate((es, cs), axis=1)
y = np.asarray(y)

es,cs=[],[]
for e, c in test_data:
  es.append(e)
  cs.append(c)
test_x = np.concatenate((es, cs), axis=1)



print("Train!")
model = nn.MLPRegressor(max_iter=2, hidden_layer_sizes=(500,500,50), verbose=True, alpha=0.0005, learning_rate_init=0.001, batch_size=100)
model.fit(x, y)

my_y = model.predict(x)
#
print("PEARSON:", pearsonr(y, my_y))
print("MSE", np.mean(np.power(my_y-y, 2)))
print("MAE", np.mean(np.abs(my_y-y)))
print("DONE!")
test_y = model.predict(test_x)
print(test_y)
for y in test_y:
  print(y)