"""
This is experiment 4

pretrained word embbedings, bi-lstm,


"""

import torch
import torch.nn as nn
import random

from models.lstm import LSTM
from utils.tools import normalize_embeddings
from utils.resourceManager import getEmbeddedResource
from utils.statsmanager import StatsManager

print("Getting data...")
data = getEmbeddedResource("exp4", "FastText", "zh", "train")
val_data = getEmbeddedResource("exp4", "FastText", "zh", "dev")
print("Tokenized data")

model = LSTM(lstms_in_out=((300, 100), (300, 100)), linear_layers=(100,50), out_size=1, hidden_activation=nn.ReLU, final_activation=None).float()
print("Model loaded.")
learningRate = 0.01
epochs = 50
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)
batch_size = 100
print("Starting training...")
stats = StatsManager("exp4.000")

val_a_normalized, val_a_len = normalize_embeddings([row[0] for row in val_data])
val_b_normalized, val_b_len = normalize_embeddings([row[1] for row in val_data])
val_a = torch.tensor(val_a_normalized)
val_b = torch.tensor(val_b_normalized)
val_labels = torch.tensor([row[2] for row in val_data]).view((len(val_data), 1))


for epoch in range(epochs):
    random.shuffle(data)
    for batch in range(int(len(data)/batch_size)-1):
        print(".", end='')
        # Converting inputs and labels to Variable
        #print([row[0] for row in data[batch*batch_size:(batch+1)*batch_size]])
        a_normalized, a_len = normalize_embeddings([row[0] for row in data[batch * batch_size:(batch + 1) * batch_size]])
        b_normalized, b_len = normalize_embeddings([row[1] for row in data[batch * batch_size:(batch + 1) * batch_size]])
        a = torch.tensor(a_normalized)
        b = torch.tensor(b_normalized)
        labels = torch.tensor([row[2] for row in data[batch*batch_size:(batch+1)*batch_size]]).view((batch_size, 1))

        # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
        optimizer.zero_grad()

        # get output from the model, given the inputs
        outputs = model(a.float(), a_len, b.float(), b_len)

        # get loss for the predicted output
        loss = criterion(outputs, labels)
        stats.put(loss, outputs, labels)

        # get gradients w.r.t to parameters
        loss.backward()

        # update parameters
        optimizer.step()

    stats.printEpoch()

    val_outputs = model(val_a.float(), val_a_len, val_b.float(), val_b_len)
    stats.computeValidation(val_outputs, val_labels)

stats.plot()
