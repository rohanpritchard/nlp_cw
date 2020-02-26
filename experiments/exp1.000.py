"""
This is experiment 1

trained embeddings, bi-lstm

learned embeddings for both languages, each go through their own lstm, then the final state concatenated
and a dense network used
"""

import torch
import torch.nn as nn
import random

from models.lstm import LSTM
from utils.tools import get_word_to_index, tokenize, normalize
from utils.resourceManager import get_data
from utils.statsmanager import StatsManager

print("Getting data...")
data = get_data("train", "zh")
a_to_index = get_word_to_index([d[0] for d in data])
b_to_index = get_word_to_index([d[1] for d in data])
data = [(tokenize(a, a_to_index), tokenize(b, b_to_index), score) for a, b, score in data]

val_data = get_data("dev", "zh")
val_data = [(tokenize(a, a_to_index), tokenize(b, b_to_index), score) for a, b, score in val_data]

val_a_normalized, val_a_len = normalize([row[0] for row in val_data])
val_b_normalized, val_b_len = normalize([row[1] for row in val_data])
val_a = torch.tensor(val_a_normalized, dtype=int)
val_b = torch.tensor(val_b_normalized, dtype=int)
val_labels = torch.tensor([row[2] for row in val_data]).view((len(val_data), 1))

print("Tokenized data")

model = LSTM(a_vocab_size=len(a_to_index), b_vocab_size=len(b_to_index), padding_index=0, lstms_in_out=((5, 5), (5, 5)), linear_layers=(10, 5), out_size=1, hidden_activation=nn.ReLU, final_activation=None)
print("Model loaded.")
learningRate = 0.01
epochs = 30
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)
batch_size = 100
print("Starting training...")
stats = StatsManager()

for epoch in range(epochs):
    random.shuffle(data)
    for batch in range(int(len(data)/batch_size)-1):
        print(".", end='')
        # Converting inputs and labels to Variable
        #print([row[0] for row in data[batch*batch_size:(batch+1)*batch_size]])
        a_normalized, a_len = normalize([row[0] for row in data[batch * batch_size:(batch + 1) * batch_size]])
        b_normalized, b_len = normalize([row[1] for row in data[batch * batch_size:(batch + 1) * batch_size]])
        a = torch.tensor(a_normalized, dtype=int)
        b = torch.tensor(b_normalized, dtype=int)
        labels = torch.tensor([row[2] for row in data[batch*batch_size:(batch+1)*batch_size]]).view((batch_size, 1))

        # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
        optimizer.zero_grad()

        # get output from the model, given the inputs
        outputs = model(a, a_len, b, b_len)

        # get loss for the predicted output
        loss = criterion(outputs, labels)

        stats.put(loss, outputs, labels)

        # get gradients w.r.t to parameters
        loss.backward()

        # update parameters
        optimizer.step()

    stats.printEpoch()

    val_outputs = model(val_a, val_a_len, val_b, val_b_len)
    stats.computeValidation(val_outputs, val_labels)

stats.plot()
