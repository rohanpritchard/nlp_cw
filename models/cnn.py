import math

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import random
import matplotlib.pyplot as plt
import pickle
from scipy.stats.stats import pearsonr

from torch import Tensor

from utils.get_data import get_word_to_index, tokenize, get_data, normalize, Embedder, normalize_embeddings


class CNN(torch.nn.Module):
    def __init__(self, padding_index=0, lstms_in_out=((300, 100), (300, 100)), linear_layers=(100,50), out_size=1, hidden_activation=nn.ReLU, final_activation=None):
        super(CNN, self).__init__()
        self.final_activation = final_activation
        self.hidden_activation = hidden_activation

        self.eng_cnn = nn.Conv1d(in_channels=300, out_channels=300, kernel_size=3)
        self.ger_cnn = nn.Conv1d(in_channels=300, out_channels=300, kernel_size=3)
        self.concat_size = 600
        self.linears = [nn.Linear(i, o) for i, o in zip([self.concat_size]+list(linear_layers), list(linear_layers)+[out_size])]

    def forward(self, eng, eng_len, ger, ger_len):

        eng = self.eng_cnn.forward(eng.transpose(2, 1))
        ger = self.eng_cnn.forward(ger.transpose(2, 1))

        eng = torch.mean(eng, 2)
        ger = torch.mean(ger, 2)

        cat = torch.cat((eng, ger), -1).view(-1, self.concat_size)

        val = cat
        for l in self.linears:
            val = self.hidden_activation()(val)
            val = l(val)

        if self.final_activation is not None:
            out = self.final_activation()(val)
        else:
            out = val
        return out


print("Getting data...")
with open("embedded_data.train", "rb") as f:
    data = pickle.load(f)
with open("embedded_data.dev", "rb") as f:
    val = pickle.load(f)
print("Tokenized data")

model = CNN().float()
print("Model loaded.")
learningRate = 0.001
epochs = 20
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)
batch_size = 50
print("Starting training...")
losses = []
pearsons = []
maes = []
for epoch in range(epochs):
    elosses = []
    epearsons = []
    emaes = []
    random.shuffle(data)
    for batch in range(int(len(data)/batch_size)-1):
        print(".", end='')
        # Converting inputs and labels to Variable
        #print([row[0] for row in data[batch*batch_size:(batch+1)*batch_size]])
        eng_normalized, eng_len = normalize_embeddings([row[0] for row in data[batch*batch_size:(batch+1)*batch_size]])
        ger_normalized, ger_len = normalize_embeddings([row[1] for row in data[batch*batch_size:(batch+1)*batch_size]])
        eng = torch.tensor(eng_normalized)
        ger = torch.tensor(ger_normalized)
        labels = torch.tensor([row[2] for row in data[batch*batch_size:(batch+1)*batch_size]]).resize(batch_size, 1)

        # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
        optimizer.zero_grad()

        # get output from the model, given the inputs
        outputs = model(eng.float(), eng_len, ger.float(), ger_len)

        # get loss for the predicted output
        loss = criterion(outputs, labels)

        vx = outputs - torch.mean(outputs)
        vy = labels - torch.mean(labels)

        cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        epearsons.append(cost.item())
        pearsons.append(cost.item())

        mae = (outputs-labels).abs().mean()
        maes.append(mae.item())
        emaes.append(mae.item())
        losses.append(loss.item())
        elosses.append(loss.item())

        # get gradients w.r.t to parameters
        loss.backward()

        # update parameters
        optimizer.step()

    print('\nepoch {}, loss {}, mae {}, pearson {}'.format(epoch, np.mean(elosses), np.mean(emaes), np.mean(epearsons)))
    eng_normalized, eng_len = normalize_embeddings([row[0] for row in val])
    ger_normalized, ger_len = normalize_embeddings([row[1] for row in val])
    eng = torch.tensor(eng_normalized)
    ger = torch.tensor(ger_normalized)
    labels = torch.tensor([row[2] for row in val])
    outputs = model(eng.float(), eng_len, ger.float(), ger_len).detach().numpy().flatten()
    print("PEARSON", pearsonr(outputs, labels))

plt.ioff()
fig = plt.figure()
ax = fig.add_subplot()
line1, = ax.plot(losses)
line2, = ax.plot(maes)
plt.show()
