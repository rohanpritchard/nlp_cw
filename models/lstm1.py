import math

import numpy as np
import torch
import torch.nn as nn
import random

from junk_scripts.view_data import get_data


class LSTM1(torch.nn.Module):
    def __init__(self, lstms_in_out=((10, 50), (10, 50)), linear_layers=(10,10), out_size=1, hidden_activation=nn.ReLU, final_activation=nn.ReLU):
        super(LSTM1, self).__init__()
        self.final_activation = final_activation
        self.hidden_activation = hidden_activation

        self.eng_lstm = nn.LSTM(*lstms_in_out[0])  # Input dim is 3, output dim is 3
        self.ger_lstm = nn.LSTM(*lstms_in_out[1])
        concat_size = lstms_in_out[0][1] + lstms_in_out[1][1]
        self.linears = [nn.Linear(i, o) for i, o in zip([concat_size]+linear_layers[:], linear_layers[:]+[out_size])]

    def forward(self, eng, ger):
        eng_out, eng_hidden = self.eng_lstm(eng)
        ger_out, ger_hidden = self.ger_lstm(ger)

        cat = torch.cat((eng_hidden[0], ger_hidden[0]), 0)

        val = cat
        for l in self.linears:
            val = self.hidden_activation()(val)
            val = l(val)

        out = self.final_activation()(val)
        return out

data = get_data("train")

model = LSTM1()
learningRate = 0.01
epochs = 10
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)
batch_size = 100

for epoch in range(epochs):
    random.shuffle(data)
    losses = []
    for batch in range(len(data)/batch_size-1):
        # Converting inputs and labels to Variable
        eng = [row[0] for row in data[batch*batch_size:(batch+1)*batch_size]]
        ger = [row[1] for row in data[batch * batch_size:(batch + 1) * batch_size]]
        labels = [row[2] for row in data[batch*batch_size:(batch+1)*batch_size]]

        # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
        optimizer.zero_grad()

        # get output from the model, given the inputs
        outputs = model(eng, ger)

        # get loss for the predicted output
        loss = criterion(outputs, labels)
        losses.append(loss)
        # get gradients w.r.t to parameters
        loss.backward()

        # update parameters
        optimizer.step()

    print('epoch {}, loss {}'.format(epoch, np.mean(losses)))