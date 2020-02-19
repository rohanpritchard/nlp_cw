import math

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import random
import matplotlib.pyplot as plt
import pickle

from torch import Tensor

from utils.get_data import get_word_to_index, tokenize, get_data, normalize, Embedder, normalize_embeddings

SENTENCE_ENCODING_DIFFERENCE_NODES = 1

class LSTM3(torch.nn.Module):
    def __init__(self, padding_index=0, lstms_in_out=((300, 100), (300, 100)), linear_layers=(100,50), out_size=1, hidden_activation=nn.ReLU, final_activation=None):
        super(LSTM3, self).__init__()
        self.final_activation = final_activation
        self.hidden_activation = hidden_activation

        self.eng_lstm = nn.LSTM(*lstms_in_out[0], batch_first=True)
        self.ger_lstm = nn.LSTM(*lstms_in_out[1], batch_first=True)
        self.concat_size = lstms_in_out[0][1] + lstms_in_out[1][1]
        self.linears = [nn.Linear(i, o) for i, o in zip([self.concat_size + SENTENCE_ENCODING_DIFFERENCE_NODES]+list(linear_layers), list(linear_layers)+[out_size])]

    def forward(self, eng, eng_len, ger, ger_len, similarities):
        # Ignore the embeddings of zeros corresponding to the end of the sentence.
        eng_embedded_chopped = torch.nn.utils.rnn.pack_padded_sequence(eng, eng_len, batch_first=True,
                                                                  enforce_sorted=False)
        ger_embedded_chopped = torch.nn.utils.rnn.pack_padded_sequence(ger, ger_len, batch_first=True,
                                                                  enforce_sorted=False)

        #print("Eng chopped:", eng_embedded_chopped.shape)

        eng_out, eng_hidden = self.eng_lstm(eng_embedded_chopped)
        ger_out, ger_hidden = self.ger_lstm(ger_embedded_chopped)

        #print("Eng hidden:", eng_hidden[0][0].shape)
        #print("Ger hidden:", ger_hidden[0][0].shape)
        #print("Similarities:", similarities.view(similarities.shape[0], 1).shape)

        cat = torch.cat((eng_hidden[0][0], ger_hidden[0][0], similarities.view(similarities.shape[0], 1)), -1)#.view(-1, self.concat_size)

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
with open("embedded_data.txt", "rb") as f:
    data = pickle.load(f)
with open("similarities.txt", "rb") as f:
    similarities = pickle.load(f)
print("Tokenized data")

model = LSTM3().float()
print("Model loaded.")
learningRate = 0.01
epochs = 20
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())
batch_size = 20
print("Starting training...")
losses = []
pearsons = []
maes = []
for epoch in range(epochs):
    elosses = []
    epearsons = []
    emaes = []
    # random.shuffle(data)
    for batch in range(int(len(data)/batch_size)-1):
        print(".", end='')
        # Converting inputs and labels to Variable
        #print([row[0] for row in data[batch*batch_size:(batch+1)*batch_size]])
        eng_normalized, eng_len = normalize_embeddings([row[0] for row in data[batch*batch_size:(batch+1)*batch_size]])
        ger_normalized, ger_len = normalize_embeddings([row[1] for row in data[batch*batch_size:(batch+1)*batch_size]])
        eng = torch.tensor(eng_normalized)
        ger = torch.tensor(ger_normalized)
        labels = torch.tensor([row[2] for row in data[batch*batch_size:(batch+1)*batch_size]])

        # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to accumulate gradients
        optimizer.zero_grad()

        # get output from the model, given the inputs
        outputs = model(eng.float(), eng_len, ger.float(), ger_len, similarities[batch*batch_size:(batch+1)*batch_size])

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

plt.ioff()
fig = plt.figure()
ax = fig.add_subplot()
line1, = ax.plot(losses)
line2, = ax.plot(maes)
plt.show()
