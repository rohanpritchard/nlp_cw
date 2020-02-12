import math

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import random
import matplotlib.pyplot as plt

from torch import Tensor

from utils.get_data import get_word_to_index, tokenize, get_data, normalize


class LSTM1(torch.nn.Module):
    def __init__(self, eng_vocab_size, ger_vocab_size, padding_index=0, lstms_in_out=((5, 5), (5, 5)), linear_layers=(10,5), out_size=1, hidden_activation=nn.ReLU, final_activation=None):
        super(LSTM1, self).__init__()
        self.eng_embedding_layer = nn.Embedding(eng_vocab_size, lstms_in_out[0][0], padding_idx=padding_index)
        self.ger_embedding_layer = nn.Embedding(ger_vocab_size, lstms_in_out[1][0], padding_idx=padding_index)
        self.final_activation = final_activation
        self.hidden_activation = hidden_activation

        self.eng_lstm = nn.LSTM(*lstms_in_out[0], batch_first=True)
        self.ger_lstm = nn.LSTM(*lstms_in_out[1], batch_first=True)
        self.concat_size = lstms_in_out[0][1] + lstms_in_out[1][1]
        self.linears = [nn.Linear(i, o) for i, o in zip([self.concat_size]+list(linear_layers), list(linear_layers)+[out_size])]

    def forward(self, eng, eng_len, ger, ger_len):
        eng_embedded = self.eng_embedding_layer(eng)
        ger_embedded = self.ger_embedding_layer(ger)

        # Ignore the embeddings of zeros corresponding to the end of the sentence.
        eng_embedded_chopped = torch.nn.utils.rnn.pack_padded_sequence(eng_embedded, eng_len, batch_first=True,
                                                                  enforce_sorted=False)
        ger_embedded_chopped = torch.nn.utils.rnn.pack_padded_sequence(ger_embedded, ger_len, batch_first=True,
                                                                  enforce_sorted=False)

        eng_out, eng_hidden = self.eng_lstm(eng_embedded_chopped)
        ger_out, ger_hidden = self.ger_lstm(ger_embedded_chopped)

        cat = torch.cat((eng_hidden[0], ger_hidden[0]), -1).view(-1, self.concat_size)

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
data = get_data("train")
eng_to_index = get_word_to_index([d[0] for d in data])
ger_to_index = get_word_to_index([d[1] for d in data])
data = [(tokenize(eng, eng_to_index), tokenize(ger, ger_to_index), score) for eng, ger, score in data]
print("Tokenized data")

model = LSTM1(eng_vocab_size=len(eng_to_index), ger_vocab_size=len(ger_to_index))
print("Model loaded.")
learningRate = 0.01
epochs = 3
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)
batch_size = 100
print("Starting training...")
losses = []
for epoch in range(epochs):
    random.shuffle(data)
    for batch in range(int(len(data)/batch_size)-1):
        print(".", end='')
        # Converting inputs and labels to Variable
        #print([row[0] for row in data[batch*batch_size:(batch+1)*batch_size]])
        eng_normalized, eng_len = normalize([row[0] for row in data[batch*batch_size:(batch+1)*batch_size]])
        ger_normalized, ger_len = normalize([row[1] for row in data[batch*batch_size:(batch+1)*batch_size]])
        eng = torch.tensor(eng_normalized, dtype=int)
        ger = torch.tensor(ger_normalized, dtype=int)
        labels = torch.tensor([row[2] for row in data[batch*batch_size:(batch+1)*batch_size]])

        # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
        optimizer.zero_grad()

        # get output from the model, given the inputs
        outputs = model(eng, eng_len, ger, ger_len)

        # get loss for the predicted output
        loss = criterion(outputs, labels)

        losses.append(loss.item())

        # get gradients w.r.t to parameters
        loss.backward()

        # update parameters
        optimizer.step()

    print('\nepoch {}, loss {}'.format(epoch + 1, np.mean(losses)))

plt.ioff()
fig = plt.figure()
ax = fig.add_subplot()
line1, = ax.plot(losses)
plt.show()
