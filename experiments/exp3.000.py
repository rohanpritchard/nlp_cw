"""
This is experiment 3

pretrained word embbedings, convo layer, average to sentence, concat, mlp


"""

import torch
import torch.nn as nn
import random

from utils.resourceManager import getEmbeddedResource
from utils.statsmanager import StatsManager
from utils.tools import normalize_embeddings


class CNN(torch.nn.Module):
  def __init__(self, padding_index=0, lstms_in_out=((300, 100), (300, 100)), linear_layers=(100, 50), out_size=1,
               hidden_activation=nn.ReLU, final_activation=None):
    super(CNN, self).__init__()
    self.final_activation = final_activation
    self.hidden_activation = hidden_activation

    self.eng_cnn = nn.Conv1d(in_channels=300, out_channels=300, kernel_size=3)
    self.ger_cnn = nn.Conv1d(in_channels=300, out_channels=300, kernel_size=3)
    self.concat_size = 600
    self.linears = [nn.Linear(i, o) for i, o in
                    zip([self.concat_size] + list(linear_layers), list(linear_layers) + [out_size])]

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
data = getEmbeddedResource("exp3", "FastText", "zh", "train")
val_data = getEmbeddedResource("exp3", "FastText", "zh", "dev")
print("Tokenized data")

model = CNN().float()
print("Model loaded.")
learningRate = 0.001
epochs = 20
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)
batch_size = 50
print("Starting training...")
stats = StatsManager()

val_a_normalized, val_a_len = normalize_embeddings([row[0] for row in val_data])
val_b_normalized, val_b_len = normalize_embeddings([row[1] for row in val_data])
val_a = torch.tensor(val_a_normalized)
val_b = torch.tensor(val_b_normalized)
val_labels = torch.tensor([row[2] for row in val_data]).view((len(val_data), 1))

for epoch in range(epochs):
  random.shuffle(data)
  for batch in range(int(len(data) / batch_size) - 1):
    print(".", end='')
    # Converting inputs and labels to Variable
    # print([row[0] for row in data[batch*batch_size:(batch+1)*batch_size]])
    a_normalized, a_len = normalize_embeddings([row[0] for row in data[batch * batch_size:(batch + 1) * batch_size]])
    b_normalized, b_len = normalize_embeddings([row[1] for row in data[batch * batch_size:(batch + 1) * batch_size]])
    a = torch.tensor(a_normalized)
    b = torch.tensor(b_normalized)
    labels = torch.tensor([row[2] for row in data[batch * batch_size:(batch + 1) * batch_size]]).view((batch_size, 1))

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
