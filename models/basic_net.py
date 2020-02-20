import pickle
import random

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr


class Net(torch.nn.Module):
    def __init__(self, linear_layers, activation_func=torch.nn.functional.relu):
        super(Net, self).__init__()
        self.linears = [torch.nn.Linear(i, o) for i, o in linear_layers]
        self.activation_func = activation_func
        self.final_layer = torch.nn.Linear(linear_layers[-1][1], 1)

    def forward(self, x):
        for layer in self.linears:
            x = layer.forward(x)
            x = self.activation_func(x)
        x = self.final_layer.forward(x)
        return x


print("Getting data...")
with open("embedded_data.txt", "rb") as f:
    data = pickle.load(f)
print("Tokenized data")

data_averaged = []
for e, c, l in data:
    e_avg = np.mean(e, axis=0)
    c_avg = np.mean(c, axis=0)
    data_averaged.append((np.array(list(e_avg)+list(c_avg)), l))

print("Data size:", len(data_averaged))
val = data_averaged[-1000:]
data_averaged = data_averaged[:-1000]

model = Net([(600, 1000)]).float()
print("Model loaded.")
learningRate = 0.005
weight_decay = 0.0005
epochs = 50
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learningRate, weight_decay=weight_decay)
batch_size = 200
print("Starting training...")
losses = []
pearsons = []
maes = []
for epoch in range(epochs):
    random.shuffle(data_averaged)
    elosses = []
    epearsons = []
    emaes = []
    for batch in range(int(len(data_averaged)/batch_size)-1):

        # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
        optimizer.zero_grad()

        xs = torch.Tensor([row[0] for row in data_averaged[batch * batch_size:(batch + 1) * batch_size]])
        labels = torch.Tensor([row[1] for row in data_averaged[batch * batch_size:(batch + 1) * batch_size]]).resize(batch_size, 1)

        # get output from the model, given the inputs
        outputs = model(xs)

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

    xs = torch.Tensor([row[0] for row in val])
    labels = np.array([row[1] for row in val])
    outs = model(xs).detach().numpy().flatten()
    print('\nepoch {}, loss {}, mae {}, pearson {}'.format(epoch, np.mean(elosses), np.mean(emaes), np.mean(epearsons)))
    print("PEARSON", pearsonr(outs, labels))

plt.ioff()
fig = plt.figure()
ax = fig.add_subplot()
line1, = ax.plot(losses)
line2, = ax.plot(maes)
plt.show()