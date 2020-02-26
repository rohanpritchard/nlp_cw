import warnings

import torch
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

class StatsManager:
  def __init__(self):
    self.losses = []
    self.pearsons = []
    self.pearsons_val = []
    self.maes = []
    self.elosses = []
    self.epearsons = []
    self.emaes = []
    self.epoch = []
    self.epochs = 0
    self.batchs = 0

  def put(self, loss, outputs, labels):
    loss = loss.float().detach()
    outputs = outputs.float().detach()
    labels = labels.float().detach()

    vx = outputs - torch.mean(outputs)
    vy = labels - torch.mean(labels)
    cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
    p = pearsonr(np.asarray(outputs).flatten(), np.asarray(labels).flatten())
    self.epearsons.append(cost.item())
    self.pearsons.append(cost.item())

    mae = (outputs - labels).abs().mean()
    self.maes.append(mae.item())
    self.emaes.append(mae.item())

    self.losses.append(loss.item())
    self.elosses.append(loss.item())
    self.batchs += 1

  def printEpoch(self):
    print('\nepoch {}, loss {}, mae {}, pearson {}'.format(self.epochs, np.mean(self.elosses), np.mean(self.emaes), np.mean(self.epearsons)))
    self.epochs += 1
    self.epoch.append(self.batchs)
    self.elosses = []
    self.epearsons = []
    self.emaes = []

  def computeValidation(self, outputs, labels):
    outputs = outputs.detach()
    labels = labels.detach()
    p = pearsonr(np.asarray(outputs).flatten(), np.asarray(labels).flatten())
    print("PEARSON:", p)
    self.pearsons_val.append(p[0])

  def plot(self):
    plt.ioff()
    fig = plt.figure()
    ax = fig.add_subplot()
    line1, = ax.plot(np.arange(self.batchs), self.losses, label="loss")
    line2, = ax.plot(np.arange(self.batchs), self.maes, label="mae")
    line3, = ax.plot(np.arange(self.batchs), self.pearsons, label="pearson")
    if len(self.epoch) == len(self.pearsons_val):
      line4, = ax.plot(self.epoch, self.pearsons_val, label="validation pearson")
    else:
      warnings.warn("peasons_val has not been populated, use 'computeValidation' after each epoch")
    ax.legend()
    plt.show()


