import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import *
from torch.autograd import *
from preprocess import *
from batchify import *
from model import *

batch_size = 32
lr=1e-3
epochs = 10

c = Corpus()
train_batch = Batchify(c.train, batch_size)
valid_batch = Batchify(c.valid, batch_size)
test_batch = Batchify(c.test, batch_size)
model = PSMM(batch_size, 10000, 300)

optimizer = optim.Adam(model.parameters(), lr=lr)
for epoch in range(epochs):
    model.train()
    for idx, (data, label) in enumerate(train_batch, 1):
        result = model(data)
        loss = F.nll_loss(result, Variable(label.view(-1)))
        ppl = np.exp(loss.data[0])
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), 0.25)
        optimizer.step()
        print "Epoch {}/{}, batch {}/{}: Perplexity: {}".format(epoch, epochs, idx, len(train_batch), ppl)