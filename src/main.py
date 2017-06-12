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

parser = argparse.ArgumentParser("Pointer Sentinel Mixture Models")
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--hidden', type=int, default=300)
args = parser.parse_args()

c = Corpus()
train_batch = Batchify(c.train, args.batch_size)
valid_batch = Batchify(c.valid, args.batch_size)
test_batch = Batchify(c.test, args.batch_size)
model = PSMM(args.batch_size, len(c.dict), args.hidden, args.cuda)

optimizer = optim.Adam(model.parameters(), lr=args.lr)
for epoch in range(args.epochs):
    model.train()
    for idx, (data, label) in enumerate(train_batch, 1):
        result = model(data)
        loss = F.nll_loss(result, Variable(label.view(-1)))
        ppl = np.exp(loss.data[0])
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), 0.25)
        optimizer.step()
        print "Epoch {}/{}, batch {}/{}: Perplexity: {}".format(epoch, args.epochs, idx, len(train_batch), ppl)