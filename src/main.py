import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import *
from torch.autograd import *
from tensorboard_logger import configure, log_value
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

configure("../runs/psmm", flush_secs=5)

c = Corpus()
train_batch = Batchify(c.train, args.batch_size)
valid_batch = Batchify(c.valid, args.batch_size)
test_batch = Batchify(c.test, args.batch_size)
model = PSMM(args.batch_size, len(c.dict), args.hidden, args.cuda)
if args.cuda:
    model = model.cuda()

optimizer = optim.Adam(model.parameters(), lr=args.lr)
min_ppl = 1e9

for epoch in range(args.epochs):
    model.train()
    log_ppl = 0
    for idx, (data, label) in enumerate(train_batch, 1):
        result = model(data)
        label = Variable(label.view(-1))
        if args.cuda:
            label = label.cuda()

        loss = F.nll_loss(result, label)
        log_ppl += loss
        ppl = np.exp(loss.data[0])
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), 0.25)
        optimizer.step()
        print "Epoch {}/{}, batch {}/{}: Perplexity: {}".format(epoch, args.epochs, idx, len(train_batch), ppl)

    log_ppl /= len(train_batch)
    ppl = np.exp(log_ppl)
    print "Train perplexity: {}".format(ppl)
    log_value('train_ppl', ppl, epoch)

    model.eval()
    log_ppl = 0
    for data, label in valid_batch:
        result = model(data)
        label = Variable(label.view(-1))
        if args.cuda:
            label = label.cuda()

        log_ppl += F.nll_loss(result, label)

    log_ppl /= len(valid_batch)
    ppl = np.exp(log_ppl)
    print "Evaluation perplexity: {}".format(ppl)
    log_value('eval_ppl', ppl, epoch)

    if ppl < min_ppl:
        min_ppl = ppl
        with open('../params/model.params', 'wb') as f:
            torch.save(model, f)
    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 4.0

model.eval()
log_ppl = 0
for data, label in test_batch:
    result = model(data)
    label = Variable(label.view(-1))
    if args.cuda:
        label = label.cuda()

    log_ppl += F.nll_loss(result, label)

log_ppl /= len(test_batch)
ppl = np.exp(log_ppl)
print "Test perplexity: {}".format(ppl)
