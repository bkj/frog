#!/usr/bin/env python

"""
  train_search.py
"""

from __future__ import print_function, division

import os
import sys
import json
import argparse
import itertools
import numpy as np
from tqdm import tqdm
from time import time
from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.data
from torchvision import datasets

import utils
from genotypes import PRIMITIVES
from model_search import DARTSearchNetwork

from basenet import BaseNet, Metrics, HPSchedule
from basenet.helpers import set_seeds

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

# --
# CLI

def parse_args():
  parser = argparse.ArgumentParser("cifar")
  parser.add_argument('--outpath', type=str, default="results/search/0")
  parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
  parser.add_argument('--batch-size', type=int, default=64, help='batch size')
  
  parser.add_argument('--lr-max', type=float, default=0.025)
  parser.add_argument('--lr-min', type=float, default=0.001)
  parser.add_argument('--momentum', type=float, default=0.9)
  parser.add_argument('--weight-decay', type=float, default=3e-4)
  
  parser.add_argument('--num-classes', type=int, default=10)
  parser.add_argument('--init-channels', type=int, default=16)
  parser.add_argument('--layers', type=int, default=8)
  parser.add_argument('--steps', type=int, default=4)
  
  parser.add_argument('--arch-lr', type=float, default=3e-4)
  parser.add_argument('--arch-weight-decay', type=float, default=1e-3)
  
  parser.add_argument('--seed', type=int, default=123)
  return parser.parse_args()

# --
# Run

args = parse_args()

num_ops = len(PRIMITIVES)

set_seeds(args.seed)

train_transform, valid_transform = utils._data_transforms_cifar10(cutout=False)
train_data = datasets.CIFAR10(root='./data', train=True, download=False, transform=train_transform)
valid_data = datasets.CIFAR10(root='./data', train=False, download=False, transform=valid_transform)

# Is this necessary?
train_indices, search_indices = train_test_split(np.arange(len(train_data)), train_size=0.5)

class ZipDataloader:
  def __init__(self, dataloaders):
    self.dataloaders = dataloaders
  
  def __len__(self):
    return max([len(d) for d in self.dataloaders])
  
  def __iter__(self):
    counter = 0
    iters = [iter(d) for d in self.dataloaders]
    while counter < len(self):
      yield tuple(zip(*[next(it) for it in iters]))
      counter += 1

dataloaders = {
  "train"  : ZipDataloader([
    torch.utils.data.DataLoader(
      dataset=train_data,
      batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(train_indices),
      pin_memory=False,
      num_workers=4,
    ),
    torch.utils.data.DataLoader(
      dataset=train_data,
      batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(search_indices),
      pin_memory=False,
      num_workers=4,
    )
  ]),
  "valid"  : torch.utils.data.DataLoader(
    dataset=valid_data,
    batch_size=args.batch_size,
    shuffle=False,
    pin_memory=True,
    num_workers=2,
  )
}

# --
# Define model

class Architecture(BaseNet):
  def __init__(self, steps, num_ops, scale=1e-3):
    super().__init__(loss_fn=F.cross_entropy)
    
    self.steps   = steps
    self.num_ops = num_ops
    
    n_edges     = 2 * steps + sum(range(steps)) # Connections to inputs + connections to earlier steps
    self.normal = nn.Parameter(torch.FloatTensor(np.random.normal(0, scale, (n_edges, num_ops))))
    self.reduce = nn.Parameter(torch.FloatTensor(np.random.normal(0, scale, (n_edges, num_ops))))
  
  def __repr__(self):
    return 'Architecture(steps=%d | num_ops=%d)' % (self.steps, self.num_ops)

cuda = torch.device('cuda')
arch = Architecture(steps=args.steps, num_ops=num_ops).to(cuda)
arch.init_optimizer(
  opt=torch.optim.Adam,
  params=arch.parameters(),
  lr=args.arch_lr,
  betas=(0.5, 0.999),
  weight_decay=args.arch_weight_decay,
  clip_grad_norm=10.
)

model = DARTSearchNetwork(
  arch=arch,
  C=args.init_channels,
  num_classes=args.num_classes,
  layers=args.layers,
  steps=args.steps,
).to(cuda)
model.verbose = True

model.init_optimizer(
  opt=torch.optim.SGD,
  params=model.parameters(),
  hp_scheduler={
    "lr" : HPSchedule.sgdr(hp_max=args.lr_max, period_length=args.epochs, hp_min=args.lr_min),
  },
  momentum=args.momentum,
  weight_decay=args.weight_decay,
  clip_grad_norm=5.0,
)

# --
# Run

t = time()
for epoch in range(args.epochs):
  train = model.train_epoch(dataloaders, mode='train', compute_acc=True)
  valid = model.eval_epoch(dataloaders, mode='valid', compute_acc=True)
  
  print(json.dumps({
    "epoch"      : int(epoch),
    "train_loss" : float(np.mean(train['loss'][-10:])),
    "train_acc"  : float(train['acc']),
    "valid_loss" : float(np.mean(valid['loss'])),
    "valid_acc"  : float(valid['acc']),
  }))
  sys.stdout.flush()
  
  torch.save(model.state_dict(), os.path.join(args.outpath, 'weights.pt'))
  torch.save(arch.normal, os.path.join(args.outpath, 'normal_arch_e%d.pt' % epoch))
  torch.save(arch.reduce, os.path.join(args.outpath, 'reduce_arch_e%d.pt' % epoch))



