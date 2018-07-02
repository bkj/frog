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
from operations import PRIMITIVES
from model_search import DARTSearchNetwork, DARTArchitecture

from basenet import BaseNet, Metrics, HPSchedule
from basenet.helpers import set_seeds
from basenet.vision import transforms as btransforms

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# --
# CLI

def parse_args():
  parser = argparse.ArgumentParser("cifar")
  parser.add_argument('--outpath', type=str, default="results/search/0")
  parser.add_argument('--dataset', type=str, default='cifar10')
  parser.add_argument('--genotype', type=str)
  
  parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
  parser.add_argument('--batch-size', type=int, default=64, help='batch size')
  
  parser.add_argument('--lr-max', type=float, default=0.025)
  parser.add_argument('--lr-min', type=float, default=0.001)
  parser.add_argument('--momentum', type=float, default=0.9)
  parser.add_argument('--weight-decay', type=float, default=3e-4)
  
  parser.add_argument('--op-channels', type=int, default=16)
  parser.add_argument('--num-layers', type=int, default=8)
  parser.add_argument('--num-nodes', type=int, default=4)
  
  parser.add_argument('--arch-lr', type=float, default=3e-4)
  parser.add_argument('--arch-weight-decay', type=float, default=1e-3)
  
  parser.add_argument('--seed', type=int, default=123)
  return parser.parse_args()

# --
# Run

args = parse_args()

num_ops = len(PRIMITIVES)

set_seeds(args.seed)

if args.dataset == 'cifar10':
  dataset_fn = datasets.CIFAR10
  in_channels = 3
  num_classes = 10
elif args.dataset == 'fashion_mnist':
  dataset_fn = datasets.FashionMNIST
  in_channels = 1
  num_classes = 10

train_transform, valid_transform = btransforms.DatasetPipeline(dataset=args.dataset)
# if args.cutout:
#   cutout = btransforms.Cutout(cut_h=args.cutout_length, cut_w=args.cutout_length)
#   train_transform.transforms.append(cutout)

train_data = dataset_fn(root='./data', train=True, download=False, transform=train_transform)
valid_data = dataset_fn(root='./data', train=False, download=False, transform=valid_transform)


# Is this necessary?
train_indices, search_indices = train_test_split(np.arange(len(train_data)), train_size=0.5)

dataloaders = {
  "train"  : utils.ZipDataloader([
    torch.utils.data.DataLoader(
      dataset=train_data,
      batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(train_indices),
      pin_memory=False,
      num_workers=0,
    ),
    torch.utils.data.DataLoader(
      dataset=train_data,
      batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(search_indices),
      pin_memory=False,
      num_workers=0,
    )
  ]),
  "valid"  : torch.utils.data.DataLoader(
    dataset=valid_data,
    batch_size=args.batch_size,
    shuffle=False,
    pin_memory=True,
    num_workers=0,
  )
}

cuda = torch.device('cuda')

if not args.genotype:
  arch = DARTArchitecture(num_nodes=args.num_nodes, num_ops=num_ops).to(cuda)
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
    in_channels=in_channels,
    num_classes=num_classes,
    op_channels=args.op_channels,
    num_layers=args.num_layers,
    num_nodes=args.num_nodes,
  ).to(cuda)
else:
  # Check that `num_nodes` and `genotype` sizes work
  model = DARTTrainNetwork(
    genotype=np.load(args.genotype),
    in_channels=in_channels,
    num_classes=num_classes,
    op_channels=args.op_channels,
    num_layers=args.num_layers,
    num_nodes=args.num_nodes,
  )

model.verbose = True
print(model, file=sys.stderr)

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
