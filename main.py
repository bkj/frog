#!/usr/bin/env python

"""
  train_search.py
"""

from __future__ import print_function, division

import os
import sys
import json
import warnings
import argparse
import numpy as np
from time import time
from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.data
from torchvision import datasets

from basenet.data import ZipDataloader
from basenet.helpers import set_seeds
from basenet import BaseNet, Metrics, HPSchedule
from basenet.vision import transforms as btransforms

from frog.models import cifar, fashion_mnist

NUM_WORKERS = 6
warnings.simplefilter(action='ignore', category=FutureWarning)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

# --
# CLI

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outpath', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--genotype', type=str)

    parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='batch size')

    parser.add_argument('--lr-max', type=float, default=0.025)
    parser.add_argument('--lr-min', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=3e-4)
    parser.add_argument('--cutout-length', type=int, default=0)

    parser.add_argument('--op-channels', type=int, default=16)
    parser.add_argument('--num-layers', type=int, default=8)
    parser.add_argument('--num-nodes', type=int, default=4)

    parser.add_argument('--arch-lr', type=float, default=3e-4)
    parser.add_argument('--arch-weight-decay', type=float, default=1e-3)
    parser.add_argument('--unrolled', action="store_true")

    parser.add_argument('--seed', type=int, default=123)
    return parser.parse_args()

# --
# Init

args = parse_args()
set_seeds(args.seed)

if args.outpath is not None:
    json.dump(vars(args), open(os.path.join(args.outpath, 'config.json'), 'w'))
else:
    print('args.outpath is None -- will not save weights', file=sys.stderr)

print(json.dumps(vars(args)), file=sys.stderr)

# --
# Data

if args.dataset == 'cifar10':
    model = cifar
    dataset_fn = datasets.CIFAR10
    in_channels = 3
    num_classes = 10
elif args.dataset == 'fashion_mnist':
    raise Exception('no fashion_mnist model!')
    dataset_fn = datasets.FashionMNIST
    in_channels = 1
    num_classes = 10

train_transform, valid_transform = btransforms.DatasetPipeline(dataset=args.dataset)
if args.cutout_length > 0:
    cutout = btransforms.Cutout(cut_h=args.cutout_length, cut_w=args.cutout_length)
    train_transform.transforms.append(cutout)

train_data = dataset_fn(root='./data', train=True, download=False, transform=train_transform)
valid_data = dataset_fn(root='./data', train=False, download=False, transform=valid_transform)

if not args.genotype:
    train_indices, search_indices = train_test_split(np.arange(len(train_data)), train_size=0.5, random_state=789)
    
    dataloaders = {
        "train"  : ZipDataloader([
            torch.utils.data.DataLoader(
                dataset=train_data,
                batch_size=args.batch_size,
                sampler=torch.utils.data.sampler.SubsetRandomSampler(train_indices),
                pin_memory=False,
                num_workers=NUM_WORKERS,
            ),
            torch.utils.data.DataLoader(
                dataset=train_data,
                batch_size=args.batch_size,
                sampler=torch.utils.data.sampler.SubsetRandomSampler(search_indices),
                pin_memory=False,
                num_workers=NUM_WORKERS,
            )
        ]),
        "valid"  : torch.utils.data.DataLoader(
            dataset=valid_data,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=NUM_WORKERS,
        )
    }
else:
    dataloaders = {
        "train"  : torch.utils.data.DataLoader(
            dataset=train_data,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=False,
            num_workers=NUM_WORKERS,
        ),
        "valid"  : torch.utils.data.DataLoader(
            dataset=valid_data,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=NUM_WORKERS,
        )
    }

# --
# Define model

net = model.Network(
    in_channels=in_channels,
    num_classes=num_classes,
    op_channels=args.op_channels,
    num_layers=args.num_layers,
    num_nodes=args.num_nodes,
)

if not args.genotype:
    arch = model.Architecture(num_nodes=args.num_nodes).to('cuda')
    arch.init_optimizer(
        opt=torch.optim.Adam,
        params=arch.parameters(),
        lr=args.arch_lr,
        betas=(0.5, 0.999),
        weight_decay=args.arch_weight_decay,
        clip_grad_norm=10.0,
    )
    
    net.init_search(arch=arch, unrolled=args.unrolled)
else:
    net.init_train(genotype=np.load(args.genotype))


lr_scheduler = HPSchedule.sgdr(hp_max=args.lr_max, period_length=args.epochs, hp_min=args.lr_min)
net.init_optimizer(
    opt=torch.optim.SGD,
    params=net.parameters(),
    hp_scheduler={
        "lr" : lr_scheduler,
    },
    momentum=args.momentum,
    weight_decay=args.weight_decay,
    clip_grad_norm=5.0,
)

net = net.to('cuda')
net.verbose = True
print(net, file=sys.stderr)

# --
# Run

t = time()
for epoch in range(args.epochs):
    train = net.train_epoch(dataloaders, mode='train', compute_acc=True)
    valid = net.eval_epoch(dataloaders, mode='valid', compute_acc=True)
    
    print(json.dumps({
        "epoch"      : int(epoch),
        "train_loss" : float(np.mean(train['loss'][-10:])),
        "train_acc"  : float(train['acc']),
        "valid_loss" : float(np.mean(valid['loss'])),
        "valid_acc"  : float(valid['acc']),
        "net_lr"     : float(net.hp['lr']),
    }))
    sys.stdout.flush()
    
    if args.outpath is not None:
        net.checkpoint(outpath=args.outpath, epoch=epoch)
