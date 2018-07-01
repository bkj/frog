import os
import sys
import time
import glob
import json
import numpy as np
import torch
import itertools
import argparse
from tqdm import tqdm
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
from torchvision import datasets

import utils
from torch.autograd import Variable
from model_search import Network
from genotypes import PRIMITIVES

from basenet.helpers import set_seeds

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True


def parse_args():
  parser = argparse.ArgumentParser("cifar")
  parser.add_argument('--outpath', type=str, default="results/search/0")
  parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
  parser.add_argument('--batch_size', type=int, default=64, help='batch size')
  
  parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
  parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
  parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
  parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
  
  parser.add_argument('--num-classes', type=int, default=10)
  parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
  parser.add_argument('--layers', type=int, default=8, help='total number of layers')
  
  parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
  parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
  
  parser.add_argument('--seed', type=int, default=123)
  return parser.parse_args()


if __name__ == "__main__":
  args = parse_args()
  
  steps = 4
  num_ops = len(PRIMITIVES)
  
  set_seeds(args.seed)
  
  train_transform, valid_transform = utils._data_transforms_cifar10(cutout=False)
  train_data = datasets.CIFAR10(root='../data', train=True, download=False, transform=train_transform)
  valid_data = datasets.CIFAR10(root='../data', train=False, download=False, transform=valid_transform)
  
  indices = np.arange(len(train_data))
  split   = int(np.floor(0.5 * len(train_data)))
  
  dataloaders = {
    "train"  : torch.utils.data.DataLoader(
      dataset=train_data,
      batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]), # !! Is there a real reason to keep these separate?
      pin_memory=True,
      num_workers=2,
    ),
    "search" : torch.utils.data.DataLoader(
      dataset=train_data,
      batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:]), # !! Is there a real reason to keep these separate?
      pin_memory=True,
      num_workers=2,
    ),
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
  
  model = Network(args.init_channels, args.num_classes, args.layers, steps=steps).cuda()
  optimizer = torch.optim.SGD(
      params=model.parameters(),
      lr=args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay
  )
  model_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs), eta_min=args.learning_rate_min)
  
  # --
  # Define meta-model
  
  k = sum(1 for i in range(steps) for n in range(2+i))
  scale = 1e-3
  arch_parameters = [
    Variable(scale * torch.randn(k, num_ops).cuda(), requires_grad=True),
    Variable(scale * torch.randn(k, num_ops).cuda(), requires_grad=True),
  ]
  
  arch_optimizer = torch.optim.Adam(
    params=arch_parameters,
    lr=args.arch_learning_rate,
    betas=(0.5, 0.999),
    weight_decay=args.arch_weight_decay
  )
  
  # --
  # Run
  
  for epoch in range(args.epochs):
    _ = model_lr_scheduler.step()
    
    # --
    # Train loop
    
    _ = model.train()
    train_n_total, train_n_correct, train_loss = 0, 0, 0
    gen = zip(dataloaders['train'], itertools.cycle(dataloaders['search']))
    gen = tqdm(gen, total=len(dataloaders['train']))
    for ((input_train, target_train), (input_search, target_search)) in gen:
      
      # Arch step (no unrolled yet)
      arch_optimizer.zero_grad()
      input_search  = Variable(input_search, requires_grad=False).cuda()
      target_search = Variable(target_search, requires_grad=False).cuda(async=True)
      loss  = F.cross_entropy(model(input_search, arch_parameters), target_search)
      loss.backward()
      nn.utils.clip_grad_norm(arch_parameters, 10.)
      arch_optimizer.step()
      arch_optimizer.zero_grad()
      
      # Model step
      optimizer.zero_grad()
      input_train  = Variable(input_train, requires_grad=False).cuda()
      target_train = Variable(target_train, requires_grad=False).cuda(async=True)
      preds = model(input_train, arch_parameters)
      loss  = F.cross_entropy(preds, target_train)
      loss.backward()
      nn.utils.clip_grad_norm(model.parameters(), 5.)
      optimizer.step()
      optimizer.zero_grad()
      
      train_n_correct += int((preds.max(dim=-1)[1] == target_train).int().sum())
      train_n_total   += int(input_train.shape[0])
      train_loss      += float(loss.data)
      
      gen.set_postfix(**{
        "epoch"      : int(epoch),
        "train_loss" : float(train_loss / train_n_total),
        "train_acc"  : float(train_n_correct / train_n_total),
      })
      
      del input_search
      del target_search
      del input_train
      del target_train
      del preds
    
    # --
    # Valid loop
    
    _ = model.eval()
    
    valid_n_total, valid_n_correct, valid_loss = 0, 0, 0
    gen = tqdm(dataloaders['valid'], total=len(dataloaders['valid']))
    for (input_valid, target_valid) in gen:
      input_valid  = Variable(input_valid, requires_grad=False).cuda()
      target_valid = Variable(target_valid, requires_grad=False).cuda(async=True)
      
      preds           = model(input_valid, arch_parameters)
      valid_n_correct += int((preds.max(dim=-1)[1] == target_valid).int().sum().cpu())
      valid_n_total   += int(input_valid.shape[0])
      valid_loss      += F.cross_entropy(preds, target_valid).data.cpu()
      
      gen.set_postfix(**{
        "epoch"      : int(epoch),
        "valid_loss" : float(valid_loss / valid_n_total),
        "valid_acc"  : float(valid_n_correct / valid_n_total),
      })
      
      del input_valid
      del target_valid
      del preds
    
    print(json.dumps({
      "epoch"      : int(epoch),
      "train_loss" : float(train_loss / train_n_total),
      "train_acc"  : float(train_n_correct / train_n_total),
      "valid_loss" : float(valid_loss / valid_n_total),
      "valid_acc"  : float(valid_n_correct / valid_n_total),
    }))
    sys.stdout.flush()
    
    torch.save(model.state_dict(), os.path.join(args.outpath, 'weights.pt'))
    torch.save(arch_parameters[0], os.path.join(args.outpath, 'normal_arch_e%d.pt' % epoch))
    torch.save(arch_parameters[1], os.path.join(args.outpath, 'reduce_arch_e%d.pt' % epoch))



