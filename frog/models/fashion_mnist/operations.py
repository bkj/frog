#!/usr/bin/env python

"""
  operations.py
"""

import torch
import torch.nn as nn
from collections import namedtuple
Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

OPS = {
  'none'         : lambda C, stride, affine: Zero(),
  'max_pool_3x3' : lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
  'skip_connect' : lambda C, stride, affine: Identity()
  'conv_3x3'     : lambda C, stride, affine: ReLUConvBN(C, C, kernel_size=3, stride=stride, padding=1, affine=True),
}

PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'skip_connect',
    'conv_3x3',
]


class ReLUConvBN(nn.Module):
  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super().__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
    )
    
  def forward(self, x):
    return self.op(x)
  
  def __repr__(self):
    return self.op.__repr__()


class Identity(nn.Module):
  def __init__(self):
    super().__init__()
    
  def forward(self, x):
    return x
    
  def __repr__(self):
    return "Identity()"


class Zero(nn.Module):
  def forward(self, x):
    raise Exception