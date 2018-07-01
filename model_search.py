#!/usr/bin/env python

"""
  model_search.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from basenet import BaseNet

from operations import *
from genotypes import PRIMITIVES

class MixedOp(nn.Module):
  def __init__(self, C, stride):
    super().__init__()
    
    self._ops = nn.ModuleList()
    for primitive in PRIMITIVES:
      op = OPS[primitive](C, stride, False)
      if 'pool' in primitive:
        op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
      self._ops.append(op)
      
  def forward(self, x, weights):
    return sum(w * op(x) for w, op in zip(weights, self._ops))


class DARTSearchCell(nn.Module):
  def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
    super().__init__()
    
    self.reduction   = reduction
    self._steps      = steps
    self._multiplier = multiplier
    
    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
    
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
    
    self._ops = nn.ModuleList()
    for i in range(self._steps):
      for idx in range(2+i):
        stride = 2 if reduction and idx < 2 else 1
        self._ops.append(MixedOp(C, stride))
    
  def forward(self, s0, s1, weights):
    states = [
      self.preprocess0(s0),
      self.preprocess1(s1),
    ]
    offset = 0
    for i in range(self._steps):
      s = sum(self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
      offset += len(states)
      states.append(s)
      
    return torch.cat(states[-self._multiplier:], dim=1)


class DARTSearchNetwork(BaseNet):
  def __init__(self, arch, C, num_classes, layers, steps=4, multiplier=4, stem_multiplier=3):
    super().__init__(loss_fn=F.cross_entropy)
    
    self._arch        = arch
    self._C           = C
    self._num_classes = num_classes
    self._layers      = layers
    
    self._steps       = steps
    self._multiplier  = multiplier
    
    C_curr = stem_multiplier * C
    self.stem = nn.Sequential(
      nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
      nn.BatchNorm2d(C_curr)
    )
    
    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
    self.cells = nn.ModuleList()
    reduction_prev = False
    
    for i in range(layers):
      reduction = i in [layers//3, 2 * layers//3]
      if reduction:
        C_curr *= 2
      
      cell = DARTSearchCell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, multiplier * C_curr
    
    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier     = nn.Linear(C_prev, num_classes)

  def forward(self, input):
    s0 = s1 = self.stem(input)
    
    normal_weights = F.softmax(self._arch.normal, dim=-1)
    reduce_weights = F.softmax(self._arch.reduce, dim=-1)
    
    for cell in self.cells:
      s0, s1 = s1, cell(s0, s1, normal_weights if not cell.reduction else reduce_weights)
    
    out = self.global_pooling(s1)
    out = out.view(out.size(0),-1)
    out = self.classifier(out)
    return out
  
  def train_batch(self, data, target, metric_fns=None):
    data_train, data_search = data
    target_train, target_search = target
    self._arch.train_batch(data_search, target_search, forward=self.forward)
    return super().train_batch(data_train, target_train, metric_fns=metric_fns)
