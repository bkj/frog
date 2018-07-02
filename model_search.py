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
  def __init__(self, num_channels, stride):
    super().__init__()
    
    self._ops = nn.ModuleList()
    for primitive in PRIMITIVES:
      op = OPS[primitive](num_channels, stride, False)
      if 'pool' in primitive:
        op = nn.Sequential(op, nn.BatchNorm2d(num_channels, affine=False))
      self._ops.append(op)
      
  def forward(self, x, weights):
    return sum(w * op(x) for w, op in zip(weights, self._ops))


class DARTSearchCell(nn.Module):
  def __init__(self, steps, multiplier, num_channels, reduction, prev_layers):
    super().__init__()
    
    self.reduction = reduction
    self.output_channels = multiplier * num_channels
    
    self._steps      = steps
    self._multiplier = multiplier
    
    if reduction_prev:
      self.preprocess0 = FactorizedReduce(prev_layers[0]['channels'], num_channels, affine=False)
    else:
      self.preprocess0 = ReLUConvBN(prev_layers[0]['channels'], num_channels, 1, 1, 0, affine=False)
    
    self.preprocess1 = ReLUConvBN(prev_layers[1]['channels'], num_channels, 1, 1, 0, affine=False)
    
    self._ops = nn.ModuleList()
    for i in range(self._steps):
      for idx in range(i+2):
        stride = 2 if reduction and idx < 2 else 1
        self._ops.append(MixedOp(num_channels=num_channels, stride=stride))
  
  def forward(self, s0, s1, weights):
    states = [
      self.preprocess0(s0),
      self.preprocess1(s1),
    ]
    offset = 0
    for _ in range(self._steps):
      s = sum(self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
      offset += len(states)
      states.append(s)
      
    return torch.cat(states[-self._multiplier:], dim=1)


class DARTSearchNetwork(BaseNet):
  def __init__(self, arch, num_channels, num_classes, layers, steps=4, multiplier=4, stem_multiplier=3):
    super().__init__(loss_fn=F.cross_entropy)
    
    reduction_idx = [layers//3, 2 * layers//3]
    
    self._arch        = arch
    self._num_classes = num_classes
    self._layers      = layers
    
    self._steps       = steps
    self._multiplier  = multiplier
    
    self.stem = nn.Sequential(
      nn.Conv2d(3, stem_multiplier * num_channels, 3, padding=1, bias=False),
      nn.BatchNorm2d(stem_multiplier * num_channels)
    )
    
    self.layer_infos = [
      {"steps" : steps, "reduction" : False, "channels" : stem_multiplier * num_channels},
      {"steps" : steps, "reduction" : False, "channels" : stem_multiplier * num_channels},
      {"steps" : steps, "reduction" : False, "channels" : num_channels},
    ]
    for layer_idx in range(layers):
      num_channels *= multiplier
      reduction = layer_idx in reduction_idx
      if reduction:
        num_channels *= 2
      
      self.layer_infos.append({
        "steps"     : steps,
        "reduction" : reduction,
        "channels"  : num_channels
      })
    
    self.cells = nn.ModuleList([DARTSearchCell(layer_infos=self.layer_infos[i-2:i+1]) 
      for layer_idx in range(2, len(self.layer_infos))])
    
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
