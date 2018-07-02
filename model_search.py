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
    
    self.ops = nn.ModuleList()
    for primitive in PRIMITIVES:
      op = OPS[primitive](num_channels, stride, False)
      if 'pool' in primitive:
        op = nn.Sequential(op, nn.BatchNorm2d(num_channels, affine=False))
      self.ops.append(op)
      
  def forward(self, x, weights):
    return sum(w * op(x) for w, op in zip(weights, self.ops))


class DARTSearchCell(nn.Module):
  def __init__(self, layer_infos):
    super().__init__()
    
    print(layer_infos)
    assert len(layer_infos) == 3
    layer_minus_2, layer_minus_1, layer = layer_infos
    
    self.channels     = layer['channels']
    self.reduction    = layer['reduction']
    self.steps        = layer['steps']
    self.cat_last     = layer['cat_last']
    
    op_channels = self.channels // self.cat_last
    
    if layer_minus_1['reduction']:
      self.preprocess0 = FactorizedReduce(layer_minus_2['channels'], op_channels, affine=False)
    else:
      self.preprocess0 = ReLUConvBN(layer_minus_2['channels'], op_channels, 1, 1, 0, affine=False)
    
    self.preprocess1 = ReLUConvBN(layer_minus_1['channels'], op_channels, 1, 1, 0, affine=False)
    
    self.ops = nn.ModuleList()
    for i in range(self.steps):
      for idx in range(i+2):
        stride = 2 if self.reduction and idx < 2 else 1
        self.ops.append(MixedOp(num_channels=op_channels, stride=stride))
  
  def forward(self, s0, s1, weights):
    states = [
      self.preprocess0(s0),
      self.preprocess1(s1),
    ]
    offset = 0
    for _ in range(self.steps):
      s = sum(self.ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
      offset += len(states)
      states.append(s)
      
    return torch.cat(states[-self.cat_last:], dim=1)


class DARTSearchNetwork(BaseNet):
  def __init__(self, arch, num_channels, num_classes, layers, steps=4, cat_last=4, stem_multiplier=3):
    super().__init__(loss_fn=F.cross_entropy)
    
    reduction_idx = [layers//3, 2 * layers//3]
    
    self._arch        = arch
    self._num_classes = num_classes
    self._layers      = layers
    
    self._steps     = steps
    self._cat_last  = cat_last
    
    self.stem = nn.Sequential(
      nn.Conv2d(3, stem_multiplier * num_channels, 3, padding=1, bias=False),
      nn.BatchNorm2d(stem_multiplier * num_channels)
    )
    
    self.layer_infos = [
      {"steps" : steps, "cat_last" : cat_last, "reduction" : False, "channels" : stem_multiplier * num_channels},
      {"steps" : steps, "cat_last" : cat_last, "reduction" : False, "channels" : stem_multiplier * num_channels},
      {"steps" : steps, "cat_last" : cat_last, "reduction" : False, "channels" : num_channels},
    ]
    for layer_idx in range(1, layers):
      num_channels *= self.layer_infos[-1]['cat_last']
      reduction = layer_idx in reduction_idx
      if reduction:
        num_channels *= 2
      
      self.layer_infos.append({
        "steps"     : steps,
        "cat_last"  : cat_last,
        "reduction" : reduction,
        "channels"  : num_channels,
      })
    
    self.cells = nn.ModuleList([DARTSearchCell(layer_infos=self.layer_infos[layer_idx-3:layer_idx]) 
      for layer_idx in range(3, len(self.layer_infos) + 1)])
    
    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier     = nn.Linear(self.layer_infos[-1]['channels'], num_classes)
  
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
