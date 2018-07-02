#!/usr/bin/env python

"""
  model_search.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from basenet import BaseNet

from operations import *

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
  def __init__(self, layer_infos):
    super().__init__()
    
    assert len(layer_infos) == 3
    layer_minus_2, layer_minus_1, layer = layer_infos
    
    self.reduction       = layer['reduction']
    self.steps           = layer['steps']
    self.cat_last        = layer['cat_last']
    
    if layer_minus_1['reduction']:
      self.preprocess0 = FactorizedReduce(layer_minus_2['output_channels'], layer['op_channels'], affine=False)
    else:
      self.preprocess0 = ReLUConvBN(layer_minus_2['output_channels'], layer['op_channels'], 1, 1, 0, affine=False)
    
    self.preprocess1 = ReLUConvBN(layer_minus_1['output_channels'], layer['op_channels'], 1, 1, 0, affine=False)
    
    self._ops = nn.ModuleList()
    for step in range(self.steps):
      for idx in range(step + 2):
        stride = 2 if self.reduction and idx < 2 else 1
        self._ops.append(MixedOp(num_channels=layer['op_channels'], stride=stride))
  
  def forward(self, s0, s1, weights):
    states = [
      self.preprocess0(s0),
      self.preprocess1(s1),
    ]
    offset = 0
    for step in range(self.steps):
      s = sum(self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
      offset += len(states)
      states.append(s)
    
    out = states[-self.cat_last:]
    out = torch.cat(out, dim=1)
    return out


class DARTSearchNetwork(BaseNet):
  def __init__(self, arch, in_channels, num_classes, layers, steps=4, cat_last=4, stem_multiplier=3):
    super().__init__(loss_fn=F.cross_entropy)
    
    reduction_idx = [layers//3, 2 * layers//3]
    
    self._arch = arch
    
    # Define architecture
    
    self.stem = nn.Sequential(
      nn.Conv2d(3, stem_multiplier * in_channels, 3, padding=1, bias=False),
      nn.BatchNorm2d(stem_multiplier * in_channels)
    )
    
    self.layer_infos = [
      {"steps" : steps, "cat_last" : 1, "reduction" : False, "op_channels" : stem_multiplier * in_channels, "output_channels" : stem_multiplier * in_channels}, # Stem info
      {"steps" : steps, "cat_last" : 1, "reduction" : False, "op_channels" : stem_multiplier * in_channels, "output_channels" : stem_multiplier * in_channels}, # Stem info
    ]
    self.cells = []
    op_channels = in_channels
    for layer_idx in range(layers):
      reduction = layer_idx in reduction_idx
      if reduction:
        op_channels *= 2
      
      self.layer_infos.append({
        "steps"           : steps,
        "cat_last"        : cat_last,
        "reduction"       : reduction,
        "op_channels"     : op_channels,
        "output_channels" : op_channels * cat_last,
      })
      
      self.cells.append(DARTSearchCell(layer_infos=self.layer_infos[-3:]))
    
    self.cells          = nn.ModuleList(self.cells)
    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier     = nn.Linear(self.layer_infos[-1]['output_channels'], num_classes)
  
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
