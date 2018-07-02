#!/usr/bin/env python

"""
    model_search.py
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from basenet import BaseNet

from operations import *

class DARTEdge(nn.Module):
  def __init__(self, id, num_channels, stride, primitives=PRIMITIVES, simple_repr=True):
    super().__init__()
    
    self.name = 'edge(%d)' % id
    
    self._primitives  = primitives
    self._simple_repr = simple_repr
    
    for primitive in primitives:
      op = OPS[primitive](num_channels, stride, False)
      if 'pool' in primitive:
        op = nn.Sequential(op, nn.BatchNorm2d(num_channels, affine=False))
      self.add_module(primitive, op)
  
  def forward(self, x, weights):
    return sum(w * getattr(self, p)(x) for w, p in zip(weights, self._primitives)) # Slower?
  
  def __repr__(self):
    if self._simple_repr:
      return 'DARTEdge(%s)' % str(self._primitives)
    else:
      return super().__repr__()


class DARTNode(nn.Module):
  def __init__(self, id):
    super().__init__()
    
    self.name = 'node(%d)' % id
    self.meta = []


class DARTSearchCell(nn.Module):
  def __init__(self, layer_infos):
    super().__init__()
    self.meta = []
    
    assert len(layer_infos) == 3
    layer_minus_2, layer_minus_1, layer = layer_infos
    
    self.reduction = layer['reduction']
    self.num_nodes = layer['num_nodes']
    self.cat_last  = layer['cat_last']
    
    if layer_minus_1['reduction']:
      self.prep0 = FactorizedReduce(layer_minus_2['output_channels'], layer['op_channels'], affine=False)
    else:
      self.prep0 = ReLUConvBN(layer_minus_2['output_channels'], layer['op_channels'], 1, 1, 0, affine=False)
    
    self.prep1 = ReLUConvBN(layer_minus_1['output_channels'], layer['op_channels'], 1, 1, 0, affine=False)
    
    weight_offset = 0
    for node_idx in range(self.num_nodes):
      node = DARTNode(id=node_idx)
      for state_offset, edge_idx in enumerate(range(-2, node_idx)):
        
        stride = 2 if self.reduction and edge_idx < 0 else 1
        edge = DARTEdge(id=edge_idx, num_channels=layer['op_channels'], stride=stride)
        
        node.add_module(edge.name, edge)
        node.meta.append({
          "edge_name"     : edge.name,
          "weight_offset" : weight_offset,
          "state_offset"  : state_offset,
        })
        
        weight_offset += 1
      
      self.add_module(node.name, node)
      self.meta.append({
        "node_name" : node.name
      })
  
  def forward(self, states, weights):
    states = [
      self.prep0(states[0]),
      self.prep1(states[1]),
    ]
    for node_meta in self.meta:
      node = getattr(self, node_meta['node_name'])
      s = sum(getattr(node, edge_meta['edge_name'])(states[edge_meta['state_offset']], weights[edge_meta['weight_offset']]) for edge_meta in node.meta)
      states.append(s)
    
    out = states[-self.cat_last:]
    out = torch.cat(out, dim=1)
    return out


class DARTSearchNetwork(BaseNet):
  def __init__(self, arch, in_channels, num_classes, num_layers, num_nodes=4, cat_last=4, stem_multiplier=3, num_inputs=2):
    super().__init__(loss_fn=F.cross_entropy)
    
    reduction_idxs = [num_layers // 3, 2 * num_layers // 3]
    
    self._arch = arch
    
    self.stem = nn.Sequential(
      nn.Conv2d(3, stem_multiplier * in_channels, 3, padding=1, bias=False),
      nn.BatchNorm2d(stem_multiplier * in_channels)
    )
    
    self.layer_infos = [
      {"num_nodes" : num_nodes, "cat_last" : 1, "reduction" : False, "op_channels" : stem_multiplier * in_channels, "output_channels" : stem_multiplier * in_channels}, # Stem info
    ] * num_inputs
    self.cells = []
    op_channels = in_channels
    for layer_idx in range(num_layers):
      reduction = layer_idx in reduction_idxs
      if reduction:
        op_channels *= 2
      
      self.layer_infos.append({
        "num_nodes"       : num_nodes,
        "cat_last"        : cat_last,
        "reduction"       : reduction,
        "op_channels"     : op_channels,
        "output_channels" : op_channels * cat_last,
      })
      
      self.cells.append(DARTSearchCell(layer_infos=self.layer_infos[-(num_inputs+1):]))
    
    self.cells          = nn.ModuleList(self.cells)
    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier     = nn.Linear(self.layer_infos[-1]['output_channels'], num_classes)
    
    self.num_inputs = num_inputs
  
  def forward(self, x):
    normal_weights, reduce_weights = self._arch.get_weights()
    
    x = self.stem(x)
    states = [x] * self.num_inputs
    for cell in self.cells:
      states = states[1:] + [cell(states, normal_weights if not cell.reduction else reduce_weights)]
    
    out = self.global_pooling(states[-1])
    out = out.view(out.size(0),-1)
    out = self.classifier(out)
    return out
  
  def train_batch(self, data, target, metric_fns=None):
    data_train, data_search = data
    target_train, target_search = target
    self._arch.train_batch(data_search, target_search, forward=self.forward)
    return super().train_batch(data_train, target_train, metric_fns=metric_fns)
