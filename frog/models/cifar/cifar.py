#!/usr/bin/env python

"""
  cifar.py
"""

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from basenet import BaseNet

from .operations import *
from ...frog import FROGArchitecture, FROGSearchMixin

# --
# Architecture

class Architecture(FROGArchitecture):
    def __init__(self, num_nodes, num_prev=2, scale=1e-3):
        super().__init__(loss_fn=F.cross_entropy)
        
        num_ops = len(PRIMITIVES)
        
        self.num_nodes = num_nodes
        self.num_ops   = num_ops
        self.num_prev  = num_prev
        
        n_edges     = num_prev * num_nodes + sum(range(num_nodes)) # Connections to inputs + connections to earlier num_nodes
        self.normal = nn.Parameter(torch.Tensor(np.random.normal(0, scale, (n_edges, num_ops))))
        self.reduce = nn.Parameter(torch.Tensor(np.random.normal(0, scale, (n_edges, num_ops))))
        
        self._arch_params = [self.normal, self.reduce]
        
    def __repr__(self):
        return 'CIFARArchitecture(num_nodes=%d | num_ops=%d)' % (self.num_nodes, self.num_ops)

# --
# DART Network

class DARTEdge(nn.Module):
  def __init__(self, id, num_channels, stride, primitives=PRIMITIVES, simple_repr=True):
    super().__init__()
    
    self.name = 'edge(%d)' % id
    
    self._primitives   = primitives
    self._simple_repr  = simple_repr
    self._num_channels = num_channels
    
    self._fixed   = False
    self._weights = None
    
    for primitive in primitives:
      op = OPS[primitive](num_channels, stride, False)
      if 'pool' in primitive:
        op = nn.Sequential(op, nn.BatchNorm2d(num_channels, affine=False)) # No batchnorm when self._fixed
      self.add_module(primitive, op)
  
  def forward(self, x, weights):
    if self._fixed:
      weights = self._weights
      assert len(weights) == len(self._primitives)
    
    return sum(w * getattr(self, p)(x) for w, p in zip(weights, self._primitives)) # Slower?
  
  def fix_weights(self, weights):
    # !! Untested
    assert not self._fixed, 'DARTEdge: already fixed'
    self._fixed = True
    tmp_primitives, tmp_weights = [], []
    for w, p in zip(weights, self._primitives):
      if w > 0:
        tmp_primitives.append(p)
        tmp_weights.append(float(w))
      else:
        delattr(self, p)
    
    self._primitives = tmp_primitives
    self._weights    = tmp_weights
  
  def __repr__(self):
    if self._simple_repr:
      return 'DARTEdge(%s | num_channels=%d)' % (str(self._primitives), self._num_channels)
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
    assert len(layer_infos) == 3
    layer_minus_2, layer_minus_1, layer = layer_infos
    
    self.meta = []
    self._fixed = False
    self._weights = None
    
    self.reduction = layer['reduction']
    self.num_nodes = layer['num_nodes']
    self.cat_last  = layer['cat_last']
    
    # ?? Affine=True when self._fixed
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
      if not self._fixed:
        s = sum(getattr(node, edge_meta['edge_name'])(states[edge_meta['state_offset']], weights=weights[edge_meta['weight_offset']]) for edge_meta in node.meta)
      else:
        s = sum(getattr(node, edge_meta['edge_name'])(states[edge_meta['state_offset']], weights=None) for edge_meta in node.meta)
      
      states.append(s)
    
    out = states[-self.cat_last:]
    out = torch.cat(out, dim=1)
    return out
  
  def fix_weights(self, fixed_weights):
    # !! Untested
    if not self.reduction:
      fixed_weights = fixed_weights[0]
    else:
      fixed_weights = fixed_weights[1]
    
    assert not self._fixed, 'DARTEdge: already fixed'
    self._fixed = True
    for node_meta in self.meta:
      node = getattr(self, node_meta['node_name'])
      tmp_node_meta = []
      for edge_meta in node.meta:
        w = fixed_weights[edge_meta['weight_offset']]
        if w.max() > 0:
          getattr(node, edge_meta['edge_name']).fix_weights(w)
          tmp_node_meta.append(edge_meta)
        else:
          delattr(node, edge_meta['edge_name'])
        
        node.meta = tmp_node_meta
  
  def __repr__(self):
    if self.reduction:
      return 'Reduce' + super().__repr__()
    else:
      return super().__repr__()


class Network(FROGSearchMixin, BaseNet):
  def __init__(self, in_channels, op_channels, num_classes, num_layers, num_nodes=4, cat_last=4, stem_multiplier=3, num_inputs=2):
    super().__init__(loss_fn=F.cross_entropy)
    
    reduction_idxs = [num_layers // 3, 2 * num_layers // 3]
    
    self.stem = nn.Sequential(
      nn.Conv2d(in_channels, stem_multiplier * op_channels, kernel_size=3, padding=1, bias=False),
      nn.BatchNorm2d(stem_multiplier * op_channels)
    )
    
    self.layer_infos = [
      {"num_nodes" : num_nodes, "cat_last" : 1, "reduction" : False, "op_channels" : stem_multiplier * op_channels, "output_channels" : stem_multiplier * op_channels}, # Stem info
    ] * num_inputs
    self.cells = []
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
    weights = None
    if not self._fixed:
      normal_weights, reduce_weights = self._arch_get_weights()
    
    x = self.stem(x)
    states = [x] * self.num_inputs
    for cell in self.cells:
      if not self._fixed:
        weights = normal_weights if not cell.reduction else reduce_weights
      states = states[1:] + [cell(states, weights)]
    
    out = self.global_pooling(states[-1])
    out = out.view(out.size(0),-1)
    out = self.classifier(out)
    return out
