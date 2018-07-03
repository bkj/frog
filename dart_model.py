#!/usr/bin/env python

"""
    dart_model.py
"""

import os
import sys
import numpy as np
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from basenet import BaseNet
from basenet.helpers import to_device, set_seeds

from operations import *

TORCH_VERSION_4 = '0.4' == torch.__version__[:3]

# --
# Architecture


class DARTArchitecture(BaseNet):
  def __init__(self, num_nodes, num_ops, num_prev=2, scale=1e-3):
    super().__init__(loss_fn=F.cross_entropy)
    
    self.num_nodes = num_nodes
    self.num_ops   = num_ops
    self.num_prev  = num_prev
    
    n_edges     = num_prev * num_nodes + sum(range(num_nodes)) # Connections to inputs + connections to earlier num_nodes
    self.normal = nn.Parameter(torch.Tensor(np.random.normal(0, scale, (n_edges, num_ops))))
    self.reduce = nn.Parameter(torch.Tensor(np.random.normal(0, scale, (n_edges, num_ops))))
  
  def get_logits(self):
    return self.normal, self.reduce
    
  def get_weights(self):
    # !! Could do other logic in here
    normal_weights = F.softmax(self.normal, dim=-1)
    reduce_weights = F.softmax(self.reduce, dim=-1)
    return normal_weights, reduce_weights
  
  def __repr__(self):
    return 'DARTArchitecture(num_nodes=%d | num_ops=%d)' % (self.num_nodes, self.num_ops)
  
  def unrolled_train_batch(self, model, data_train, target_train, data_search, target_search, **kwargs):
    # !! Could probably be simplified
    
    assert self.loss_fn is not None, 'DARTArchitecture: self.loss_fn is None'
    assert self.training, 'DARTArchitecture: self.training == False'
    
    self.opt.zero_grad()
    
    if not TORCH_VERSION_4:
        data_train, target_train = Variable(data_train), Variable(target_train)
        data_search, target_search = Variable(data_search), Variable(target_search)
    
    data_train, target_train = to_device(data_train, self.device), to_device(target_train, self.device)
    data_search, target_search = to_device(data_search, self.device), to_device(target_search, self.device)
    
    search_loss = self._compute_unrolled_grads(model, data_train, target_train, data_search, target_search)
    
    if self.clip_grad_norm > 0:
        if TORCH_VERSION_4:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.params, self.clip_grad_norm)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm(self.params, self.clip_grad_norm)
    
    self.opt.step()
    
    return float(search_loss)

  def _compute_unrolled_grads(self, model, data_train, target_train, data_search, target_search):
    
    # Store model state
    state_dict     = deepcopy(model.state_dict())
    opt_state_dict = deepcopy(model.opt.state_dict())
    wd = model.opt.state_dict()['param_groups'][0]['weight_decay']
    assert wd == 3e-4 # !! Sanity check for now
    
    # Take a step
    model.opt.zero_grad()
    train_loss = self.loss_fn(model(data_train), target_train)
    train_loss.backward()
    model.opt.step()
    
    # Compute grad w.r.t architecture
    search_loss    = self.loss_fn(model(data_search), target_search)
    grads          = torch.autograd.grad(search_loss, [self.normal, self.reduce], retain_graph=True)
    dtheta         = torch.autograd.grad(search_loss, model.parameters())
    vector         = [dt.add(wd, t).data for dt, t in zip(dtheta, model.parameters())]
    implicit_grads = self._hessian_vector_product(model, vector, data_train, target_train)
    _ = [g.data.sub_(model.hp['lr'], ig.data) for g, ig in zip(grads, implicit_grads)]
    
    # Reset model
    model.load_state_dict(state_dict)
    model.opt.load_state_dict(opt_state_dict)
    
    # Add gradient to architecture
    for v, g in zip([self.normal, self.reduce], grads):
      if v.grad is None:
        v.grad = g.data
      else:
        v.grad.data.copy_(g.data)
      
    return float(search_loss)
    
  def _hessian_vector_product(self, model, vector, data, target, r=1e-2):
    # Changes model weights
    
    R = r / torch.cat([v.view(-1) for v in vector]).norm()
    
    # plus R
    _ = [p.data.add_(R, v) for p, v in zip(model.parameters(), vector)]
    grads_pos = torch.autograd.grad(self.loss_fn(model(data), target), [self.normal, self.reduce])
    
    # minus R
    _ = [p.data.sub_(2*R, v) for p, v in zip(model.parameters(), vector)]
    grads_neg = torch.autograd.grad(self.loss_fn(model(data), target), [self.normal, self.reduce])
    
    return [(x - y).div_(2*R) for x, y in zip(grads_pos, grads_neg)]


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


class _DARTNetwork(BaseNet):
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


class DARTSearchNetwork(_DARTNetwork):
  def __init__(self, arch, unrolled=False, **kwargs):
    super().__init__(**kwargs)
    assert self.loss_fn == arch.loss_fn, "DARTSearchNetwork.loss_fn != arch.loss_fn"
    
    # A little awkward, but we don't want arch.parameters to show up in model.parameters
    self._unrolled         = unrolled
    self._arch_get_weights = arch.get_weights
    self._arch_get_logits  = arch.get_logits
    self._arch_train_batch = arch.unrolled_train_batch if unrolled else arch.train_batch
    
    self._fixed = False
  
  def train_batch(self, data, target, metric_fns=None):
    data_train, data_search = data
    target_train, target_search = target
    if self._unrolled:
      self._arch_train_batch(model=self, data_train=data_train, target_train=target_train, data_search=data_search, target_search=target_search)
    else:
      self._arch_train_batch(data_search, target_search, forward=self.forward)
    
    loss, metrics = super().train_batch(data_train, target_train, metric_fns=metric_fns)
    print('loss', float(loss))
    return loss, metrics
  
  def checkpoint(self, outpath, epoch):
    torch.save(self.state_dict(), os.path.join(outpath, 'weights.pt'))
    normal_logits, reduce_logits = self._arch_get_logits()
    torch.save(normal_logits, os.path.join(outpath, 'normal_arch_e%d.pt' % epoch))
    torch.save(reduce_logits, os.path.join(outpath, 'reduce_arch_e%d.pt' % epoch))


class DARTTrainNetwork(_DARTNetwork):
  def __init__(self, genotype, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._fixed = True
    for cell in self.cells:
      cell.fix_weights(genotype)
  
  def checkpoint(self, outpath, epoch):
    torch.save(self.state_dict(), os.path.join(outpath, 'weights_e%d.pt' % epoch))