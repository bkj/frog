#!/usr/bin/env python

"""
    frog.py
"""

import os
import sys
import numpy as np
from time import time
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from basenet import BaseNet
from basenet.helpers import to_device, set_seeds

TORCH_VERSION_4 = '0.4' == torch.__version__[:3]

# --
# Architecture

class FROGArchitecture(BaseNet):
    def get_params(self):
        return self._arch_params
    
    def get_weights(self):
        return [F.softmax(w, dim=-1) for w in self._arch_params]
    
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
        assert model.hp is not None, '_compute_unrolled_grads: model.hp is None'
        assert 'lr' in model.hp, '_compute_unrolled_grads: lr not in model.hp'
        
        # Store model state
        model = model.deepcopy()
        wd = model.opt.state_dict()['param_groups'][0]['weight_decay']
        
        # Take a step
        model.opt.zero_grad()
        train_loss = self.loss_fn(model(data_train), target_train)
        train_loss.backward()
        model.opt.step()
        
        # Compute grad w.r.t architecture
        search_loss    = self.loss_fn(model(data_search), target_search)
        grads          = torch.autograd.grad(search_loss, self._arch_params, retain_graph=True)
        dtheta         = torch.autograd.grad(search_loss, model.parameters())
        dtheta         = [dt.add(wd, t).data for dt, t in zip(dtheta, model.parameters())]
        implicit_grads = self._hessian_vector_product(model, dtheta, data_train, target_train)
        _ = [g.data.sub_(model.hp['lr'], ig.data) for g, ig in zip(grads, implicit_grads)]
        
        # Add gradient to architecture
        for v, g in zip(self._arch_params, grads):
            if v.grad is None:
                v.grad = g.data
            else:
                v.grad.data.copy_(g.data)
          
        return float(search_loss)
    
    def _hessian_vector_product(self, model, dtheta, data, target, r=1e-2):
        # Changes model weights
        
        R = r / torch.cat([v.view(-1) for v in dtheta]).norm()
        
        # plus R
        _ = [p.data.add_(R, v) for p, v in zip(model.parameters(), dtheta)]
        grads_pos = torch.autograd.grad(self.loss_fn(model(data), target), self._arch_params)
        
        # minus R
        _ = [p.data.sub_(2*R, v) for p, v in zip(model.parameters(), dtheta)]
        grads_neg = torch.autograd.grad(self.loss_fn(model(data), target), self._arch_params)
        
        return [(x - y).div_(2*R) for x, y in zip(grads_pos, grads_neg)]


class FROGSearchMixin:
  def init_search(self, arch, unrolled=False):
    # A little awkward, but we don't want arch.parameters to show up in model.parameters
    self._fixed            = False
    self._unrolled         = unrolled
    self._arch_get_weights = arch.get_weights
    self._arch_get_params  = arch.get_params
    self._arch_train_batch = arch.unrolled_train_batch if unrolled else arch.train_batch
    
    self.train_batch = self._search_train_batch
  
  def init_train(self, genotype):
    assert getattr(self, '_fixed', None) is None, "FROGSearchMixin: cannot call `init_train` right now"
    self._fixed = True
    for cell in self.cells:
      cell.fix_weights(genotype)
  
  def deepcopy(self):
    assert self._fixed == False, "FROGSearchMixin: can only call `deepcopy` on a search network"
    new_model = super().deepcopy()
    new_model._arch_get_weights = self._arch_get_weights
    new_model._arch_get_params  = self._arch_get_params 
    new_model._arch_train_batch = self._arch_train_batch
    return new_model
  
  def _search_train_batch(self, data, target, metric_fns=None):
    data_train, data_search = data
    target_train, target_search = target
    
    if self._unrolled:
      self._arch_train_batch(model=self, data_train=data_train, target_train=target_train, data_search=data_search, target_search=target_search)
    else:
      self._arch_train_batch(data_search, target_search, forward=self.forward)
    
    return super().train_batch(data_train, target_train, metric_fns=metric_fns)
  
  def checkpoint(self, outpath, epoch):
    if self._fixed:
        torch.save(self.state_dict(), os.path.join(outpath, 'fixed_weights_e%s.pt' % str(epoch)))
    else:
        torch.save(self.state_dict(), os.path.join(outpath, 'search_weights_e%s.pt' % str(epoch)))
        torch.save(self._arch_get_params(), os.path.join(outpath, 'search_arch_params_e%s.pt' % str(epoch)))
