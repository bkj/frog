import torch
import torch.nn as nn
import torch.nn.functional as F

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


class Cell(nn.Module):
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
      for j in range(2+i):
        stride = 2 if reduction and j < 2 else 1
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


class Network(nn.Module):
  def __init__(self, C, num_classes, layers, steps=4, multiplier=4, stem_multiplier=3):
    super(Network, self).__init__()
    self._C = C
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
      
      cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, multiplier*C_curr
    
    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier     = nn.Linear(C_prev, num_classes)

  def forward(self, input, arch_parameters):
    s0 = s1 = self.stem(input)
    
    normal_weights = F.softmax(arch_parameters[0], dim=-1)
    reduce_weights = F.softmax(arch_parameters[1], dim=-1)
    
    for cell in self.cells:
      s0, s1 = s1, cell(s0, s1, normal_weights if not cell.reduction else reduce_weights)
    
    out = self.global_pooling(s1)
    out = out.view(out.size(0),-1)
    out = self.classifier(out)
    return out

  # def genotype(self):

  #   def _parse(weights):
  #     gene = []
  #     n = 2
  #     start = 0
  #     for i in range(self._steps):
  #       end = start + n
  #       W = weights[start:end].copy()
  #       edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
  #       for j in edges:
  #         k_best = None
  #         for k in range(len(W[j])):
  #           if k != PRIMITIVES.index('none'):
  #             if k_best is None or W[j][k] > W[j][k_best]:
  #               k_best = k
  #         gene.append((PRIMITIVES[k_best], j))
  #       start = end
  #       n += 1
  #     return gene

  #   gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
  #   gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())

  #   concat = range(2+self._steps-self._multiplier, self._steps+2)
  #   genotype = Genotype(
  #     normal=gene_normal, normal_concat=concat,
  #     reduce=gene_reduce, reduce_concat=concat
  #   )
  #   return genotype

