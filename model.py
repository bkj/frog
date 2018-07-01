import torch
import torch.nn as nn
from operations import *
from torch.autograd import Variable
from utils import drop_path


def drop_path(x, drop_prob):
  if drop_prob > 0.:
    keep_prob = 1.-drop_prob
    mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
    x.div_(keep_prob)
    x.mul_(mask)
  
  return x


class Cell(nn.Module):
  def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev):
    super().__init__()
    
    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
    
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)
    
    if reduction:
      op_names, indices = zip(*genotype.reduce)
      concat = genotype.reduce_concat
    else:
      op_names, indices = zip(*genotype.normal)
      concat = genotype.normal_concat
    
    assert len(op_names) == len(indices)
    self._steps = len(op_names) // 2
    self._concat = concat
    self.multiplier = len(concat)
    
    self._ops = nn.ModuleList()
    for name, idx in zip(op_names, indices):
      stride = 2 if reduction and idx < 2 else 1
      self._ops.append(OPS[name](C, stride, True))
    
    self._indices = indices
    
  def forward(self, s0, s1, drop_prob):
    states = [
      self.preprocess0(s0),
      self.preprocess1(s1),
    ]
    
    for i in range(self._steps):
      op1 = self._ops[2*i]
      op2 = self._ops[2*i+1]
      
      h1  = op1(states[self._indices[2*i]])
      h2  = op2(states[self._indices[2*i+1]])
      
      if self.training and drop_prob > 0:
        if not isinstance(op1, Identity):
          h1 = drop_path(h1, drop_prob)
        if not isinstance(op2, Identity):
          h2 = drop_path(h2, drop_prob)
      
      s = h1 + h2
      states.append(s)
    
    return torch.cat([states[i] for i in self._concat], dim=1)


class AuxiliaryHeadCIFAR(nn.Module):

  def __init__(self, C, num_classes):
    super().__init__()
    self.features = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False), # image size = 2 x 2
      nn.Conv2d(C, 128, 1, bias=False),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True),
      nn.Conv2d(128, 768, 2, bias=False),
      nn.BatchNorm2d(768),
      nn.ReLU(inplace=True)
    )
    self.classifier = nn.Linear(768, num_classes)

  def forward(self, x):
    x = self.features(x)
    x = x.view(x.size(0),-1)
    x = self.classifier(x)
    return x


class NetworkCIFAR(nn.Module):

  def __init__(self, C, num_classes, layers, auxiliary, genotype, stem_multiplier=3):
    super().__init__()
    
    self._C           = C
    self._num_classes = num_classes
    self._layers      = layers
    
    self._auxiliary   = auxiliary
    
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
      
      cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
      if i == 2 * layers // 3:
        C_to_auxiliary = C_prev
    
    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)
    
    if auxiliary:
      self.auxiliary_head = AuxiliaryHeadCIFAR(C_to_auxiliary, num_classes)

  def forward(self, input):
    logits_aux = None
    s0 = s1 = self.stem(input)
    
    for i, cell in enumerate(self.cells):
      s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
      if i == 2 * self._layers//3:
        if self._auxiliary and self.training:
          logits_aux = self.auxiliary_head(s1)
    
    out = self.global_pooling(s1)
    out = out.view(out.size(0),-1)
    out = self.classifier()
    return out, logits_aux
