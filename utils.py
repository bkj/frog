#!/usr/bin/env python

"""
  utils.py
"""

import itertools

class ZipDataloader:
  def __init__(self, dataloaders):
    self.dataloaders = dataloaders
  
  def __len__(self):
    return len(self.dataloaders[0])
  
  def __iter__(self):
    counter = 0
    iters = [iter(self.dataloaders[0])] + [itertools.cycle(iter(d)) for d in self.dataloaders[1:]]
    while counter < len(self):
      yield tuple(zip(*[next(it) for it in iters]))
      counter += 1