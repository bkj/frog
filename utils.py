#!/usr/bin/env python

"""
  utils.py
"""

import itertools

class ZipDataloader:
  def __init__(self, dataloaders):
    self.dataloaders = dataloaders
  
  def __len__(self):
    return max([len(d) for d in self.dataloaders])
  
  def __iter__(self):
    counter = 0
    iters = [itertools.cycle(iter(d)) for d in self.dataloaders]
    while counter < len(self):
      yield tuple(zip(*[next(it) for it in iters]))
      counter += 1