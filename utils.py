#!/usr/bin/env python

"""
  utils.py
"""

import itertools

def loopy_wrapper(gen):
  while True:
    for x in gen:
      yield x


class ZipDataloader:
  def __init__(self, dataloaders):
    self.dataloaders = dataloaders
  
  def __len__(self):
    return len(self.dataloaders[0])
  
  def __iter__(self):
    counter = 0
    iters = [loopy_wrapper(d) for d in self.dataloaders]
    while counter < len(self):
      yield tuple(zip(*[next(it) for it in iters]))
      counter += 1
