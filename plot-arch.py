#!/usr/bin/env python

"""
    plot-arch.py
"""

import sys
from rsub import *
from matplotlib import pyplot as plt
import seaborn as sns

import torch
from torch.nn import functional as F

from basenet.helpers import to_numpy

arch = torch.load(sys.argv[1], map_location=lambda *x: x[0])
arch = F.softmax(arch, dim=-1)
arch = to_numpy(arch)

# Heatmap of weights
_ = sns.heatmap(arch)
show_plot()

# Plot "none" prob
_ = plt.plot(arch[:,0])
_ = plt.axvline(1, c='grey')
_ = plt.axvline(4, c='grey')
_ = plt.axvline(8, c='grey')
_ = plt.axvline(13, c='grey')
show_plot()