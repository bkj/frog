#!/usr/bin/env python

"""
    plot.py
"""

import os
import sys
import json
import pandas as pd
import numpy as np

from rsub import *
from matplotlib import pyplot as plt

def smart_json_loads(x):
    try:
        return json.loads(x)
    except:
        pass

all_data = []

frog_final = []
rand_final = []
for p in sys.argv[1:]:
    x = list(map(json.loads, open(p).read().splitlines()))
    valid_acc = np.array([xx['valid_acc'] for xx in x])
    
    if 'random' in p:
        c = 'grey'
    else:
        c= 'red'
    
    _ = plt.plot(valid_acc, alpha=0.75, label='%s_valid' % os.path.basename(p), c=c)
    
    if len(valid_acc):
        if 'random' in p:
            rand_final.append(valid_acc[-1])
        else:
            frog_final.append(valid_acc[-1])
    
    # train_acc = np.array([xx['train_acc'] for xx in x])
    # _ = plt.plot(train_acc, alpha=0.75, label='%s_train' % os.path.basename(p))

_ = plt.grid(alpha=0.25)
for t in np.arange(0.90, 1.0, 0.01):
    _ = plt.axhline(t, c='grey', alpha=0.25, lw=1)

# _ = plt.legend(fontsize=4)
_ = plt.ylim(0.50, 1.0)
show_plot()

# _ = plt.hist(frog_final, 32, alpha=0.25)
# _ = plt.hist(rand_final, 32, alpha=0.25)
# show_plot()
