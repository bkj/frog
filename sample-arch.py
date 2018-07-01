#!/usr/bin/env python

"""
    sample-arch.py
    
    Notice the pattern of "none" connections -- there's a bias
    towards taking an edge from earlier layers.  This is stronger
    in the normal layers, but still present in the reduce layers.
"""

import pickle
import argparse
import numpy as np
from hashlib import md5
from pprint import pprint

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from basenet.helpers import to_numpy
from genotypes import PRIMITIVES, Genotype

np.set_printoptions(linewidth=120)

assert PRIMITIVES[0] == 'none'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--normal-path', type=str, default='results/search/0/normal_arch_e49.pt')
    parser.add_argument('--reduce-path', type=str, default='results/search/0/reduce_arch_e49.pt')
    parser.add_argument('--outpath', type=str)
    parser.add_argument('--seed', type=int, default=123)
    
    parser.add_argument('--steps', type=int, default=4)
    parser.add_argument('--multiplier', type=int, default=4)
    
    
    return parser.parse_args()


# --
# Helpers

def parse_genotype(weights, steps):
    gene = []
    start = 0
    for i in range(steps):
        end = start + 2 + i
        W = weights[start:end].copy()
        W = W[:,1:] # Drop nones
        top_edges = W.max(axis=-1).argsort()[::-1][:2] # 
        print('--')
        print(top_edges)
        print(weights[start:end])
        for edge in top_edges:
            gene.append((PRIMITIVES[W[edge].argmax() + 1], edge))
        
        start = end
    
    return gene



if __name__ == "__main__":
    args = parse_args()
    
    normal_logits = torch.load(args.normal_path, map_location=lambda storage, loc: storage)
    reduce_logits = torch.load(args.reduce_path, map_location=lambda storage, loc: storage)
    
    normal_logits = Variable(normal_logits.data)
    reduce_logits = Variable(reduce_logits.data)
    
    # normal_logits[:,0] = -1000
    # reduce_logits[:,0] = -1000
    
    normal_weights = to_numpy(F.softmax(normal_logits, dim=-1))
    reduce_weights = to_numpy(F.softmax(reduce_logits, dim=-1))
    
    from rsub import *
    from matplotlib import pyplot as plt
    _ = plt.plot(normal_weights[:,0])
    _ = plt.plot(reduce_weights[:,0])
    show_plot()
    
    import seaborn as sns
    sns.heatmap(normal_weights)
    show_plot()

    import seaborn as sns
    sns.heatmap(reduce_weights)
    show_plot()
    
    normal_gene = parse_genotype(normal_weights, steps=args.steps)
    print('-' * 50)
    reduce_gene = parse_genotype(reduce_weights, steps=args.steps)
    
    # concat = range(2 + args.steps - args.multiplier, args.steps + 2)
    # genotype = Genotype(
    #   normal=normal_gene,
    #   normal_concat=concat,
    #   reduce=reduce_gene,
    #   reduce_concat=concat,
    # )
    # print(md5(str(genotype).encode()).hexdigest())
    # pprint(genotype)
    # pickle.dump(genotype, open(args.outpath, 'wb'))