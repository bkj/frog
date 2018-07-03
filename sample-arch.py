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
from operations import PRIMITIVES

from collections import namedtuple
Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

np.set_printoptions(linewidth=120)

assert PRIMITIVES[0] == 'none'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--normal-path', type=str, default='results/search/0/normal_arch_e49.pt')
    parser.add_argument('--reduce-path', type=str, default='results/search/0/reduce_arch_e49.pt')
    parser.add_argument('--outpath', type=str)
    
    parser.add_argument('--num-nodes', type=int, default=4)
    parser.add_argument('--cat-last', type=int, default=4)
    parser.add_argument('--as-matrix', action="store_true")
    
    return parser.parse_args()


# --
# Helpers

def parse_genotype(weights, num_nodes):
    gene = []
    start = 0
    weights = weights[:,1:] # Drop nones
    for i in range(num_nodes):
        end = start + 2 + i
        W = weights[start:end].copy()
        top_edges = W.max(axis=-1).argsort()[::-1][:2]
        for edge in top_edges:
            gene.append((PRIMITIVES[W[edge].argmax() + 1], edge))
        
        start = end
    
    return gene


def parse_weights(weights, num_nodes):
    weights = weights.copy()
    start = 0
    for i in range(num_nodes):
        end = start + 2 + i
        W = weights[start:end].copy()
        W_new = np.zeros(W.shape)
        
        top_edges = W[:,1:].max(axis=-1).argsort()[::-1][:2]
        for edge in top_edges:
            W_new[edge,W[edge,1:].argmax() + 1] = 1
        
        weights[start:end] = W_new
        start = end
    
    return weights


if __name__ == "__main__":
    args = parse_args()
    
    normal_logits = torch.load(args.normal_path, map_location=lambda storage, loc: storage)
    reduce_logits = torch.load(args.reduce_path, map_location=lambda storage, loc: storage)
    
    normal_logits = Variable(normal_logits.data)
    reduce_logits = Variable(reduce_logits.data)
    
    normal_weights = to_numpy(F.softmax(normal_logits, dim=-1))
    reduce_weights = to_numpy(F.softmax(reduce_logits, dim=-1))
    
    if not args.as_matrix:
        normal_gene = parse_genotype(normal_weights, num_nodes=args.num_nodes)
        reduce_gene = parse_genotype(reduce_weights, num_nodes=args.num_nodes)
        
        concat = list(range(2 + args.num_nodes - args.cat_last, args.num_nodes + 2))
        genotype = Genotype(
          normal=normal_gene,
          normal_concat=concat,
          reduce=reduce_gene,
          reduce_concat=concat,
        )
        pickle.dump(genotype, open(args.outpath, 'wb'))
    else:
        normal_gene = parse_weights(normal_weights, num_nodes=args.num_nodes)
        reduce_gene = parse_weights(reduce_weights, num_nodes=args.num_nodes)
        genotype    = np.stack([normal_gene, reduce_gene])
        np.save(args.outpath, genotype)
        
    print('hash=%s' % md5(str(genotype).encode()).hexdigest())
    print(genotype)

