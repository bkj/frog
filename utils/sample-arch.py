#!/usr/bin/env python

"""
    utils/sample-arch.py
"""

import json
import pickle
import argparse
import numpy as np
from hashlib import md5
from collections import namedtuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from basenet.helpers import to_numpy

np.set_printoptions(linewidth=120)
Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch-path', type=str)
    parser.add_argument('--config-path', type=str)
    parser.add_argument('--outpath', type=str, required=True)
    
    parser.add_argument('--dart-format', action="store_true")
    parser.add_argument('--random', action="store_true")
    parser.add_argument('--seed', type=int, default=123)
    return parser.parse_args()


# --
# Helpers

def prep_weight(logits, random):
    logits = Variable(logits.data)
    if random:
        logits.data = torch.randn(logits.shape)
    
    return to_numpy(F.softmax(logits, dim=-1))


def parse_genotype(weights, num_nodes):
    raise Exception('!! need to point this to primitives')
    from operations import PRIMITIVES
    assert PRIMITIVES[0] == 'none'
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
    
    logits = torch.load(args.arch_path, map_location=lambda storage, loc: storage)
    config = json.load(open(args.config_path))
    
    num_nodes = config['num_nodes']
    cat_last  = config['cat_last']
    
    weights = [prep_weight(logit, random=args.random) for logit in logits]
    
    if args.dart_format:
        assert len(weights) == 2, "len(weights) != 2 -- unsupported by Genotype"
        concat = list(range(2 + num_nodes - cat_last, num_nodes + 2))
        genotype = Genotype(
          normal=parse_genotype(normal_weights, num_nodes=num_nodes),
          normal_concat=concat,
          reduce=parse_genotype(reduce_weights, num_nodes=num_nodes),
          reduce_concat=concat,
        )
        pickle.dump(genotype, open(args.outpath, 'wb'))
    else:
        genes    = [parse_weights(weight, num_nodes=num_nodes) for weight in weights]
        genotype = np.stack(genes)
        np.save(args.outpath, genotype)
        
    print('hash=%s' % md5(str(genotype).encode()).hexdigest())
    print(genotype)

