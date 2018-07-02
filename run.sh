#!/bin/bash

# run.sh

source activate dart_env

rm -rf results/search/0
mkdir -p results/search/0

# CUDA_VISIBLE_DEVICES=1 python train_search.py --outpath results/search/0 |\
#     tee results/search/0/log.jl

# python sample-arch.py \
#     --normal-path results/search/0/normal_arch_e49.pt \
#     --reduce-path results/search/0/reduce_arch_e49.pt \
#     --outpath genotype.pkl

CUDA_VISIBLE_DEVICES=4 python train_search.py --outpath results/search/1 |\
    tee results/search/1/log.jl

# --

rm -rf results/search-mnist/0
mkdir -p results/search-mnist/0

CUDA_VISIBLE_DEVICES=1 python train_search_mnist.py --outpath results/search-mnist/0 |\
    tee results/search-mnist/0/log.jl