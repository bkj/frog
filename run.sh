#!/bin/bash

# run.sh

source activate dart_env

rm -rf results/search/1
mkdir -p results/search/1
CUDA_VISIBLE_DEVICES=5 python main.py \
    --outpath results/search/1 \
    --unrolled | tee results/search/1/log.jl

python sample-arch.py \
    --normal-path results/search/0/normal_arch_e49.pt \
    --reduce-path results/search/0/reduce_arch_e49.pt \
    --outpath results/search/0/genotype.pkl

python sample-arch.py \
    --normal-path results/search/0/normal_arch_e49.pt \
    --reduce-path results/search/0/reduce_arch_e49.pt \
    --as-matrix \
    --outpath results/search/0/genotype.npy

CUDA_VISIBLE_DEVICES=4 python main.py --outpath results/search/1 |\
    tee results/search/1/log.jl

# --

EXP_DIR="results/search/mnist/0/"
# rm -rf $EXP_DIR
mkdir -p $EXP_DIR
CUDA_VISIBLE_DEVICES=1 python main.py \
    --outpath $EXP_DIR \
    --dataset fashion_mnist \
    --unrolled |tee $EXP_DIR/search.jl

i=4
python sample-arch.py \
    --normal-path $EXP_DIR/normal_arch_e$i.pt \
    --reduce-path $EXP_DIR/reduce_arch_e$i.pt \
    --as-matrix \
    --outpath $EXP_DIR/genotype-$i.npy

mkdir -p $EXP_DIR/$i
CUDA_VISIBLE_DEVICES=1 python main.py \
    --genotype $EXP_DIR/genotype-$i.npy \
    --outpath $EXP_DIR/$i \
    --epochs 10 \
    --lr-max 0.1 \
    --dataset fashion_mnist \
    --unrolled | tee $EXP_DIR/$i/train.jl