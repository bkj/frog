#!/bin/bash

# scripts/fashion_mnist.sh

source activate py36

PREFIX="results/fashion-mnist/small-20-dev"

# --
# DARTS

for i in $(seq 10); do
    RUN_ID="$PREFIX/search/$i"
    mkdir -p $RUN_ID
    
    CUDA_VISIBLE_DEVICES=1 python main.py --unrolled \
        --outpath $RUN_ID \
        --seed $i \
        --dataset fashion_mnist --epochs 20 --num-layers 1 --op-channels 64 --lr-schedule linear --lr-max 0.1 --cat-last 1 \
        | tee $RUN_ID/log.jl

    ARCH_PATH="$RUN_ID/search_arch_params_efinal.pt"
    CONFIG_PATH="$RUN_ID/search_config.json"    
    python utils/sample-arch.py \
        --arch-path $ARCH_PATH \
        --config-path $CONFIG_PATH \
        --outpath $RUN_ID/genotype.npy
    
    CUDA_VISIBLE_DEVICES=1 python main.py --genotype $RUN_ID/genotype.npy \
        --outpath $RUN_ID \
        --seed $i \
        --dataset fashion_mnist --epochs 20 --num-layers 1 --op-channels 64 --lr-schedule linear --lr-max 0.1 --cat-last 1 \
        | tee $RUN_ID/train.jl
done

# --
# RANDOM

for i in $(seq 100); do
    RUN_ID="$PREFIX/random/$i"
    mkdir -p $RUN_ID
    
    ARCH_PATH="$PREFIX/search/1/search_arch_params_efinal.pt"
    CONFIG_PATH="$PREFIX/search/1/search_config.json"    
    python utils/sample-arch.py \
        --arch-path $ARCH_PATH \
        --config-path $CONFIG_PATH \
        --outpath $RUN_ID/genotype.npy \
        --random --seed $i
    
    CUDA_VISIBLE_DEVICES=0 python main.py --genotype $RUN_ID/genotype.npy \
        --outpath $RUN_ID \
        --seed $i \
        --dataset fashion_mnist --epochs 20 --num-layers 1 --op-channels 64 --lr-schedule linear --lr-max 0.1 --cat-last 1 \
        | tee $RUN_ID/train.jl
done


python utils/plot.py $PREFIX/*/*/train.jl
