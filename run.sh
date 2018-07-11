#!/bin/bash

# run.sh

source activate py36


RUN_ID="results/search/2"
rm -rf $RUN_ID
mkdir -p $RUN_ID
mkdir -p $RUN_ID/train
CUDA_VISIBLE_DEVICES=5 python main.py \
    --outpath $RUN_ID \
    --unrolled | tee $RUN_ID/log.jl

python sample-arch.py \
    --normal-path $RUN_ID/normal_arch_e49.pt \
    --reduce-path $RUN_ID/reduce_arch_e49.pt \
    --outpath $RUN_ID/genotype.pkl

# CUDA_VISIBLE_DEVICES=7 python main.py \
#     --genotype $RUN_ID/genotype.npy \
#     --layers 20 \
#     --epochs 600 \
#     --outpath $RUN_ID/train | tee $RUN_ID/train.jl


# --

EXP_DIR="results/search/mnist/0/"
# rm -rf $EXP_DIR
mkdir -p $EXP_DIR
CUDA_VISIBLE_DEVICES=1 python main.py \
    --outpath $EXP_DIR \
    --dataset fashion_mnist \
    --unrolled |tee $EXP_DIR/search.jl

for i in $(seq 0 49); do
    python sample-arch.py \
        --normal-path $EXP_DIR/normal_arch_e$i.pt \
        --reduce-path $EXP_DIR/reduce_arch_e$i.pt \
        --as-matrix \
        --outpath $EXP_DIR/genotype-$i.npy
done

for i in $(seq 0 49 | tac); do
    mkdir -p $EXP_DIR/$i
    CUDA_VISIBLE_DEVICES=0 python main.py \
        --genotype $EXP_DIR/genotype-$i.npy \
        --outpath $EXP_DIR/$i \
        --epochs 10 \
        --lr-max 0.1 \
        --dataset fashion_mnist | tee $EXP_DIR/$i/train.jl
done

for i in $(seq 0 49 | shuf); do
    mkdir -p $EXP_DIR/$i.e20
    CUDA_VISIBLE_DEVICES=0 python tmp-main.py \
        --genotype $EXP_DIR/genotype-$i.npy \
        --outpath $EXP_DIR/$i.e20 \
        --epochs 30 \
        --lr-max 0.1 \
        --dataset fashion_mnist | tee $EXP_DIR/$i.e20/train.jl
done

for i in $(seq 0 49 | shuf); do
    mkdir -p $EXP_DIR/$i.e20_l16
    CUDA_VISIBLE_DEVICES=0 python tmp-main.py \
        --genotype $EXP_DIR/genotype-$i.npy \
        --outpath $EXP_DIR/$i.e20_l16 \
        --num-layers 16 \
        --epochs 30 \
        --lr-max 0.1 \
        --dataset fashion_mnist | tee $EXP_DIR/$i.e20_l16/train.jl
done

intervals=(0 9 19 29 39 49)
for i in ${intervals[@]}; do
    mkdir -p $EXP_DIR/$i.e64_l20
    CUDA_VISIBLE_DEVICES=0 python tmp-main.py \
        --genotype $EXP_DIR/genotype-$i.npy \
        --outpath $EXP_DIR/$i.e64_l20 \
        --num-layers 20 \
        --epochs 64 \
        --lr-max 0.1 \
        --dataset fashion_mnist | tee $EXP_DIR/$i.e64_l20/train.jl
done
