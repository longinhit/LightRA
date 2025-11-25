#!/bin/bash

DATA=./data
TRAINER=ZeroshotCLIP
# TRAINER=ZeroshotCLIP2
seed=1
shot=1

# DATASET=$1
datasets=(caltech101 oxford_pets stanford_cars oxford_flowers food101 fgvc_aircraft sun397 dtd eurosat ucf101 imagenet)

for DATASET in "${datasets[@]}"
do
    python train.py \
        --root ${DATA} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/zsclip/config.yaml \
        --output-dir output/zero_shot/${DATASET}/${TRAINER}/ \
        --eval-only
done