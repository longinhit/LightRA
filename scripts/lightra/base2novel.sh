#!/bin/bash

# custom config
DATA="./data"
TRAINER=LightRA
DATASET=$1
SEED=$2
SHOTS=16

DIR=output/base2novel/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/seed${SEED}
COMMON_DIR=${DATASET}/shots_${SHOTS}/${TRAINER}/seed${SEED}
MODEL_DIR=output/base2novel/train_base/${COMMON_DIR}
if [ -d "$DIR" ]; then
        echo "Evaluating model! Results are available in ${DIR}. Evaluating on ${DATASET} with ${SHOTS} shot and ${SEED} seed ..."
        python train.py \
                --root ${DATA} \
                --seed ${SEED} \
                --trainer ${TRAINER} \
                --dataset-config-file configs/datasets/${DATASET}.yaml \
                --config-file configs/trainers/${TRAINER}/base2novel.yaml \
                --model-dir ${MODEL_DIR} \
                --eval-only \
                --output-dir ${DIR} \
                DATASET.NUM_SHOTS ${SHOTS} \
                DATASET.SUBSAMPLE_CLASSES base
else
        echo "Start fintune on ${DATASET} with ${SHOTS} shot and ${SEED} seed ..."
        python train.py \
                --root ${DATA} \
                --seed ${SEED} \
                --trainer ${TRAINER} \
                --dataset-config-file configs/datasets/${DATASET}.yaml \
                --config-file configs/trainers/${TRAINER}/base2novel.yaml \
                --output-dir ${DIR} \
                DATASET.NUM_SHOTS ${SHOTS} \
                DATASET.SUBSAMPLE_CLASSES base
fi

TEST_OUTPUT_DIR=output/base2novel/test_novel/${COMMON_DIR}
echo "Evaluating model! Results are available in ${DIR}. Resuming..."
python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/base2novel.yaml \
        --output-dir ${TEST_OUTPUT_DIR} \
        --model-dir ${MODEL_DIR} \
        --eval-only \
        DATASET.NUM_SHOTS ${SHOTS} \
        DATASET.SUBSAMPLE_CLASSES new
