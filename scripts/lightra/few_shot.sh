#!/bin/bash

DATA="./data"
TRAINER=LightRA
seeds=(1 2 3)
gpus=(0 1 2)
shots=(1 2 4 8 16)

DATASET=$1

if [ "$DATASET" = "eurosat" ]; then
    opt="TRAINER.LightRA.LOSS_WEIGHT 1"
elif [[ "$DATASET" = "imagenet" || "$DATASET" = "caltech101" || "$DATASET" = "oxford_pets" || "$DATASET" = "food101" || "$DATASET" = "dtd" || "$DATASET" = "sun397" ]]; then
    opt="TRAINER.LightRA.LOSS_WEIGHT 15"
else
    opt=""
fi

declare -A gpu_pid_map

get_free_gpus() {
    free_gpus=()
    for i in "${!gpus[@]}"; do
        gpu=${gpus[$i]}
        if [[ -z "${gpu_pid_map[$gpu]}" ]]; then
            free_gpus+=($gpu)
        fi
    done
    echo "${free_gpus[@]}"
}

for shot in "${shots[@]}"
do
    for SEED in "${seeds[@]}"
    do
        free_gpus=($(get_free_gpus))
        gpu=${free_gpus[0]}

        DIR=output/few_shot/${DATASET}/shots_${shot}/${TRAINER}/seed${SEED}
        echo "Run this job and save the output to ${DIR}"
        CUDA_VISIBLE_DEVICES=$gpu python train.py \
            --root ${DATA} \
            --seed ${SEED} \
            --trainer ${TRAINER} \
            --dataset-config-file configs/datasets/${DATASET}.yaml \
            --config-file configs/trainers/${TRAINER}/few_shot.yaml \
            --output-dir ${DIR} \
            DATASET.NUM_SHOTS ${shot} $opt \
            DATASET.SUBSAMPLE_CLASSES all &

        gpu_pid_map[$gpu]=$!
        while [ ${#gpu_pid_map[@]} -eq ${#gpus[@]} ]; do # Each GPU has a task assigned. Wait for the task to complete before releasing the GPU.
            for gpu in "${!gpu_pid_map[@]}"; do
                pid=${gpu_pid_map[$gpu]}
                if ! kill -0 $pid 2>/dev/null; then
                        unset gpu_pid_map[$gpu]
                fi
            done
            sleep 5
        done
    done
done

# Wait for all processes to complete.
for gpu in "${!gpu_pid_map[@]}"; do
    pid=${gpu_pid_map[$gpu]}
    wait $pid
done
unset gpu_pid_map
declare -A gpu_pid_map