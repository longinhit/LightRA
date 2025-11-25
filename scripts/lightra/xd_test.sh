#!/bin/bash

# custom config
DATA="./data"
TRAINER=LightRA

CFG=x2d
SHOTS=16

seeds=("$@") # Pass the seed from an external source.
DATASET="${seeds[0]}" 
unset seeds[0] 
seeds=("${seeds[@]}")  

gpus=(0 1 2)

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

for SEED in "${seeds[@]}"
do
    free_gpus=($(get_free_gpus))
    gpu=${free_gpus[0]}
    DIR=output/${CFG}/target/${DATASET}/shots_${SHOTS}/${TRAINER}/seed${SEED}
    echo "Run this job and save the output to ${DIR}"
    CUDA_VISIBLE_DEVICES=$gpu python train.py \
                                            --root ${DATA} \
                                            --seed ${SEED} \
                                            --trainer ${TRAINER} \
                                            --dataset-config-file configs/datasets/${DATASET}.yaml \
                                            --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
                                            --output-dir ${DIR} \
                                            --model-dir output/${CFG}/source/imagenet/shots_${SHOTS}/${TRAINER}/seed${SEED}  \
                                            --eval-only &
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

# Wait for all processes to complete.
for gpu in "${!gpu_pid_map[@]}"; do
    pid=${gpu_pid_map[$gpu]}
    wait $pid
done
unset gpu_pid_map
declare -A gpu_pid_map


