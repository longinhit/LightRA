#! /bin/bash

seeds=(1 2 3)
gpus=(0 1 2)
shots=(16)

datasets=("$@") # Pass the datasets from an external source.

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

for dataset in "${datasets[@]}"
do
    for seed in "${seeds[@]}"
    do
        for ((i = 0; i < ${#shots[@]}; i++))
        do
            shot=${shots[$i]}
            free_gpus=($(get_free_gpus))
            gpu=${free_gpus[0]}
            CUDA_VISIBLE_DEVICES=$gpu scripts/lightra/base2novel.sh $dataset $seed &
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
done

# Wait for all processes to complete.
for gpu in "${!gpu_pid_map[@]}"; do
    pid=${gpu_pid_map[$gpu]}
    wait $pid
done
unset gpu_pid_map
declare -A gpu_pid_map 