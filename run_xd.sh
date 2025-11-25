#! /bin/bash
seed=(1 2 3)

datasets=(caltech101 oxford_pets stanford_cars oxford_flowers food101 fgvc_aircraft sun397 dtd eurosat ucf101 imagenetv2 imagenet_sketch imagenet_a imagenet_r)

scripts/lightra/xd_train.sh "${seed[@]}"

echo "train x2d completely, start eval..."

# testing
for DATASET in "${datasets[@]}"
do
    bash scripts/lightra/xd_test.sh ${DATASET} "${seed[@]}"
done

python parse_multi_res.py --test-log \
                         --keyword accuracy \
                         --directory ./output/x2d \
                         --trainer LightRA \
                         --experiments source \
                         --shots 16 \
                         --datasets imagenet

python parse_multi_res.py --test-log \
                         --keyword accuracy \
                         --directory ./output/x2d \
                         --trainer LightRA \
                         --experiments target \
                         --shots 16 \
                         --datasets "${datasets[@]}"