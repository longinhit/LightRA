#! /bin/bash

shots="1 2 4 8 16"
datasets=(imagenet caltech101 oxford_pets stanford_cars oxford_flowers food101 fgvc_aircraft sun397 dtd eurosat ucf101)

for dataset in "${datasets[@]}"
do
    ./scripts/lightra/few_shot.sh ${dataset}
done

for dataset in "${datasets[@]}"
do
    python parse_multi_res.py --test-log \
                    --keyword accuracy \
                    --directory ./output \
                    --trainer LightRA \
                    --experiments few_shot \
                    --shots ${shots} \
                    --datasets ${dataset}
done