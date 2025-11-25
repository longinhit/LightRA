datasets=(imagenet caltech101 oxford_pets stanford_cars oxford_flowers food101 fgvc_aircraft sun397 dtd eurosat ucf101)

bash scripts/lightra/run_multi_gpu.sh "${datasets[@]}"

python parse_multi_res.py --test-log \
                         --keyword accuracy \
                         --directory ./output/base2novel \
                         --trainer LightRA \
                         --experiments train_base test_novel \
                         --shots 16 \
                         --datasets "${datasets[@]}"