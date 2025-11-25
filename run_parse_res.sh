datasets=(imagenet caltech101 oxford_pets stanford_cars oxford_flowers food101 fgvc_aircraft sun397 dtd eurosat ucf101)

datasets=(fgvc_aircraft)

COS_WEIGHT=2.5
LOSS_WEIGHT=15

python parse_multi_res.py --test-log \
                    --keyword accuracy \
                    --directory ./output/base2novel \
                    --trainer LightRA \
                    --experiments train_base_${COS_WEIGHT}_${LOSS_WEIGHT} test_novel_${COS_WEIGHT}_${LOSS_WEIGHT} \
                    --shots 16 \
                    --datasets "${datasets[@]}"



