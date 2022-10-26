#! /bin/bash

python3 run.py --model=VGG3 --dataset=FMNIST --an-sim=1 --mapping-distr=mapping_example/distr/mapping_cm_32.npy --array-size=32 --load-model-path=models/fmnist/model_fmnist_vgg3_13-09-22.pt --test-error-distr=3
