#! /bin/bash

echo "Generating models"

python3 run.py --model=VGG3 --dataset=FMNIST --batch-size=256 --epochs=100 --lr=0.001 --step-size=10 --gamma=0.5 --train-model=1 --save-model="fmnist_vgg3_13-09-22" --save-training-state="fmnist_vgg3_13-09-22"

python3 run.py --model=VGG3 --dataset=KMNIST --batch-size=256 --epochs=100 --lr=0.001 --step-size=10 --gamma=0.5 --train-model=1 --save-model="kmnist_vgg3_13-09-22" --save-training-state="kmnist_vgg3_13-09-22"
