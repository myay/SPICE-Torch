#! /bin/bash

echo "Generating models"

# python3 run.py --model=VGG3 --dataset=FMNIST --batch-size=256 --epochs=100 --lr=0.001 --step-size=10 --gamma=0.5 --train-model=1 --save-model="fmnist_vgg3" --save-training-state="fmnist_vgg3"

# python3 run.py --model=VGG3 --dataset=KMNIST --batch-size=256 --epochs=100 --lr=0.001 --step-size=10 --gamma=0.5 --train-model=1 --save-model="kmnist_vgg3" --save-training-state="kmnist_vgg3"

# python3 run.py --model=VGG7 --dataset=SVHN --batch-size=256 --epochs=100 --lr=0.001 --step-size=10 --gamma=0.5 --train-model=1 --save-model="svhn_vgg7" --save-training-state="svhn_vgg7"

python3 run.py --model=VGG7 --dataset=CIFAR10 --batch-size=256 --epochs=200 --lr=0.001 --step-size=50 --gamma=0.5 --train-model=1 --save-model="cifar10_vgg7" --save-training-state="cifar10_vgg7"
