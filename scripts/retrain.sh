#! /bin/bash

for i in {14..2..2}
do
  python3 run.py --model=VGG3 --dataset=FMNIST --an-sim=1 --array-size=32 --mapping=mapping_example/direct/mapping_${i}.npy --performance-mode=1 --train-model=1 --epochs=5 --lr=0.0001 --gamma=0.5 --step-size=10 --batch-size=256 --load-training-state=models/fmnist/model_checkpoint_fmnist_vgg3_13-09-22.pt
done

for i in {14..2..2}
do
  python3 run.py --model=VGG3 --dataset=KMNIST --an-sim=1 --array-size=32 --mapping=mapping_example/direct/mapping_${i}.npy --performance-mode=1 --train-model=1 --epochs=5 --lr=0.0001 --gamma=0.5 --step-size=10 --batch-size=256 --load-training-state=models/kmnist/model_checkpoint_kmnist_vgg3_13-09-22.pt
done
# python3 run.py --model=VGG3 --dataset=FMNIST --print-accuracy=1 --an-sim=1 --array-size=32 --mapping=mapping_example/direct/mapping_14.npy --performance-mode=1 --load-model-path=models/fmnist/model_fmnist_vgg3_13-09-22.pt

# python3 run.py --model=VGG3 --dataset=FMNIST --print-accuracy=1 --train-model=1
