#! /bin/bash

mapping="mapping_cm"
# nr_mappings=31

for i in {32..32}
do
  echo "Executing for ${mapping}_${i}"
  python3 run.py --model=VGG7 --dataset=CIFAR10 --load-model-path=models/cifar10/model_cifar10_vgg7.pt --an-sim=1 --array-size=32 --mapping-distr=mapping_example/distr/mapping_cm_${i}.npy --test-error-distr=3 --test-batch-size=64
done
