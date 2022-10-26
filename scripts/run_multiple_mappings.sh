#! /bin/bash

mapping="mapping"
# nr_mappings=31

for i in {31..1}
do
  echo "Executing for ${mapping}_absf_${i}"
  python3 run.py --model=VGG7 --dataset=SVHN --load-model-path=models/svhn/model_svhn_vgg7_13-09-22.pt --an-sim=1 --array-size=32 --mapping=mapping_example/direct_absf/mapping_absf_${i}.npy --performance-mode=1 --print-accuracy=1 --test-batch-size=64
done
