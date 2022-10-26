#! /bin/bash

mapping="mapping_cm"
# nr_mappings=31

for i in {32..2..2}
do
  echo "Executing for ${mapping}_${i}"
  python3 run.py --model=VGG3 --dataset=FMNIST --load-model-path=models/fmnist/model_fmnist_vgg3_13-09-22.pt --an-sim=1 --array-size=32 --mapping-distr=mapping_example/distr/mapping_cm_${i}.npy --test-error-distr=5
done
