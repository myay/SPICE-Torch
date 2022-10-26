#! /bin/bash
reps=10
#
# # baseline
# echo " --- Baseline VGG3 ---"
# python3 run.py --model=VGG3 --dataset=FMNIST --load-model-path=models/fmnist/model_fmnist_vgg3_13-09-22.pt --print-accuracy=1 --profile-time=${reps}
#
# # direct mapping
# echo " --- Direct mapping VGG3 ---"
# python3 run.py --model=VGG3 --dataset=FMNIST --load-model-path=models/fmnist/model_fmnist_vgg3_13-09-22.pt --print-accuracy=1 --an-sim=1 --array-size=32 --mapping=mapping_example/direct/mapping_30.npy --profile-time=${reps} --performance-mode=1
#
# # distribution based mapping
# echo " --- Distribution based mapping VGG3 ---"
# python3 run.py --model=VGG3 --dataset=FMNIST --load-model-path=models/fmnist/model_fmnist_vgg3_13-09-22.pt --print-accuracy=1 --an-sim=1 --array-size=32 --mapping-distr=mapping_example/distr/mapping_cm_30.npy --profile-time=${reps} --performance-mode=1

# baseline
echo " --- Baseline VGG7 ---"
python3 run.py --model=VGG7 --dataset=CIFAR10 --load-model-path=models/cifar10/model_cifar10_vgg7.pt --print-accuracy=1 --profile-time=${reps}

# direct mapping
echo " --- Direct mapping VGG7 ---"
python3 run.py --model=VGG7 --dataset=CIFAR10 --load-model-path=models/cifar10/model_cifar10_vgg7.pt --print-accuracy=1 --an-sim=1 --array-size=32 --mapping=mapping_example/direct/mapping_30.npy --profile-time=${reps} --performance-mode=1

# distribution based mapping
echo " --- Distribution based mapping VGG7 ---"
python3 run.py --model=VGG7 --dataset=CIFAR10 --load-model-path=models/cifar10/model_cifar10_vgg7.pt --print-accuracy=1 --an-sim=1 --array-size=32 --mapping-distr=mapping_example/distr/mapping_cm_30.npy --profile-time=${reps} --performance-mode=1
