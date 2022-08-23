from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import numpy as np
import time
import json
import sys
import os
from datetime import datetime
sys.path.append("code/python/")

from Utils import Scale, Clippy, set_layer_mode, parse_args, dump_exp_data, create_exp_folder, store_exp_data, Criterion, binary_hingeloss

from QuantizedNN import QuantizedLinear, QuantizedConv2d, QuantizedActivation

from Models import FC, VGG3, VGG7

from Traintest_Utils import train, test, test_error

import binarizePM1
import binarizePM1FI
import quantization

class SymmetricBitErrorsBinarizedPM1:
    def __init__(self, method, p):
        self.method = method
        self.p = p
    def updateErrorModel(self, p_updated):
        self.p = p_updated
    def resetErrorModel(self):
        self.p = 0
    def applyErrorModel(self, input):
        return self.method(input, self.p, self.p)

class Quantization1:
    def __init__(self, method):
        self.method = method
    def applyQuantization(self, input):
        return self.method(input)

binarizepm1 = Quantization1(binarizePM1.binarize)
binarizepm1fi = SymmetricBitErrorsBinarizedPM1(binarizePM1FI.binarizeFI, 0.1)

cel_train = Criterion(method=nn.CrossEntropyLoss(reduction="none"), name="CEL_train")
cel_test = Criterion(method=nn.CrossEntropyLoss(reduction="none"), name="CEL_test")

q_train = True # quantization during training
q_eval = True # quantization during evaluation
# an_sim = True

# python3 run.py --model=FC --dataset=FMNIST --batch-size=256 --epochs=5 --lr=0.001 --step-size=2 --gamma=0.5 --load-model="model_fc_test.pt"

# capacitor model
# t = - tau * torch.log(1-(a/v_o))

# class Snn_RC:
#     def __init__(self, method):
#         self.r_l = method
#         self.v_th = v_th
#         self.
#         self.c_mem = c_mem
#     def applyQuantization(self, input):
#         return self.method(input)

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Training Process')
    parse_args(parser)
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    available_gpus = [i for i in range(torch.cuda.device_count())]
    print("Available GPUs: ", available_gpus)
    gpu_select = args.gpu_num
    # change GPU that is being used
    torch.cuda.set_device(gpu_select)
    # which GPU is currently used
    print("Currently used GPU: ", torch.cuda.current_device())

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    nn_model = None
    model = None
    dataset1 = None
    dataset2 = None

    if args.model == "FC":
        nn_model = FC
        model = nn_model().cuda()
    if args.model == "VGG3":
        nn_model = VGG3
        model = nn_model().cuda()
    if args.model == "VGG7":
        nn_model = VGG7
        model = nn_model().cuda()

    if args.dataset == "MNIST":
        transform=transforms.Compose([
            transforms.ToTensor(),
            ])
        dataset1 = datasets.MNIST('data', train=True, download=True, transform=transform)
        dataset2 = datasets.MNIST('data', train=False, transform=transform)

    if args.dataset == "FMNIST":
        transform=transforms.Compose([
            transforms.ToTensor(),
            ])
        dataset1 = datasets.FashionMNIST('data', train=True, download=True, transform=transform)
        dataset2 = datasets.FashionMNIST('data', train=False, transform=transform)

    if args.dataset == "KMNIST":
        transform=transforms.Compose([
            transforms.ToTensor(),
            ])
        dataset1 = datasets.KMNIST(root="data/KMNIST/", train=True, download=True, transform=transform)
        dataset2 = datasets.KMNIST('data/KMNIST/', train=False, download=True, transform=transform)

    if args.dataset == "SVHN":
        transform=transforms.Compose([
            transforms.ToTensor(),
            ])
        dataset1 = datasets.SVHN(root="data/SVHN/", split="train", download=True, transform=transform)
        dataset2 = datasets.SVHN(root="data/SVHN/", split="test", download=True, transform=transform)

    if args.dataset == "CIFAR10":
        transform_train=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        transform_test=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        dataset1 = datasets.CIFAR10('data', train=True, download=True, transform=transform_train)
        dataset2 = datasets.CIFAR10('data', train=False, transform=transform_test)

    if args.dataset == "CIFAR100":
        transform_train=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        transform_test=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        dataset1 = datasets.CIFAR100('data', train=True, download=True, transform=transform_train)
        dataset2 = datasets.CIFAR100('data', train=False, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    mac_mapping = None
    if args.mapping is not None:
        mac_mapping = torch.from_numpy(np.load(args.mapping)).float().cuda()
        print("mapping", mac_mapping)

    model = nn_model(quantMethod=binarizepm1, an_sim=args.an_sim, array_size=args.array_size, mapping=mac_mapping, quantize_train=q_train, quantize_eval=q_eval, error_model=None).to(device)

    # print(model.name)
    # create experiment folder and file
    to_dump_path = create_exp_folder(model)
    if not os.path.exists(to_dump_path):
        open(to_dump_path, 'w').close()

    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    optimizer = Clippy(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    if args.train_model is not None:
        time_elapsed = 0
        times = []
        for epoch in range(1, args.epochs + 1):
            torch.cuda.synchronize()
            since = int(round(time.time()*1000))
            #
            train(args, model, device, train_loader, optimizer, epoch)
            #
            time_elapsed += int(round(time.time()*1000)) - since
            print('Epoch training time elapsed: {}ms'.format(int(round(time.time()*1000)) - since))
            # test(model, device, train_loader)
            since = int(round(time.time()*1000))
            #
            test(model, device, test_loader)
            #
            time_elapsed += int(round(time.time()*1000)) - since
            print('Test time elapsed: {}ms'.format(int(round(time.time()*1000)) - since))
            # test(model, device, train_loader)
            scheduler.step()

    if args.save_model is not None:
        torch.save(model.state_dict(), "model_{}.pt".format(args.save_model))

    # load model
    if args.load_model_path is not None:
            to_load = args.load_model_path
            print("Loaded model: ", to_load)
            model.load_state_dict(torch.load(to_load, map_location='cuda:0'))

    if args.test_error is not None:
        all_accuracies = test_error(model, device, test_loader)
        to_dump_data = dump_exp_data(model, args, all_accuracies)
        store_exp_data(to_dump_path, to_dump_data)

    test(model, device, test_loader)


if __name__ == '__main__':
    main()
