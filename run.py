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

from Utils import set_layer_mode, parse_args, dump_exp_data, create_exp_folder, store_exp_data, get_model_and_datasets, print_tikz_data, cuda_profiler

from QuantizedNN import QuantizedLinear, QuantizedConv2d, QuantizedActivation

from Models import FC, VGG3, VGG7

from Traintest_Utils import train, test, test_error, Clippy, Criterion, binary_hingeloss

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

# crit_train = Criterion(method=nn.CrossEntropyLoss(reduction="none"), name="CEL_train")
# crit_test = Criterion(method=nn.CrossEntropyLoss(reduction="none"), name="CEL_test")
crit_train = Criterion(binary_hingeloss, "MHL_train", param=128)
crit_test = Criterion(binary_hingeloss, "MHL_test", param=128)

q_train = True # quantization during training
q_eval = True # quantization during evaluation

#python3 run.py --model=FC --dataset=FMNIST --load-model="model_fc_test.pt" --mapping=mapping_example/mapping.npy --array-size=32 --an-sim=1

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

    nn_model, dataset1, dataset2 = get_model_and_datasets(args)

    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    mac_mapping = None
    mac_mapping_distr = None
    sorted_mac_mapping_idx = None
    if args.mapping is not None:
        print("Mapping: ", args.mapping)
        mac_mapping = torch.from_numpy(np.load(args.mapping)).float().cuda()
        # print("mapping", mac_mapping)
    if args.mapping_distr is not None:
        # print("hello")
        # print("Mapping distr.: ", args.mapping_distr)
        sorted_mac_mapping_idx = torch.from_numpy(np.argsort(np.load(args.mapping_distr))).float().cuda()
        mac_mapping_distr = torch.from_numpy(np.load(args.mapping_distr)).float().cuda()
        # calculate cumulative distribution
        # flag = 1
        # a = []
        # print("MAC mapping distr", mac_mapping_distr[2])
        for i in range(mac_mapping_distr.shape[0]):
            # flag = 1
            for j in range(mac_mapping_distr.shape[1]):
                # the first entry that is not zero needs to be left alone
                # print("mac1", mac_mapping_distr[i][int(sorted_mac_mapping_idx[i][j])])
                # print("mac2", mac_mapping_distr[i][int(sorted_mac_mapping_idx[i][j+1])])
                # if ((mac_mapping_distr[i][int(sorted_mac_mapping_idx[i][j])] > 0) and (flag is None)):
                #     flag = None
                #     continue
                if (mac_mapping_distr[i][int(sorted_mac_mapping_idx[i][j])] > 0):
                    mac_mapping_distr[i][int(sorted_mac_mapping_idx[i][j])] = mac_mapping_distr[i][int(sorted_mac_mapping_idx[i][j])] + mac_mapping_distr[i][int(sorted_mac_mapping_idx[i][j-1])]
                    # print("map", mac_mapping_distr[i][int(sorted_mac_mapping_idx[i][j])])
        # print("MAC mapping distr", mac_mapping_distr[2])
        # print sorted array
        # for i in range(mac_mapping_distr.shape[1]):
        #     print(mac_mapping_distr[2][int(sorted_mac_mapping_idx[2][i])])
        # print(mac_mapping_distr[2])
        # print(sorted_mac_mapping_idx[2])
        # use later: mapping[sorted[i]]
        # print("Mapping from distr: ", mac_mapping_distr)
        # print("Mapping from distr idx: ", sorted_mac_mapping_idx)


    model = nn_model(crit_train, crit_test, quantMethod=binarizepm1, an_sim=args.an_sim, array_size=args.array_size, mapping=mac_mapping, mapping_distr=mac_mapping_distr, sorted_mapping_idx=sorted_mac_mapping_idx, performance_mode=args.performance_mode, quantize_train=q_train, quantize_eval=q_eval, error_model=None).to(device)

    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    optimizer = Clippy(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    # load training state or create new model
    if args.load_training_state is not None:
        print("Loaded training state: ", args.load_training_state)
        checkpoint = torch.load(args.load_training_state)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # epoch = checkpoint['epoch']

    # print(model.name)
    # create experiment folder and file
    to_dump_path = create_exp_folder(model)
    if not os.path.exists(to_dump_path):
        open(to_dump_path, 'w').close()

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

    if args.save_training_state is not None:
        path = "model_checkpoint_{}.pt".format(args.save_training_state)

        torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        }, path)

    # load model
    if args.load_model_path is not None:
            to_load = args.load_model_path
            print("Loaded model: ", to_load)
            model.load_state_dict(torch.load(to_load, map_location='cuda:0'))

    if args.test_error is not None:
        all_accuracies = test_error(model, device, test_loader)
        to_dump_data = dump_exp_data(model, args, all_accuracies)
        store_exp_data(to_dump_path, to_dump_data)

    if args.test_error_distr is not None:
        # perform repeated experiments and return in tikz format
        acc_list = []
        for i in range(args.test_error_distr):
            acc_list.append(test(model, device, test_loader))
        # print("acclist", acc_list)
        print_tikz_data(acc_list)

    if args.print_accuracy is not None:
        print("Accuracy: ")
        test(model, device, test_loader)

    if args.profile_time is not None:
        print("Measuring time: ")
        times_list = []
        for rep in range(args.profile_time):
            profiled = cuda_profiler(test, model, device, test_loader, pr=None)
            times_list.append(profiled)
        print_tikz_data(times_list)

if __name__ == '__main__':
    main()
