import torch
import torch.nn as nn
from torchvision import datasets, transforms
import numpy as np
import argparse
import os
from datetime import datetime
import json

from QuantizedNN import QuantizedLinear, QuantizedConv2d, QuantizedActivation

from Models import FC, VGG3, VGG7, ResNet, BasicBlock

def get_model_and_datasets(args):
    nn_model = None
    # model = None
    dataset1 = None
    dataset2 = None

    if args.model == "FC":
        nn_model = FC
        # model = nn_model().cuda()
    if args.model == "VGG3":
        nn_model = VGG3
        # model = nn_model().cuda()
    if args.model == "VGG7":
        nn_model = VGG7
        # model = nn_model().cuda()
    if args.model == "ResNet":
        nn_model = ResNet# nn_model(BasicBlock, [2, 2, 2, 2]).to(device)

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

    if args.dataset == "IMAGENETTE":
        transform_train = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        transform_test = transforms.Compose([
            transforms.Resize((64, 64)),
            # transforms.RandomCrop(64, padding=4),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        dataset1 = datasets.ImageFolder('data/imagenette2/train', transform=transform_train)
        dataset2 = datasets.ImageFolder('data/imagenette2/val', transform=transform_test)

def set_layer_mode(model, mode):
    for layer in model.children():
        if isinstance(layer, (QuantizedActivation, QuantizedLinear, QuantizedConv2d)):
            if mode == "train":
                layer.training = True
            if mode == "eval":
                layer.eval = False

def parse_args(parser):
    parser.add_argument('--model', type=str, default=None,
                    help='VGG3/VGG7')
    parser.add_argument('--dataset', type=str, default=None,
                    help='MNIST/FMNIST/QMNIST/SVHN/CIFAR10')
    parser.add_argument('--performance-mode', type=int, default=None, help='Specify whether to activate the faster and more memory-efficient performance mode (sub-MAC results can only be changed in cuda-kernel!)')
    parser.add_argument('--train-model', type=int, default=None, help='Whether to train a model')
    parser.add_argument('--load-model-path', type=str, default=None, help='Specify path to model if it should be loaded')
    parser.add_argument('--gpu-num', type=int, default=0, metavar='N', help='Specify the GPU on which the training should be performed')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                        help='input batch size for testing (default: 256)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.5, metavar='M',
                        help='Learning rate step gamma (default: 0.5)')
    parser.add_argument('--step-size', type=int, default=25, metavar='M',
                        help='Learning step size (default: 5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', type=str, default=None,
                        help='Specify name for saving model')
    parser.add_argument('--load-training-state', type=str, default=None,
                        help='Specify path for loading the training state')
    parser.add_argument('--save-training-state', type=str, default=None,
                        help='Specify path for saving the training state')
    parser.add_argument('--an-sim', type=int, default=None, help='Whether to turn on the mapping based on SPICE')
    parser.add_argument('--mapping', type=str, default=None,
                        help='Specify the direct mapping to import')
    parser.add_argument('--mapping-distr', type=str, default=None,
                        help='Specify the distribution-based mapping to import')
    parser.add_argument('--array-size', type=int, default=32, help='Specify the array size')
    parser.add_argument('--test-error', type=int, default=None, help='Whether to test the model')
    parser.add_argument('--test-error-distr', type=int, default=None, help='Specify the number of repetitions to perform in accuracy evaluations')
    parser.add_argument('--print-accuracy', type=int, default=None, help='Specify whether to print inference accuracy')
    parser.add_argument('--profile-time', type=int, default=None, help='Specify whether to profile the execution time by specifying the repetitions')
    parser.add_argument('--extract-absfreq', type=int, default=None, help='Specify whether to extract the absolute frequencies of MAC values')
    parser.add_argument('--extract-absfreq-resnet', type=int, default=None, help='Specify whether to extract the absolute frequencies of MAC values for ResNet')



def dump_exp_data(model, args, all_accuracies):
    to_dump = dict()
    to_dump["model"] = model.name
    # to_dump["method"] = model.method
    to_dump["batchsize"] = args.batch_size
    to_dump["epochs"] = args.epochs
    to_dump["learning_rate"] = args.lr
    to_dump["gamma"] = args.gamma
    to_dump["stepsize"] = args.step_size
    # to_dump["traincrit"] = model.traincriterion.name
    # to_dump["testcrit"] = model.testcriterion.name
    to_dump["test_error"] = all_accuracies
    return to_dump

def create_exp_folder(model):
    exp_path = ""
    access_rights = 0o755
    this_path = os.getcwd()
    exp_path += this_path+"/experiments/"+model.name+"/"+"results-"+datetime.now().strftime('%d-%m-%Y-%H:%M:%S')
    try:
        os.makedirs(exp_path, access_rights, exist_ok=False)
    except OSError:
        print ("Creation of the directory %s failed" % exp_path)
    else:
        print ("Successfully created the directory %s" % exp_path)
    return exp_path + "/results.jsonl"

def store_exp_data(to_dump_path, to_dump_data):
    with open(to_dump_path, 'a') as outfile:
        json.dump(to_dump_data, outfile)
        print ("Successfully stored results in %s" % to_dump_path)

def print_tikz_data(in_array):
    accs_mean = np.mean(np.array(in_array), axis=0)
    accs_min = np.min(np.array(in_array), axis=0)
    accs_max = np.max(np.array(in_array), axis=0)

    # x_counter = 0
    # print(accs_mean)
    # for idx in range(len(accs_mean)):
    #     # print("&", end='')
    #     print("{} {} {} {}".format(str(x_counter+1), accs_mean[idx], accs_max[idx] - accs_mean[idx], accs_mean[idx] - accs_min[idx]))
    #     x_counter += 1
    print("{} {} {}".format(accs_mean, accs_max - accs_mean, accs_mean - accs_min))

# wrapper for profiling functions
def cuda_profiler(profile_function, *args, **kwargs):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output = profile_function(*args, **kwargs)
    end.record()
    torch.cuda.synchronize()
    print("Run time (ms):", start.elapsed_time(end))
    return start.elapsed_time(end)
