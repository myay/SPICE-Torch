from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import time
import json
import sys
import os
from datetime import datetime
sys.path.append("code/python/")

from Utils import set_layer_mode, parse_args, dump_exp_data, create_exp_folder, store_exp_data

from QuantizedNN import QuantizedLinear, QuantizedConv2d, QuantizedActivation

import binarizePM1
import binarizePM1FI
import quantization

def binary_hingeloss(yhat, y, b=128):
    #print("yhat", yhat.mean(dim=1))
    #print("y", y)
    # print("BINHINGE")
    y_enc = 2 * torch.nn.functional.one_hot(y, yhat.shape[-1]) - 1.0
    #print("y_enc", y_enc)
    l = (b - y_enc * yhat).clamp(min=0)
    #print(l)
    return l.mean(dim=1) / b

class Clippy(torch.optim.Adam):
    def step(self, closure=None):
        loss = super(Clippy, self).step(closure=closure)
        for group in self.param_groups:
            for p in group['params']:
                p.data.clamp(-1,1)
        return loss

class Criterion:
    def __init__(self, method, name, param=None):
        self.method = method
        self.param = param
        self.name = name
    def applyCriterion(self, output, target):
        if self.param is not None:
            return self.method(output, target, self.param)
        else:
            return self.method(output, target)

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    set_layer_mode(model, "train") # propagate informaton about training to all layers

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # loss = F.nll_loss(output, target)
        criterion = nn.CrossEntropyLoss(reduction="none")
        loss = criterion(output, target).mean()
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader, pr=1):
    model.eval()
    set_layer_mode(model, "eval") # propagate informaton about eval to all layers

    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
    #     test_loss, correct, len(test_loader.dataset),
    #     100. * correct / len(test_loader.dataset)))
    if pr is not None:
        print('\nAccuracy: {:.2f}%\n'.format(
            100. * correct / len(test_loader.dataset)))

    accuracy = 100. * (correct / len(test_loader.dataset))

    return accuracy


def test_error(model, device, test_loader):
    model.eval()
    set_layer_mode(model, "eval") # propagate informaton about eval to all layers
    perrors = [i/100 for i in range(10)]

    all_accuracies = []
    for perror in perrors:
        # update perror in every layer
        for layer in model.children():
            if isinstance(layer, (QuantizedActivation, QuantizedLinear, QuantizedConv2d)):
                if layer.error_model is not None:
                    layer.error_model.updateErrorModel(perror)

        print("Error rate: ", perror)
        accuracy = test(model, device, test_loader)
        all_accuracies.append(
            {
                "perror":perror,
                "accuracy": accuracy
            }
        )

    # reset error models
    for layer in model.children():
        if isinstance(layer, (QuantizedActivation, QuantizedLinear, QuantizedConv2d)):
            if layer.error_model is not None:
                layer.error_model.resetErrorModel()
    return all_accuracies
