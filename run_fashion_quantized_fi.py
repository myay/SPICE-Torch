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

from Utils import Scale, Clippy, set_layer_mode, parse_args, dump_exp_data, create_exp_folder, store_exp_data, Criterion, binary_hingeloss

from QuantizedNN import QuantizedLinear, QuantizedConv2d, QuantizedActivation

import binarizePM1
import binarizePM1FI
import quantization
import quantizationFI

### settings
# 8 bit
#python3 run_fashion_quantized_8bit.py --batch-size=256 --epochs=200 --lr=0.0001 --step-size=5 --gamma=0.5
#scale 1e-5

# 4 bit
#python3 run_fashion_quantized_fi.py --batch-size=256 --epochs=10 --lr=0.0001 --step-size=5 --gamma=0.5 --test-error
#scale 1e-5

# bit error case
#python3 run_fashion_bin_fi.py --batch-size=256 --epochs=5 --lr=0.001 --step-size=25 --test-error

class SymmetricBitErrorsQNN:
    def __init__(self, method_errors, method_enc_dec, p, bits, type):
        self.method_errors = method_errors
        self.method_enc_dec = method_enc_dec
        self.p = p
        self.bits = bits
        self.type = type
    def updateErrorModel(self, p_updated):
        self.p = p_updated
    def resetErrorModel(self):
        self.p = 0
    def applyErrorModel(self, input):
        output = self.method_enc_dec(input,
         input.min().item(), input.max().item(), self.bits, 1) # to unsigned, encode
        output = self.method_errors(input, self.p, self.p, self.bits) # inject bit flips
        output = self.method_enc_dec(input,
         input.min().item(), input.max().item(), self.bits, 0) # to signed again, decode
        return output

class Quantization2:
    def __init__(self, method, bits=None, unsign=0):
        self.method = method
        self.bits = bits
        self.unsigned = unsign # 0: use signed, 1: use unsigned
    def applyQuantization(self, input):
        return self.method(input,
         input.min().item(), input.max().item(), self.bits, self.unsigned)

q4bit_errors = SymmetricBitErrorsQNN(quantizationFI.bfi_8bit, quantization.quantize, 0, 4, "flip4bit")
# q4bit_errors = None
q4bit = Quantization2(quantization.quantize, 4, 0)
cel_train = Criterion(method=nn.CrossEntropyLoss(reduction="none"), name="CEL_train")
cel_test = Criterion(method=nn.CrossEntropyLoss(reduction="none"), name="CEL_test")

mhl_train = Criterion(binary_hingeloss, "MHL_train", param=1024)
mhl_test = Criterion(binary_hingeloss, "MHL_test", param=1024)

q_train = True # quantization during training
q_eval = True # quantization during evaluation

class QNN_FMNIST(nn.Module):
    def __init__(self):
        super(QNN_FMNIST, self).__init__()
        self.relu = nn.ReLU()
        self.name = "QNN_FMNIST"
        self.method = {"type": q4bit_errors.type, "p": q4bit_errors.p}
        self.traincriterion = cel_train
        self.testcriterion = cel_test
        # self.traincriterion = mhl_train
        # self.testcriterion = mhl_test

        self.conv1 = QuantizedConv2d(1, 64, kernel_size=3, padding=1, stride=1, quantization=q4bit, error_model=q4bit_errors, quantize_train=q_train, quantize_eval=q_eval)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = QuantizedConv2d(64, 64, kernel_size=3, padding=1, stride=1, quantization=q4bit, error_model=q4bit_errors, quantize_train=q_train, quantize_eval=q_eval)
        self.bn2 = nn.BatchNorm2d(64)

        self.fc1 = QuantizedLinear(7*7*64, 2048, quantization=q4bit, error_model=q4bit_errors, quantize_train=q_train, quantize_eval=q_eval)
        self.bn3 = nn.BatchNorm1d(2048)

        self.fc2 = QuantizedLinear(2048, 10, quantization=q4bit, error_model=q4bit_errors, quantize_train=q_train, quantize_eval=q_eval)
        # self.fc2 = nn.Linear(2048, 10)
        self.scale = Scale(init_value=1e-5)

    def forward(self, x):
        #print(self)
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.bn2(x)
        x = self.relu(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.scale(x)
        # output = F.log_softmax(x, dim=1)
        return x

    def train_m(self, args, device, train_loader, optimizer, epoch):
        self.train()
        set_layer_mode(self, "train") # propagate informaton about training to all layers

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = self(data)
            loss = self.traincriterion.applyCriterion(output, target)
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
                if args.dry_run:
                    break


    def test_m(self, device, test_loader):
        self.eval()
        set_layer_mode(self, "eval") # propagate informaton about eval to all layers

        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = self(data)
                test_loss += self.testcriterion.applyCriterion(output, target).mean()
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

        accuracy = 100. * (correct / len(test_loader.dataset))

        return accuracy


    def test_m_error(self, device, test_loader):
        self.eval()
        set_layer_mode(self, "eval") # propagate informaton about eval to all layers
        perrors = [i/100 for i in range(10)]

        all_accuracies = []
        for perror in perrors:
            # update perror in every layer
            for layer in self.children():
                if isinstance(layer, (QuantizedActivation, QuantizedLinear, QuantizedConv2d)):
                    if layer.error_model is not None:
                        layer.error_model.updateErrorModel(perror)

            print("Error rate: ", perror)
            accuracy = self.test_m(device, test_loader)
            all_accuracies.append(
                {
                    "perror":perror,
                    "accuracy": accuracy
                }
            )

        # reset error models
        for layer in self.children():
            if isinstance(layer, (QuantizedActivation, QuantizedLinear, QuantizedConv2d)):
                if layer.error_model is not None:
                    layer.error_model.resetErrorModel()
        return all_accuracies


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Training Process')
    parse_args(parser)
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        ])
    dataset1 = datasets.FashionMNIST('data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.FashionMNIST('data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = QNN_FMNIST().to(device)

    # create experiment folder and file
    to_dump_path = create_exp_folder(model)
    if not os.path.exists(to_dump_path):
        open(to_dump_path, 'w').close()

    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    optimizer = Clippy(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    time_elapsed = 0
    times = []
    for epoch in range(1, args.epochs + 1):
        torch.cuda.synchronize()
        since = int(round(time.time()*1000))
        #
        model.train_m(args, device, train_loader, optimizer, epoch)
        #
        time_elapsed += int(round(time.time()*1000)) - since
        print('Epoch training time elapsed: {}ms'.format(int(round(time.time()*1000)) - since))
        # test(model, device, train_loader)
        since = int(round(time.time()*1000))
        #
        model.test_m(device, test_loader)
        #
        time_elapsed += int(round(time.time()*1000)) - since
        print('Test time elapsed: {}ms'.format(int(round(time.time()*1000)) - since))
        # test(model, device, train_loader)
        scheduler.step()

    if args.test_error:
        all_accuracies = model.test_m_error(device, test_loader)
        to_dump_data = dump_exp_data(model, args, all_accuracies)
        store_exp_data(to_dump_path, to_dump_data)

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")

    # TODO: ONNX save

if __name__ == '__main__':
    main()
