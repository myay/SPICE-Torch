'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from QuantizedNN import QuantizedLinear, QuantizedConv2d, QuantizedActivation
c
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, quantMethod=None, an_sim=None, array_size=None, mapping=None, mapping_distr=None, sorted_mapping_idx=None, performance_mode=None, quantize_train=True, quantize_eval=True, error_model=None, train_model=None, extract_absfreq=None):
        super(BasicBlock, self).__init__()
        self.htanh = nn.Hardtanh()
        self.qact = QuantizedActivation(quantization=quantMethod)
        self.conv1 = QuantizedConv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, quantization=quantMethod, an_sim=an_sim, array_size=array_size, mac_mapping=mapping, mac_mapping_distr=mapping_distr, sorted_mac_mapping_idx=sorted_mapping_idx,
            performance_mode=performance_mode,
            error_model=error_model, bias=False, train_model=train_model, extract_absfreq=extract_absfreq)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = QuantizedConv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, quantization=quantMethod, an_sim=an_sim, array_size=array_size, mac_mapping=mapping, mac_mapping_distr=mapping_distr, sorted_mac_mapping_idx=sorted_mapping_idx,
                               performance_mode=performance_mode,
                               error_model=error_model, bias=False, train_model=train_model, extract_absfreq=extract_absfreq)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                QuantizedConv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, quantization=quantMethod, an_sim=an_sim, array_size=array_size, mac_mapping=mapping, mac_mapping_distr=mapping_distr, sorted_mac_mapping_idx=sorted_mapping_idx,
                          performance_mode=performance_mode,
                          error_model=error_model, bias=False, train_model=train_model, extract_absfreq=extract_absfreq),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.qact(self.htanh(self.bn1(self.conv1(x))))
        # out = self.qact(self.htanh(self.bn2(self.conv2(out))))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.qact(self.htanh(out))
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, train_crit, test_crit, quantMethod=None, an_sim=None, array_size=None, mapping=None, mapping_distr=None, sorted_mapping_idx=None, performance_mode=None, quantize_train=True, quantize_eval=True, error_model=None, train_model=None, extract_absfreq=None, num_classes=10):
        super(ResNet, self).__init__()
        self.name = "ResNet18"
        self.traincriterion = train_crit
        self.testcriterion = test_crit
        self.quantization = quantMethod
        self.q_train = quantize_train
        self.q_test = quantize_eval
        self.error_model = error_model
        self.an_sim = an_sim
        self.array_size = array_size
        self.mapping = mapping
        self.mapping_distr = mapping_distr
        self.sorted_mapping_idx = sorted_mapping_idx
        self.performance_mode = performance_mode
        self.train_model = train_model
        self.extract_absfreq = extract_absfreq
        self.in_planes = 64

        self.htanh = nn.Hardtanh()
        self.qact = QuantizedActivation(quantization=self.quantization)

        self.conv1 = QuantizedConv2d(3, 64, kernel_size=3, stride=1, padding=1, quantization=self.quantization, error_model=self.error_model, bias=False, array_size=self.array_size)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = QuantizedLinear(512*block.expansion, num_classes, quantization=self.quantization, an_sim=self.an_sim, array_size=self.array_size, mac_mapping=self.mapping, mac_mapping_distr=self.mapping_distr, sorted_mac_mapping_idx=self.sorted_mapping_idx, performance_mode=self.performance_mode, error_model=self.error_model, bias=False, train_model=self.train_model, extract_absfreq=self.extract_absfreq)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, quantMethod=self.quantization, an_sim=self.an_sim, array_size=self.array_size, mapping=self.mapping, mapping_distr=self.mapping_distr, sorted_mapping_idx=self.sorted_mapping_idx, performance_mode=self.performance_mode, error_model=self.error_model, train_model=self.train_model, extract_absfreq=self.extract_absfreq))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.qact(self.htanh(self.bn1(self.conv1(x))))
        out = self.layer1(out)
        # out = F.max_pool2d(out, 2)
        out = self.layer2(out)
        # out = F.max_pool2d(out, 2)
        out = self.layer3(out)
        # out = F.max_pool2d(out, 2)
        out = F.max_pool2d(out, 2)
        out = self.layer4(out)
        out = F.max_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        # print("---")
        return out
