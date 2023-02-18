# SPICE-Torch
A framework for connecting SPICE simulations of analog computing neuron circuits with PyTorch accuracy evaluations for Binarized (and soon Quantized) Neural Networks.

Tested setups:
- Python 3.10.6 (pip + venv), PyTorch 1.13.1, GeForce RTX 3060 Ti 8GB (Driver Version: 525.60.13, CUDA Version: 12.0), Ubuntu 22.04.1
- Python 3.6.9, PyTorch 1.5.0, GeForce GTX 1060 6GB (Driver Version: 440.100, CUDA Version: 10.2), Ubuntu 18.04.4
- Python 3.6.13 (conda), 1.7.0+cu110, GeForce GTX 1080 8GB (Driver Version: 450.102.04, CUDA Version: 11.0), Ubuntu 18.04.6
- Python 3.9.7, PyTorch 1.9.0, GeForce GTX 3080 10GB (Driver Version: 512.15, CUDA Version: 11.6)

Supported features:
- BNN models for FC, VGG3, VGG7, ResNet18
- Datasets: FashionMNIST, KMNIST, SVHN, CIFAR10, IMAGENETTE
- Error model application for Linear and Conv2d layers (direct mapping and distribution-based mapping)
- Variable crossbar array size
- Training with error models
- Performance mode (MAC engine with error application are fused into one GPU-kernel)

TODOs:
- Optimize MAC engine for execution time
- Support for QNNs
- More NN models

## CUDA-based Error Model Application and Binarization/Quantization

First, install PyTorch. For fast binarization/quantization and error injection during training, CUDA support is needed. To enable it, install pybind11 and CUDA toolkit.

Then, to install the CUDA-kernels, go to the folder ```code/cuda/``` and run

```./install_kernels.sh```

After successful installation of all kernels, run the binarization/quantization-aware training with error injection for BNNs using

```python3 run.py --model=VGG3 --dataset=FMNIST --batch-size=256 --epochs=5 --lr=0.001 --step-size=2 --gamma=0.5 --train-model=1 --save-model=vgg3_test```.

Here is a list of the command line parameters for running the error evaluations with SPICE-Torch:
| Command line parameter | Options |
| :------------- |:-------------|
| --model      | FC, VGG3, VGG7, ResNet |
| --dataset      | MNIST, FMNIST, QMNIST, SVHN, CIFAR10, IMAGENETTE |
| --an-sim      | int, whether to turn on the mapping from SPICE, default: None |
| --mapping      | string, loads a direct mapping from the specified path, default: None |
| --mapping-distr      | string, loads a distribution-based mapping from the specified path, default: None |
| --array-size      | int, specifies the size of the crossbar array, default: None |
| --performance-mode      | int, specify whether to activate the faster and more memory-efficient performance mode (when using this sub-MAC results can only be changed in cuda-kernel!), default: None |
| --print-accuracy      | int, specifies whether to print inference accuracy, default: None |
| --test-error-distr      | int, specifies the number of repetitions to perform in accuracy evaluations for distribution based evaluation, default: None |
| --train-model      | bool, whether to train a model, default: None|
| --epochs      | int, number of epochs to train, default: 10|
| --lr      | float, learning rate, default: 1.0|
| --gamma      | float, learning rate step, default: 0.5|
| --step-size      | int, learning rate step site, default: 5|
| --batch-size      | int, specifies the batch size in training, default: 64|
| --test-batch-size      | int, specifies the batch size in testing, default: 1000|
| --save-model | string, saves a trained model with the specified name in the string, default:None |
| --load-model-path | string, loads a model from the specified path in the string, default: None |
| --load-training-state | string, saves a training state with the specified name in the string, default:None |
| --save-training-state | string, loads a training state from the specified path in the string, default: None |
| --gpu-num | int, specifies the GPU on which the training should be performed, default: 0 |
| --profile-time | int, Specify whether to profile the execution time by specifying the repetitions, default: None |
| --extract-absfreq | int, Specify whether to extract the absolute frequency of MAC values, default: None |
| --extract-absfreq-resnet | int, Specify whether to extract the absolute frequency of MAC values for resnet, default: None |

More information on the command line parameters can be found [here](https://github.com/myay/IFneuronSPICE/blob/main/code/python/Utils.py#L55).

Please contact me if you have any questions: mikail.yayla@tu-dortmund.de.
