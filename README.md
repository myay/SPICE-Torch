# BNN-QNN-ErrorEvaluation
A framework for error tolerance training and evaluation of Binarized and Quantized Neural Networks

## CUDA-based Binarization, Quantization, and Error Injection

First, install PyTorch. For fast binarization/quantization and error injection during training, CUDA support is needed. To enable it, install pybind11 and CUDA toolkit.

Then, to install the CUDA-kernels, go to the folder ```code/cuda/``` and run

```./install_kernels.sh```

After successful installation of all kernels, run the binarization/quantization-aware training with error injection for BNNs using

```python3 run.py --model=VGG3 --dataset=FMNIST --batch-size=256 --epochs=5 --lr=0.001 --step-size=2 --gamma=0.5 --train-model=1```,

and for QNNs using

TODO

The code is an extension based on the MNIST example in https://github.com/pytorch/examples/tree/master/mnist.

Please contact me if you have any questions: mikail.yayla@tu-dortmund.de.
