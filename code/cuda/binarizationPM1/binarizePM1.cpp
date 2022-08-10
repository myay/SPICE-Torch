#include <torch/extension.h>

#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor binarizePM1_cuda(
    torch::Tensor input
  );

torch::Tensor binarizePM1(
    torch::Tensor input
  ) {
  CHECK_INPUT(input);
  return binarizePM1_cuda(input);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("binarize", &binarizePM1, "BINARIZEPM1");
}
