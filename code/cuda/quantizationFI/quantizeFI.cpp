#include <torch/extension.h>
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// 8 bit function
torch::Tensor fi_uint8_cuda(
    torch::Tensor input,
    float f01,
    float f10,
    int nrbits
  );

torch::Tensor fi_uint8(
    torch::Tensor input,
    float f01,
    float f10,
    int nrbits
  ) {
  CHECK_INPUT(input);
  return fi_uint8_cuda(input, f01, f10, nrbits);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("bfi_8bit", &fi_uint8, "BFI_8BIT");
}
