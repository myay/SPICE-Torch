#include <torch/extension.h>
#include <vector>

// CUDA forward declaration
torch::Tensor custommac2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor output,
    int array_size
  );

#define CHECK_CUDA(x) AT_ASSERTM(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor custommac2d(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor output,
    int array_size
  ) {
  CHECK_INPUT(input);
  CHECK_INPUT(weight);
  CHECK_INPUT(output);
  return custommac2d_cuda(input, weight, output, array_size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("custommac2d", &custommac2d, "CUSTOMMAC2D");
}
