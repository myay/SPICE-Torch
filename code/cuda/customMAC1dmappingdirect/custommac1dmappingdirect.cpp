#include <torch/extension.h>
#include <vector>

// CUDA forward declaration
torch::Tensor custommac1dmappingdirect_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor output,
    torch::Tensor mapping,
    int array_size
  );

#define CHECK_CUDA(x) AT_ASSERTM(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor custommac1dmappingdirect(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor output,
    torch::Tensor mapping,
    int array_size
  ) {
  CHECK_INPUT(input);
  CHECK_INPUT(weight);
  CHECK_INPUT(output);
  CHECK_INPUT(mapping);
  return custommac1dmappingdirect_cuda(input, weight, output, mapping, array_size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("custommac1dmappingdirect", &custommac1dmappingdirect, "CUSTOMMAC1DMAPPINGDIRECT");
}
