#include <torch/extension.h>

#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor mappingdirect_cuda(
    torch::Tensor input,
    torch::Tensor mapping
  );

torch::Tensor mappingdirect(
    torch::Tensor input,
    torch::Tensor mapping
  ) {
  CHECK_INPUT(input);
  CHECK_INPUT(mapping);
  return mappingdirect_cuda(input, mapping);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mappingdirect", &mappingdirect, "MAPPINGDIRECT");
}
