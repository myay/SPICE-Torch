#include <torch/extension.h>

#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor mappingdistr_cuda(
    torch::Tensor input,
    torch::Tensor mapping_distr,
    torch::Tensor mapping_distr_sorted_idx
  );

torch::Tensor mappingdistr(
    torch::Tensor input,
    torch::Tensor mapping_distr,
    torch::Tensor mapping_distr_sorted_idx
  ) {
  CHECK_INPUT(input);
  CHECK_INPUT(mapping_distr);
  CHECK_INPUT(mapping_distr_sorted_idx);
  return mappingdistr_cuda(input, mapping_distr, mapping_distr_sorted_idx);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mappingdistr", &mappingdistr, "MAPPINGDISTR");
}
