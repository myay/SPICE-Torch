#include <torch/extension.h>

#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor quantize_cuda(
    torch::Tensor input, // tensor to quantize
    float min_range, // min value
    float max_range, // max value
    int q, // 2^q is desired range
    int unsign // when 0, use unsigned, otherwise signed
  );

torch::Tensor quantize(
    torch::Tensor input,
    float min_range,
    float max_range,
    int q,
    int unsign
  ) {
  CHECK_INPUT(input);
  return quantize_cuda(input, min_range, max_range, q, unsign);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("quantize", &quantize, "QUANTIZE");
}
