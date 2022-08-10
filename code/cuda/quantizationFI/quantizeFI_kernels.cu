#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cstdint>
#include <curand.h>
#include <curand_kernel.h>
#include <chrono>

// threads per block
#define TPB_X 8
#define TPB_Y 8
#define TPB_Z 8

// 8 bit function
template <typename scalar_t>
__global__ void fi_uint8_cuda_kernel(
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> input,
    float f01,
    float f10,
    unsigned long long seed0,
    int nrbits
  ) {

  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int d = blockIdx.y * blockDim.y + threadIdx.y;
  const int e = blockIdx.z * blockDim.z + threadIdx.z;

  if ((c < input.size(0)) && (d < input.size(1)) && (e < input.size(2)))
  {
    // scalar_t input_val_f = input[c][d][e];

    // uint8_t inj_val = *(uint8_t *)&(input_val_f);
    uint8_t inj_val = input[c][d][e];

    // creates a unique value from three values with cantor pairing function
    // the combination of (c,d,e) -> cantor_val (N^3->N) is bijective
    unsigned long long k1 = 0.5*(c+d)*(c+d+1)+d;
    unsigned long long cantor_val = 0.5*(k1+e)*(k1+e+1)+e;

    curandState_t state0;
    curand_init(seed0+cantor_val, 0, 0, &state0);

    uint8_t injector = 0x1;
    for (int i = 0; i < nrbits; i++)
    {
      // check whether it is a zero or a 1 at current bit position i
      if ((injector & inj_val) == 0)
      {
        // case 0 & 1
        inj_val ^= ((uint8_t)(curand_uniform(&state0) < f01) << i);
      }
      else
      {
        // case 1 & 1
        inj_val ^= ((uint8_t)(curand_uniform(&state0) < f10) << i);
      }
      injector = injector << 1;
    }
    input[c][d][e] = inj_val;
  }
}

torch::Tensor fi_uint8_cuda(
  torch::Tensor input,
  float f01,
  float f10,
  int nrbits
) {

  int64_t shape_len = input.dim();
  std::vector<int64_t> shape_original;
  for (int i = 0; i < shape_len; i++)
  {
    shape_original.push_back(input.size(i));
  }

  if (shape_len == 1)
  {
    input = input.reshape({input.size(0),1,1});
  }
  if (shape_len == 2)
  {
    input = input.reshape({input.size(0),input.size(1),1});
  }
  if (shape_len > 3)
  {
    input = input.reshape({input.size(0),input.size(1),-1});
  }

  const int input_size_x = input.size(0);
  const int input_size_y = input.size(1);
  const int input_size_z = input.size(2);
  int threads_x = TPB_X; // per block, 8
  int threads_y = TPB_Y;
  int threads_z = TPB_Z;

  const dim3 threads(threads_x,threads_y, threads_z);
  const dim3 blocks((input_size_x + threads_x - 1) / threads_x,
                    (input_size_y + threads_y - 1) / threads_y,
                    (input_size_z + threads_z - 1) / threads_z);

  // create a seed from the current time in nanoseconds
  auto now = std::chrono::system_clock::now();
  auto now_ms = std::chrono::time_point_cast<std::chrono::nanoseconds>(now);
  auto value = now_ms.time_since_epoch();
  unsigned long long seed0 = value.count();

  AT_DISPATCH_ALL_TYPES(input.type(), "fi_uint8_cuda", ([&] {
    fi_uint8_cuda_kernel<scalar_t><<<blocks, threads>>>(
        input.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
        f01,
        f10,
        seed0,
        nrbits
    );
  }));
  input = input.reshape(shape_original);
  return input;
}
