#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdio.h>
#include <cstdint>

#include <curand.h>
#include <curand_kernel.h>
#include <chrono>

#define DEBUG_1D 0
#define DEBUG_THREAD_INFO_FLOAT32 0
#define DEBUG_THREAD_INFO_INT32 0
#define DEBUG_BITS 0
#define DEBUG_SEEDS 0

// bit stuff https://stackoverflow.com/questions/111928/is-there-a-printf-converter-to-print-in-binary-format

// for rng stuff: http://ianfinlayson.net/class/cpsc425/notes/cuda-random
template <typename scalar_t>
__global__ void binarizePM1FI_kernel(
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> input,
    float f01,
    float f10,
    unsigned long long seed0
  ) {

  // handle access indices
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int d = blockIdx.y * blockDim.y + threadIdx.y;
  const int e = blockIdx.z * blockDim.z + threadIdx.z;

  if ((c < input.size(0)) && (d < input.size(1)) && (e < input.size(2)))
  {
    // creates a unique value from three values with cantor pairing function
    // the combination of (c,d,e) -> cantor_val (N^3->N) is bijective
    unsigned long long k1 = 0.5*(c+d)*(c+d+1)+d;
    unsigned long long cantor_val = 0.5*(k1+e)*(k1+e+1)+e;

    curandState_t state0;
    curand_init(seed0+cantor_val, 0, 0, &state0);

    // if input is negative
    if (input[c][d][e] <= 0)
    {
      if (curand_uniform(&state0) < f01)
      {
        input[c][d][e] *= (-1);
      }
    }
    else
    {
      if (curand_uniform(&state0) < f10)
      {
        input[c][d][e] *= (-1);
      }
    }
  }
}

torch::Tensor binarizePM1FI_cuda(
  torch::Tensor input,
  float f01,
  float f10
) {
  // The number of thread blocks in a grid is usually dictated by the size of the data being processed, which typically exceeds the number of processors in the system.
  // dim3 threadsPerBlock(8,8,8)
  // <<<number of blocks per grid, number of threads ber block>>>
  // grid is created with enough blocks to have one thread per matrix element

  // https://devtalk.nvidia.com/default/topic/1028226/how-many-concurrent-threads-are-running-on-my-geforce-gtx-1080-ti-/

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
  int threads_x = 8; // per block, 8
  int threads_y = 8; // per block, 8
  int threads_z = 8; // per block, 8

  #if DEBUG_1D
    threads_x = 1;
    threads_y = 1;
    threads_z = 1;
  #endif

  const dim3 threads(threads_x,threads_y, threads_z);
  const dim3 blocks((input_size_x + threads_x - 1) / threads_x,
                    (input_size_y + threads_y - 1) / threads_y,
                    (input_size_z + threads_z - 1) / threads_z);

  // create a seed from the current time in nanoseconds
  auto now = std::chrono::system_clock::now();
  auto now_ms = std::chrono::time_point_cast<std::chrono::nanoseconds>(now);
  auto value = now_ms.time_since_epoch();
  unsigned long long seed0 = value.count();

  AT_DISPATCH_ALL_TYPES(input.type(), "binarizePM1FI_cuda", ([&] {
    binarizePM1FI_kernel<scalar_t><<<blocks, threads>>>(
        input.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
        f01,
        f10,
        seed0
    );
  }));
  input = input.reshape(shape_original);
  return input;
}
