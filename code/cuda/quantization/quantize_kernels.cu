#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cstdint>

// threads per block
#define TPB_X 8
#define TPB_Y 8
#define TPB_Z 8

template <typename scalar_t>
__global__ void quantize_kernel(
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> input,
    float min_range,
    float max_range,
    float q_range,
    int unsign
  ) {

  // float m_range = 0;
  // if (max_range > min_range)
  // {
  //   m_range = max_range;
  // }
  // else
  // {
  //   m_range = max_range;
  // }

  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int d = blockIdx.y * blockDim.y + threadIdx.y;
  const int e = blockIdx.z * blockDim.z + threadIdx.z;

  if ((c < input.size(0)) && (d < input.size(1)) && (e < input.size(2)))
  {
    input[c][d][e] = std::round((input[c][d][e] - min_range) * q_range / (max_range - min_range));
    if (unsign == 0)
    {
      input[c][d][e] -= (q_range + 1) / 2.0;
    }
    // input[c][d][e] /= q_range;
    // input[c][d][e] *= m_range;
  }
}

torch::Tensor quantize_cuda(
  torch::Tensor input,
  float min_range,
  float max_range,
  int q,
  int unsign
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
  // The number of thread blocks in a grid is usually dictated by the size of the data being processed, which typically exceeds the number of processors in the system.
  // dim3 threadsPerBlock(8,8,8)
  // <<<number of blocks per grid, number of threads ber block>>>
  // grid is created with enough blocks to have one thread per matrix element

  // https://devtalk.nvidia.com/default/topic/1028226/how-many-concurrent-threads-are-running-on-my-geforce-gtx-1080-ti-/
  const int input_size_x = input.size(0);
  const int input_size_y = input.size(1);
  const int input_size_z = input.size(2);
  int threads_x = TPB_X; // per block, 8
  int threads_y = TPB_Y;
  int threads_z = TPB_Z;

  const float q_range = std::pow(2, q) - 1.0;

  const dim3 threads(threads_x,threads_y, threads_z);
  const dim3 blocks((input_size_x + threads_x - 1) / threads_x,
                    (input_size_y + threads_y - 1) / threads_y,
                    (input_size_z + threads_z - 1) / threads_z);

  AT_DISPATCH_ALL_TYPES(input.type(), "quantize_cuda", ([&] {
    quantize_kernel<scalar_t><<<blocks, threads>>>(
        input.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
        min_range,
        max_range,
        q_range,
        unsign
    );
  }));

  input = input.reshape(shape_original);
  return input;
}
