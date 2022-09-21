#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdio.h>
#include <cstdint>

#define DEBUG_1D 0
#define DEBUG_THREAD_INFO_FLOAT32 0
#define DEBUG_THREAD_INFO_INT32 0
#define DEBUG_BITS 0
#define DEBUG_SEEDS 0

template <typename scalar_t>
__global__ void custommac1dmappingdirect_kernel(
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> input,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> weight,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> output,
    torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> mapping,
    int array_size
  )
{

  // handle access indices
  const int c = blockIdx.x * blockDim.x + threadIdx.x; // y
  const int d = blockIdx.y * blockDim.y + threadIdx.y; // x

  // make sure we don't modify memory regions outside of output
  if ((d < output.size(0)) && (c < output.size(1)))
  {
    int cycle_counter = 0;
    float shifted_mac_result = 0;
    float sub_mac_result = 0;
    for(int i = 0; i < weight.size(1); i++)
    {
        //printf("Thread: (%d,%d,%d)\nWeight: %.4f, Input: %.4f\n", c, d, e, weight[c][i], input[d][i][e]);
        sub_mac_result += (weight[c][i] * input[d][i]);
        cycle_counter += 1;

        if((cycle_counter == array_size) || (i == (weight.size(1)-1)))
        {
          shifted_mac_result = (sub_mac_result + array_size)/2;
          shifted_mac_result = mapping[int(shifted_mac_result)];
          sub_mac_result = 2*shifted_mac_result - array_size;
          output[d][c] += sub_mac_result;
          sub_mac_result = 0;
          shifted_mac_result = 0;
          cycle_counter = 0;
        }
    }
  }
}

torch::Tensor custommac1dmappingdirect_cuda(
  torch::Tensor input,
  torch::Tensor weight,
  torch::Tensor output,
  torch::Tensor mapping,
  int array_size
) {
  // The number of thread blocks in a grid is usually dictated by the size of the data being processed, which typically exceeds the number of processors in the system.
  // dim3 threadsPerBlock(8,8,8)
  // <<<number of blocks per grid, number of threads ber block>>>
  // grid is created with enough blocks to have one thread per matrix element

  // https://devtalk.nvidia.com/default/topic/1028226/how-many-concurrent-threads-are-running-on-my-geforce-gtx-1080-ti-/
  const int output_size_x = output.size(1);
  const int output_size_y = output.size(0);
  int threads_x = 16; // per block, 16
  int threads_y = 16; // per block, 16

  #if DEBUG_1D
    threads_x = 1;
    threads_y = 1;
  #endif

  const dim3 threads(threads_x,threads_y);
  const dim3 blocks((output_size_x + threads_x - 1) / threads_x,
                    (output_size_y + threads_y - 1) / threads_y);

  AT_DISPATCH_ALL_TYPES(input.type(), "custommac1dmappingdirect_cuda", ([&] {
    custommac1dmappingdirect_kernel<scalar_t><<<blocks, threads>>>(
        input.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        weight.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        output.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        mapping.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
        array_size
    );
  }));

  return output;
}
