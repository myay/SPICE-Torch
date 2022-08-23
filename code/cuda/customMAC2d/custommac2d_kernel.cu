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
__global__ void custommac1d_kernel(
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> input,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> weight,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> output,
    int array_size
  )
{

  // handle access indices
  const int c = blockIdx.x * blockDim.x + threadIdx.x; // y
  const int d = blockIdx.y * blockDim.y + threadIdx.y; // x
  const int e = blockIdx.z * blockDim.z + threadIdx.z; // z

  // make sure we don't modify memory regions outside of output
  if ((d < output.size(0)) && (c < output.size(1)) && (e < output.size(2)))
  {
    // this is (c,d,e), we have as many threads as we have pixels in output out
    // each thread of out calculates a MAC (row of filter times column of input)

    // every thread is responsible for one sum, there are as many threads as mac sums in output
    int cycle_counter = 0; // nr of ready inputs to majority gate
    int global_cycles = 0; // counter for all cycles
    float mac_result = 0;
    for(int i = 0; i < weight.size(1); i++)
    {
      //printf("Thread: (%d,%d,%d)\nWeight: %.4f, Input: %.4f\n", c, d, e, weight[c][i], input[d][i][e]);
      mac_result += (weight[c][i] * input[d][i][e]);
      // output[d][c][] += mult_result;
      cycle_counter += 1;

      // when at last element, store mac result
      if(i == (weight.size(1)-1))
      {
        output[d][c][e][global_cycles] = mac_result;
        mac_result = 0;
        cycle_counter = 0;
        global_cycles += 1;
      }
      else if(cycle_counter == array_size)
      {
        output[d][c][e][global_cycles] = mac_result;
        mac_result = 0;
        cycle_counter = 0;
        global_cycles += 1;
      }
    }
  }
}

torch::Tensor custommac2d_cuda(
  torch::Tensor input,
  torch::Tensor weight,
  torch::Tensor output,
  int array_size
) {
  // The number of thread blocks in a grid is usually dictated by the size of the data being processed, which typically exceeds the number of processors in the system.
  // dim3 threadsPerBlock(8,8,8)
  // <<<number of blocks per grid, number of threads ber block>>>
  // grid is created with enough blocks to have one thread per matrix element

  // https://devtalk.nvidia.com/default/topic/1028226/how-many-concurrent-threads-are-running-on-my-geforce-gtx-1080-ti-/
  const int output_size_x = output.size(1);
  const int output_size_y = output.size(0);
  const int output_size_z = output.size(2);
  int threads_x = 8; // per block, 8
  int threads_y = 8; // per block, 8
  int threads_z = 8; // per block, 8

  #if DEBUG_1D
    threads_x = 1;
    threads_y = 1;
    threads_z = 1;
  #endif

  const dim3 threads(threads_x, threads_y, threads_z);
  const dim3 blocks((output_size_x + threads_x - 1) / threads_x,
                    (output_size_y + threads_y - 1) / threads_y,
                    (output_size_z + threads_z - 1) / threads_z);

  AT_DISPATCH_ALL_TYPES(input.type(), "custommac2d_cuda", ([&] {
    custommac1d_kernel<scalar_t><<<blocks, threads>>>(
        input.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
        weight.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        output.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
        array_size
    );
  }));

  return output;
}
