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

template <typename scalar_t>
__global__ void custommac2dmappingdistr_kernel(
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> input,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> weight,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> output,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> mapping_distr,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> mapping_distr_sorted_idx,
    int array_size,
    unsigned long long seed0
  )
{

  // handle access indices
  const int c = blockIdx.x * blockDim.x + threadIdx.x; // y
  const int d = blockIdx.y * blockDim.y + threadIdx.y; // x
  const int e = blockIdx.z * blockDim.z + threadIdx.z; // z

  // make sure we don't modify memory regions outside of output
  if ((d < output.size(0)) && (c < output.size(1)) && (e < output.size(2)))
  {
    // unsigned long long k1 = 0.5*(c+d)*(c+d+1)+d;
    // unsigned long long cantor_val = 0.5*(k1+e)*(k1+e+1)+e;
    unsigned long long cantor_val = 0.5*(c+d)*(c+d+1)+d;
    curandState_t state0;
    curand_init(seed0+cantor_val, 0, 0, &state0);

    int cycle_counter = 0;
    float shifted_mac_result = 0;
    float sub_mac_result = 0;
    for(int i = 0; i < weight.size(1); i++)
    {
      //printf("Thread: (%d,%d,%d)\nWeight: %.4f, Input: %.4f\n", c, d, e, weight[c][i], input[d][i][e]);
      sub_mac_result += (weight[c][i] * input[d][i][e]);
      cycle_counter += 1;

      if((cycle_counter == array_size) || (i == (weight.size(1)-1)))
      {
        shifted_mac_result = (sub_mac_result + array_size)/2;
        // shifted_mac_result = mapping[int(shifted_mac_result)];
        // apply prob based mapping here
        // sample a random number
        float rand_val = curand_uniform(&state0);
        // find out where this random value lies in mapping_distr[mac_value]
        for (int i = 0; i < mapping_distr.size(0); i++)
        {
          if (i < mapping_distr.size(0)-1)
          {
            float prob_tmp = mapping_distr[int(shifted_mac_result)][int(mapping_distr_sorted_idx[int(shifted_mac_result)][i])];
            float prob_tmp_next = mapping_distr[int(shifted_mac_result)][int(mapping_distr_sorted_idx[int(shifted_mac_result)][i+1])];
            if ((rand_val >= prob_tmp) && (rand_val <= prob_tmp_next))
            {
              // set mac value
              sub_mac_result = mapping_distr_sorted_idx[int(shifted_mac_result)][i+1];
              break;
            }
          }
          else if (i == mapping_distr.size(0)-1)
          {
            float prob_tmp = mapping_distr[int(shifted_mac_result)][int(mapping_distr_sorted_idx[int(shifted_mac_result)][i])];
            if(rand_val <= prob_tmp)
            {
              // set mac value
              sub_mac_result = mapping_distr_sorted_idx[int(shifted_mac_result)][i];
              break;
            }
          }
        }
        sub_mac_result = 2*sub_mac_result - array_size;
        output[d][c][e] += sub_mac_result;
        sub_mac_result = 0;
        shifted_mac_result = 0;
        cycle_counter = 0;
      }
    }
  }
}

torch::Tensor custommac2dmappingdistr_cuda(
  torch::Tensor input,
  torch::Tensor weight,
  torch::Tensor output,
  torch::Tensor mapping_distr,
  torch::Tensor mapping_distr_sorted_idx,
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

  // create a seed from the current time in nanoseconds
  auto now = std::chrono::system_clock::now();
  auto now_ms = std::chrono::time_point_cast<std::chrono::nanoseconds>(now);
  auto value = now_ms.time_since_epoch();
  unsigned long long seed0 = value.count();

  AT_DISPATCH_ALL_TYPES(input.type(), "custommac2dmappingdistr_cuda", ([&] {
    custommac2dmappingdistr_kernel<scalar_t><<<blocks, threads>>>(
        input.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
        weight.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        output.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
        mapping_distr.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        mapping_distr_sorted_idx.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        array_size,
        seed0
    );
  }));

  return output;
}
