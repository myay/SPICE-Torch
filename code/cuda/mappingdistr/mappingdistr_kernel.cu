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

template <typename scalar_t>
__global__ void mappingdistr_kernel(
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> input,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> mapping_distr,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> mapping_distr_sorted_idx,
    unsigned long long seed0
  ) {

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

    // find MAC-value
    int mac_value = int(input[c][d][e]);
    // mapping_distr[mac_value][i] and mapping_distr_sorted_idx[mac_value][i] needs to be used

    // print MAC value
    // #if 0
    //   if (c == 0 && d == 0 && e == 0)
    //   {
    //     printf(" (-1) Mac-value: %d\n", mac_value);
    //   }
    // #endif

    // sample a random number
    float rand_val = curand_uniform(&state0);

    // #if 0
    //   if (c == 0 && d == 0 && e == 0)
    //   {
    //     printf(" (0) Rand. nr: %.2f\n", rand_val);
    //   }
    // #endif

    // find out where this random value lies in mapping_distr[mac_value]
    for (int i = 0; i < mapping_distr.size(0); i++)
    {
      if (i < mapping_distr.size(0)-1)
      {
        float prob_tmp = mapping_distr[mac_value][int(mapping_distr_sorted_idx[mac_value][i])];
        float prob_tmp_next = mapping_distr[mac_value][int(mapping_distr_sorted_idx[mac_value][i+1])];
        if ((rand_val >= prob_tmp) && (rand_val <= prob_tmp_next))
        {
          // set mac value
          input[c][d][e] = mapping_distr_sorted_idx[mac_value][i+1];

          // print set MAC value
          // #if 0
          //   if (c == 0 && d == 0 && e == 0)
          //   {
          //     printf(" (1) Set mac-value: %.2f\n", input[c][d][e]);
          //   }
          // #endif

          break;
        }
      }
      else if (i == mapping_distr.size(0)-1)
      {
        float prob_tmp = mapping_distr[mac_value][int(mapping_distr_sorted_idx[mac_value][i])];
        if(rand_val <= prob_tmp)
        {
          // set mac value
          input[c][d][e] = mapping_distr_sorted_idx[mac_value][i];

          // print set MAC value
          // #if 0
          //   if (c == 0 && d == 0 && e == 0)
          //   {
          //     printf(" (2) Set mac-value: %.2f\n", input[c][d][e]);
          //   }
          // #endif

          break;
        }
      }
    }
    // input[c][d][e] = input[c][d][e];
    // #if 0
    //   if (c == 0 && d == 0 && e == 0)
    //   {
    //     printf("\n---\n");
    //   }
    // #endif
  }
}

torch::Tensor mappingdistr_cuda(
  torch::Tensor input,
  torch::Tensor mapping_distr,
  torch::Tensor mapping_distr_sorted_idx
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

  const dim3 threads(threads_x,threads_y, threads_z);
  const dim3 blocks((input_size_x + threads_x - 1) / threads_x,
                    (input_size_y + threads_y - 1) / threads_y,
                    (input_size_z + threads_z - 1) / threads_z);

  // create a seed from the current time in nanoseconds
  auto now = std::chrono::system_clock::now();
  auto now_ms = std::chrono::time_point_cast<std::chrono::nanoseconds>(now);
  auto value = now_ms.time_since_epoch();
  unsigned long long seed0 = value.count();

  AT_DISPATCH_ALL_TYPES(input.type(), "mappingdistr_cuda", ([&] {
    mappingdistr_kernel<scalar_t><<<blocks, threads>>>(
        input.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
        mapping_distr.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        mapping_distr_sorted_idx.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        seed0
    );
  }));

  input = input.reshape(shape_original);
  return input;
}
