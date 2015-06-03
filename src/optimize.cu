#include <cassert>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <cstdio>
#include <cuda_runtime.h>
#include "optimize.cuh"

#include <curand.h>


using namespace std;


/* User defined objective function goes here. */
__device__ __host__ float f(float x) {
  return 20 * log(x) - .0004 * x * x * sin(x) + x * cos(x);
}

/* Generate a float between low and high, given r between 0 and 1. */
__device__ float rand_float(float low, float high, float r) {
  return low + r * (high - low);
}

/* Apply objective function to all inputs. */
__global__
void applyFunctionKernel(float *input, float *output, int N, float low, float high) {

  int index = blockIdx.x * blockDim.x + threadIdx.x;

  while (index < N) {
    float r = input[index];

    /* Convert r (which is in [0, 1]) into a valid input and
     * write this to the input. This is convenient to do here
     */
    input[index] = rand_float(low, high, r);

    output[index] = f(rand_float(low, high, r));

    index += gridDim.x * blockDim.x;
  }
}

void callApplyFunctionKernel(float *input, float *output, int N, float low, float high) {

  int block_count = 32;
  int threads_per_block = 32;

  applyFunctionKernel<<<block_count, threads_per_block>>>(input, output, N, low, high);
}

/* A bit of boilerplate to allow for a critical section. Borrowed from
 * http://stackoverflow.com/questions/18963293/cuda-atomics-change-flag
 */

__device__ volatile int sem = 0;

__device__ void acquire_semaphore(volatile int *lock){
  while (atomicCAS((int *)lock, 0, 1) != 0);
  }

__device__ void release_semaphore(volatile int *lock){
  *lock = 0;
  __threadfence();
}

/* Find max of a subset by using shmem, then combine solutions
 * across blocks. */
__global__
void findMaxKernel(float *output, int N, float *max, int *max_index) {
  /* Local max of the values looked at by this block */
  __shared__ float local_max;
  __shared__ int local_max_index;

  int index = blockIdx.x * blockDim.x + threadIdx.x;

  /* Initialize shmem */
  if (threadIdx.x == 0) {
    local_max = output[0];
    local_max_index = 0;
  }

  if (index == 0) {
    *max = output[0];
    *max_index = 0;
  }

  while (index < N) {

    float tmp = output[index];

    if (tmp > local_max) {
      local_max = tmp;
      local_max_index = index;
    }

    index += gridDim.x * blockDim.x;
  }

  __syncthreads();

  /* One thread per block compares values across blocks */

  if (threadIdx.x == 0) {
    __syncthreads();
    if (threadIdx.x == 0) {
      acquire_semaphore(&sem);
    }
    __syncthreads();

    // critical section
    if (local_max > *max) {
      *max = local_max;
      *max_index = local_max_index;
    }

    __syncthreads();
    if (threadIdx.x == 0) {
      release_semaphore(&sem);
    }
    __syncthreads();

  }
}

OptimizationOutput cudaCallMaximumKernel(float *input, float *output, int N) {
  
  int block_count = 32;
  int threads_per_block = 32;

  /* Set up buffers for GPU */

  float *dev_max;
  int *dev_max_index;

  gpuErrChk(cudaMalloc(&dev_max, sizeof(float)));
  gpuErrChk(cudaMalloc(&dev_max_index, sizeof(int)));

  /* Call kernel */
  findMaxKernel<<<block_count, threads_per_block>>>(output, N, dev_max,
    dev_max_index);

  /* Now get values off of GPU. This part is a bit awkward: need the
   * index in order to find the corresponding input */

  float *host_maximizer = (float *) malloc(sizeof(float));
  float *host_max = (float *) malloc(sizeof(float));
  int *host_max_index = (int *) malloc(sizeof(int));

  *host_maximizer = -1.;
  *host_max = -1.;
  *host_max_index = -1;

  gpuErrChk(cudaMemcpy(host_max, dev_max,
    sizeof(float), cudaMemcpyDeviceToHost));

  gpuErrChk(cudaMemcpy(host_max_index, dev_max_index,
    sizeof(int), cudaMemcpyDeviceToHost));

  /* Use the max index to look into the input array */
  gpuErrChk(cudaMemcpy(host_maximizer, input + *host_max_index, sizeof(int), 
    cudaMemcpyDeviceToHost));

  OptimizationOutput result;
  result.maximizer = *host_maximizer;
  result.maximum = *host_max;

  return result;
}

/* High level function which performs uniform random optimization */
OptimizationOutput optimize_UR(int N, float low, float high) {
  float *dev_input;

  gpuErrChk(cudaMalloc(&dev_input, N * sizeof(float)));

  cout << "Generating GPU inputs" << endl;

  /* Create random inputs */
  curandGenerator_t gen;
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  curandGenerateUniform(gen, dev_input, N);

  cout << "Generating GPU outputs" << endl;

  float *dev_output;
  gpuErrChk(cudaMalloc(&dev_output, N * sizeof(float)));

  /* Apply the objective function to the inputs */
  callApplyFunctionKernel(dev_input, dev_output, N, low, high);

  /* Find the maximum of the outputs. */
  return cudaCallMaximumKernel(dev_input, dev_output, N);
}

OptimizationOutput optimize(int N, float low, float high, OptimizationAlgorithm algorithm) {
  switch (algorithm) {
    case UNIFORM_RANDOM:
      return optimize_UR(N, low, high);
  }

  cerr << "Unrecognized algorithm." << endl;
  throw 20;
}














