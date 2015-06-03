#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <sstream>

#include <cuda_runtime.h>

#include "optimize.cuh"

using namespace std;

// (From Eric's code)
    cudaEvent_t start;
    cudaEvent_t stop;

#define START_TIMER() {                         \
      gpuErrChk(cudaEventCreate(&start));       \
      gpuErrChk(cudaEventCreate(&stop));        \
      gpuErrChk(cudaEventRecord(start));        \
    }

#define STOP_RECORD_TIMER(name) {                           \
      gpuErrChk(cudaEventRecord(stop));                     \
      gpuErrChk(cudaEventSynchronize(stop));                \
      gpuErrChk(cudaEventElapsedTime(&name, start, stop));  \
      gpuErrChk(cudaEventDestroy(start));                   \
      gpuErrChk(cudaEventDestroy(stop));                    \
    }

////////////////////////////////////////////////////////////////////////////////
// Start non boilerplate code

float rand_float(float low, float high) {
  return low + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(high - low)));
}

int main(int argc, char** argv) {

  /* N function evaluations */

  int N = 1e8;

  float low = 0.;
  float high = 1000.;

  /* CPU STUFF */

  float cpu_time = -1.;

  START_TIMER();
{
  printf("Starting CPU stuff\n");

  srand(time(NULL));

  float maximizer = rand_float(low, high);
  float maximum = f(maximizer);

  for (int i = 0; i < N - 1; i++) {
    float input = rand_float(low, high);
    float output = f(input);

    if (output > maximum) {
      maximum = output;
      maximizer = input;
    }
  }

  printf("CPU max value:\t%f\n", maximum);
  printf("for input value:\t%f\n", maximizer);
}

  STOP_RECORD_TIMER(cpu_time);

  printf("\nCPU time:\t%f\n\n", cpu_time);


  float gpu_time = -1.;
  START_TIMER();
{

  // N, low, high, optimization type
  OptimizationOutput gpu_result = optimize(N, low, high, UNIFORM_RANDOM);

  printf("CPU max value:\t%f\n", gpu_result.maximum);
  printf("for input value:\t%f\n", gpu_result.maximizer);
}

  STOP_RECORD_TIMER(gpu_time);

  printf("\nGPU time:\t%f\n\n", gpu_time);
}