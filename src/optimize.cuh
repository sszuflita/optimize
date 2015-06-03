#ifndef OPTIMIZE_CUH
#define OPTIMIZE_CUH

__device__ __host__ float f(float x);

/*
NOTE: You can use this macro to easily check cuda error codes
and get more information.

Modified from:
http://stackoverflow.com/questions/14038589/
what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
*/
#define gpuErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code,
                      const char *file,
                      int line,
                      bool abort=true) {
  if (code != cudaSuccess) {
    fprintf(stderr,"GPUassert: %s %s %d\n",
            cudaGetErrorString(code), file, line);
    exit(code);
  }
}

#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { printf("Error at %s:%d\n",__FILE__,__LINE__); return EXIT_FAILURE;}} while(0)

enum OptimizationAlgorithm { UNIFORM_RANDOM };

typedef struct {
	float maximizer;
	float maximum;
} OptimizationOutput;

OptimizationOutput optimize(int N, float low, float high, OptimizationAlgorithm algorithm);


#endif
