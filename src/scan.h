#include <cuda_runtime.h>

__global__ void scanKernel(int *input, int *output, int n);
int scan(int *d_input, int *d_output, int n);