#include <cuda_runtime.h>

__global__ void scanKernel(const int *input, int *output, int n);
__global__ void addScannedBlockSums(int *output, const int *blockSums, int n);
__global__ void scanKernelMultiBlock(const int *input, int *output, int *blockSums, int n);
int scan(const int *d_input, int *d_output, int n);
int scan_multiblock(const int *d_input, int *d_output, int n);