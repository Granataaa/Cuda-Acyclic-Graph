// This file implements the prefix sum (scan) algorithm for compacting the list of active nodes based on their flags.

#include <cuda_runtime.h>
#include <stdio.h>
#include "graph_utils.h"
// Multi-block scan kernel using Blelloch scan and block-level partial sums
__global__ void scanKernelMultiBlock(const int *input, int *output, int *blockSums, int n) {
    extern __shared__ int temp[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x * 2 + tid;
    int offset = 1;
    int blockOffset = 2 * blockIdx.x * blockDim.x;

    // Load input into shared memory (zero padding if out of bounds)
    temp[2 * tid]     = (blockOffset + 2 * tid     < n) ? input[blockOffset + 2 * tid]     : 0;
    temp[2 * tid + 1] = (blockOffset + 2 * tid + 1 < n) ? input[blockOffset + 2 * tid + 1] : 0;

    // Up-sweep (reduce) phase
    for (int d = blockDim.x; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }

    // Save total sum of this block to blockSums
    if (tid == 0) {
        blockSums[blockIdx.x] = temp[2 * blockDim.x - 1];
        temp[2 * blockDim.x - 1] = 0;
    }

    // Down-sweep phase
    for (int d = 1; d < 2 * blockDim.x; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();

    // Write results to output
    if (blockOffset + 2 * tid     < n) output[blockOffset + 2 * tid]     = temp[2 * tid];
    if (blockOffset + 2 * tid + 1 < n) output[blockOffset + 2 * tid + 1] = temp[2 * tid + 1];
}

// Kernel to add scanned block sums to each block's output
__global__ void addScannedBlockSums(int *output, const int *blockSums, int n) {
    int blockOffset = 2 * blockIdx.x * blockDim.x;
    int tid = threadIdx.x;
    int addVal = blockSums[blockIdx.x];
    if (blockIdx.x == 0) return; // First block does not need to add anything

    if (blockOffset + 2 * tid     < n) output[blockOffset + 2 * tid]     += addVal;
    if (blockOffset + 2 * tid + 1 < n) output[blockOffset + 2 * tid + 1] += addVal;
}

__global__ void scanKernel(const int *input, int *output, int n) {
    extern __shared__ int temp[]; // Shared memory for the scan operation
    int localID = threadIdx.x;
    int globalID = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = 1;

    // Proteggi gli accessi!
    if (2 * globalID < n)
        temp[2 * localID] = input[2 * globalID];
    else
        temp[2 * localID] = 0;

    if (2 * globalID + 1 < n)
        temp[2 * localID + 1] = input[2 * globalID + 1];
    else
        temp[2 * localID + 1] = 0;
    
    // Up-sweep (reduce) phase
    for (int d = n >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (localID < d) {
            int ai = offset * (2 * localID + 1) - 1;
            int bi = offset * (2 * localID + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }

    // Clear the last element
    if (localID == 0) {
        temp[n - 1] = 0; // Set the last element to 0
    }

    // Down-sweep phase
    for (int d = 1; d < n; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (localID < d) {
            int ai = offset * (2 * localID + 1) - 1;
            int bi = offset * (2 * localID + 2) - 1;
            int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();

    // Write results to output
    if (2 * globalID < n)
        output[2 * globalID] = temp[2 * localID];
    if (2 * globalID + 1 < n)
        output[2 * globalID + 1] = temp[2 * localID + 1];
}

int scan(const int *d_input, int *d_output, int n) {
    int Npad = 1;
    while (Npad < n) Npad <<= 1;
    int lws = 256;
    int numBlocks = (Npad + lws - 1) / lws;
    int sharedMemSize = 2 * lws * sizeof(int);

    cudaEvent_t start_scan, stop_scan;
    check_cuda_error(cudaEventCreate(&start_scan), "cudaEventCreate start_scan");
    check_cuda_error(cudaEventCreate(&stop_scan), "cudaEventCreate stop_scan");

    cudaEventRecord(start_scan);
    scanKernel<<<numBlocks, lws, sharedMemSize>>>(d_input, d_output, Npad);
    check_cuda_error(cudaDeviceSynchronize(), "scanKernel execution");
    cudaEventRecord(stop_scan);
    check_cuda_error(cudaEventSynchronize(stop_scan), "cudaEventSynchronize stop_scan");
    float time_scan = 0.0f;
    check_cuda_error(cudaEventElapsedTime(&time_scan, start_scan, stop_scan), "cudaEventElapsedTime scan");
    printf("Scan time: %f ms, %g GB/s, %g GE/s\n", time_scan, (2 * n * sizeof(int)) / time_scan / 1e6, n / time_scan / 1e6);

    int *h_output = (int*)malloc(n * sizeof(int));
    cudaMemcpy(h_output, d_output, n * sizeof(int), cudaMemcpyDeviceToHost);
    printf("d_output: ");
    for (int i = 0; i < n; ++i) {
        printf("%d ", h_output[i]);
    }
    printf("\n");
    free(h_output);

    int last_scan = 0, last_flag = 0;
    check_cuda_error(cudaMemcpy(&last_scan, d_output + n - 1, sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy last_scan");
    check_cuda_error(cudaMemcpy(&last_flag, d_input + n - 1, sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy last_flag");
    int active_count = last_scan + last_flag;
    return active_count;
}

// Multi-block scan main (copy of the original main, using scanKernelMultiBlock)
int scan_multiblock(const int *d_input, int *d_output, int n) {
    int Npad = 1;
    while (Npad < n) Npad <<= 1;
    int lws = 256;
    int numBlocks = (Npad + 2 * lws - 1) / (2 * lws);
    int sharedMemSize = 2 * lws * sizeof(int);

    int *d_blockSums, *d_scannedBlockSums;
    cudaMalloc(&d_blockSums, numBlocks * sizeof(int));
    cudaMalloc(&d_scannedBlockSums, numBlocks * sizeof(int));

    cudaEvent_t start_scan, stop_scan;
    check_cuda_error(cudaEventCreate(&start_scan), "cudaEventCreate start_scan");
    check_cuda_error(cudaEventCreate(&stop_scan), "cudaEventCreate stop_scan");

    cudaEventRecord(start_scan);
    scanKernelMultiBlock<<<numBlocks, lws, sharedMemSize>>>(d_input, d_output, d_blockSums, Npad);
    check_cuda_error(cudaDeviceSynchronize(), "scanKernelMultiBlock execution");

    // Scan the block sums (single block, enough threads)
    if (numBlocks > 1) {
        scanKernel<<<1, lws, 2 * lws * sizeof(int)>>>(d_blockSums, d_scannedBlockSums, numBlocks);
        check_cuda_error(cudaDeviceSynchronize(), "scanKernel blockSums execution");
        addScannedBlockSums<<<numBlocks, lws>>>(d_output, d_scannedBlockSums, Npad);
        check_cuda_error(cudaDeviceSynchronize(), "addScannedBlockSums execution");
    }

    cudaEventRecord(stop_scan);
    check_cuda_error(cudaEventSynchronize(stop_scan), "cudaEventSynchronize stop_scan");
    float time_scan = 0.0f;
    check_cuda_error(cudaEventElapsedTime(&time_scan, start_scan, stop_scan), "cudaEventElapsedTime scan");
    printf("Multi-block scan time: %f ms, %g GB/s, %g GE/s\n", time_scan, (2 * n * sizeof(int)) / time_scan / 1e6, n / time_scan / 1e6);

    int *h_output = (int*)malloc(n * sizeof(int));
    cudaMemcpy(h_output, d_output, n * sizeof(int), cudaMemcpyDeviceToHost);
    printf("d_output (multi-block): ");
    for (int i = 0; i < n; ++i) {
        printf("%d ", h_output[i]);
    }
    printf("\n");
    free(h_output);

    int last_scan = 0, last_flag = 0;
    check_cuda_error(cudaMemcpy(&last_scan, d_output + n - 1, sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy last_scan");
    check_cuda_error(cudaMemcpy(&last_flag, d_input + n - 1, sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy last_flag");
    int active_count = last_scan + last_flag;

    cudaFree(d_blockSums);
    cudaFree(d_scannedBlockSums);

    return active_count;
}