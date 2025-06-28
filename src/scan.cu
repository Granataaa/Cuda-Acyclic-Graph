// This file implements the prefix sum (scan) algorithm for compacting the list of active nodes based on their flags.

#include <cuda_runtime.h>
#include <stdio.h>

__global__ void scanKernel(int *input, int *output, int n) {
    extern __shared__ int temp[]; // Shared memory for the scan operation
    int localID = threadIdx.x;
    int offset = 1;

    // Load input into shared memory
    temp[2 * localID] = input[2 * localID];
    temp[2 * localID + 1] = input[2 * localID + 1];
    
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
    output[2 * localID] = temp[2 * localID];
    output[2 * localID + 1] = temp[2 * localID + 1];
}

int scan(int *d_input, int *d_output, int n) {
    int lws = 256;
    int numBlocks = (n + lws - 1) / lws;
    int sharedMemSize = 2 * lws * sizeof(int);

    scanKernel<<<numBlocks, lws, sharedMemSize>>>(d_input, d_output, n);
    cudaDeviceSynchronize();

    int *h_output = (int*)malloc(n * sizeof(int));
    cudaMemcpy(h_output, d_output, n * sizeof(int), cudaMemcpyDeviceToHost);
    printf("d_output: ");
    for (int i = 0; i < n; ++i) {
        printf("%d ", h_output[i]);
    }
    printf("\n");
    free(h_output);

    // Copia l'ultimo elemento dell'output su host per sapere quanti nodi attivi ci sono
    int last_scan = 0, last_flag = 0;
    cudaMemcpy(&last_scan, d_output + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&last_flag, d_input + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
    int active_count = last_scan + last_flag;
    return active_count;
}