#ifndef GRAPH_UTILS_H
#define GRAPH_UTILS_H

#include <cuda_runtime.h>

__global__ void calculate_in_degree(const int* adj, int* in_degree, int n);
__global__ void flag_zero_in_degree(const int* in_degree, int* flags, int n);
__global__ void remove_active_nodes(const int* adj, int* in_degree, const int* active_nodes, int active_count, int n);
void check_cuda_err(cudaError_t err, const char *msg);
int* loadGraphFromFile(const char* filename, int *n);
void check_acyclic(int* adj, int n);

#endif // GRAPH_UTILS_H