// graph_utils.cu
#include "graph_utils.h"
#include "scan.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

// calcola il grado di ingresso per ogni nodo
__global__ void calculate_in_degree(int* adj, int* in_degree, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // global_id
    if (idx < n) {
        for (int j = 0; j < n; j++) {
            if (adj[j * n + idx] == 1) {
                atomicAdd(&in_degree[idx], 1);
            }
        }
    }
}

__global__ void flag_zero_in_degree(int* in_degree, int* flags, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        flags[idx] = (in_degree[idx] == 0) ? 1 : 0;
    }
}

__global__ void remove_active_nodes(int* adj, int* in_degree, int* active_nodes, int active_count, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < active_count) {
        int node = active_nodes[idx];
        in_degree[node] = -1; // Mark as removed
        for (int j = 0; j < n; j++) {
            if (adj[node * n + j] == 1) {
                atomicSub(&in_degree[j], 1);
            }
        }
    }
}

__global__ void compact_active_nodes(const int* flags, const int* scan, int* active_nodes, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && flags[idx]) {
        active_nodes[scan[idx]] = idx;
    }
}

void check_cuda_err(cudaError_t err, const char *msg)
{
	if (err != cudaSuccess) {
		fprintf(stderr, "%s failed: %d - %s\n", msg, err, cudaGetErrorString(err));
		exit(17);
	}
}

int* loadGraphFromFile(const char* filename, int *n) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Could not open file %s\n", filename);
        exit(EXIT_FAILURE);
    }

    fscanf(file, "%d", n);
    int *adj = (int *)malloc((*n) * (*n) * sizeof(int));
    for (int i = 0; i < (*n) * (*n); i++) {
        fscanf(file, "%d", &adj[i]);
    }

    fclose(file);
    return adj;
}

void check_acyclic(int* adj, int n) {
    int* d_adj;
    int* d_in_degree;
    int* d_flags;
    int* d_output;
    int* d_active_nodes;
    int* in_degree = new int[n]();
    int* flags = new int[n]();
    
    cudaMalloc(&d_adj, n * n * sizeof(int));
    cudaMalloc(&d_in_degree, n * sizeof(int));
    cudaMalloc(&d_flags, n * sizeof(int));
    cudaMalloc(&d_output, n * sizeof(int));
    cudaMalloc(&d_active_nodes, n * sizeof(int));

    cudaMemcpy(d_adj, adj, n * n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_in_degree, in_degree, n * sizeof(int), cudaMemcpyHostToDevice);
    
    // Calculate in-degrees
    calculate_in_degree<<<(n + 255) / 256, 256>>>(d_adj, d_in_degree, n); // <<<numBlocks, lws>>>
    cudaDeviceSynchronize();

    printf("Initial in-degrees:\n");
    cudaMemcpy(in_degree, d_in_degree, n * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++) {
        printf("%d ", in_degree[i]);
    }
    printf("\n");

    
    // Main loop for topological sort
    while (true) {
        flag_zero_in_degree<<<(n + 255) / 256, 256>>>(d_in_degree, d_flags, n);
        cudaDeviceSynchronize();

        cudaMemcpy(flags, d_flags, n * sizeof(int), cudaMemcpyDeviceToHost);
        printf("d_flags: ");
        for (int i = 0; i < n; i++) {
            printf("%d ", flags[i]);
        }
        printf("\n");
        
        // Scan operation to compact active nodes
        int active_count = scan(d_flags, d_output, n);

        // Compatta gli indici dei nodi attivi
        compact_active_nodes<<<(n + 255) / 256, 256>>>(d_flags, d_output, d_active_nodes, n);
        cudaDeviceSynchronize();
        
        printf("Active nodes count: %d\n", active_count);
        
        remove_active_nodes<<<(active_count + 255) / 256, 256>>>(d_adj, d_in_degree, d_active_nodes, active_count, n);
        cudaDeviceSynchronize();

        cudaMemcpy(in_degree, d_in_degree, n * sizeof(int), cudaMemcpyDeviceToHost);
        printf("Updated in-degrees:\n");
        for (int i = 0; i < n; i++) {
            printf("%d ", in_degree[i]);
        }
        printf("\n");

        if (active_count == 0) {
            break; // No active nodes left
        }
    }

    // Check for remaining nodes with in-degree >= 0
    cudaMemcpy(in_degree, d_in_degree, n * sizeof(int), cudaMemcpyDeviceToHost);
    bool is_acyclic = true;
    for (int i = 0; i < n; ++i) {
        if (in_degree[i] >= 0) {
            is_acyclic = false;
            break;
        }
    }

    // Clean up
    cudaFree(d_adj);
    cudaFree(d_in_degree);
    cudaFree(d_flags);
    cudaFree(d_output);
    cudaFree(d_active_nodes);
    delete[] in_degree;
    delete[] flags;

    if (is_acyclic) {
        printf("The graph is acyclic.\n");
    } else {
        printf("The graph is cyclic.\n");
    }
}