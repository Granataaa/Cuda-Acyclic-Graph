// graph_utils.cu
#include "graph_utils.h"
#include "scan.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

// calcola il grado di ingresso per ogni nodo
__global__ void calculate_in_degree(const int* adj, int* in_degree, int n, int Npad) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // global_id
    if (idx < n) {
        for (int j = 0; j < n; j++) {
            if (adj[j * Npad + idx] == 1) {
                atomicAdd(&in_degree[idx], 1);
            }
        }
    }
}

__global__ void flag_zero_in_degree(const int* in_degree, int* flags, int Npad) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < Npad) {
        flags[idx] = (in_degree[idx] == 0) ? 1 : 0;
    }
}

__global__ void remove_active_nodes(const int* adj, int* in_degree, int* active_nodes, int active_count, int n, int Npad) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < active_count) {
        int node = active_nodes[idx];
        // printf("Rimuovo nodo %d con in_degree %d\n", node, in_degree[node]);
        in_degree[node] = -1; // Mark as removed
        for (int j = 0; j < n; j++) {
            // printf("%d -> %d: %d\n", node, j, adj[node * Npad + j]);
            if (adj[node * Npad + j] == 1) {
                // printf("Nodo %d decrementa in_degree[%d]\n", node, j);
                atomicSub(&in_degree[j], 1);
            }
        }
    }
}

__global__ void compact_active_nodes(const int* flags, const int* scan, int* active_nodes, int Npad) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < Npad && flags[idx]) {
        active_nodes[scan[idx]] = idx;
    }
}

void check_cuda_error(cudaError_t err, const char *msg)
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

    char tipo[32];
    fscanf(file, "%s", tipo);
    printf("Loaded graph with %d that is %s\n", *n, tipo);

    fclose(file);
    return adj;
}

void check_acyclic(int* adj, int n) {
    int* d_adj;
    int* d_in_degree;
    int* d_flags;
    int* d_output;
    int* d_active_nodes;
    int lws = 256; // Local work size
    // Trova la potenza di 2 successiva a n
    int Npad = 1;
    while (Npad < n) Npad <<= 1;
    int* in_degree = new int[Npad]();
    printf("Graph size: %d, padded size: %d\n", n, Npad);
    
    check_cuda_error(cudaMalloc(&d_in_degree, Npad * sizeof(int)), "cudaMalloc d_in_degree");
    check_cuda_error(cudaMalloc(&d_flags, Npad * sizeof(int)), "cudaMalloc d_flags");
    check_cuda_error(cudaMalloc(&d_output, Npad * sizeof(int)), "cudaMalloc d_output");
    check_cuda_error(cudaMalloc(&d_active_nodes, Npad * sizeof(int)), "cudaMalloc d_active_nodes");
    check_cuda_error(cudaMemcpy(d_in_degree, in_degree, Npad * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy d_in_degree");

    // Per la matrice di adiacenza (n*n -> Npad*Npad)
    int* adj_pad = (int*)calloc(Npad * Npad, sizeof(int));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            adj_pad[i * Npad + j] = adj[i * n + j];

    check_cuda_error(cudaMalloc(&d_adj, Npad * Npad * sizeof(int)), "cudaMalloc d_adj");
    check_cuda_error(cudaMemcpy(d_adj, adj_pad, Npad * Npad * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy d_adj");
    free(adj_pad);

    cudaEvent_t start_calculate_in_degree, stop_calculate_in_degree;
    cudaEvent_t start_flag_zero_in_degree, stop_flag_zero_in_degree;
    cudaEvent_t start_compact_active_nodes, stop_compact_active_nodes;
    cudaEvent_t start_remove_active_nodes, stop_remove_active_nodes;
    check_cuda_error(cudaEventCreate(&start_calculate_in_degree), "create start_calculate_in_degree event");
    check_cuda_error(cudaEventCreate(&stop_calculate_in_degree), "create stop_calculate_in_degree event");
    check_cuda_error(cudaEventCreate(&start_flag_zero_in_degree), "create start_flag_zero_in_degree event");
    check_cuda_error(cudaEventCreate(&stop_flag_zero_in_degree), "create stop_flag_zero_in_degree event");
    check_cuda_error(cudaEventCreate(&start_compact_active_nodes), "create start_compact_active_nodes event");
    check_cuda_error(cudaEventCreate(&stop_compact_active_nodes), "create stop_compact_active_nodes event");
    check_cuda_error(cudaEventCreate(&start_remove_active_nodes), "create start_remove_active_nodes event");
    check_cuda_error(cudaEventCreate(&stop_remove_active_nodes), "create stop_remove_active_nodes event");
    float ms_calculate_in_degree = 0.0f, ms_flag_zero_in_degree = 0.0f, ms_compact_active_nodes = 0.0f, ms_remove_active_nodes = 0.0f;
    
    // Calculate in-degrees
    cudaEventRecord(start_calculate_in_degree);
    calculate_in_degree<<<(Npad + (lws-1)) / lws, lws>>>(d_adj, d_in_degree, n,Npad); // <<<numBlocks, lws>>>
    check_cuda_error(cudaDeviceSynchronize(), "calculate_in_degree kernel");
    cudaEventRecord(stop_calculate_in_degree);
    check_cuda_error(cudaEventSynchronize(stop_calculate_in_degree), "synchronize stop_calculate_in_degree");
    check_cuda_error(cudaEventElapsedTime(&ms_calculate_in_degree, start_calculate_in_degree, stop_calculate_in_degree), "elapsed time calculate_in_degree");
    printf("calculate_in_degree: %g ms, %g GB/s, %g GE/s\n", ms_calculate_in_degree, ((n * n * sizeof(int)) + (n * sizeof(int))) / ms_calculate_in_degree / 1e6, (n * n) / ms_calculate_in_degree / 1e6);

    // printf("Initial in-degrees:\n");
    // cudaMemcpy(in_degree, d_in_degree, n * sizeof(int), cudaMemcpyDeviceToHost);
    // for (int i = 0; i < n; i++) {
    //     printf("%d ", in_degree[i]);
    // }
    // printf("\n");

    int i = 1; // Iteration counter
    // Main loop for topological sort
    while (true) {
        printf("Iteration: %d\n", i);
        i++;

        cudaEventRecord(start_flag_zero_in_degree);
        flag_zero_in_degree<<<(Npad + (lws-1)) / lws, lws>>>(d_in_degree, d_flags, Npad);
        check_cuda_error(cudaDeviceSynchronize(), "flag_zero_in_degree kernel");
        cudaEventRecord(stop_flag_zero_in_degree);
        check_cuda_error(cudaEventSynchronize(stop_flag_zero_in_degree), "synchronize stop_flag_zero_in_degree");
        check_cuda_error(cudaEventElapsedTime(&ms_flag_zero_in_degree, start_flag_zero_in_degree, stop_flag_zero_in_degree), "elapsed time flag_zero_in_degree");
        printf("flag_zero_in_degree: %g ms, %g GB/s, %g GE/s\n", ms_flag_zero_in_degree, (n * sizeof(int) + n * sizeof(int)) / ms_flag_zero_in_degree / 1e6, n / ms_flag_zero_in_degree / 1e6);

        // cudaMemcpy(flags, d_flags, n * sizeof(int), cudaMemcpyDeviceToHost);
        // printf("d_flags: ");
        // for (int i = 0; i < n; i++) {
        //     printf("%d ", flags[i]);
        // }
        // printf("\n");
        
        // Scan operation to compact active nodes
        int active_count = scan_multiblock(d_flags, d_output, n); // Using scan to count active nodes 

        // Compatta gli indici dei nodi attivi
        cudaEventRecord(start_compact_active_nodes);
        compact_active_nodes<<<(Npad + (lws-1)) / lws, lws>>>(d_flags, d_output, d_active_nodes, Npad);
        check_cuda_error(cudaDeviceSynchronize(), "compact_active_nodes kernel");
        cudaEventRecord(stop_compact_active_nodes);
        check_cuda_error(cudaEventSynchronize(stop_compact_active_nodes), "synchronize stop_compact_active_nodes");
        check_cuda_error(cudaEventElapsedTime(&ms_compact_active_nodes, start_compact_active_nodes, stop_compact_active_nodes), "elapsed time compact_active_nodes");
        printf("compact_active_nodes: %g ms, %g GB/s, %g GE/s\n", ms_compact_active_nodes, (n * sizeof(int) + n * sizeof(int)) / ms_compact_active_nodes / 1e6, n / ms_compact_active_nodes / 1e6);
        
        int* h_active_nodes = new int[active_count];
        cudaMemcpy(h_active_nodes, d_active_nodes, active_count * sizeof(int), cudaMemcpyDeviceToHost);
        printf("Active nodes: ");
        for (int i = 0; i < active_count; ++i) printf("%d ", h_active_nodes[i]);
        printf("\n");
        delete[] h_active_nodes;

        printf("Active nodes count: %d\n", active_count);      

        cudaEventRecord(start_remove_active_nodes);
        remove_active_nodes<<<(active_count + (lws-1)) / lws, lws>>>(d_adj, d_in_degree, d_active_nodes, active_count, n, Npad);
        check_cuda_error(cudaDeviceSynchronize(), "remove_active_nodes kernel");
        cudaEventRecord(stop_remove_active_nodes);
        check_cuda_error(cudaEventSynchronize(stop_remove_active_nodes), "synchronize stop_remove_active_nodes");
        check_cuda_error(cudaEventElapsedTime(&ms_remove_active_nodes, start_remove_active_nodes, stop_remove_active_nodes), "elapsed time remove_active_nodes");
        printf("remove_active_nodes: %g ms, %g GB/s, %g GE/s\n", ms_remove_active_nodes, ((n * n * sizeof(int)) + (n * sizeof(int))) / ms_remove_active_nodes / 1e6, (n * n) / ms_remove_active_nodes / 1e6);

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
    check_cuda_error(cudaMemcpy(in_degree, d_in_degree, n * sizeof(int), cudaMemcpyDeviceToHost), "Final memcpy d_in_degree");
    bool is_acyclic = true;
    for (int i = 0; i < n; ++i) {
        if (in_degree[i] >= 0) {
            is_acyclic = false;
            break;
        }
    }

    // Clean up
    check_cuda_error(cudaFree(d_adj), "cudaFree d_adj");
    check_cuda_error(cudaFree(d_in_degree), "cudaFree d_in_degree");
    check_cuda_error(cudaFree(d_flags), "cudaFree d_flags");
    check_cuda_error(cudaFree(d_output), "cudaFree d_output");
    check_cuda_error(cudaFree(d_active_nodes), "cudaFree d_active_nodes");
    delete[] in_degree;

    if (is_acyclic) {
        printf("The graph is acyclic.\n");
    } else {
        printf("The graph is cyclic.\n");
    }
}