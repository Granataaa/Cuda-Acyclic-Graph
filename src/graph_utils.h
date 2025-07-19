#ifndef GRAPH_UTILS_H
#define GRAPH_UTILS_H

#include <cuda_runtime.h>

// Dichiarazioni dei kernel CUDA
__global__ void calculate_in_degree(const int* adj, int* in_degree, int n, int Npad);
// Kernel per il calcolo del grado di ingresso con approccio multi-passata (se attivi)
__global__ void calculate_in_degree_partial(const int* adj, int* partial, int n, int Npad);
__global__ void calculate_in_degree_reduce_final(const int* partial, int* in_degree, int n, int num_blocks);

__global__ void flag_zero_in_degree(const int* in_degree, int* flags, int Npad);
__global__ void compact_active_nodes(const int* flags, const int* scan, int* active_nodes, int Npad);

// Kernel per la rimozione dei nodi attivi
__global__ void remove_active_nodes_2D(const int* adj, int* in_degree, int* active_nodes, int active_count, int n, int Npad);
__global__ void remove_active_nodes_2D_sm(const int* adj, int* in_degree, int* active_nodes, int active_count, int n, int Npad);

// Dichiarazione della funzione di gestione errori CUDA
void check_cuda_error(cudaError_t err, const char *msg);

// Dichiarazione della funzione per il caricamento del grafo da file (Host)
int* load_graph_from_file(const char* filename, int *n);

// Dichiarazione della funzione CPU per la rimozione dei nodi attivi (matrice densa)
void remove_active_nodes_cpu_dense(const int* h_adj, int* h_in_degree, const int* h_active_nodes, int active_count, int n, int Npad);

void check_acyclic(int* adj, int n, int lws);

#endif // GRAPH_UTILS_H