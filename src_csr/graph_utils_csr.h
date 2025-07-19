#ifndef GRAPH_UTILS_CSR_H
#define GRAPH_UTILS_CSR_H

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#ifdef __cplusplus
extern "C" {
#endif

// Struttura per rappresentare un grafo in formato Compressed Sparse Row (CSR)
typedef struct {
    int n;          // Numero di nodi nel grafo
    int nnz;        // Numero di archi (elementi non zero) nel grafo
    int *row_ptr;   // Puntatori di riga: row_ptr[i] indica l'inizio degli archi del nodo i in col_idx
                    // Dimensione: n + 1
    int *col_idx;   // Indici di colonna: memorizza i nodi di destinazione per ciascun arco
                    // Dimensione: nnz
    char tipo[32];  // Stringa per indicare il tipo di grafo (e.g., "ciclica", "aciclica")
} CSRGraph;

// Struttura per rappresentare un grafo in formato Compressed Sparse Column (CSC)
typedef struct {
    int n;          // Numero di nodi nel grafo
    int nnz;        // Numero di archi (elementi non zero) nel grafo
    int *col_ptr;   // Puntatori di colonna: col_ptr[j] indica l'inizio degli archi che arrivano
                    // al nodo j in row_idx. Utile per calcolare l'in-degree.
                    // Dimensione: n + 1
    int *row_idx;   // Indici di riga: memorizza i nodi di origine per ciascun arco
                    // Dimensione: nnz
} CSCGraph;


CSRGraph* load_graph_from_file_csr(const char* filename);

void free_csr_graph(CSRGraph* g);

CSCGraph* convert_csr_to_csc(const CSRGraph* csr_g);

void free_csc_graph(CSCGraph* g);

void check_acyclic_csr(const CSRGraph* g, int lws);

inline void check_cuda_error(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "%s failed: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE); // Usa EXIT_FAILURE per indicare un errore
    }
}

#ifdef __cplusplus
}
#endif

#endif // GRAPH_UTILS_CSR_H