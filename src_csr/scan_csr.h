#ifndef SCAN_H
#define SCAN_H

#include <cuda_runtime.h>

// Kernel per la scan (prefix sum) su singolo blocco
__global__ void scanKernel_csr(const int *input, int *output, int n);

// Kernel per la scan multi-blocco (fase 1: calcola somme parziali)
__global__ void scanKernelMultiBlock_csr(const int *input, int *output, int *blockSums, int n);

// Kernel per la scan multi-blocco (fase 2: aggiunge le somme dei blocchi scansionati)
__global__ void addScannedBlockSums_csr(int *output, const int *blockSums, int n);

// Funzione wrapper per l'esecuzione della scan multi-blocco
// Restituisce il conteggio degli elementi attivi e aggiorna il tempo di esecuzione.
int scan_multiblock_csr(const int *d_input, int *d_output, int n, int lws, float *time_ms);

#endif // SCAN_H