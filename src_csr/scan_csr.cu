#include "scan_csr.h"
#include "graph_utils_csr.h"
#include <cuda_runtime.h>
#include <stdio.h>

// Kernel CUDA: Scan (Prefix Sum) su singolo blocco Blelloch
// Questo kernel è progettato per operare su un singolo blocco di dati.
// La memoria condivisa `temp` deve essere allocata dinamicamente con `extern __shared__`.
__global__ void scanKernel_csr(const int *input, int *output, int n_elements) {
    extern __shared__ int temp[]; // Memoria condivisa per la scan
    int localID = threadIdx.x;
    int globalID = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = 1; // Usato nelle fasi di up-sweep e down-sweep

    // Carica gli elementi dal global memory alla shared memory
    // Protegge gli accessi per evitare out-of-bounds se n_elements non è potenza di 2 o multiplo di blockDim.x*2
    int val_left = (2 * globalID < n_elements) ? input[2 * globalID] : 0;
    int val_right = (2 * globalID + 1 < n_elements) ? input[2 * globalID + 1] : 0;

    temp[2 * localID] = val_left;
    temp[2 * localID + 1] = val_right;
    
    // Fase di Up-Sweep (Riduzione): Calcola le somme parziali
    for (int d = blockDim.x; d > 0; d >>= 1) { // Iterazioni basate sulla dimensione del blocco
        __syncthreads(); // Sincronizza i thread prima di ogni livello
        if (localID < d) {
            int ai = offset * (2 * localID + 1) - 1; // Indice sinistro
            int bi = offset * (2 * localID + 2) - 1; // Indice destro
            temp[bi] += temp[ai]; // Somma accumulata
        }
        offset *= 2; // Raddoppia l'offset per il livello successivo
    }

    // Salva l'ultimo elemento (somma totale del blocco) e lo azzera per la fase di Down-Sweep
    if (localID == 0) {
        temp[2 * blockDim.x - 1] = 0; // L'ultimo elemento diventa 0 per una scan esclusiva
    }

    // Fase di Down-Sweep: Calcola la scan esclusiva (prefix sum)
    for (int d = 1; d < 2 * blockDim.x; d *= 2) { // Iterazioni basate sulla dimensione della shared memory
        offset >>= 1; // Dimezza l'offset
        __syncthreads(); // Sincronizza i thread
        if (localID < d) {
            int ai = offset * (2 * localID + 1) - 1;
            int bi = offset * (2 * localID + 2) - 1;
            int t = temp[ai]; // Salva il valore sinistro
            temp[ai] = temp[bi]; // Assegna il valore destro al sinistro
            temp[bi] += t; // Somma il valore salvato al destro
        }
    }
    __syncthreads(); // Sincronizza prima di scrivere i risultati

    // Scrive i risultati della scan nell'array di output globale
    if (2 * globalID < n_elements) output[2 * globalID] = temp[2 * localID];
    if (2 * globalID + 1 < n_elements) output[2 * globalID + 1] = temp[2 * localID + 1];
}

// Kernel CUDA: Scan Multi-Block (Fase 1: Calcola somme parziali per ogni blocco)
// Ogni blocco esegue una scan interna e il thread 0 del blocco salva la somma totale.
__global__ void scanKernelMultiBlock_csr(const int *input, int *output, int *blockSums, int n_padded) {
    extern __shared__ int temp[]; 
    int tid = threadIdx.x;
    int offset = 1;
    int blockOffset = 2 * blockIdx.x * blockDim.x; // Indice di partenza globale per il blocco corrente

    // Carica i dati in memoria condivisa (con padding a zero se fuori dai limiti effettivi di n_padded)
    temp[2 * tid]     = (blockOffset + 2 * tid     < n_padded) ? input[blockOffset + 2 * tid]     : 0;
    temp[2 * tid + 1] = (blockOffset + 2 * tid + 1 < n_padded) ? input[blockOffset + 2 * tid + 1] : 0;

    // Up-sweep (riduzione) per calcolare la somma totale del blocco
    for (int d = blockDim.x; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }

    // Il thread 0 del blocco salva la somma totale del blocco
    if (tid == 0) {
        blockSums[blockIdx.x] = temp[2 * blockDim.x - 1]; // Salva la somma totale del blocco
        temp[2 * blockDim.x - 1] = 0; // Azzera l'ultimo elemento per la fase down-sweep
    }

    // Down-sweep per calcolare la scan esclusiva all'interno del blocco
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

    // Scrive i risultati della scan del blocco nell'array di output globale
    if (blockOffset + 2 * tid     < n_padded) output[blockOffset + 2 * tid]     = temp[2 * tid];
    if (blockOffset + 2 * tid + 1 < n_padded) output[blockOffset + 2 * tid + 1] = temp[2 * tid + 1];
}

// Kernel CUDA: Aggiunge le somme scansionate dei blocchi all'output
// Questo kernel è la seconda fase della scan multi-blocco.
__global__ void addScannedBlockSums_csr(int *output, const int *blockSums, int n_padded) {
    int blockOffset = 2 * blockIdx.x * blockDim.x;
    int tid = threadIdx.x;
    
    if (blockIdx.x == 0) return; // Il primo blocco non ha somme da aggiungere

    int addVal = blockSums[blockIdx.x]; // La somma cumulativa dei blocchi precedenti
    
    // Aggiunge la somma cumulativa a tutti gli elementi del blocco corrente
    if (blockOffset + 2 * tid     < n_padded) output[blockOffset + 2 * tid]     += addVal;
    if (blockOffset + 2 * tid + 1 < n_padded) output[blockOffset + 2 * tid + 1] += addVal;
}

// Funzione wrapper host per l'esecuzione della scan multi-blocco
// Prende `d_input` (flags), `d_output` (scan result), `n` (dimensione reale), `lws` (threads per blocco),
// e un puntatore a float per accumulare il tempo di esecuzione.
int scan_multiblock_csr(const int *d_input, int *d_output, int n, int lws, float *time_ms_accumulator) {
    // Calcola Npad (dimensione padded per la scan)
    int Npad = 1;
    while (Npad < n) Npad <<= 1;
    // La scan opera su `Npad` elementi. Ogni thread elabora 2 elementi, quindi 2*lws elementi per blocco.
    int numBlocks = (Npad + 2 * lws - 1) / (2 * lws);
    // Dimensione della shared memory per blocco (2 * lws elementi * sizeof(int))
    size_t sharedMemSize = 2 * lws * sizeof(int);

    // Allocazioni per le somme dei blocchi e le somme scansionate dei blocchi
    int *d_blockSums, *d_scannedBlockSums;
    check_cuda_error(cudaMalloc(&d_blockSums, numBlocks * sizeof(int)), "cudaMalloc d_blockSums");
    check_cuda_error(cudaMalloc(&d_scannedBlockSums, numBlocks * sizeof(int)), "cudaMalloc d_scannedBlockSums");

    // Eventi CUDA per misurare il tempo della scan
    cudaEvent_t start_scan, stop_scan;
    check_cuda_error(cudaEventCreate(&start_scan), "cudaEventCreate start_scan");
    check_cuda_error(cudaEventCreate(&stop_scan), "cudaEventCreate stop_scan");

    cudaEventRecord(start_scan);

    // Fase 1: Esegue la scanKernelMultiBlock. Ogni blocco calcola la sua scan interna
    // e la somma totale del blocco viene salvata in `d_blockSums`.
    scanKernelMultiBlock_csr<<<numBlocks, lws, sharedMemSize>>>(d_input, d_output, d_blockSums, Npad);
    check_cuda_error(cudaDeviceSynchronize(), "scanKernelMultiBlock execution");

    // Fase 2: Se ci sono più di un blocco, esegue una scan sui `d_blockSums`
    // per ottenere le somme cumulative dei blocchi.
    if (numBlocks > 1) {
        // La scan sui blockSums può essere eseguita da un singolo blocco GPU
        // se il numero di blockSums non è troppo grande, altrimenti necessita di un'altra scan multi-blocco.
        // Assumiamo che `numBlocks` sia gestibile da un singolo blocco per semplicità.
        check_cuda_error(cudaMalloc(&d_scannedBlockSums, numBlocks * sizeof(int)), "cudaMalloc d_scannedBlockSums");
        scanKernel_csr<<<1, (numBlocks + lws - 1) / lws * lws / 2, 2 * ((numBlocks + lws - 1) / lws * lws / 2) * sizeof(int)>>>(d_blockSums, d_scannedBlockSums, numBlocks); // Adatta lws
        check_cuda_error(cudaDeviceSynchronize(), "scanKernel blockSums execution");
        
        // Fase 3: Aggiunge le somme cumulative dei blocchi all'output della scan precedente
        addScannedBlockSums_csr<<<numBlocks, lws>>>(d_output, d_scannedBlockSums, Npad);
        check_cuda_error(cudaDeviceSynchronize(), "addScannedBlockSums execution");
    }

    cudaEventRecord(stop_scan);
    check_cuda_error(cudaEventSynchronize(stop_scan), "cudaEventSynchronize stop_scan");
    float time_scan_ms = 0.0f;
    check_cuda_error(cudaEventElapsedTime(&time_scan_ms, start_scan, stop_scan), "cudaEventElapsedTime scan");
    *time_ms_accumulator += time_scan_ms; // Accumula il tempo

    // Calcola il numero di nodi attivi dalla scan
    int last_scan_value = 0;
    int last_flag_value = 0;
    // Copia l'ultimo elemento di d_output (la somma totale della scan) e l'ultimo elemento di d_input
    // per ottenere il conteggio totale degli elementi "flagged"
    check_cuda_error(cudaMemcpy(&last_scan_value, d_output + n - 1, sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy last_scan_value");
    check_cuda_error(cudaMemcpy(&last_flag_value, d_input + n - 1, sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy last_flag_value");
    int active_count = last_scan_value + last_flag_value;

    // Cleanup delle risorse CUDA della scan
    cudaFree(d_blockSums);
    if (numBlocks > 1) {
        cudaFree(d_scannedBlockSums);
    }
    cudaEventDestroy(start_scan);
    cudaEventDestroy(stop_scan);

    return active_count;
}