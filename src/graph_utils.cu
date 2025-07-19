#include "graph_utils.h"
#include "scan.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <chrono> // Per misurare il tempo in modo più preciso su host

// Funzione di utilità per il controllo degli errori CUDA
void check_cuda_error(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s failed: %d - %s\n", msg, err, cudaGetErrorString(err));
        exit(EXIT_FAILURE); // Usa EXIT_FAILURE per indicare un errore
    }
}

// Carica il grafo da un file nel formato a matrice di adiacenza densa
int* load_graph_from_file(const char* filename, int *n) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error: Could not open file %s\n", filename);
        exit(EXIT_FAILURE);
    }
    
    fscanf(file, "%d", n); // Legge la dimensione del grafo
    int *adj = (int *)malloc((*n) * (*n) * sizeof(int));
    if (!adj) {
        fprintf(stderr, "Error: Failed to allocate memory for adjacency matrix.\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < (*n) * (*n); i++) {
        fscanf(file, "%d", &adj[i]);
    }

    char type_str[32]; // Buffer per leggere il tipo di grafo
    fscanf(file, "%s", type_str);
    printf("Loaded graph of size %d. Type: %s\n", *n, type_str);

    fclose(file);
    return adj;
}

// Kernel CUDA: Calcola il grado di ingresso per ogni nodo
// Questo kernel è funzionale ma accede a memoria non coalesced, potenzialmente più lento, ma in realtà il più veloce.
__global__ void calculate_in_degree(const int* adj, int* in_degree, int n, int Npad) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Global ID del nodo
    if (idx < n) {
        // Itera sulle righe per trovare gli archi che puntano a 'idx'
        for (int j = 0; j < n; j++) {
            if (adj[j * Npad + idx] == 1) { // Accesso alla colonna 'idx' per ogni riga 'j'
                in_degree[idx]++;
            }
        }
    }
}
// Versione che usa memoria condivisa
// Prima passata: ogni blocco calcola una somma parziale per ogni nodo
__global__ void calculate_in_degree_partial(const int* adj, int* partial, int n, int Npad) {
    int node = blockIdx.x; // ogni blocco gestisce un nodo
    int tid = threadIdx.x;
    int block = blockIdx.y;
    __shared__ int sum[256];

    int local_sum = 0;
    for (int j = tid + block * blockDim.x; j < n; j += gridDim.y * blockDim.x) {
        if (adj[j * Npad + node] == 1)
            local_sum++;
    }
    sum[tid] = local_sum;
    __syncthreads();

    // Riduzione in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sum[tid] += sum[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        partial[node * gridDim.y + block] = sum[0];
}
// Seconda passata: somma le parziali per ogni nodo
__global__ void calculate_in_degree_reduce_final(const int* partial, int* in_degree, int n, int num_blocks) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node < n) {
        int sum = 0;
        for (int b = 0; b < num_blocks; b++)
            sum += partial[node * num_blocks + b];
        in_degree[node] = sum;
    }
}

// Kernel CUDA: Flagga i nodi con grado di ingresso zero
__global__ void flag_zero_in_degree(const int* in_degree, int* flags, int Npad) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < Npad) {
        flags[idx] = (in_degree[idx] == 0) ? 1 : 0;
    }
}

// Kernel CUDA: Compatta i nodi attivi in un nuovo array
__global__ void compact_active_nodes(const int* flags, const int* scan, int* active_nodes, int Npad) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < Npad && flags[idx]) { // Se il nodo 'idx' è flagged (ha in-degree zero)
        active_nodes[scan[idx]] = idx; // Inseriscilo nella posizione indicata dal prefix sum
    }
}

// Kernel CUDA: Rimuove i nodi attivi (versione 2D con atomic)
// Versione con operazione atomica
__global__ void remove_active_nodes_2D(const int* adj, int* in_degree, int* active_nodes, int active_count, int n, int Npad) {
    int node_idx_in_active_list = blockIdx.x * blockDim.x + threadIdx.x; // Indice del nodo nella lista `active_nodes`
    int dest_node_col = blockIdx.y * blockDim.y + threadIdx.y;          // Colonna della matrice (nodo di destinazione)

    if (node_idx_in_active_list < active_count && dest_node_col < n) {
        int source_node = active_nodes[node_idx_in_active_list]; // Il nodo che sta per essere "rimosso"

        // Controlla se esiste un arco dal nodo sorgente al nodo di destinazione
        if (adj[source_node * Npad + dest_node_col] == 1) {
            // Decrementa atomicamente il grado di ingresso del nodo di destinazione
            atomicSub(&in_degree[dest_node_col], 1);
        }
        
        // Solo un thread per ogni `source_node` deve marcare il suo `in_degree` a -1
        // Per ogni `source_node` avrai `blockDim.y` thread con lo stesso `node_idx_in_active_list`.
        // Assicuriamoci che solo il primo thread (quello con `threadIdx.y == 0`) lo faccia.
        if (threadIdx.y == 0) {
            in_degree[source_node] = -1; // Marca il nodo come processato/rimosso
        }
    }
}
// Questo kernel è più efficiente riducendo gli accessi atomici globali utilizzando la shared memory
__global__ void remove_active_nodes_2D_sm(const int* adj, int* in_degree, int* active_nodes, int active_count, int n, int Npad) {
    // `local_delta_in_degree` è una matrice bidimensionale concettuale in shared memory
    // (blockDim.y righe x blockDim.x colonne)
    extern __shared__ int local_delta_in_degree[]; 

    int node_idx_in_active_list = blockIdx.x * blockDim.x + threadIdx.x; // Indice del nodo attivo (riga)
    int j_col_idx = blockIdx.y * blockDim.y + threadIdx.y;             // Indice della colonna (nodo di destinazione)

    int val = 0;
    if (node_idx_in_active_list < active_count && j_col_idx < n) {
        int source_node = active_nodes[node_idx_in_active_list];
        if (adj[source_node * Npad + j_col_idx] == 1) {
            val = 1; // Questo arco contribuisce a decrementare l'in-degree
        }
    }
    
    // Ogni thread contribuisce al suo elemento nella shared memory
    // Mappa threadIdx.x e threadIdx.y a un indice 1D in `local_delta_in_degree`
    local_delta_in_degree[threadIdx.y * blockDim.x + threadIdx.x] = val;
    __syncthreads(); // Assicura che tutti i valori siano scritti prima della riduzione

    // Riduzione per colonna: sommiamo i `val` per ogni `j_col_idx` all'interno del blocco
    // Solo un thread per ogni `j_col_idx` (es. threadIdx.x == 0) esegue questa somma
    if (threadIdx.x == 0 && j_col_idx < n) {
        int sum = 0;
        for (int i = 0; i < blockDim.x; i++) {
            sum += local_delta_in_degree[threadIdx.y * blockDim.x + i];
        }
        if (sum > 0) {
            atomicSub(&in_degree[j_col_idx], sum); // Aggiornamento atomico del grado di ingresso
        }
    }

    // Marcare il nodo sorgente come rimosso
    // Solo un thread per ogni `node_idx_in_active_list` (es. threadIdx.y == 0) lo fa
    if (node_idx_in_active_list < active_count && threadIdx.y == 0) {
        int node = active_nodes[node_idx_in_active_list];
        in_degree[node] = -1; // Marca il nodo come processato
    }
}

// Funzione host: rimuove i nodi attivi e aggiorna i gradi di ingresso (matrice densa)
void remove_active_nodes_cpu_dense(const int* h_adj, int* h_in_degree, const int* h_active_nodes, int active_count, int n, int Npad) {
    for (int i = 0; i < active_count; ++i) { // Per ogni nodo attivo (source_node)
        int source_node = h_active_nodes[i];
        
        // Marca il nodo come rimosso
        h_in_degree[source_node] = -1; 

        // Itera su tutti i possibili nodi di destinazione (j)
        for (int j = 0; j < n; ++j) {
            // Se esiste un arco da 'source_node' a 'j'
            if (h_adj[source_node * Npad + j] == 1) {
                // Decrementa l'in-degree del nodo di destinazione 'j'
                // Solo se 'j' non è già stato rimosso (-1)
                if (h_in_degree[j] != -1) {
                    h_in_degree[j]--;
                }
            }
        }
    }
}


// Funzione principale che esegue il controllo aciclico usando Kahn e CUDA (Dense/Hybrid)
void check_acyclic(int* adj, int n, int lws) {
    // Dichiarazioni dei puntatori a memoria device
    int* d_adj;
    int* d_in_degree;
    int* d_flags;
    int* d_output;
    int* d_active_nodes;
    
    // lws2 è una dimensione di blocco tipica per kernel 2D (es. 16x16)
    int lws2 = 16; 
    const int CPU_THRESHOLD = 10; // Soglia per decidere se usare CPU o GPU per `remove_active_nodes`

    // Calcola Npad: la dimensione del grafo padded alla successiva potenza di 2
    int Npad = 1;
    while (Npad < n) Npad <<= 1;
    
    // Allocazioni memoria Host per la logica CPU (se usata) e per il risultato finale
    int* h_in_degree_cpu = (int*)calloc(Npad, sizeof(int)); 
    int* h_active_nodes_cpu = (int*)calloc(Npad, sizeof(int));
    int* h_in_degree_final_result = (int*)malloc(Npad * sizeof(int)); // Per copiare il risultato finale dalla GPU

    // Prepara la matrice di adiacenza con padding su Host (usata per copia iniziale e per CPU)
    int* h_adj_padded = (int*)calloc(Npad * Npad, sizeof(int));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            h_adj_padded[i * Npad + j] = adj[i * n + j];
        }
    }

    // Allocazioni memoria Device
    check_cuda_error(cudaMalloc(&d_in_degree, Npad * sizeof(int)), "cudaMalloc d_in_degree");
    check_cuda_error(cudaMalloc(&d_flags, Npad * sizeof(int)), "cudaMalloc d_flags");
    check_cuda_error(cudaMalloc(&d_output, Npad * sizeof(int)), "cudaMalloc d_output"); // Per l'output della scan
    check_cuda_error(cudaMalloc(&d_active_nodes, Npad * sizeof(int)), "cudaMalloc d_active_nodes");
    check_cuda_error(cudaMalloc(&d_adj, Npad * Npad * sizeof(int)), "cudaMalloc d_adj");

    // Inizializza d_in_degree a zero sulla GPU
    check_cuda_error(cudaMemset(d_in_degree, 0, Npad * sizeof(int)), "cudaMemset d_in_degree");
    // Copia la matrice di adiacenza (padded) dalla Host alla Device
    check_cuda_error(cudaMemcpy(d_adj, h_adj_padded, Npad * Npad * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy d_adj");

    // Se si usasse una matrice trasposta
    // check_cuda_error(cudaMalloc(&d_adj_T, Npad * Npad * sizeof(int)), "cudaMalloc d_adj");
    // check_cuda_error(cudaMemcpy(d_adj_T, adj_T_host, Npad * Npad * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy d_adj");

    // Setup degli eventi CUDA per la misurazione del tempo
    cudaEvent_t start_calculate_in_degree, stop_calculate_in_degree;
    cudaEvent_t start_flag_zero_in_degree, stop_flag_zero_in_degree;
    cudaEvent_t start_compact_active_nodes, stop_compact_active_nodes;
    cudaEvent_t start_remove_active_nodes, stop_remove_active_nodes;
    cudaEvent_t start_total, stop_total;

    check_cuda_error(cudaEventCreate(&start_calculate_in_degree), "create start_calculate_in_degree event");
    check_cuda_error(cudaEventCreate(&stop_calculate_in_degree), "create stop_calculate_in_degree event");
    check_cuda_error(cudaEventCreate(&start_flag_zero_in_degree), "create start_flag_zero_in_degree event");
    check_cuda_error(cudaEventCreate(&stop_flag_zero_in_degree), "create stop_flag_zero_in_degree event");
    check_cuda_error(cudaEventCreate(&start_compact_active_nodes), "create start_compact_active_nodes event");
    check_cuda_error(cudaEventCreate(&stop_compact_active_nodes), "create stop_compact_active_nodes event");
    check_cuda_error(cudaEventCreate(&start_remove_active_nodes), "create start_remove_active_nodes event");
    check_cuda_error(cudaEventCreate(&stop_remove_active_nodes), "create stop_remove_active_nodes event");
    check_cuda_error(cudaEventCreate(&start_total), "create start_total event");
    check_cuda_error(cudaEventCreate(&stop_total), "create stop_total event");

    float ms_calculate_in_degree = 0.0f;
    float total_flag_zero_in_degree_ms = 0.0f;
    float total_compact_active_nodes_ms = 0.0f;
    float total_remove_active_nodes_ms = 0.0f;
    float total_scan_ms = 0.0f;
    int iterations = 0;

    // Inizia il timing totale
    cudaEventRecord(start_total);

    // Fase 1: Calcolo dei gradi di ingresso iniziali
    cudaEventRecord(start_calculate_in_degree);
    calculate_in_degree<<<(Npad + lws - 1) / lws, lws>>>(d_adj, d_in_degree, n, Npad);
    check_cuda_error(cudaDeviceSynchronize(), "calculate_in_degree kernel");
    // int num_blocks_y = (n + lws - 1) / lws; // scegli in base a n e risorse
    // dim3 grid_dim(n, num_blocks_y);
    // dim3 block_dim(lws);
    // int* d_partial;
    // check_cuda_error(cudaMalloc(&d_partial, n * num_blocks_y * sizeof(int)), "cudaMalloc d_partial");
    // calculate_in_degree_partial<<<grid_dim, block_dim>>>(d_adj, d_partial, n, Npad);
    // check_cuda_error(cudaDeviceSynchronize(), "calculate_in_degree_partial");
    // calculate_in_degree_reduce_final<<<(n + lws - 1) / lws, lws>>>(d_partial, d_in_degree, n, num_blocks_y);
    // check_cuda_error(cudaDeviceSynchronize(), "calculate_in_degree_reduce_final");
    cudaEventRecord(stop_calculate_in_degree);
    check_cuda_error(cudaEventSynchronize(stop_calculate_in_degree), "synchronize stop_calculate_in_degree");
    check_cuda_error(cudaEventElapsedTime(&ms_calculate_in_degree, start_calculate_in_degree, stop_calculate_in_degree), "elapsed time calculate_in_degree");
    
    printf("calculate_in_degree: %.3f ms, %.3f GB/s, %.3f GE/s\n", 
           ms_calculate_in_degree, 
           ((double)(n * n * sizeof(int)) + (n * sizeof(int))) / (ms_calculate_in_degree / 1000.0) / 1e9, 
           (double)(n * n) / (ms_calculate_in_degree / 1000.0) / 1e9);

    // Ciclo principale dell'algoritmo di Kahn
    while (true) {
        iterations++;

        // Fase 2: Flagga i nodi con grado di ingresso zero
        cudaEventRecord(start_flag_zero_in_degree);
        flag_zero_in_degree<<<(Npad + lws - 1) / lws, lws>>>(d_in_degree, d_flags, Npad);
        check_cuda_error(cudaDeviceSynchronize(), "flag_zero_in_degree kernel");
        cudaEventRecord(stop_flag_zero_in_degree);
        float ms_flag_zero_in_degree;
        check_cuda_error(cudaEventSynchronize(stop_flag_zero_in_degree), "synchronize stop_flag_zero_in_degree");
        check_cuda_error(cudaEventElapsedTime(&ms_flag_zero_in_degree, start_flag_zero_in_degree, stop_flag_zero_in_degree), "elapsed time flag_zero_in_degree");
        total_flag_zero_in_degree_ms += ms_flag_zero_in_degree;
        
        // Fase 3: Esegue la scan per contare e compattare i nodi attivi
        int active_count = scan_multiblock(d_flags, d_output, n, lws, &total_scan_ms);

        // Fase 4: Compattazione dei nodi attivi (copia nelle posizioni finali)
        cudaEventRecord(start_compact_active_nodes);
        compact_active_nodes<<<(Npad + lws - 1) / lws, lws>>>(d_flags, d_output, d_active_nodes, Npad);
        check_cuda_error(cudaDeviceSynchronize(), "compact_active_nodes kernel");
        cudaEventRecord(stop_compact_active_nodes);
        float ms_compact_active_nodes;
        check_cuda_error(cudaEventSynchronize(stop_compact_active_nodes), "synchronize stop_compact_active_nodes");
        check_cuda_error(cudaEventElapsedTime(&ms_compact_active_nodes, start_compact_active_nodes, stop_compact_active_nodes), "elapsed time compact_active_nodes");
        total_compact_active_nodes_ms += ms_compact_active_nodes;

        // Condizione di terminazione: nessun nodo con grado zero trovato
        if (active_count == 0) {
            break;
        }
        
        // Fase 5: Rimozione dei nodi attivi e aggiornamento dei gradi di ingresso
        

        // cudaEventRecord(start_remove_active_nodes);

        // if (active_count <= CPU_THRESHOLD) {
        //     // **NON copiare d_adj in adj_pad qui ad ogni iterazione!**
        //     // adj_pad è già stato popolato e contiene la matrice iniziale.
        //     // La matrice di adiacenza non cambia durante l'algoritmo di Kahn.
            
        //     check_cuda_error(cudaMemcpy(h_in_degree_cpu, d_in_degree, Npad * sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy d_in_degree to host for CPU processing");
        //     check_cuda_error(cudaMemcpy(h_active_nodes_cpu, d_active_nodes, active_count * sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy d_active_nodes to host for CPU processing");

        //     // Esegui la logica sulla CPU
        //     // Passa direttamente adj_pad, che è già sulla memoria host
        //     remove_active_nodes_cpu_dense(adj_pad, h_in_degree_cpu, h_active_nodes_cpu, active_count, n, Npad);

        //     // Copia i risultati aggiornati di in_degree dalla CPU alla GPU
        //     check_cuda_error(cudaMemcpy(d_in_degree, h_in_degree_cpu, Npad * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy h_in_degree_cpu to device after CPU processing");
            
        //     // free(adj_pad); // NON liberare adj_pad qui!
        // } else {
        //     // Usa il kernel CUDA esistente
        //     dim3 block(lws2, lws2);
        //     dim3 grid((active_count + lws2 - 1) / lws2, (n + lws2 - 1) / lws2);
        //     size_t sharedMemSize = lws2 * sizeof(int); 
        //     // remove_active_nodes_2D<<<grid, block>>>(d_adj, d_in_degree, d_active_nodes, active_count, n, Npad);      
        //     remove_active_nodes_2D_sm<<<grid, block, sharedMemSize>>>(d_adj, d_in_degree, d_active_nodes, active_count, n, Npad);
        //     check_cuda_error(cudaDeviceSynchronize(), "remove_active_nodes kernel");
        // }

        // cudaEventRecord(stop_remove_active_nodes);
        // float ms_remove_active_nodes;
        // check_cuda_error(cudaEventSynchronize(stop_remove_active_nodes), "synchronize stop_remove_active_nodes");
        // check_cuda_error(cudaEventElapsedTime(&ms_remove_active_nodes, start_remove_active_nodes, stop_remove_active_nodes), "elapsed time remove_active_nodes");
        // total_remove_active_nodes_ms += ms_remove_active_nodes;


        dim3 block(lws2, lws2);
        dim3 grid((active_count + lws2 - 1) / lws2, (n + lws2 - 1) / lws2);
        size_t sharedMemSize = lws2 * sizeof(int);
        cudaEventRecord(start_remove_active_nodes);
        // remove_active_nodes_2D<<<grid, block>>>(d_adj, d_in_degree, d_active_nodes, active_count, n, Npad);
        remove_active_nodes_2D_sm<<<grid, block, sharedMemSize>>>(d_adj, d_in_degree, d_active_nodes, active_count, n, Npad);
        check_cuda_error(cudaDeviceSynchronize(), "remove_active_nodes kernel");
        cudaEventRecord(stop_remove_active_nodes);
        float ms_remove_active_nodes;
        check_cuda_error(cudaEventSynchronize(stop_remove_active_nodes), "synchronize stop_remove_active_nodes");
        check_cuda_error(cudaEventElapsedTime(&ms_remove_active_nodes, start_remove_active_nodes, stop_remove_active_nodes), "elapsed time remove_active_nodes");
        total_remove_active_nodes_ms += ms_remove_active_nodes;
    }

    printf("Total iterations: %d\n", iterations);

    // Copia i gradi di ingresso finali dalla GPU alla Host per il controllo del ciclo
    check_cuda_error(cudaMemcpy(h_in_degree_final_result, d_in_degree, n * sizeof(int), cudaMemcpyDeviceToHost), "Final memcpy d_in_degree");
    
    // Controlla se il grafo è aciclico
    bool is_acyclic = true;
    for (int i = 0; i < n; ++i) {
        if (h_in_degree_final_result[i] >= 0) { // Se un nodo ha ancora in-degree >= 0, significa un ciclo
            is_acyclic = false;
            break;
        }
    }

    // Ferma il timing totale
    cudaEventRecord(stop_total);
    check_cuda_error(cudaEventSynchronize(stop_total), "synchronize stop_total");
    float ms_total_execution = 0.0f;
    check_cuda_error(cudaEventElapsedTime(&ms_total_execution, start_total, stop_total), "elapsed time total");

    printf("Total execution time: %.3f ms\n", ms_total_execution);

    if (iterations > 0) {
        printf("Avg flag_zero_in_degree: %.3f ms (BW: %.3f GB/s, Ops: %.3f GOps/s)\n",
            total_flag_zero_in_degree_ms / iterations,
            (2.0 * n * sizeof(int)) / (total_flag_zero_in_degree_ms / iterations / 1000.0) / 1e9,
            (double)n / (total_flag_zero_in_degree_ms / iterations / 1000.0) / 1e9);

        printf("Avg scan: %.3f ms (BW: %.3f GB/s, Ops: %.3f GOps/s)\n",
            total_scan_ms / iterations,
            (2.0 * n * sizeof(int)) / (total_scan_ms / iterations / 1000.0) / 1e9,
            (double)n / (total_scan_ms / iterations / 1000.0) / 1e9);

        printf("Avg compact_active_nodes: %.3f ms (BW: %.3f GB/s, Ops: %.3f GOps/s)\n",
            total_compact_active_nodes_ms / iterations,
            (2.0 * n * sizeof(int)) / (total_compact_active_nodes_ms / iterations / 1000.0) / 1e9,
            (double)n / (total_compact_active_nodes_ms / iterations / 1000.0) / 1e9);

        printf("Avg remove_active_nodes: %.3f ms\n",
            total_remove_active_nodes_ms / iterations);
    }

    // Cleanup delle risorse CUDA
    check_cuda_error(cudaFree(d_adj), "cudaFree d_adj");
    check_cuda_error(cudaFree(d_in_degree), "cudaFree d_in_degree");
    check_cuda_error(cudaFree(d_flags), "cudaFree d_flags");
    check_cuda_error(cudaFree(d_output), "cudaFree d_output");
    check_cuda_error(cudaFree(d_active_nodes), "cudaFree d_active_nodes");

    // Cleanup delle risorse Host
    free(h_in_degree_cpu);
    free(h_active_nodes_cpu);
    free(h_adj_padded);
    free(h_in_degree_final_result); // Libera il risultato finale


    if (is_acyclic) {
        printf("The graph is acyclic.\n\n");
    } else {
        printf("The graph is cyclic.\n\n");
    }

    // Distruzione degli eventi CUDA
    cudaEventDestroy(start_calculate_in_degree);
    cudaEventDestroy(stop_calculate_in_degree);
    cudaEventDestroy(start_flag_zero_in_degree);
    cudaEventDestroy(stop_flag_zero_in_degree);
    cudaEventDestroy(start_compact_active_nodes);
    cudaEventDestroy(stop_compact_active_nodes);
    cudaEventDestroy(start_remove_active_nodes);
    cudaEventDestroy(stop_remove_active_nodes);
    cudaEventDestroy(start_total);
    cudaEventDestroy(stop_total);
}