#include "graph_utils_csr.h"
#include "scan_csr.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>

// Implementazione della funzione per caricare un grafo da file in formato CSR
CSRGraph* load_graph_from_file_csr(const char* filename) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Errore: Impossibile aprire il file %s\n", filename);
        exit(EXIT_FAILURE);
    }

    CSRGraph *g = (CSRGraph*)malloc(sizeof(CSRGraph));
    if (!g) {
        fprintf(stderr, "Errore: Impossibile allocare memoria per CSRGraph\n");
        fclose(file);
        exit(EXIT_FAILURE);
    }

    // Leggi il numero di nodi (n) e di archi (nnz)
    if (fscanf(file, "%d %d", &g->n, &g->nnz) != 2) {
        fprintf(stderr, "Errore: Formato file non valido per n e nnz\n");
        free(g);
        fclose(file);
        exit(EXIT_FAILURE);
    }

    // Alloca memoria per row_ptr e col_idx
    g->row_ptr = (int*)malloc((g->n + 1) * sizeof(int));
    g->col_idx = (int*)malloc(g->nnz * sizeof(int));
    if (!g->row_ptr || !g->col_idx) {
        fprintf(stderr, "Errore: Impossibile allocare memoria per row_ptr o col_idx\n");
        free(g->row_ptr);
        free(g->col_idx);
        free(g);
        fclose(file);
        exit(EXIT_FAILURE);
    }

    // Leggi i dati di row_ptr
    for (int i = 0; i < g->n + 1; ++i) {
        if (fscanf(file, "%d", &g->row_ptr[i]) != 1) {
            fprintf(stderr, "Errore: Formato file non valido per row_ptr\n");
            free_csr_graph(g);
            fclose(file);
            exit(EXIT_FAILURE);
        }
    }
    // Leggi i dati di col_idx
    for (int i = 0; i < g->nnz; ++i) {
        if (fscanf(file, "%d", &g->col_idx[i]) != 1) {
            fprintf(stderr, "Errore: Formato file non valido per col_idx\n");
            free_csr_graph(g);
            fclose(file);
            exit(EXIT_FAILURE);
        }
    }
    // Leggi il tipo di grafo (ciclica/aciclica)
    if (fscanf(file, "%31s", g->tipo) != 1) { // Limita la lettura a 31 caratteri per sicurezza
        fprintf(stderr, "Errore: Formato file non valido per tipo\n");
        free_csr_graph(g);
        fclose(file);
        exit(EXIT_FAILURE);
    }

    printf("Grafo CSR caricato: nodi=%d, archi=%d, tipo=%s\n", g->n, g->nnz, g->tipo);

    fclose(file);
    return g;
}

// Implementazione della funzione per liberare la memoria di un grafo CSR
void free_csr_graph(CSRGraph* g) {
    if (!g) return;
    free(g->row_ptr);
    free(g->col_idx);
    free(g);
}

// Implementazione della funzione per convertire CSR in CSC sulla CPU
CSCGraph* convert_csr_to_csc(const CSRGraph* csr_g) {
    if (!csr_g) return NULL;

    CSCGraph* csc_g = (CSCGraph*)malloc(sizeof(CSCGraph));
    if (!csc_g) {
        fprintf(stderr, "Errore: Impossibile allocare memoria per CSCGraph\n");
        exit(EXIT_FAILURE);
    }
    csc_g->n = csr_g->n;
    csc_g->nnz = csr_g->nnz;

    csc_g->col_ptr = (int*)calloc(csc_g->n + 1, sizeof(int)); // Inizializza a zero
    csc_g->row_idx = (int*)malloc(csc_g->nnz * sizeof(int));
    if (!csc_g->col_ptr || !csc_g->row_idx) {
        fprintf(stderr, "Errore: Impossibile allocare memoria per col_ptr o row_idx CSC\n");
        free_csc_graph(csc_g);
        exit(EXIT_FAILURE);
    }

    // Primo passaggio: contare gli elementi per ogni colonna per popolare col_ptr
    // Questo ci darà la dimensione di ogni colonna nel CSC
    for (int i = 0; i < csr_g->n; ++i) { // Iteriamo su ogni riga del CSR (nodo di partenza)
        for (int j = csr_g->row_ptr[i]; j < csr_g->row_ptr[i+1]; ++j) {
            int col_index = csr_g->col_idx[j]; // Nodo di destinazione dell'arco (i -> col_index)
            csc_g->col_ptr[col_index + 1]++;   // Incrementa il contatore per questa colonna
        }
    }

    // Trasformare col_ptr in un array di puntatori cumulativi (prefix sum)
    // col_ptr[k] indicherà dove inizia la colonna k in row_idx
    for (int i = 0; i < csc_g->n; ++i) {
        csc_g->col_ptr[i+1] += csc_g->col_ptr[i];
    }

    // Secondo passaggio: popolare row_idx del CSC
    // Usiamo un array temporaneo per tenere traccia della posizione corrente in ogni colonna
    std::vector<int> current_col_pos(csc_g->n, 0); // Inizializza a zero

    for (int i = 0; i < csr_g->n; ++i) { // Iteriamo sulle righe del CSR (nodi di partenza)
        for (int j = csr_g->row_ptr[i]; j < csr_g->row_ptr[i+1]; ++j) {
            int col_index = csr_g->col_idx[j]; // Nodo di destinazione
            // L'indice di riga nel CSC è la riga di partenza nel CSR
            csc_g->row_idx[csc_g->col_ptr[col_index] + current_col_pos[col_index]] = i;
            current_col_pos[col_index]++; // Avanza la posizione per la prossima voce in questa colonna
        }
    }

    return csc_g;
}

// Implementazione della funzione per liberare la memoria di un grafo CSC
void free_csc_graph(CSCGraph* g) {
    if (g) {
        free(g->col_ptr);
        free(g->row_idx);
        free(g);
    }
}   

// kernel CUDA

/**
 * @brief Kernel CUDA per calcolare l'in-degree di tutti i nodi di un grafo
 * rappresentato in formato CSC (Compressed Sparse Column).
 * Ogni thread gestisce un nodo e calcola il suo in-degree usando col_ptr.
 * @param d_col_ptr Puntatore CUDA all'array col_ptr del grafo CSC.
 * @param d_in_degree Puntatore CUDA all'array dove memorizzare gli in-degree.
 * @param n Il numero totale di nodi nel grafo.
 */
__global__ void calculate_in_degree_csc(const int* d_col_ptr, int* d_in_degree, int n) {
    int node_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (node_idx < n) {
        // L'in-degree per il nodo 'node_idx' è dato dalla dimensione della sua colonna nel CSC,
        // ovvero la differenza tra col_ptr[node_idx + 1] e col_ptr[node_idx].
        d_in_degree[node_idx] = d_col_ptr[node_idx + 1] - d_col_ptr[node_idx];
    }
}


/**
 * @brief Kernel CUDA per calcolare l'in-degree di tutti i nodi di un grafo
 * rappresentato in formato CSR usando operazioni atomiche.
 * Ogni thread processa gli archi di una riga e incrementa atomicamente
 * l'in-degree dei nodi di destinazione.
 * @param row_ptr Puntatore CUDA all'array row_ptr del grafo CSR.
 * @param col_idx Puntatore CUDA all'array col_idx del grafo CSR.
 * @param in_degree Puntatore CUDA all'array dove memorizzare gli in-degree.
 * @param n Il numero totale di nodi nel grafo.
 */
__global__ void calculate_in_degree_csr_atomic(const int* row_ptr, const int* col_idx, int* in_degree, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n) {
        for (int k = row_ptr[row]; k < row_ptr[row + 1]; ++k) {
            // Incrementa atomicamente l'in-degree del nodo di destinazione
            atomicAdd(&in_degree[col_idx[k]], 1);
        }
    }
}

/**
 * @brief Kernel CUDA per flaggare i nodi che hanno un in-degree pari a zero.
 * Questi nodi sono i candidati per essere rimossi in un DAG.
 * @param in_degree Puntatore CUDA all'array degli in-degree.
 * @param flags Puntatore CUDA all'array di flag (1 se in-degree è 0, 0 altrimenti).
 * @param n Il numero totale di nodi.
 */
__global__ void flag_zero_in_degree_csr(const int* in_degree, int* flags, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Se l'in-degree è 0, setta il flag a 1, altrimenti a 0.
        flags[idx] = (in_degree[idx] == 0) ? 1 : 0;
    }
}

/**
 * @brief Kernel CUDA per compattare i nodi attivi (quelli flaggati) in un array contiguo.
 * Utilizza i risultati di una scan per determinare la posizione finale di ciascun nodo attivo.
 * @param flags Puntatore CUDA all'array di flag (1 per nodi attivi, 0 altrimenti).
 * @param scan Puntatore CUDA all'array dei risultati della scan (prefix sum) sui flags.
 * @param active_nodes Puntatore CUDA all'array dove verranno memorizzati gli indici dei nodi attivi.
 * @param n Il numero totale di nodi.
 */
__global__ void compact_active_nodes_csr(const int* flags, const int* scan, int* active_nodes, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && flags[idx]) { // Se il nodo è flaggato (attivo)
        // La sua nuova posizione nell'array active_nodes è data dal risultato della scan su flags[idx].
        // Nota: scan[idx] dovrebbe essere l'indice corretto se scan è un exclusive scan.
        // Se è un inclusive scan, la posizione è scan[idx] - 1.
        // Assumendo uno "scan esclusivo" (prefix sum), la posizione è scan[idx].
        active_nodes[scan[idx]] = idx;
    }
}

/**
 * @brief Kernel CUDA per rimuovere i nodi attivi dal grafo e aggiornare gli in-degree
 * dei loro vicini. Utilizza operazioni atomiche per decrementare gli in-degree.
 * @param row_ptr Puntatore CUDA all'array row_ptr del grafo CSR.
 * @param col_idx Puntatore CUDA all'array col_idx del grafo CSR.
 * @param in_degree Puntatore CUDA all'array degli in-degree da aggiornare.
 * @param active_nodes Puntatore CUDA all'array degli indici dei nodi attivi da rimuovere.
 * @param active_count Il numero di nodi attivi.
 */
__global__ void remove_active_nodes_csr(const int* row_ptr, const int* col_idx, int* in_degree, const int* active_nodes, int active_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < active_count) {
        int node = active_nodes[idx]; // L'indice del nodo attivo da rimuovere

        // Marca il nodo come rimosso impostando il suo in-degree a -1.
        // Questo impedisce che venga processato nuovamente e indica che è stato "consumato".
        in_degree[node] = -1;

        // Processa gli archi uscenti dal nodo rimosso.
        // Questi archi non esisteranno più e quindi decrementeranno l'in-degree dei nodi di destinazione.
        int start = row_ptr[node];
        int end = row_ptr[node + 1];

        // Loop unroll o operazioni atomiche per ridurre la contesa atomica sui vicini.
        // Qui usiamo atomicSub per sicurezza.
        for (int k = start; k < end; ++k) {
            // Decrementa atomicamente l'in-degree del nodo di destinazione.
            // È importante che i nodi con in-degree a -1 non vengano ulteriormente decrementati.
            // L'atomicSub lavora solo su valori validi.
            atomicSub(&in_degree[col_idx[k]], 1);
        }
    }
}

/**
 * @brief Kernel CUDA per generare i decrementi per gli in-degree dei nodi vicini.
 * Ogni thread processa gli archi uscenti da un nodo attivo e incrementa
 * un contatore di decremento per il nodo di destinazione.
 * Questa è la prima fase di un approccio a due passaggi per ridurre le atomiche.
 * @param row_ptr Puntatore CUDA all'array row_ptr del grafo CSR.
 * @param col_idx Puntatore CUDA all'array col_idx del grafo CSR.
 * @param active_nodes Puntatore CUDA all'array degli indici dei nodi attivi da rimuovere.
 * @param active_count Il numero di nodi attivi.
 * @param d_decrement Puntatore CUDA all'array dove accumulare i decrementi per ogni nodo.
 * Deve essere inizializzato a zero prima di chiamare questo kernel.
 */
__global__ void remove_active_nodes_csr_reduce_global(
    const int* row_ptr, const int* col_idx, const int* active_nodes, int active_count,
    int* d_decrement)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < active_count) {
        int node = active_nodes[idx];
        // Ogni thread processa gli archi uscenti da un nodo attivo.
        for (int k = row_ptr[node]; k < row_ptr[node + 1]; ++k) {
            int dest = col_idx[k]; // Nodo di destinazione
            // Incrementa atomicamente il contatore di decrementi per il nodo di destinazione.
            // Questo kernel aggrega i decrementi per ogni nodo di destinazione.
            atomicAdd(&d_decrement[dest], 1);
        }
    }
}

/**
 * @brief Kernel CUDA per applicare i decrementi aggregati agli in-degree dei nodi.
 * Questa è la seconda fase dell'approccio a due passaggi.
 * Ogni thread applica il decremento calcolato a un in-degree specifico,
 * evitando contese atomiche in questa fase.
 * @param d_in_degree Puntatore CUDA all'array degli in-degree da aggiornare.
 * @param d_decrement Puntatore CUDA all'array dei decrementi aggregati.
 * @param n Il numero totale di nodi.
 */
__global__ void apply_decrement_to_in_degree(int* d_in_degree, const int* d_decrement, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Se c'è un decremento da applicare (d_decrement[idx] > 0)
        if (d_decrement[idx] > 0) {
            // Applica il decremento. Questa scrittura non richiede atomiche
            // perché ogni thread scrive su una posizione unica d_in_degree[idx].
            d_in_degree[idx] -= d_decrement[idx];
        }
    }
}

/**
 * @brief Kernel CUDA per marcare i nodi attivi come rimossi (impostando in-degree a -1).
 * Questo kernel è separato per evitare contese con le operazioni di decremento
 * e per garantire che ogni nodo rimosso sia marcato una sola volta.
 * @param d_in_degree Puntatore CUDA all'array degli in-degree.
 * @param d_active_nodes Puntatore CUDA all'array degli indici dei nodi attivi.
 * @param active_count Il numero di nodi attivi.
 */
__global__ void mark_removed_nodes(int* d_in_degree, const int* d_active_nodes, int active_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < active_count) {
        int node = d_active_nodes[idx];
        // Marca il nodo come rimosso. Questa scrittura non è atomica e va bene
        // perché ogni thread scrive su un nodo distinto.
        d_in_degree[node] = -1;
    }
}

/**
 * @brief Implementazione CPU della rimozione dei nodi attivi e dell'aggiornamento degli in-degree.
 * Questa funzione è usata come fallback o per test su piccoli set di dati.
 * @param g Il grafo CSR.
 * @param h_in_degree L'array degli in-degree sulla CPU.
 * @param h_active_nodes L'array degli indici dei nodi attivi sulla CPU.
 * @param active_count Il numero di nodi attivi da rimuovere.
 */
void remove_active_nodes_cpu(const CSRGraph* g, int* h_in_degree, const int* h_active_nodes, int active_count) {
    const int* h_row_ptr = g->row_ptr;
    const int* h_col_idx = g->col_idx;

    for (int i = 0; i < active_count; ++i) {
        int node = h_active_nodes[i];
        
        // Marca il nodo come rimosso.
        h_in_degree[node] = -1; 

        // Processa gli archi uscenti dal nodo attivo
        int start = h_row_ptr[node];
        int end = h_row_ptr[node + 1];

        for (int k = start; k < end; ++k) {
            int dest_node = h_col_idx[k];
            // Decrementa l'in-degree del nodo di destinazione
            // Assicurati di non decrementare un nodo già rimosso.
            if (h_in_degree[dest_node] != -1) {
                h_in_degree[dest_node]--;
            }
        }
    }
}


/**
 * @brief Controlla se un grafo è aciclico utilizzando l'algoritmo di rimozione topologica
 * con implementazione mista CPU/GPU.
 * @param g Il grafo CSR da controllare.
 * @param lws La dimensione del blocco (local work size) per i kernel CUDA.
 */
void check_acyclic_csr(const CSRGraph* g, int lws) {
    int n = g->n; // Numero di nodi
    
    // Dichiarazione puntatori per la memoria sul device (GPU)
    int *d_row_ptr, *d_col_idx, *d_in_degree, *d_flags, *d_scan, *d_active_nodes;
    int *d_csc_col_ptr; // Puntatore per la col_ptr del grafo CSC (usato solo all'inizio)
    int *d_decrement; // Usato nell'approccio a due passaggi per la rimozione

    // Dichiarazione puntatori per la memoria sull'host (CPU) per i risultati intermedi e fallback CPU
    int *h_in_degree_host = (int*)calloc(n, sizeof(int)); // Copia finale degli in-degree per controllo
    int *h_in_degree_cpu = (int*)malloc(n * sizeof(int)); // Per il fallback CPU
    int *h_active_nodes_cpu = (int*)malloc(n * sizeof(int)); // Per il fallback CPU (max n nodi attivi)

    // Variabili per il timing e statistiche
    float total_flag_zero_in_degree = 0.0f;
    float total_compact_active_nodes = 0.0f;
    float total_remove_active_nodes = 0.0f;
    float total_scan = 0.0f;
    int iterazioni = 0;
    long long total_active_nodes_processed = 0; // Totale nodi attivi processati in tutte le iterazioni

    // Soglia per decidere se usare CPU o GPU per la rimozione dei nodi
    // Questo valore può essere ottimizzato in base al sistema e al tipo di grafo.
    const int CPU_THRESHOLD = 10; 

    // Converti CSR a CSC sulla CPU una volta sola all'inizio
    // CSC è più efficiente per il calcolo iniziale dell'in-degree.
    CSCGraph* csc_h = convert_csr_to_csc(g);

    // -------------------------------------------------------------------------
    // Allocazione Memoria su Device (GPU)
    // -------------------------------------------------------------------------
    check_cuda_error(cudaMalloc(&d_row_ptr, (n + 1) * sizeof(int)), "cudaMalloc d_row_ptr");
    check_cuda_error(cudaMalloc(&d_col_idx, g->nnz * sizeof(int)), "cudaMalloc d_col_idx");
    check_cuda_error(cudaMalloc(&d_csc_col_ptr, (n + 1) * sizeof(int)), "cudaMalloc d_csc_col_ptr");
    check_cuda_error(cudaMalloc(&d_in_degree, n * sizeof(int)), "cudaMalloc d_in_degree");
    check_cuda_error(cudaMalloc(&d_flags, n * sizeof(int)), "cudaMalloc d_flags");
    check_cuda_error(cudaMalloc(&d_scan, n * sizeof(int)), "cudaMalloc d_scan");
    check_cuda_error(cudaMalloc(&d_active_nodes, n * sizeof(int)), "cudaMalloc d_active_nodes");
    check_cuda_error(cudaMalloc(&d_decrement, n * sizeof(int)), "cudaMalloc d_decrement"); // Per l'approccio a due passaggi

    // -------------------------------------------------------------------------
    // Copia Dati Host su Device
    // -------------------------------------------------------------------------
    check_cuda_error(cudaMemcpy(d_row_ptr, g->row_ptr, (n + 1) * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy d_row_ptr");
    check_cuda_error(cudaMemcpy(d_col_idx, g->col_idx, g->nnz * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy d_col_idx");
    check_cuda_error(cudaMemcpy(d_csc_col_ptr, csc_h->col_ptr, (n + 1) * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy d_csc_col_ptr");
    
    // Inizializza d_in_degree a zero (sarà popolato dal kernel)
    check_cuda_error(cudaMemset(d_in_degree, 0, n * sizeof(int)), "cudaMemset d_in_degree");

    // -------------------------------------------------------------------------
    // Eventi CUDA per il Timing delle Operazioni
    // -------------------------------------------------------------------------
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

    cudaEventRecord(start_total); // Inizia il timing totale

    // -------------------------------------------------------------------------
    // Fase 1: Calcolo In-Degree Iniziale (Usa CSC per efficienza)
    // -------------------------------------------------------------------------
    int blockSize = lws;
    int gridSize = (n + blockSize - 1) / blockSize;

    cudaEventRecord(start_calculate_in_degree);
    calculate_in_degree_csc<<<gridSize, blockSize>>>(d_csc_col_ptr, d_in_degree, n);
    check_cuda_error(cudaDeviceSynchronize(), "calculate_in_degree_csc kernel");
    cudaEventRecord(stop_calculate_in_degree);
    check_cuda_error(cudaEventSynchronize(stop_calculate_in_degree), "synchronize stop_calculate_in_degree");
    float ms_calculate_in_degree = 0.0f;
    check_cuda_error(cudaEventElapsedTime(&ms_calculate_in_degree, start_calculate_in_degree, stop_calculate_in_degree), "elapsed time calculate_in_degree");

    // cudaEventRecord(start_calculate_in_degree);
    // calculate_in_degree_csr_atomic<<<gridSize, blockSize>>>(d_row_ptr, d_col_idx, d_in_degree, n);
    // check_cuda_error(cudaDeviceSynchronize(), "calculate_in_degree_csc kernel");
    // cudaEventRecord(stop_calculate_in_degree);
    // check_cuda_error(cudaEventSynchronize(stop_calculate_in_degree), "synchronize stop_calculate_in_degree");
    // float ms_calculate_in_degree = 0.0f;
    // check_cuda_error(cudaEventElapsedTime(&ms_calculate_in_degree, start_calculate_in_degree, stop_calculate_in_degree), "elapsed time calculate_in_degree");

    // Dopo aver calcolato l'in-degree iniziale, la rappresentazione CSC non è più necessaria
    cudaFree(d_csc_col_ptr);
    free_csc_graph(csc_h); // Libera la memoria CSC sull'host

    // -------------------------------------------------------------------------
    // Fase 2: Iterazioni di Rimozione Topologica (Kahn's Algorithm)
    // -------------------------------------------------------------------------
    while (true) {
        iterazioni++;

        // 1. Flagga i nodi con in-degree zero
        cudaEventRecord(start_flag_zero_in_degree);
        flag_zero_in_degree_csr<<<gridSize, blockSize>>>(d_in_degree, d_flags, n);
        check_cuda_error(cudaDeviceSynchronize(), "flag_zero_in_degree_csr kernel");
        cudaEventRecord(stop_flag_zero_in_degree);
        check_cuda_error(cudaEventSynchronize(stop_flag_zero_in_degree), "synchronize stop_flag_zero_in_degree");
        float ms_flag_zero_in_degree = 0.0f;
        check_cuda_error(cudaEventElapsedTime(&ms_flag_zero_in_degree, start_flag_zero_in_degree, stop_flag_zero_in_degree), "elapsed time flag_zero_in_degree");
        total_flag_zero_in_degree += ms_flag_zero_in_degree;

        // 2. Esegui la scan sui flag per contare e preparare la compattazione
        // (Assumiamo che scan_multiblock_csr sia fornita e gestisca il suo timing interno)
        int active_count = scan_multiblock_csr(d_flags, d_scan, n, lws, &total_scan);

        // Se nessun nodo ha in-degree zero, il processo si ferma
        if (active_count == 0) break;
        total_active_nodes_processed += active_count;

        // 3. Compatta i nodi attivi in un array contiguo
        cudaEventRecord(start_compact_active_nodes);
        compact_active_nodes_csr<<<gridSize, blockSize>>>(d_flags, d_scan, d_active_nodes, n);
        check_cuda_error(cudaDeviceSynchronize(), "compact_active_nodes_csr kernel");
        cudaEventRecord(stop_compact_active_nodes);
        check_cuda_error(cudaEventSynchronize(stop_compact_active_nodes), "synchronize stop_compact_active_nodes");
        float ms_compact_active_nodes = 0.0f;
        check_cuda_error(cudaEventElapsedTime(&ms_compact_active_nodes, start_compact_active_nodes, stop_compact_active_nodes), "elapsed time compact_active_nodes");
        total_compact_active_nodes += ms_compact_active_nodes;

        // 4. Rimuovi i nodi attivi e aggiorna gli in-degree dei vicini (Strategia mista CPU/GPU)
        cudaEventRecord(start_remove_active_nodes); // Inizia il timing per la rimozione

        if (active_count <= CPU_THRESHOLD) {
            // Esegui sulla CPU per piccoli numeri di nodi attivi
            check_cuda_error(cudaMemcpy(h_in_degree_cpu, d_in_degree, n * sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy d_in_degree to host for CPU processing");
            check_cuda_error(cudaMemcpy(h_active_nodes_cpu, d_active_nodes, active_count * sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy d_active_nodes to host for CPU processing");
            
            remove_active_nodes_cpu(g, h_in_degree_cpu, h_active_nodes_cpu, active_count);

            check_cuda_error(cudaMemcpy(d_in_degree, h_in_degree_cpu, n * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy h_in_degree_cpu to device after CPU processing");
        } else {
            // Esegui sulla GPU con l'approccio a due passaggi per ridurre le atomiche
            // a) Resetta l'array di decremento per questa iterazione
            check_cuda_error(cudaMemset(d_decrement, 0, n * sizeof(int)), "cudaMemset d_decrement");

            // b) Genera i decrementi per tutti i nodi di destinazione (con atomiche)
            int remove_gridSize = (active_count + blockSize - 1) / blockSize;
            remove_active_nodes_csr_reduce_global<<<remove_gridSize, blockSize>>>(d_row_ptr, d_col_idx, d_active_nodes, active_count, d_decrement);
            check_cuda_error(cudaDeviceSynchronize(), "remove_active_nodes_csr_reduce_global kernel"); 

            // c) Applica i decrementi aggregati a d_in_degree (SENZA ATOMICHE)
            int apply_gridSize = (n + blockSize - 1) / blockSize; 
            apply_decrement_to_in_degree<<<apply_gridSize, blockSize>>>(d_in_degree, d_decrement, n);
            check_cuda_error(cudaDeviceSynchronize(), "apply_decrement_to_in_degree kernel"); 

            // d) Marca i nodi attivi come rimossi (SENZA ATOMICHE)
            mark_removed_nodes<<<remove_gridSize, blockSize>>>(d_in_degree, d_active_nodes, active_count);
            check_cuda_error(cudaDeviceSynchronize(), "mark_removed_nodes kernel"); 
        }
        
        cudaEventRecord(stop_remove_active_nodes); // Ferma il timing per la rimozione
        check_cuda_error(cudaEventSynchronize(stop_remove_active_nodes), "synchronize stop_remove_active_nodes");
        float ms_remove_active_nodes = 0.0f;
        check_cuda_error(cudaEventElapsedTime(&ms_remove_active_nodes, start_remove_active_nodes, stop_remove_active_nodes), "elapsed time remove_active_nodes");
        total_remove_active_nodes += ms_remove_active_nodes;


        // Rimozione nodi attivi
        // cudaEventRecord(start_remove_active_nodes);
        // remove_active_nodes_csr<<<(active_count + blockSize - 1) / blockSize, blockSize>>>(d_row_ptr, d_col_idx, d_in_degree, d_active_nodes, active_count);
        // check_cuda_error(cudaDeviceSynchronize(), "remove_active_nodes_csr_2 kernel");
        // cudaEventRecord(stop_remove_active_nodes);
        // check_cuda_error(cudaEventSynchronize(stop_remove_active_nodes), "synchronize stop_remove_active_nodes");
        // float ms_remove_active_nodes = 0.0f;
        // check_cuda_error(cudaEventElapsedTime(&ms_remove_active_nodes, start_remove_active_nodes, stop_remove_active_nodes), "elapsed time remove_active_nodes");
        // total_remove_active_nodes += ms_remove_active_nodes;


        // // --- INIZIO NUOVA SEQUENZA DI RIMOZIONE ---
        // cudaEventRecord(start_remove_active_nodes);

        // // a) Resetta l'array di decremento per la prossima iterazione
        // check_cuda_error(cudaMemset(d_decrement, 0, n * sizeof(int)), "cudaMemset d_decrement");

        // // b) Genera i decrementi per tutti i nodi di destinazione (con atomiche)
        // int remove_gridSize = (active_count + blockSize - 1) / blockSize;
        // remove_active_nodes_csr_reduce_global<<<remove_gridSize, blockSize>>>(d_row_ptr, d_col_idx, d_active_nodes, active_count, d_decrement);
        // check_cuda_error(cudaDeviceSynchronize(), "remove_active_nodes_csr_reduce_global kernel"); // Sincronizzazione necessaria prima di applicare

        // // c) Applica i decrementi aggregati a d_in_degree (SENZA ATOMICHE)
        // int apply_gridSize = (n + blockSize - 1) / blockSize; // O un numero di blocchi proporzionale ai soli nodi con decremento > 0
        // apply_decrement_to_in_degree<<<apply_gridSize, blockSize>>>(d_in_degree, d_decrement, n);
        // check_cuda_error(cudaDeviceSynchronize(), "apply_decrement_to_in_degree kernel"); // Sincronizzazione necessaria prima di marcare rimossi

        // // d) Marca i nodi attivi come rimossi (SENZA ATOMICHE)
        // mark_removed_nodes<<<remove_gridSize, blockSize>>>(d_in_degree, d_active_nodes, active_count);
        // check_cuda_error(cudaDeviceSynchronize(), "mark_removed_nodes kernel"); // Sincronizzazione necessaria per il timing

        // cudaEventRecord(stop_remove_active_nodes);
        // check_cuda_error(cudaEventSynchronize(stop_remove_active_nodes), "synchronize stop_remove_active_nodes");
        // float ms_remove_active_nodes = 0.0f;
        // check_cuda_error(cudaEventElapsedTime(&ms_remove_active_nodes, start_remove_active_nodes, stop_remove_active_nodes), "elapsed time remove_active_nodes");
        // total_remove_active_nodes += ms_remove_active_nodes;

    }

    // -------------------------------------------------------------------------
    // Controllo Finale e Risultati
    // -------------------------------------------------------------------------
    // Copia gli in-degree finali dal device all'host per il controllo
    check_cuda_error(cudaMemcpy(h_in_degree_host, d_in_degree, n * sizeof(int), cudaMemcpyDeviceToHost), "Final memcpy d_in_degree");
    
    // Un grafo è aciclico se e solo se tutti i nodi possono essere rimossi
    // (cioè, il loro in-degree finale è -1).
    bool is_acyclic = true;
    for (int i = 0; i < n; ++i) {
        if (h_in_degree_host[i] >= 0) { // Se un nodo ha ancora in-degree >= 0, significa che non è stato rimosso
            is_acyclic = false;
            break;
        }
    }

    cudaEventRecord(stop_total); // Ferma il timing totale
    check_cuda_error(cudaEventSynchronize(stop_total), "synchronize stop_total");
    float ms_total = 0.0f;
    check_cuda_error(cudaEventElapsedTime(&ms_total, start_total, stop_total), "elapsed time total");

    // -------------------------------------------------------------------------
    // Stampa Statistiche
    // -------------------------------------------------------------------------
    printf("\n--- Risultati Analisi Grafo ---\n");
    printf("Tempo totale CUDA: %.3f ms\n", ms_total);
    printf("Numero di iterazioni: %d\n", iterazioni);

    if (iterazioni > 0) {
        // calculate_in_degree_csr
        // printf("calculate_in_degree_csr: %g ms, %g GB/s, %g GE/s\n",
        //     ms_calculate_in_degree,
        //     ((g->nnz * sizeof(int)) + ((n + 1) * sizeof(int)) + (n * sizeof(int))) / (ms_calculate_in_degree / 1000.0) / 1e9,
        //     g->nnz / (ms_calculate_in_degree / 1000.0) / 1e9);

        // calculate_in_degree_csc
        printf("calculate_in_degree_csc: %g ms, %g GB/s, %g GE/s\n",
            ms_calculate_in_degree,
            (((n + 1) * sizeof(int)) + (n * sizeof(int))) / (ms_calculate_in_degree / 1000.0) / 1e9,
            n / (ms_calculate_in_degree / 1000.0) / 1e9);

        // flag_zero_in_degree_csr
        printf("flag_zero_in_degree (avg): %g ms, %g GB/s, %g GE/s\n",
            total_flag_zero_in_degree / iterazioni,
            (n * sizeof(int) + n * sizeof(int)) / ((total_flag_zero_in_degree / iterazioni) / 1000.0) / 1e9,
            n / ((total_flag_zero_in_degree / iterazioni) / 1000.0) / 1e9);

        // compact_active_nodes_csr
        printf("compact_active_nodes (avg): %g ms, %g GB/s, %g GE/s\n",
            total_compact_active_nodes / iterazioni,
            (3 * n * sizeof(int)) / ((total_compact_active_nodes / iterazioni) / 1000.0) / 1e9,
            n / ((total_compact_active_nodes / iterazioni) / 1000.0) / 1e9);

        // remove_active_nodes_csr (o CPU-based)
        printf("remove_active_nodes (avg): %g ms\n",
            total_remove_active_nodes / iterazioni);
        
        // scan
        printf("scan (avg): %g ms, %g GB/s, %g GE/s\n",
            total_scan / iterazioni,
            (2 * n * sizeof(int)) / ((total_scan / iterazioni) / 1000.0) / 1e9,
            n / ((total_scan / iterazioni) / 1000.0) / 1e9);
    }

    if (is_acyclic) {
        printf("The graph is acyclic.\n\n");
    } else {
        printf("The graph is cyclic.\n\n");
    }

    // -------------------------------------------------------------------------
    // Liberazione Memoria su Device e Host
    // -------------------------------------------------------------------------
    cudaFree(d_row_ptr);
    cudaFree(d_col_idx);
    cudaFree(d_in_degree);
    cudaFree(d_flags);
    cudaFree(d_scan);
    cudaFree(d_active_nodes);
    cudaFree(d_decrement); // Libera l'array di decremento

    free(h_in_degree_host);
    free(h_in_degree_cpu);
    free(h_active_nodes_cpu);

    // Distrugge gli eventi CUDA
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