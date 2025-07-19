__global__ void calculate_in_degree_csc(const int* d_col_ptr, const int* d_row_idx, int* d_in_degree, int n) {
    int node_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (node_idx < n) {
        // Il grado di entrata per il nodo 'node_idx' è semplicemente la dimensione della sua colonna nel CSC
        // ovvero col_ptr[node_idx + 1] - col_ptr[node_idx]
        d_in_degree[node_idx] = d_col_ptr[node_idx + 1] - d_col_ptr[node_idx];
    }
}

__global__ void calculate_in_degree_csr(const int* row_ptr, const int* col_idx, int* in_degree, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n) {
        for (int k = row_ptr[row]; k < row_ptr[row + 1]; ++k) {
            atomicAdd(&in_degree[col_idx[k]], 1);
            //in_degree[col_idx[k]] += 1; // Non atomico, ma richiede attenzione per la contesa
        }
    }
}

__global__ void calculate_in_degree_no_atomic(const int* row_ptr, const int* col_idx, int* in_degree, int n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n) return;

    int count = 0;
    for (int row = 0; row < n; ++row) {
        for (int i = row_ptr[row]; i < row_ptr[row + 1]; ++i) {
            if (col_idx[i] == col) {
                count++;
            }
        }
    }
    in_degree[col] = count;
}

__global__ void calculate_in_degree_csr_chunked(const int* row_ptr, const int* col_idx, int* in_degree, int n) {
    int chunk_size = blockDim.x;
    int chunk_start = blockIdx.x * chunk_size;
    int chunk_end = min(chunk_start + chunk_size, n);

    for (int row = chunk_start + threadIdx.x; row < chunk_end; row += blockDim.x) {
        for (int k = row_ptr[row]; k < row_ptr[row + 1]; ++k) {
            atomicAdd(&in_degree[col_idx[k]], 1);
        }
    }
}

// Primo kernel: ogni blocco accumula localmente gli in-degree
__global__ void calculate_in_degree_csr_local(const int* row_ptr, const int* col_idx, int* local_in_degree, int n) {
    extern __shared__ int s_in_degree[];
    int tid = threadIdx.x;
    int row = blockIdx.x * blockDim.x + tid;
    int offset = blockIdx.x * n;

    // Inizializza shared memory
    for (int i = tid; i < n; i += blockDim.x)
        s_in_degree[i] = 0;
    __syncthreads();

    if (row < n) {
        for (int k = row_ptr[row]; k < row_ptr[row + 1]; ++k) {
            atomicAdd(&s_in_degree[col_idx[k]], 1);
        }
    }
    __syncthreads();

    // Scrivi i risultati nella memoria globale locale (per blocco)
    for (int i = tid; i < n; i += blockDim.x)
        local_in_degree[offset + i] = s_in_degree[i];
}

// Secondo kernel: somma i risultati di tutti i blocchi
__global__ void reduce_in_degree_blocks(const int* local_in_degree, int* in_degree, int n, int numBlocks) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int sum = 0;
        for (int b = 0; b < numBlocks; ++b)
            sum += local_in_degree[b * n + idx];
        in_degree[idx] = sum;
    }
}

__global__ void calculate_in_degree_partial_csr(const int* row_ptr, const int* col_idx,
                                                int* partial, int n, int num_partials) {
    int node = blockIdx.x;  // Nodo target v
    int tid = threadIdx.x;
    int part_id = blockIdx.y;  // Blocco parziale

    __shared__ int local_sum[256];  // Assumiamo max 256 thread

    int sum = 0;
    for (int row = tid + part_id * blockDim.x; row < n; row += gridDim.y * blockDim.x) {
        for (int j = row_ptr[row]; j < row_ptr[row + 1]; ++j) {
            if (col_idx[j] == node) {
                sum++;
            }
        }
    }

    local_sum[tid] = sum;
    __syncthreads();

    // Riduzione in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            local_sum[tid] += local_sum[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        partial[node * num_partials + part_id] = local_sum[0];
}

__global__ void calculate_in_degree_reduce_final(const int* partial, int* in_degree, int n, int num_blocks) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node < n) {
        int sum = 0;
        for (int b = 0; b < num_blocks; b++)
            sum += partial[node * num_blocks + b];
        in_degree[node] = sum;
    }
}


__global__ void flag_zero_in_degree_csr(const int* in_degree, int* flags, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        flags[idx] = (in_degree[idx] == 0) ? 1 : 0;
    }
}

__global__ void compact_active_nodes_csr(const int* flags, const int* scan, int* active_nodes, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && flags[idx]) {
        active_nodes[scan[idx]] = idx;
    }
}

__global__ void remove_active_nodes_csr(const int* row_ptr, const int* col_idx, int* in_degree, const int* active_nodes, int active_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < active_count) {
        int node = active_nodes[idx];
        in_degree[node] = -1; // Mark as removed
        for (int k = row_ptr[node]; k < row_ptr[node + 1]; ++k) {
            int dest = col_idx[k];
            atomicSub(&in_degree[dest], 1);
        }
    }
}

__global__ void remove_active_nodes_csr_2(const int* row_ptr, const int* col_idx, int* in_degree, const int* active_nodes, int active_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < active_count) {
        int node = active_nodes[idx];
        in_degree[node] = -1; // Mark as removed

        // Processa gli archi uscenti dal nodo attivo
        int start = row_ptr[node];
        int end = row_ptr[node + 1];

        // Loop unroll per ridurre la contesa atomica
        for (int k = start; k < end; ++k) {
            atomicSub(&in_degree[col_idx[k]], 1);
        }
    }
}

__global__ void remove_active_nodes_csr_chunked_reduce_direct(
    const int* row_ptr, const int* col_idx, const int* active_nodes, int active_count,
    int n, int chunk_start, int chunk_size, int* in_degree)
{
    extern __shared__ int local_decrement[];
    int tid = threadIdx.x;

    // Inizializza solo il chunk
    for (int i = tid; i < chunk_size; i += blockDim.x)
        local_decrement[i] = 0;
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + tid;
    if (idx < active_count) {
        int node = active_nodes[idx];
        for (int k = row_ptr[node]; k < row_ptr[node + 1]; ++k) {
            int dest = col_idx[k];
            if (dest >= chunk_start && dest < chunk_start + chunk_size)
                atomicAdd(&local_decrement[dest - chunk_start], 1);
        }
        // Marca come rimosso
        atomicExch(&in_degree[node], -1);
    }
    __syncthreads();

    // Applica decremento direttamente
    for (int i = tid; i < chunk_size; i += blockDim.x)
        atomicSub(&in_degree[chunk_start + i], local_decrement[i]);
}

__global__ void remove_active_nodes_csr_reduce_global(
    const int* row_ptr, const int* col_idx, const int* active_nodes, int active_count,
    int* d_decrement)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < active_count) {
        int node = active_nodes[idx];
        // Ogni thread processa un nodo attivo
        for (int k = row_ptr[node]; k < row_ptr[node + 1]; ++k) {
            int dest = col_idx[k];
            // atomicAdd aggiorna il contatore di decrementi per il nodo di destinazione
            atomicAdd(&d_decrement[dest], 1);
        }
    }
}

__global__ void apply_decrement_to_in_degree(int* d_in_degree, const int* d_decrement, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Se d_decrement[idx] è stato incrementato (cioè > 0), allora lo sottraiamo
        // Senza atomiche, perché ogni thread scrive su una locazione unica d_in_degree[idx].
        // Solo se d_decrement[idx] > 0 è necessario scrivere.
        if (d_decrement[idx] > 0) { // Ottimizzazione: evita scritture inutili
            d_in_degree[idx] -= d_decrement[idx];
        }
    }
}

__global__ void mark_removed_nodes(int* d_in_degree, const int* d_active_nodes, int active_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < active_count) {
        int node = d_active_nodes[idx];
        d_in_degree[node] = -1; // Scrittura non atomica, ogni thread scrive su un nodo unico
    }
}