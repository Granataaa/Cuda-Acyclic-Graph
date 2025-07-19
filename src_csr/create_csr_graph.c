#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NUM_GRAPHS 2
#define N_MIN 30000
#define N_MAX 30000

int rand_int(int min, int max) {
    return min + rand() % (max - min + 1);
}

void generate_cyclic(int **mat, int n) {
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
           mat[i][j] = rand() % 2;
}

void generate_acyclic(int **mat, int n) {
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            mat[i][j] = (j > i) ? rand() % 2 : 0;
}

void matrix_to_csr(int **mat, int n, int **row_ptr, int **col_idx, int *nnz) {
    *row_ptr = (int *)malloc((n + 1) * sizeof(int));
    int max_edges = n * n;
    *col_idx = (int *)malloc(max_edges * sizeof(int));
    int count = 0;
    (*row_ptr)[0] = 0;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (mat[i][j]) {
                (*col_idx)[count++] = j;
            }
        }
        (*row_ptr)[i + 1] = count;
    }
    *nnz = count;
}

int main() {
    srand((unsigned int)time(NULL));
    char filename[128];

    for (int graph_idx = 0 ; graph_idx < NUM_GRAPHS + 4 ; ++graph_idx) {
        int n = rand_int(N_MIN, N_MAX);
        int **mat = (int **)malloc(n * sizeof(int *));
        for (int i = 0; i < n; ++i)
            mat[i] = (int *)calloc(n, sizeof(int));

        int is_cyclic = (graph_idx < NUM_GRAPHS / 2);

        if (is_cyclic)
            generate_cyclic(mat, n);
        else
            generate_acyclic(mat, n);

        int *row_ptr, *col_idx, nnz;
        matrix_to_csr(mat, n, &row_ptr, &col_idx, &nnz);

        snprintf(filename, sizeof(filename), "../data_csr/sample_graph_csr%d.txt", graph_idx + 1);
        FILE *f = fopen(filename, "w");
        if (!f) {
            fprintf(stderr, "Errore apertura file %s\n", filename);
            exit(1);
        }

        fprintf(f, "%d %d\n", n, nnz); // n nodi, nnz archi
        for (int i = 0; i < n + 1; ++i)
            fprintf(f, "%d ", row_ptr[i]);
        fprintf(f, "\n");
        for (int i = 0; i < nnz; ++i)
            fprintf(f, "%d ", col_idx[i]);
        fprintf(f, "\n");
        fprintf(f, "%s\n", is_cyclic ? "ciclica" : "aciclica");
        fclose(f);

        free(row_ptr);
        free(col_idx);
        for (int i = 0; i < n; ++i)
            free(mat[i]);
        free(mat);
    }

    printf("Grafi CSR generati in ../data_csr/sample_graph_csrX.txt\n");
    return 0;
}