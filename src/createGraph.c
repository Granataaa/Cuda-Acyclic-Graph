#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NUM_GRAPHS 4    // Numero totale di grafi da generare
#define N_MIN 20000     // Dimensione minima del grafo
#define N_MAX 20000     // Dimensione massima del grafo

// Funzione per generare un intero casuale tra min e max (inclusi)
int rand_int(int min, int max) {
    return min + rand() % (max - min + 1);
}

// Genera una matrice ciclica nxn (binaria casuale)
void generate_cyclic(int **mat, int n) {
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            mat[i][j] = rand() % 2;
}

// Genera una matrice aciclica nxn (binaria casuale nella parte triangolare superiore)
void generate_acyclic(int **mat, int n) {
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            mat[i][j] = (j > i) ? rand() % 2 : 0;
}

int main() {
    srand((unsigned int)time(NULL));
    char filename[128];

    for (int graph_idx = 3 ; graph_idx < NUM_GRAPHS ; ++graph_idx) {
        int n = rand_int(N_MIN, N_MAX);
        int **mat = (int **)malloc(n * sizeof(int *));
        for (int i = 0; i < n; ++i)
            mat[i] = (int *)malloc(n * sizeof(int));

        int is_cyclic = (graph_idx < NUM_GRAPHS / 2); // Prima metà ciclici, seconda metà aciclici

        if (is_cyclic)
            generate_cyclic(mat, n);
        else
            generate_acyclic(mat, n);

        snprintf(filename, sizeof(filename), "../data/sample_graph%d.txt", graph_idx + 1);
        FILE *f = fopen(filename, "w");
        if (!f) {
            fprintf(stderr, "Errore apertura file %s\n", filename);
            exit(1);
        }

        fprintf(f, "%d\n", n);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j)
                fprintf(f, "%d ", mat[i][j]);
            fprintf(f, "\n");
        }
        fprintf(f, "%s\n", is_cyclic ? "ciclica" : "aciclica");
        fclose(f);

        for (int i = 0; i < n; ++i)
            free(mat[i]);
        free(mat);
    }

    printf("Grafi generati in ../data/sample_graphX.txt\n");
    return 0;
}