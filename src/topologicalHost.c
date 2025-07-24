#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>

// Calcola il grado di ingresso per ogni nodo
void calculate_in_degree(const int* adj, int* in_degree, int n) {
    memset(in_degree, 0, n * sizeof(int));
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
            if (adj[i * n + j])
                in_degree[j]++;
}

// Topological sort host (ritorna true se aciclico)
bool check_acyclic_host(const int* adj, int n) {
    int* in_degree = (int*)malloc(n * sizeof(int));
    int* queue = (int*)malloc(n * sizeof(int));
    int front = 0, rear = 0, count = 0;

    calculate_in_degree(adj, in_degree, n);

    // Inserisci tutti i nodi con in_degree 0 nella coda
    for (int i = 0; i < n; ++i)
        if (in_degree[i] == 0)
            queue[rear++] = i;

    while (front < rear) {
        int u = queue[front++];
        count++;
        for (int v = 0; v < n; ++v) {
            if (adj[u * n + v]) {
                in_degree[v]--;
                if (in_degree[v] == 0)
                    queue[rear++] = v;
            }
        }
    }

    free(in_degree);
    free(queue);
    return count == n; // Aciclico se tutti i nodi sono stati rimossi
}

int main(int argc, char** argv) {
    for (int i = 0; i < 4; ++i) {
        printf("\n--- Test Iteration %d ---\n", i + 1);

        int n;
        char filename[50];
        snprintf(filename, sizeof(filename), "../data/sample_graph%d.txt", i + 1);
        FILE* f = fopen(filename, "r"); // Riapri il file ad ogni iterazione
        if (!f) {
            perror("File open");
            return 1; // Termina se il file non può essere aperto
        }

        fscanf(f, "%d", &n);
        int* adj = (int*)malloc(n * n * sizeof(int));
        if (!adj) { // Controllo allocazione memoria
            perror("Memory allocation failed");
            fclose(f);
            return 1;
        }

        for (int j = 0; j < n * n; ++j) {
            fscanf(f, "%d", &adj[j]);
        }

        char buffer[10];
        // Assicurati che il buffer sia abbastanza grande per il token letto
        if (fscanf(f, "%9s", buffer) == 1) { // Limita la lettura a 9 caratteri + null terminator
             printf("The graph definition in file is: %s\n", buffer);
        } else {
            printf("Could not read graph definition from file.\n");
        }


        fclose(f); // Chiudi il file alla fine di ogni iterazione

        clock_t start = clock();
        bool acyclic = check_acyclic_host(adj, n); // Chiama la tua funzione
        clock_t end = clock();
        double ms = 1000.0 * (end - start) / CLOCKS_PER_SEC;

        printf("Graph is %s\n", acyclic ? "acyclic" : "cyclic");
        printf("Host time: %.3f ms\n", ms);

        free(adj); // Libera la memoria allocata per il grafo ad ogni iterazione
    }

    return 0;
}