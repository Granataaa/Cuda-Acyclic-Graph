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
    if (argc < 2) {
        printf("Usage: %s <graph_file.txt>\n", argv[0]);
        return 1;
    }
    int n;
    FILE* f = fopen(argv[1], "r");
    if (!f) { perror("File open"); return 1; }
    fscanf(f, "%d", &n);
    int* adj = (int*)malloc(n * n * sizeof(int));
    for (int i = 0; i < n * n; ++i)
        fscanf(f, "%d", &adj[i]);
    char buffer[10];
    fscanf(f, "%s", &buffer);
    printf("the graph is %s\n", buffer);
    fclose(f);

    clock_t start = clock();
    bool acyclic = check_acyclic_host(adj, n);
    clock_t end = clock();
    double ms = 1000.0 * (end - start) / CLOCKS_PER_SEC;
    double seconds = ms / 1000.0;
    double bytes = n * n * sizeof(int); // dati letti dalla matrice
    double elements = n * n;            // celle della matrice
    printf("Graph is %s\n", acyclic ? "acyclic" : "cyclic");
    printf("Host time: %.3f ms\n", ms);
    // printf("Host GB/s: %.6f\n", bytes / seconds / 1e9);
    // printf("Host GE/s: %.6f\n", elements / seconds / 1e9);

    free(adj);
    return 0;
}