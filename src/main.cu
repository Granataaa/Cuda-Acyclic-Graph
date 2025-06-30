#include <iostream>
#include <cuda_runtime.h>
#include "graph_utils.h"
#include "scan.h"

int main() {
    int n;
    int* adj = loadGraphFromFile("data/sample_graph10.txt", &n);
    check_acyclic(adj, n);
    delete[] adj;
    return 0;
}