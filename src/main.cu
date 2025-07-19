#include <iostream>
#include <sstream>
#include <cstdlib>
#include "graph_utils.h"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <lws>" << std::endl;
        return 1;
    }
    
    // lws (local work size) è la dimensione del blocco per i kernel 1D
    int lws = std::atoi(argv[1]);

    // Ciclo per testare con diversi file di grafo
    // Assumi che i file siano nominati sample_graph1.txt, sample_graph2.txt, etc.
    for (int i = 1; i <= 10; i++){
        int n; // Dimensione del grafo
        std::stringstream ss;
        ss << "../data/sample_graph" << i << ".txt"; // Percorso relativo al file del grafo
        std::string filename = ss.str();

        // Carica il grafo dalla memoria Host
        int* adj = load_graph_from_file(filename.c_str(), &n);
        
        // Esegui il controllo di aciclicità (ibrido GPU/CPU)
        std::cout << "--- Processing graph: " << filename << " ---\n";
        check_acyclic(adj, n, lws); 
        
        // Libera la memoria allocata per la matrice di adiacenza host
        free(adj); // Usa free() perché load_graph_from_file usa malloc()
    }

    return 0;
}