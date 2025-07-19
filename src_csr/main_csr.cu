#include <iostream>
#include <sstream>
#include <cstdlib> // Per std::atoi
#include "graph_utils_csr.h" // Include il nostro header con le utilità

int main(int argc, char* argv[]) {
    // Controlla il numero di argomenti della riga di comando
    if (argc < 2) {
        std::cerr << "Uso: " << argv[0] << " <lws>" << std::endl;
        std::cerr << "  <lws>: Dimensione del blocco (local work size) per i kernel CUDA." << std::endl;
        return 1; // Termina con un codice di errore
    }

    // Converti l'argomento lws da stringa a intero
    int lws = std::atoi(argv[1]);
    if (lws <= 0) {
        std::cerr << "Errore: <lws> deve essere un numero intero positivo." << std::endl;
        return 1;
    }

    // Loop per testare diversi grafi di esempio
    for (int i = 1; i <= 7; ++i) {
        std::stringstream ss;
        // Costruisce il percorso del file del grafo. Assicurati che i file siano in "../data_csr/"
        ss << "../data_csr/sample_graph_csr" << i << ".txt";
        std::string filename = ss.str();

        printf("\n--- Elaborazione del grafo: %s ---\n", filename.c_str());

        // Carica il grafo dal file
        CSRGraph* g = load_graph_from_file_csr(filename.c_str());
        if (!g) {
            std::cerr << "Errore: Impossibile caricare il grafo da " << filename << std::endl;
            continue; // Passa al prossimo file
        }

        // Esegui il controllo di aciclicità sul grafo caricato
        check_acyclic_csr(g, lws);

        // Libera la memoria del grafo dopo l'uso
        free_csr_graph(g);
    }

    return 0; // Termina con successo
}