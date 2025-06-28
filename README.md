# CUDA Acyclic Graph Checker

This project implements a CUDA-based algorithm to determine if a directed graph, represented as an adjacency matrix, is acyclic. The algorithm utilizes parallel processing capabilities of NVIDIA GPUs to efficiently compute in-degrees, perform topological sorting, and check for cycles in the graph.

## Project Structure

```
cuda-acyclic-graph
├── src
│   ├── main.cu          # Entry point for the CUDA application
│   ├── graph_utils.cu   # Utility functions for graph operations
│   ├── graph_utils.h    # Header file for graph utility functions
│   └── scan.cu          # Implementation of the prefix sum (scan) algorithm
├── data
│   └── sample_graph.txt  # Sample graph data in adjacency matrix format
├── Makefile              # Build instructions for the project
└── README.md             # Project documentation
```

## Overview of the Algorithm

1. **Input**: The algorithm takes an adjacency matrix `adj[n][n]` and the number of nodes `n`.
2. **In-Degree Calculation**: It computes the in-degree for each node in parallel.
3. **Topological Sort Loop**:
   - Flags nodes with in-degree zero.
   - Uses a prefix sum (scan) to compact the list of active nodes.
   - Extracts nodes with in-degree zero for processing.
   - Removes active nodes and updates the in-degrees of their adjacent nodes.
4. **Termination Check**: If all nodes are processed, the graph is acyclic; otherwise, a cycle is present.

## Building the Project

To build the project, navigate to the project directory and run the following command:

```
make
```

This will compile the CUDA source files and generate the executable.

## Running the Project

After building the project, you can run the executable with the following command:

```
./cuda-acyclic-graph <data/sample_graph.txt>
```

Make sure to replace `<data/sample_graph.txt>` with the path to your input graph file.

## Sample Input

The `data/sample_graph.txt` file contains the adjacency matrix representation of a graph. Here is an example format:

```
0 1 0 0
0 0 1 0
0 0 0 1
1 0 0 0
```

This represents a directed graph with 4 nodes.

## Conclusion

This project demonstrates the use of CUDA for parallel processing of graph algorithms, specifically for checking the acyclicity of directed graphs. The implementation leverages efficient memory management and parallel computation to handle potentially large graphs.