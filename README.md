# CUDA Cycle Detection in Directed Graphs using Kahn's Algorithm

This project implements a **CUDA-accelerated algorithm** to detect cycles in a directed graph using **Kahn's topological sort algorithm**. It supports two graph representations:
- Adjacency matrix
- Compressed Sparse Row (CSR) format
By leveraging NVIDIA GPUs, the algorithm achieves high parallel performance for in-degree computation, topological sorting, and cycle detection.

## Project Structure

```
cuda-acyclic-graph/
├── src/                  # Dense adjacency matrix implementation
│   ├── main.cu           # Main CUDA entry point
│   ├── graph_utils.cu    # Graph-related CUDA utilities
│   ├── graph_utils.h     # Header for graph utilities
│   ├── scan.cu           # Parallel prefix sum (scan) implementation
│   ├── scan.h            # Header for scan functions
│   ├── Makefile          # Build file for the dense version
│   ├── topologicalHost.c # Host-only (CPU) version of the algorithm
│   └── createGraph.c     # Script to generate acyclic/cyclic graphs
├── data/
│   └── sample_graph.txt  # Sample graph in adjacency matrix format
├── src_csr/              # CSR matrix implementation
│   ├── main_csr.cu       # Main CUDA entry point for CSR
│   ├── graph_utils_csr.cu
│   ├── graph_utils_csr.h
│   ├── scan_csr.cu
│   ├── scan_csr.h
│   ├── Makefile          # Build file for the CSR version
│   ├── kernel.cu         # Collection of kernel experiments
│   └── create_csr_graph.c# Script to generate CSR-based graphs
├── data_csr/
│   └── sample_graph_csr.txt  # Sample CSR-formatted graph
├── .gitignore            # Git ignore rules
└── README.md             # Project documentation

```

## Algorithm Overview

This implementation is based on **Kahn’s Algorithm** for topological sorting. The steps are:
1. **Input**: Graph data in adjacency matrix or CSR format.
2. **In-Degree Calculation**: Parallel computation of in-degree for each node.
3. **Topological Sorting Loop**:
   - Identify all nodes with in-degree 0 (parallelized).
   - Use a prefix sum (scan) to compact the list of active nodes.
   - Remove these nodes and decrement in-degrees of their neighbors.
   - Repeat until all nodes are processed or a cycle is detected.
4. **Cycle Detection**:
   - If all nodes are processed → Acyclic
   - If any nodes remain → Cycle detected

## Requirements & Build Configuration

To build and run this project, ensure you have the following installed:
- NVIDIA GPU with CUDA Compute Capability (e.g., 5.0+)
- CUDA Toolkit installed (version ≥ 11.x recommended)
  [Download CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
- A C++ compiler compatible with your CUDA version (e.g., g++, clang++)
- Make (usually pre-installed on Unix systems)

You can verify CUDA installation by running:

```
nvcc --version
```
If not installed, follow NVIDIA's [official installation guide](https://docs.nvidia.com/cuda/).

## Customizing the Makefile

Depending on your hardware or preferences, you may need to modify the `Makefile`. Below are some common recommendations:

1. **Set Target GPU Architecture**

Make sure the `Makefile` targets your GPU’s compute capability.
In both `src/Makefile` and `src_csr/Makefile`, you may see something like:

```
CFLAGS = -O2 -arch=sm_86
```

Check your GPU here: [CUDA GPU Capability List](https://developer.nvidia.com/cuda-gpus)

## Building the Project

1. Navigate to the desired version (either `src` for dense matrix or `src_csr` for CSR).
2. Run:

```
make
```

This will compile the CUDA source code and produce an executable.

## Running the Project

After building, run the executable as follows:

```
./acyclic_graph(_csr).exe <lws>
```

- `lws`: Local work size (or other parameter expected by your CUDA kernels). 

## Notes

- You can switch between dense and CSR implementations by compiling and running the appropriate version inside `src/` or `src_csr/`.
- Graphs can be generated manually via `createGraph.c` or `create_csr_graph.c`.