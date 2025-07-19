CC = nvcc
CFLAGS = -O2 -arch=sm_86
SRC = main_csr.cu graph_utils_csr.cu scan_csr.cu
OBJ = $(SRC:.cu=.o)
TARGET = acyclic_graph_csr

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(OBJ) -o $(TARGET)

%.o: %.cu
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	del /Q $(OBJ) $(TARGET) 2>nul