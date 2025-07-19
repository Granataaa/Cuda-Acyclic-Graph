CC = nvcc
CFLAGS = -O2 -arch=sm_86
SRC = main.cu graph_utils.cu scan.cu
OBJ = $(SRC:.cu=.o)
TARGET = acyclic_graph

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(OBJ) -o $(TARGET)

%.o: %.cu
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	del /Q $(OBJ) $(TARGET) 2>nul