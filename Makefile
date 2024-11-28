NVCC = nvcc
CXX = g++
NVCC_FLAGS = -O3 -arch=sm_86
CXX_FLAGS = -O3 -std=c++17

CUDA_INCLUDE = /usr/local/cuda/include
CUDA_LIB = /usr/local/cuda/lib64

all: grnn_program

grnn_program: main.o grnn.o
	$(NVCC) $(NVCC_FLAGS) main.o grnn.o -o grnn_program

main.o: main.cpp grnn.h
	$(CXX) $(CXX_FLAGS) -I$(CUDA_INCLUDE) -c main.cpp

grnn.o: grnn.cu grnn.h
	$(NVCC) $(NVCC_FLAGS) -c grnn.cu

clean:
	rm -f *.o grnn_program
