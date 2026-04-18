.PHONY: all clean
ARCH = sm_86

all: vector_add_v1 vector_add_v2 matmul

vector_add_v1:
	nvcc -arch=$(ARCH) -o vector_add_v1 00_vector_add_v1.cu

vector_add_v2:
	nvcc -arch=$(ARCH) -o vector_add_v2 01_vector_add_v2.cu

matmul:
	nvcc -arch=$(ARCH) -o matmul 02_matmul.cu

clean:
	rm -f vector_add_v1 vector_add_v2 matmul *_cu