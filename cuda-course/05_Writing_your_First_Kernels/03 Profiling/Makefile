.PHONY: 00_nvtx_matmul 01_naive_matmul 02_tiled_matmul all clean
ARCH = sm_86
LINK_FLAGS = -lnvToolsExt

00_nvtx_matmul:
	nvcc -arch=$(ARCH) -o 00_nvtx_matmul 00_nvtx_matmul.cu $(LINK_FLAGS)

01_naive_matmul:
	nvcc -arch=$(ARCH) -o 01_naive_matmul 01_naive_matmul.cu $(LINK_FLAGS)

02_tiled_matmul:
	nvcc -arch=$(ARCH) -o 02_tiled_matmul 02_tiled_matmul.cu $(LINK_FLAGS)

all: 00_nvtx_matmul 01_naive_matmul 02_tiled_matmul

clean:
	rm -f 00_nvtx_matmul 01_naive_matmul 02_tiled_matmul *_cu