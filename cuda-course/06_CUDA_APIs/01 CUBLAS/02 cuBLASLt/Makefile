.PHONY: all clean
ARCH = sm_86
LIBS_CUBLASLt = -lcublasLt
LIBS_CUBLAS = -lcublas

all: 01_LtMatmul 02_compare

01_LtMatmul:
	nvcc -arch=$(ARCH) -o 01_LtMatmul_cu 01_LtMatmul.cu $(LIBS_CUBLASLt)

02_compare:
	nvcc -arch=$(ARCH) -o 02_compare_cu 02_compare.cu $(LIBS_CUBLASLt) $(LIBS_CUBLAS)

clean:
	rm -f 01_LtMatmul *_cu 02_compare *_cu