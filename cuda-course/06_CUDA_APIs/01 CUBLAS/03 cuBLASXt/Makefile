.PHONY: all clean

ARCH = sm_86
LIBS_CUBLAS = -lcublas

all: 01_demo 02_compare

01_demo: 
	nvcc -arch=$(ARCH) -o 01_demo_cu 01_demo.cu $(LIBS_CUBLAS)

02_compare: 
	nvcc -arch=$(ARCH) -o 02_compare_cu 02_compare.cu $(LIBS_CUBLAS)

clean:
	rm -f 01_demo *_cu