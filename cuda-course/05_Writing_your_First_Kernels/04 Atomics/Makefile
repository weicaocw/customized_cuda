.PHONY: all clean
ARCH = sm_86

all: 00_atomicAdd

00_atomicAdd:
	nvcc -arch=$(ARCH) -o 00_atomicAdd_cu 00_atomicAdd.cu

clean:
	rm -f 00_atomicAdd *_cu