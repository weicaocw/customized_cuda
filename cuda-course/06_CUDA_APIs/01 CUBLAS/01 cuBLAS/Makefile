.PHONY: all clean
ARCH = sm_86
LIBS = -lcublas

all: 01_Hgemm_Sgemm

01_Hgemm_Sgemm:
	nvcc -arch=$(ARCH) -o 01_Hgemm_Sgemm_cu 01_Hgemm_Sgemm.cu $(LIBS)

clean:
	rm -f 01_Hgemm_Sgemm *_cu