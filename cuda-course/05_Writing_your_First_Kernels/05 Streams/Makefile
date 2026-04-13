.PHONY: all clean
ARCH = sm_86

all: 01_stream_basics 02_stream_advanced

01_stream_basics:
	nvcc -arch=$(ARCH) -o 01_stream_basics_cu 01_stream_basics.cu

02_stream_advanced:
	nvcc -arch=$(ARCH) -o 02_stream_advanced_cu 02_stream_advanced.cu

clean:
	rm -f 01_stream_basics *_cu 02_stream_advanced *_cu