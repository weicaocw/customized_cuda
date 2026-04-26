.PHONY: all clean

ARCH = sm_86
LIBS_CUDNN = -lcudnn

all: 00_Tanh 01_Conv2d_NCHW 02_compare_conv

00_Tanh:
	nvcc -arch=$(ARCH) -o 00_Tanh_cu 00_Tanh.cu $(LIBS_CUDNN)

01_Conv2d_NCHW:
	nvcc -arch=$(ARCH) -o 01_Conv2d_NCHW_cu 01_Conv2d_NCHW.cu $(LIBS_CUDNN)

02_compare_conv:
	nvcc -arch=$(ARCH) -o 02_compare_conv_cu 02_compare_conv.cu $(LIBS_CUDNN)

clean:
	rm -f *_cu