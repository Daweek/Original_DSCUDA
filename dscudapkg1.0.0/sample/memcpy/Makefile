NVCC            ?= nvcc
RCUDACC          = ../../bin/rcudacc
CUDAPATH        ?= /usr/local/cuda/
CUDASDKPATH     ?= $(CUDAPATH)/NVIDIA_GPU_Computing_SDK
CUDAINCLUDES     = -I. -I$(CUDAPATH)/include -I$(CUDASDKPATH)/C/common/inc

TARGET = memcpy memcpylocal

all:	$(TARGET)

memcpy:	memcpy.cu ../../lib/librcuda.a
	$(RCUDACC) -o $@ -I. -i $< -lrcuda -lrdmacm -libverbs -lpthread


memcpylocal:	memcpy.cu
	$(NVCC) -o $@ $(CUDAINCLUDES) $<


clean:
	rm -rf *.o *.ptx $(TARGET) *.linkinfo ./rcudatmp
