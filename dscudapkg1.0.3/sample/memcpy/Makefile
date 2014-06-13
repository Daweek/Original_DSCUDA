NVCC            ?= nvcc
DSCUDACPP        = ../../bin/dscudacpp
CUDAPATH        ?= /usr/local/cuda/
CUDASDKPATH     ?= $(CUDAPATH)/NVIDIA_GPU_Computing_SDK
CUDAINCLUDES     = -I. -I$(CUDAPATH)/include -I$(CUDASDKPATH)/C/common/inc

TARGET = memcpy memcpylocal

all:	$(TARGET)

memcpy:	memcpy.cu ../../lib/libdscuda.a
	$(DSCUDACPP) -o $@ -I. -i $< -ldscuda -lrdmacm -libverbs -lpthread


memcpylocal:	memcpy.cu
	$(NVCC) -o $@ $(CUDAINCLUDES) $<


clean:
	rm -rf *.o *.ptx $(TARGET) *.linkinfo ./dscudatmp
