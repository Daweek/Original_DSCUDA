NVCC            ?= nvcc
DSCUDACPP        = ../../bin/dscudacpp
CUDAPATH        ?= /usr/local/cuda/
CUDASDKPATH     ?= $(CUDAPATH)/NVIDIA_GPU_Computing_SDK
CUDAINCLUDES     = -I. -I$(CUDAPATH)/include -I$(CUDASDKPATH)/C/common/inc

TARGET = memcpy memcpylocal

all:	$(TARGET)

memcpy:	memcpy.cu ../../lib/libdscuda.a
	$(DSCUDACPP) -o $@ -I. -i $< -ldscuda -lrdmacm -libverbs -lpthread

memcpy_omp:	memcpy.cu ../../lib/libdscuda.a
	$(DSCUDACPP) -o $@ -I. -i $< -ldscuda -lrdmacm -libverbs -lpthread  -Xcompiler -fopenmp

memcpy_rpc:	memcpy.cu ../../lib/libdscuda_rpc.a
	$(DSCUDACPP) -o $@ -I. -i $< -ldscuda_rpc


memcpylocal:	memcpy.cu
	$(NVCC) -o $@ $(CUDAINCLUDES) $<

memcpylocal_omp:	memcpy.cu
	$(NVCC) -o $@ $(CUDAINCLUDES) $< -Xcompiler -fopenmp


clean:
	rm -rf *.o *.ptx $(TARGET) *.linkinfo ./dscudatmp
