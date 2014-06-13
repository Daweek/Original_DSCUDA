NVCC            ?= nvcc
DSCUDACPP        = ../../bin/dscudacpp
CUDAPATH        ?= /usr/local/cuda/
CUDASDKPATH     ?= $(CUDAPATH)/NVIDIA_GPU_Computing_SDK
CUDAINCLUDES     = -I. -I$(CUDAPATH)/include -I$(CUDASDKPATH)/C/common/inc

TARGET = bandwidth bandwidthlocal bandwidth_omp bandwidth_rpc

all:	$(TARGET)

bandwidth:	bandwidth.cu ../../lib/libdscuda.a
	$(DSCUDACPP) -g -o $@ -I. -i $< -ldscuda -lrdmacm -libverbs -lpthread

bandwidth_omp:	bandwidth.cu ../../lib/libdscuda.a
	$(DSCUDACPP) -o $@ -I. -i $< -ldscuda -lrdmacm -libverbs -lpthread -Xcompiler -fopenmp

bandwidth_rpc:	bandwidth.cu ../../lib/libdscuda_rpc.a
	$(DSCUDACPP) -o $@ -I. -i $< -ldscuda_rpc

bandwidth_rpconly:	bandwidth.cu ../../lib/libdscuda_rpc.a
	$(DSCUDACPP) -DRPC_ONLY=1 -o $@ -I. -i $< -ldscuda_rpc


bandwidthlocal:	bandwidth.cu
	$(NVCC) -o $@ $(CUDAINCLUDES) $<

bandwidthlocal_omp:	bandwidth.cu
	$(NVCC) -o $@ $(CUDAINCLUDES) $< -Xcompiler -fopenmp

broadcast:	broadcast.cu ../../lib/libdscuda.a
	$(DSCUDACPP) -o $@ -I. -i $< -ldscuda -lrdmacm -libverbs -lpthread

broadcast_omp:	broadcast.cu ../../lib/libdscuda.a
	$(DSCUDACPP) -o $@ -I. -i $< -ldscuda -lrdmacm -libverbs -lpthread -Xcompiler -fopenmp


clean:
	rm -rf *.o *.ptx $(TARGET) *.linkinfo ./dscudatmp
