DSCUDACPP     ?= ../../bin/dscudacpp -g
CUDAPATH      ?= /usr/local/cuda
CUDASDKPATH   ?= /usr/local/cuda/NVIDIA_GPU_Computing_SDK
CUDAINCLUDES   = -I. -I$(CUDAPATH)/include -I$(CUDASDKPATH)/C/common/inc -I$(CUDASDKPATH)/shared/inc
CUDALIBS       = -L. -I$(CUDAPATH)/lib64 -L$(CUDASDKPATH)/C/lib -L$(CUDASDKPATH)/shared/lib

NVCC          ?= $(CUDAPATH)/bin/nvcc -use_fast_math -O0 -g


TARGET = reduceMultiGPU


$(TARGET):	$(TARGET).cu $(TARGET).cuh ../../lib/libdscuda.a
	$(DSCUDACPP) -o $@ -I. -i $< -ldscuda -libverbs -lrdmacm -lpthread

$(TARGET)_rpc:	$(TARGET).cu $(TARGET).cuh ../../lib/libdscuda_rpc.a
	$(DSCUDACPP) -o $@ -I. -i $< -ldscuda_rpc

$(TARGET)_rpconly:	$(TARGET).cu $(TARGET).cuh ../../lib/libdscuda_rpc.a
	$(DSCUDACPP) -DRPC_ONLY=1 -o $@ -I. -i $< -ldscuda_rpc

$(TARGET).o:	$(TARGET).cu $(TARGET).cuh ../../lib/libdscuda.a
	$(DSCUDACPP) -c -o $@ -I. -i $< -ldscuda -libverbs -lrdmacm -lpthread


$(TARGET)_local:	$(TARGET).cu $(TARGET).cuh
	$(NVCC) $(CUDAINCLUDES) $(CUDALIBS) -o $@ reduceMultiGPU.cu -lshrutil_x86_64 -lcutil_x86_64

clean:
	rm -rf *.o *.ptx $(TARGET) $(TARGET)_local *.linkinfo *.snapshot ./dscudatmp
