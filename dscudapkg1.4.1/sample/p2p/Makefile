DSCUDACPP        = ../../bin/dscudacpp
CUDAPATH        ?= /usr/local/cuda/
CUDASDKPATH     ?= $(CUDAPATH)/NVIDIA_GPU_Computing_SDK
# CUDAINCLUDES     = -I. -I$(CUDAPATH)/include -I$(CUDASDKPATH)/C/common/inc
NVCC             = $(CUDAPATH)/bin/nvcc

TARGET = p2p

$(TARGET):	p2p.cu ../../lib/libdscuda.a
	$(DSCUDACPP) -o $@ -I. -i $< -ldscuda -libverbs -lrdmacm -lpthread

$(TARGET)_rpc:	p2p.cu ../../lib/libdscuda_rpc.a
	$(DSCUDACPP) -o $@ -I. -i $< -ldscuda_rpc

$(TARGET)_rpconly:	p2p.cu ../../lib/libdscuda_rpc.a
	$(DSCUDACPP) -DRPC_ONLY=1 -o $@ -I. -i $< -ldscuda_rpc

$(TARGET)_local:	p2p.cu
	$(NVCC) -o $@ -I. -I$(CUDASDKPATH)/C/common/inc $<

clean:
	rm -rf *.o *.ptx $(TARGET) $(TARGET)_rpc *.linkinfo ./dscudatmp
