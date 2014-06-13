DSCUDACPP        = ../../bin/dscudacpp
CUDAPATH        ?= /usr/local/cuda/
CUDASDKPATH     ?= $(CUDAPATH)/NVIDIA_GPU_Computing_SDK
# CUDAINCLUDES     = -I. -I$(CUDAPATH)/include -I$(CUDASDKPATH)/C/common/inc
LIBPATH          = ../../lib

TARGET = userapp

$(TARGET)_rpc:	userapp.cu $(LIBPATH)/libdscuda_rpc.so
	nvcc -g -o $@ -I. $<

$(TARGET):	userapp.cu $(LIBPATH)/libdscuda_ibv.so
	nvcc -g -o $@ -I. $< -libverbs -lrdmacm -lpthread

#	nvcc -g -o $@ -I. $< -L$(LIBPATH) -ldscuda_ibv -libverbs -lrdmacm -lpthread

$(TARGET)_rpconly:	userapp.cu ../../lib/libdscuda_rpc.a
	$(DSCUDACPP) -DRPC_ONLY=1 -o $@ -I. -i $< -ldscuda_rpc

$(TARGET).o:	userapp.cu ../../lib/libdscuda.a
	$(DSCUDACPP) -c -o $@ -I. -i $< -ldscuda -libverbs -lrdmacm -lpthread


clean:
	rm -rf *.o *.ptx $(TARGET) $(TARGET)_rpc *.linkinfo ./dscudatmp
