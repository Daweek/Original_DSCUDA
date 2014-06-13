CPP              = g++
DSCUDACPP        = ../../bin/dscudacpp
CUDAPATH        ?= /usr/local/cuda/
CUDASDKPATH     ?= $(CUDAPATH)/NVIDIA_GPU_Computing_SDK

TARGET = userapp

$(TARGET):	userapp.cu ../../lib/libdscuda.a
	$(DSCUDACPP) -g -c -o $@.o -I. -i $<
	$(CPP) -g -L$(DSCUDA_PATH)/lib -o $@ $@.o -ldscuda_ibv -lrdmacm -libverbs -lpthread -lcudart

$(TARGET)_tcp:	userapp.cu ../../lib/libdscuda_tcp.a
	$(DSCUDACPP) -c -o $@.o -I. -i $<
	$(CPP) -L$(DSCUDA_PATH)/lib -o $@ $@.o -ldscuda_tcp -lcudart

$(TARGET)_tcponly:	userapp.cu ../../lib/libdscuda_tcp.a
	$(DSCUDACPP) -DTCP_ONLY=1 -c -o $@.o -I. -i $<
	$(CPP) -L$(DSCUDA_PATH)/lib -o $@ $@.o -ldscuda_tcp -lcudart

clean:
	rm -rf *.o *.ptx $(TARGET) $(TARGET)_tcp *.linkinfo ./dscudatmp
