NVCC             = nvcc
 CPP              = g++
DSCUDACPP        = ../../bin/dscudacpp
CUDAPATH        ?= /usr/local/cuda/
CUDASDKPATH     ?= $(CUDAPATH)/NVIDIA_GPU_Computing_SDK
BASENAME         = userapp
TARGETS          = $(BASENAME) $(BASENAME)_tcp $(BASENAME)_tcponly $(BASENAME)_nvcc

all:	$(TARGETS)

$(BASENAME):	userapp.cu ../../lib/libdscuda.a
	$(DSCUDACPP) -g -c -o $@.o -I. -i $<
	$(CPP) -g -L$(DSCUDA_PATH)/lib -o $@ $@.o -ldscuda_ibv -lrdmacm -libverbs -lpthread -lcudart

$(BASENAME)_tcp:	userapp.cu ../../lib/libdscuda_tcp.a
	$(DSCUDACPP) -c -o $@.o -I. -i $<
	$(CPP) -L$(DSCUDA_PATH)/lib -o $@ $@.o -ldscuda_tcp -lcudart

$(BASENAME)_tcponly:	userapp.cu ../../lib/libdscuda_tcp.a
	$(DSCUDACPP) -DTCP_ONLY=1 -c -o $@.o -I. -i $<
	$(CPP) -L$(DSCUDA_PATH)/lib -o $@ $@.o -ldscuda_tcp -lcudart


$(BASENAME)_nvcc:	userapp.cu
	$(NVCC) -g  -o $@ $<    # CUDA 5.0 or earlier.

#	$(NVCC) -g --cudart=shared -o $@ $<  # CUDA 5.5 or later.

clean:
	rm -rf *.o *.ptx $(TARGETS) *.linkinfo ./dscudatmp
