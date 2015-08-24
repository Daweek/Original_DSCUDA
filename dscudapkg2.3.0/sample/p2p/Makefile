DSCUDACPP       ?= ../../bin/dscudacpp -arch sm_35 -rdc=true
CUDAPATH        ?= /usr/local/cuda/
CUDASDKPATH     ?= $(CUDAPATH)/NVIDIA_GPU_Computing_SDK
NVCC            ?= $(CUDAPATH)/bin/nvcc
TARGET           = p2p

$(TARGET):	p2p.cu ../../lib/libdscuda.a
	$(DSCUDACPP) -o $@ -I. -i $<

$(TARGET)_tcp:	p2p.cu ../../lib/libdscuda_tcp.a
	$(DSCUDACPP) -o $@ -I. -i $<

$(TARGET)_tcponly:	p2p.cu ../../lib/libdscuda_tcp.a
	$(DSCUDACPP) -DTCP_ONLY=1 -o $@ -I. -i $<

$(TARGET)_local:	p2p.cu
	$(NVCC) -o $@ -I. -I$(CUDASDKPATH)/C/common/inc $<

clean:
	rm -rf $(TARGET)_local *.o *.ptx *_ibv *_tcp *.svr.cu *.svr *.linkinfo ./dscudatmp
