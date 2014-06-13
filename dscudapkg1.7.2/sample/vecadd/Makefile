DSCUDACPP        = ../../bin/dscudacpp
CUDAPATH        ?= /usr/local/cuda/
CUDASDKPATH     ?= $(CUDAPATH)/NVIDIA_GPU_Computing_SDK
# CUDAINCLUDES     = -I. -I$(CUDAPATH)/include -I$(CUDASDKPATH)/C/common/inc

TARGET = userapp

$(TARGET):	userapp.cu ../../lib/libdscuda.a
	$(DSCUDACPP) -o $@ -I. -i $< --cudart=shared -ldscuda -libverbs -lrdmacm -lpthread

#	$(DSCUDACPP) --cpp -o $@ -I. -i $< --cudart=shared -ldscuda -libverbs -lrdmacm -lpthread

$(TARGET)_tcp:	userapp.cu ../../lib/libdscuda_tcp.a
	$(DSCUDACPP) -o $@ -I. -i $< --cudart=shared -ldscuda_tcp

#	$(DSCUDACPP) --cpp -o $@ -I. -i $< --cudart=shared -ldscuda_tcp


$(TARGET)_tcponly:	userapp.cu ../../lib/libdscuda_tcp.a
	$(DSCUDACPP) -DTCP_ONLY=1 -o $@ -I. -i $< --cudart=shared -ldscuda_tcp

#	$(DSCUDACPP) --cpp -DTCP_ONLY=1 -o $@ -I. -i $< --cudart=shared -ldscuda_tcp

$(TARGET).o:	userapp.cu ../../lib/libdscuda.a
	$(DSCUDACPP) -c -o $@ -I. -i $< -ldscuda -libverbs -lrdmacm -lpthread


clean:
	rm -rf *.o *.ptx $(TARGET) $(TARGET)_tcp *.linkinfo ./dscudatmp
