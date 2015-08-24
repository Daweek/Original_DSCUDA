DSCUDACPP     ?= ../../bin/dscudacpp -g -arch sm_35 -rdc=true
CUDAPATH      ?= /usr/local/cuda
CUDASDKPATH   ?= /usr/local/cuda/NVIDIA_GPU_Computing_SDK
CUDAINCLUDES   = -I. -I$(CUDAPATH)/include -I$(CUDASDKPATH)/C/common/inc -I$(CUDASDKPATH)/shared/inc
CUDALIBS       = -L. -I$(CUDAPATH)/lib64 -L$(CUDASDKPATH)/C/lib -L$(CUDASDKPATH)/shared/lib

NVCC          ?= $(CUDAPATH)/bin/nvcc -use_fast_math -O0 -g


TARGET = reduceMultiGPU


$(TARGET):	$(TARGET).cu $(TARGET).cuh
	$(DSCUDACPP) -o $@ -I. -i $<

$(TARGET)_tcp:	$(TARGET).cu $(TARGET).cuh
	$(DSCUDACPP) -o $@ -I. -i $<

$(TARGET)_tcponly:	$(TARGET).cu $(TARGET).cuh
	$(DSCUDACPP) -DTCP_ONLY=1 -o $@ -I. -i $<

$(TARGET).o:	$(TARGET).cu $(TARGET).cuh
	$(DSCUDACPP) -c -o $@ -I. -i $<


$(TARGET)_local:	$(TARGET).cu $(TARGET).cuh
	$(NVCC) $(CUDAINCLUDES) $(CUDALIBS) -o $@ reduceMultiGPU.cu -lshrutil_x86_64 -lcutil_x86_64

clean:
	rm -rf *.o *.ptx *_ibv *_tcp *.svr.cu *.svr *.linkinfo ./dscudatmp
