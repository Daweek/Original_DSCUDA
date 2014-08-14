NVCC            ?= nvcc
DSCUDACPP        = ../../bin/dscudacpp
NVCCOPT          = -arch sm_35 -rdc=true
CUDAPATH        ?= /usr/local/cuda/
CUDASDKPATH     ?= $(CUDAPATH)/samples
CUDAINCLUDES     = -I. -I$(CUDAPATH)/include -I$(CUDASDKPATH)/common/inc

TARGET = bandwidth

all:	$(TARGET)

bandwidth:	bandwidth.cu
	$(DSCUDACPP) $(NVCCOPT) $(CUDAINCLUDES) -g -o $@ -I. -i $<

bandwidth_omp:	bandwidth.cu
	$(DSCUDACPP) $(NVCCOPT) $(CUDAINCLUDES) -o $@ -I. -i $< -Xcompiler -fopenmp

bandwidthlocal:	bandwidth.cu
	$(NVCC) $(NVCCOPT) -o $@ $(CUDAINCLUDES) $<

bandwidthlocal_omp:	bandwidth.cu
	$(NVCC) $(NVCCOPT) -o $@ $(CUDAINCLUDES) $< -Xcompiler -fopenmp

broadcast:	broadcast.cu
	$(DSCUDACPP) $(NVCCOPT) -o $@ -I. -i $<

broadcast_omp:	broadcast.cu
	$(DSCUDACPP) $(NVCCOPT) -o $@ -I. -i $< -Xcompiler -fopenmp

clean:
	rm -rf *.o ./dscudatmp ./dscudasvr $(TARGET)_tcp $(TARGET)_ibv *.linkinfo
