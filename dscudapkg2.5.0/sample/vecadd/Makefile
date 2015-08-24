NVCC             = nvcc -ccbin g++
NVCCOPT          = -arch sm_35 -rdc=true
# NVCCOPT          += -DTCP_ONLY=1
CPP              = g++
CUDAPATH        ?= /usr/local/cuda/
CUDALIBPATH      = $(CUDAPATH)/lib64
CUDASDKPATH      = $(CUDAPATH)/samples
DSCUDAPATH      ?= ../..
DSCUDACPP        = $(DSCUDAPATH)/bin/dscudacpp
DSCUDALIBPATH    = $(DSCUDAPATH)/lib
TARGETBASE       = userapp
TARGET           = $(TARGETBASE)_ibv

all:	$(TARGET)

$(TARGET):	main.o userapp.o
	$(DSCUDACPP) $(NVCCOPT) -o $(TARGETBASE) -i $^

%.o: %.cu
	$(DSCUDACPP) $(NVCCOPT) -c -I. -I$(CUDASDKPATH)/common/inc -o $@ -i $<

main.o:		main.cu userapp.cuh
userapp.o:	userapp.cu

clean:
	rm -rf *.o *.ptx *_ibv *_tcp *.svr.cu *.svr *.linkinfo ./dscudatmp
