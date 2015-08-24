NVCC             = nvcc -ccbin g++
NVCCOPT          = -arch sm_35 -rdc=true
CPP              = g++
CUDAPATH        ?= /usr/local/cuda/
CUDALIBPATH      = $(CUDAPATH)/lib64
CUDASDKPATH      = $(CUDAPATH)/samples
DSCUDAPATH      ?= ../..
DSCUDACPP        = $(DSCUDAPATH)/bin/dscudacpp
DSCUDALIBPATH    = $(DSCUDAPATH)/lib
TARGET          = userapp

all:	$(TARGET)

$(TARGET):	main.o
	$(DSCUDACPP) $(NVCCOPT) -o $@ -i $^

%.o: %.cu
	$(DSCUDACPP) $(NVCCOPT) -c -I. -I$(CUDASDKPATH)/common/inc -o $@ -i $<

clean:
	rm -rf *.o *.ptx *_ibv *_tcp *.svr.cu *.svr *.linkinfo ./dscudatmp
