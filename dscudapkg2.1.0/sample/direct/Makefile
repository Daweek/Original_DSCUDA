CPP            = g++
DSCUDACPP     ?= ../../bin/dscudacpp -arch sm_35 -rdc=true
CUDAPATH      ?= /usr/local/cuda
CUDASDKPATH   ?= /usr/local/cuda/NVIDIA_GPU_Computing_SDK
CUDAINCLUDES   = -I. -I$(CUDAPATH)/include -I$(CUDASDKPATH)/C/common/inc

NVCC          ?= $(CUDAPATH)/bin/nvcc -use_fast_math -O 
TARGET         = dsdirectMultiGPU

default:	$(TARGET)

run:	dsdirect pl2k pl8k pl64k
	./dsdirect pl2k pl2k.snapshot
	./dsdirect pl8k pl8k.snapshot

bigrun:	dsdirect pl64k
	./dsdirect pl64k pl64k.snapshot

dsdirectMultiGPU:	directMultiGPU.cu util.o ../../lib/libdscuda_ibv.a
	$(DSCUDACPP) -c -o $@.o $(CUDAINCLUDES) -i directMultiGPU.cu
	$(DSCUDACPP) -o $@ -i $@.o util.o

dsdirectMultiGPU_omp:	directMultiGPU.cu util.o ../../lib/libdscuda_ibv.a
	$(DSCUDACPP) -c -o $@.o $(CUDAINCLUDES) -i directMultiGPU.cu -Xcompiler -fopenmp 
	$(DSCUDACPP) -o $@ -i $@.o util.o -fopenmp 

dsdirect:	direct.cu util.o ../../lib/libdscuda_ibv.a
	$(DSCUDACPP) -c -o $@.o $(CUDAINCLUDES) -i direct.cu
	$(DSCUDACPP) -o $@ -i $@.o util.o

directMultiGPU_omp:	directMultiGPU.cu util.o
	$(NVCC) -o $@ $(CUDAINCLUDES) directMultiGPU.cu util.o -lpthread -Xcompiler -fopenmp 

direct:		direct.cu util.o
	$(NVCC)  -o $@ $(CUDAINCLUDES) direct.cu util.o

util.o:		util.c
	$(CPP) -c -o $@ util.c

mkdist:		mkdist.c util.o
	$(CPP) -o $@ mkdist.c util.o -lm

pl2k:	mkdist
	./mkdist 2048 1 pl2k

pl8k:	mkdist
	./mkdist 8192 1 pl8k

pl64k:	mkdist
	./mkdist 65536 1 pl64k

clean:
	rm -rf *.o ./dscudatmp ./dscudasvr $(TARGET)_tcp $(TARGET)_ibv *.linkinfo pl2k pl8k pl64k *.snapshot
