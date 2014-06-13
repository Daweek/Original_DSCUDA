CPP            = g++
DSCUDACPP     ?= ../../bin/dscudacpp
CUDAPATH      ?= /usr/local/cuda
CUDASDKPATH   ?= /usr/local/cuda/NVIDIA_GPU_Computing_SDK
CUDAINCLUDES   = -I. -I$(CUDAPATH)/include -I$(CUDASDKPATH)/C/common/inc

NVCC          ?= $(CUDAPATH)/bin/nvcc -use_fast_math -O 
TARGET         = dsdirectMultiGPU dsdirectMultiGPU_omp dsdirectMultiGPU_rpc

all:	$(TARGET)

run:	dsdirect pl2k pl8k pl64k
	./dsdirect pl2k pl2k.snapshot
	./dsdirect pl8k pl8k.snapshot

bigrun:	dsdirect pl64k
	./dsdirect pl64k pl64k.snapshot

dsdirectMultiGPU:	directMultiGPU.cu util.o ../../lib/libdscuda_ibv.a
	$(DSCUDACPP) -c -o $@.o $(CUDAINCLUDES) -i directMultiGPU.cu
	$(CPP) -L$(DSCUDA_PATH)/lib -o $@ $@.o util.o -ldscuda_ibv -lrdmacm -libverbs -lpthread -lcudart

dsdirectMultiGPU_rpc:	directMultiGPU.cu util.o ../../lib/libdscuda_rpc.a
	$(DSCUDACPP) -c -DRPC_ONLY=1 -o $@.o $(CUDAINCLUDES) -i directMultiGPU.cu
	$(CPP) -L$(DSCUDA_PATH)/lib -o $@ $@.o util.o -ldscuda_rpc -lcudart

dsdirectMultiGPU_omp:	directMultiGPU.cu util.o ../../lib/libdscuda_ibv.a
	$(DSCUDACPP) -c -o $@.o $(CUDAINCLUDES) -i directMultiGPU.cu -Xcompiler -fopenmp 
	$(CPP) -L$(DSCUDA_PATH)/lib -o $@ $@.o util.o -ldscuda_ibv -lrdmacm -libverbs -lpthread -lcudart -fopenmp 

dsdirect:	direct.cu util.o ../../lib/libdscuda_ibv.a
	$(DSCUDACPP) -c -o $@.o $(CUDAINCLUDES) -i direct.cu
	$(CPP) -L$(DSCUDA_PATH)/lib -o $@ $@.o util.o -ldscuda_ibv -lrdmacm -libverbs -lpthread -lcudart

dsdirect_rpc:	direct.cu util.o ../../lib/libdscuda_rpc.a
	$(DSCUDACPP) -c -DRPC_ONLY=1 -o $@.o $(CUDAINCLUDES) -i direct.cu
	$(CPP) -L$(DSCUDA_PATH)/lib -o $@ $@.o util.o -ldscuda_rpc -lcudart

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
	rm -rf *.o *.ptx $(TARGET) *.linkinfo pl2k pl8k pl64k *.snapshot ./dscudatmp
