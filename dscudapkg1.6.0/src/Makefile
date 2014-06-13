# ARCH = $(shell uname -m)
# ifeq ($(ARCH), x86_64)

ifeq ($(shell uname -m), x86_64)
	LP64 = 1
else
	LP64 = 0
endif

CC            = gcc -O0 -g -ffast-math -funroll-loops
CXX           = g++ -O0 -g -ffast-math -funroll-loops
CUDAPATH     ?= /usr/local/cuda
NVCC          = $(CUDAPATH)/bin/nvcc -O0 -g
CUDASDKPATH  ?= $(CUDAPATH)/NVIDIA_GPU_Computing_SDK
CUDAINCLUDES  = -I. -I$(CUDAPATH)/include -I$(CUDASDKPATH)/C/common/inc -I$(CUDASDKPATH)/common/inc
CUDALIBS      = -L. -L$(CUDAPATH)/lib64

TARGET         = dscudad dscudasvr libdscuda_ibv.a libdscuda_tcp.a libcudart.so.3
TARGET_TCPONLY = dscudad_tcp dscudasvr_tcp libdscuda_tcp.a libcudart.so.3
DSCUDAHDR      = dscuda.h dscudadefs.h dscudamacros.h
CLIENTBASE     = dscudaverb.h libdscuda.h libdscuda.cu dscudaverb.o

default :	$(TARGET)

tcponly:	$(TARGET_TCPONLY)

libdscuda_ibv.so :	$(CLIENTBASE) libdscuda_ibv.cu $(DSCUDAHDR) \
                        dscudautil.o ibvutil.o sockutil.o
	$(NVCC) $(CUDAINCLUDES) -c -o libdscuda_ibv.o libdscuda_ibv.cu
	$(CXX) -shared -Wl,-soname,libdscuda.so -o $@ libdscuda_ibv.o dscudautil.o \
               ibvutil.o dscudaverb.o sockutil.o

libdscuda_tcp.so :	$(CLIENTBASE) libdscuda_tcp.cu $(DSCUDAHDR) \
                        dscudautil.o sockutil.o
	$(NVCC) $(CUDAINCLUDES) -c -o libdscuda_tcp.o libdscuda_tcp.cu
	$(CXX) -shared -Wl,-soname,libdscuda.so -o $@ libdscuda_tcp.o \
               dscudautil.o dscudaverb.o sockutil.o

libdscuda_ibv.a :	$(CLIENTBASE) libdscuda_ibv.cu $(DSCUDAHDR) \
                        dscudautil.o ibvutil.o sockutil.o
	$(NVCC) $(CUDAINCLUDES) -c -o libdscuda_ibv.o libdscuda_ibv.cu
	ar rvu $@ libdscuda_ibv.o dscudautil.o ibvutil.o dscudaverb.o sockutil.o
	ranlib $@

libdscuda_tcp.a :	$(CLIENTBASE) libdscuda_tcp.cu $(DSCUDAHDR) \
                        dscudautil.o sockutil.o
	$(NVCC) $(CUDAINCLUDES) -DTCP_ONLY=1 -c -o libdscuda_tcp.o libdscuda_tcp.cu
	ar rvu $@ libdscuda_tcp.o dscudautil.o dscudaverb.o  sockutil.o
	ranlib $@

# A dummy library to override CUDA runtime library functions.
libcudart.so.3:	cudart_dummy.cu
	$(NVCC) $(CUDAINCLUDES) -shared -Xcompiler -fPIC -o $@ $<

dscudaverb.o:	dscudaverb.cu
	$(NVCC) $(CUDAINCLUDES) -DTCP_ONLY=1 -c -o $@ $<

dscudad :	dscudad.c sockutil.c sockutil.h
	$(CC) -o $@ dscudad.c sockutil.c

dscudad_tcp:	dscudad.c sockutil.c sockutil.h
	$(CC) -DTCP_ONLY=1 -o $@ dscudad.c sockutil.c

dummysvr :	dummysvr.c

dummyclient :	dummyclient.c

dscudasvr :	dscudasvr.cu dscudasvr_ibv.cu dscudasvr_tcp.cu $(DSCUDAHDR) \
                dscudautil.o ibvutil.o sockutil.o
	$(NVCC) $(CUDAINCLUDES) -c -o $@.o dscudasvr.cu
	$(NVCC) $(CUDALIBS) -o $@ dscudasvr.o dscudautil.o ibvutil.o  sockutil.o \
                -lcufft -lcudart -lcuda -lrdmacm -libverbs -lpthread

dscudasvr_tcp:	dscudasvr.cu dscudasvr_tcp.cu $(DSCUDAHDR) \
                dscudautil.o sockutil.o
	$(NVCC) $(CUDAINCLUDES) -DTCP_ONLY=1 -c -o $@.o dscudasvr.cu
	$(NVCC) $(CUDALIBS) -o $@ $@.o dscudautil.o  sockutil.o \
                -lcufft -lcudart -lcuda -lpthread

dscudatest :	dscudatest.cu libdscuda_ibv.so
	$(NVCC) $(CUDAINCLUDES) -o $@ $< -L../lib -ldscuda_ibv -lrdmacm -libverbs -lpthread

dscudautil.o:	dscudautil.cu
	$(NVCC) $(CUDAINCLUDES) -c -o $@ $<

ibvutil.o:	ibvutil.cpp ibvdefs.h
	$(CXX) $(CUDAINCLUDES) -c -o $@ ibvutil.cpp

clean:
	/bin/rm $(TARGET) $(TARGET_TCPONLY) *.o *~ *.linkinfo
