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
NVCC0         = $(CUDAPATH)/bin/nvcc -g -use_fast_math # -arch sm_35 should not be given.
NVCC          = $(CUDAPATH)/bin/nvcc -g -arch sm_35 -rdc=true -use_fast_math
PICOPT        = -Xcompiler -fPIC
SVRLIBOPT     = -D_DSCUDA_RUNTIME_API_LAUNCH=1

CUDASDKPATH  ?= $(CUDAPATH)/NVIDIA_GPU_Computing_SDK
CUDAINCLUDES  = -I. -I$(CUDAPATH)/include -I$(CUDASDKPATH)/C/common/inc -I$(CUDASDKPATH)/common/inc
CUDALIBS      = -L. -L$(CUDAPATH)/lib64


BINCOMPAT      = -DBINCOMPATIBLE=1

TARGET         = dscudad dscudasvr libdscuda_ibv.a libdscuda_tcp.a libdscudasvr.a libcudart.so.3
MIN_TARGET     = dscudad libdscuda_tcp.a libdscudasvr.a libcudart.so.3

#                 libdscuda_ibv.so libdscuda_tcp.so

TARGET_TCPONLY = dscudad_tcp dscudasvr_tcp libdscuda_tcp.a libcudart.so.3
DSCUDAHDR      = dscuda.h dscudadefs.h dscudamacros.h
CLIENTBASE     = dscudaverb.h libdscuda.h libdscuda.cu dscudaverb.o

default :	$(MIN_TARGET)

all :	$(TARGET)

tcponly:	$(TARGET_TCPONLY)

libdscuda_ibv.so :	$(CLIENTBASE) libdscuda_ibv.cu $(DSCUDAHDR) \
                        dscudautil.o ibvutil.o sockutil.o
	$(NVCC) $(PICOPT) $(BINCOMPAT) $(CUDAINCLUDES) -c -o libdscuda_ibv.o libdscuda_ibv.cu
	$(CXX) -shared -Wl,-soname,libdscuda.so -o $@ libdscuda_ibv.o dscudautil.o \
               ibvutil.o dscudaverb.o sockutil.o -lrdmacm -libverbs -lpthread

libdscuda_tcp.so :	$(CLIENTBASE) libdscuda_tcp.cu $(DSCUDAHDR) \
                        dscudautil.o sockutil.o
	$(NVCC) $(PICOPT) $(BINCOMPAT) $(CUDAINCLUDES) -c -o libdscuda_tcp.o libdscuda_tcp.cu
	$(CXX) -shared -Wl,-soname,libdscuda.so -o $@ libdscuda_tcp.o \
               dscudautil.o dscudaverb.o sockutil.o

libdscuda_ibv.a :	$(CLIENTBASE) libdscuda_ibv.cu $(DSCUDAHDR) \
                        dscudautil.o ibvutil.o sockutil.o
	$(NVCC0) $(CUDAINCLUDES) -c -o libdscuda_ibv.o libdscuda_ibv.cu
	ar rvu $@ libdscuda_ibv.o dscudautil.o ibvutil.o dscudaverb.o sockutil.o
	ranlib $@

libdscuda_tcp.a :	$(CLIENTBASE) libdscuda_tcp.cu $(DSCUDAHDR) \
                        dscudautil.o sockutil.o
	$(NVCC0) $(CUDAINCLUDES) -DTCP_ONLY=1 -c -o libdscuda_tcp.o libdscuda_tcp.cu
	ar rvu $@ libdscuda_tcp.o dscudautil.o dscudaverb.o  sockutil.o
	ranlib $@

# A dummy library to override CUDA runtime library functions.
libcudart.so.3:	cudart_dummy.cu
	$(NVCC) $(CUDAINCLUDES) -shared --cudart=shared -Xcompiler -fPIC -o $@ $< # CUDA 5.5 or later.

#	$(NVCC) $(CUDAINCLUDES) -shared -Xcompiler -fPIC -o $@ $< # CUDA 5.0 or earlier

dscudaverb.o:	dscudaverb.cu
	$(NVCC) $(PICOPT) $(CUDAINCLUDES) -DTCP_ONLY=1 -c -o $@ $<

dscudad :	dscudad.c sockutil.c sockutil.h
	$(CC) -o $@ dscudad.c sockutil.c -lpthread

dscudad_tcp:	dscudad.c sockutil.c sockutil.h
	$(CC) -DTCP_ONLY=1 -o $@ dscudad.c sockutil.c

dummysvr :	dummysvr.c

dummyclient :	dummyclient.c

dscudasvr :	dscudasvr.cu dscudasvr_ibv.cu dscudasvr_tcp.cu $(DSCUDAHDR) \
                dscudautil.o ibvutil.o sockutil.o
	$(NVCC) $(CUDAINCLUDES) -c -o $@.o dscudasvr.cu
	$(NVCC) $(CUDALIBS) -o $@ dscudasvr.o dscudautil.o ibvutil.o  sockutil.o \
                -lcufft -lcudart -lcuda -lrdmacm -libverbs -lpthread

libdscudasvr.a	:	dscudasvr_ibv.cu dscudasvr_tcp.cu $(DSCUDAHDR) \
			libdscudasvr.o dscudautil.o ibvutil.o sockutil.o
			ar rvu $@ libdscudasvr.o dscudautil.o ibvutil.o  sockutil.o

libdscudasvr.o:	dscudasvr.cu dscudasvr_ibv.cu dscudasvr_tcp.cu $(DSCUDAHDR)
	$(NVCC) $(CUDAINCLUDES) $(SVRLIBOPT) -c -o $@ dscudasvr.cu

dscudasvr_tcp:	dscudasvr.cu dscudasvr_tcp.cu $(DSCUDAHDR) \
                dscudautil.o sockutil.o
	$(NVCC) $(CUDAINCLUDES) -DTCP_ONLY=1 -c -o $@.o dscudasvr.cu
	$(NVCC) $(CUDALIBS) -o $@ $@.o dscudautil.o  sockutil.o \
                -lcufft -lcudart -lcuda -lpthread

dscudatest :	dscudatest.cu libdscuda_ibv.so
	$(NVCC) $(CUDAINCLUDES) -o $@ $< -L../lib -ldscuda_ibv -lrdmacm -libverbs -lpthread

dscudautil.o:	dscudautil.cu
	$(NVCC) $(PICOPT) $(CUDAINCLUDES) -c -o $@ $<

sockutil.o:	sockutil.c
	$(NVCC) $(PICOPT) $(CUDAINCLUDES) -c -o $@ $<

ibvutil.o:	ibvutil.cpp ibvdefs.h
	$(CXX) $(PICOPT) $(CUDAINCLUDES) -c -o $@ $<

clean:
	/bin/rm $(TARGET) $(TARGET_TCPONLY) *.o *~ *.linkinfo
