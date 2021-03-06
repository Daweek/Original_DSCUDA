# ARCH = $(shell uname -m)
# ifeq ($(ARCH), x86_64)

ifeq ($(shell uname -m), x86_64)
	LP64 = 1
else
	LP64 = 0
endif

RPCGEN        = rpcgen -N -D__LP64__=$(LP64)
CC            = gcc -O0 -g -ffast-math -funroll-loops
CXX           = g++ -O0 -g -ffast-math -funroll-loops
CUDAPATH     ?= /usr/local/cuda
NVCC          = $(CUDAPATH)/bin/nvcc -O0 -g
CUDASDKPATH  ?= $(CUDAPATH)/NVIDIA_GPU_Computing_SDK
CUDAINCLUDES  = -I. -I$(CUDAPATH)/include -I$(CUDASDKPATH)/C/common/inc
CUDALIBS      = -L. -L$(CUDAPATH)/lib64

TARGET         = dscudad dscudasvr libdscuda_ibv.a libdscuda_rpc.a libcudart.so.3
TARGET_RPCONLY = dscudad_rpc dscudasvr_rpc libdscuda_rpc.a libcudart.so.3
DSCUDAIF       = dscudarpc
DSCUDAHDR      = dscuda.h dscudadefs.h dscudamacros.h
CLIENTBASE     = dscudaverb.h libdscuda.h libdscuda.cu dscudaverb.o

default :	$(TARGET)

rpconly:	$(TARGET_RPCONLY)

libdscuda_ibv.so :	$(CLIENTBASE) libdscuda_ibv.cu $(DSCUDAHDR) $(DSCUDAIF)_clnt.o \
                        dscudautil.o ibv_rdma.o sockutil.o
	$(NVCC) $(CUDAINCLUDES) -c -o libdscuda_ibv.o libdscuda_ibv.cu
	$(CXX) -shared -Wl,-soname,libdscuda.so -o $@ libdscuda_ibv.o dscudautil.o \
               ibv_rdma.o dscudaverb.o sockutil.o

libdscuda_rpc.so :	$(CLIENTBASE) libdscuda_rpc.cu $(DSCUDAHDR) $(DSCUDAIF)_clnt.o \
                        $(DSCUDAIF)_xdr.o dscudautil.o sockutil.o
	$(NVCC) $(CUDAINCLUDES) -c -o libdscuda_rpc.o libdscuda_rpc.cu
	$(CXX) -shared -Wl,-soname,libdscuda.so -o $@ libdscuda_rpc.o $(DSCUDAIF)_clnt.o \
               $(DSCUDAIF)_xdr.o dscudautil.o dscudaverb.o sockutil.o

libdscuda_ibv.a :	$(CLIENTBASE) libdscuda_ibv.cu $(DSCUDAHDR) $(DSCUDAIF)_clnt.o \
                        dscudautil.o ibv_rdma.o sockutil.o
	$(NVCC) $(CUDAINCLUDES) -c -o libdscuda_ibv.o libdscuda_ibv.cu
	ar rvu $@ libdscuda_ibv.o dscudautil.o ibv_rdma.o dscudaverb.o sockutil.o
	ranlib $@

libdscuda_rpc.a :	$(CLIENTBASE) libdscuda_rpc.cu $(DSCUDAHDR) $(DSCUDAIF)_clnt.o \
                        $(DSCUDAIF)_xdr.o dscudautil.o sockutil.o
	$(NVCC) $(CUDAINCLUDES) -DRPC_ONLY=1 -c -o libdscuda_rpc.o libdscuda_rpc.cu
	ar rvu $@ libdscuda_rpc.o $(DSCUDAIF)_clnt.o $(DSCUDAIF)_xdr.o dscudautil.o dscudaverb.o  sockutil.o
	ranlib $@

# A dummy library to override CUDA runtime library functions.
libcudart.so.3:	cudart_dummy.cu
	$(NVCC) $(CUDAINCLUDES) -shared -Xcompiler -fPIC -o $@ $<

dscudaverb.o:	dscudaverb.cu
	$(NVCC) $(CUDAINCLUDES) -DRPC_ONLY=1 -c -o $@ $<

dscudad :	dscudad.c sockutil.c sockutil.h
	$(CC) -o $@ dscudad.c sockutil.c

dscudad_rpc :	dscudad.c sockutil.c sockutil.h
	$(CC) -DRPC_ONLY=1 -o $@ dscudad.c sockutil.c

dummysvr :	dummysvr.c

dummyclient :	dummyclient.c

dscudasvr :	dscudasvr.cu dscudasvr_ibv.cu dscudasvr_rpc.cu $(DSCUDAHDR) $(DSCUDAIF)_svc.o \
                $(DSCUDAIF)_xdr.o dscudautil.o ibv_rdma.o sockutil.o
	$(NVCC) $(CUDAINCLUDES) -c -o $@.o dscudasvr.cu
	$(NVCC) $(CUDALIBS) -o $@ dscudasvr.o $(DSCUDAIF)_svc.o $(DSCUDAIF)_xdr.o dscudautil.o ibv_rdma.o  sockutil.o \
                -lcufft -lcudart -lcuda -lrdmacm -libverbs -lpthread

dscudasvr_rpc :	dscudasvr.cu dscudasvr_rpc.cu $(DSCUDAHDR) $(DSCUDAIF)_svc.o $(DSCUDAIF)_xdr.o \
                dscudautil.o sockutil.o
	$(NVCC) $(CUDAINCLUDES) -DRPC_ONLY=1 -c -o $@.o dscudasvr.cu
	$(NVCC) $(CUDALIBS) -o $@ $@.o $(DSCUDAIF)_svc.o $(DSCUDAIF)_xdr.o dscudautil.o  sockutil.o \
                -lcufft -lcudart -lcuda -lpthread

dscudatest :	dscudatest.cu libdscuda_ibv.so
	$(NVCC) $(CUDAINCLUDES) -o $@ $< -L../lib -ldscuda_ibv -lrdmacm -libverbs -lpthread


$(DSCUDAIF)_clnt.o:	$(DSCUDAIF)_clnt.c $(DSCUDAIF).h
	$(CXX) $(CUDAINCLUDES) -c -o $@ $(DSCUDAIF)_clnt.c

$(DSCUDAIF)_svc.o:	$(DSCUDAIF)_svc.c $(DSCUDAIF).h
	$(CXX) $(CUDAINCLUDES) -c -o $@ $(DSCUDAIF)_svc.c

$(DSCUDAIF)_xdr.o:	$(DSCUDAIF)_xdr.c $(DSCUDAIF).h
	$(CXX) $(CUDAINCLUDES) -c -o $@ $(DSCUDAIF)_xdr.c

dscudautil.o:	dscudautil.cu
	$(NVCC) $(CUDAINCLUDES) -c -o $@ $<

ibv_rdma.o:	$(DSCUDAIF).h ibv_rdma.cpp ibv_rdma.h
	$(CXX) $(CUDAINCLUDES) -c -o $@ ibv_rdma.cpp

$(DSCUDAIF)_clnt.c :	$(DSCUDAIF).x
	$(RPCGEN) -l $< > $@

$(DSCUDAIF)_svc.c:	$(DSCUDAIF).x
	$(RPCGEN) -m $< > $@

#	$(RPCGEN) -s tcp -s udp $< > $@

$(DSCUDAIF)_xdr.c:	$(DSCUDAIF).x
	$(RPCGEN) -c $< > $@

$(DSCUDAIF).h:		$(DSCUDAIF).x
	$(RPCGEN) -h $< > $@

clean:
	/bin/rm $(TARGET) $(TARGET_RPCONLY) $(DSCUDAIF).h $(DSCUDAIF)_svc.* $(DSCUDAIF)_clnt.* $(DSCUDAIF)_xdr.* *.o *~ *.linkinfo
