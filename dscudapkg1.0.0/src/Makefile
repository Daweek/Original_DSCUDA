RPCGEN        = rpcgen -N
CC            = gcc -O0 -g -ffast-math -funroll-loops
CXX           = g++ -O0 -g -ffast-math -funroll-loops
CUDAPATH     ?= /usr/local/cuda
NVCC          = $(CUDAPATH)/bin/nvcc -O0 -g
CUDASDKPATH  ?= $(CUDAPATH)/NVIDIA_GPU_Computing_SDK
CUDAINCLUDES  = -I. -I$(CUDAPATH)/include -I$(CUDASDKPATH)/C/common/inc
CUDALIBS      = -L. -L$(CUDAPATH)/lib64

TARGET   = dscudasvr libdscuda.a libcudart.so.3 dscudatest
DSCUDAIF  = dscudarpc
DSCUDAHDR = dscuda.h

USE_IBV  = 1    # set 1 to use InfiniBand Verbs for remote procedure call. otherwise use RPC.

default :	$(TARGET)

libdscuda.a :	libdscuda.cu $(DSCUDAHDR) $(DSCUDAIF)_clnt.o $(DSCUDAIF)_xdr.o dscudautil.o ibv_rdma.o
	$(NVCC) $(CUDAINCLUDES) -c -o libdscuda.o libdscuda.cu -DUSE_IBV=$(USE_IBV)
	ar rvu $@ libdscuda.o $(DSCUDAIF)_clnt.o $(DSCUDAIF)_xdr.o dscudautil.o ibv_rdma.o
	ranlib $@

# A dummy library to override CUDA runtime library functions.
libcudart.so.3:	cudart_dummy.c
	$(CXX) $(CUDAINCLUDES) -fPIC -shared -o $@ $<

dscudasvr :	dscudasvr.cu $(DSCUDAHDR) $(DSCUDAIF)_svc.o $(DSCUDAIF)_xdr.o dscudautil.o ibv_rdma.o
	$(NVCC) $(CUDAINCLUDES) -c -o $@.o dscudasvr.cu
	$(NVCC) $(CUDALIBS) -o $@ dscudasvr.o $(DSCUDAIF)_svc.o $(DSCUDAIF)_xdr.o dscudautil.o ibv_rdma.o -lcudart -lcuda -lrdmacm -libverbs -lpthread

#	$(NVCC) $(CUDAINCLUDES) -o $@ dscudasvr.cu $(DSCUDAIF)_svc.o $(DSCUDAIF)_xdr.o dscudautil.o ibv_rdma.o -lcuda -lrdmacm


dscudatest :	dscudatest.cu libdscuda.a
	$(NVCC) $(CUDAINCLUDES) -o $@ $< -L../lib -ldscuda -lrdmacm -libverbs -lpthread


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
	/bin/rm $(TARGET) $(DSCUDAIF).h $(DSCUDAIF)_svc.* $(DSCUDAIF)_clnt.* $(DSCUDAIF)_xdr.* *.o *~ *.linkinfo
