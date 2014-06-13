RPCGEN        = rpcgen -N
CC            = gcc -O0 -g -ffast-math -funroll-loops
CXX           = g++ -O0 -g -ffast-math -funroll-loops
CUDAPATH     ?= /usr/local/cuda
NVCC          = $(CUDAPATH)/bin/nvcc -O0 -g
CUDASDKPATH  ?= $(CUDAPATH)/NVIDIA_GPU_Computing_SDK
CUDAINCLUDES  = -I. -I$(CUDAPATH)/include -I$(CUDASDKPATH)/C/common/inc -Xcompiler -fPIC 
CUDALIBS      = -L. -L$(CUDAPATH)/lib64

TARGET        = dscudasvr libdscuda_ibv.a libdscuda_rpc.a libcudart.so.3 dscudatest
DSCUDAIF      = dscudarpc
DSCUDAHDR     = dscuda.h
CLIENTBASE    = dscudaverb.h libdscuda.h libdscuda.cu dscudaverb.o

default :	$(TARGET)

libdscuda_ibv.so :	$(CLIENTBASE) libdscuda_ibv.cu $(DSCUDAHDR) $(DSCUDAIF)_clnt.o dscudautil.o ibv_rdma.o
	$(NVCC) $(CUDAINCLUDES) -c -o libdscuda_ibv.o libdscuda_ibv.cu
	$(CXX) -shared -Wl,-soname,libdscuda.so -o $@ libdscuda_ibv.o dscudautil.o ibv_rdma.o dscudaverb.o

libdscuda_rpc.so :	$(CLIENTBASE) libdscuda_rpc.cu $(DSCUDAHDR) $(DSCUDAIF)_clnt.o $(DSCUDAIF)_xdr.o dscudautil.o
	$(NVCC) $(CUDAINCLUDES) -c -o libdscuda_rpc.o libdscuda_rpc.cu
	$(CXX) -shared -Wl,-soname,libdscuda.so -o $@ libdscuda_rpc.o $(DSCUDAIF)_clnt.o $(DSCUDAIF)_xdr.o dscudautil.o dscudaverb.o

libdscuda_ibv.a :	$(CLIENTBASE) libdscuda_ibv.cu $(DSCUDAHDR) $(DSCUDAIF)_clnt.o dscudautil.o ibv_rdma.o
	$(NVCC) $(CUDAINCLUDES) -c -o libdscuda_ibv.o libdscuda_ibv.cu
	ar rvu $@ libdscuda_ibv.o dscudautil.o ibv_rdma.o dscudaverb.o
	ranlib $@

libdscuda_rpc.a :	$(CLIENTBASE) libdscuda_rpc.cu $(DSCUDAHDR) $(DSCUDAIF)_clnt.o $(DSCUDAIF)_xdr.o dscudautil.o
	$(NVCC) $(CUDAINCLUDES) -c -o libdscuda_rpc.o libdscuda_rpc.cu
	ar rvu $@ libdscuda_rpc.o $(DSCUDAIF)_clnt.o $(DSCUDAIF)_xdr.o dscudautil.o dscudaverb.o
	ranlib $@

# A dummy library to override CUDA runtime library functions.
libcudart.so.3:	cudart_dummy.cu
	$(NVCC) $(CUDAINCLUDES) -shared -o $@ $<

dscudaverb.o:	dscudaverb.cu
	$(NVCC) $(CUDAINCLUDES) -c -o $@ $<


dscudasvr :	dscudasvr.cu dscudasvr_ibv.cu dscudasvr_rpc.cu $(DSCUDAHDR) $(DSCUDAIF)_svc.o $(DSCUDAIF)_xdr.o dscudautil.o ibv_rdma.o
	$(NVCC) $(CUDAINCLUDES) -c -o $@.o dscudasvr.cu
	$(NVCC) $(CUDALIBS) -o $@ dscudasvr.o $(DSCUDAIF)_svc.o $(DSCUDAIF)_xdr.o dscudautil.o ibv_rdma.o -lcudart -lcuda -lrdmacm -libverbs -lpthread

#	$(NVCC) $(CUDAINCLUDES) -o $@ dscudasvr.cu $(DSCUDAIF)_svc.o $(DSCUDAIF)_xdr.o dscudautil.o ibv_rdma.o -lcuda -lrdmacm


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
	/bin/rm $(TARGET) $(DSCUDAIF).h $(DSCUDAIF)_svc.* $(DSCUDAIF)_clnt.* $(DSCUDAIF)_xdr.* *.o *~ *.linkinfo
