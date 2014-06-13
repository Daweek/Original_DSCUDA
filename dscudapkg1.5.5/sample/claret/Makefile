CPP            = g++
DSCUDAPATH = ../..
DSCUDACPP     ?= ../../bin/dscudacpp
CUDAPATH      ?= /usr/local/cuda
CUDASDKPATH   ?= /usr/local/cuda/NVIDIA_GPU_Computing_SDK
CUDAINCLUDES   = -I. -I$(CUDAPATH)/include -I$(CUDASDKPATH)/C/common/inc
NVCC          ?= $(CUDAPATH)/bin/nvcc -use_fast_math -O 
TARGET         = direct rdirect mkdist

CFLAGS         = -DVTGRAPE -O -ffast-math -funroll-loops 
NVCCFLAGS      = -DVTGRAPE -O

MDGRAPE_LIB         = $(CUDASDKPATH)/C/lib/libcutil_x86_64.a -L$(DSCUDAPATH)/lib -lcudart mr3.o
MDGRAPE_LIB_RPCONLY = $(CUDASDKPATH)/C/lib/libcutil_x86_64.a -L$(DSCUDAPATH)/lib -lcudart mr3_rpconly.o

# MDGRAPE_LIB += $(CUDASDKPATH)/C/lib/libcutil_x86_64.a -L$(CUDAPATH)/lib -lcudart


GL_INCDIR = $(CUDASDKPATH)/shared/inc
GL_LIBDIR = /usr/local/lib
GL_LIBS = -L$(GL_LIBDIR) -lglut -lGL -lGLU
SOCKET_LIB = sockhelp.o
LIB         = $(SOCKET_LIB) $(MDGRAPE_LIB) $(GL_LIBS) -lm -lstdc++
LIB_RPCONLY = $(SOCKET_LIB) $(MDGRAPE_LIB_RPCONLY) $(GL_LIBS) -lm -lstdc++

TARGET     = cras_gpu_ibv cras_gpu_rpc

all:	$(TARGET)

mr3.o:	mr3.cu
	$(DSCUDACPP) -o $@ $(CUDAINCLUDES) -c -use_fast_math -O -i $< 

mr3_rpconly.o:	mr3.cu
	$(DSCUDACPP) -DRPC_ONLY=1 -o $@ $(CUDAINCLUDES) -c -use_fast_math -O -i $< 

cras_gpu_ibv : cras36.c sockhelp.o mr3.o $(DSCUDAPATH)/lib/libdscuda_ibv.a
	$(NVCC) $(NVCCFLAGS) -I$(GL_INCDIR) $< -o $@ $(LIB) -ldscuda_ibv -lrdmacm -libverbs -lpthread

cras_gpu_rpc : cras36.c sockhelp.o mr3.o $(DSCUDAPATH)/lib/libdscuda_rpc.a
	$(NVCC) $(NVCCFLAGS) -I$(GL_INCDIR) $< -o $@ $(LIB) -ldscuda_rpc

cras_gpu_rpconly : cras36.c sockhelp.o mr3_rpconly.o $(DSCUDAPATH)/lib/libdscuda_rpc.a
	$(NVCC) $(NVCCFLAGS) -I$(GL_INCDIR) $< -o $@ $(LIB_RPCONLY) -ldscuda_rpc

cras_host : cras36.c sockhelp.o mr3_host.o
	$(CC) $(CFLAGS) -I$(GL_INCDIR) $< -o $@ $(LIB) 

mr3_host.o : mr3_host.c
	$(CC) $(CFLAGS) -c $<

sockhelp.o : sockhelp.c
	$(CC) $(CFLAGS) -o $@ -c $<

clean:
	rm -rf *.o *.ptx $(TARGET) *.linkinfo ./dscudatmp
