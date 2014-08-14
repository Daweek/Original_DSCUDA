NVCC             = nvcc -ccbin g++
NVCCOPT          = -arch sm_35 -rdc=true
CPP              = g++
CUDAPATH        ?= /usr/local/cuda/
CUDALIBPATH      = $(CUDAPATH)/lib64
CUDASDKPATH      = $(CUDAPATH)/samples
DSCUDAPATH      ?= ../..
DSCUDACPP        = $(DSCUDAPATH)/bin/dscudacpp
DSCUDALIBPATH    = $(DSCUDAPATH)/lib
CLIENT           = userapp
SERVER           = dscudasvr
TARGET          = $(CLIENT)_ibv $(CLIENT)_tcp $(SERVER)
MIN_TARGET      = $(CLIENT)_tcp $(SERVER)

default:	$(MIN_TARGET)

all:	$(TARGET)

$(CLIENT)_ibv:	main.o $(DSCUDALIBPATH)/libdscuda_ibv.a
	$(NVCC) $(NVCCOPT) -o $@ $^ -L$(DSCUDALIBPATH) -ldscuda_ibv -lrdmacm -libverbs -lpthread -lcudart

$(CLIENT)_tcp:	main.o $(DSCUDALIBPATH)/libdscuda_tcp.a
	$(NVCC) $(NVCCOPT) -o $@ $^ -L$(DSCUDALIBPATH) -ldscuda_tcp -lcudart

dscudasvr:	main.svr.o $(DSCUDALIBPATH)/libdscudasvr.a
	$(NVCC) $(NVCCOPT) -o $@ $^ -L$(CUDALIBPATH) -lcuda -lrdmacm -libverbs -lpthread

%.o: %.cu
	$(DSCUDACPP) $(NVCCOPT) -c -I. -I$(CUDASDKPATH)/common/inc -o $@ -i $<

%.svr.o:	%.cu
	$(NVCC) -DNOMAIN $(NVCCOPT) -c -I. -I$(CUDASDKPATH)/common/inc -o $@ $<

# %.svr.o:	%.ii
#	$(NVCC) -DNOMAIN $(NVCCOPT) -c -I. -I$(CUDASDKPATH)/common/inc -o $@ $<

%.ii:	%.cu
	$(NVCC) -DNOMAIN $(NVCCOPT) -cuda -I. -I$(CUDASDKPATH)/common/inc -o $@ $<

clean:
	rm -rf *.o $(TARGET) ./dscudatmp
