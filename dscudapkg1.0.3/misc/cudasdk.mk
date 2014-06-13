.SUFFIXES : .cu .cu_dbg.o .c_dbg.o .cpp_dbg.o .cu_rel.o .c_rel.o .cpp_rel.o .cubin .ptx

RCUDACC        ?= $(RCUDA_PATH)/bin/rcudacc
CUDAPATH       ?= /usr/local/cuda/
CUDASDKPATH    ?= $(CUDAPATH)/NVIDIA_GPU_Computing_SDK

NVCC           ?= $(CUDAPATH)/bin/nvcc
CXX            ?= g++
CC             ?= g++

INCLUDES       += -I. -I$(CUDAPATH)/include -I$(CUDASDKPATH)/C/common/inc -I$(CUDASDKPATH)/shared/inc
LIBPATH        += -L$(RCUDA_PATH)/lib -L$(CUDASDKPATH)/C/lib -L$(CUDASDKPATH)/shared/lib
LIB            += -lshrutil_i386 -lcutil_i386

ifeq ($(USECUFFT),1)
  ifeq ($(emu),1)
    LIB += -lcufftemu
  else
    LIB += -lcufft
  endif
endif

ifeq ($(USECUBLAS),1)
  ifeq ($(emu),1)
    LIB += -lcublasemu
  else
    LIB += -lcublas
  endif
endif

ifeq ($(USECURAND),1)
    LIB += -lcurand
endif

ifeq ($(USECUSPARSE),1)
  LIB += -lcusparse
endif

COMMONFLAGS    += $(INCLUDES) -DUNIX
NVCCFLAGS      += -m32 --compiler-options -fno-strict-aliasing -O2
CXXFLAGS       += -m32 -O2 -fno-strict-aliasing
CFLAGS         += -m32 -O2 -fno-strict-aliasing
LDFLAGS        += --compiler-options -fPIC -m32

NVCCFLAGS      += $(COMMONFLAGS)
CXXFLAGS       += $(COMMONFLAGS)
CFLAGS         += $(COMMONFLAGS)

OBJS +=  $(patsubst %.cu,%.cu.o,$(CUFILES))
OBJS +=  $(patsubst %.cpp,%.cpp.o,$(CCFILES))
OBJS +=  $(patsubst %.c,%.c.o,$(CFILES))

SM_VERSIONS   := 10 11 12 13 20
GENCODE_SM10 := -gencode=arch=compute_10,code=\\\"sm_10,compute_10\\\"
GENCODE_SM20 := -gencode=arch=compute_20,code=\\\"sm_20,compute_20\\\"

define SMVERSION_template
OBJS += $(patsubst %.cu,%.cu_$(1).o,$(notdir $(CUFILES_sm_$(1))))
%.cu_$(1).o : %.cu $(CU_DEPS)
	$(RCUDACC) -gencode=arch=compute_$(1),code=\\\"sm_$(1),compute_$(1)\\\" -o $$@ $(CUDAINCLUDES) $(NVCCFLAGS) -c -i $$<
endef


#	$(RCUDACC) -gencode=arch=compute_$(1),code=\\\"sm_$(1),compute_$(1)\\\" $(GENCODE_SM20) -o $$@ $(CUDAINCLUDES) $(NVCCFLAGS) -c -i $$<


$(foreach smver,$(SM_VERSIONS),$(eval $(call SMVERSION_template,$(smver))))


TARGET = $(EXECUTABLE)

$(TARGET):	$(OBJS)
	$(NVCC) $(LDFLAGS) -o $@ $(LIBPATH) $^ -lrcuda $(LIB)

%.cu.o:		%.cu $(CU_DEPS)
	echo objs $(OBJS)
	echo src $^
	$(RCUDACC) -o $@ $(CUDAINCLUDES) $(NVCCFLAGS) -c -i $<

%.cpp.o:	%.cpp $(C_DEPS)
	$(CXX) -o $@ $(CXXFLAGS) -c $<

%.c.o:	%.c $(C_DEPS)
	$(CC) -o $@ $(CFLAGS) -c $<

clean:
	rm -rf $(OBJS) *.ptx $(TARGET) *.linkinfo ./rcudatmp
