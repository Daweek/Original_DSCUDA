.SUFFIXES : .cu .cu_dbg.o .c_dbg.o .cpp_dbg.o .cu_rel.o .c_rel.o .cpp_rel.o .cubin .ptx

RCUDACC        ?= $(RCUDA_PATH)/bin/rcudacc
CUDAPATH       ?= /usr/local/cuda/
CUDASDKPATH    ?= $(CUDAPATH)/NVIDIA_GPU_Computing_SDK

# Basic directory setup for SDK
# (override directories only if they are not already defined)
SRCDIR     ?= 
ROOTDIR    ?= ..
ROOTBINDIR ?= $(ROOTDIR)/../bin
BINDIR     ?= $(ROOTBINDIR)/$(OSLOWER)
ROOTOBJDIR ?= obj
LIBDIR     := $(ROOTDIR)/../lib
COMMONDIR  := $(ROOTDIR)/../common
SHAREDDIR  := $(ROOTDIR)/../../shared/


NVCC           ?= $(CUDAPATH)/bin/nvcc
CXX            ?= g++
CC             ?= g++



# architecture flag for nvcc and gcc compilers build
CUBIN_ARCH_FLAG :=
CXX_ARCH_FLAGS  :=
NVCCFLAGS       :=
LIB_ARCH        := $(OSARCH)

# Determining the necessary Cross-Compilation Flags
# 32-bit OS, but we target 64-bit cross compilation
ifeq ($(x86_64),1) 
    NVCCFLAGS       += -m64
    LIB_ARCH         = x86_64
    CUDPPLIB_SUFFIX  = x86_64
    ifneq ($(DARWIN),)
         CXX_ARCH_FLAGS += -arch x86_64
    else
         CXX_ARCH_FLAGS += -m64
    endif
else 
# 64-bit OS, and we target 32-bit cross compilation
    ifeq ($(i386),1)
        NVCCFLAGS       += -m32
        LIB_ARCH         = i386
        CUDPPLIB_SUFFIX  = i386
        ifneq ($(DARWIN),)
             CXX_ARCH_FLAGS += -arch i386
        else
             CXX_ARCH_FLAGS += -m32
        endif
    else 
        ifeq "$(strip $(HP_64))" ""
            LIB_ARCH        = i386
            CUDPPLIB_SUFFIX = i386
            NVCCFLAGS      += -m32
            ifneq ($(DARWIN),)
               CXX_ARCH_FLAGS += -arch i386
            else
               CXX_ARCH_FLAGS += -m32
            endif
        else
            LIB_ARCH        = x86_64
            CUDPPLIB_SUFFIX = x86_64
            NVCCFLAGS      += -m64
            ifneq ($(DARWIN),)
               CXX_ARCH_FLAGS += -arch x86_64
            else
               CXX_ARCH_FLAGS += -m64
            endif
        endif
    endif
endif

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
NVCCFLAGS      += --compiler-options -fno-strict-aliasing -O2
CXXFLAGS       += -O2 -fno-strict-aliasing
CFLAGS         += -O2 -fno-strict-aliasing
LDFLAGS        += --compiler-options -fPIC

NVCCFLAGS      += $(COMMONFLAGS) $(GENCODE_ARCH)
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

# detect OS
OSUPPER = $(shell uname -s 2>/dev/null | tr [:lower:] [:upper:])
OSLOWER = $(shell uname -s 2>/dev/null | tr [:upper:] [:lower:])

# 'linux' is output for Linux system, 'darwin' for OS X
DARWIN = $(strip $(findstring DARWIN, $(OSUPPER)))
ifneq ($(DARWIN),)
   SNOWLEOPARD = $(strip $(findstring 10.6, $(shell egrep "<string>10\.6" /System/Library/CoreServices/SystemVersion.plist)))
endif

# OpenGL is used or not (if it is used, then it is necessary to include GLEW)
ifeq ($(USEGLLIB),1)
    ifneq ($(DARWIN),)
        OPENGLLIB := -L/System/Library/Frameworks/OpenGL.framework/Libraries 
        OPENGLLIB += -lGL -lGLU $(COMMONDIR)/lib/$(OSLOWER)/libGLEW.a
    else
# this case for linux platforms
	OPENGLLIB := -lGL -lGLU -lX11 -lXi -lXmu
# check if x86_64 flag has been set, otherwise, check HP_64 is i386/x86_64
        ifeq ($(x86_64),1) 
	       OPENGLLIB += -lGLEW_x86_64 -L/usr/X11R6/lib64
        else
             ifeq ($(i386),)
                 ifeq "$(strip $(HP_64))" ""
	             OPENGLLIB += -lGLEW -L/usr/X11R6/lib
                 else
	             OPENGLLIB += -lGLEW_x86_64 -L/usr/X11R6/lib64
                 endif
             endif
        endif
# check if i386 flag has been set, otehrwise check HP_64 is i386/x86_64
        ifeq ($(i386),1)
	       OPENGLLIB += -lGLEW -L/usr/X11R6/lib
        else
             ifeq ($(x86_64),)
                 ifeq "$(strip $(HP_64))" ""
	             OPENGLLIB += -lGLEW -L/usr/X11R6/lib
                 else
	             OPENGLLIB += -lGLEW_x86_64 -L/usr/X11R6/lib64
                 endif
             endif
        endif
    endif
endif

ifeq ($(USEGLUT),1)
    ifneq ($(DARWIN),)
	OPENGLLIB += -framework GLUT
    else
        ifeq ($(x86_64),1)
	     OPENGLLIB += -lglut -L/usr/lib64 
        endif
        ifeq ($(i386),1)
	     OPENGLLIB += -lglut -L/usr/lib 
        endif

        ifeq ($(x86_64),)
            ifeq ($(i386),)  
	        OPENGLLIB += -lglut
            endif
        endif
    endif
endif

ifeq ($(USEPARAMGL),1)
	PARAMGLLIB := -lparamgl_$(LIB_ARCH)$(LIBSUFFIX)
endif

ifeq ($(USERENDERCHECKGL),1)
	RENDERCHECKGLLIB := -lrendercheckgl_$(LIB_ARCH)$(LIBSUFFIX)
endif

ifeq ($(USECUDPP), 1)
    CUDPPLIB := -lcudpp_$(CUDPPLIB_SUFFIX)

    ifeq ($(emu), 1)
        CUDPPLIB := $(CUDPPLIB)_emu
    endif
endif

ifeq ($(USENVCUVID), 1)
     ifneq ($(DARWIN),)
         NVCUVIDLIB := -L$(ROOTDIR)/common/lib/darwin -lnvcuvid
     endif
endif

# Libs
ifneq ($(DARWIN),)
    LIB       := -L$(CUDA_INSTALL_PATH)/lib -L$(LIBDIR) -L$(COMMONDIR)/lib/$(OSLOWER) -L$(SHAREDDIR)/lib $(NVCUVIDLIB) 
else
  ifeq "$(strip $(HP_64))" ""
    ifeq ($(x86_64),1)
       LIB       := -L$(CUDA_INSTALL_PATH)/lib64 -L$(LIBDIR) -L$(COMMONDIR)/lib/$(OSLOWER) -L$(SHAREDDIR)/lib 
    else
       LIB       := -L$(CUDA_INSTALL_PATH)/lib -L$(LIBDIR) -L$(COMMONDIR)/lib/$(OSLOWER) -L$(SHAREDDIR)/lib
    endif
  else
    ifeq ($(i386),1)
       LIB       := -L$(CUDA_INSTALL_PATH)/lib -L$(LIBDIR) -L$(COMMONDIR)/lib/$(OSLOWER) -L$(SHAREDDIR)/lib
    else
       LIB       := -L$(CUDA_INSTALL_PATH)/lib64 -L$(LIBDIR) -L$(COMMONDIR)/lib/$(OSLOWER) -L$(SHAREDDIR)/lib
    endif
  endif
endif

# If dynamically linking to CUDA and CUDART, we exclude the libraries from the LIB
ifeq ($(USECUDADYNLIB),1)
     LIB += ${OPENGLLIB} $(PARAMGLLIB) $(RENDERCHECKGLLIB) $(CUDPPLIB) ${LIB} -ldl -rdynamic 
else
# static linking, we will statically link against CUDA and CUDART
  ifeq ($(USEDRVAPI),1)
     LIB += -lcuda   ${OPENGLLIB} $(PARAMGLLIB) $(RENDERCHECKGLLIB) $(CUDPPLIB) ${LIB} 
  else
     ifeq ($(emu),1) 
         LIB += -lcudartemu
     else 
         LIB += -lcudart
     endif
     LIB += ${OPENGLLIB} $(PARAMGLLIB) $(RENDERCHECKGLLIB) $(CUDPPLIB) ${LIB}
  endif
endif

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


INCLUDES       += -I. -I$(CUDAPATH)/include -I$(CUDASDKPATH)/C/common/inc -I$(CUDASDKPATH)/shared/inc
CUDAINCLUDES   += $(INCLUDES)
LIBPATH        += -L$(RCUDA_PATH)/lib -L$(CUDASDKPATH)/C/lib -L$(CUDASDKPATH)/shared/lib -L$(CUDASDKPATH)/shared/lib/linux
LIB            += -lshrutil_x86_64 -lcutil_x86_64 -lrdmacm -lGLEW_x86_64 -lglut -lparamgl_x86_64 -lGL




#	$(RCUDACC) -gencode=arch=compute_$(1),code=\\\"sm_$(1),compute_$(1)\\\" $(GENCODE_SM20) -o $$@ $(CUDAINCLUDES) $(NVCCFLAGS) -c -i $$<


$(foreach smver,$(SM_VERSIONS),$(eval $(call SMVERSION_template,$(smver))))


TARGET = $(EXECUTABLE)

$(TARGET):	$(OBJS) $(RCUDA_PATH)/lib/librcuda.a
	$(NVCC) $(LDFLAGS) -o $@ $(LIBPATH) $^ -lrcuda $(LIB)

%.cu.o:		%.cu $(CU_DEPS)
	$(RCUDACC) -o $@ $(CUDAINCLUDES) $(NVCCFLAGS) -c -i $<

%.cpp.o:	%.cpp $(C_DEPS)
	$(CXX) -o $@ $(CXXFLAGS) -c $<

%.c.o:	%.c $(C_DEPS)
	$(CC) -o $@ $(CFLAGS) -c $<

clean:
	rm -rf $(OBJS) *.ptx $(TARGET) *.linkinfo ./rcudatmp
