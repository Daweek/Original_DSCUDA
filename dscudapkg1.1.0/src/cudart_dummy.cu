#include <stdio.h>
#include <stdlib.h>
#include <cutil.h>
#include <cutil_inline.h>

extern "C" {
    void**
    __cudaRegisterFatBinary(void *fatCubin)
    {
        // fprintf(stderr, "my __cudaRegisterFatBinary.\n");
        return NULL;
    }

    void
    __cudaUnregisterFatBinary(void **fatCubinHandle)
    {
        // fprintf(stderr, "my __cudaUnregisterFatBinary.\n");
    }

    void
    __cudaRegisterFunction(void       **fatCubinHandle,
                           const char *hostFun,
                           char       *deviceFun,
                           const char *deviceName,
                           int        thread_limit,
                           uint3      *tid,
                           uint3      *bid,
                           dim3       *bDim,
                           dim3       *gDim,
                           int        *wSize)
    {
        // fprintf(stderr, "my __cudaRegisterFunction.\n");
    }

    void
    __cudaRegisterVar(void **fatCubinHandle,
                      char  *hostVar,
                      char  *deviceAddress,
                      const char  *deviceName,
                      int    ext,
                      int    size,
                      int    constant,
                      int    global)
    {
        // fprintf(stderr, "my __cudaRegisterVar.\n");
    }

    // this seems to be actually doing some work. cannot be a dummy?
    void
    __cudaRegisterTexture(void **fatCubinHandle,
                          const struct textureReference *hostVar,
                          const void **deviceAddress,
                          const char *deviceName,
                          int dim,       
                          int norm,      
                          int ext)

    {
        fprintf(stderr, "my __cudaRegisterTexture.\n");
    }

} // "C"
