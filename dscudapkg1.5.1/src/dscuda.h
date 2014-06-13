#ifndef _DSCUDA_H
#define _DSCUDA_H

#include <cuda_runtime_api.h>
// #include <cutil.h> // !!!
#include <builtin_types.h>
#include <driver_types.h>
#include <cuda_texture_types.h>
#include <texture_types.h>
#include "dscudadefs.h"
#include "dscudamacros.h"
#include "ibvdefs.h"
#include "tcpdefs.h"


typedef enum {
    RCMethodNone = 0,
    RCMethodMemcpyH2D,
    RCMethodMemcpyD2H,
    RCMethodMemcpyD2D,
    RCMethodMalloc,
    RCMethodFree,
    RCMethodGetErrorString,
    RCMethodGetDeviceProperties,
    RCMethodRuntimeGetVersion,
    RCMethodThreadSynchronize,
    RCMethodThreadExit,
    RCMethodDeviceSynchronize,

    // APIs w/wrapper
    RCMethodDscudaMemcpyToSymbolH2D,
    RCMethodDscudaMemcpyToSymbolD2D,
    RCMethodDscudaMemcpyFromSymbolD2H,
    RCMethodDscudaMemcpyFromSymbolD2D,
    RCMethodDscudaMemcpyToSymbolAsyncH2D,
    RCMethodDscudaMemcpyToSymbolAsyncD2D,
    RCMethodDscudaMemcpyFromSymbolAsyncD2H,
    RCMethodDscudaMemcpyFromSymbolAsyncD2D,
    RCMethodDscudaLoadModule,
    RCMethodDscudaLaunchKernel,
    RCMethodDscudaRecvP2P,
    RCMethodDscudaSendP2P,

    // add some more methods here...

    RCMethodEnd, // end of index for remotecalls.

    // APIs w/o remotecall
    RCMethodSetDevice,

} RCMethod;

// default method
typedef struct {
    RCMethod method;
    int payload;
} RCHdr;

// MemcpyH2D
typedef struct {
    RCMethod method;
    int payload;
    size_t count;
    RCadr dstadr;
    void *srcbuf;
} RCMemcpyH2DInvokeHdr;

typedef struct {
    RCMethod method;
    int payload;
    cudaError_t err;
} RCMemcpyH2DReturnHdr;

// MemcpyD2H
typedef struct {
    RCMethod method;
    int payload;
    size_t count;
    RCadr srcadr;
} RCMemcpyD2HInvokeHdr;

typedef struct {
    RCMethod method;
    int payload;
    cudaError_t err;
    void *dstbuf;
} RCMemcpyD2HReturnHdr;

// MemcpyD2D
typedef struct {
    RCMethod method;
    int payload;
    size_t count;
    RCadr dstadr;
    RCadr srcadr;
} RCMemcpyD2DInvokeHdr;

typedef struct {
    RCMethod method;
    int payload;
    cudaError_t err;
} RCMemcpyD2DReturnHdr;

// Malloc
typedef struct {
    RCMethod method;
    int payload;
    size_t size;
} RCMallocInvokeHdr;

typedef struct {
    RCMethod method;
    int payload;
    cudaError_t err;
    RCadr devAdr;
} RCMallocReturnHdr;

// Free
typedef struct {
    RCMethod method;
    int payload;
    RCadr devAdr;
} RCFreeInvokeHdr;

typedef struct {
    RCMethod method;
    int payload;
    cudaError_t err;
} RCFreeReturnHdr;

// cudaGetErrorString
typedef struct {
    RCMethod method;
    int payload;
    int device;
    cudaError_t err;
} RCGetErrorStringInvokeHdr;

typedef struct {
    RCMethod method;
    int payload;
    char *errmsg;
} RCGetErrorStringReturnHdr;

// cudaGetDeviceProperties
typedef struct {
    RCMethod method;
    int payload;
    int device;
} RCGetDevicePropertiesInvokeHdr;

typedef struct {
    RCMethod method;
    int payload;
    cudaError_t err;
    cudaDeviceProp prop;
} RCGetDevicePropertiesReturnHdr;

// cudaRuntimeGetVersion
typedef struct {
    RCMethod method;
    int payload;
    char dummy[8];
} RCRuntimeGetVersionInvokeHdr;

typedef struct {
    RCMethod method;
    int payload;
    cudaError_t err;
    int version;
} RCRuntimeGetVersionReturnHdr;

// cudaThreadSynchronize
typedef struct {
    RCMethod method;
    int payload;
    char dummy[8];
} RCThreadSynchronizeInvokeHdr;

typedef struct {
    RCMethod method;
    int payload;
    cudaError_t err;
} RCThreadSynchronizeReturnHdr;

// cudaThreadExit
typedef struct {
    RCMethod method;
    int payload;
    char dummy[8];
} RCThreadExitInvokeHdr;

typedef struct {
    RCMethod method;
    int payload;
    cudaError_t err;
} RCThreadExitReturnHdr;

// cudaDeviceSynchronize
typedef struct {
    RCMethod method;
    int payload;
    char dummy[8];
} RCDeviceSynchronizeInvokeHdr;

typedef struct {
    RCMethod method;
    int payload;
    cudaError_t err;
} RCDeviceSynchronizeReturnHdr;

// dscudaMemcpyToSymbolH2D
typedef struct {
    RCMethod method;
    int payload;
    int moduleid;
    char symbol[RC_SNAMELEN];
    size_t count;
    size_t offset;
    void *src;
} RCDscudaMemcpyToSymbolH2DInvokeHdr;

typedef struct {
    RCMethod method;
    int payload;
    cudaError_t err;
} RCDscudaMemcpyToSymbolH2DReturnHdr;

// dscudaMemcpyToSymbolD2D
typedef struct {
    RCMethod method;
    int payload;
    int moduleid;
    char symbol[RC_SNAMELEN];
    size_t count;
    size_t offset;
    RCadr srcadr;
} RCDscudaMemcpyToSymbolD2DInvokeHdr;

typedef struct {
    RCMethod method;
    int payload;
    cudaError_t err;
} RCDscudaMemcpyToSymbolD2DReturnHdr;


// dscudaMemcpyFromSymbolD2H
typedef struct {
    RCMethod method;
    int payload;
    int moduleid;
    char symbol[RC_SNAMELEN];
    size_t count;
    size_t offset;
} RCDscudaMemcpyFromSymbolD2HInvokeHdr;

typedef struct {
    RCMethod method;
    int payload;
    cudaError_t err;
    void *dst;
} RCDscudaMemcpyFromSymbolD2HReturnHdr;

// dscudaMemcpyFromSymbolD2D
typedef struct {
    RCMethod method;
    int payload;
    int moduleid;
    char symbol[RC_SNAMELEN];
    size_t count;
    size_t offset;
    RCadr dstadr;
} RCDscudaMemcpyFromSymbolD2DInvokeHdr;

typedef struct {
    RCMethod method;
    int payload;
    cudaError_t err;
} RCDscudaMemcpyFromSymbolD2DReturnHdr;

// dscudaMemcpyToSymbolAsyncH2D
typedef struct {
    RCMethod method;
    int payload;
    int moduleid;
    char symbol[RC_SNAMELEN];
    size_t count;
    size_t offset;
    RCstream stream;
    void *src;
} RCDscudaMemcpyToSymbolAsyncH2DInvokeHdr;

typedef struct {
    RCMethod method;
    int payload;
    cudaError_t err;
} RCDscudaMemcpyToSymbolAsyncH2DReturnHdr;

// dscudaMemcpyToSymbolAsyncD2D
typedef struct {
    RCMethod method;
    int payload;
    int moduleid;
    char symbol[RC_SNAMELEN];
    size_t count;
    size_t offset;
    RCstream stream;
    RCadr srcadr;
} RCDscudaMemcpyToSymbolAsyncD2DInvokeHdr;

typedef struct {
    RCMethod method;
    int payload;
    cudaError_t err;
} RCDscudaMemcpyToSymbolAsyncD2DReturnHdr;

// dscudaMemcpyFromSymbolAsyncD2H
typedef struct {
    RCMethod method;
    int payload;
    int moduleid;
    char symbol[RC_SNAMELEN];
    size_t count;
    size_t offset;
    RCstream stream;
} RCDscudaMemcpyFromSymbolAsyncD2HInvokeHdr;

typedef struct {
    RCMethod method;
    int payload;
    cudaError_t err;
    void *dst;
} RCDscudaMemcpyFromSymbolAsyncD2HReturnHdr;

// dscudaMemcpyFromSymbolAsyncD2D
typedef struct {
    RCMethod method;
    int payload;
    int moduleid;
    char symbol[RC_SNAMELEN];
    size_t count;
    size_t offset;
    RCstream stream;
    RCadr dstadr;
} RCDscudaMemcpyFromSymbolAsyncD2DInvokeHdr;

typedef struct {
    RCMethod method;
    int payload;
    cudaError_t err;
} RCDscudaMemcpyFromSymbolAsyncD2DReturnHdr;

// dscudaLoadModuleReturnHdr
typedef struct {
    RCMethod method;
    int payload;
    unsigned long long int ipaddr;
    unsigned long int pid;
    char modulename[RC_KMODULENAMELEN];
    void *moduleimage;
} RCDscudaLoadModuleInvokeHdr;

typedef struct {
    RCMethod method;
    int payload;
    cudaError_t err;
    int moduleid;
} RCDscudaLoadModuleReturnHdr;

// dscudaLaunchKernel
typedef struct {
    RCMethod method;
    int payload;
    int moduleid;
    int kernelid;
    char kernelname[RC_KNAMELEN];
    unsigned int gdim[3];
    unsigned int bdim[3];
    unsigned int smemsize;
    RCstream stream;
    int narg;
    void *args;
} RCDscudaLaunchKernelInvokeHdr;

typedef struct {
    RCMethod method;
    int payload;
    cudaError_t err;
} RCDscudaLaunchKernelReturnHdr;

// dscudaSendP2P
typedef struct {
    RCMethod method;
    int payload;
    size_t count;
    RCadr srcadr;
    RCadr dstadr;
    unsigned int dstip;
    int port;
} RCDscudaSendP2PInvokeHdr;

typedef struct {
    RCMethod method;
    int payload;
    cudaError_t err;
} RCDscudaSendP2PReturnHdr;

// dscudaRecvP2P
typedef struct {
    RCMethod method;
    int payload;
    unsigned int srcip;
    int port;
} RCDscudaRecvP2PInvokeHdr;

typedef struct {
    RCMethod method;
    int payload;
    cudaError_t err;
} RCDscudaRecvP2PReturnHdr;

// P2P
typedef struct {
    RCMethod dummy;
    int payload;
    int size;
    size_t count;
    RCadr dstadr;
    void *srcbuf;
} P2PInvokeHdr;

typedef struct {
    RCMethod dummy;
    int payload;
    int size;
    cudaError_t err;
} P2PReturnHdr;

typedef struct {
    int type;
    union {
        unsigned long pointerval;
        unsigned int intval;
        float floatval;
        char customval[RC_KARGMAX];
    } val;
    unsigned int offset;
    unsigned int size;
} RCArg;


enum {
    RC_REMOTECALL_TYPE_TCP,
    RC_REMOTECALL_TYPE_IBV,
};

// defined in dscudautil.cu
char *dscudaMemcpyKindName(cudaMemcpyKind kind);
// const char *dscudaGetIpaddrString(unsigned int addr);
unsigned int dscudaServerNameToAddr(char *svrname);
unsigned int dscudaServerNameToDevid(char *svrname);
unsigned int dscudaServerIpStrToAddr(char *ipstr);
char *       dscudaAddrToServerIpStr(unsigned int addr);
int          dscudaAlignUp(int off, int align);
unsigned int dscudaRoundUp(unsigned int src, unsigned int by);
double       RCgetCputime(double *t0);

// defined in libdscuda.cu
void *dscudaUvaOfAdr(void *adr, int devid);
int dscudaDevidOfUva(void *adr);
void *dscudaAdrOfUva(void *adr);
int dscudaNredundancy(void);
void dscudaSetAutoVerb(int verb);
int dscudaRemoteCallType(void);
void dscudaSetErrorHandler(void (*handler)(void *), void *handler_arg);
void dscudaGetMangledFunctionName(char *name, const char *funcif, const char *ptxdata);
int *dscudaLoadModule(char *srcname, char *strdata);
void dscudaLaunchKernelWrapper(int *moduleid, int kid, char *kname,
                               int *gdim, int *bdim, RCsize smemsize, RCstream stream,
                               int narg, RCArg *arg);

cudaError_t dscudaFuncGetAttributesWrapper(int *moduleid, struct cudaFuncAttributes *attr, const char *func);

cudaError_t dscudaMemcpyToSymbolWrapper(int *moduleid, const char *symbol, const void *src,
                                       size_t count, size_t offset = 0,
                                       enum cudaMemcpyKind kind = cudaMemcpyHostToDevice);

cudaError_t dscudaMemcpyToSymbolAsyncWrapper(int *moduleid, const char *symbol, const void *src,
					    size_t count, size_t offset = 0,
					    enum cudaMemcpyKind kind = cudaMemcpyHostToDevice, cudaStream_t stream = 0);

cudaError_t dscudaMemcpyFromSymbolWrapper(int *moduleid, void *dst, const char *symbol,
					 size_t count, size_t offset = 0,
					 enum cudaMemcpyKind kind = cudaMemcpyDeviceToHost);

cudaError_t dscudaMemcpyFromSymbolAsyncWrapper(int *moduleid, void *dst, const char *symbol,
					      size_t count, size_t offset = 0,
					      enum cudaMemcpyKind kind = cudaMemcpyDeviceToHost, cudaStream_t stream = 0);


#if 0 // texture is not supported yet

cudaError_t dscudaBindTextureWrapper(int *moduleid, char *texname,
                                    size_t *offset,
                                    const struct textureReference *tex,
                                    const void *devPtr,
                                    const struct cudaChannelFormatDesc *desc,
                                    size_t size = UINT_MAX);

template<class T, int dim, enum cudaTextureReadMode readMode>
cudaError_t dscudaBindTextureWrapper(int *moduleid, char *texname,
                                    size_t *offset,
                                    const struct texture<T, dim, readMode> &tex,
                                    const void *devPtr,
                                    const struct cudaChannelFormatDesc &desc,
                                    size_t size = UINT_MAX)
{
  return cudaBindTexture(offset, &tex, devPtr, &desc, size);
}

template<class T, int dim, enum cudaTextureReadMode readMode>
cudaError_t dscudaBindTextureWrapper(int *moduleid, char *texname,
                                    size_t *offset,
                                    const struct texture<T, dim, readMode> &tex,
                                    const void *devPtr,
                                    size_t size = UINT_MAX)
{
  return cudaBindTexture(offset, tex, devPtr, tex.channelDesc, size);
}


cudaError_t dscudaBindTexture2DWrapper(int *moduleid, char *texname,
                                      size_t *offset,
                                      const struct textureReference *tex,
                                      const void *devPtr,
                                      const struct cudaChannelFormatDesc *desc,
                                      size_t width, size_t height, size_t pitch);

template<class T, int dim, enum cudaTextureReadMode readMode>
cudaError_t dscudaBindTexture2DWrapper(int *moduleid, char *texname,
                                      size_t *offset,
                                      const struct texture<T, dim, readMode> &tex,
                                      const void *devPtr,
                                      const struct cudaChannelFormatDesc &desc,
                                      size_t width, size_t height, size_t pitch)
{
    return dscudaBindTexture2DWrapper(moduleid, texname,
                                     offset, &tex, devPtr, &desc, width, height, pitch);
}

template<class T, int dim, enum cudaTextureReadMode readMode>
cudaError_t dscudaBindTexture2DWrapper(int *moduleid, char *texname,
                                      size_t *offset,
                                      const struct texture<T, dim, readMode> &tex,
                                      const void *devPtr,
                                      size_t width, size_t height, size_t pitch)
{
    return dscudaBindTexture2DWrapper(moduleid, texname,
                                     offset, &tex, devPtr, &tex.channelDesc, width, height, pitch);
}

cudaError_t dscudaBindTextureToArrayWrapper(int *moduleid, char *texname,
                                           const struct textureReference *tex,
                                           const struct cudaArray * array,
                                           const struct cudaChannelFormatDesc *desc);

template<class T, int dim, enum cudaTextureReadMode readMode>
cudaError_t dscudaBindTextureToArrayWrapper(int *moduleid, char *texname,
                                           const struct texture<T, dim, readMode> &tex,
                                           const struct cudaArray * array,
                                           const struct cudaChannelFormatDesc & desc)
{
    return dscudaBindTextureToArrayWrapper(moduleid, texname, &tex, array, &desc);
}

template<class T, int dim, enum cudaTextureReadMode readMode>
cudaError_t dscudaBindTextureToArrayWrapper(int *moduleid, char *texname,
                                           const struct texture<T, dim, readMode> &tex,
                                           const struct cudaArray * array)
{
    struct cudaChannelFormatDesc desc;
    cudaError_t err = cudaGetChannelDesc(&desc, array);
    return err == cudaSuccess ? dscudaBindTextureToArrayWrapper(moduleid, texname, &tex, array, &desc) : err;
}

#endif // texture support

#endif // _DSCUDA_H
