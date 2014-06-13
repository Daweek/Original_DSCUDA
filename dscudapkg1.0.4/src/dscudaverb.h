#ifndef _DSCUDAVERB_H
#define _DSCUDAVERB_H

typedef struct {
    int device;
} cudaSetDeviceArgsType;

typedef struct {
    struct cudaDeviceProp *prop;
    int device;
} cudaGetDevicePropertiesArgsType;

typedef struct {
    void **devPtr;
    size_t size;
} cudaMallocArgsType;

typedef struct {
    void *dst;
    const void *src;
    size_t count;
    enum cudaMemcpyKind kind;  
} cudaMemcpyArgsType;

typedef struct {
    int *moduleid;
    const char *symbol;
    const void *src;
    size_t count;
    size_t offset;
    enum cudaMemcpyKind kind;
} cudaMemcpyToSymbolArgsType;

typedef struct {
    void *devPtr;
} cudaFreeArgsType;

typedef struct {
    char *srcname;
} cudaLoadModuleArgsType;

typedef struct {
    int *moduleid;
    int kid;
    char *kname;
    RCdim3 gdim;
    RCdim3 bdim;
    RCsize smemsize;
    RCstream stream;
    RCargs args;
} rpcCudaLaunchKernelArgsType;

typedef struct {
    int *moduleid;
    int kid;
    char *kname;
    int *gdim;
    int *bdim;
    RCsize smemsize;
    RCstream stream;
    int narg;
    IbvArg *arg;
} ibvCudaLaunchKernelArgsType;

typedef union {
    cudaSetDeviceArgsType cudaSetDeviceArgs; 
    cudaGetDevicePropertiesArgsType cudaGetDevicePropertiesArgs;
    cudaMallocArgsType cudaMallocArgs;
    cudaMemcpyArgsType cudaMemcpyArgs;
    cudaMemcpyToSymbolArgsType cudaMemcpyToSymbolArgs;
    cudaFreeArgsType cudaFreeArgs;
    cudaLoadModuleArgsType cudaLoadModuleArgs;
    rpcCudaLaunchKernelArgsType rpcCudaLaunchKernelArgs;
    ibvCudaLaunchKernelArgsType ibvCudaLaunchKernelArgs;
} cudaArgs;

typedef struct {
    int funcID;
    cudaArgs args;
} verbHist;


extern void verbAddHist(int, cudaArgs);
extern void verbClearHist(void);
extern void verbRecallHist(void);

#endif // _DSCUDAVERB_H
