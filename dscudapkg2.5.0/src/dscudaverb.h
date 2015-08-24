#ifndef _DSCUDAVERB_H
#define _DSCUDAVERB_H
#define DSCUDAVERB_HISTMAX_GROWSIZE (10)

typedef struct {
    int device;
} cudaSetDeviceArgs;

typedef struct {
    struct cudaDeviceProp *prop;
    int device;
} cudaGetDevicePropertiesArgs;

typedef struct {
    void **devPtr;
    size_t size;
} cudaMallocArgs;

typedef struct {
    void *dst;
    void *src;
    size_t count;
    enum cudaMemcpyKind kind;  
} cudaMemcpyArgs;

typedef struct {
    int *moduleid;
    char *symbol;
    void *src;
    size_t count;
    size_t offset;
    enum cudaMemcpyKind kind;
} cudaMemcpyToSymbolArgs;

typedef struct {
    void *devPtr;
} cudaFreeArgs;

typedef struct {
    char *srcname;
} cudaLoadModuleArgs;

typedef struct {
    int *moduleid;
    int kid;
    char *kname;
    int *gdim;
    int *bdim;
    RCsize smemsize;
    RCstream stream;
    int narg;
    RCArg *arg;
} cudaLaunchKernelArgs;

typedef struct {
    int *gdim;
    int *bdim;
    RCsize smemsize;
    RCstream stream;
} cudaConfigureCallArgs;

typedef struct {
    int size;
    int offset;
    void *arg;
} cudaSetupArgumentArgs;

typedef struct {
    int size;
    int offset;
    void *arg;
} cudaSetupArgumentOfTypePArgs;

typedef struct {
    void **kadrp;
    char *key;
} cudaDscudaLaunchWrapperArgs;

typedef struct {
    int funcID;
    void *args;
} dscudaVerbHist;


extern void dscudaVerbInit(void);
extern void dscudaVerbAddHist(int, void *);
extern void dscudaVerbClearHist(void);
extern void dscudaVerbRecallHist(void);

#endif // _DSCUDAVERB_H
