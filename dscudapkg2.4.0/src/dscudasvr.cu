#include <string>
#include <time.h>
#include <map>
#include <regex.h>
#include <cuda.h>
#include <crt/host_runtime.h>

extern "C" {
    extern void** CUDARTAPI __cudaRegisterFatBinary(void *fatCubin);

    extern void CUDARTAPI __cudaUnregisterFatBinary(void **fatCubinHandle);

#if 0 // not used
    extern void CUDARTAPI __cudaRegisterVar(void **fatCubinHandle,
                                            char  *hostVar,
                                            char  *deviceAddress,
                                            const char  *deviceName,
                                            int    ext,
                                            int    size,
                                            int    constant,
                                            int    global);

    extern void CUDARTAPI __cudaRegisterFunction(void   **fatCubinHandle,
                                                 const char    *hostFun,
                                                 char    *deviceFun,
                                                 const char    *deviceName,
                                                 int      thread_limit,
                                                 uint3   *tid,
                                                 uint3   *bid,
                                                 dim3    *bDim,
                                                 dim3    *gDim,
                                                 int     *wSize);
#endif
}


#if CUDA_VERSION >= 5000 && CUDA_VERSION <= 5500
#include <helper_cuda_drvapi.h>
#elif CUDA_VERSION < 5000
#include <cutil.h>
// remove definition of some macros which will be redefined in \"cutil_inline.h\".

#ifdef MIN
#undef MIN
#endif
#ifdef MAX
#undef MAX
#endif
#include <cutil_inline.h>

#endif

#include <netdb.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <poll.h>
#include <errno.h>
#include <pthread.h>
#include <cuda_runtime_api.h>
#include <cufft.h>
#include "dscudadefs.h"
#include "dscudamacros.h"
#include "dscuda.h"
#include "sockutil.h"
#include "ibvdefs.h"
#include "tcpdefs.h"

static std::map<std::string, void *> Symbol2AddressTable;

typedef struct {
    int valid;
    unsigned int id;
    unsigned int ipaddr;
    unsigned int pid;
    time_t loaded_time;
    char name[RC_KMODULENAMELEN];
    CUmodule handle;
    CUfunction kfunc[RC_NKFUNCMAX]; // this is not used for now.
} Module;
static Module Modulelist[RC_NKMODULEMAX] = {0};

// a list of kernel functions registered.
// possibly includes functions defined in different fatbins.
typedef struct {
    void *hostFuncPtr;
    CUfunction hostFuncHandle;
    char deviceFuncSymbol[RC_KNAMELEN];
} KfuncEntry;
static int Nkfunc = 0; // # of kernel functions registered.
static KfuncEntry Kfunc[RC_NKFUNCMAX];

// a list of global pointers registered.
// possibly includes pointers defined in different fatbins.
typedef struct {
    void *hostVar;
    CUdeviceptr dptr;
    int size;
    char symbol[RC_SNAMELEN];
} GptrEntry;
static int Ngptr = 0; // # of global pointers registered.
static GptrEntry Gptr[RC_NSYMBOLMAX];

typedef struct RCipcMem_t {
    cudaIpcMemHandle_t handle;
    void *adr; // Adress obtained by cudaIpcOpenMemHandle(&adr, handle, ...);
    time_t usedat; // Duration since adr is lastly refered to.
    RCipcMem_t *prev;
    RCipcMem_t *next;
} RCipcMem;

static void *RCipcMemRegister(cudaIpcMemHandle_t handle);
static void RCipcMemUnregister(RCipcMem *ipcmem);
static RCipcMem *RCipcMemQuery(cudaIpcMemHandle_t handle);

static RCipcMem *RCipcMemListTop = NULL;
static RCipcMem *RCipcMemListTail = NULL;

// vars used for a single cudaLaunch().
typedef struct {
    char val[RC_KARGLEN];
    int offset;
    int size;
} KparamEntry;
static int Nkparam = 0;
static KparamEntry Kparam[RC_NKARGMAX];
static dim3 Kgdim;
static dim3 Kbdim;
static int Ksmemsize = 0;
static int Kstream = 0;

static int D2Csock = -1; // socket for sideband communication to the client. inherited from the daemon.
static int TcpPort = RC_SERVER_IP_PORT;
static int Connected = 0;
static int UseIbv = 0; // use IB Verbs if set to 1. use socket by default.
static int UseGD3 = 0; // use RDMA from/to GPU to/from IB HCA.
static int Ndevice = 1;                 // # of devices in the system.
static int Devid[RC_NSERVERMAX] = {0,}; // real device ids of the ones in the system.
static int dscuDevice;                   // virtual device id of the one used in the current context.
static CUcontext dscuContext = NULL;
static int Devid2Vdevid[RC_NDEVICEMAX]; // device id conversion table from real to virtual.
static int (*RCStub[RCMethodEnd])(RCHdr *, RCHdr *);

static char Dscudapath[512];    // where DS-CUDA is installed.
static char Dscudasvrpath[512]; // where DS-CUDA server built on the client side is copied at runtime.

static void notifyIamReady(void);
static void showUsage(char *command);
static void showConf(void);
static void parseArgv(int argc, char **argv);
static cudaError_t initDscuda(void);
static cudaError_t createDscuContext(void);
static cudaError_t destroyDscuContext(void);
static void initEnv(void);
static void releaseModules(bool releaseall);
static CUresult getFunctionByName(CUfunction *kfuncp, char *kname, int moduleid);
static void getGlobalSymbol(int moduleid, char *symbolname, CUdeviceptr *dptr, size_t *size);
static int dscudaLoadModule(RCipaddr ipaddr, RCpid pid, char *mname, char *image);
static void * dscudaLaunchKernel(int moduleid, int kid, char *kname,
                                 RCdim3 gdim, RCdim3 bdim, RCsize smemsize, RCstream stream, int narg, RCArg *args);
static cudaError_t setTextureParams(CUtexref texref, RCtexture texbuf, char *texname, CUDA_ARRAY_DESCRIPTOR *descp = NULL);
static void * addressLookupByRegexp(char *restr);

#undef WARN
#define WARN(lv, fmt, args...) if (lv <= dscudaWarnLevel()) \
        fprintf(stderr, "dscudasvr[%d] : " fmt, dscudaMyServerId(), ## args);

#define SET_STUB(mthd) {                            \
        RCStub[RCMethod ## mthd] = RC ## mthd;        \
}

int
dscudaMyServerId(void)
{
    return TcpPort - RC_SERVER_IP_PORT;
}

#if 0
inline void
fatal_error(int exitcode)
{
    fprintf(stderr,
            "%s(%i) : fatal_error().\n"
            "Probably you need to restart dscudasvr.\n",
            __FILE__, __LINE__);
    exit(exitcode);
}

inline void
check_cuda_error(cudaError err)
{
    if (cudaSuccess != err) {
        fprintf(stderr,
                "%s(%i) : check_cuda_error() Runtime API error : %s.\n"
                "You may need to restart dscudasvr.\n",
                __FILE__, __LINE__, cudaGetErrorString(err));
    }
}
#else

#define fatal_error(exitcode)\
{\
    fprintf(stderr,\
            "%s(%i) : fatal_error().\n"\
            "Probably you need to restart dscudasvr.\n",\
            __FILE__, __LINE__);\
    exit(exitcode);\
}\

#define check_cuda_error(err)\
{\
    if (cudaSuccess != err) {\
        fprintf(stderr,\
                "%s(%i) : check_cuda_error() Runtime API error : %s.\n"\
                "You may need to restart dscudasvr.\n",\
                __FILE__, __LINE__, cudaGetErrorString(err));\
    }\
}
#endif


#define SETUP_PACKET_BUF(mthd)                            \
    RC ## mthd ## InvokeHdr *rpkt = (RC ## mthd ## InvokeHdr *)rpkt0; \
    RC ## mthd ## ReturnHdr *spkt = (RC ## mthd ## ReturnHdr *)spkt0; \
    int spktsize = sizeof(RC ## mthd ## ReturnHdr);                    \
    spkt->method = RCMethod ## mthd ;                                   \
    WARN(3, "cuda" #mthd "(");                                          \
    if (!dscuContext) createDscuContext();


#if TCP_ONLY
#define WAIT_READY_TO_KICKOFF(conn) /* nop */
#else
#define WAIT_READY_TO_KICKOFF(conn)             \
    if (UseIbv) {                               \
        rdmaWaitReadyToKickoff(conn);           \
    }                                           \
    else {                                      \
        /* nop */                               \
    }
#include "dscudasvr_ibv.cu"
#endif

#include "dscudasvr_tcp.cu"

/*
 * CUDA API stubs
 */

static int
RCMemcpyH2D(RCHdr *rpkt0, RCHdr *spkt0)
{
    cudaError_t err;
    SETUP_PACKET_BUF(MemcpyH2D);

    err = cudaMemcpy((void *)rpkt->dstadr, &rpkt->srcbuf, rpkt->count, cudaMemcpyHostToDevice);
    check_cuda_error(err);

    spkt->err = err;
    WARN(3, "0x%08llx, 0x%08lx, %d, %s) done.\n",
         rpkt->dstadr, (unsigned long)&rpkt->srcbuf, rpkt->count,
         dscudaMemcpyKindName(cudaMemcpyHostToDevice));

    WAIT_READY_TO_KICKOFF(IbvConn);
    return spktsize;
}

static int
RCMemcpyD2H(RCHdr *rpkt0, RCHdr *spkt0)
{
    cudaError_t err;
    SETUP_PACKET_BUF(MemcpyD2H);

    err = cudaMemcpy(&spkt->dstbuf, (void *)rpkt->srcadr, rpkt->count, cudaMemcpyDeviceToHost);
    check_cuda_error(err);
    spkt->err = err;
    WARN(3, "0x%08lx, 0x%08llx, %d, %s) done.\n",
         (unsigned long)&spkt->dstbuf, rpkt->srcadr, rpkt->count,
         dscudaMemcpyKindName(cudaMemcpyDeviceToHost));

    WAIT_READY_TO_KICKOFF(IbvConn);
    spktsize += rpkt->count;


#if 0 // !!! [debugging purpose only] destroy some part of the returning data
      // !!! in order to emulate a malfunctional GPU.
    {
        static int firstcall = 1;
        static int err_in_prev_call = 0; // avoid bad data generation in adjacent calls.
        if (firstcall) {
            firstcall = 0;
            srand48(time(NULL));
        }
        if (drand48() < 1.0/10.0 && !err_in_prev_call) {
            WARN(2, "################ bad data generatad.\n\n");
            ((int *)&spkt->dstbuf)[0] = 123;
            err_in_prev_call = 1;
        }
        else {
            err_in_prev_call = 0;
        }
    }
#endif

    return spktsize;
}

static int
RCMemcpyD2D(RCHdr *rpkt0, RCHdr *spkt0)
{
    cudaError_t err;
    SETUP_PACKET_BUF(MemcpyD2D);

    err = cudaMemcpy((void*)rpkt->dstadr, (void*)rpkt->srcadr, rpkt->count, cudaMemcpyDeviceToDevice);
    check_cuda_error(err);

    spkt->err = err;
    WARN(3, "0x%08llx, 0x%08lx, %d, %s) done.\n",
         rpkt->dstadr, rpkt->srcadr, rpkt->count,
         dscudaMemcpyKindName(cudaMemcpyDeviceToDevice));

    WAIT_READY_TO_KICKOFF(IbvConn);
    return spktsize;
}

/*
 * Look for a 'handle' opened and, if any, return its 'adr'.
 * Otherwise register a new entry and open the 'handle' to obtain its 'adr'.
 * An entry not used for a long time is automatically closed.
 */
static void *
RCipcMemRegister(cudaIpcMemHandle_t handle)
{
    cudaError_t err;
    RCipcMem *ipcmem = RCipcMemQuery(handle);

    if (ipcmem) return ipcmem->adr;

    ipcmem = (RCipcMem *)malloc(sizeof(RCipcMem));
    ipcmem->handle = handle;
    err = cudaIpcOpenMemHandle(&ipcmem->adr, handle, cudaIpcMemLazyEnablePeerAccess);
    check_cuda_error(err);
    ipcmem->usedat = time(NULL);
    ipcmem->prev = RCipcMemListTail;
    ipcmem->next = NULL;
    if (!RCipcMemListTop) { // ipcmem will be the 1st entry.
        RCipcMemListTop = ipcmem;
    }
    else {
        RCipcMemListTail->next = ipcmem;
    }
    RCipcMemListTail = ipcmem;
    WARN(3, "RCipcMemRegister: adr:0x%08llx\n", ipcmem->adr);

    return ipcmem->adr;
}

static void
RCipcMemUnregister(RCipcMem *ipcmem)
{
    cudaError_t err;

    if (ipcmem->prev) { // reconnect the linked list.
        ipcmem->prev->next = ipcmem->next;
    }
    else { // ipcmem was the 1st entry.
        RCipcMemListTop = ipcmem->next;
        if (ipcmem->next) {
            ipcmem->next->prev = NULL;
        }
    }

    if (ipcmem->next) {
        ipcmem->next->prev = ipcmem->prev;
    }
    else { // ipcmem was the last entry.
        RCipcMemListTail = ipcmem->prev;
    }

    WARN(3, "cudaIpcCloseMemHandle(0x%016llx)\n", ipcmem->adr);
    err = cudaIpcCloseMemHandle(ipcmem->adr);
    check_cuda_error(err);

    free(ipcmem);
}

static RCipcMem *
RCipcMemQuery(cudaIpcMemHandle_t handle)
{
    RCipcMem *im = RCipcMemListTop;
    RCipcMem *ipcmem = NULL;
    time_t tnow = time(NULL);

    while (im) {
        if (!memcmp(&im->handle, &handle, sizeof(handle))) { // found.
            ipcmem = im;
            im->usedat = tnow;
        }
        if (tnow - im->usedat > RC_SERVER_CACHE_LIFETIME) { // clean up an unused one.
            RCipcMemUnregister(im);
        }
        im = im->next;
    }
    return ipcmem;
}

static int
RCMemcpyLocalP2P(RCHdr *rpkt0, RCHdr *spkt0)
{
    void *srcadr;
    cudaError_t err;

    SETUP_PACKET_BUF(MemcpyLocalP2P);

    WARN(3, "dstadr:0x%08llx, shandle:0x%08llx, count:%d\n",
         rpkt->dstadr, rpkt->shandle, rpkt->count);

#if RC_CACHE_IPCMEM
    srcadr = RCipcMemRegister(rpkt->shandle);
#else
    err = cudaIpcOpenMemHandle(&srcadr, rpkt->shandle, cudaIpcMemLazyEnablePeerAccess);
    check_cuda_error(err);
#endif
    err = cudaMemcpy((void*)rpkt->dstadr, srcadr, rpkt->count, cudaMemcpyDefault);

    check_cuda_error(err);
    spkt->err = err;

#if !RC_CACHE_IPCMEM
    WARN(3, "cudaIpcCloseMemHandle(0x%016llx)\n", srcadr);
    err = cudaIpcCloseMemHandle(srcadr);
    check_cuda_error(err);
#endif

    WARN(3, "0x%08llx, 0x%08llx, %d) done.\n",
         rpkt->shandle, rpkt->dstadr, rpkt->count);

    WAIT_READY_TO_KICKOFF(IbvConn);
    return spktsize;
}

static int
RCMemset(RCHdr *rpkt0, RCHdr *spkt0)
{
    cudaError_t err;
    SETUP_PACKET_BUF(Memset);

    err = cudaMemset((void *)rpkt->devptr, rpkt->value, rpkt->count);
    check_cuda_error(err);

    spkt->err = err;
    WARN(3, "0x%08lx, %d, %d) done.\n",
         rpkt->devptr, rpkt->value, rpkt->count);

    WAIT_READY_TO_KICKOFF(IbvConn);
    return spktsize;
}

static int
RCMalloc(RCHdr *rpkt0, RCHdr *spkt0)
{
    cudaError_t err;
    SETUP_PACKET_BUF(Malloc);

    err = cudaMalloc((void**)&spkt->devAdr, rpkt->size);
    check_cuda_error(err);
    spkt->err = err;
    WARN(3, "0x%08llx, %d) done. devadr:0x%08llx\n", &spkt->devAdr, rpkt->size, spkt->devAdr);

    WAIT_READY_TO_KICKOFF(IbvConn);
    return spktsize;
}

static int
RCFree(RCHdr *rpkt0, RCHdr *spkt0)
{
    cudaError_t err;
    SETUP_PACKET_BUF(Free);

    err = cudaFree((void*)rpkt->devAdr);
    check_cuda_error(err);
    spkt->err = err;
    WARN(3, "0x%08llx) done.\n", rpkt->devAdr);

    WAIT_READY_TO_KICKOFF(IbvConn);
    return spktsize;
}

static int
RCGetErrorString(RCHdr *rpkt0, RCHdr *spkt0)
{
    SETUP_PACKET_BUF(GetErrorString);
    int len;
    const char *str;

    str = cudaGetErrorString(rpkt->err);
    strncpy((char *)&spkt->errmsg, str, 256);
    len = strlen((char *)&spkt->errmsg);
    WARN(3, "%d) errmsg:%s  done.\n", rpkt->err, &spkt->errmsg);

    WAIT_READY_TO_KICKOFF(IbvConn);
    spktsize += len;
    return spktsize;
}

static int
RCGetLastError(RCHdr *rpkt0, RCHdr *spkt0)
{
    cudaError_t err;

    SETUP_PACKET_BUF(GetLastError);

    spkt->err = cudaGetLastError();
    WARN(3, ") err:%d  done.\n", spkt->err);

    WAIT_READY_TO_KICKOFF(IbvConn);

    return spktsize;
}

static int
RCGetDeviceProperties(RCHdr *rpkt0, RCHdr *spkt0)
{
    cudaError_t err;
    SETUP_PACKET_BUF(GetDeviceProperties);

    if (1 < Ndevice) {
        WARN(0, "ibvGetDeviceProperties() cannot handle multiple devices for now. Ndevice:%d\n",
             Ndevice);
        exit(1);
    }
    err = cudaGetDeviceProperties(&spkt->prop, Devid[0]);
    check_cuda_error(err);
    spkt->err = err;
    WARN(3, "0x%08lx, %d) done.\n", (unsigned long)&spkt->prop, rpkt->device);

    WAIT_READY_TO_KICKOFF(IbvConn);
    return spktsize;
}

static int
RCRuntimeGetVersion(RCHdr *rpkt0, RCHdr *spkt0)
{
    cudaError_t err;
    SETUP_PACKET_BUF(RuntimeGetVersion);

    err = cudaRuntimeGetVersion(&spkt->version);
    check_cuda_error(err);
    spkt->err = err;
    WARN(3, "0x%08lx) done.\n", (unsigned long)&spkt->version);

    WAIT_READY_TO_KICKOFF(IbvConn);
    return spktsize;
}

static int
RCThreadSynchronize(RCHdr *rpkt0, RCHdr *spkt0)
{
    cudaError_t err;
    SETUP_PACKET_BUF(ThreadSynchronize);

    err = cudaThreadSynchronize();
    check_cuda_error(err);
    spkt->err = err;
    WARN(3, "done.\n");

    WAIT_READY_TO_KICKOFF(IbvConn);
    return spktsize;
}

static int
RCThreadExit(RCHdr *rpkt0, RCHdr *spkt0)
{
    cudaError_t err;
    SETUP_PACKET_BUF(ThreadExit);

    err = cudaThreadExit();
    check_cuda_error(err);
    spkt->err = err;
    WARN(3, "done.\n");

    WAIT_READY_TO_KICKOFF(IbvConn);
    return spktsize;
}

static int
RCDeviceSynchronize(RCHdr *rpkt0, RCHdr *spkt0)
{
    cudaError_t err;
    SETUP_PACKET_BUF(DeviceSynchronize);

    err = cudaDeviceSynchronize();
    check_cuda_error(err);
    spkt->err = err;
    WARN(3, ") done.\n");

    WAIT_READY_TO_KICKOFF(IbvConn);
    return spktsize;
}

static int
RCDscudaMemcpyToSymbolH2D(RCHdr *rpkt0, RCHdr *spkt0)
{
    cudaError_t err;
    CUdeviceptr gsptr;
    size_t gssize;

    SETUP_PACKET_BUF(DscudaMemcpyToSymbolH2D);

    getGlobalSymbol(rpkt->moduleid, rpkt->symbol, &gsptr, &gssize);
    WARN(3, "gsptr: 0x%08lx\n", gsptr);

    err = ::cudaMemcpyToSymbol((char *)gsptr, &rpkt->srcbuf, rpkt->count, rpkt->offset, cudaMemcpyHostToDevice);

    check_cuda_error(err);
    spkt->err = err;

    WARN(3, "0x%08lx, 0x%08lx, %d, %d, %s) done. module name:%s  symbol:%s\n",
         gsptr, (unsigned long)&rpkt->srcbuf, rpkt->count, rpkt->offset,
         dscudaMemcpyKindName(cudaMemcpyHostToDevice),
         Modulelist[rpkt->moduleid].name, rpkt->symbol);
    //    WARN(3, "float val: %f\n", *((float *)&rpkt->srcbuf));

    WAIT_READY_TO_KICKOFF(IbvConn);

    return spktsize;
}


static int
RCDscudaMemcpyToSymbolD2D(RCHdr *rpkt0, RCHdr *spkt0)
{
    cudaError_t err;
    CUdeviceptr gsptr;
    size_t gssize;

    SETUP_PACKET_BUF(DscudaMemcpyToSymbolD2D);

    getGlobalSymbol(rpkt->moduleid, rpkt->symbol, &gsptr, &gssize);
    err = ::cudaMemcpyToSymbol((char *)gsptr, (void *)rpkt->srcadr, rpkt->count, rpkt->offset, cudaMemcpyDeviceToDevice);
    spkt->err = err;

    WARN(3, "0x%08lx, 0x%08lx, %d, %d, %s) done. module name:%s  symbol:%s\n",
         gsptr, (unsigned long)rpkt->srcadr, rpkt->count, rpkt->offset,
         dscudaMemcpyKindName(cudaMemcpyHostToDevice),
         Modulelist[rpkt->moduleid].name, rpkt->symbol);

    WAIT_READY_TO_KICKOFF(IbvConn);

    return spktsize;
}


static int
RCDscudaMemcpyFromSymbolD2H(RCHdr *rpkt0, RCHdr *spkt0)
{
    cudaError_t err;
    CUdeviceptr gsptr;
    size_t gssize;

    SETUP_PACKET_BUF(DscudaMemcpyFromSymbolD2H);

    getGlobalSymbol(rpkt->moduleid, rpkt->symbol, &gsptr, &gssize);

    err = ::cudaMemcpyFromSymbol(&spkt->dstbuf, (char *)gsptr, rpkt->count, rpkt->offset, cudaMemcpyDeviceToHost);
    check_cuda_error(err);
    spkt->err = err;
    WARN(3, "0x%08lx, 0x%08lx, %d, %d, %s) done. module name:%s  symbol:%s\n",
         (unsigned long)&spkt->dstbuf, gsptr, rpkt->count, rpkt->offset,
         dscudaMemcpyKindName(cudaMemcpyDeviceToHost),
         Modulelist[rpkt->moduleid].name, rpkt->symbol);

    WAIT_READY_TO_KICKOFF(IbvConn);
    spktsize += rpkt->count;

    return spktsize;
}

static int
RCDscudaMemcpyFromSymbolD2D(RCHdr *rpkt0, RCHdr *spkt0)
{
    cudaError_t err;
    CUdeviceptr gsptr;
    size_t gssize;

    SETUP_PACKET_BUF(DscudaMemcpyFromSymbolD2D);

    getGlobalSymbol(rpkt->moduleid, rpkt->symbol, &gsptr, &gssize);

    err = ::cudaMemcpyFromSymbol((void *)rpkt->dstadr, (char *)gsptr, rpkt->count, rpkt->offset, cudaMemcpyDeviceToHost);
    check_cuda_error(err);
    spkt->err = err;
    WARN(3, "0x%08lx, 0x%08lx, %d, %d, %s) done. module name:%s  symbol:%s\n",
         (unsigned long)rpkt->dstadr, gsptr, rpkt->count, rpkt->offset,
         dscudaMemcpyKindName(cudaMemcpyDeviceToHost),
         Modulelist[rpkt->moduleid].name, rpkt->symbol);

    WAIT_READY_TO_KICKOFF(IbvConn);

    return spktsize;
}

static int
RCDscudaMemcpyToSymbolAsyncH2D(RCHdr *rpkt0, RCHdr *spkt0)
{
#warning fill this part in ibvDscudaMemcpyToSymbolAsyncH2D

    SETUP_PACKET_BUF(DscudaMemcpyToSymbolAsyncH2D);

#if 0
    RCbuf srcbuf;
    srcbuf.RCbuf_len = rpkt->count;
    srcbuf.RCbuf_val = (char *)&rpkt->src;
    dscudaResult *resp = dscudamemcpytosymbolasynch2did_1_svc(rpkt->moduleid,
                                                            rpkt->symbol,
                                                            srcbuf,
                                                            rpkt->count,
                                                            rpkt->offset,
                                                            rpkt->stream,
                                                            NULL);
    spkt->err = (cudaError_t)cudaSuccess;

    WAIT_READY_TO_KICKOFF(IbvConn);

#endif

    return spktsize;
}


static int
RCDscudaMemcpyToSymbolAsyncD2D(RCHdr *rpkt0, RCHdr *spkt0)
{
#warning fill this part in ibvDscudaMemcpyToSymbolAsyncD2D

    SETUP_PACKET_BUF(DscudaMemcpyToSymbolAsyncD2D);

#if 0
    dscudaResult *resp = dscudamemcpytosymbolasyncd2did_1_svc(rpkt->moduleid,
                                                            rpkt->symbol,
                                                            rpkt->srcadr,
                                                            rpkt->count,
                                                            rpkt->offset,
                                                            rpkt->stream,
                                                            NULL);
    spkt->err = (cudaError_t)cudaSuccess;

    WAIT_READY_TO_KICKOFF(IbvConn);

#endif

    return spktsize;
}


static int
RCDscudaMemcpyFromSymbolAsyncD2H(RCHdr *rpkt0, RCHdr *spkt0)
{
#warning fill this part in ibvDscudaMemcpyFromSymbolAsyncD2H

    SETUP_PACKET_BUF(DscudaMemcpyFromSymbolAsyncD2H);

#if 0
    dscudaMemcpyFromSymbolAsyncD2HResult *resp = dscudamemcpyfromsymbolasyncd2hid_1_svc(rpkt->moduleid,
                                                                                      rpkt->symbol,
                                                                                      rpkt->count,
                                                                                      rpkt->offset,
                                                                                      rpkt->stream,
                                                                                      NULL);
    spkt->err = (cudaError_t)cudaSuccess;

    WAIT_READY_TO_KICKOFF(IbvConn);
    spktsize += rpkt->count;
    memcpy(&spkt->dst, resp->buf.RCbuf_val, rpkt->count);

#endif

    return spktsize;
}

static int
RCDscudaMemcpyFromSymbolAsyncD2D(RCHdr *rpkt0, RCHdr *spkt0)
{
#warning fill this part in ibvDscudaMemcpyFromSymbolAsyncD2D

    SETUP_PACKET_BUF(DscudaMemcpyFromSymbolAsyncD2D);

#if 0
    dscudaResult *resp = dscudamemcpyfromsymbolasyncd2did_1_svc(rpkt->moduleid,
                                                            rpkt->dstadr,
                                                            rpkt->symbol,
                                                            rpkt->count,
                                                            rpkt->offset,
                                                            rpkt->stream,
                                                            NULL);
    spkt->err = (cudaError_t)cudaSuccess;

    WAIT_READY_TO_KICKOFF(IbvConn);

#endif

    return spktsize;
}


// for deprecated cuParamSetSize(), cuParamSeti() etc.
static int
RCUnpackKernelParam(CUfunction *kfuncp, int narg, RCArg *args)
{
    CUresult cuerr;
    CUfunction kfunc = *kfuncp;
    RCArg noarg;
    RCArg *argp = &noarg;
    int i;
    int ival;
    float fval;
    void *pval;

    noarg.offset = 0;
    noarg.size = 0;

    for (i = 0; i < narg; i++) {
        argp = args + i;

        switch (argp->type) {
          case dscudaArgTypeP:
            pval = (void*)&(argp->val.pointerval);
            cuerr = cuParamSetv(kfunc, argp->offset, pval, argp->size);
            if (cuerr != CUDA_SUCCESS) {
                WARN(0, "cuParamSetv(0x%08llx, %d, 0x%08llx, %d) failed. %s\n",
                     kfunc, argp->offset, pval, argp->size,
                     cudaGetErrorString((cudaError_t)cuerr));
                fatal_error(1);
            }
            break;

          case dscudaArgTypeI:
            ival = argp->val.intval;
            cuerr = cuParamSeti(kfunc, argp->offset, ival);
            if (cuerr != CUDA_SUCCESS) {
                WARN(0, "cuParamSeti(0x%08llx, %d, %d) failed. %s\n",
                     kfunc, argp->offset, ival,
                     cudaGetErrorString((cudaError_t)cuerr));
                fatal_error(1);
            }
            break;

          case dscudaArgTypeF:
            fval = argp->val.floatval;
            cuerr = cuParamSetf(kfunc, argp->offset, fval);
            if (cuerr != CUDA_SUCCESS) {
                WARN(0, "cuParamSetf(0x%08llx, %d, %f) failed. %s\n",
                     kfunc, argp->offset, fval,
                     cudaGetErrorString((cudaError_t)cuerr));
                fatal_error(1);
            }
            break;

          case dscudaArgTypeV:
            pval = argp->val.customval;
            cuerr = cuParamSetv(kfunc, argp->offset, pval, argp->size);
            if (cuerr != CUDA_SUCCESS) {
                WARN(0, "cuParamSetv(0x%08llx, %d, 0x%08llx, %d) failed. %s\n",
                     kfunc, argp->offset, pval, argp->size,
                     cudaGetErrorString((cudaError_t)cuerr));
                fatal_error(1);
            }
            break;

          default:
            WARN(0, "RCUnpackKernelParam: invalid RCargType\n", argp->type);
            fatal_error(1);
        }
    }
    return argp->offset + argp->size;
}

static int
RCDscudaLoadModule(RCHdr *rpkt0, RCHdr *spkt0)
{
    SETUP_PACKET_BUF(DscudaLoadModule);
    spkt->moduleid = dscudaLoadModule((RCipaddr)rpkt->ipaddr,
                                      (RCpid)rpkt->pid,
                                      rpkt->modulename,
                                      (char *)&rpkt->moduleimage);
    spkt->err = (cudaError_t)cudaSuccess;

    WAIT_READY_TO_KICKOFF(IbvConn);
    return spktsize;
}

static int
RCDscudaLaunchKernel(RCHdr *rpkt0, RCHdr *spkt0)
{
#warning fill this part in ibvDscudaLaunchKernel

    SETUP_PACKET_BUF(DscudaLaunchKernel);

    RCdim3 gdim, bdim;

    gdim.x = rpkt->gdim[0];
    gdim.y = rpkt->gdim[1];
    gdim.z = rpkt->gdim[2];

    bdim.x = rpkt->bdim[0];
    bdim.y = rpkt->bdim[1];
    bdim.z = rpkt->bdim[2];

    dscudaLaunchKernel(rpkt->moduleid, rpkt->kernelid, rpkt->kernelname,
                       gdim, bdim, rpkt->smemsize, rpkt->stream,
                       rpkt->narg, (RCArg *)&rpkt->args);

    spkt->err = (cudaError_t)cudaSuccess;

    WAIT_READY_TO_KICKOFF(IbvConn);

    WARN(3, ") done.\n");

    return spktsize;
}


static void
getTexRef(int moduleid, char *texname, CUtexref *texref)
{
    CUresult cuerr;
    Module *mp;

    if (moduleid < 0 || RC_NKMODULEMAX <= moduleid) {
        WARN(0, "getTexRef() : invalid module id:%d.\n", moduleid);
        fatal_error(1);
    }
    mp = Modulelist + moduleid;
    cuerr = cuModuleGetTexRef(texref, mp->handle, texname);
    if (cuerr == CUDA_SUCCESS) {
        WARN(3, "cuModuleGetTexRef(0x%08lx, 0x%08lx, %s) done."
             " modulename:%s  texname:%s  *texref:0x%08lx\n",
             texref, mp->handle, texname,
             mp->name, texname, *texref);
    }
    else {
        WARN(0, "cuModuleGetTexRef(0x%08lx, 0x%08lx, %s) failed."
             " modulename:%s  texname:%s  cuerr:%d\n",
             texref, mp->handle, texname,
             mp->name, texname, cuerr);
        fatal_error(1);
    }
}


static cudaError_t
setTextureParams(CUtexref texref, RCtexture texbuf, char *texname, CUDA_ARRAY_DESCRIPTOR *descp)
{
    cudaError_t err;
    int ncomponent, i;
    unsigned int texref_flags = 0;
    unsigned int fmt_high, fmt_low, fmt_index;

    CUarray_format fmt[] = {
        CU_AD_FORMAT_UNSIGNED_INT8,    // 00_00
        CU_AD_FORMAT_UNSIGNED_INT16,   // 00_01
        CU_AD_FORMAT_UNSIGNED_INT32,   // 00_10
        (CUarray_format)0,
        CU_AD_FORMAT_SIGNED_INT8,      // 01_00
        CU_AD_FORMAT_SIGNED_INT16,     // 01_01
        CU_AD_FORMAT_SIGNED_INT32,     // 01_10
        (CUarray_format)0,
        (CUarray_format)0,
        CU_AD_FORMAT_HALF,             // 10_01
        (CUarray_format)0,
        (CUarray_format)0,
        (CUarray_format)0,
        (CUarray_format)0,
        CU_AD_FORMAT_FLOAT,            // 11_10
        (CUarray_format)0,
    };

    // set addressmode (wrap/clamp/mirror/border)
    //
    for (i = 0; i < 3; i++) {
        err = (cudaError_t)cuTexRefSetAddressMode(texref, i, (CUaddress_mode_enum)texbuf.addressMode[i]);
        if (err != cudaSuccess) {
            check_cuda_error(err);
            return err;
        }
    }

    // set filtermode (point/linear)
    //
    err = (cudaError_t)cuTexRefSetFilterMode(texref, (CUfilter_mode_enum)texbuf.filterMode);
    if (err != cudaSuccess) {
        check_cuda_error(err);
        return err;
    }

    // set flags (integer/normalized)
    //
    if (texbuf.normalized) {
        texref_flags |= CU_TRSF_NORMALIZED_COORDINATES;
    }
    else {
        texref_flags |= CU_TRSF_READ_AS_INTEGER;
    }
    err = (cudaError_t)cuTexRefSetFlags(texref, texref_flags);
    if (err != cudaSuccess) {
        check_cuda_error(err);
        return err;
    }

    // set format (unsigned/signed/float, 32/16/8-bit)
    //
    switch (texbuf.x) {
      case 8:
        fmt_low = 0;
        break;
      case 16:
        fmt_low = 1;
        break;
      case 32:
        fmt_low = 2;
        break;
      default:
        WARN(0, "cuModuleGetTexRef() invalid channel format. texture name:%s descriptor.x:%d\n",
             texname, texbuf.x);
        err = cudaErrorInvalidValue;
        return err;
    }
    switch (texbuf.f) {
      case cudaChannelFormatKindUnsigned:
        fmt_high = 0;
        break;

      case cudaChannelFormatKindSigned:
        fmt_high = 1;
        break;

      case cudaChannelFormatKindFloat:
        fmt_high = 3;
        break;

      case cudaChannelFormatKindNone:
        WARN(0, "cuModuleGetTexRef() invalid channel format. texture name:%s descriptor.f:%s\n",
             texname, "cudaChannelFormatKindNone");
        err = cudaErrorInvalidValue;
        return err;

      default:
        WARN(0, "cuModuleGetTexRef() invalid channel format. texture name:%s descriptor.f:%s\n",
             texname, texbuf.f);
        err = cudaErrorInvalidValue;
        return err;
    }
    fmt_index = fmt_high << 2 | fmt_low;
    ncomponent = 1;
    if (texbuf.y) ncomponent = 2;
    if (texbuf.z) ncomponent = 3;
    if (texbuf.w) ncomponent = 4;
    if (descp) {
        descp->Format = fmt[fmt_index];
        descp->NumChannels = ncomponent;
    }
    WARN(4, "cuTexRefSetFormat(0x%08llx, %d, %d)\n", texref, fmt[fmt_index], ncomponent);
    err = (cudaError_t)cuTexRefSetFormat(texref, fmt[fmt_index], ncomponent);
    if (err != cudaSuccess) {
        check_cuda_error(err);
        return err;
    }

    return cudaSuccess;
}






static void
getTextureParams(RCtexture *texbufp, struct textureReference *tex, struct cudaChannelFormatDesc *desc)
{
    tex->normalized = texbufp->normalized;
    tex->filterMode = (cudaTextureFilterMode)texbufp->filterMode;
    tex->addressMode[0] = (cudaTextureAddressMode)texbufp->addressMode[0];
    tex->addressMode[1] = (cudaTextureAddressMode)texbufp->addressMode[1];
    tex->addressMode[2] = (cudaTextureAddressMode)texbufp->addressMode[2];

    desc->x = texbufp->x;
    desc->y = texbufp->y;
    desc->z = texbufp->z;
    desc->w = texbufp->w;
    desc->f = (cudaChannelFormatKind)texbufp->f;
}

static int
RCDscudaBindTexture(RCHdr *rpkt0, RCHdr *spkt0)
{
#if _DSCUDA_RUNTIME_API_LAUNCH // use runtime API

    cudaError_t err;
    char restr[RC_SNAMELEN+16];
    textureReference *tex;
    struct cudaChannelFormatDesc desc;

    SETUP_PACKET_BUF(DscudaBindTexture);

    tex = (textureReference *)Symbol2AddressTable[rpkt->texname];
    if (!tex) {
        WARN(0, "texture %s did not match with any of the symbols.\n", rpkt->texname);
        exit(1);
    }

    if (!tex) {
        sprintf(restr, "^_ZN\\w+_GLOBAL__N__.*%sE", rpkt->texname);
        tex = (textureReference *)addressLookupByRegexp(restr); // look for a texname in unnamed namespace.
        if (!tex) {
            WARN(3, "texture %s did not match with %s.\n", rpkt->texname, restr);
            sprintf(restr, "^_ZN\\w+%sE", rpkt->texname);
            tex = (textureReference *)addressLookupByRegexp(restr); // look for a texname in arbitrary namespace.
            if (!tex) {
                WARN(0, "texture %s did not match with %s.\n", rpkt->texname, restr);
                fatal_error(1);
            }
        }
    }

    WARN(4, "texture:%s  address:0x%016llx\n", rpkt->texname, tex);

    getTextureParams(&rpkt->texbuf, tex, &desc);
    err = ::cudaBindTexture(&spkt->offset, tex, (void *)rpkt->devptr, &desc, rpkt->size); // !!!

    spkt->err = err;
    WARN(3, "0x%016llx, 0x%016llx, 0x%016llx, 0x%016llx, %d) done. texref:0x%08lx  texname:%s  offset:%d\n",
         &spkt->offset, tex, rpkt->devptr, &desc, rpkt->size,
         tex, rpkt->texname, spkt->offset);

    WAIT_READY_TO_KICKOFF(IbvConn);

    return spktsize;

#else // driver API.

    cudaError_t err;
    CUtexref texref;

    SETUP_PACKET_BUF(DscudaBindTexture);

    Module *mp = Modulelist + rpkt->moduleid;

    getTexRef(rpkt->moduleid, rpkt->texname, &texref);
    err = setTextureParams(texref, rpkt->texbuf, rpkt->texname);
    if (err == cudaSuccess) {
        WARN(4, "cuTexRefSetAddress(0x%08llx, 0x%08llx, 0x%08llx, %d)\n", &spkt->offset, texref, rpkt->devptr, rpkt->size);
        err = (cudaError_t)cuTexRefSetAddress((size_t *)&spkt->offset, texref, (CUdeviceptr)(rpkt->devptr), rpkt->size);
        check_cuda_error(err);
    }
    spkt->err = err;
    WARN(3, ") done. texref:0x%08lx  module name:%s  texname:%s\n",
         texref, Modulelist[rpkt->moduleid].name, rpkt->texname);

    WAIT_READY_TO_KICKOFF(IbvConn);

    return spktsize;

#endif
}

static int
RCDscudaUnbindTexture(RCHdr *rpkt0, RCHdr *spkt0)
{
    cudaError_t err;
    CUtexref texref;

    SETUP_PACKET_BUF(DscudaUnbindTexture);

    Module *mp = Modulelist + rpkt->moduleid;

    getTexRef(rpkt->moduleid, rpkt->texname, &texref);
    err = cudaSuccess;
    spkt->err = err;
    WARN(3, ") done. texref:0x%08lx  module name:%s  texname:%s\n"
         "Note: cudaUnbindTexture() does nothing but returning cudaSuccess.\n",
         texref, Modulelist[rpkt->moduleid].name, rpkt->texname);
    WAIT_READY_TO_KICKOFF(IbvConn);

    return spktsize;
}


static int
RCCreateChannelDesc(RCHdr *rpkt0, RCHdr *spkt0)
{
    cudaError_t err;
    cudaChannelFormatDesc desc;

    SETUP_PACKET_BUF(CreateChannelDesc);

    spkt->desc = cudaCreateChannelDesc(rpkt->x, rpkt->y, rpkt->z, rpkt->w, rpkt->f);
    WARN(3, "%d, %d, %d, %d, %d) done.\n",
         rpkt->x, rpkt->y, rpkt->z, rpkt->w, rpkt->f)

    WAIT_READY_TO_KICKOFF(IbvConn);
    return spktsize;
}

static int
RCDeviceSetLimit(RCHdr *rpkt0, RCHdr *spkt0)
{
    cudaError_t err;

    SETUP_PACKET_BUF(DeviceSetLimit);

    err = cudaDeviceSetLimit(rpkt->limit, rpkt->value);
    check_cuda_error(err);
    spkt->err = err;
    WARN(3, "done.\n");

    WAIT_READY_TO_KICKOFF(IbvConn);
    return spktsize;
}

static int
RCIpcGetMemHandle(RCHdr *rpkt0, RCHdr *spkt0)
{
    cudaError_t err;
    SETUP_PACKET_BUF(IpcGetMemHandle);

    err = cudaIpcGetMemHandle((cudaIpcMemHandle_t *)&spkt->handle, (void *)rpkt->adr);
    check_cuda_error(err);
    spkt->err = err;
    WARN(3, "0x%08llx, 0x%08llx) done. handle:0x%08llx\n",
         &spkt->handle, rpkt->adr, spkt->handle);

    WAIT_READY_TO_KICKOFF(IbvConn);
    return spktsize;
}


#if 1 // CUDA 5.0 or later

static int
RCDeviceSetSharedMemConfig(RCHdr *rpkt0, RCHdr *spkt0)
{
    cudaError_t err;

    SETUP_PACKET_BUF(DeviceSetSharedMemConfig);

    switch (rpkt->config) {
      case 1:
        err = cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte);
        break;
      case 2:
        err = cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
        break;
      default:
        err = cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeDefault);
        break;
    }
    check_cuda_error(err);
    spkt->err = err;
    WARN(3, "done.\n");

    WAIT_READY_TO_KICKOFF(IbvConn);
    return spktsize;
}

#else

static int
RCDeviceSetSharedMemConfig(RCHdr *rpkt0, RCHdr *spkt0)
{
    WARN(0, "cudaDeviceSetSharedMemConfig() is not supported by CUDA 4.2 or earlier.\n");
    exit(1);
}

#endif


static void
notifyIamReady(void)
{
    char msg[] = "ready";
    if (D2Csock >= 0) {
        WARN(3, "send \"ready\" to the client.\n");
        sendMsgBySocket(D2Csock, msg);
    }
}

static int
receiveProtocolPreference(void)
{
    char msg[256], rc[64];

    if (D2Csock >= 0) {
        WARN(3, "wait for remotecall preference (\"sock\" or \"ibv\") from the client.\n");
        recvMsgBySocket(D2Csock, msg, sizeof(msg));
        sscanf(msg, "remotecall:%s", rc);
        WARN(2, "method of remote procedure call: %s\n", rc);
        if (!strncmp("ibv", rc, strlen("ibv"))) {
            return 1;
        }
        else {
            return 0;
        }
    }
    else {
        return UseIbv; // do not modify the preference.
    }
}

static void
showUsage(char *command)
{
    fprintf(stderr,
            "usage: %s [-s server_id] [-d 'deviceid'] [-p port] [-S socket]\n"
            "       (-p & -S are used by the daemon only.)\n",
            command);
}

static void
showConf(void)
{
    int i;
    char str[1024], str0[1024];

    WARN(2, "TCP port : %d (base + %d)\n", TcpPort, TcpPort - RC_SERVER_IP_PORT);
    WARN(2, "ndevice : %d\n", Ndevice);
    sprintf(str, "real device%s      :", Ndevice > 1 ? "s" : " ");
    for (i = 0; i < Ndevice; i++) {
        sprintf(str0, " %d", Devid[i]);
        strcat(str, str0);
    }
    WARN(2, "%s\n", str);
    sprintf(str, "virtual device%s   :", Ndevice > 1 ? "s" : " ");
    for (i = 0; i < Ndevice; i++) {
        sprintf(str0, " %d", Devid2Vdevid[Devid[i]]);
        strcat(str, str0);
    }
    WARN(2, "%s\n", str);
}

extern char *optarg;
extern int optind;
static void
parseArgv(int argc, char **argv)
{
    int c, ic;
    char *param = "d:hp:s:S:";
    char *num;
    char buf[256];
    int device_used[RC_NDEVICEMAX] = {0,};
    int tcpport_set = 0;
    int serverid = 0;

    while ((c = getopt(argc, argv, param)) != EOF) {
        switch (c) {
          case 'p':
            TcpPort = atoi(optarg);
            tcpport_set = 1;
            break;

          case 's':
            serverid = atoi(optarg);
            break;

          case 'S':
            D2Csock = atoi(optarg);
            break;

          case 'd':
            Ndevice = 0;
            strncpy(buf, optarg, sizeof(buf));
            num = strtok(buf, " ");
            while (num) {
                ic = atoi(num);
                if (ic < 0 || RC_NDEVICEMAX <= ic ) {
                    fprintf(stderr, "device id out of range: %d\n", ic);
                    exit(2);
                }
                if (!device_used[ic]) { // care for malformed optarg value such as "0 1 2 2 3".
                    device_used[ic] = 1;
                    Devid[Ndevice] = ic;
                    Ndevice++;
                }
                num = strtok(NULL, " ");
            }
            break;

          case 'h':
          default:
            showUsage(argv[0]);
            exit(1);
        }
    }
    if (!tcpport_set) {
        TcpPort = RC_SERVER_IP_PORT + serverid;
        WARN(3, "TCP port number not given by '-p' option. Use default (%d).\n", TcpPort);
    }
}

// should be called only once in a run.
static cudaError_t
initDscuda(void)
{
    int i;
    unsigned int flags = 0; // should always be 0.
    CUresult err;

    WARN(4, "initDscuda...\n");

    for (i = 0; i < Ndevice; i++) {
        Devid2Vdevid[Devid[i]] = i;
    }

    err = cuInit(flags);
    if (err != CUDA_SUCCESS) {
        WARN(0, "cuInit(%d) failed.\n", flags);
        exit(1);
    }
    err = (CUresult)cudaSetValidDevices(Devid, Ndevice);
    if (err != CUDA_SUCCESS) {
        WARN(0, "cudaSetValidDevices(0x%08llx, %d) failed.\n", Devid, Ndevice);
        exit(1);
    }
    dscuDevice = Devid[0];
    WARN(3, "cudaSetValidDevices(0x%08llx, %d). dscuDevice:%d\n",
         Devid, Ndevice, dscuDevice);
    WARN(4, "initDscuda done.\n");
    return (cudaError_t)err;
}

static cudaError_t
createDscuContext(void)
{
    return cudaSuccess; // !!!
    //    unsigned int flags = 0; // should always be 0.
    CUdevice dev = 0;
    CUresult err;

    err = cuDeviceGet(&dev, dscuDevice);
    if (err != CUDA_SUCCESS) {
        WARN(0, "cuDeviceGet() failed.\n");
        return (cudaError_t)err;
    }

#if 0
    err = cuCtxCreate(&dscuContext, flags, dev);
    if (err != CUDA_SUCCESS) {
        WARN(0, "cuCtxCreate() failed.\n");
        return (cudaError_t)err;
    }
#else // not used. set a dummy value not to be called repeatedly.
    dscuContext = (CUcontext)-1;
#endif

    return (cudaError_t)err;
}

static cudaError_t
destroyDscuContext(void)
{
#if 0

    CUresult cuerr;
    bool all = true;

    WARN(3, "destroyDscuContext(");
    releaseModules(all);

    cuerr = cuCtxDestroy(dscuContext);
    WARN(4, "cuCtxDestroy(0x%08llx", dscuContext);
    if (cuerr != CUDA_SUCCESS) {
        WARN(0, "cuCtxDestroy() failed.\n");
        fatal_error(1);
        return (cudaError_t)cuerr;
    }
    dscuContext = NULL;
    WARN(4, ") done.\n");
    WARN(3, ") done.\n");

#else

    dscuContext = NULL;

#endif
    return cudaSuccess;
}

static void
initEnv(void)
{
    static int firstcall = 1;
    char *env;

    if (!firstcall) return;

    firstcall = 0;

    // DSCUDA_WARNLEVEL
    env = getenv("DSCUDA_WARNLEVEL");
    if (env) {
        int tmp;
        tmp = atoi(strtok(env, " "));
        if (0 <= tmp) {
            dscudaSetWarnLevel(tmp);
        }
        WARN(1, "WarnLevel: %d\n", dscudaWarnLevel());
    }

    // DSCUDA_PATH
    env = getenv("DSCUDA_PATH");
    if (!env) {
        fprintf(stderr, "An environment variable 'DSCUDA_PATH' not set.\n");
        exit(1);
    }
    strncpy(Dscudapath, env, sizeof(Dscudapath));

    // DSCUDA_SVRPATH
    env = getenv("DSCUDA_SVRPATH");
    if (!env) {
        fprintf(stderr, "An environment variable 'DSCUDA_SVRPATH' not set.\n");
        exit(1);
    }
    strncpy(Dscudasvrpath, env, sizeof(Dscudasvrpath));

    // DSCUDA_REMOTECALL
    env = getenv("DSCUDA_REMOTECALL");
#if TCP_ONLY
    UseIbv = 0;
    WARN(2, "method of remote procedure call: TCP\n");
#else
    if (D2Csock >= 0) { // launched by daemon.
        WARN(3, "A server launched by the daemon "
             "does not use the evironment variable 'DSCUDA_REMOTECALL'.\n");
    }
    else { // launched by hand.
        if (!env) {
            fprintf(stderr, "Set an environment variable 'DSCUDA_REMOTECALL' to 'ibv' or 'sock'.\n");
            exit(1);
        }
        if (!strcmp(env, "ibv")) {
            UseIbv = 1;
            WARN(2, "method of remote procedure call: InfiniBand Verbs\n");
        }
        else if (!strcmp(env, "tcp")) {
            UseIbv = 0;
            WARN(2, "method of remote procedure call: TCP\n");
        }
        else {
            UseIbv = 0;
            WARN(2, "method of remote procedure call '%s' is not available. use TCP.\n", env);
        }
    }
#endif
    // DSCUDA_USEGPUDIRECT3
    env = getenv("DSCUDA_USEGD3");
    if (env && atoi(env)) {
        WARN(2, "Use GPU Direct ver3 (RDMA).\n");
        UseGD3 = 1;
    }
    else {
        WARN(2, "Do not use GPU Direct ver3 (RDMA).\n");
        UseGD3 = 0;
    }
}

/*
 * Unload Modules never been used for a long time.
 */
static void
releaseModules(bool releaseall = false)
{
    Module *mp;
    int i;

    for (i = 0, mp = Modulelist; i < RC_NKMODULEMAX; i++, mp++) {
        if (!mp->valid) continue;
        if (releaseall || time(NULL) - mp->loaded_time > RC_SERVER_CACHE_LIFETIME) {
            cuModuleUnload((CUmodule)mp->handle);
            mp->valid = 0;
            mp->handle = NULL;
            for (i = 0; i < RC_NKFUNCMAX; i++) {
                mp->kfunc[i] = NULL;
            }
            WARN(3, "releaseModules() unloaded a module. name:%s pid:%d ip:%s age:%d\n",
                 mp->name, mp->pid, dscudaAddrToServerIpStr(mp->ipaddr),
                 time(NULL) - mp->loaded_time);
        }
    }
}

static CUresult
getFunctionByName(CUfunction *kfuncp, char *kname, int moduleid)
{
    CUresult cuerr;
    Module *mp = Modulelist + moduleid;

    cuerr = cuModuleGetFunction(kfuncp, mp->handle, kname);
    if (cuerr == CUDA_SUCCESS) {
        WARN(3, "cuModuleGetFunction() : function '%s' found.\n", kname);
    }
    else {
        WARN(0, "cuModuleGetFunction() : function:'%s'. %s\n",
             kname, cudaGetErrorString((cudaError_t)cuerr));
	WARN(0, "moduleid:%d module valid:%d id:%d name:%s\n",
	     moduleid, mp->valid, mp->id, mp->name);
        fatal_error(1);
    }
    return cuerr;
}

static void
getGlobalSymbol(int moduleid, char *symbolname, CUdeviceptr *dptr, size_t *size)
{
#if _DSCUDA_RUNTIME_API_LAUNCH // use runtime API
    cudaError_t err;
    char restr[RC_SNAMELEN+16];
    
    void **hostptr = (void **)Symbol2AddressTable[symbolname]; // look for a raw symbolname.
    if (!hostptr) {
        sprintf(restr, "^_ZN\\w+_GLOBAL__N__.*%sE", symbolname);
        hostptr = (void **)addressLookupByRegexp(restr); // look for a symbolname in unnamed namespace.
        if (!hostptr) {
            WARN(3, "global symbol %s did not match with %s.\n", symbolname, restr);
            sprintf(restr, "^_ZN\\w+%sE", symbolname);
            hostptr = (void **)addressLookupByRegexp(restr); // look for a symbolname in arbitrary namespace.
            if (!hostptr) {
                WARN(0, "global symbol %s did not match with %s.\n", symbolname, restr);
                fatal_error(1);
            }
        }
    }
    *dptr = (CUdeviceptr)hostptr;

#else // driver API.

    CUresult cuerr;
    Module *mp;

    if (moduleid < 0 || RC_NKMODULEMAX <= moduleid) {
        WARN(0, "getGlobalSymbol() : invalid module id:%d.\n", moduleid);
        fatal_error(1);
    }
    mp = Modulelist + moduleid;
    cuerr = cuModuleGetGlobal(dptr, size, mp->handle, symbolname);
    if (cuerr == CUDA_SUCCESS) {
    WARN(3, "cuModuleGetGlobal(0x%08lx, 0x%08lx, 0x%08lx, %s) done."
	 " modulename:%s  symbolname:%s  *dptr:0x%08lx\n",
	 dptr, size, mp->handle, symbolname,
	 mp->name, symbolname, *dptr);
    }
    else {
        WARN(0, "cuModuleGetGlobal(0x%08llx, 0x%08llx, 0x%08llx, 0x%08llx) failed."
             " modulename:%s  symbolname:%s  %s\n",
             dptr, size, mp->handle, symbolname,
             mp->name, symbolname, cudaGetErrorString((cudaError_t)cuerr));
        fatal_error(1);
    }
#endif
}

static int
dscudaLoadModule(RCipaddr ipaddr, RCpid pid, char *mname, char *image)
{
    CUresult cuerr;
    Module   *mp;
    int      i;

#if RC_CACHE_MODULE
    // look for mname in the module list, which may found if the client
    // resent multiple requests for the same mname:pid:ipaddr.
    int found = 0;
    for (i = 0, mp = Modulelist; i < RC_NKMODULEMAX; i++, mp++) {
        if (!mp->valid) continue;
        if ((unsigned int)ipaddr == mp->ipaddr &&
            pid    == mp->pid &&
            !strcmp(mname, mp->name)) {
            found = 1;
            break;
        }
	WARN(4, "ip:%x  %x    pid:%d  %d    name:%s  %s\n",
	     (unsigned int)ipaddr, mp->ipaddr, pid, mp->pid, mname, mp->name);
    }

    if (found) { // module found. i.e, it's already loaded.
        WARN(3, "\n\n------------------------------------------------------------------\n"
             "dscudaloadmoduleid_1_svc() got multiple requests for\n"
             "  the same module name : %s,\n"
             "  the same process id  : %d, and\n"
             "  the same IP address  : %s,\n"
             "which means a client resent the same module twice or more.\n"
             "If you see this message too often, you may want to increase\n"
             "$dscuda/include/dscudadefs.h:RC_CLIENT_CACHE_LIFETIME\n"
             "for better performance.\n"
             "------------------------------------------------------------------\n\n",
             mname, pid, dscudaAddrToServerIpStr(ipaddr));
        WARN(3, "cuModuleLoadData() : a module found in the cache. id:%d  name:%s  age:%d\n",
             mp->id, mname, time(NULL) - mp->loaded_time);
    }
    else  // module not found in the cache. load it from image.
#endif // RC_CACHE_MODULE

    {
        for (i = 0, mp = Modulelist; i < RC_NKMODULEMAX; i++, mp++) {
            if (!mp->valid) break;
            if (i == RC_NKMODULEMAX) {
                WARN(0, "module cache is full.\n");
                fatal_error(1);
            }
        }
        mp->id = i;
        cuerr = cuModuleLoadData(&mp->handle, image);

#if 0
        if (cuerr != CUDA_SUCCESS) {

            FILE *fp = fopen("tmp_knlrcvd", "w");
            //            fprintf(fp, "%s", image);
            fclose(fp);

            WARN(0, "cuModuleLoadData() failed. cuerr:%d\n", cuerr);
            fatal_error(1);
        }
#endif

        mp->valid = 1;
        mp->ipaddr = ipaddr;
        mp->pid = pid;
        strncpy(mp->name, mname, sizeof(Modulelist[0].name));
        for (i = 0; i < RC_NKFUNCMAX; i++) {
            mp->kfunc[i] = NULL;
        }
        WARN(3, "cuModuleLoadData() : a module loaded. id:%d  name:%s\n", mp->id, mname);
    }
    mp->loaded_time = time(NULL); // (re)set the lifetime of the cache.
    releaseModules();
    return mp->id;
}

static void *
dscudaLaunchKernel(int moduleid, int kid, char *kname,
                   RCdim3 gdim, RCdim3 bdim, RCsize smemsize, RCstream stream,
                   int narg, RCArg *args)
                   
{
    static int dummyres = 123;
    int paramsize;
    CUresult cuerr;

#if !RC_SUPPORT_CONCURRENT_EXEC
    stream = 0;
#endif

    if (!dscuContext) createDscuContext();

    // load a kernel function into Module[moduleid].kfunc[kid]
    // form Module[moduleid].handle.
    if (moduleid < 0 || RC_NKMODULEMAX <= moduleid) {
        WARN(0, "dscudalaunchkernelid_1_svc() : invalid module id:%d.\n", moduleid);
        fatal_error(1);
    }

#if 1 // look inside a module for a function by name.
    CUfunction kfunc;
    getFunctionByName(&kfunc, kname, moduleid);
#else // look for a function by its ID.
    // this is faster, but not used since it would cause a problem
    // when called from a kernel function that uses C++ template.
    // in that case kid might not be unique for each instance of the template.
    Module *mp = Modulelist + moduleid;
    CUfunction kfunc = mp->kfunc[kid];
    if (!kfunc) {
        getFunctionByName(&kfunc, kname, moduleid);
        mp->kfunc[kid] = kfunc;
    }
#endif


#if 1 // implementation w/new APIs.
    int i;
    void *kparams[RC_NKARGMAX];

    if (RC_NKARGMAX < narg) {
        WARN(0, "number of arguments (=%d) exceed the limit (=%d).", narg, RC_NKARGMAX);
        exit(1);
    }

    for (i = 0; i < narg; i++) {
        kparams[i] = &(args[i].val);
    }
    cuLaunchKernel(kfunc,
                   gdim.x, gdim.y, gdim.z,
                   bdim.x, bdim.y, bdim.z,
                   smemsize,
                   (CUstream)stream,
                   (void **)kparams,
                   NULL);

#else // implementation w/depricated APIs.

    // a kernel function found.
    // now make it run.
    paramsize = RCUnpackKernelParam(&kfunc, narg, (RCArg *)args);
    cuerr = cuParamSetSize(kfunc, paramsize);
    if (cuerr != CUDA_SUCCESS) {
        WARN(0, "cuParamSetSize() failed. size:%d %s\n",
             paramsize, cudaGetErrorString((cudaError_t)cuerr));
        fatal_error(1);
    }
    WARN(5, "cuParamSetSize() done.\n");

    cuerr = cuFuncSetBlockShape(kfunc, bdim.x, bdim.y, bdim.z);
    if (cuerr != CUDA_SUCCESS) {
        WARN(0, "cuFuncSetBlockShape() failed. %s\n", cudaGetErrorString((cudaError_t)cuerr));
        fatal_error(1);
    }
    WARN(5, "cuFuncSetBlockShape() done.\n");

    if (smemsize != 0) {
        cuerr = cuFuncSetSharedSize(kfunc, smemsize);
        if (cuerr != CUDA_SUCCESS) {
            WARN(0, "cuFuncSetSharedSize() failed. %s\n", cudaGetErrorString((cudaError_t)cuerr));
            fatal_error(1);
        }
        WARN(5, "cuFuncSetSharedSize() done.\n");
    }

    if ((cudaStream_t)stream == NULL) {
        cuerr = cuLaunchGrid(kfunc, gdim.x, gdim.y);
        if (cuerr != CUDA_SUCCESS) {
            WARN(0, "cuLaunchGrid() failed. kname:%s %s\n",
                 kname, cudaGetErrorString((cudaError_t)cuerr));
            fatal_error(1);
        }
        WARN(3, "cuLaunchGrid() done. kname:%s\n", kname);
    }
    else {
        cuerr = cuLaunchGridAsync(kfunc, gdim.x, gdim.y, (cudaStream_t)stream);
        if (cuerr != CUDA_SUCCESS) {
            WARN(0, "cuLaunchGridAsync() failed. kname:%s  %s\n",
                 kname, cudaGetErrorString((cudaError_t)cuerr));
            fatal_error(1);
        }
        WARN(3, "cuLaunchGridAsync() done.  kname:%s  stream:0x%08llx\n", kname, stream);
    }

#endif

    return &dummyres; // seems necessary to return something even if it's not used by the client.
}


/*
 * Kernel Execution
 * used only by CUDA binary compatible shared library.
 */

static int
RCLaunch(RCHdr *rpkt0, RCHdr *spkt0)
{
    CUresult cuerr;
    int i;
    int paramsize = 0;
    CUfunction kfunc;
    char *kname;
    bool found = false;

    SETUP_PACKET_BUF(Launch);

    WARN(3, "cudaLaunch(\n");
    if (!dscuContext) createDscuContext();

    for (i = 0; i < Nkfunc; i++) {
        if ((void *)rpkt->func == Kfunc[i].hostFuncPtr) {
            found = true;
            break;
        }
    }
    if (!found) {
        WARN(0, "function pointer: 0x%016llx not found in Kfunc[] list. abort.\n", rpkt->func);
        exit(1);
    }
    kfunc = Kfunc[i].hostFuncHandle;
    kname = Kfunc[i].deviceFuncSymbol;

    WARN(3, "Kfunc[%d]\n", i);
    WARN(3, "  hostFuncPtr      : 0x%016llx\n", rpkt->func);
    WARN(3, "  hostFuncHandle   : 0x%016llx\n", kfunc);
    WARN(3, "  deviceFuncSymbol : %s\n", kname);

    void *valp;
    int offset, size;
    for (i = 0; i < Nkparam; i++) {
        offset = Kparam[i].offset;
        size = Kparam[i].size;
        valp = Kparam[i].val;

        cuerr = cuParamSetv(kfunc, offset, valp, size);
        WARN(3, "cuParamSetv(0x%08llx, %d, 0x%08llx, %d)\n",
             kfunc, offset, valp, size);

        if (cuerr != CUDA_SUCCESS) {
            WARN(0, "cuParamSetv(0x%08llx, %d, 0x%08llx, %d) failed. %s\n",
                 kfunc, offset, valp, size,
                 cudaGetErrorString((cudaError_t)cuerr));
            fatal_error(1);
        }
    }
    paramsize = offset + size;

    cuerr = cuParamSetSize(kfunc, paramsize);
    WARN(3, "cuParamSetSize(0x%08llx, %d)\n", kfunc, paramsize);
    if (cuerr != CUDA_SUCCESS) {
        WARN(0, "cuParamSetSize() failed. size:%d %s\n",
             paramsize, cudaGetErrorString((cudaError_t)cuerr));
        fatal_error(1);
    }
    WARN(5, "cuParamSetSize() done.\n");

    cuerr = cuFuncSetBlockShape(kfunc, Kbdim.x, Kbdim.y, Kbdim.z);
    WARN(3, "cuFuncSetBlockShape(0x%08llx, %d, %d, %d)\n", kfunc, Kbdim.x, Kbdim.y, Kbdim.z);
    if (cuerr != CUDA_SUCCESS) {
        WARN(0, "cuFuncSetBlockShape() failed. %s\n", cudaGetErrorString((cudaError_t)cuerr));
        fatal_error(1);
    }
    WARN(5, "cuFuncSetBlockShape() done.\n");

    if (Ksmemsize != 0) {
        cuerr = cuFuncSetSharedSize(kfunc, Ksmemsize);
        WARN(3, "cuFuncSetSharedSize(0x%08llx, %d)\n", kfunc, Ksmemsize);
        if (cuerr != CUDA_SUCCESS) {
            WARN(0, "cuFuncSetSharedSize() failed. %s\n", cudaGetErrorString((cudaError_t)cuerr));
            fatal_error(1);
        }
        WARN(5, "cuFuncSetSharedSize() done.\n");
    }

    cuerr = cuLaunchGrid(kfunc, Kgdim.x, Kgdim.y);
    WARN(3, "cuLaunchGrid(0x%08llx, %d, %d)\n", kfunc, Kgdim.x, Kgdim.y);
    if (cuerr != CUDA_SUCCESS) {
        WARN(0, "cuLaunchGrid() failed. kname:%s %s\n",
             kname, cudaGetErrorString((cudaError_t)cuerr));
        fatal_error(1);
    }
    WARN(3, "cuLaunchGrid() done. kname:%s\n", kname);


    Nkparam = 0;
    Ksmemsize = 0;
    Kstream = 0;

    WARN(3, ") done.\n");

    spkt->err = (cudaError_t)cudaSuccess;

    WAIT_READY_TO_KICKOFF(IbvConn);
    return spktsize;
}

static int
RCConfigureCall(RCHdr *rpkt0, RCHdr *spkt0)
{
    cudaError_t err;

    SETUP_PACKET_BUF(ConfigureCall);
    WARN(3, "cudaConfigureCall(");
    if (!dscuContext) createDscuContext();

#if _DSCUDA_RUNTIME_API_LAUNCH // use runtime API to support Dynamic Parallelism

    dim3 grid(rpkt->gdim[0], rpkt->gdim[1], rpkt->gdim[2]);
    dim3 block(rpkt->bdim[0], rpkt->bdim[1], rpkt->bdim[2]);
    err = cudaConfigureCall(grid, block, rpkt->smemsize, (cudaStream_t)rpkt->stream);
    check_cuda_error(err);

#else // driver API.

    Kgdim.x = rpkt->gdim[0];
    Kgdim.y = rpkt->gdim[1];
    Kgdim.z = rpkt->gdim[2];
    Kbdim.x = rpkt->bdim[0];
    Kbdim.y = rpkt->bdim[1];
    Kbdim.z = rpkt->bdim[2];
    Ksmemsize = rpkt->smemsize;
    Kstream = rpkt->stream;

#endif

    WARN(3, "[%d, %d, %d], [%d, %d, %d], %d, %d) done.\n",
         rpkt->gdim[0], rpkt->gdim[1], rpkt->gdim[2],
         rpkt->bdim[0], rpkt->bdim[1], rpkt->bdim[2],
         rpkt->smemsize, rpkt->stream);

    spkt->err = (cudaError_t)cudaSuccess;

    WAIT_READY_TO_KICKOFF(IbvConn);
    return spktsize;
}

static int
RCSetupArgument(RCHdr *rpkt0, RCHdr *spkt0)
{
    SETUP_PACKET_BUF(SetupArgument);

    cudaError_t err;
    CUresult cuerr;
    char *argbuf = (char *)&rpkt->argbuf;
    int size = rpkt->size;
    int offset = rpkt->offset;

    WARN(3, "cudaSetupArgument(0x%llx (0x%llx), %d, %d)\n",
         &rpkt->argbuf, *((uint64_t *)&rpkt->argbuf), size, offset);
    WARN(3, "in int: %d\n", *((int *)&rpkt->argbuf));
    WARN(3, "in float: %f\n", *((float *)&rpkt->argbuf));


    if (!dscuContext) createDscuContext();

#if _DSCUDA_RUNTIME_API_LAUNCH // use runtime API to support Dynamic Parallelism
    err = cudaSetupArgument(&rpkt->argbuf, size, offset);
    check_cuda_error(err);
    WARN(3, "done.\n");
#else // driver API.
    if (RC_KARGLEN < size) {
        WARN(0, "size of a parameter of a kernel function too large: %d\n", size);
        exit(1);
    }

    memcpy(Kparam[Nkparam].val, argbuf, size);
    Kparam[Nkparam].size = size;
    Kparam[Nkparam].offset = offset;
    Nkparam++;
#endif

    spkt->err = (cudaError_t)cudaSuccess;

    WAIT_READY_TO_KICKOFF(IbvConn);
    return spktsize;
}

/*
 * Kernel Execution
 * obtain kernel function address from a given __PRETTY_FUNCTION__ value,
 * and then passes it to cudaLaunch(adr).
 */

static int
RCDscudaLaunch(RCHdr *rpkt0, RCHdr *spkt0)
{
    static char mangler[256] = {0, };
    cudaError_t err;
    CUresult cuerr;
    char cmd[1024];
    char sname[RC_SNAMELEN];
    char cname[RC_SNAMELEN];
    char restr[RC_SNAMELEN+16];
    FILE *outpipe;
    void *adr;
    regex_t re;
    regmatch_t match[16];

    SETUP_PACKET_BUF(DscudaLaunch);

    WARN(3, "dscudaLaunch(%s\n", rpkt->prettyname);
    if (!dscuContext) createDscuContext();

    adr = (void *)rpkt->kadr;

    if (!adr) { // this is executed only once for each kernel (for better performance).

        // At 1st, look for a function name in C style (i.e. non mangled).
        // Obtain the name from rpkt->prettyname.
        sprintf(restr, "\\b(\\w+)\\(");
        WARN(3, "restr:%s\n", restr);
        if (regcomp(&re, restr, REG_EXTENDED | REG_NEWLINE) != 0) {
            WARN(0, "compilation failed for regexp:%s.\n", restr);
            fatal_error(1);
        }
        if (regexec(&re, rpkt->prettyname, 2, match, 0) == 0) {
            int so = match[1].rm_so;
            int len = match[1].rm_eo - so;
            strncpy(cname, (rpkt->prettyname) + so, len);
            cname[len] = 0;
            WARN(3, "%s Matched. symbol name obtained:%s\n", rpkt->prettyname, cname);
        }
        regfree(&re);
        adr = Symbol2AddressTable[cname];
    }

    if (!adr) {
        WARN(3, "C-style symbol %s not found. Look for C++-style one.\n", cname);

        // C-style name not found. Next look for a name in C++ style (mangled one).
        // exec 'pretty2mangled' to obtain the mangled name,
        // which is stored to sname.
        if (!mangler[0]) {
            sprintf(mangler, "%s/bin/pretty2mangled", Dscudapath);
        }
        sprintf(cmd, "echo '%s' | %s \n", rpkt->prettyname, mangler);
        outpipe = popen(cmd, "r");
        if (!outpipe) {
            perror("RCDscudaLaunch()");
            fatal_error(1);
        }
        fgets(sname, sizeof(sname), outpipe);
        pclose(outpipe);
        WARN(3, "mangled functon name body : '%s'\n", sname);

        // Note that we need care for a symbol in unnamed namespace.
        // e.g.) _ZN63_GLOBAL__N__39_tmpxft_000036d2_00000000_6_main_cpp1_ii_f28793c56vecAddEPfS0_S0_
        sprintf(restr, "^(_ZN\\w+_GLOBAL__N__.*%s.*|%s.*)", cname, sname);
        // sprintf(restr, "^%s.*", sname);

        adr = addressLookupByRegexp(restr);
    }

    WARN(3, "  address               : %016llx\n", adr);

    err = cudaLaunch(adr);
    check_cuda_error(err);

    WARN(3, ") done.\n");

    spkt->kadr = (RCadr)adr;
    spkt->err = (cudaError_t)cudaSuccess;

    WAIT_READY_TO_KICKOFF(IbvConn);
    return spktsize;
}

static void *
addressLookupByRegexp(char *restr)
{
    std::map<std::string, void *>::iterator it;
    regex_t re;
    regmatch_t match[16];
    int nmatch;
    void *adr = 0;

    WARN(3, "restr:%s\n", restr);
    if (regcomp(&re, restr, REG_EXTENDED | REG_NEWLINE) != 0) {
        WARN(0, "compilation failed for regexp:%s.\n", restr);
        fatal_error(1);
    }

    nmatch = 0;
    for (it = Symbol2AddressTable.begin(); it != Symbol2AddressTable.end(); it++) {
        if (regexec(&re, it->first.c_str(), 1, match, 0) == 0) {
            WARN(3, "%s Matched.  adr:0x%016llx\n", it->first.c_str(), it->second);
            adr = it->second;
            nmatch++;
        }
        else {
            //                printf("%s  Not matched.\n", it->first.c_str());
        }
    }
    switch (nmatch) {
      case 0:
        WARN(3, "%s did not match.\n", restr);
        break;
      case 1:
        break;
      default:
        WARN(0, "%s matched with multiple items.\n", restr);
        exit(1);
    }
    regfree(&re);
    //    WARN(3, "  address               : %016llx\n", adr);
    return adr;
}


/*
 * Undocumented APIs
 */

static int
RCDscudaRegisterFatBinary(RCHdr *rpkt0, RCHdr *spkt0)
{
    void **handle = NULL;
    SETUP_PACKET_BUF(DscudaRegisterFatBinary);

    int m = rpkt->m;
    int v = rpkt->v;
    char *symbol = rpkt->f;
    char *imgbuf = (char *)&rpkt->fatbinbuf;
    int imgsize = rpkt->count;

    if (strlen(symbol) == 0) {
        symbol = 0;
    }

    WARN(3, "RCDscudaRegisterFatBinary(%d, %d, 0x%llx, 0x%llx) done. size:%d symbol:%s\n",
         m, v, imgbuf, symbol, imgsize, symbol);

    if (!dscuContext) createDscuContext();

    static int cnt = 0;
    if (cnt < 3) { // use __cudaRegisterFatBinary() if invoked from inside libcudart.so.
        fatDeviceText_t *fdt = (fatDeviceText_t *)calloc(sizeof(fatDeviceText_t), 1);

        fdt->m = m;
        fdt->v = v;
        fdt->d = (unsigned long long *)calloc(imgsize, 1);
        memcpy(fdt->d, imgbuf, imgsize);
        fdt->f = symbol;
        handle = __cudaRegisterFatBinary(fdt);

        WARN(3, "size: 0x%llx byte, 0x%llx word\n", imgsize, imgsize / 8);

        int i;
        unsigned long long int *img = fdt->d;
        for (i = 0; i < 3 * 4; i += 4) {
            WARN(3, "%d: %016llx %016llx %016llx %016llx\n",
                 i, img[i + 0], img[i + 1], img[i + 2], img[i + 3]);
        }
    }
    else { // use cuModuleLoadFatBinary() if invoked by the constructor of the user code.

        CUresult cuerr;

        // In CUDA5.0 or earlier, this cudaMalloc() dummy call
        // makes cuModuleLoadFatBinary() work. I don't know why.
        // In CUDA5.5 this does not help.
        cudaError_t err;
        int *dum;
        err = cudaMalloc((void**)&dum, 128);
        check_cuda_error(err);

        WARN(3, "size: 0x%llx byte, 0x%llx word\n", imgsize, imgsize / 8);

        int i;
        unsigned long long int *img = (unsigned long long int *)imgbuf;
        for (i = 0; i < 3 * 4; i += 4) {
            WARN(3, "%d: %016llx %016llx %016llx %016llx\n",
                 i, img[i + 0], img[i + 1], img[i + 2], img[i + 3]);
        }

        cuerr = cuModuleLoadFatBinary((CUmodule *)&handle, imgbuf);
        if (cuerr != CUDA_SUCCESS) {
            WARN(0, "cuModuleLoadFatBinary() failed. %s\n", cudaGetErrorString((cudaError_t)cuerr));
            fatal_error(1);
        }
    }
    cnt++;


    WARN(3, "handle: 0x%llx\n", handle);
    spkt->handle = (RCadr)handle;
    spkt->err = (cudaError_t)cudaSuccess;

    WAIT_READY_TO_KICKOFF(IbvConn);
    return spktsize;
}

static int
RCDscudaUnregisterFatBinary(RCHdr *rpkt0, RCHdr *spkt0)
{
    SETUP_PACKET_BUF(DscudaUnregisterFatBinary);

    WARN(3, "dscudaUnregisterFatBinary() does nothing but returning cudaSuccess.");

    // WARN(3, "dscudaUnregisterFatBinary(\n");
    if (!dscuContext) createDscuContext();

    // __cudaUnregisterFatBinary() is called as a exit() hook,
    // and thus this process kills itself almost at the same moment.
    // Therefore no completion reply might be sent to the client,
    // causing the client wait forever.
    //
    // __cudaUnregisterFatBinary((void **)rpkt->handle); // !!!
    //
    // WARN(3, "0x%llx) done.\n", rpkt->handle);

    spkt->err = (cudaError_t)cudaSuccess;

    WAIT_READY_TO_KICKOFF(IbvConn);

    return spktsize;
}

static int
RCDscudaRegisterFunction(RCHdr *rpkt0, RCHdr *spkt0)
{
    SETUP_PACKET_BUF(DscudaRegisterFunction);

    WARN(3, "dscudaRegisterFunction(\n");
    if (!dscuContext) createDscuContext();

    CUfunction kfunchandle;
    CUresult cuerr;
    cuerr = cuModuleGetFunction(&kfunchandle, (CUmodule)rpkt->handle, rpkt->dfunc);
    if (cuerr == CUDA_SUCCESS) {
        WARN(3, "cuModuleGetFunction() : function '%s' found.\n", rpkt->dfunc);
    }
    else {
        WARN(0, "cuModuleGetFunction() : function:'%s'. %s\n",
             rpkt->dfunc, cudaGetErrorString((cudaError_t)cuerr));
        fatal_error(1);
    }

    if (sizeof(Kfunc) / sizeof(KfuncEntry) <= Nkfunc) {
        WARN(0, "the number of kernel functions exceeds the limit (=%d).\n",
             sizeof(Kfunc) / sizeof(KfuncEntry));
        exit(1);
    }

    Kfunc[Nkfunc].hostFuncPtr = (void *)rpkt->hfunc;
    Kfunc[Nkfunc].hostFuncHandle = kfunchandle;
    memcpy(Kfunc[Nkfunc].deviceFuncSymbol, rpkt->dfunc, strlen(rpkt->dfunc) + 1);
    Nkfunc++;

    WARN(3, "0x%llx, 0x%llx, %s, %s, %d) done.\n",
         rpkt->handle, rpkt->hfunc, rpkt->dfunc, rpkt->dname, rpkt->tlimit);

    spkt->err = (cudaError_t)cudaSuccess;

    WAIT_READY_TO_KICKOFF(IbvConn);
    return spktsize;
}

static int
RCDscudaRegisterVar(RCHdr *rpkt0, RCHdr *spkt0)
{
    SETUP_PACKET_BUF(DscudaRegisterVar);

    WARN(3, "dscudaRegisterVar(\n");
    if (!dscuContext) createDscuContext();

    CUdeviceptr dptr;
    size_t bytes;
    CUresult cuerr;
    cuerr = cuModuleGetGlobal(&dptr, &bytes, (CUmodule)rpkt->handle, rpkt->dvar);
    if (cuerr == CUDA_SUCCESS) {
        WARN(3, "cuModuleGetGlobal() : global pointer '%s' found.\n", rpkt->dvar);
    }
    else {
        WARN(0, "cuModuleGetGlobal() : global pointer:'%s'. %s\n",
             rpkt->dvar, cudaGetErrorString((cudaError_t)cuerr));
        fatal_error(1);
    }

    if (sizeof(Gptr) / sizeof(GptrEntry) <= Ngptr) {
        WARN(0, "the number of global pointers exceeds the limit (=%d).\n",
             sizeof(Gptr) / sizeof(GptrEntry));
        exit(1);
    }

    Gptr[Ngptr].hostVar = (void *)rpkt->hvar;
    Gptr[Ngptr].dptr = dptr;
    Gptr[Ngptr].size = bytes;
    memcpy(Gptr[Ngptr].symbol, rpkt->dvar, strlen(rpkt->dvar) + 1);
    Ngptr++;

    WARN(3, "0x%llx, 0x%llx, %s, %s) done.\n",
         rpkt->handle, rpkt->hvar, rpkt->dvar, rpkt->dname);

    spkt->err = (cudaError_t)cudaSuccess;

    WAIT_READY_TO_KICKOFF(IbvConn);
    return spktsize;
}

#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/scan.h>

static int
RCDscudaSortIntBy32BitKey(RCHdr *rpkt0, RCHdr *spkt0)
{
    SETUP_PACKET_BUF(DscudaSortIntBy32BitKey);

    WARN(3, "dscudaSortIntBy32BitKey(\n");
    if (!dscuContext) createDscuContext();

    thrust::device_ptr<int> keyBegin((int *)rpkt->key);
    thrust::device_ptr<int> keyEnd((int *)rpkt->key + rpkt->nitems);
    thrust::device_ptr<int> valueBegin((int *)rpkt->value);
    thrust::sort_by_key(keyBegin, keyEnd, valueBegin);

    spkt->err = (cudaError_t)cudaSuccess;

    WAIT_READY_TO_KICKOFF(IbvConn);
    return spktsize;
}

static int
RCDscudaSortIntBy64BitKey(RCHdr *rpkt0, RCHdr *spkt0)
{
    SETUP_PACKET_BUF(DscudaSortIntBy64BitKey);

    WARN(3, "dscudaSortIntBy64BitKey(\n");
    if (!dscuContext) createDscuContext();

    thrust::device_ptr<uint64_t> keyBegin((uint64_t *)rpkt->key);
    thrust::device_ptr<uint64_t> keyEnd((uint64_t *)rpkt->key + rpkt->nitems);
    thrust::device_ptr<int> valueBegin((int *)rpkt->value);
    thrust::sort_by_key(keyBegin, keyEnd, valueBegin);

    spkt->err = (cudaError_t)cudaSuccess;

    WAIT_READY_TO_KICKOFF(IbvConn);
    return spktsize;
}

static int
RCDscudaScanIntBy64BitKey(RCHdr *rpkt0, RCHdr *spkt0)
{
    SETUP_PACKET_BUF(DscudaScanIntBy64BitKey);

    WARN(3, "dscudaScanIntBy64BitKey(\n");
    if (!dscuContext) createDscuContext();

    thrust::device_ptr<uint64_t> keyBegin((uint64_t *)rpkt->key);
    thrust::device_ptr<uint64_t> keyEnd((uint64_t *)rpkt->key + rpkt->nitems);
    thrust::device_ptr<int> valueBegin((int *)rpkt->value);
    thrust::inclusive_scan_by_key(keyBegin, keyEnd, valueBegin, valueBegin);

    spkt->err = (cudaError_t)cudaSuccess;

    WAIT_READY_TO_KICKOFF(IbvConn);
    return spktsize;
}



static void
setupStub(void)
{
    int i;
    memset(RCStub, 0, sizeof(RCMethod) * RCMethodEnd);
    SET_STUB(MemcpyH2D);
    SET_STUB(MemcpyD2H);
    SET_STUB(MemcpyD2D);
    SET_STUB(MemcpyLocalP2P);
    SET_STUB(Memset);
    SET_STUB(Malloc);
    SET_STUB(Free);
    SET_STUB(GetErrorString);
    SET_STUB(GetLastError);
    SET_STUB(GetDeviceProperties);
    SET_STUB(RuntimeGetVersion);
    SET_STUB(ThreadSynchronize);
    SET_STUB(ThreadExit);
    SET_STUB(DeviceSynchronize);
    SET_STUB(CreateChannelDesc);
    SET_STUB(DeviceSetLimit);
    SET_STUB(DeviceSetSharedMemConfig);
    SET_STUB(IpcGetMemHandle);

    SET_STUB(DscudaMemcpyToSymbolH2D);
    SET_STUB(DscudaMemcpyToSymbolD2D);
    SET_STUB(DscudaMemcpyFromSymbolD2H);
    SET_STUB(DscudaMemcpyFromSymbolD2D);
    SET_STUB(DscudaMemcpyToSymbolAsyncH2D);
    SET_STUB(DscudaMemcpyToSymbolAsyncD2D);
    SET_STUB(DscudaMemcpyFromSymbolAsyncD2H);
    SET_STUB(DscudaMemcpyFromSymbolAsyncD2D);
    SET_STUB(DscudaLoadModule);
    SET_STUB(DscudaLaunchKernel);
    SET_STUB(DscudaBindTexture);
    SET_STUB(DscudaUnbindTexture);
    SET_STUB(DscudaLaunch);
    SET_STUB(DscudaSendP2PInit);
    SET_STUB(DscudaRecvP2PInit);
    SET_STUB(DscudaSendP2P);
    SET_STUB(DscudaRecvP2P);

    SET_STUB(Launch);
    SET_STUB(ConfigureCall);
    SET_STUB(SetupArgument);
    SET_STUB(DscudaRegisterFatBinary);
    SET_STUB(DscudaUnregisterFatBinary);
    SET_STUB(DscudaRegisterFunction);
    SET_STUB(DscudaRegisterVar);

    SET_STUB(DscudaRegisterMR);

    SET_STUB(DscudaSortIntBy32BitKey);
    SET_STUB(DscudaSortIntBy64BitKey);
    SET_STUB(DscudaScanIntBy64BitKey);

    for (i = 1; i < RCMethodEnd; i++) {
        if (RCStub[i]) continue;
        WARN(0, "setupStub: RCStub[%d] is not initialized.\n", i);
        exit(1);
    }
}

static void
createSymbol2AddressTable(char* objfile)
{
    int line;
    void *adr;
    char cmd[256];
    char linebuf[1024];
    char adrstr[32], symstr[1024];
    FILE *outpipe;

    WARN(3, "symbol2address table of file:%s\n", objfile);
    sprintf(cmd, "nm %s | awk '{if (2 < NF) {print $1, $3}}' | grep -v '_device_stub_'", objfile);
    outpipe = popen(cmd, "r");
    if (!outpipe) {
        perror("createSymbol2AddressTable()");
        exit(1);
    }
    line = 0;
    while (!feof(outpipe)) {
        fgets(linebuf, sizeof(linebuf), outpipe);
        sscanf(linebuf, "%s %s", &adrstr, &symstr);
        adr = (void *)strtoul(adrstr, (char **)NULL, 16);
        Symbol2AddressTable[symstr] = adr;
        // WARN(4, "A %s", linebuf);
        // WARN(4, "B %s %s\n", adrstr, symstr);
        WARN(4, "C %016x '%s'\n\n", adr, symstr);
        // WARN(4, "D %016x %s\n\n", Symbol2AddressTable[symstr], symstr);
        line++;
    }
    pclose(outpipe);
}

/*
 * dummy functions just to avoid error when linking to client objs.
 */
void *
dscudaAdrOfUva(void *adr)
{
    // nop
}

void
dscudaLaunchWrapper(char *key)
{
    // nop
}

cudaError_t
dscudaSortIntBy32BitKey(const int size, int *key, int *value)
{
    // nop
}

cudaError_t
dscudaSortIntBy64BitKey(const int size, uint64_t *key, int *value)
{
    // nop
}

cudaError_t
dscudaScanIntBy64BitKey(const int size, uint64_t *key, int *value)
{
    // nop
}

cudaError_t
dscudaDeviceSetSharedMemConfig(cudaSharedMemConfig config)
{
    // nop
}

void
dscudaMemcopies(void **dbufs, void **sbufs, int *counts, int ncopies)
{
    // nop
}

void
dscudaBroadcast(void **dbufs, void *sbuf, int count, int ncopies)
{
    // nop
}

int
main(int argc, char **argv)
{
    parseArgv(argc, argv);
    initEnv();
    initDscuda();
    showConf();
    createSymbol2AddressTable(argv[0]);
    UseIbv = receiveProtocolPreference();
    setupStub();
    if (UseIbv) {
#if !TCP_ONLY
        notifyIamReady();
        ibvMainLoop(NULL);
#endif
    }
    else {
        notifyIamReady();
        tcpMainLoop();
    }
    fprintf (stderr, "main loop returned.\n"); // never reached.
    exit (1);
}
