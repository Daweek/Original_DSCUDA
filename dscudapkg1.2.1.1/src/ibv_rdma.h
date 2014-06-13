#ifndef RDMA_COMMON_H
#define RDMA_COMMON_H

#include <netdb.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <rdma/rdma_cma.h>
#include <cuda_runtime_api.h>
#include "dscudadefs.h"
#include "dscudarpc.h"
#include "dscudamacros.h"

#define TEST_NZ(x) do { if ( (x)) {fprintf(stderr, "error: " #x " failed (returned non-zero)." ); exit(EXIT_FAILURE); } } while (0)
#define TEST_Z(x)  do { if (!(x)) {fprintf(stderr, "error: " #x " failed (returned zero/null)."); exit(EXIT_FAILURE); } } while (0)

#define RDMA_BUFFER_SIZE (1024 * 1024 * 32)
// static const int RDMA_BUFFER_SIZE = 1024 * 4;
#define RC_IBV_IP_PORT_BASE  (65432)
#define RC_IBV_TIMEOUT (500)  // in milli second.
#define RC_IBV_EOP (12345678)

enum msgtype_t{
    MSG_MR,   // let the peer know my MR.
    MSG_DONE, // let the peer know my RDMA is done. not used for now.
};

struct message {
    enum msgtype_t type;
    union {
        struct ibv_mr mr;
    } data;
};

enum rdma_state_t {
    STATE_INIT,
    STATE_READY,
    STATE_BUSY0,
    STATE_BUSY1,
};

typedef struct {
    struct rdma_cm_id *id;
    struct ibv_qp *qp;

    struct ibv_mr *recv_mr;
    struct ibv_mr *send_mr;
    struct ibv_mr *rdma_local_mr;
    struct ibv_mr *rdma_remote_mr;
    struct ibv_mr peer_mr;

    struct ibv_context *ibvctx;
    struct ibv_pd *pd;
    struct ibv_cq *cq;
    struct ibv_comp_channel *comp_channel;

    int connected;

    pthread_t cq_poller_thread;

    struct message *recv_msg;
    struct message *send_msg;

    char *rdma_local_region;
    char *rdma_remote_region;
    enum rdma_state_t rdma_state;
    int rdma_nreq_pending;
} IbvConnection;

typedef enum {
    RCMethodNone = 0,
    RCMethodMemcpyH2D,
    RCMethodMemcpyD2H,
    RCMethodMalloc,
    RCMethodFree,
    RCMethodGetErrorString,
    RCMethodGetDeviceProperties,
    RCMethodRuntimeGetVersion,
    RCMethodThreadSynchronize,
    RCMethodThreadExit,
    RCMethodDeviceSynchronize,
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

    // add some more methods...

    RCMethodEnd
} RCMethod;

// default method
typedef struct {
    RCMethod method;
    int payload;
} IbvHdr;

// MemcpyH2D
typedef struct {
    RCMethod method;
    size_t count;
    RCadr dstadr;
    void *srcbuf;
} IbvMemcpyH2DInvokeHdr;

typedef struct {
    RCMethod method;
    cudaError_t err;
} IbvMemcpyH2DReturnHdr;

// MemcpyD2H
typedef struct {
    RCMethod method;
    size_t count;
    RCadr srcadr;
} IbvMemcpyD2HInvokeHdr;

typedef struct {
    RCMethod method;
    cudaError_t err;
    void *dstbuf;
} IbvMemcpyD2HReturnHdr;

// Malloc
typedef struct {
    RCMethod method;
    size_t size;
} IbvMallocInvokeHdr;

typedef struct {
    RCMethod method;
    cudaError_t err;
    RCadr devAdr;
} IbvMallocReturnHdr;

// Free
typedef struct {
    RCMethod method;
    RCadr devAdr;
} IbvFreeInvokeHdr;

typedef struct {
    RCMethod method;
    cudaError_t err;
} IbvFreeReturnHdr;

// cudaGetErrorString
typedef struct {
    RCMethod method;
    int device;
    cudaError_t err;
} IbvGetErrorStringInvokeHdr;

typedef struct {
    RCMethod method;
    char *errmsg;
} IbvGetErrorStringReturnHdr;

// cudaGetDeviceProperties
typedef struct {
    RCMethod method;
    int device;
} IbvGetDevicePropertiesInvokeHdr;

typedef struct {
    RCMethod method;
    cudaError_t err;
    cudaDeviceProp prop;
} IbvGetDevicePropertiesReturnHdr;

// cudaRuntimeGetVersion
typedef struct {
    RCMethod method;
    char dummy[8];
} IbvRuntimeGetVersionInvokeHdr;

typedef struct {
    RCMethod method;
    cudaError_t err;
    int version;
} IbvRuntimeGetVersionReturnHdr;

// cudaThreadSynchronize
typedef struct {
    RCMethod method;
    char dummy[8];
} IbvThreadSynchronizeInvokeHdr;

typedef struct {
    RCMethod method;
    cudaError_t err;
} IbvThreadSynchronizeReturnHdr;

// cudaThreadExit
typedef struct {
    RCMethod method;
    char dummy[8];
} IbvThreadExitInvokeHdr;

typedef struct {
    RCMethod method;
    cudaError_t err;
} IbvThreadExitReturnHdr;

// cudaDeviceSynchronize
typedef struct {
    RCMethod method;
    char dummy[8];
} IbvDeviceSynchronizeInvokeHdr;

typedef struct {
    RCMethod method;
    cudaError_t err;
} IbvDeviceSynchronizeReturnHdr;

// dscudaMemcpyToSymbolH2D
typedef struct {
    RCMethod method;
    int moduleid;
    char symbol[RC_SNAMELEN];
    size_t count;
    size_t offset;
    void *src;
} IbvDscudaMemcpyToSymbolH2DInvokeHdr;

typedef struct {
    RCMethod method;
    cudaError_t err;
} IbvDscudaMemcpyToSymbolH2DReturnHdr;

// dscudaMemcpyToSymbolD2D
typedef struct {
    RCMethod method;
    int moduleid;
    char symbol[RC_SNAMELEN];
    size_t count;
    size_t offset;
    RCadr srcadr;
} IbvDscudaMemcpyToSymbolD2DInvokeHdr;

typedef struct {
    RCMethod method;
    cudaError_t err;
} IbvDscudaMemcpyToSymbolD2DReturnHdr;


// dscudaMemcpyFromSymbolD2H
typedef struct {
    RCMethod method;
    int moduleid;
    char symbol[RC_SNAMELEN];
    size_t count;
    size_t offset;
} IbvDscudaMemcpyFromSymbolD2HInvokeHdr;

typedef struct {
    RCMethod method;
    cudaError_t err;
    void *dst;
} IbvDscudaMemcpyFromSymbolD2HReturnHdr;

// dscudaMemcpyFromSymbolD2D
typedef struct {
    RCMethod method;
    int moduleid;
    char symbol[RC_SNAMELEN];
    size_t count;
    size_t offset;
    RCadr dstadr;
} IbvDscudaMemcpyFromSymbolD2DInvokeHdr;

typedef struct {
    RCMethod method;
    cudaError_t err;
} IbvDscudaMemcpyFromSymbolD2DReturnHdr;

// dscudaMemcpyToSymbolAsyncH2D
typedef struct {
    RCMethod method;
    int moduleid;
    char symbol[RC_SNAMELEN];
    size_t count;
    size_t offset;
    RCstream stream;
    void *src;
} IbvDscudaMemcpyToSymbolAsyncH2DInvokeHdr;

typedef struct {
    RCMethod method;
    cudaError_t err;
} IbvDscudaMemcpyToSymbolAsyncH2DReturnHdr;

// dscudaMemcpyToSymbolAsyncD2D
typedef struct {
    RCMethod method;
    int moduleid;
    char symbol[RC_SNAMELEN];
    size_t count;
    size_t offset;
    RCstream stream;
    RCadr srcadr;
} IbvDscudaMemcpyToSymbolAsyncD2DInvokeHdr;

typedef struct {
    RCMethod method;
    cudaError_t err;
} IbvDscudaMemcpyToSymbolAsyncD2DReturnHdr;


// dscudaMemcpyFromSymbolAsyncD2H
typedef struct {
    RCMethod method;
    int moduleid;
    char symbol[RC_SNAMELEN];
    size_t count;
    size_t offset;
    RCstream stream;
} IbvDscudaMemcpyFromSymbolAsyncD2HInvokeHdr;

typedef struct {
    RCMethod method;
    cudaError_t err;
    void *dst;
} IbvDscudaMemcpyFromSymbolAsyncD2HReturnHdr;

// dscudaMemcpyFromSymbolAsyncD2D
typedef struct {
    RCMethod method;
    int moduleid;
    char symbol[RC_SNAMELEN];
    size_t count;
    size_t offset;
    RCstream stream;
    RCadr dstadr;
} IbvDscudaMemcpyFromSymbolAsyncD2DInvokeHdr;

typedef struct {
    RCMethod method;
    cudaError_t err;
} IbvDscudaMemcpyFromSymbolAsyncD2DReturnHdr;


// dscudaLoadModuleReturnHdr
typedef struct {
    RCMethod method;
    uint64_t ipaddr;
    unsigned long int pid;
    char modulename[RC_KMODULENAMELEN];
    void *moduleimage;
} IbvDscudaLoadModuleInvokeHdr;

typedef struct {
    RCMethod method;
    cudaError_t err;
    int moduleid;
} IbvDscudaLoadModuleReturnHdr;

// dscudaLaunchKernel
typedef struct {
    RCMethod method;
    int moduleid;
    int kernelid;
    char kernelname[RC_KNAMELEN];
    unsigned int gdim[3];
    unsigned int bdim[3];
    unsigned int smemsize;
    RCstream stream;
    int narg;
    void *args;
} IbvDscudaLaunchKernelInvokeHdr;

typedef struct {
    RCMethod method;
    cudaError_t err;
} IbvDscudaLaunchKernelReturnHdr;




typedef struct {
    int type;
    union {
        uint64_t pointerval;
        unsigned int intval;
        float floatval;
        char customval[RC_KARGMAX];
    } val;
    unsigned int offset;
    unsigned int size;
} IbvArg;


void rdmaBuildConnection(struct rdma_cm_id *id);
void rdmaBuildParams(struct rdma_conn_param *params);
void rdmaDestroyConnection(IbvConnection *conn);
void rdmaSetOnCompletionHandler(void (*handler)(struct ibv_wc *));
void rdmaOnCompletionClient(struct ibv_wc *);
void rdmaOnCompletionServer(struct ibv_wc *);
void rdmaWaitEvent(struct rdma_event_channel *ec, int et, int (*handler)(struct rdma_cm_id *id));
void rdmaWaitReadyToKickoff(IbvConnection *conn);
void rdmaWaitReadyToDisconnect(IbvConnection *conn);
void rdmaKickoff(IbvConnection *conn, int length);
void rdmaPipelinedKickoff(IbvConnection *conn, int length, char *payload_buf, char *payload_src, int payload_size);
void rdmaSendMr(IbvConnection *conn);

#endif
