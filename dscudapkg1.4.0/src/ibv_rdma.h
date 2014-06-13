#ifndef RDMA_COMMON_H
#define RDMA_COMMON_H

#ifdef RPC_ONLY

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

#else

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

#define TEST_NZ(x) do { if ( (x)) {WARN(0, #x " failed (returned non-zero).\n" ); exit(EXIT_FAILURE); } } while (0)
#define TEST_Z(x)  do { if (!(x)) {WARN(0, #x " failed (returned zero/null).\n"); exit(EXIT_FAILURE); } } while (0)

// RDMA buffer
#define RC_NWR_PER_POST (16) // max # of work requests in a single post.
#define RC_SGE_SIZE (1024 * 1024 * 2) // size per segment.
#define RC_WR_MAX (RC_NWR_PER_POST * 16) // max # of work requests stored in QP.
#define RC_RDMA_BUF_SIZE (RC_NWR_PER_POST * RC_SGE_SIZE) // size of the rdma buf.


#if RC_RDMA_BUF_SIZE  < RC_KMODULEIMAGELEN
#error "RC_RDMA_BUF_SIZE too small."
// you can reduce RC_KMODULEIMAGELEN if you know your .ptx files are small enough.
#endif

#define RC_SERVER_IBV_CQ_SIZE (RC_WR_MAX)
#define RC_CLIENT_IBV_CQ_SIZE (65536)
//#define RC_CLIENT_IBV_CQ_SIZE (RC_WR_MAX)
#define RC_IBV_IP_PORT_BASE  (65432)
#define RC_IBV_TIMEOUT (500)  // in milli second.

struct message {
    struct ibv_mr mr[RC_NWR_PER_POST];
};

enum rdma_state_t {
    STATE_INIT,
    STATE_READY,
    STATE_BUSY,
};

typedef struct {
    // IB Verb resources.
    struct rdma_cm_id *id;
    struct ibv_qp *qp;
    struct ibv_context *ibvctx;
    struct ibv_pd *pd;
    struct ibv_cq *cq;
    struct ibv_comp_channel *comp_channel;

    // message buf.
    struct message *recv_msg;
    struct message *send_msg;

    // rdma buf.
    char *rdma_local_region;
    char *rdma_remote_region;

    // MR for message buf.
    struct ibv_mr *recv_mr;
    struct ibv_mr *send_mr;
    struct ibv_mr peer_mr[RC_NWR_PER_POST];

    // MR for rdma buf.
    struct ibv_mr *rdma_local_mr[RC_NWR_PER_POST];
    struct ibv_mr *rdma_remote_mr[RC_NWR_PER_POST];

    // misc.
    pthread_t cq_poller_thread;
    int connected;
    enum rdma_state_t rdma_state;
    int rdma_nreq_pending;
} IbvConnection;

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

// MemcpyD2D
typedef struct {
    RCMethod method;
    size_t count;
    RCadr dstadr;
    RCadr srcadr;
} IbvMemcpyD2DInvokeHdr;

typedef struct {
    RCMethod method;
    cudaError_t err;
} IbvMemcpyD2DReturnHdr;

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

void rdmaBuildConnection(struct rdma_cm_id *id, bool is_server);
void rdmaBuildParams(struct rdma_conn_param *params);
void rdmaDestroyConnection(IbvConnection *conn);
void rdmaSetOnCompletionHandler(void (*handler)(struct ibv_wc *));
void rdmaOnCompletionClient(struct ibv_wc *);
void rdmaOnCompletionServer(struct ibv_wc *);
void rdmaWaitEvent(struct rdma_event_channel *ec, rdma_cm_event_type et, int (*handler)(struct rdma_cm_id *id));
void rdmaWaitReadyToKickoff(IbvConnection *conn);
void rdmaWaitReadyToDisconnect(IbvConnection *conn);
void rdmaKickoff(IbvConnection *conn, int length);
void rdmaPipelinedKickoff(IbvConnection *conn, int length, char *payload_buf, char *payload_src, int payload_size);
void rdmaSendMr(IbvConnection *conn);

#endif // RPC_ONLY

#endif // RDMA_COMMON_H
