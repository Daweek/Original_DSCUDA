#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <rpc/rpc.h>
#include <rpc/pmap_clnt.h>
#include <cutil.h>
// remove definition of some macros which will be redefined in \"cutil_inline.h\".
#ifdef MIN
#undef MIN
#endif
#ifdef MAX
#undef MAX
#endif
#include <cutil_inline.h>
#include <pthread.h>
#include "dscudarpc.h"
#include "dscuda.h"
#include "ibv_rdma.h"

extern void dscuda_prog_1(struct svc_req *rqstp, register SVCXPRT *transp);

typedef struct {
    int valid;
    unsigned int id;
    unsigned int ipaddr;
    unsigned int pid;
    time_t loaded_time;
    char name[256];
    CUmodule handle;
    CUfunction kfunc[RC_NKFUNCMAX]; // this is not used for now.
} Module;

static int ServerId = 0;                // dscudasvr id.
static int Ndevice = 1;                 // # of devices in the system.
static int Devid[RC_NDEVICEMAX] = {0,}; // real device ids of the ones in the system.
static int rcuDevice;                   // virtual device id of the one used in the current context.
static CUcontext rcuContext = NULL;
static int Devid2Vdevid[RC_NDEVICEMAX]; // device id conversion table from real to virtual.
static Module Modulelist[RC_NKMODULEMAX] = {0};

static int UseIbv = 0; // use IB Verbs if set to 1. use RPC by default.
static IbvConnection *IbvConn = NULL;

static int ibvMemcpyH2D(IbvHdr *rpkt0, IbvHdr *spkt0);
static int ibvMemcpyD2H(IbvHdr *rpkt0, IbvHdr *spkt0);
static int ibvMalloc(IbvHdr *rpkt0, IbvHdr *spkt0);
static int ibvFree(IbvHdr *rpkt0, IbvHdr *spkt0);
static int ibvGetErrorString(IbvHdr *rpkt0, IbvHdr *spkt0);
static int ibvGetDeviceProperties(IbvHdr *rpkt0, IbvHdr *spkt0);
static int ibvRuntimeGetVersion(IbvHdr *rpkt0, IbvHdr *spkt0);
static int ibvDeviceSynchronize(IbvHdr *rpkt0, IbvHdr *spkt0);
static int ibvDscudaMemcpyToSymbolAsyncH2D(IbvHdr *rpkt0, IbvHdr *spkt0);
static int ibvDscudaMemcpyToSymbolAsyncD2D(IbvHdr *rpkt0, IbvHdr *spkt0);
static int ibvDscudaMemcpyFromSymbolAsyncD2H(IbvHdr *rpkt0, IbvHdr *spkt0);
static int ibvDscudaMemcpyFromSymbolAsyncD2D(IbvHdr *rpkt0, IbvHdr *spkt0);
static int ibvDscudaLoadModule(IbvHdr *rpkt0, IbvHdr *spkt0);
static int ibvDscudaLaunchKernel(IbvHdr *rpkt0, IbvHdr *spkt0);
static int ibvUnpackKernelParam(CUfunction *kfuncp, int narg, IbvArg *args);
static int rpcUnpackKernelParam(CUfunction *kfuncp, RCargs *argsp);

static int (*IbvStub[RCMethodEnd])(IbvHdr *, IbvHdr *);
static cudaError_t initDscuda(void);
static cudaError_t createRcuContext(void);
static cudaError_t destroyRcuContext(void);
static void parseArgv(int argc, char **argv);
static void initEnv(void);
static void releaseModules(bool releaseall);
static CUresult getFunctionByName(CUfunction *kfuncp, char *kname, int moduleid);
static void getGlobalSymbol(int moduleid, char *symbolname, CUdeviceptr *dptr, size_t *size);
static cudaError_t setTextureParams(CUtexref texref, RCtexture texbuf, char *texname, CUDA_ARRAY_DESCRIPTOR *descp);

#undef WARN
#define WARN(lv, fmt, args...) if (lv <= dscudaWarnLevel()) fprintf(stderr, "dscudasvr[%d] : " fmt, ServerId, ## args);

#if 0
inline void
fatal_error(int exitcode)
{
    fprintf(stderr,
            "%s(%i) : fatal_error().\n"
            "Probably you need to restart dscudasvr.\n",
            __FILE__, __LINE__);
    // exit(exitcode);
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

static void
showUsage(char *command)
{
    fprintf(stderr, "usage: %s [-c cluster_id] [-d 'deviceid0 deviceid1 ...']\n",
            command);
}

static void
showConf(void)
{
    int i;
    printf("server id : %d\n", ServerId);
    printf("ndevice : %d\n", Ndevice);
    printf("real device%s      :", Ndevice > 1 ? "s" : " ");
    for (i = 0; i < Ndevice; i++) {
        printf(" %d", Devid[i]);
    }
    printf("\n");
    printf("virtual device%s   :", Ndevice > 1 ? "s" : " ");
    for (i = 0; i < Ndevice; i++) {
        printf(" %d", Devid2Vdevid[Devid[i]]);
    }
    printf("\n\n");
}

extern char *optarg;
extern int optind;
static void
parseArgv(int argc, char **argv)
{
    int c, ic;
    char* param = "c:d:h";
    char *num;
    char buf[256];
    int device_used[RC_NDEVICEMAX] = {0,};

    while ((c = getopt(argc, argv, param)) != EOF) {
        switch (c) {
          case 'c':
            ServerId = atoi(optarg);
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
}

// should be called only once in a run.
static cudaError_t
initDscuda(void)
{
    int i;
    unsigned int flags = 0; // should always be 0.
    CUresult err;

    WARN(4, "initDscuda(");

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
    WARN(3, "cudaSetValidDevices(0x%08llx, %d).\n", Devid, Ndevice);
    rcuDevice = Devid[0];
    return (cudaError_t)err;
}

static cudaError_t
createRcuContext(void)
{
    //    unsigned int flags = 0; // should always be 0.
    CUdevice dev = 0;
    CUresult err;

    err = cuDeviceGet(&dev, rcuDevice);
    if (err != CUDA_SUCCESS) {
        WARN(0, "cuDeviceGet() failed.\n");
        return (cudaError_t)err;
    }

#if 0
    err = cuCtxCreate(&rcuContext, flags, dev);
    if (err != CUDA_SUCCESS) {
        WARN(0, "cuCtxCreate() failed.\n");
        return (cudaError_t)err;
    }
#else // not used. set a dummy value not to be called repeatedly.
    rcuContext = (CUcontext)-1;
#endif

    return (cudaError_t)err;
}

static cudaError_t
destroyRcuContext(void)
{
#if 0

    CUresult cuerr;
    bool all = true;

    WARN(3, "destroyRcuContext(");
    releaseModules(all);

    cuerr = cuCtxDestroy(rcuContext);
    WARN(4, "cuCtxDestroy(0x%08llx", rcuContext);
    if (cuerr != CUDA_SUCCESS) {
        WARN(0, "cuCtxDestroy() failed.\n");
        fatal_error(1);
        return (cudaError_t)cuerr;
    }
    rcuContext = NULL;
    WARN(4, ") done.\n");
    WARN(3, ") done.\n");

#else

    rcuContext = NULL;

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

    // DSCUDA_REMOTECALL
    env = getenv("DSCUDA_REMOTECALL");
    if (!env) {
        fprintf(stderr, "Set an environment variable 'DSCUDA_REMOTECALL' to 'ibv' or 'rpc'.\n");
        exit(1);
    }
    if (!strcmp(env, "ibv")) {
        UseIbv = 1;
        WARN(2, "method of remote procedure call: InfiniBand Verbs\n");
    }
    else if (!strcmp(env, "rpc")) {
        UseIbv = 0;
        WARN(2, "method of remote procedure call: RPC\n");
    }
    else {
        UseIbv = 0;
        WARN(2, "method of remote procedure call '%s' is not available. use RPC.\n", env);
    }
}

static int
on_connection_request(struct rdma_cm_id *id)
{
    struct rdma_conn_param cm_params;

    WARN(2, "received a connection request.\n");
    build_connection(id);
    build_params(&cm_params);
    TEST_NZ(rdma_accept(id, &cm_params));
    IbvConn = (IbvConnection *)id->context;

    return 0;
}

static int
on_connection(struct rdma_cm_id *id)
{
    WARN(2, "accepted a connection request.\n");
    ((IbvConnection *)id->context)->connected = 1;
    return 0;
}

static int
on_disconnection(struct rdma_cm_id *id)
{
    WARN(0, "going to disconnect...\n");
    ((IbvConnection *)id->context)->connected = 0;
    return 0;
}

static void *
ib_watch_disconnection_event(void *arg)
{
    struct rdma_event_channel *ec = *(struct rdma_event_channel **)arg;
    wait_event(ec, RDMA_CM_EVENT_DISCONNECTED, on_disconnection);

    return NULL;
}

static void
setupIbv(void)
{
    int i;
    int (*func)(IbvHdr *, IbvHdr *);

    IbvStub[0] = NULL;
    for (i = 1; i != RCMethodEnd; i++) {
        switch (i) {
          case RCMethodMemcpyH2D:
            func = ibvMemcpyH2D;
            break;
          case RCMethodMemcpyD2H:
            func = ibvMemcpyD2H;
            break;
          case RCMethodMalloc:
            func = ibvMalloc;
            break;
          case RCMethodFree:
            func = ibvFree;
            break;
          case RCMethodGetErrorString:
            func = ibvGetErrorString;
            break;
          case RCMethodGetDeviceProperties:
            func = ibvGetDeviceProperties;
            break;
          case RCMethodRuntimeGetVersion:
            func = ibvRuntimeGetVersion;
            break;
          case RCMethodDeviceSynchronize:
            func = ibvDeviceSynchronize;
            break;
          case RCMethodDscudaMemcpyToSymbolAsyncH2D:
            func = ibvDscudaMemcpyToSymbolAsyncH2D;
            break;
          case RCMethodDscudaMemcpyToSymbolAsyncD2D:
            func = ibvDscudaMemcpyToSymbolAsyncD2D;
            break;
          case RCMethodDscudaMemcpyFromSymbolAsyncD2H:
            func = ibvDscudaMemcpyFromSymbolAsyncD2H;
            break;
          case RCMethodDscudaMemcpyFromSymbolAsyncD2D:
            func = ibvDscudaMemcpyFromSymbolAsyncD2D;
            break;
          case RCMethodDscudaLoadModule:
            func = ibvDscudaLoadModule;
            break;
          case RCMethodDscudaLaunchKernel:
            func = ibvDscudaLaunchKernel;
            break;
          default:
            fprintf(stderr, "setupIbv:unknown method.\n");
            exit(1);
        }
        IbvStub[i] = func;
    }
}

static void *
ib_main_loop(void *arg)
{
    uint16_t port = 0;
    struct sockaddr_in addr;
    struct rdma_cm_id *listener = NULL;
    struct rdma_event_channel *ec = NULL;

    setupIbv();

    while (true) { // for each connection

        TEST_Z(ec = rdma_create_event_channel());
        TEST_NZ(rdma_create_id(ec, &listener, NULL, RDMA_PS_TCP));

        memset(&addr, 0, sizeof(addr));
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = 0;
        addr.sin_port = htons(RC_IBV_IP_PORT_BASE + ServerId);

        TEST_NZ(rdma_bind_addr(listener, (struct sockaddr *)&addr));
        TEST_NZ(rdma_listen(listener, 10)); // backlog=10 is arbitrary.
        set_on_completion_handler(on_completion_server);

        port = ntohs(rdma_get_src_port(listener));
        printf("listening on port %d.\n", port);

        wait_event(ec, RDMA_CM_EVENT_CONNECT_REQUEST, on_connection_request);

        // IbvConn->rdma_local/remote_region are now set.
        volatile IbvConnection *conn = IbvConn;
        IbvHdr *spkt = (IbvHdr *)conn->rdma_local_region;
        IbvHdr *rpkt = (IbvHdr *)conn->rdma_remote_region;
        rpkt->method = RCMethodNone;

        wait_event(ec, RDMA_CM_EVENT_ESTABLISHED, on_connection);

        pthread_t tid;
        TEST_NZ(pthread_create(&tid, NULL, ib_watch_disconnection_event, &ec));

        while (conn->connected) {
            if (!rpkt->method) continue; // wait a packet arrival.
            int spktsize = (IbvStub[rpkt->method])(rpkt, spkt);
            kickoff_rdma((IbvConnection *)conn, spktsize);
            rpkt->method = RCMethodNone; // prepare for the next packet arrival.
        }

        //        destroy_connection(conn);
        rdma_destroy_id(conn->id);
        rdma_destroy_id(listener);
        rdma_destroy_event_channel(ec);
        WARN(0, "disconnected.\n");

    } // for each connection

    return NULL;
}

static void
setupRpc(void)
{
    register SVCXPRT *transp;
    unsigned long int prog;

    prog = DSCUDA_PROG + ServerId;
    pmap_unset (prog, DSCUDA_VER);

#if 1 // TCP
    transp = svctcp_create(RPC_ANYSOCK, RC_BUFSIZE, RC_BUFSIZE);
    if (transp == NULL) {
        fprintf (stderr, "%s", "cannot create tcp service.");
        exit(1);
    }
    if (!svc_register(transp, prog, DSCUDA_VER, dscuda_prog_1, IPPROTO_TCP)) {
        fprintf (stderr, "unable to register (prog:0x%x DSCUDA_VER:%d, TCP).\n",
        prog, DSCUDA_VER);
        exit(1);
    }

#else // UDP

    transp = svcudp_create(RPC_ANYSOCK);
    if (transp == NULL) {
        fprintf (stderr, "%s", "cannot create udp service.");
        exit(1);
    }
    if (!svc_register(transp, prog, DSCUDA_VER, dscuda_prog_1, IPPROTO_UDP)) {
        fprintf (stderr, "%s", "unable to register (prog, DSCUDA_VER, udp).");
        exit(1);
    }

#endif
}

int
main (int argc, char **argv)
{
    parseArgv(argc, argv);
    initEnv();
    initDscuda();
    showConf();

    if (UseIbv) {
#if 0
        pthread_t tid;
        TEST_NZ(pthread_create(&tid, NULL, ib_main_loop, NULL));
        while (true) sleep(10);
#else
        ib_main_loop(NULL);
#endif
    }
    else {
        setupRpc();
        svc_run();
    }
    fprintf (stderr, "main loop returned"); // never reached.
    exit (1);
}

/*
 * Dscuda server-side counterparts for CUDA runtime APIs:
 */

/*
 * Thread Management
 */

dscudaResult *
dscudathreadexitid_1_svc(struct svc_req *sr)
{
    cudaError_t err;
    static dscudaResult res;

    WARN(3, "cudaThreadExit(\n");
    if (!rcuContext) createRcuContext();

    err = cudaThreadExit();
    check_cuda_error(err);
    res.err = err;
    WARN(3, ") done.\n");
    return &res;
}

dscudaResult *
dscudathreadsynchronizeid_1_svc(struct svc_req *sr)
{
    cudaError_t err;
    static dscudaResult res;

    WARN(3, "cudaThreadSynchronize(");
    if (!rcuContext) createRcuContext();

    err = cudaThreadSynchronize();
    check_cuda_error(err);
    res.err = err;
    WARN(3, ") done.\n");
    return &res;
}

dscudaResult *
dscudathreadsetlimitid_1_svc(int limit, RCsize value, struct svc_req *sr)
{
    cudaError_t err;
    static dscudaResult res;

    WARN(3, "cudaThreadSetLimit(");
    if (!rcuContext) createRcuContext();

    err = cudaThreadSetLimit((enum cudaLimit)limit, value);
    check_cuda_error(err);
    res.err = err;
    WARN(3, "%d, %d) done.\n", limit, value);
    return &res;
}

dscudaThreadGetLimitResult *
dscudathreadgetlimitid_1_svc(int limit, struct svc_req *sr)
{
    cudaError_t err;
    static dscudaThreadGetLimitResult res;
    size_t value;

    WARN(3, "cudaThreadGetLimit(");
    if (!rcuContext) createRcuContext();

    err = cudaThreadGetLimit(&value, (enum cudaLimit)limit);
    check_cuda_error(err);
    res.err = err;
    res.value = value;
    WARN(3, "0x%08llx, %d) done.  value:%d\n", &value, limit, value);
    return &res;
}

dscudaResult *
dscudathreadsetcacheconfigid_1_svc(int cacheConfig, struct svc_req *sr)
{
    cudaError_t err;
    static dscudaResult res;

    WARN(3, "cudaThreadSetCacheConfig(");
    if (!rcuContext) createRcuContext();

    err = cudaThreadSetCacheConfig((enum cudaFuncCache)cacheConfig);
    check_cuda_error(err);
    res.err = err;
    WARN(3, "%d) done.\n", cacheConfig);
    return &res;
}

dscudaThreadGetCacheConfigResult *
dscudathreadgetcacheconfigid_1_svc(struct svc_req *sr)
{
    cudaError_t err;
    static dscudaThreadGetCacheConfigResult res;
    int cacheConfig;

    WARN(3, "cudaThreadGetCacheConfig(");
    if (!rcuContext) createRcuContext();

    err = cudaThreadGetCacheConfig((enum cudaFuncCache *)&cacheConfig);
    check_cuda_error(err);
    res.err = err;
    res.cacheConfig = cacheConfig;
    WARN(3, "0x%08llx) done.  cacheConfig:%d\n", &cacheConfig, cacheConfig);
    return &res;
}


/*
 * Error Handling
 */

dscudaResult *
dscudagetlasterrorid_1_svc(struct svc_req *sr)
{
    cudaError_t err;
    static dscudaResult res;

    WARN(5, "cudaGetLastError(");
    if (!rcuContext) createRcuContext();

    err = cudaGetLastError();
    check_cuda_error(err);
    res.err = err;
    WARN(5, ") done.\n");
    return &res;
}

dscudaResult *
dscudapeekatlasterrorid_1_svc(struct svc_req *sr)
{
    cudaError_t err;
    static dscudaResult res;

    WARN(5, "cudaPeekAtLastError(");
    if (!rcuContext) createRcuContext();

    err = cudaPeekAtLastError();
    check_cuda_error(err);
    res.err = err;
    WARN(5, ") done.\n");
    return &res;
}

dscudaGetErrorStringResult *
dscudageterrorstringid_1_svc(int err, struct svc_req *sr)
{
    static dscudaGetErrorStringResult res;

    WARN(3, "cudaGetErrorString(");
    if (!rcuContext) createRcuContext();

    res.errmsg = (char *)cudaGetErrorString((cudaError_t)err);
    WARN(3, "%d) done.\n", err);
    return &res;
}


/*
 * Device Management
 */

dscudaGetDeviceResult *
dscudagetdeviceid_1_svc(struct svc_req *sr)
{
    cudaError_t err;
    int device;
    static dscudaGetDeviceResult res;

    WARN(3, "cudaGetDevice(");
    if (!rcuContext) createRcuContext();

    err = cudaGetDevice(&device);
    check_cuda_error(err);
    res.device = Devid2Vdevid[device];
    res.err = err;
    WARN(3, "0x%08llx) done. device:%d  virtual device:%d\n",
         (unsigned long)&device, device, res.device);
    return &res;
}

dscudaGetDeviceCountResult *
dscudagetdevicecountid_1_svc(struct svc_req *sr)
{
    int count;
    static dscudaGetDeviceCountResult res;

    WARN(3, "cudaGetDeviceCount(");

#if 0
// this returns # of devices in the system, even if the number of
// valid devices set by cudaSetValidDevices() is smaller.
    cudaError_t err;
    err = cudaGetDeviceCount(&count);
    check_cuda_error(err);
    res.count = count;
    res.err = err;
#else
    res.count = count = Ndevice;
    res.err = cudaSuccess;
#endif
    WARN(3, "0x%08llx) done. count:%d\n", (unsigned long)&count, count);
    return &res;
}

dscudaGetDevicePropertiesResult *
dscudagetdevicepropertiesid_1_svc(int device, struct svc_req *sr)
{
    cudaError_t err;
    static int firstcall = 1;
    static dscudaGetDevicePropertiesResult res;

    WARN(3, "cudaGetDeviceProperties(");

    if (firstcall) {
        firstcall = 0;
        res.prop.RCbuf_val = (char*)malloc(sizeof(cudaDeviceProp));
        res.prop.RCbuf_len = sizeof(cudaDeviceProp);
    }
    if (1 < Ndevice) {
        WARN(0, "dscudagetdevicepropertiesid_1_svc() cannot handle multiple devices for now. Ndevice:%d\n",
             Ndevice);
        exit(1);
    }
    err = cudaGetDeviceProperties((cudaDeviceProp *)res.prop.RCbuf_val, Devid[0]);
    check_cuda_error(err);
    res.err = err;
    WARN(3, "0x%08llx, %d) done.\n", (unsigned long)res.prop.RCbuf_val, Devid[0]);
    return &res;
}

dscudaDriverGetVersionResult *
dscudadrivergetversionid_1_svc(struct svc_req *sr)
{
    cudaError_t err;
    int ver;
    static dscudaDriverGetVersionResult res;

    WARN(3, "cudaDriverGetVersion(");

    if (!rcuContext) createRcuContext();

    err = cudaDriverGetVersion(&ver);
    check_cuda_error(err);
    res.ver = ver;
    res.err = err;
    WARN(3, "0x%08llx) done.\n", (unsigned long)&ver);
    return &res;
}

dscudaRuntimeGetVersionResult *
dscudaruntimegetversionid_1_svc(struct svc_req *sr)
{
    cudaError_t err;
    int ver;
    static dscudaRuntimeGetVersionResult res;

    WARN(3, "cudaRuntimeGetVersion(");

    if (!rcuContext) createRcuContext();

    err = cudaRuntimeGetVersion(&ver);
    check_cuda_error(err);
    res.ver = ver;
    res.err = err;
    WARN(3, "0x%08llx) done.\n", (unsigned long)&ver);
    return &res;
}

dscudaResult *
dscudasetdeviceid_1_svc(int device, struct svc_req *sr)
{
    cudaError_t err;
    static dscudaResult res;

    WARN(3, "cudaSetDevice(");

    if (rcuContext) destroyRcuContext();

    rcuDevice = Devid[device];
    err = createRcuContext();
    res.err = err;
    WARN(3, "%d) done.  rcuDevice: %d\n",
         device, rcuDevice);
    return &res;
}

dscudaResult *
dscudasetdeviceflagsid_1_svc(unsigned int flags, struct svc_req *sr)
{
    cudaError_t err;
    static dscudaResult res;

    WARN(3, "cudaSetDeviceFlags(");

    /* cudaSetDeviceFlags() API should be called only when
     * the device is not active, i.e., rcuContext does not exist.
     * Before invoking the API, destroy the context if valid. */
    if (rcuContext) destroyRcuContext();

    err = cudaSetDeviceFlags(flags);
    check_cuda_error(err);
    res.err = err;
    WARN(3, "0x%08x)\n", flags);

    return &res;
}

dscudaChooseDeviceResult *
dscudachoosedeviceid_1_svc(RCbuf prop, struct svc_req *sr)
{
    cudaError_t err;
    int device;
    static dscudaChooseDeviceResult res;

    WARN(3, "cudaGetDevice(");
    if (!rcuContext) createRcuContext();

    err = cudaChooseDevice(&device, (const struct cudaDeviceProp *)&prop.RCbuf_val);
    check_cuda_error(err);
    res.device = Devid2Vdevid[device];
    res.err = err;
    WARN(3, "0x%08llx) done. device:%d  virtual device:%d\n",
         (unsigned long)&device, device, res.device);
    return &res;
}


dscudaResult *
dscudadevicesynchronize_1_svc(struct svc_req *sr)
{
    cudaError_t err;
    static dscudaResult res;

    WARN(3, "cudaDeviceSynchronize(");
    if (!rcuContext) createRcuContext();

    err = cudaDeviceSynchronize();
    check_cuda_error(err);
    res.err = err;
    WARN(3, ") done.\n");

    return &res;
}

dscudaResult *
dscudadevicereset_1_svc(struct svc_req *sr)
{
    cudaError_t err;
    bool all = true;
    static dscudaResult res;

    WARN(3, "cudaDeviceReset(");
    if (!rcuContext) createRcuContext();

    err = cudaDeviceReset();
    check_cuda_error(err);
    res.err = err;
    releaseModules(all);
    WARN(3, ") done.\n");

    return &res;
}

/*
 * Stream Management
 */

dscudaStreamCreateResult *
dscudastreamcreateid_1_svc(struct svc_req *sr)
{
    static dscudaStreamCreateResult res;
    cudaError_t err;
    cudaStream_t stream;

    WARN(3, "cudaStreamCreate(");
    if (!rcuContext) createRcuContext();
    err = cudaStreamCreate(&stream);
    res.stream = (RCadr)stream;
    check_cuda_error(err);
    res.err = err;
    WARN(3, "0x%08llx) done. stream:0x%08llx\n", &stream, stream);

    return &res;
}

dscudaResult *
dscudastreamdestroyid_1_svc(RCstream stream, struct svc_req *sr)
{
    static dscudaResult res;
    cudaError_t err;

    WARN(3, "cudaStreamDestroy(");
    if (!rcuContext) createRcuContext();
    err = cudaStreamDestroy((cudaStream_t)stream);
    check_cuda_error(err);
    res.err = err;
    WARN(3, "0x%08llx) done.\n", stream);

    return &res;
}

dscudaResult *
dscudastreamwaiteventid_1_svc(RCstream stream, RCevent event, unsigned int flags, struct svc_req *sr)
{
    static dscudaResult res;
    cudaError_t err;

    WARN(3, "cudaStreamWaitEvent(");
    if (!rcuContext) createRcuContext();
    err = cudaStreamWaitEvent((cudaStream_t)stream, (cudaEvent_t)event, flags);
    check_cuda_error(err);
    res.err = err;
    WARN(3, "0x%08llx 0x%08llx, 0x%08x) done.\n", stream, event, flags);

    return &res;
}

dscudaResult *
dscudastreamsynchronizeid_1_svc(RCstream stream, struct svc_req *sr)
{
    static dscudaResult res;
    cudaError_t err;

    WARN(3, "cudaStreamSynchronize(");
    if (!rcuContext) createRcuContext();
    err = cudaStreamSynchronize((cudaStream_t)stream);
    check_cuda_error(err);
    res.err = err;
    WARN(3, "0x%08llx) done.\n", stream);

    return &res;
}

dscudaResult *
dscudastreamqueryid_1_svc(RCstream stream, struct svc_req *sr)
{
    static dscudaResult res;
    cudaError_t err;

    WARN(3, "cudaStreamQuery(");
    if (!rcuContext) createRcuContext();
    err = cudaStreamQuery((cudaStream_t)stream);
    // should not check error due to the nature of this API.
    // check_cuda_error(err);
    res.err = err;
    WARN(3, "0x%08llx) done.\n", stream);

    return &res;
}

/*
 * Event Management
 */

dscudaEventCreateResult *
dscudaeventcreateid_1_svc(struct svc_req *sr)
{
    static dscudaEventCreateResult res;
    cudaError_t err;
    cudaEvent_t event;

    WARN(3, "cudaEventCreate(");
    if (!rcuContext) createRcuContext();
    err = cudaEventCreate(&event);
    res.event = (RCadr)event;
    check_cuda_error(err);
    res.err = err;
    WARN(3, "0x%08llx) done. event:0x%08llx\n", &event, event);

    return &res;
}

dscudaEventCreateResult *
dscudaeventcreatewithflagsid_1_svc(unsigned int flags, struct svc_req *sr)
{
    static dscudaEventCreateResult res;
    cudaError_t err;
    cudaEvent_t event;

    WARN(3, "cudaEventCreateWithFlags(");
    if (!rcuContext) createRcuContext();
    err = cudaEventCreateWithFlags(&event, flags);
    res.event = (RCadr)event;
    check_cuda_error(err);
    res.err = err;
    WARN(3, "0x%08llx, 0x%08x) done. event:0x%08llx\n", &event, flags, event);

    return &res;
}

dscudaResult *
dscudaeventdestroyid_1_svc(RCevent event, struct svc_req *sr)
{
    static dscudaResult res;
    cudaError_t err;

    WARN(3, "cudaEventDestroy(");
    if (!rcuContext) createRcuContext();
    err = cudaEventDestroy((cudaEvent_t)event);
    check_cuda_error(err);
    res.err = err;
    WARN(3, "0x%08llx) done.\n", event);

    return &res;
}

dscudaEventElapsedTimeResult *
dscudaeventelapsedtimeid_1_svc(RCevent start, RCevent end, struct svc_req *sr)
{
    static dscudaEventElapsedTimeResult res;
    cudaError_t err;
    float millisecond;

    WARN(3, "cudaEventElapsedTime(");
    if (!rcuContext) createRcuContext();
    err = cudaEventElapsedTime(&millisecond, (cudaEvent_t)start, (cudaEvent_t)end);
    check_cuda_error(err);
    res.ms = millisecond;
    res.err = err;
    WARN(3, "%5.3f 0x%08llx 0x%08llx) done.\n", millisecond, start, end);

    return &res;
}

dscudaResult *
dscudaeventrecordid_1_svc(RCevent event, RCstream stream, struct svc_req *sr)
{
    static dscudaResult res;
    cudaError_t err;

    WARN(3, "cudaEventRecord(");
    if (!rcuContext) createRcuContext();
    err = cudaEventRecord((cudaEvent_t)event, (cudaStream_t)stream);
    check_cuda_error(err);
    res.err = err;
    WARN(3, "0x%08llx 0x%08llx) done.\n", event, stream);

    return &res;
}

dscudaResult *
dscudaeventsynchronizeid_1_svc(RCevent event, struct svc_req *sr)
{
    static dscudaResult res;
    cudaError_t err;

    WARN(3, "cudaEventSynchronize(");
    if (!rcuContext) createRcuContext();
    err = cudaEventSynchronize((cudaEvent_t)event);
    check_cuda_error(err);
    res.err = err;
    WARN(3, "0x%08llx) done.\n", event);

    return &res;
}

dscudaResult *
dscudaeventqueryid_1_svc(RCevent event, struct svc_req *sr)
{
    static dscudaResult res;
    cudaError_t err;

    WARN(3, "cudaEventQuery(");
    if (!rcuContext) createRcuContext();
    err = cudaEventQuery((cudaEvent_t)event);
    // should not check error due to the nature of this API.
    // check_cuda_error(err);
    res.err = err;
    WARN(3, "0x%08llx) done.\n", event);

    return &res;
}

/*
 * Execution Control
 */
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
                 mp->name, mp->pid, dscudaGetIpaddrString(mp->ipaddr),
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

dscudaLoadModuleResult *
dscudaloadmoduleid_1_svc(RCipaddr ipaddr, RCpid pid, char *mname, char *image, struct svc_req *sr)
{
    CUresult              cuerr;
    Module *mp;
    int                   i;
    static dscudaLoadModuleResult res;

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
        WARN(1, "\n\n----------------------\n"
             "dscudaloadmoduleid_1_svc() got multiple requests for the same module name:%s,\n"
             "the same process id:%d, and the same IP address:%s,\n"
             "which means a client resent the same module more than twice.\n"
             "If you see this message too often, you may want to increase\n"
             "$dscuda/include/dscudadefs.h:RC_CLIENT_CACHE_LIFETIME for better performance.\n"
             "----------------------\n\n",
             mname, pid, dscudaGetIpaddrString(ipaddr));
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
        if (cuerr != CUDA_SUCCESS) {
            WARN(0, "cuModuleLoadData() failed. %s\n", cudaGetErrorString((cudaError_t)cuerr));
            fatal_error(1);
        }
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
    res.id = mp->id;

    releaseModules();

    return &res;
}


static int
ibvUnpackKernelParam(CUfunction *kfuncp, int narg, IbvArg *args)
{
    CUresult cuerr;
    CUfunction kfunc = *kfuncp;
    IbvArg noarg;
    IbvArg *argp = &noarg;
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
                WARN(0, "cuParamSetv(0x%08llx, %d, %f) failed. %s\n",
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
            WARN(0, "ibvUnpackKernelParam: invalid RCargType\n", argp->type);
            fatal_error(1);
        }
    }
    return argp->offset + argp->size;
}

static int
rpcUnpackKernelParam(CUfunction *kfuncp, RCargs *argsp)
{
    CUresult cuerr;
    CUfunction kfunc = *kfuncp;
    int ival;
    float fval;
    void *pval;
    RCarg noarg;
    RCarg *argp = &noarg;

    noarg.offset = 0;
    noarg.size = 0;

    for (int i = 0; i < argsp->RCargs_len; i++) {
        argp = &(argsp->RCargs_val[i]);

        switch (argp->val.type) {
          case dscudaArgTypeP:
            pval = (void*)&(argp->val.RCargVal_u.address);
            cuerr = cuParamSetv(kfunc, argp->offset, pval, argp->size);
            if (cuerr != CUDA_SUCCESS) {
                WARN(0, "cuParamSetv(0x%08llx, %d, 0x%08llx, %d) failed. %s\n",
                     kfunc, argp->offset, pval, argp->size,
                     cudaGetErrorString((cudaError_t)cuerr));
                fatal_error(1);
            }
            break;

          case dscudaArgTypeI:
            ival = argp->val.RCargVal_u.valuei;
            cuerr = cuParamSeti(kfunc, argp->offset, ival);
            if (cuerr != CUDA_SUCCESS) {
                WARN(0, "cuParamSeti(0x%08llx, %d, %d) failed. %s\n",
                     kfunc, argp->offset, ival,
                     cudaGetErrorString((cudaError_t)cuerr));
                fatal_error(1);
            }
            break;

          case dscudaArgTypeF:
            fval = argp->val.RCargVal_u.valuef;
            cuerr = cuParamSetf(kfunc, argp->offset, fval);
            if (cuerr != CUDA_SUCCESS) {
                WARN(0, "cuParamSetv(0x%08llx, %d, %f) failed. %s\n",
                     kfunc, argp->offset, fval,
                     cudaGetErrorString((cudaError_t)cuerr));
                fatal_error(1);
            }
            break;

          case dscudaArgTypeV:
            pval = argp->val.RCargVal_u.valuev;
            cuerr = cuParamSetv(kfunc, argp->offset, pval, argp->size);
            if (cuerr != CUDA_SUCCESS) {
                WARN(0, "cuParamSetv(0x%08llx, %d, 0x%08llx, %d) failed. %s\n",
                     kfunc, argp->offset, pval, argp->size,
                     cudaGetErrorString((cudaError_t)cuerr));
                fatal_error(1);
            }
            break;

          default:
            WARN(0, "rpcUnpackKernelParam: invalid RCargType\n", argp->val.type);
            fatal_error(1);
        }
    }
    return argp->offset + argp->size;
}

/*
 * launch a kernel function of id 'kid' (or name 'kname', if it's not loaded yet),
 * defined in a module of id 'moduleid'.
 */
void *
dscudalaunchkernelid_1_svc(int moduleid, int kid, char *kname,
                          RCdim3 gdim, RCdim3 bdim, RCsize smemsize, RCstream stream, RCargs args, struct svc_req *)
{
    static int dummyres = 123;
    int paramsize;
    CUresult cuerr;

#if !RC_SUPPORT_CONCURRENT_EXEC
    stream = 0;
#endif

    if (!rcuContext) createRcuContext();

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

    // a kernel function found.
    // now make it run.
    if (UseIbv) {
        paramsize = ibvUnpackKernelParam(&kfunc, args.RCargs_len, (IbvArg *)args.RCargs_val);
    }
    else {
        paramsize = rpcUnpackKernelParam(&kfunc, &args);
    }
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
        WARN(4, "cuLaunchGrid() done. kname:%s\n", kname);
    }
    else {
        cuerr = cuLaunchGridAsync(kfunc, gdim.x, gdim.y, (cudaStream_t)stream);
        if (cuerr != CUDA_SUCCESS) {
            WARN(0, "cuLaunchGridAsync() failed. kname:%s  %s\n",
                 kname, cudaGetErrorString((cudaError_t)cuerr));
            fatal_error(1);
        }
        WARN(4, "cuLaunchGridAsync() done.  kname:%s  stream:0x%08llx\n", kname, stream);
    }

    return &dummyres; // seems necessary to return something even if it's not used by the client.
}

dscudaFuncGetAttributesResult *
dscudafuncgetattributesid_1_svc(int moduleid, char *kname, struct svc_req *sr)
{
    static dscudaFuncGetAttributesResult res;
    CUresult err;
    CUfunction kfunc;

    if (!rcuContext) createRcuContext();

    err = getFunctionByName(&kfunc, kname, moduleid);
    check_cuda_error((cudaError_t)err);

    WARN(3, "cuFuncGetAttribute(");
    err = cuFuncGetAttribute(&res.attr.binaryVersion, CU_FUNC_ATTRIBUTE_BINARY_VERSION, kfunc);
    check_cuda_error((cudaError_t)err);
    WARN(3, "0x%08llx, %d, 0x%08llx) done.\n", &res.attr.binaryVersion, CU_FUNC_ATTRIBUTE_BINARY_VERSION, kfunc);

    err = cuFuncGetAttribute((int *)&res.attr.constSizeBytes, CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES, kfunc);
    check_cuda_error((cudaError_t)err);
    WARN(3, "0x%08llx, %d, 0x%08llx) done.\n", &res.attr.constSizeBytes, CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES, kfunc);

    err = cuFuncGetAttribute((int *)&res.attr.localSizeBytes, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, kfunc);
    check_cuda_error((cudaError_t)err);
    WARN(3, "0x%08llx, %d, 0x%08llx) done.\n", &res.attr.localSizeBytes, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, kfunc);

    err = cuFuncGetAttribute(&res.attr.maxThreadsPerBlock, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, kfunc);
    check_cuda_error((cudaError_t)err);
    WARN(3, "0x%08llx, %d, 0x%08llx) done.\n", &res.attr.maxThreadsPerBlock, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, kfunc);

    err = cuFuncGetAttribute(&res.attr.numRegs, CU_FUNC_ATTRIBUTE_NUM_REGS, kfunc);
    check_cuda_error((cudaError_t)err);
    WARN(3, "0x%08llx, %d, 0x%08llx) done.\n", &res.attr.numRegs, CU_FUNC_ATTRIBUTE_NUM_REGS, kfunc);

    err = cuFuncGetAttribute(&res.attr.ptxVersion, CU_FUNC_ATTRIBUTE_PTX_VERSION, kfunc);
    check_cuda_error((cudaError_t)err);
    WARN(3, "0x%08llx, %d, 0x%08llx) done.\n", &res.attr.ptxVersion, CU_FUNC_ATTRIBUTE_PTX_VERSION, kfunc);

    err = cuFuncGetAttribute((int *)&res.attr.sharedSizeBytes, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, kfunc);
    check_cuda_error((cudaError_t)err);
    WARN(3, "0x%08llx, %d, 0x%08llx) done.\n", &res.attr.sharedSizeBytes, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, kfunc);

    res.err = err;

    return &res;
}

/*
 * Memory Management
 */

dscudaMallocResult * 
dscudamallocid_1_svc(RCsize size, struct svc_req *sr)
{
    static dscudaMallocResult res;
    cudaError_t err;
    int *devadr;

    WARN(3, "cudaMalloc(");
    if (!rcuContext) createRcuContext();
    err = cudaMalloc((void**)&devadr, size);
    res.devAdr = (RCadr)devadr;
    check_cuda_error(err);
    res.err = err;
    WARN(3, "0x%08llx, %d) done. devadr:0x%08llx\n", &devadr, size, devadr);

    return &res;
}

dscudaResult *
dscudafreeid_1_svc(RCadr mem, struct svc_req *)
{
    static dscudaResult res;
    cudaError_t err;

    WARN(3, "cudaFree(");
    if (!rcuContext) createRcuContext();
    err = cudaFree((void*)mem);
    check_cuda_error(err);
    res.err = err;
    WARN(3, "0x%08llx) done.\n", mem);

    return &res;
}

dscudaMemcpyH2HResult *
dscudamemcpyh2hid_1_svc(RCadr dst, RCbuf srcbuf, RCsize count, struct svc_req *sr)
{
    static dscudaMemcpyH2HResult res;
    WARN(0, "dscudaMemcpy() does not support cudaMemcpyHostToHost transfer yet.\n");
    return &res;
}

static int
ibvMemcpyH2D(IbvHdr *rpkt0, IbvHdr *spkt0)
{
    IbvMemcpyH2DInvokeHdr *rpkt = (IbvMemcpyH2DInvokeHdr *)rpkt0;
    IbvMemcpyH2DReturnHdr *spkt = (IbvMemcpyH2DReturnHdr *)spkt0;
    int spktsize;
    cudaError_t err;

    WARN(3, "cudaMemcpy(");
    if (!rcuContext) createRcuContext();

    err = cudaMemcpy((void*)rpkt->dstadr, &rpkt->srcbuf, rpkt->count, cudaMemcpyHostToDevice);
    check_cuda_error(err);
    spkt->err = err;
    WARN(3, "0x%08llx, 0x%08lx, %d, %s) done.\n",
         rpkt->dstadr, (unsigned long)&rpkt->srcbuf, rpkt->count,
         dscudaMemcpyKindName(cudaMemcpyHostToDevice));

    wait_ready_to_rdma(IbvConn);
    spktsize = sizeof(IbvMemcpyH2DReturnHdr);
    spkt->method = RCMethodMemcpyH2D;
    return spktsize;
}

static int
ibvMemcpyD2H(IbvHdr *rpkt0, IbvHdr *spkt0)
{
    IbvMemcpyD2HInvokeHdr *rpkt = (IbvMemcpyD2HInvokeHdr *)rpkt0;
    IbvMemcpyD2HReturnHdr *spkt = (IbvMemcpyD2HReturnHdr *)spkt0;
    int spktsize;
    cudaError_t err;

    WARN(3, "cudaMemcpy(");
    if (!rcuContext) createRcuContext();

    err = cudaMemcpy(&spkt->dstbuf, (void *)rpkt->srcadr, rpkt->count, cudaMemcpyDeviceToHost);
    check_cuda_error(err);
    spkt->err = err;
    WARN(3, "0x%08lx, 0x%08llx, %d, %s) done.\n",
         (unsigned long)&spkt->dstbuf, rpkt->srcadr, rpkt->count,
         dscudaMemcpyKindName(cudaMemcpyDeviceToHost));

    wait_ready_to_rdma(IbvConn);
    spktsize = sizeof(IbvMemcpyD2HReturnHdr) + rpkt->count;
    spkt->method = RCMethodMemcpyD2H;
    return spktsize;
}

static int
ibvMalloc(IbvHdr *rpkt0, IbvHdr *spkt0)
{
    IbvMallocInvokeHdr *rpkt = (IbvMallocInvokeHdr *)rpkt0;
    IbvMallocReturnHdr *spkt = (IbvMallocReturnHdr *)spkt0;
    int spktsize;
    cudaError_t err;

    WARN(3, "cudaMalloc(");
    if (!rcuContext) createRcuContext();

    err = cudaMalloc((void**)&spkt->devAdr, rpkt->size);
    check_cuda_error(err);
    spkt->err = err;
    WARN(3, "0x%08llx, %d) done. devadr:0x%08llx\n", &spkt->devAdr, rpkt->size, spkt->devAdr);

    wait_ready_to_rdma(IbvConn);
    spktsize = sizeof(IbvMallocReturnHdr);
    spkt->method = RCMethodMalloc;
    return spktsize;
}

static int
ibvFree(IbvHdr *rpkt0, IbvHdr *spkt0)
{
    IbvFreeInvokeHdr *rpkt = (IbvFreeInvokeHdr *)rpkt0;
    IbvFreeReturnHdr *spkt = (IbvFreeReturnHdr *)spkt0;
    int spktsize;
    cudaError_t err;

    WARN(3, "cudaFree(");
    if (!rcuContext) createRcuContext();

    err = cudaFree((void*)rpkt->devAdr);
    check_cuda_error(err);
    spkt->err = err;
    WARN(3, "0x%08llx) done.\n", rpkt->devAdr);

    wait_ready_to_rdma(IbvConn);
    spktsize = sizeof(IbvFreeReturnHdr);
    spkt->method = RCMethodFree;
    return spktsize;
}

static int
ibvGetErrorString(IbvHdr *rpkt0, IbvHdr *spkt0)
{
    IbvGetErrorStringInvokeHdr *rpkt = (IbvGetErrorStringInvokeHdr *)rpkt0;
    IbvGetErrorStringReturnHdr *spkt = (IbvGetErrorStringReturnHdr *)spkt0;
    int spktsize;
    int len;
    const char *str;

    WARN(3, "cudaGetErrorString(");
    if (!rcuContext) createRcuContext();

    str = cudaGetErrorString(rpkt->err);
    strncpy((char *)&spkt->errmsg, str, 256);
    len = strlen((char *)&spkt->errmsg);
    WARN(3, "%d) errmsg:%s  done.\n", rpkt->err, &spkt->errmsg);

    wait_ready_to_rdma(IbvConn);
    spktsize = sizeof(IbvGetErrorStringReturnHdr) + len;
    spkt->method = RCMethodGetErrorString;
    return spktsize;
}

static int
ibvGetDeviceProperties(IbvHdr *rpkt0, IbvHdr *spkt0)
{
    IbvGetDevicePropertiesInvokeHdr *rpkt = (IbvGetDevicePropertiesInvokeHdr *)rpkt0;
    IbvGetDevicePropertiesReturnHdr *spkt = (IbvGetDevicePropertiesReturnHdr *)spkt0;
    int spktsize;
    cudaError_t err;

    WARN(3, "cudaGetDeviceProperties(");
    if (!rcuContext) createRcuContext();

    if (1 < Ndevice) {
        WARN(0, "ibvGetDeviceProperties() cannot handle multiple devices for now. Ndevice:%d\n",
             Ndevice);
        exit(1);
    }
    //    err = cudaGetDeviceProperties(&spkt->prop, rpkt->device); // !! NG
    err = cudaGetDeviceProperties(&spkt->prop, Devid[0]);
    check_cuda_error(err);
    spkt->err = err;
    WARN(3, "0x%08lx, %d) done.\n", (unsigned long)&spkt->prop, rpkt->device);

    wait_ready_to_rdma(IbvConn);
    spktsize = sizeof(IbvGetDevicePropertiesReturnHdr);
    spkt->method = RCMethodGetDeviceProperties;
    return spktsize;
}

static int
ibvRuntimeGetVersion(IbvHdr *rpkt0, IbvHdr *spkt0)
{
    //    IbvRuntimeGetVersionInvokeHdr *rpkt = (IbvRuntimeGetVersionInvokeHdr *)rpkt0;
    IbvRuntimeGetVersionReturnHdr *spkt = (IbvRuntimeGetVersionReturnHdr *)spkt0;
    int spktsize;
    cudaError_t err;

    WARN(3, "cudaRuntimeGetVersion(");
    if (!rcuContext) createRcuContext();

    err = cudaRuntimeGetVersion(&spkt->version);
    check_cuda_error(err);
    spkt->err = err;
    WARN(3, "0x%08lx) done.\n", (unsigned long)&spkt->version);

    wait_ready_to_rdma(IbvConn);
    spktsize = sizeof(IbvRuntimeGetVersionReturnHdr);
    spkt->method = RCMethodRuntimeGetVersion;
    return spktsize;
}

static int
ibvDeviceSynchronize(IbvHdr *rpkt0, IbvHdr *spkt0)
{
    IbvDeviceSynchronizeInvokeHdr *rpkt = (IbvDeviceSynchronizeInvokeHdr *)rpkt0;
    IbvDeviceSynchronizeReturnHdr *spkt = (IbvDeviceSynchronizeReturnHdr *)spkt0;
    int spktsize;
    cudaError_t err;

    WARN(3, "cudaDeviceSynchronize()");
    if (!rcuContext) createRcuContext();

    err = cudaDeviceSynchronize();
    check_cuda_error(err);
    spkt->err = err;
    WARN(3, "done.\n");

    wait_ready_to_rdma(IbvConn);
    spktsize = sizeof(IbvDeviceSynchronizeReturnHdr);
    spkt->method = RCMethodDeviceSynchronize;
    return spktsize;
}

static int
ibvDscudaMemcpyToSymbolAsyncH2D(IbvHdr *rpkt0, IbvHdr *spkt0)
{
    IbvDscudaMemcpyToSymbolAsyncH2DInvokeHdr *rpkt = (IbvDscudaMemcpyToSymbolAsyncH2DInvokeHdr *)rpkt0;
    IbvDscudaMemcpyToSymbolAsyncH2DReturnHdr *spkt = (IbvDscudaMemcpyToSymbolAsyncH2DReturnHdr *)spkt0;
    int spktsize;

    // dscudamemcpytosymbolasynch2did_1_svc(int moduleid, char *symbol, RCbuf src, RCsize count, RCsize offset, RCstream stream, struct svc_req *sr)
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

    wait_ready_to_rdma(IbvConn);
    spktsize = sizeof(IbvDscudaMemcpyToSymbolAsyncH2DReturnHdr);
    spkt->method = RCMethodDscudaMemcpyToSymbolAsyncH2D;
    return spktsize;
}


static int
ibvDscudaMemcpyToSymbolAsyncD2D(IbvHdr *rpkt0, IbvHdr *spkt0)
{
    IbvDscudaMemcpyToSymbolAsyncD2DInvokeHdr *rpkt = (IbvDscudaMemcpyToSymbolAsyncD2DInvokeHdr *)rpkt0;
    IbvDscudaMemcpyToSymbolAsyncD2DReturnHdr *spkt = (IbvDscudaMemcpyToSymbolAsyncD2DReturnHdr *)spkt0;
    int spktsize;


    // dscudamemcpytosymbolasyncd2did_1_svc(int moduleid, char *symbol, RCadr src, RCsize count, RCsize offset, RCstream stream, struct svc_req *sr)
    dscudaResult *resp = dscudamemcpytosymbolasyncd2did_1_svc(rpkt->moduleid,
                                                            rpkt->symbol,
                                                            rpkt->srcadr,
                                                            rpkt->count,
                                                            rpkt->offset,
                                                            rpkt->stream,
                                                            NULL);

    spkt->err = (cudaError_t)cudaSuccess;

    wait_ready_to_rdma(IbvConn);
    spktsize = sizeof(IbvDscudaMemcpyToSymbolAsyncD2DReturnHdr);
    spkt->method = RCMethodDscudaMemcpyToSymbolAsyncD2D;
    return spktsize;
}


static int
ibvDscudaMemcpyFromSymbolAsyncD2H(IbvHdr *rpkt0, IbvHdr *spkt0)
{
    IbvDscudaMemcpyFromSymbolAsyncD2HInvokeHdr *rpkt = (IbvDscudaMemcpyFromSymbolAsyncD2HInvokeHdr *)rpkt0;
    IbvDscudaMemcpyFromSymbolAsyncD2HReturnHdr *spkt = (IbvDscudaMemcpyFromSymbolAsyncD2HReturnHdr *)spkt0;
    int spktsize;

    // dscudamemcpytosymbolasynch2did_1_svc(int moduleid, char *symbol, RCbuf src, RCsize count, RCsize offset, RCstream stream, struct svc_req *sr)
    dscudaMemcpyFromSymbolAsyncD2HResult *resp = dscudamemcpyfromsymbolasyncd2hid_1_svc(rpkt->moduleid,
                                                                                      rpkt->symbol,
                                                                                      rpkt->count,
                                                                                      rpkt->offset,
                                                                                      rpkt->stream,
                                                                                      NULL);

    spkt->err = (cudaError_t)cudaSuccess;

    wait_ready_to_rdma(IbvConn);
    spktsize = sizeof(IbvDscudaMemcpyFromSymbolAsyncD2HReturnHdr) + rpkt->count;
    spkt->method = RCMethodDscudaMemcpyFromSymbolAsyncD2H;
    memcpy(&spkt->dst, resp->buf.RCbuf_val, rpkt->count);
    return spktsize;
}


static int
ibvDscudaMemcpyFromSymbolAsyncD2D(IbvHdr *rpkt0, IbvHdr *spkt0)
{
    IbvDscudaMemcpyFromSymbolAsyncD2DInvokeHdr *rpkt = (IbvDscudaMemcpyFromSymbolAsyncD2DInvokeHdr *)rpkt0;
    IbvDscudaMemcpyFromSymbolAsyncD2DReturnHdr *spkt = (IbvDscudaMemcpyFromSymbolAsyncD2DReturnHdr *)spkt0;
    int spktsize;


    // dscudamemcpytosymbolasyncd2did_1_svc(int moduleid, char *symbol, RCadr src, RCsize count, RCsize offset, RCstream stream, struct svc_req *sr)
    dscudaResult *resp = dscudamemcpyfromsymbolasyncd2did_1_svc(rpkt->moduleid,
                                                            rpkt->dstadr,
                                                            rpkt->symbol,
                                                            rpkt->count,
                                                            rpkt->offset,
                                                            rpkt->stream,
                                                            NULL);

    spkt->err = (cudaError_t)cudaSuccess;

    wait_ready_to_rdma(IbvConn);
    spktsize = sizeof(IbvDscudaMemcpyFromSymbolAsyncD2DReturnHdr);
    spkt->method = RCMethodDscudaMemcpyFromSymbolAsyncD2D;
    return spktsize;
}



static int
ibvDscudaLoadModule(IbvHdr *rpkt0, IbvHdr *spkt0)
{
    IbvDscudaLoadModuleInvokeHdr *rpkt = (IbvDscudaLoadModuleInvokeHdr *)rpkt0;
    IbvDscudaLoadModuleReturnHdr *spkt = (IbvDscudaLoadModuleReturnHdr *)spkt0;
    int spktsize;

    // dscudaloadmoduleid_1_svc(RCipaddr ipaddr, RCpid pid, char *mname, char *image, struct svc_req *sr)
    dscudaLoadModuleResult *resp = dscudaloadmoduleid_1_svc((RCipaddr)rpkt->ipaddr,
                                                          (RCpid)rpkt->pid,
                                                          rpkt->modulename,
                                                          (char *)&rpkt->moduleimage,
                                                          NULL);

    spkt->err = (cudaError_t)cudaSuccess;
    spkt->moduleid = resp->id;

    wait_ready_to_rdma(IbvConn);
    spktsize = sizeof(IbvDscudaLoadModuleReturnHdr);
    spkt->method = RCMethodDscudaLoadModule;
    return spktsize;
}

static int
ibvDscudaLaunchKernel(IbvHdr *rpkt0, IbvHdr *spkt0)
{
    IbvDscudaLaunchKernelInvokeHdr *rpkt = (IbvDscudaLaunchKernelInvokeHdr *)rpkt0;
    IbvDscudaLaunchKernelReturnHdr *spkt = (IbvDscudaLaunchKernelReturnHdr *)spkt0;
    int spktsize;
    RCdim3 gdim, bdim;
    RCargs args;

    gdim.x = rpkt->gdim[0];
    gdim.y = rpkt->gdim[1];
    gdim.z = rpkt->gdim[2];

    bdim.x = rpkt->bdim[0];
    bdim.y = rpkt->bdim[1];
    bdim.z = rpkt->bdim[2];

    args.RCargs_len = rpkt->narg;
    args.RCargs_val = (RCarg *)&rpkt->args;

    //    dscudalaunchkernelid_1_svc(int moduleid, int kid, char *kname,
    //                              RCdim3 gdim, RCdim3 bdim, RCsize smemsize, RCstream stream, RCargs args, struct svc_req *)
    dscudalaunchkernelid_1_svc(rpkt->moduleid, rpkt->kernelid, rpkt->kernelname,
                              gdim, bdim, rpkt->smemsize, rpkt->stream, args, NULL);



    spkt->err = (cudaError_t)cudaSuccess;

    wait_ready_to_rdma(IbvConn);
    spktsize = sizeof(IbvDscudaLaunchKernelReturnHdr);
    spkt->method = RCMethodDscudaLaunchKernel;
    return spktsize;
}

dscudaResult *
dscudamemcpyh2did_1_svc(RCadr dst, RCbuf srcbuf, RCsize count, struct svc_req *sr)
{
    static dscudaResult res;
    cudaError_t err;

    WARN(3, "cudaMemcpy(");
    if (!rcuContext) createRcuContext();
    err = cudaMemcpy((void*)dst, srcbuf.RCbuf_val, count, cudaMemcpyHostToDevice);
    check_cuda_error(err);
    res.err = err;
    WARN(3, "0x%08llx, 0x%08lx, %d, %s) done.\n",
            dst, (unsigned long)srcbuf.RCbuf_val, count, dscudaMemcpyKindName(cudaMemcpyHostToDevice));
    return &res;
}

dscudaMemcpyD2HResult *
dscudamemcpyd2hid_1_svc(RCadr src, RCsize count, struct svc_req *sr)
{
    static RCsize maxcount = 0;
    static dscudaMemcpyD2HResult res;
    cudaError_t err;

    WARN(3, "cudaMemcpy(");
    if (!rcuContext) createRcuContext();
    if (maxcount == 0) {
        res.buf.RCbuf_val = NULL;
    }
    if (maxcount < count) {
        res.buf.RCbuf_val = (char*)realloc(res.buf.RCbuf_val, count);
        maxcount = count;
    }
    res.buf.RCbuf_len = count;
    err = cudaMemcpy(res.buf.RCbuf_val, (const void*)src, count, cudaMemcpyDeviceToHost);
    WARN(3, "0x%08lx, 0x%08llx, %d, %s) done.\n",
         (unsigned long)res.buf.RCbuf_val, src, count, dscudaMemcpyKindName(cudaMemcpyDeviceToHost));
    check_cuda_error(err);
    res.err = err;

#if 0 // destroy some part of the returning data. debugging purpose only.
    {
        srand48(time(NULL));
        if (ServerId == 0 && drand48() < 1.0/100.0) {
            res.buf.RCbuf_val[0] = 0;
        }
    }
#endif

    return &res;
}

dscudaResult *
dscudamemcpyd2did_1_svc(RCadr dst, RCadr src, RCsize count, struct svc_req *sr)
{
    cudaError_t err;
    static dscudaResult res;
    WARN(3, "cudaMemcpy(");
    err = cudaMemcpy((void *)dst, (void *)src, count, cudaMemcpyDeviceToDevice);
    check_cuda_error(err);
    res.err = err;
    WARN(3, "0x%08llx, 0x%08llx, %d, %s) done.\n",
            dst, src, count, dscudaMemcpyKindName(cudaMemcpyDeviceToDevice));
    return &res;
}

dscudaMallocArrayResult *
dscudamallocarrayid_1_svc(RCchanneldesc desc, RCsize width, RCsize height, unsigned int flags, struct svc_req *sr)
{
    static dscudaMallocArrayResult res;
    cudaError_t err;
    cudaArray *devadr;
    cudaChannelFormatDesc descbuf = cudaCreateChannelDesc(desc.x, desc.y, desc.z, desc.w, (enum cudaChannelFormatKind)desc.f);

    WARN(3, "cudaMallocArray(");
    if (!rcuContext) createRcuContext();
    err = cudaMallocArray((cudaArray**)&devadr, &descbuf, width, height, flags);
    res.array = (RCadr)devadr;
    check_cuda_error(err);
    res.err = err;
    WARN(3, "0x%08llx, 0x%08llx, %d, %d, 0x%08x) done. devadr:0x%08llx\n",
         &devadr, &descbuf, width, height, flags, devadr)

    return &res;
}

dscudaResult *
dscudafreearrayid_1_svc(RCadr array, struct svc_req *sr)
{
    static dscudaResult res;
    cudaError_t err;

    WARN(3, "cudaFreeArray(");
    if (!rcuContext) createRcuContext();
    err = cudaFreeArray((cudaArray*)array);
    check_cuda_error(err);
    res.err = err;
    WARN(3, "0x%08llx) done.\n", array);

    return &res;
}

dscudaMemcpyToArrayH2HResult *
dscudamemcpytoarrayh2hid_1_svc(RCadr dst, RCsize wOffset, RCsize hOffset, RCbuf src, RCsize count, struct svc_req *sr)
{
    static dscudaMemcpyToArrayH2HResult res;
    WARN(0, "dscudaMemcpyToArray() does not support cudaMemcpyHostToHost transfer yet.\n");
    return &res;
}

dscudaResult *
dscudamemcpytoarrayh2did_1_svc(RCadr dst, RCsize wOffset, RCsize hOffset, RCbuf src, RCsize count, struct svc_req *sr)
{
    static dscudaResult res;
    cudaError_t err;

    WARN(3, "cudaMemcpyToArray(");
    if (!rcuContext) createRcuContext();
    err = cudaMemcpyToArray((cudaArray *)dst, wOffset, hOffset, src.RCbuf_val, count, cudaMemcpyHostToDevice);
    check_cuda_error(err);
    res.err = err;
    WARN(3, "0x%08llx, %d, %d, 0x%08llx, %d, %s) done.\n",
         dst, wOffset, hOffset, (unsigned long)src.RCbuf_val, count, dscudaMemcpyKindName(cudaMemcpyHostToDevice));
    return &res;
}

dscudaMemcpyToArrayD2HResult *
dscudamemcpytoarrayd2hid_1_svc(RCsize wOffset, RCsize hOffset, RCadr src, RCsize count, struct svc_req *sr)
{
    static dscudaMemcpyToArrayD2HResult res;
    WARN(0, "dscudaMemcpyToArray() does not support cudaMemcpyDeviceToHost transfer yet.\n");
    return &res;
}

dscudaResult *
dscudamemcpytoarrayd2did_1_svc(RCadr dst, RCsize wOffset, RCsize hOffset, RCadr src, RCsize count, struct svc_req *sr)
{
    cudaError_t err;
    static dscudaResult res;
    WARN(3, "cudaMemcpyToArray(");
    err = cudaMemcpyToArray((cudaArray *)dst, wOffset, hOffset, (void *)src, count, cudaMemcpyDeviceToDevice);
    check_cuda_error(err);
    res.err = err;
    WARN(3, "0x%08llx, %d, %d, 0x%08llx, %d, %s) done.\n",
         dst, wOffset, hOffset, src, count, dscudaMemcpyKindName(cudaMemcpyDeviceToDevice));
    return &res;
}

static void
getGlobalSymbol(int moduleid, char *symbolname, CUdeviceptr *dptr, size_t *size)
{
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
}

dscudaResult *
dscudamemcpytosymbolh2did_1_svc(int moduleid, char *symbol, RCbuf src, RCsize count, RCsize offset, struct svc_req *sr)
{
    static dscudaResult res;
    cudaError_t err;
    CUdeviceptr gsptr;
    size_t gssize;

    WARN(3, "cudaMemcpyToSymbol(");
    if (!rcuContext) createRcuContext();

    getGlobalSymbol(moduleid, symbol, &gsptr, &gssize);
    err = cudaMemcpy((char *)gsptr + offset, src.RCbuf_val, count, cudaMemcpyHostToDevice);
                             
    WARN(3, "0x%08lx, 0x%08lx, %d, %d, %s) done. module name:%s  symbol:%s\n",
         gsptr, (unsigned long)src.RCbuf_val, count, offset,
         dscudaMemcpyKindName(cudaMemcpyHostToDevice),
         Modulelist[moduleid].name, symbol);
    check_cuda_error(err);
    res.err = err;
    return &res;
}

dscudaResult *
dscudamemcpytosymbold2did_1_svc(int moduleid, char *symbol, RCadr src, RCsize count, RCsize offset, struct svc_req *sr)
{
    static dscudaResult res;
    cudaError_t err;
    CUdeviceptr gsptr;
    size_t gssize;

    WARN(3, "cudaMemcpyToSymbol(");
    if (!rcuContext) createRcuContext();

    getGlobalSymbol(moduleid, symbol, &gsptr, &gssize);

    err = cudaMemcpy((char *)gsptr + offset, (void*)src, count, cudaMemcpyDeviceToDevice);
    check_cuda_error(err);
    res.err = err;
    WARN(3, "0x%08llx, 0x%08llx, %d, %d, %s) done.\n",
         gsptr, (unsigned long)src, count, offset,
         dscudaMemcpyKindName(cudaMemcpyDeviceToDevice));

    return &res;
}

dscudaMemcpyFromSymbolD2HResult *
dscudamemcpyfromsymbold2hid_1_svc(int moduleid, char *symbol, RCsize count, RCsize offset, struct svc_req *sr)
{
    static RCsize maxcount = 0;
    static dscudaMemcpyFromSymbolD2HResult res;
    cudaError_t err;
    CUdeviceptr gsptr;
    size_t gssize;

    WARN(3, "cudaMemcpyFromSymbol(");
    if (!rcuContext) createRcuContext();
    if (maxcount == 0) {
        res.buf.RCbuf_val = NULL;
    }
    if (maxcount < count) {
        res.buf.RCbuf_val = (char*)realloc(res.buf.RCbuf_val, count);
        maxcount = count;
    }
    res.buf.RCbuf_len = count;

    getGlobalSymbol(moduleid, symbol, &gsptr, &gssize);
    err = cudaMemcpy(res.buf.RCbuf_val, (char *)gsptr + offset, count, cudaMemcpyDeviceToHost);
                             
    WARN(3, "0x%08llx, 0x%08llx, %d, %d, %s) done. module name:%s  symbol:%s\n",
         (unsigned long)res.buf.RCbuf_val, gsptr, count, offset,
         dscudaMemcpyKindName(cudaMemcpyDeviceToHost),
         Modulelist[moduleid].name, symbol);
    check_cuda_error(err);
    res.err = err;
    return &res;
}

dscudaResult *
dscudamemcpyfromsymbold2did_1_svc(int moduleid, RCadr dst, char *symbol, RCsize count, RCsize offset, struct svc_req *sr)
{
    static dscudaResult res;
    cudaError_t err;
    CUdeviceptr gsptr;
    size_t gssize;

    WARN(3, "cudaMemcpyFromSymbol(");
    if (!rcuContext) createRcuContext();

    getGlobalSymbol(moduleid, symbol, &gsptr, &gssize);

    err = cudaMemcpy((void*)dst, (char *)gsptr + offset, count, cudaMemcpyDeviceToDevice);
    check_cuda_error(err);
    res.err = err;
    WARN(3, "0x%08llx, 0x%08llx, %d, %d, %s) done.\n",
         (unsigned long)dst, gsptr, count, offset,
         dscudaMemcpyKindName(cudaMemcpyDeviceToDevice));

    return &res;
}


dscudaResult *
dscudamemcpytosymbolasynch2did_1_svc(int moduleid, char *symbol, RCbuf src, RCsize count, RCsize offset, RCstream stream, struct svc_req *sr)
{
    static dscudaResult res;
    cudaError_t err;
    CUdeviceptr gsptr;
    size_t gssize;

    WARN(3, "cudaMemcpyToSymbolAsync(");
    if (!rcuContext) createRcuContext();
    getGlobalSymbol(moduleid, symbol, &gsptr, &gssize);
    err = cudaMemcpyAsync((char *)gsptr + offset, src.RCbuf_val, count, cudaMemcpyHostToDevice, (cudaStream_t)stream);
                             
    WARN(3, "0x%08lx, 0x%08lx, %d, %d, %s, 0x%08llx) done. module name:%s  symbol:%s\n",
         gsptr, (unsigned long)src.RCbuf_val, count, offset,
         dscudaMemcpyKindName(cudaMemcpyHostToDevice), stream,
         Modulelist[moduleid].name, symbol);

    check_cuda_error(err);
    res.err = err;
    return &res;
}

dscudaResult *
dscudamemcpytosymbolasyncd2did_1_svc(int moduleid, char *symbol, RCadr src, RCsize count, RCsize offset, RCstream stream, struct svc_req *sr)
{
    static dscudaResult res;
    cudaError_t err;
    CUdeviceptr gsptr;
    size_t gssize;

    WARN(3, "cudaMemcpyToSymbolAsync(");
    if (!rcuContext) createRcuContext();

    getGlobalSymbol(moduleid, symbol, &gsptr, &gssize);

    err = cudaMemcpyAsync((char *)gsptr + offset, (void*)src, count, cudaMemcpyDeviceToDevice, (cudaStream_t)stream);
    check_cuda_error(err);
    res.err = err;
    WARN(3, "0x%08lx, 0x%08lx, %d, %d, %s, 0x%08llx) done.\n",
         gsptr, (unsigned long)src, count, offset, stream,
         dscudaMemcpyKindName(cudaMemcpyDeviceToDevice));

    return &res;
}

dscudaMemcpyFromSymbolAsyncD2HResult *
dscudamemcpyfromsymbolasyncd2hid_1_svc(int moduleid, char *symbol, RCsize count, RCsize offset, RCstream stream, struct svc_req *sr)
{
    static RCsize maxcount = 0;
    static dscudaMemcpyFromSymbolAsyncD2HResult res;
    cudaError_t err;
    CUdeviceptr gsptr;
    size_t gssize;

    WARN(3, "cudaMemcpyFromSymbolAsync(");
    if (!rcuContext) createRcuContext();
    if (maxcount == 0) {
        res.buf.RCbuf_val = NULL;
    }
    if (maxcount < count) {
        res.buf.RCbuf_val = (char*)realloc(res.buf.RCbuf_val, count);
        maxcount = count;
    }
    res.buf.RCbuf_len = count;

    getGlobalSymbol(moduleid, symbol, &gsptr, &gssize);
    err = cudaMemcpyAsync(res.buf.RCbuf_val, (char *)gsptr + offset, count, cudaMemcpyDeviceToHost, (cudaStream_t)stream);
                             
    WARN(3, "0x%08lx, 0x%08lx, %d, %d, %s, 0x%08llx) done. module name:%s  symbol:%s\n",
         (unsigned long)res.buf.RCbuf_val, gsptr, count, offset, stream,
         dscudaMemcpyKindName(cudaMemcpyDeviceToHost),
         Modulelist[moduleid].name, symbol);
    check_cuda_error(err);
    res.err = err;
    return &res;
}

dscudaResult *
dscudamemcpyfromsymbolasyncd2did_1_svc(int moduleid, RCadr dst, char *symbol, RCsize count, RCsize offset, RCstream stream, struct svc_req *sr)
{
    static dscudaResult res;
    cudaError_t err;
    CUdeviceptr gsptr;
    size_t gssize;

    WARN(3, "cudaMemcpyFromSymbolAsync(");
    if (!rcuContext) createRcuContext();

    getGlobalSymbol(moduleid, symbol, &gsptr, &gssize);

    err = cudaMemcpyAsync((void*)dst, (char *)gsptr + offset, count, cudaMemcpyDeviceToDevice, (cudaStream_t)stream);
    check_cuda_error(err);
    res.err = err;
    WARN(3, "0x%08lx, 0x%08lx, %d, %d, %s, 0x%08llx) done.\n",
         (unsigned long)dst, gsptr, count, offset, stream,
         dscudaMemcpyKindName(cudaMemcpyDeviceToDevice));

    return &res;
}


dscudaResult *
dscudamemsetid_1_svc(RCadr dst, int value, RCsize count, struct svc_req *sq)
{
    static dscudaResult res;
    cudaError_t err;

    WARN(3, "cudaMemset(");
    if (!rcuContext) createRcuContext();
    err = cudaMemset((void *)dst, value, count);
    check_cuda_error(err);
    res.err = err;
    WARN(3, "0x%08llx, %d, %d) done.\n", dst, value, count);
    return &res;
}

dscudaHostAllocResult *
dscudahostallocid_1_svc(RCsize size, unsigned int flags, struct svc_req *sr)
{
    static dscudaHostAllocResult res;
    cudaError_t err;
    int *devadr;

    WARN(3, "cudaHostAlloc(");
    if (!rcuContext) createRcuContext();
    err = cudaHostAlloc((void**)&devadr, size, flags);
    res.pHost = (RCadr)devadr;
    check_cuda_error(err);
    res.err = err;
    WARN(3, "0x%08llx, %d, 0x%08x) done.\n", res.pHost, size, flags);

    return &res;
}

dscudaMallocHostResult *
dscudamallochostid_1_svc(RCsize size, struct svc_req *sr)
{
    static dscudaMallocHostResult res;
    cudaError_t err;
    int *devadr;

    WARN(3, "cudaMallocHost(");
    if (!rcuContext) createRcuContext();
    err = cudaMallocHost((void**)&devadr, size);
    res.ptr = (RCadr)devadr;
    check_cuda_error(err);
    res.err = err;
    WARN(3, "0x%08llx, %d) done. devadr:0x%08llx\n", &devadr, size, devadr);

    return &res;
}

dscudaResult *
dscudafreehostid_1_svc(RCadr ptr, struct svc_req *sr)
{
    static dscudaResult res;
    cudaError_t err;

    WARN(3, "cudaFreeHost(");
    if (!rcuContext) createRcuContext();
    err = cudaFreeHost((void*)ptr);
    check_cuda_error(err);
    res.err = err;
    WARN(3, "0x%08llx) done.\n", ptr);

    return &res;
}

dscudaHostGetDevicePointerResult *
dscudahostgetdevicepointerid_1_svc(RCadr pHost, unsigned int flags , struct svc_req *sr)
{
    cudaError_t err;
    static dscudaHostGetDevicePointerResult res;
    RCadr pDevice;

    WARN(3, "cudaHostGetDevicePointer(");
    if (!rcuContext) createRcuContext();

    err = cudaHostGetDevicePointer((void **)&pDevice, (void *)pHost, flags);
    check_cuda_error(err);
    res.pDevice = pDevice;
    res.err = err;
    WARN(3, ") done.\n");
    return &res;
}

dscudaHostGetFlagsResult *
dscudahostgetflagsid_1_svc(RCadr pHost, struct svc_req *sr)
{
    cudaError_t err;
    static dscudaHostGetFlagsResult res;
    unsigned int flags;

    WARN(3, "cudaHostGetFlags(");
    if (!rcuContext) createRcuContext();

    err = cudaHostGetFlags(&flags, (void *)pHost);
    check_cuda_error(err);
    res.err = err;
    res.flags = flags;
    WARN(3, ") done.\n");
    return &res;
}

dscudaMemcpyAsyncH2HResult *
dscudamemcpyasynch2hid_1_svc(RCadr dst, RCbuf src, RCsize count, RCstream stream, struct svc_req *sr)
{
    static dscudaMemcpyAsyncH2HResult res;
    WARN(0, "dscudaMemcpyAsync() does not support cudaMemcpyHostToHost transfer yet.\n");
    return &res;
}

dscudaResult *
dscudamemcpyasynch2did_1_svc(RCadr dst, RCbuf src, RCsize count, RCstream stream, struct svc_req *sr)
{
    static dscudaResult res;
    cudaError_t err;

    WARN(3, "cudaMemcpyAsync(");
    if (!rcuContext) createRcuContext();
    err = cudaMemcpyAsync((void*)dst, src.RCbuf_val, count, cudaMemcpyHostToDevice, (cudaStream_t)stream);
    check_cuda_error(err);
    res.err = err;
    WARN(3, "0x%08lx, 0x%08lx, %d, %s, 0x%08lx) done.\n",
         dst, (unsigned long)src.RCbuf_val, count, dscudaMemcpyKindName(cudaMemcpyHostToDevice), stream);
    return &res;
}

dscudaMemcpyAsyncD2HResult *
dscudamemcpyasyncd2hid_1_svc(RCadr src, RCsize count, RCstream stream, struct svc_req *sr)
{
    static RCsize maxcount = 0;
    cudaError_t err;
    static dscudaMemcpyAsyncD2HResult res;

    WARN(3, "cudaMemcpyAsync(");
    if (!rcuContext) createRcuContext();
    if (maxcount == 0) {
        res.buf.RCbuf_val = NULL;
    }
    if (maxcount < count) {
        res.buf.RCbuf_val = (char*)realloc(res.buf.RCbuf_val, count);
        maxcount = count;
    }
    res.buf.RCbuf_len = count;
    err = cudaMemcpyAsync(res.buf.RCbuf_val, (const void*)src, count, cudaMemcpyDeviceToHost, (cudaStream_t)stream);
    check_cuda_error(err);
    res.err = err;
    WARN(3, "0x%08llx, 0x%08llx, %d, %s, 0x%08llx) done.\n",
         (unsigned long)res.buf.RCbuf_val, src, count, dscudaMemcpyKindName(cudaMemcpyDeviceToHost), stream);
    return &res;
}

dscudaResult *
dscudamemcpyasyncd2did_1_svc(RCadr dst, RCadr src, RCsize count, RCstream stream, struct svc_req *sr)
{
    cudaError_t err;
    static dscudaResult res;
    WARN(3, "cudaMemcpyAsync(");
    err = cudaMemcpyAsync((void *)dst, (void *)src, count, cudaMemcpyDeviceToDevice, (cudaStream_t)stream);
    check_cuda_error(err);
    res.err = err;
    WARN(3, "0x%08llx, 0x%08llx, %d, %s, 0x%08llx) done.\n",
         dst, src, count, dscudaMemcpyKindName(cudaMemcpyDeviceToDevice), stream);
    return &res;
}


dscudaMallocPitchResult *
dscudamallocpitchid_1_svc(RCsize width, RCsize height, struct svc_req *sr)
{
    static dscudaMallocPitchResult res;
    cudaError_t err;
    int *devadr;
    size_t pitch;

    WARN(3, "cudaMallocPitch(");
    if (!rcuContext) createRcuContext();
    err = cudaMallocPitch((void**)&devadr, &pitch, width, height);
    res.devPtr = (RCadr)devadr;
    res.pitch = pitch;
    check_cuda_error(err);
    res.err = err;
    WARN(3, "0x%08llx, %d, %d) done. devadr:0x%08llx\n", &devadr, width, height, devadr);

    return &res;
}

dscudaMemcpy2DToArrayH2HResult *
dscudamemcpy2dtoarrayh2hid_1_svc(RCadr dst, RCsize wOffset, RCsize hOffset, RCbuf src, RCsize spitch, RCsize width, RCsize height, struct svc_req *sr)
{
    static dscudaMemcpy2DToArrayH2HResult res;
    WARN(0, "dscudaMemcpy2DToArray() does not support cudaMemcpyHostToHost transfer yet.\n");
    return &res;
}

dscudaResult *
dscudamemcpy2dtoarrayh2did_1_svc(RCadr dst, RCsize wOffset, RCsize hOffset, RCbuf srcbuf, RCsize spitch, RCsize width, RCsize height, struct svc_req *sr)
{
    static dscudaResult res;
    cudaError_t err;

    WARN(3, "cudaMemcpy2DToArray(");
    if (!rcuContext) createRcuContext();
    err = cudaMemcpy2DToArray((cudaArray*)dst, wOffset, hOffset, srcbuf.RCbuf_val, spitch, width, height, cudaMemcpyHostToDevice);
    check_cuda_error(err);
    res.err = err;
    WARN(3, "0x%08lx, %d, %d, 0x%08llx, %d, %d, %d, %s) done.\n",
         dst, wOffset, hOffset, (unsigned long)srcbuf.RCbuf_val, spitch, width, height, dscudaMemcpyKindName(cudaMemcpyHostToDevice));
    return &res;
}

dscudaMemcpy2DToArrayD2HResult *
dscudamemcpy2dtoarrayd2hid_1_svc(RCsize wOffset, RCsize hOffset, RCadr src, RCsize spitch, RCsize width, RCsize height, struct svc_req *sr)
{
    static RCsize maxcount = 0;
    cudaError_t err;
    static dscudaMemcpy2DToArrayD2HResult res;
    int count = spitch * height;

    WARN(3, "cudaMemcpy2DToArray(");
    if (!rcuContext) createRcuContext();
    if (maxcount == 0) {
        res.buf.RCbuf_val = NULL;
    }
    if (maxcount < count) {
        res.buf.RCbuf_val = (char*)realloc(res.buf.RCbuf_val, count);
        maxcount = count;
    }
    res.buf.RCbuf_len = count;
    err = cudaMemcpy2DToArray((cudaArray *)res.buf.RCbuf_val, wOffset, hOffset, (void *)src, spitch, width, height, cudaMemcpyDeviceToHost);
    check_cuda_error(err);
    res.err = err;
    WARN(3, "0x%08llx, %d, %d, 0x%08llx, %d, %d, %d, %s) done. 2D buf size : %d\n",
         (unsigned long)res.buf.RCbuf_val, wOffset, hOffset, src, spitch, width, height, dscudaMemcpyKindName(cudaMemcpyDeviceToHost), count);
    return &res;
}

dscudaResult *
dscudamemcpy2dtoarrayd2did_1_svc(RCadr dst, RCsize wOffset, RCsize hOffset, RCadr src, RCsize spitch, RCsize width, RCsize height, struct svc_req *sr)
{
    cudaError_t err;
    static dscudaResult res;
    WARN(3, "cudaMemcpy2DToArray(");
    err = cudaMemcpy2DToArray((cudaArray *)dst, wOffset, hOffset, (void *)src, spitch, width, height, cudaMemcpyDeviceToDevice);
    check_cuda_error(err);
    res.err = err;
    WARN(3, "0x%08llx, %d, %d, 0x%08llx, %d, %d, %d, %s) done.\n",
         dst, wOffset, hOffset, src, spitch, width, height, dscudaMemcpyKindName(cudaMemcpyDeviceToDevice));
    return &res;
}

dscudaMemcpy2DH2HResult *
dscudamemcpy2dh2hid_1_svc(RCadr dst, RCsize dpitch, RCbuf src, RCsize spitch, RCsize width, RCsize height, struct svc_req *sr)
{
    static dscudaMemcpy2DH2HResult res;
    WARN(0, "dscudaMemcpy2D() does not support cudaMemcpyHostToHost transfer yet.\n");
    return &res;
}

dscudaResult *
dscudamemcpy2dh2did_1_svc(RCadr dst, RCsize dpitch, RCbuf srcbuf, RCsize spitch, RCsize width, RCsize height, struct svc_req *sr)
{
    static dscudaResult res;
    cudaError_t err;

    WARN(3, "cudaMemcpy2D(");
    if (!rcuContext) createRcuContext();
    err = cudaMemcpy2D((void*)dst, dpitch, srcbuf.RCbuf_val, spitch, width, height, cudaMemcpyHostToDevice);
    check_cuda_error(err);
    res.err = err;
    WARN(3, "0x%08lx, %d, 0x%08lx, %d, %d, %d, %s) done.\n",
         dst, dpitch, (unsigned long)srcbuf.RCbuf_val, spitch, width, height, dscudaMemcpyKindName(cudaMemcpyHostToDevice));
    return &res;
}

dscudaMemcpy2DD2HResult *
dscudamemcpy2dd2hid_1_svc(RCsize dpitch, RCadr src, RCsize spitch, RCsize width, RCsize height, struct svc_req *sr)
{
    static RCsize maxcount = 0;
    cudaError_t err;
    static dscudaMemcpy2DD2HResult res;
    int count = spitch * height;

    WARN(3, "cudaMemcpy2D(");
    if (!rcuContext) createRcuContext();
    if (maxcount == 0) {
        res.buf.RCbuf_val = NULL;
    }
    if (maxcount < count) {
        res.buf.RCbuf_val = (char*)realloc(res.buf.RCbuf_val, count);
        maxcount = count;
    }
    res.buf.RCbuf_len = count;
    err = cudaMemcpy2D(res.buf.RCbuf_val, dpitch, (void *)src, spitch, width, height, cudaMemcpyDeviceToHost);
    check_cuda_error(err);
    res.err = err;
    WARN(3, "0x%08llx, %d, 0x%08llx, %d, %d, %d, %s) done. 2D buf size : %d\n",
         (unsigned long)res.buf.RCbuf_val, dpitch, src, spitch, width, height, dscudaMemcpyKindName(cudaMemcpyDeviceToHost), count);
    return &res;
}

dscudaResult *
dscudamemcpy2dd2did_1_svc(RCadr dst, RCsize dpitch, RCadr src, RCsize spitch, RCsize width, RCsize height, struct svc_req *sr)
{
    cudaError_t err;
    static dscudaResult res;
    WARN(3, "cudaMemcpy2D(");
    err = cudaMemcpy2D((void *)dst, dpitch, (void *)src, spitch, width, height, cudaMemcpyDeviceToDevice);
    check_cuda_error(err);
    res.err = err;
    WARN(3, "0x%08llx, %d, 0x%08llx, %d, %d, %d, %s) done.\n",
         dst, dpitch, src, spitch, width, height, dscudaMemcpyKindName(cudaMemcpyDeviceToDevice));
    return &res;
}

dscudaResult *
dscudamemset2did_1_svc(RCadr dst, RCsize pitch, int value, RCsize width, RCsize height, struct svc_req *sq)
{
    static dscudaResult res;
    cudaError_t err;

    WARN(3, "cudaMemset2D(");
    if (!rcuContext) createRcuContext();
    err = cudaMemset2D((void *)dst, pitch, value, width, height);
    check_cuda_error(err);
    res.err = err;
    WARN(3, "0x%08llx, %d, %d, %d, %d) done.\n", dst, pitch, value, width, height);
    return &res;
}


/*
 * Texture Reference Management
 */

dscudaCreateChannelDescResult *
dscudacreatechanneldescid_1_svc(int x, int y, int z, int w, RCchannelformat f, struct svc_req *sr)
{
    static dscudaCreateChannelDescResult res;
    cudaChannelFormatDesc desc;

    WARN(3, "cudaCreateChannelDesc(");
    if (!rcuContext) createRcuContext();
    desc = cudaCreateChannelDesc(x, y, z, w, (enum cudaChannelFormatKind)f);
    res.x = desc.x;
    res.y = desc.y;
    res.z = desc.z;
    res.w = desc.w;
    res.f = desc.f;
    WARN(3, "%d, %d, %d, %d, %d) done.\n", x, y, z, w, f)
    return &res;
}

dscudaGetChannelDescResult *
dscudagetchanneldescid_1_svc(RCadr array, struct svc_req *sr)
{
    static dscudaGetChannelDescResult res;
    cudaError_t err;
    cudaChannelFormatDesc desc;

    WARN(3, "cudaGetChannelDesc(");
    if (!rcuContext) createRcuContext();
    err = cudaGetChannelDesc(&desc, (const struct cudaArray*)array);
    res.err = err;
    res.x = desc.x;
    res.y = desc.y;
    res.z = desc.z;
    res.w = desc.w;
    res.f = desc.f;
    WARN(3, "0x%08llx, 0x&08llx) done.\n", &desc, array)
    return &res;
}

static cudaError_t
setTextureParams(CUtexref texref, RCtexture texbuf, char *texname, CUDA_ARRAY_DESCRIPTOR *descp = NULL)
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

dscudaBindTextureResult *
dscudabindtextureid_1_svc(int moduleid, char *texname, RCadr devPtr, RCsize size, RCtexture texbuf, struct svc_req *sr)
{
    static dscudaBindTextureResult res;
    cudaError_t err;
    CUtexref texref;
    Module *mp = Modulelist + moduleid;

    if (!rcuContext) createRcuContext();

    err = (cudaError_t)cuModuleGetTexRef(&texref, mp->handle, texname);
    WARN(3, "cuModuleGetTexRef(0x%08llx, 0x%08llx, %s) : module: %s\n",
         &texref, mp->handle, texname, mp->name);
    if (err != cudaSuccess) {
        check_cuda_error(err);
        res.err = err;
        return &res;
    }

    err = setTextureParams(texref, texbuf, texname);
    if (err != cudaSuccess) {
        res.err = err;
        return &res;
    }

    WARN(4, "cuTexRefSetAddress(0x%08llx, 0x%08llx, 0x%08llx, %d)\n", &res.offset, texref, devPtr, size);
    err = (cudaError_t)cuTexRefSetAddress((size_t *)&res.offset, texref, (CUdeviceptr)devPtr, size);
    if (err != cudaSuccess) {
        check_cuda_error(err);
        res.err = err;
        return &res;
    }
    res.err = err;

    return &res;
}

dscudaBindTexture2DResult *
dscudabindtexture2did_1_svc(int moduleid, char *texname, RCadr devPtr, RCsize width, RCsize height, RCsize pitch, RCtexture texbuf, struct svc_req *sr)
{
    static dscudaBindTexture2DResult res;
    cudaError_t err;
    CUtexref texref;
    Module *mp = Modulelist + moduleid;
    CUDA_ARRAY_DESCRIPTOR desc;

    if (!rcuContext) createRcuContext();

    err = (cudaError_t)cuModuleGetTexRef(&texref, mp->handle, texname);
    WARN(3, "cuModuleGetTexRef(0x%08llx, 0x%08llx, %s) : module: %s\n",
         &texref, mp->handle, texname, mp->name);
    if (err != cudaSuccess) {
        check_cuda_error(err);
        res.err = err;
        return &res;
    }

    err = setTextureParams(texref, texbuf, texname, &desc);
    if (err != cudaSuccess) {
        res.err = err;
        return &res;
    }
    desc.Height = height;
    desc.Width  = width;

    WARN(4, "cuTexRefSetAddress2D(0x%08llx, 0x%08llx, 0x%08llx, %d)\n", texref, desc, devPtr, pitch);
    err = (cudaError_t)cuTexRefSetAddress2D(texref, &desc, (CUdeviceptr)devPtr, pitch);
    if (err != cudaSuccess) {
        check_cuda_error(err);
        res.err = err;
        return &res;
    }
    res.err = err;

    unsigned int align = CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT;
    unsigned long int roundup_adr = ((devPtr - 1) / align + 1) * align;
    res.offset = roundup_adr - devPtr;
    WARN(4, "align:0x%x  roundup_adr:0x%08llx  devPtr:0x%08llx  offset:0x%08llx\n",
         align, roundup_adr, devPtr, res.offset);
    return &res;
}

dscudaResult *
dscudabindtexturetoarrayid_1_svc(int moduleid, char *texname, RCadr array, RCtexture texbuf, struct svc_req *sr)
{
    static dscudaResult res;
    cudaError_t err;
    CUtexref texref;
    Module *mp = Modulelist + moduleid;

    if (!rcuContext) createRcuContext();

    err = (cudaError_t)cuModuleGetTexRef(&texref, mp->handle, texname);
    WARN(3, "cuModuleGetTexRef(0x%08llx, 0x%08llx, %s) : module: %s  moduleid:%d\n",
         &texref, mp->handle, texname, mp->name, moduleid);
    if (err != cudaSuccess) {
        check_cuda_error(err);
        res.err = err;
        return &res;
    }

    err = setTextureParams(texref, texbuf, texname);
    if (err != cudaSuccess) {
        res.err = err;
        return &res;
    }

    WARN(4, "cuTexRefSetArray(0x%08llx, 0x%08llx, %d)\n", texref, array, CU_TRSA_OVERRIDE_FORMAT);
    err = (cudaError_t)cuTexRefSetArray(texref, (CUarray)array, CU_TRSA_OVERRIDE_FORMAT);
    if (err != cudaSuccess) {
        check_cuda_error(err);
        res.err = err;
        return &res;
    }
    res.err = err;
    return &res;
}

dscudaResult *
dscudaunbindtextureid_1_svc(RCtexture texrefbuf, struct svc_req *sr)
{
    static dscudaResult res;
    cudaError_t err = cudaSuccess;

    WARN(4, "Current implementation of cudaUnbindTexture() does nothing "
         "but returning cudaSuccess.\n");

    res.err = err;
    return &res;
}
