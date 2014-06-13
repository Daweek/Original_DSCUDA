#include "libdscuda.h"
#include "ibv_rdma.h"

#define SETUP_IBV_PACKET_BUF(mthd, vid, rid)                            \
    IbvConnection *conn = (IbvConnection *)Cmid[vid][rid]->context;     \
    Ibv ## mthd ## InvokeHdr *spkt = (Ibv ## mthd ## InvokeHdr *)conn->rdma_local_region; \
    Ibv ## mthd ## ReturnHdr *rpkt = (Ibv ## mthd ## ReturnHdr *)conn->rdma_remote_region; \
    int spktsize = sizeof(Ibv ## mthd ## InvokeHdr);                    \
    rdmaWaitReadyToKickoff(conn);                                       \
    spkt->method = RCMethod ## mthd;

static void setupConnection(int idev, RCServer_t *sp);
static int on_addr_resolved(struct rdma_cm_id *id);
static int on_route_resolved(struct rdma_cm_id *id);
static int on_connection(struct rdma_cm_id *id);
static void perform_remote_call(IbvConnection *conn, RCMethod *methodp, int sendsize, RCMethod mthd);

static void ibvDscudaLaunchKernel(int moduleid, int kid, char *kname,
                                  int *gdim, int *bdim, RCsize smemsize, RCstream stream,
                                  int narg, IbvArg *arg, int vdevid, int raidid);

static const int UseIbv = 1;

int
dscudaRemoteCallType(void)
{
    return RC_REMOTECALL_TYPE_IBV;
}

static void
setupConnection(int idev, RCServer_t *sp)
{
    struct addrinfo *addr;
    struct rdma_cm_id *cmid= NULL;
    struct rdma_event_channel *ec = NULL;
    int id = sp->id;
    int cid = sp->cid;
    int sport; // port number of the server. given by the daemon, or calculated from cid.
    char *service;

    if (UseDaemon) { // access to the server via daemon.
        sport = requestDaemonForDevice(sp->ip, cid, UseIbv);
    }
    else { // directly access to the server.
        sport = RC_SERVER_IP_PORT + cid;
    }

    WARN(2, "Requesting IB Verb connection to %s:%d (port %d)...\n", sp->ip, cid, sport);
    asprintf(&service, "%d", sport);
    TEST_NZ(getaddrinfo(sp->ip, service, NULL, &addr));
    TEST_Z(ec = rdma_create_event_channel());
    TEST_NZ(rdma_create_id(ec, &cmid, NULL, RDMA_PS_TCP));
    Cmid[idev][id] = cmid;
    TEST_NZ(rdma_resolve_addr(cmid, NULL, addr->ai_addr, RC_IBV_TIMEOUT));
    freeaddrinfo(addr);
    rdmaWaitEvent(ec, RDMA_CM_EVENT_ADDR_RESOLVED,  on_addr_resolved);
    rdmaWaitEvent(ec, RDMA_CM_EVENT_ROUTE_RESOLVED, on_route_resolved);
    rdmaSetOnCompletionHandler(rdmaOnCompletionClient);
    rdmaWaitEvent(ec, RDMA_CM_EVENT_ESTABLISHED,    on_connection);
    rdmaSendMr((IbvConnection *)cmid->context);
    usleep(100);
    WARN(2, "connection established\n");
}

static int
on_addr_resolved(struct rdma_cm_id *id)
{
    WARN(3, "  IB Verb address resolved.\n");
    rdmaBuildConnection(id, false);
    TEST_NZ(rdma_resolve_route(id, RC_IBV_TIMEOUT));

    return 0;
}

static int
on_route_resolved(struct rdma_cm_id *id)
{
    struct rdma_conn_param cm_params;

    WARN(3, "  IB Verb route resolved.\n");
    rdmaBuildParams(&cm_params);
    TEST_NZ(rdma_connect(id, &cm_params));

    return 0;
}

static int
on_connection(struct rdma_cm_id *id)
{
    WARN(3, "  IB Verb connection established.\n");
    IbvConnection *conn = ((IbvConnection *)id->context);
    conn->connected = 1;

    *(RCMethod *)conn->rdma_remote_region = RCMethodFree;
    // whatever method other than RCMethodNone will do.

    return 0;
}


static int
is_nonblocking_method(RCMethod mthd)
{
    switch (mthd) {
      case RCMethodMemcpyH2D:
      case RCMethodDscudaLaunchKernel:
        //        return 0; // !!!
        return 1;
        break;
      default:
        return 0;
        break;
    }
    return 0;
}

static void
perform_remote_call(IbvConnection *conn, RCMethod *methodp, int sendsize, RCMethod mthd)
{
    while (*methodp == RCMethodNone) {
        // wait the returning packet for the previous non-blocking RPC.
    }
    rdmaWaitReadyToKickoff(conn);
    *methodp = RCMethodNone;
    rdmaKickoff(conn, sendsize);
    if (is_nonblocking_method(mthd)) {
        return; // non-blocking method returns as soon as RDMA finished.
    }

    while (*methodp == RCMethodNone) {
        // wait the returning packet.
    }
}

static void
checkResult(void *rp, RCServer_t *sp)
{
    // a dummy func.
}

/*
 * Dscuda client-side counterpart for CUDA runtime API:
 */

/*
 * Thread Management
 */

cudaError_t
cudaThreadSynchronize(void)
{
    cudaError_t err = cudaSuccess;
    int vid = vdevidIndex();

    initClient();
    WARN(3, "cudaThreadSynchronize()...");
    Vdev_t *vdev = Vdev + Vdevid[vid];
    for (int i = 0; i < vdev->nredundancy; i++) {
        SETUP_IBV_PACKET_BUF(ThreadSynchronize, Vdevid[vid], i);
        WARN(3, "spktsize:%d\n", spktsize);
        WARN(3, "Vdevid[vid]:%d\n", Vdevid[vid]);

        perform_remote_call(conn, &rpkt->method, spktsize, spkt->method);

        // unpack returned data.
        err = rpkt->err;
        WARN(3, "cudaThreadSynchronize err : %d\n", err);
    }
    WARN(3, "done.\n");

    return err;
}

cudaError_t
cudaThreadExit(void)
{
    cudaError_t err = cudaSuccess;
    int vid = vdevidIndex();

    initClient();
    WARN(3, "cudaThreadExit()...");
    Vdev_t *vdev = Vdev + Vdevid[vid];
    for (int i = 0; i < vdev->nredundancy; i++) {
        SETUP_IBV_PACKET_BUF(ThreadExit, Vdevid[vid], i);
        WARN(3, "spktsize:%d\n", spktsize);
        WARN(3, "Vdevid[vid]:%d\n", Vdevid[vid]);

        perform_remote_call(conn, &rpkt->method, spktsize, spkt->method);

        // unpack returned data.
        err = rpkt->err;
        WARN(3, "cudaThreadExit err : %d\n", err);
    }
    WARN(3, "done.\n");

    return err;
}

/*
 * Error Handling
 */

cudaError_t
cudaGetLastError(void)
{
    WARNONCE(2, "a dummy call to cudaGetLastError()\n");
    return cudaSuccess;
}

cudaError_t
cudaPeekAtLastError(void)
{
    WARNONCE(2, "a dummy call to cudaPeekAtLastError()\n");
    return cudaSuccess;
}

const char *
cudaGetErrorString(cudaError_t error)
{
    static char str[256];
    int vid = vdevidIndex();

    initClient();
    WARN(3, "cudaGetErrorString()...");
    Vdev_t *vdev = Vdev + Vdevid[vid];
    for (int i = 0; i < vdev->nredundancy; i++) {
        SETUP_IBV_PACKET_BUF(GetErrorString, Vdevid[vid], i);

        // pack send data.
        spkt->err = error;
        WARN(3, "spktsize:%d\n", spktsize);
        WARN(3, "Vdevid[vid]:%d\n", Vdevid[vid]);

        perform_remote_call(conn, &rpkt->method, spktsize, spkt->method);

        // unpack returned data.
        WARN(3, "cudaGetErrorString errmsg : %s\n", &rpkt->errmsg);
        strncpy(str, (char *)&rpkt->errmsg, strlen((char *)&rpkt->errmsg) + 1);
    }
    WARN(3, "done.\n");

    return str;
}

/*
 * Device Management
 */

cudaError_t
cudaRuntimeGetVersion(int *runtimeVersion)
{
    cudaError_t err = cudaSuccess;
    int vid = vdevidIndex();

    initClient();
    WARN(3, "cudaRuntimeGetVersion(0x%08llx)...", (unsigned long)runtimeVersion);
    Vdev_t *vdev = Vdev + Vdevid[vid];
    for (int i = 0; i < vdev->nredundancy; i++) {
        SETUP_IBV_PACKET_BUF(RuntimeGetVersion, Vdevid[vid], i);
        WARN(3, "spktsize:%d\n", spktsize);
        WARN(3, "Vdevid[vid]:%d\n", Vdevid[vid]);

        perform_remote_call(conn, &rpkt->method, spktsize, spkt->method);

        // unpack returned data.
        err = rpkt->err;
        WARN(3, "cudaRuntimeGetVersion err : %d\n", err);
        *runtimeVersion = rpkt->version;
    }
    WARN(3, "done.\n");

    return err;
}

cudaError_t
cudaDeviceSynchronize(void)
{
    cudaError_t err = cudaSuccess;
    int vid = vdevidIndex();

    initClient();
    WARN(3, "cudaDeviceSynchronize()...");
    Vdev_t *vdev = Vdev + Vdevid[vid];
    for (int i = 0; i < vdev->nredundancy; i++) {
        SETUP_IBV_PACKET_BUF(DeviceSynchronize, Vdevid[vid], i);
        WARN(3, "spktsize:%d\n", spktsize);
        WARN(3, "Vdevid[vid]:%d\n", Vdevid[vid]);

        perform_remote_call(conn, &rpkt->method, spktsize, spkt->method);

        // unpack returned data.
        err = rpkt->err;
        WARN(3, "cudaDeviceSynchronize err : %d\n", err);
    }
    WARN(3, "done.\n");

    return err;
}

cudaError_t
cudaDeviceReset(void)
{
    WARN(3, "a dummy call to cudaDeviceReset()\n");
    return cudaSuccess;
}

/*
 * Execution Control
 */

cudaError_t
cudaFuncSetCacheConfig(const char * func, enum cudaFuncCache cacheConfig)
{
    cudaError_t err = cudaSuccess;
    WARN(1, "Current implementation of cudaFuncSetCacheConfig() does nothing "
         "but returning cudaSuccess.\n");
    err = cudaSuccess;
    return err;
}

/*
 * Memory Management
 */
cudaError_t
cudaMalloc(void **devAdrPtr, size_t size)
{
    cudaError_t err = cudaSuccess;
    void *devadr;
    int vid = vdevidIndex();
    void *adrs[RC_NREDUNDANCYMAX];

    initClient();
    WARN(3, "cudaMalloc(0x%08llx, %d)...", (unsigned long)devAdrPtr, size);
    Vdev_t *vdev = Vdev + Vdevid[vid];
    for (int i = 0; i < vdev->nredundancy; i++) {
        SETUP_IBV_PACKET_BUF(Malloc, Vdevid[vid], i);
        spkt->size = size;
        WARN(3, "spktsize:%d  size:%d\n", spktsize, size);

        perform_remote_call(conn, &rpkt->method, spktsize, spkt->method);

        // unpack returned data.
        if (rpkt->err != cudaSuccess) {
            err = (cudaError_t)rpkt->err;
        }
        adrs[i] = (void*)rpkt->devAdr;
    }
    RCuvaRegister(Vdevid[vid], adrs, size);
    *devAdrPtr = dscudaUvaOfAdr(adrs[0], Vdevid[vid]);

    WARN(3, "done. *devAdrPtr:0x%08llx\n", *devAdrPtr);

    return err;
}

cudaError_t
cudaFree(void *mem)
{
    cudaError_t err = cudaSuccess;
    int vid = vdevidIndex();

    initClient();
    WARN(3, "cudaFree(0x%08llx)...", (unsigned long)mem);
    Vdev_t *vdev = Vdev + Vdevid[vid];
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        SETUP_IBV_PACKET_BUF(Free, Vdevid[vid], i);
        spkt->devAdr = (RCadr)mem;
        WARN(3, "spktsize:%d\n", spktsize);

        perform_remote_call(conn, &rpkt->method, spktsize, spkt->method);

        // unpack returned data.
        if (rpkt->err != cudaSuccess) {
            err = rpkt->err;
        }
    }

    RCuvaUnregister(mem);

    WARN(3, "done.\n");

    return err;
}

cudaError_t
cudaMemcpyH2D(void *dst, const void *src, size_t count, int vdevid)
{
    cudaError_t err = cudaSuccess;
    Vdev_t *vdev;

    vdev = Vdev + vdevid;
    for (int i = 0; i < vdev->nredundancy; i++) {
        SETUP_IBV_PACKET_BUF(MemcpyH2D, vdevid, i);
        spktsize += count;
        spkt->count = count;
        spkt->dstadr = (RCadr)dst;

        WARN(3, "spktsize:%d  count:%d\n", spktsize, count);

#if 1
        memcpy(&spkt->srcbuf, src, count);
        perform_remote_call(conn, &rpkt->method, spktsize, spkt->method);
#else
        while (rpkt->method == RCMethodNone) {
            // wait the returning packet for the previous non-blocking RPC.
        }
        rdmaWaitReadyToKickoff(conn);
        rpkt->method = RCMethodNone;
        rdmaPipelinedKickoff(conn, spktsize, (char *)&spkt->srcbuf, (char *)src, count);
#endif

        // unpack returned data.
        err = rpkt->err;
        WARN(3, "cudaMemcpy H2D err : %d\n", err);
    }

    return err;
}

cudaError_t
cudaMemcpyD2H(void *dst, const void *src, size_t count, int vdevid)
{
    cudaError_t err = cudaSuccess;
    Vdev_t *vdev;

    vdev = Vdev + vdevid;
    for (int i = 0; i < vdev->nredundancy; i++) {
        SETUP_IBV_PACKET_BUF(MemcpyD2H, vdevid, i);
        spkt->count = count;
        spkt->srcadr = (RCadr)src;
        WARN(3, "spktsize:%d  count:%d\n", spktsize, count);

        perform_remote_call(conn, &rpkt->method, spktsize, spkt->method);

        // unpack returned data.
        err = rpkt->err;
        WARN(3, "cudaMemcpy D2H err : %d\n", err);

        if (i == 0) {
            memcpy(dst, &rpkt->dstbuf, count);
        }
        else if (bcmp(dst, &rpkt->dstbuf, count) != 0) {
            WARN(1, "\n\ncudaMemcpy() data copied from device%d & device0 UNMATCHED.\n\n\n", i);
            if (autoVerb) {
                cudaMemcpyArgs args;
                args.dst = dst;
                args.src = (void *)src;
                args.count = count;
                args.kind = cudaMemcpyDeviceToHost;
                dscudaVerbAddHist(dscudaMemcpyD2HId, (void *)&args);
                dscudaVerbRecallHist();
                break;
            }
            else if (errorHandler) {
                errorHandler(errorHandlerArg);
            }
        }
        else {
            WARN(3, "cudaMemcpy() data copied from device%d & device0 matched.\n", i);
        }
    }

    return err;
}

cudaError_t
cudaMemcpyD2D(void *dst, const void *src, size_t count, int vdevid)
{
    cudaError_t err = cudaSuccess;
    Vdev_t *vdev;

    vdev = Vdev + vdevid;
    for (int i = 0; i < vdev->nredundancy; i++) {
        SETUP_IBV_PACKET_BUF(MemcpyD2D, vdevid, i);
        spkt->count = count;
        spkt->srcadr = (RCadr)src;
        spkt->dstadr = (RCadr)dst;

        WARN(3, "spktsize:%d  count:%d\n", spktsize, count);

        perform_remote_call(conn, &rpkt->method, spktsize, spkt->method);

        // unpack returned data.
        err = rpkt->err;
        WARN(3, "cudaMemcpy D2D err : %d\n", err);
    }

    return err;
}

static cudaError_t
cudaMemcpyP2P(void *dst, int ddev, const void *src, int sdev, size_t count)
{
    cudaError_t err = cudaSuccess;
    int dev0;
    int pgsz = 4096;
    static int bufsize = 0;
    static char *buf = NULL;

    if (bufsize < count) {
        bufsize = ((count - 1) / pgsz + 1) * pgsz;
        buf = (char *)realloc(buf, bufsize);
        if (!buf) {
            perror("cudaMemcpyP2P");
            exit(1);
        }
    }

    cudaGetDevice(&dev0);

    if (sdev != dev0) {
        cudaSetDevice(sdev);
    }
    err = cudaMemcpy(buf, src, count, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        if (sdev != dev0) {
            cudaSetDevice(dev0);
        }
        return err;
    }
    if (ddev != sdev) {
        cudaSetDevice(ddev);
    }
    err = cudaMemcpy(dst, buf, count, cudaMemcpyHostToDevice);
    if (ddev != dev0) {
        cudaSetDevice(dev0);
    }
    return err;

}

cudaError_t
cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind)
{
    cudaError_t err = cudaSuccess;
    int vdevid;
    Vdev_t *vdev;
    int vi = vdevidIndex();
    RCuva *suva, *duva;
    int dev0;
    void *lsrc, *ldst;

    initClient();

    WARN(3, "cudaMemcpy(0x%08lx, 0x%08lx, %d, %s)...",
         (unsigned long)dst, (unsigned long)src, count, dscudaMemcpyKindName(kind));

    vdevid = Vdevid[vi];

    lsrc = dscudaAdrOfUva((void *)src);
    ldst = dscudaAdrOfUva(dst);


    //    fprintf(stderr, ">>>> src:0x%016llx  lsrc:0x%016llx\n", src, lsrc);
    //    fprintf(stderr, ">>>> dst:0x%016llx  ldst:0x%016llx\n", dst, ldst);

    switch (kind) {
      case cudaMemcpyDeviceToHost:
        err = cudaMemcpyD2H(ldst, lsrc, count, vdevid);
        break;
      case cudaMemcpyHostToDevice:
        err = cudaMemcpyH2D(ldst, lsrc, count, vdevid);
        break;
      case cudaMemcpyDeviceToDevice:
        err = cudaMemcpyD2D(ldst, lsrc, count, vdevid);
        break;
      case cudaMemcpyDefault:
        cudaGetDevice(&dev0);
        suva = RCuvaQuery((void *)src);
        duva = RCuvaQuery(dst);
        if (!suva && !duva) {
            WARN(0, "cudaMemcpy:invalid argument.\n");
            exit(1);
        }
        else if (!suva) { // sbuf resides in the client.
            if (duva->devid != dev0) {
                cudaSetDevice(duva->devid);
            }
            err = cudaMemcpy(dst, src, count, cudaMemcpyHostToDevice);
            if (duva->devid != dev0) {
                cudaSetDevice(dev0);
            }
        }
        else if (!duva) { // dbuf resides in the client.
            if (suva->devid != dev0) {
                cudaSetDevice(suva->devid);
            }
            err = cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost);
            if (suva->devid != dev0) {
                cudaSetDevice(dev0);
            }
        }
        else {
            err = cudaMemcpyP2P(dst, duva->devid, src, suva->devid, count);
        }
        break;
      default:
        WARN(0, "Unsupported value for cudaMemcpyKind : %s\n", dscudaMemcpyKindName(kind));
        exit(1);
    }

    if (autoVerb) {
        cudaMemcpyArgs args;
        switch (kind) {
          case cudaMemcpyHostToDevice:
            args.dst = dst;
            args.src = (void *)src;
            args.count = count;
            args.kind = kind;
            dscudaVerbAddHist(dscudaMemcpyH2DId, (void *)&args);
            break;

          case cudaMemcpyDeviceToDevice:
            args.dst = dst;
            args.src = (void *)src;
            args.count = count;
            args.kind = kind;
            dscudaVerbAddHist(dscudaMemcpyD2DId, (void *)&args);
            break;

          case cudaMemcpyDeviceToHost:
            dscudaVerbClearHist();
            break;
        }
    }
    WARN(3, "done.\n");
    return err;
}

cudaError_t
cudaMemcpyPeer(void *dst, int ddev, const void *src, int sdev, size_t count)
{
    WARN(3, "cudaMemcpyPeer(0x%08lx, %d, 0x%08lx, %d, %d)...",
         (unsigned long)dst, ddev, (unsigned long)src, sdev, count);

    cudaMemcpyP2P(dst, ddev, src, sdev, count);

    WARN(3, "done.\n");
}

cudaError_t
cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device)
{
    cudaError_t err = cudaSuccess;

    initClient();
    WARN(3, "cudaGetDeviceProperties(0x%08lx, %d)...", (unsigned long)prop, device);

    // Vdev_t *vdev = Vdev + device;
    //    for (int i = 0; i < vdev->nredundancy; i++) {
    for (int i = 0; i < 1; i++) { // performs no redundant call for now.
        SETUP_IBV_PACKET_BUF(GetDeviceProperties, device, i);
        spkt->device = device;
        WARN(3, "spktsize:%d  device:%d\n", spktsize, device);

        perform_remote_call(conn, &rpkt->method, spktsize, spkt->method);

        // unpack returned data.
        err = rpkt->err;
        WARN(3, "cudaGetDeviceProperties err : %d\n", err);
        memcpy(prop, &rpkt->prop, sizeof(cudaDeviceProp));
    }
    WARN(3, "done.\n");

    return err;
}

static int
dscudaLoadModuleLocal(unsigned int ipaddr, pid_t pid, char *modulename, char *modulebuf, int vdevid, int raidid)
{
    cudaError_t err = cudaSuccess;
    SETUP_IBV_PACKET_BUF(DscudaLoadModule, vdevid, raidid);

    int moduleid;
    int namelen = strlen(modulename);
    int imagelen = strlen(modulebuf);

    if (RC_KMODULENAMELEN <= namelen) {
        WARN(0, "dscudaLoadModuleLocal:modulename too long (%d byte).\n", namelen);
        exit(1);
    }
    if (RC_KMODULEIMAGELEN <= imagelen) {
        WARN(0, "dscudaLoadModuleLocal:modulebuf too long (%d byte).\n", imagelen);
        exit(1);
    }

    spktsize += imagelen + 1;
    spkt->ipaddr = ipaddr;
    spkt->pid = pid;
    strncpy(spkt->modulename, modulename, RC_KMODULENAMELEN);
    strncpy((char *)&spkt->moduleimage, modulebuf, RC_KMODULEIMAGELEN);
    WARN(3, "spktsize:%d\n", spktsize);

    perform_remote_call(conn, &rpkt->method, spktsize, spkt->method);

    // unpack returned data.
    err = rpkt->err;
    WARN(3, "dscudaLoadModuleLocal err : %d\n", err);
    moduleid = rpkt->moduleid;

    return moduleid;
}

/*
 * launch a kernel function of id 'kid', defined in a module of id 'moduleid'.
 * 'kid' must be unique inside a single module.
 */

static void
ibvDscudaLaunchKernel(int moduleid, int kid, char *kname,
                      int *gdim, int *bdim, RCsize smemsize, RCstream stream,
                      int narg, IbvArg *arg, int vdevid, int raidid)
{
    cudaError_t err = cudaSuccess;
    int k;

    SETUP_IBV_PACKET_BUF(DscudaLaunchKernel, vdevid, raidid);

    spktsize += sizeof(IbvArg) * narg;
    spkt->moduleid = moduleid;
    spkt->kernelid = kid;
    strncpy(spkt->kernelname, kname, RC_KNAMELEN);
    for (k = 0; k < 3; k++) {
        spkt->gdim[k] = gdim[k];
        spkt->bdim[k] = bdim[k];
    }
    spkt->smemsize = smemsize;
    spkt->stream = stream;
    spkt->narg = narg;
    memcpy((char *)&spkt->args, arg, sizeof(IbvArg) * narg);
    WARN(3, "spktsize:%d\n", spktsize);

    perform_remote_call(conn, &rpkt->method, spktsize, spkt->method);

    // unpack returned data.
    err = rpkt->err;
    WARN(3, "ibvDscudaLaunchKernel err : %d\n", err);

}

void
ibvDscudaLaunchKernelWrapper(int *moduleid, int kid, char *kname,
                             int *gdim, int *bdim, RCsize smemsize, RCstream stream,
                             int narg, IbvArg *arg)
{
    RCmappedMem *mem;
    RCstreamArray *st;
    int vid = vdevidIndex();

    st = RCstreamArrayQuery((cudaStream_t)stream);
    if (!st) {
        WARN(0, "invalid stream : 0x%08llx\n", stream);
        exit(1);
    }

    mem = RCmappedMemListTop;
    while (mem) {
        cudaMemcpy(mem->pDevice, mem->pHost, mem->size, cudaMemcpyHostToDevice);
        mem = mem->next;
    }

    Vdev_t *vdev = Vdev + Vdevid[vid];
    for (int i = 0; i < vdev->nredundancy; i++) {
        ibvDscudaLaunchKernel(moduleid[i], kid, kname,
                              gdim, bdim, smemsize, (RCstream)st->s[i],
                              narg, arg, Vdevid[vid], i);
    }

    mem = RCmappedMemListTop;
    while (mem) {
        cudaMemcpy(mem->pHost, mem->pDevice, mem->size, cudaMemcpyDeviceToHost);
        mem = mem->next;
    }

    if (autoVerb) {
        cudaIbvLaunchKernelArgs args2;
        args2.moduleid = moduleid;
        args2.kid = kid;
        args2.kname = kname;
        args2.gdim = gdim;
        args2.bdim = bdim;
        args2.smemsize = smemsize;
        args2.stream = stream;
        args2.narg = narg;
        args2.arg = arg;
        dscudaVerbAddHist(dscudaLaunchKernelId, (void *)&args2);
    }
}

void
rpcDscudaLaunchKernelWrapper(int *moduleid, int kid, char *kname,
                             RCdim3 gdim, RCdim3 bdim, RCsize smemsize, RCstream stream,
                             RCargs args)
{
    // a dummy func.
}


static cudaError_t
dscudaMemcpyToSymbolH2D(int moduleid, char *symbol, const void *src,
                        size_t count, size_t offset, int vdevid, int raidid)
{
    cudaError_t err = cudaSuccess;
    SETUP_IBV_PACKET_BUF(DscudaMemcpyToSymbolH2D, vdevid, raidid);

    int snamelen = strlen(symbol);
    if (RC_SNAMELEN <= snamelen) {
        WARN(0, "dscudaMemcpyToSymbolH2D:symbol name too long (%d byte).\n", snamelen);
        exit(1);
    }

    // pack send data.
    spktsize += count;
    spkt->moduleid = moduleid;
    strncpy(spkt->symbol, symbol, RC_SNAMELEN);
    memcpy((char *)&spkt->src, src, count);
    spkt->count = count;
    spkt->offset = offset;
    WARN(3, "spktsize:%d\n", spktsize);

    perform_remote_call(conn, &rpkt->method, spktsize, spkt->method);

    // unpack returned data.
    err = rpkt->err;
    WARN(3, "dscudaMemcpyToSymbolH2D err : %d\n", err);
    return err;
}

static cudaError_t
dscudaMemcpyToSymbolD2D(int moduleid, char *symbol, const void *src,
                        size_t count, size_t offset, int vdevid, int raidid)
{
    cudaError_t err = cudaSuccess;
    SETUP_IBV_PACKET_BUF(DscudaMemcpyToSymbolD2D, vdevid, raidid);

    int snamelen = strlen(symbol);
    if (RC_SNAMELEN <= snamelen) {
        WARN(0, "dscudaMemcpyToSymbolD2D:symbol name too long (%d byte).\n", snamelen);
        exit(1);
    }

    spkt->moduleid = moduleid;
    strncpy(spkt->symbol, symbol, RC_SNAMELEN);
    spkt->srcadr = (RCadr)src;
    spkt->count = count;
    spkt->offset = offset;
    WARN(3, "spktsize:%d\n", spktsize);

    perform_remote_call(conn, &rpkt->method, spktsize, spkt->method);

    // unpack returned data.
    err = rpkt->err;
    WARN(3, "dscudaMemcpyToSymbolD2D err : %d\n", err);
    return err;
}

static cudaError_t
dscudaMemcpyFromSymbolD2H(int moduleid, void **dstbuf, char *symbol,
                          size_t count, size_t offset, int vdevid, int raidid)
{
    cudaError_t err = cudaSuccess;
    SETUP_IBV_PACKET_BUF(DscudaMemcpyFromSymbolD2H, vdevid, raidid);

    int snamelen = strlen(symbol);
    if (RC_SNAMELEN <= snamelen) {
        WARN(0, "dscudaMemcpyFromSymbolD2H:symbol name too long (%d byte).\n", snamelen);
        exit(1);
    }

    spkt->moduleid = moduleid;
    strncpy(spkt->symbol, symbol, RC_SNAMELEN);
    spkt->count = count;
    spkt->offset = offset;
    WARN(3, "spktsize:%d\n", spktsize);

    perform_remote_call(conn, &rpkt->method, spktsize, spkt->method);
        
    // unpack returned data.
    err = rpkt->err;
    memcpy(*dstbuf, (char *)&rpkt->dst, count);

    WARN(3, "dscudaMemcpyFromSymbolD2H err : %d\n", err);
    return err;
}

static cudaError_t
dscudaMemcpyFromSymbolD2D(int moduleid, void *dstadr, char *symbol,
                          size_t count, size_t offset, int vdevid, int raidid)
{
    cudaError_t err = cudaSuccess;
    SETUP_IBV_PACKET_BUF(DscudaMemcpyFromSymbolD2D, vdevid, raidid);

    int snamelen = strlen(symbol);
    if (RC_SNAMELEN <= snamelen) {
        WARN(0, "dscudaMemcpyFromSymbolD2D:symbol name too long (%d byte).\n", snamelen);
        exit(1);
    }

    spkt->moduleid = moduleid;
    spkt->dstadr = (RCadr)dstadr;
    strncpy(spkt->symbol, symbol, RC_SNAMELEN);
    spkt->count = count;
    spkt->offset = offset;
    WARN(3, "spktsize:%d\n", spktsize);

    perform_remote_call(conn, &rpkt->method, spktsize, spkt->method);

    // unpack returned data.
    err = rpkt->err;
    WARN(3, "dscudaMemcpyFromSymbolD2D err : %d\n", err);
    return err;
}



static cudaError_t
dscudaMemcpyToSymbolAsyncH2D(int moduleid, char *symbol, const void *src,
                             size_t count, size_t offset, RCstream stream, int vdevid, int raidid)
{
    cudaError_t err = cudaSuccess;
    SETUP_IBV_PACKET_BUF(DscudaMemcpyToSymbolAsyncH2D, vdevid, raidid);

    int snamelen = strlen(symbol);
    if (RC_SNAMELEN <= snamelen) {
        WARN(0, "dscudaMemcpyToSymbolAsyncH2D:symbol name too long (%d byte).\n", snamelen);
        exit(1);
    }

    // pack send data.
    spktsize += count;
    spkt->moduleid = moduleid;
    strncpy(spkt->symbol, symbol, RC_SNAMELEN);
    memcpy((char *)&spkt->src, src, count);
    spkt->count = count;
    spkt->offset = offset;
    spkt->stream = stream;
    WARN(3, "spktsize:%d\n", spktsize);

    perform_remote_call(conn, &rpkt->method, spktsize, spkt->method);

    // unpack returned data.
    err = rpkt->err;
    WARN(3, "dscudaMemcpyToSymbolAsyncH2D err : %d\n", err);
    return err;
}

static cudaError_t
dscudaMemcpyToSymbolAsyncD2D(int moduleid, char *symbol, const void *src,
                             size_t count, size_t offset, RCstream stream, int vdevid, int raidid)
{
    cudaError_t err = cudaSuccess;
    SETUP_IBV_PACKET_BUF(DscudaMemcpyToSymbolAsyncD2D, vdevid, raidid);

    int snamelen = strlen(symbol);
    if (RC_SNAMELEN <= snamelen) {
        WARN(0, "dscudaMemcpyToSymbolAsyncD2D:symbol name too long (%d byte).\n", snamelen);
        exit(1);
    }

    spkt->moduleid = moduleid;
    strncpy(spkt->symbol, symbol, RC_SNAMELEN);
    spkt->srcadr = (RCadr)src;
    spkt->count = count;
    spkt->offset = offset;
    spkt->stream = stream;
    WARN(3, "spktsize:%d\n", spktsize);

    perform_remote_call(conn, &rpkt->method, spktsize, spkt->method);

    // unpack returned data.
    err = rpkt->err;
    WARN(3, "dscudaMemcpyToSymbolAsyncD2D err : %d\n", err);
    return err;
}

static cudaError_t
dscudaMemcpyFromSymbolAsyncD2H(int moduleid, void **dstbuf, char *symbol,
                               size_t count, size_t offset, RCstream stream, int vdevid, int raidid)
{
    cudaError_t err = cudaSuccess;
    SETUP_IBV_PACKET_BUF(DscudaMemcpyFromSymbolAsyncD2H, vdevid, raidid);

    int snamelen = strlen(symbol);
    if (RC_SNAMELEN <= snamelen) {
        WARN(0, "dscudaMemcpyFromSymbolAsyncD2H:symbol name too long (%d byte).\n", snamelen);
        exit(1);
    }

    spkt->moduleid = moduleid;
    strncpy(spkt->symbol, symbol, RC_SNAMELEN);
    spkt->count = count;
    spkt->offset = offset;
    spkt->stream = stream;
    WARN(3, "spktsize:%d\n", spktsize);

    perform_remote_call(conn, &rpkt->method, spktsize, spkt->method);
        
    // unpack returned data.
    err = rpkt->err;
    memcpy(*dstbuf, (char *)&rpkt->dst, count);

    WARN(3, "dscudaMemcpyFromSymbolAsyncD2H err : %d\n", err);
    return err;
}

static cudaError_t
dscudaMemcpyFromSymbolAsyncD2D(int moduleid, void *dstadr, char *symbol,
                               size_t count, size_t offset, RCstream stream, int vdevid, int raidid)
{
    cudaError_t err = cudaSuccess;
    SETUP_IBV_PACKET_BUF(DscudaMemcpyFromSymbolAsyncD2D, vdevid, raidid);

    int snamelen = strlen(symbol);
    if (RC_SNAMELEN <= snamelen) {
        WARN(0, "dscudaMemcpyFromSymbolAsyncD2D:symbol name too long (%d byte).\n", snamelen);
        exit(1);
    }

    spkt->moduleid = moduleid;
    spkt->dstadr = (RCadr)dstadr;
    strncpy(spkt->symbol, symbol, RC_SNAMELEN);
    spkt->count = count;
    spkt->offset = offset;
    spkt->stream = stream;
    WARN(3, "spktsize:%d\n", spktsize);

    perform_remote_call(conn, &rpkt->method, spktsize, spkt->method);

    // unpack returned data.
    err = rpkt->err;
    WARN(3, "dscudaMemcpyFromSymbolAsyncD2D err : %d\n", err);
    return err;
}

/*
 * Stream Management
 */

/*
 * Event Management
 */


cudaError_t
cudaEventCreate(cudaEvent_t *event)
{
    static cudaEvent_t e;
    *event = e;
    WARN(3, "a dummy call to cudaEventCreate()\n");
    return cudaSuccess;
}

cudaError_t
cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags)
{
    static cudaEvent_t e;
    *event = e;
    WARN(3, "a dummy call to cudaEventCreateWithFlags()\n");
    return cudaSuccess;
}

cudaError_t
cudaEventDestroy(cudaEvent_t event)
{
    WARN(3, "a dummy call to cudaEventDestroy()\n");
    return cudaSuccess;
}

cudaError_t
cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end)
{
    *ms = 123.0;
    WARN(3, "a dummy call to cudaEventElapsedTime()\n");
    return cudaSuccess;
}

cudaError_t
cudaEventRecord(cudaEvent_t event, cudaStream_t stream)
{
    WARN(3, "a dummy call to cudaEventRecord()\n");
    return cudaSuccess;
}

cudaError_t
cudaEventSynchronize(cudaEvent_t event)
{
    WARN(3, "a dummy call to cudaEventSynchronize()\n");
    return cudaSuccess;
}

cudaError_t
cudaEventQuery(cudaEvent_t event)
{
    WARN(3, "a dummy call to cudaEventQuery()\n");
    return cudaSuccess;
}

#include "libdscuda.cu"
