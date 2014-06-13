#define USE_IBV 1

#include "libdscuda.h"
#include "ibv_rdma.h"

static void setupConnection(int idev, RCServer_t *sp);
static int on_addr_resolved(struct rdma_cm_id *id);
static int on_route_resolved(struct rdma_cm_id *id);
static int on_connection(struct rdma_cm_id *id);
static int ibvDscudaLoadModule(unsigned int ipaddr, pid_t pid, char *modulename, char *modulebuf, int vdevid, int raidid);
static void ibvDscudaLaunchKernel(int moduleid, int kid, char *kname,
                                 int *gdim, int *bdim, RCsize smemsize, RCstream stream,
                                 int narg, IbvArg *arg, int vdevid, int raidid);

static struct rdma_cm_id *Cmid[RC_NVDEVMAX][RC_NREDUNDANCYMAX];

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
    int portid = RC_IBV_IP_PORT_BASE + cid;
    char *service;

    WARN(2, "Requesting IB Verb connection to %s:%d (port %d)...\n", sp->ip, cid, portid);

    asprintf(&service, "%d", portid);
    TEST_NZ(getaddrinfo(sp->ip, service, NULL, &addr));
    TEST_Z(ec = rdma_create_event_channel());
    TEST_NZ(rdma_create_id(ec, &cmid, NULL, RDMA_PS_TCP));
    Cmid[idev][id] = cmid;
    TEST_NZ(rdma_resolve_addr(cmid, NULL, addr->ai_addr, RC_IBV_TIMEOUT));
    freeaddrinfo(addr);
    wait_event(ec, RDMA_CM_EVENT_ADDR_RESOLVED,  on_addr_resolved);
    wait_event(ec, RDMA_CM_EVENT_ROUTE_RESOLVED, on_route_resolved);
    set_on_completion_handler(on_completion_client);
    wait_event(ec, RDMA_CM_EVENT_ESTABLISHED,    on_connection);
    send_mr((IbvConnection *)cmid->context);
    sleep(1);
    WARN(2, "connection established\n");
}

static int
on_addr_resolved(struct rdma_cm_id *id)
{
    WARN(3, "  IB Verb address resolved.\n");
    build_connection(id);
    TEST_NZ(rdma_resolve_route(id, RC_IBV_TIMEOUT));

    return 0;
}

static int
on_route_resolved(struct rdma_cm_id *id)
{
    struct rdma_conn_param cm_params;

    WARN(3, "  IB Verb route resolved.\n");
    build_params(&cm_params);
    TEST_NZ(rdma_connect(id, &cm_params));

    return 0;
}

static int
on_connection(struct rdma_cm_id *id)
{
    WARN(3, "  IB Verb connection established.\n");
    ((IbvConnection *)id->context)->connected = 1;
    return 0;
}

/*
 * Dscuda client-side counterpart for CUDA runtime API:
 */

/*
 * Thread Management
 */



/*
 * Error Handling
 */

cudaError_t
cudaGetLastError(void)
{
    WARN(2, "a dummy call to cudaGetLastError()\n");
    return cudaSuccess;
}

cudaError_t
cudaPeekAtLastError(void)
{
    WARN(2, "a dummy call to cudaPeekAtLastError()\n");
    return cudaSuccess;
}

const char *
cudaGetErrorString(cudaError_t error)
{
    static char str[256];

    initClient();
    WARN(3, "cudaGetErrorString()...");
    Vdev_t *vdev = Vdev + Vdevid;
    for (int i = 0; i < vdev->nredundancy; i++) {
        IbvConnection *conn = (IbvConnection *)Cmid[Vdevid][i]->context;
        IbvGetErrorStringInvokeHdr *spkt = (IbvGetErrorStringInvokeHdr *)conn->rdma_local_region;
        IbvGetErrorStringReturnHdr *rpkt = (IbvGetErrorStringReturnHdr *)conn->rdma_remote_region;

        // pack send data.
        int spktsize = sizeof(IbvGetErrorStringInvokeHdr);
        spkt->method = RCMethodGetErrorString;
        spkt->err = error;
        WARN(3, "spktsize:%d\n", spktsize);
        WARN(3, "Vdevid:%d\n", Vdevid);

        // perform remote call.
        rpkt->method = RCMethodNone;
        wait_ready_to_rdma(conn);
        kickoff_rdma(conn, spktsize);
        while (!rpkt->method) {
            // wait the returning packet.
        }

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

    initClient();
    WARN(3, "cudaRuntimeGetVersion(0x%08llx)...", (unsigned long)runtimeVersion);
    Vdev_t *vdev = Vdev + Vdevid;
    for (int i = 0; i < vdev->nredundancy; i++) {
        IbvConnection *conn = (IbvConnection *)Cmid[Vdevid][i]->context;
        IbvRuntimeGetVersionInvokeHdr *spkt = (IbvRuntimeGetVersionInvokeHdr *)conn->rdma_local_region;
        IbvRuntimeGetVersionReturnHdr *rpkt = (IbvRuntimeGetVersionReturnHdr *)conn->rdma_remote_region;

        // pack send data.
        int spktsize = sizeof(IbvRuntimeGetVersionInvokeHdr);
        spkt->method = RCMethodRuntimeGetVersion;
        WARN(3, "spktsize:%d\n", spktsize);
        WARN(3, "Vdevid:%d\n", Vdevid);

        // perform remote call.
        rpkt->method = RCMethodNone;
        wait_ready_to_rdma(conn);
        kickoff_rdma(conn, spktsize);
        while (!rpkt->method) {
            // wait the returning packet.
        }

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

    initClient();
    WARN(3, "cudaDeviceSynchronize()...");
    Vdev_t *vdev = Vdev + Vdevid;
    for (int i = 0; i < vdev->nredundancy; i++) {
        IbvConnection *conn = (IbvConnection *)Cmid[Vdevid][i]->context;
        IbvDeviceSynchronizeInvokeHdr *spkt = (IbvDeviceSynchronizeInvokeHdr *)conn->rdma_local_region;
        IbvDeviceSynchronizeReturnHdr *rpkt = (IbvDeviceSynchronizeReturnHdr *)conn->rdma_remote_region;

        // pack send data.
        int spktsize = sizeof(IbvDeviceSynchronizeInvokeHdr);
        spkt->method = RCMethodDeviceSynchronize;
        WARN(3, "spktsize:%d\n", spktsize);
        WARN(3, "Vdevid:%d\n", Vdevid);

        // perform remote call.
        rpkt->method = RCMethodNone;
        wait_ready_to_rdma(conn);
        kickoff_rdma(conn, spktsize);
        while (!rpkt->method) {
            // wait the returning packet.
        }

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

    initClient();
    WARN(3, "cudaMalloc(0x%08llx, %d)...", (unsigned long)devAdrPtr, size);
    Vdev_t *vdev = Vdev + Vdevid;
    for (int i = 0; i < vdev->nredundancy; i++) {
        IbvConnection *conn = (IbvConnection *)Cmid[Vdevid][i]->context;
        IbvMallocInvokeHdr *spkt = (IbvMallocInvokeHdr *)conn->rdma_local_region;
        IbvMallocReturnHdr *rpkt = (IbvMallocReturnHdr *)conn->rdma_remote_region;

        // pack send data.
        int spktsize = sizeof(IbvMallocInvokeHdr);
        spkt->method = RCMethodMalloc;
        spkt->size = size;
        WARN(3, "spktsize:%d  size:%d\n", spktsize, size);

        // perform remote call.
        rpkt->method = RCMethodNone;
        wait_ready_to_rdma(conn);
        kickoff_rdma(conn, spktsize);
        while (!rpkt->method) {
            // wait the returning packet.
        }

        // unpack returned data.
        err = rpkt->err;
        WARN(3, "cudaMalloc err : %d\n", err);

        if (i == 0) {
            devadr = (void*)rpkt->devAdr;
        }
        else if (devadr != (void*)rpkt->devAdr) {
            WARN(0, "cudaMalloc() on device%d allocated memory address 0x%lx "
                 "different from that of device0, 0x%lx.\n", i, (void*)rpkt->devAdr, devadr);
            exit(1);
        }
    }
    *devAdrPtr = devadr;
    WARN(3, "done. *devAdrPtr:0x%08llx\n", *devAdrPtr);


    return err;
}

cudaError_t
cudaFree(void *mem)
{
    cudaError_t err = cudaSuccess;

    initClient();
    WARN(3, "cudaFree(0x%08llx)...", (unsigned long)mem);
    Vdev_t *vdev = Vdev + Vdevid;
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        IbvConnection *conn = (IbvConnection *)Cmid[Vdevid][i]->context;
        IbvFreeInvokeHdr *spkt = (IbvFreeInvokeHdr *)conn->rdma_local_region;
        IbvFreeReturnHdr *rpkt = (IbvFreeReturnHdr *)conn->rdma_remote_region;

        // pack send data.
        int spktsize = sizeof(IbvFreeInvokeHdr);
        spkt->method = RCMethodFree;
        spkt->devAdr = (RCadr)mem;
        WARN(3, "spktsize:%d\n", spktsize);

        // perform remote call.
        rpkt->method = RCMethodNone;
        wait_ready_to_rdma(conn);
        kickoff_rdma(conn, spktsize);
        while (!rpkt->method) {
            // wait the returning packet.
        }

        // unpack returned data.
        err = rpkt->err;
        WARN(3, "cudaFree err : %d\n", err);
    }

    WARN(3, "done.\n");
    return err;
}

cudaError_t
cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind)
{
    cudaError_t err = cudaSuccess;
    Vdev_t *vdev;

    initClient();

    WARN(3, "cudaMemcpy(0x%08lx, 0x%08lx, %d, %s)...",
            (unsigned long)dst, (unsigned long)src, count, dscudaMemcpyKindName(kind));

    if (RDMA_BUFFER_SIZE < count) {
        WARN(0, "count (=%d) exceeds RDMA_BUFFER_SIZE (=%d).\n",
             count, RDMA_BUFFER_SIZE);
        exit(1);
    }


    switch (kind) {
      case cudaMemcpyDeviceToHost:
        vdev = Vdev + Vdevid;
        for (int i = 0; i < vdev->nredundancy; i++) {
            IbvConnection *conn = (IbvConnection *)Cmid[Vdevid][i]->context;
            IbvMemcpyD2HInvokeHdr *spkt = (IbvMemcpyD2HInvokeHdr *)conn->rdma_local_region;
            IbvMemcpyD2HReturnHdr *rpkt = (IbvMemcpyD2HReturnHdr *)conn->rdma_remote_region;

            // pack send data.
            int spktsize = sizeof(IbvMemcpyD2HInvokeHdr);
            spkt->method = RCMethodMemcpyD2H;
            spkt->count = count;
            spkt->srcadr = (RCadr)src;
            WARN(3, "spktsize:%d  count:%d\n", spktsize, count);

            // perform remote call.
            rpkt->method = RCMethodNone;
            wait_ready_to_rdma(conn);
            kickoff_rdma(conn, spktsize);
            while (!rpkt->method) {
                // wait the returning packet.
            }

            // unpack returned data.
            err = rpkt->err;
            WARN(3, "cudaMemcpy D2H err : %d\n", err);

            if (i == 0) {
                memcpy(dst, &rpkt->dstbuf, count);
            }
            else if (bcmp(dst, &rpkt->dstbuf, count) != 0) {
                if (errorHandler) {
                    errorHandler(errorHandlerArg);
                }
            }
            else {
                WARN(3, "cudaMemcpy() data copied from device%d matched with that from device0.\n", i);
            }
        }
        break;
      case cudaMemcpyHostToDevice:
        vdev = Vdev + Vdevid;
        for (int i = 0; i < vdev->nredundancy; i++) {
            IbvConnection *conn = (IbvConnection *)Cmid[Vdevid][i]->context;
            IbvMemcpyH2DInvokeHdr *spkt = (IbvMemcpyH2DInvokeHdr *)conn->rdma_local_region;
            IbvMemcpyH2DReturnHdr *rpkt = (IbvMemcpyH2DReturnHdr *)conn->rdma_remote_region;

            // pack send data.
            int spktsize = sizeof(IbvMemcpyH2DInvokeHdr) + count;
            spkt->method = RCMethodMemcpyH2D;
            spkt->count = count;
            spkt->dstadr = (RCadr)dst;
            memcpy(&spkt->srcbuf, src, count);
            WARN(3, "spktsize:%d  count:%d\n", spktsize, count);

            // perform remote call.
            rpkt->method = RCMethodNone;
            wait_ready_to_rdma(conn);
            kickoff_rdma(conn, spktsize);
            while (!rpkt->method) {
                // wait the returning packet.
            }

            // unpack returned data.
            err = rpkt->err;
            WARN(3, "cudaMemcpy H2D err : %d\n", err);
        }
        break;
      case cudaMemcpyDeviceToDevice:
        vdev = Vdev + Vdevid;
        for (int i = 0; i < vdev->nredundancy; i++) {
            WARN(0, "Unsupported value for cudaMemcpyKind : %s\n", dscudaMemcpyKindName(kind));
            exit(1);
        }
        break;
      default:
        WARN(0, "Unsupported value for cudaMemcpyKind : %s\n", dscudaMemcpyKindName(kind));
        exit(1);
    }

    WARN(3, "done.\n");

    return err;
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
        IbvConnection *conn = (IbvConnection *)Cmid[device][i]->context;
        IbvGetDevicePropertiesInvokeHdr *spkt = (IbvGetDevicePropertiesInvokeHdr *)conn->rdma_local_region;
        IbvGetDevicePropertiesReturnHdr *rpkt = (IbvGetDevicePropertiesReturnHdr *)conn->rdma_remote_region;

        // pack send data.
        int spktsize = sizeof(IbvGetDevicePropertiesInvokeHdr);
        spkt->method = RCMethodGetDeviceProperties;
        spkt->device = device;
        WARN(3, "spktsize:%d  device:%d\n", spktsize, device);

        // perform remote call.
        rpkt->method = RCMethodNone;
        wait_ready_to_rdma(conn);
        kickoff_rdma(conn, spktsize);
        while (!rpkt->method) {
            // wait the returning packet.
        }

        // unpack returned data.
        err = rpkt->err;
        WARN(3, "cudaGetDeviceProperties err : %d\n", err);
        memcpy(prop, &rpkt->prop, sizeof(cudaDeviceProp));
    }
    WARN(3, "done.\n");

    return err;
}

static int
ibvDscudaLoadModule(unsigned int ipaddr, pid_t pid, char *modulename, char *modulebuf, int vdevid, int raidid)
{
    cudaError_t err = cudaSuccess;
    IbvConnection *conn = (IbvConnection *)Cmid[vdevid][raidid]->context;
    IbvDscudaLoadModuleInvokeHdr *spkt = (IbvDscudaLoadModuleInvokeHdr *)conn->rdma_local_region;
    IbvDscudaLoadModuleReturnHdr *rpkt = (IbvDscudaLoadModuleReturnHdr *)conn->rdma_remote_region;
    int moduleid;
    int namelen = strlen(modulename);
    int imagelen = strlen(modulebuf);

    if (RC_KMODULENAMELEN <= namelen) {
        WARN(0, "ibvDscudaLoadModule:modulename too long (%d byte).\n", namelen);
        exit(1);
    }
    if (RC_KMODULEIMAGELEN <= imagelen) {
        WARN(0, "ibvDscudaLoadModule:modulebuf too long (%d byte).\n", imagelen);
        exit(1);
    }

    // pack send data.
    int spktsize = sizeof(IbvDscudaLoadModuleInvokeHdr) + imagelen + 1;
    spkt->method = RCMethodDscudaLoadModule;
    spkt->ipaddr = ipaddr;
    spkt->pid = pid;
    strncpy(spkt->modulename, modulename, RC_KMODULENAMELEN);
    strncpy((char *)&spkt->moduleimage, modulebuf, RC_KMODULEIMAGELEN);
    WARN(3, "spktsize:%d\n", spktsize);

    // perform remote call.
    rpkt->method = RCMethodNone;
    wait_ready_to_rdma(conn);
    kickoff_rdma(conn, spktsize);
    while (!rpkt->method) {
        // wait the returning packet.
    }

    // unpack returned data.
    err = rpkt->err;
    WARN(3, "ibvDscudaLoadModule err : %d\n", err);
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
    IbvConnection *conn = (IbvConnection *)Cmid[vdevid][raidid]->context;
    IbvDscudaLaunchKernelInvokeHdr *spkt = (IbvDscudaLaunchKernelInvokeHdr *)conn->rdma_local_region;
    IbvDscudaLaunchKernelReturnHdr *rpkt = (IbvDscudaLaunchKernelReturnHdr *)conn->rdma_remote_region;
    int k;

    // pack send data.
    int spktsize = sizeof(IbvDscudaLaunchKernelInvokeHdr) + sizeof(IbvArg) * narg;
    spkt->method = RCMethodDscudaLaunchKernel;
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

    // perform remote call.
    rpkt->method = RCMethodNone;
    wait_ready_to_rdma(conn);
    kickoff_rdma(conn, spktsize);
    while (!rpkt->method) {
        // wait the returning packet.
    }

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

    Vdev_t *vdev = Vdev + Vdevid;
    for (int i = 0; i < vdev->nredundancy; i++) {
        ibvDscudaLaunchKernel(moduleid[i], kid, kname,
                             gdim, bdim, smemsize, (RCstream)st->s[i],
                             narg, arg, Vdevid, i);
    }

    mem = RCmappedMemListTop;
    while (mem) {
        cudaMemcpy(mem->pHost, mem->pDevice, mem->size, cudaMemcpyDeviceToHost);
        mem = mem->next;
    }
}

void
dscudaLaunchKernelWrapper(int *moduleid, int kid, char *kname,
                         RCdim3 gdim, RCdim3 bdim, RCsize smemsize, RCstream stream,
                         RCargs args)
{
    // a dummy func.
}

static cudaError_t
ibvDscudaMemcpyToSymbolAsyncH2D(int moduleid, char *symbol, const void *src,
                               size_t count, size_t offset, RCstream stream, int vdevid, int raidid)
{
    cudaError_t err = cudaSuccess;
    IbvConnection *conn = (IbvConnection *)Cmid[vdevid][raidid]->context;
    IbvDscudaMemcpyToSymbolAsyncH2DInvokeHdr *spkt = (IbvDscudaMemcpyToSymbolAsyncH2DInvokeHdr *)conn->rdma_local_region;
    IbvDscudaMemcpyToSymbolAsyncH2DReturnHdr *rpkt = (IbvDscudaMemcpyToSymbolAsyncH2DReturnHdr *)conn->rdma_remote_region;

    int snamelen = strlen(symbol);
    if (RC_SNAMELEN <= snamelen) {
        WARN(0, "ibvDscudaMemcpyToSymbolAsyncH2D:symbol name too long (%d byte).\n", snamelen);
        exit(1);
    }

    // pack send data.
    int spktsize = sizeof(IbvDscudaMemcpyToSymbolAsyncH2DInvokeHdr) + count;
    spkt->method = RCMethodDscudaMemcpyToSymbolAsyncH2D;
    spkt->moduleid = moduleid;
    strncpy(spkt->symbol, symbol, RC_SNAMELEN);
    memcpy((char *)&spkt->src, src, count);
    spkt->count = count;
    spkt->offset = offset;
    spkt->stream = stream;
    WARN(3, "spktsize:%d\n", spktsize);

    // perform remote call.
    rpkt->method = RCMethodNone;
    wait_ready_to_rdma(conn);
    kickoff_rdma(conn, spktsize);
    while (!rpkt->method) {
        // wait the returning packet.
    }

    // unpack returned data.
    err = rpkt->err;
    WARN(3, "ibvDscudaMemcpyToSymbolAsyncH2D err : %d\n", err);
    return err;
}

static cudaError_t
ibvDscudaMemcpyToSymbolAsyncD2D(int moduleid, char *symbol, const void *src,
                               size_t count, size_t offset, RCstream stream, int vdevid, int raidid)
{
    cudaError_t err = cudaSuccess;
    IbvConnection *conn = (IbvConnection *)Cmid[vdevid][raidid]->context;
    IbvDscudaMemcpyToSymbolAsyncD2DInvokeHdr *spkt = (IbvDscudaMemcpyToSymbolAsyncD2DInvokeHdr *)conn->rdma_local_region;
    IbvDscudaMemcpyToSymbolAsyncD2DReturnHdr *rpkt = (IbvDscudaMemcpyToSymbolAsyncD2DReturnHdr *)conn->rdma_remote_region;

    int snamelen = strlen(symbol);
    if (RC_SNAMELEN <= snamelen) {
        WARN(0, "ibvDscudaMemcpyToSymbolAsyncD2D:symbol name too long (%d byte).\n", snamelen);
        exit(1);
    }

    // pack send data.
    int spktsize = sizeof(IbvDscudaMemcpyToSymbolAsyncD2DInvokeHdr);
    spkt->method = RCMethodDscudaMemcpyToSymbolAsyncD2D;
    spkt->moduleid = moduleid;
    strncpy(spkt->symbol, symbol, RC_SNAMELEN);
    spkt->srcadr = (RCadr)src;
    spkt->count = count;
    spkt->offset = offset;
    spkt->stream = stream;
    WARN(3, "spktsize:%d\n", spktsize);

    // perform remote call.
    rpkt->method = RCMethodNone;
    wait_ready_to_rdma(conn);
    kickoff_rdma(conn, spktsize);
    while (!rpkt->method) {
        // wait the returning packet.
    }

    // unpack returned data.
    err = rpkt->err;
    WARN(3, "ibvDscudaMemcpyToSymbolAsyncD2D err : %d\n", err);
    return err;
}

static cudaError_t
ibvDscudaMemcpyFromSymbolAsyncD2H(int moduleid, void *dstbuf, char *symbol,
                               size_t count, size_t offset, RCstream stream, int vdevid, int raidid)
{
    cudaError_t err = cudaSuccess;
    IbvConnection *conn = (IbvConnection *)Cmid[vdevid][raidid]->context;
    IbvDscudaMemcpyFromSymbolAsyncD2HInvokeHdr *spkt = (IbvDscudaMemcpyFromSymbolAsyncD2HInvokeHdr *)conn->rdma_local_region;
    IbvDscudaMemcpyFromSymbolAsyncD2HReturnHdr *rpkt = (IbvDscudaMemcpyFromSymbolAsyncD2HReturnHdr *)conn->rdma_remote_region;

    int snamelen = strlen(symbol);
    if (RC_SNAMELEN <= snamelen) {
        WARN(0, "ibvDscudaMemcpyFromSymbolAsyncD2H:symbol name too long (%d byte).\n", snamelen);
        exit(1);
    }

    // pack send data.
    int spktsize = sizeof(IbvDscudaMemcpyFromSymbolAsyncD2HInvokeHdr);
    spkt->method = RCMethodDscudaMemcpyFromSymbolAsyncD2H;
    spkt->moduleid = moduleid;
    strncpy(spkt->symbol, symbol, RC_SNAMELEN);
    spkt->count = count;
    spkt->offset = offset;
    spkt->stream = stream;
    WARN(3, "spktsize:%d\n", spktsize);

    // perform remote call.
    rpkt->method = RCMethodNone;
    wait_ready_to_rdma(conn);
    kickoff_rdma(conn, spktsize);
    while (!rpkt->method) {
        // wait the returning packet.
    }

    // unpack returned data.
    err = rpkt->err;
    memcpy(dstbuf, (char *)&rpkt->dst, count);

    WARN(3, "ibvDscudaMemcpyFromSymbolAsyncD2H err : %d\n", err);
    return err;
}

static cudaError_t
ibvDscudaMemcpyFromSymbolAsyncD2D(int moduleid, void *dstadr, char *symbol,
                               size_t count, size_t offset, RCstream stream, int vdevid, int raidid)
{
    cudaError_t err = cudaSuccess;
    IbvConnection *conn = (IbvConnection *)Cmid[vdevid][raidid]->context;
    IbvDscudaMemcpyFromSymbolAsyncD2DInvokeHdr *spkt = (IbvDscudaMemcpyFromSymbolAsyncD2DInvokeHdr *)conn->rdma_local_region;
    IbvDscudaMemcpyFromSymbolAsyncD2DReturnHdr *rpkt = (IbvDscudaMemcpyFromSymbolAsyncD2DReturnHdr *)conn->rdma_remote_region;

    int snamelen = strlen(symbol);
    if (RC_SNAMELEN <= snamelen) {
        WARN(0, "ibvDscudaMemcpyFromSymbolAsyncD2D:symbol name too long (%d byte).\n", snamelen);
        exit(1);
    }

    // pack send data.
    int spktsize = sizeof(IbvDscudaMemcpyFromSymbolAsyncD2DInvokeHdr);
    spkt->method = RCMethodDscudaMemcpyFromSymbolAsyncD2D;
    spkt->moduleid = moduleid;
    spkt->dstadr = (RCadr)dstadr;
    strncpy(spkt->symbol, symbol, RC_SNAMELEN);
    spkt->count = count;
    spkt->offset = offset;
    spkt->stream = stream;
    WARN(3, "spktsize:%d\n", spktsize);

    // perform remote call.
    rpkt->method = RCMethodNone;
    wait_ready_to_rdma(conn);
    kickoff_rdma(conn, spktsize);
    while (!rpkt->method) {
        // wait the returning packet.
    }

    // unpack returned data.
    err = rpkt->err;
    WARN(3, "ibvDscudaMemcpyFromSymbolAsyncD2D err : %d\n", err);
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
