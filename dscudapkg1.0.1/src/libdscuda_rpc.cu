#include "libdscuda.h"

static CLIENT *Clnt[RC_NVDEVMAX][RC_NREDUNDANCYMAX];

int
dscudaRemoteCallType(void)
{
    return RC_REMOTECALL_TYPE_RPC;
}

static void
setupConnection(int idev, RCServer_t *sp)
{
    int id = sp->id;
    int cid = sp->cid;
    int portid = DSCUDA_PROG + cid;

    WARN(2, "Requesting socket connection to %s:%d (port 0x%x)...\n", sp->ip, cid, portid);

#if 0
    Clnt[idev][id] = clnt_create(sp->ip, DSCUDA_PROG, DSCUDA_VER, "tcp");

#elif 1 // TCP

    struct sockaddr_in sockaddr;
    struct hostent *hent;
    int sock = RPC_ANYSOCK;

    hent = gethostbyname(sp->ip);
    memset((char *)&sockaddr, 0, sizeof(sockaddr));
    sockaddr.sin_family = AF_INET;
    bcopy(hent->h_addr, (caddr_t)&sockaddr.sin_addr, hent->h_length);
    sockaddr.sin_port = htons((in_port_t)0);

    Clnt[idev][id] = clnttcp_create(&sockaddr,
                                    portid,
                                    DSCUDA_VER,
                                    &sock,
                                    RC_BUFSIZE, RC_BUFSIZE);

#else // UDP

    struct sockaddr_in sockaddr;
    struct hostent *hent;
    int sock = RPC_ANYSOCK;

    hent = gethostbyname(sp->ip);
    memset((char *)&sockaddr, 0, sizeof(sockaddr));
    sockaddr.sin_family = AF_INET;
    bcopy(hent->h_addr, (caddr_t)&sockaddr.sin_addr, hent->h_length);
    sockaddr.sin_port = htons((in_port_t)0);
    struct timeval wait = {
        1.0, // sec
        0.0, // usec
    };

    Clnt[idev][id] = clntudp_create(&sockaddr,
                                    portid,
                                    DSCUDA_VER,
                                    wait,
                                    &sock);

#endif
    if (!Clnt[idev][id]) {
        char buf[256];
        sprintf(buf, "%s:%d (port 0x%x) ", sp->ip, id, portid);
        clnt_pcreateerror(buf);
        if (0 == strcmp(sp->ip, DEFAULT_SVRIP)) {
            WARN(0, "You may need to set an environment variable 'DSCUDA_SERVER'.\n");
        }
        else {
            WARN(0, "DSCUDA server (dscudasrv on %s:%d) may be down.\n", sp->ip, id);
        }
        exit(1);
    }
}

static void
checkResult(void *rp, RCServer_t *sp)
{
    if (rp) return;
    clnt_perror(Clnt[Vdevid][sp->id], sp->ip);
    exit(1);
}

/*
 * test API for internal use only.
 */

void
dscudaWrite(size_t size, char *dst, char *src)
{
    dscudaResult *rp;
    RCbuf srcbuf;

    initClient();
    srcbuf.RCbuf_len = size;
    srcbuf.RCbuf_val = (char *)src;
    WARN(3, "cudaWrite(%d, 0x%08llx, 0x%08llx)...", size, (unsigned long)dst, (unsigned long)src);

    Vdev_t *vdev = Vdev + Vdevid;
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudawriteid_1((RCsize)size, (RCadr)dst, srcbuf, Clnt[Vdevid][sp->id]);
        checkResult(rp, sp);
    }
    WARN(3, "done.\n");
}

void
dscudaRead(size_t size, char *dst, char *src)
{
    dscudaReadResult *rp;

    initClient();
    WARN(3, "cudaRead(%d, 0x%08llx, 0x%08llx)...", size, (unsigned long)dst, (unsigned long)src);
    Vdev_t *vdev = Vdev + Vdevid;
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudareadid_1((RCsize)size, (RCadr)src, Clnt[Vdevid][sp->id]);
        checkResult(rp, sp);
    }
    memcpy(dst, rp->buf.RCbuf_val, rp->buf.RCbuf_len);
    WARN(3, "done.\n");
}

/*
 * Dscuda client-side counterpart for CUDA runtime API:
 */

/*
 * Thread Management
 */

cudaError_t
cudaThreadExit(void)
{
    cudaError_t err = cudaSuccess;
    dscudaResult *rp;

    initClient();
    WARN(3, "cudaThreadExit()...");
    Vdev_t *vdev = Vdev + Vdevid;
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudathreadexitid_1(Clnt[Vdevid][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
    }
    WARN(3, "done.\n");

    return err;
}

cudaError_t
cudaThreadSynchronize(void)
{
    cudaError_t err = cudaSuccess;
    dscudaResult *rp;

    initClient();
    WARN(3, "cudaThreadSynchronize()...");
    Vdev_t *vdev = Vdev + Vdevid;
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudathreadsynchronizeid_1(Clnt[Vdevid][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
    }
    WARN(3, "done.\n");

    return err;
}

cudaError_t
cudaThreadSetLimit(enum cudaLimit limit, size_t value)
{
    cudaError_t err = cudaSuccess;
    dscudaResult *rp;

    initClient();
    WARN(3, "cudaThreadSetLimit(%d, %d)...", limit, value);
    Vdev_t *vdev = Vdev + Vdevid;
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudathreadsetlimitid_1(limit, value, Clnt[Vdevid][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
    }
    WARN(3, "done.\n");

    return err;
}

cudaError_t
cudaThreadGetLimit(size_t *pValue, enum cudaLimit limit)
{
    cudaError_t err = cudaSuccess;
    dscudaThreadGetLimitResult *rp;

    initClient();
    WARN(3, "cudaThreadGetLimit(0x%08llx, %d)...", pValue, limit);
    Vdev_t *vdev = Vdev + Vdevid;
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudathreadgetlimitid_1(limit, Clnt[Vdevid][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
    }
    *pValue = rp->value;
    err = (cudaError_t)rp->err;
    WARN(3, "done.  *pValue: %d\n", *pValue);

    return err;
}

cudaError_t
cudaThreadSetCacheConfig(enum cudaFuncCache cacheConfig)
{
    cudaError_t err = cudaSuccess;
    dscudaResult *rp;

    initClient();
    WARN(3, "cudaThreadSetCacheConfig(%d)...", cacheConfig);
    Vdev_t *vdev = Vdev + Vdevid;
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudathreadsetcacheconfigid_1(cacheConfig, Clnt[Vdevid][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
    }
    WARN(3, "done.\n");

    return err;
}

cudaError_t
cudaThreadGetCacheConfig(enum cudaFuncCache *pCacheConfig)
{
    cudaError_t err = cudaSuccess;
    dscudaThreadGetCacheConfigResult *rp;

    initClient();
    WARN(3, "cudaThreadGetCacheConfig(0x%08llx)...", pCacheConfig);
    Vdev_t *vdev = Vdev + Vdevid;
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudathreadgetcacheconfigid_1(Clnt[Vdevid][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
    }
    *pCacheConfig = (enum cudaFuncCache)rp->cacheConfig;
    err = (cudaError_t)rp->err;
    WARN(3, "done.  *pCacheConfig: %d\n", *pCacheConfig);

    return err;
}


/*
 * Error Handling
 */

cudaError_t
cudaGetLastError(void)
{
    cudaError_t err = cudaSuccess;
    dscudaResult *rp;

    initClient();
    WARN(5, "cudaGetLastError()...");
    Vdev_t *vdev = Vdev + Vdevid;
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudagetlasterrorid_1(Clnt[Vdevid][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
    }
    WARN(5, "done.\n");

    return err;
}

cudaError_t
cudaPeekAtLastError(void)
{
    cudaError_t err = cudaSuccess;
    dscudaResult *rp;

    initClient();
    WARN(5, "cudaPeekAtLastError()...");
    Vdev_t *vdev = Vdev + Vdevid;
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudapeekatlasterrorid_1(Clnt[Vdevid][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
    }
    WARN(5, "done.\n");

    return err;
}

const char *
cudaGetErrorString(cudaError_t error)
{
    dscudaGetErrorStringResult *rp;

    initClient();
    WARN(5, "cudaGetErrorString()...");
    Vdev_t *vdev = Vdev + Vdevid;
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudageterrorstringid_1(error, Clnt[Vdevid][sp->id]);
        checkResult(rp, sp);
    }
    WARN(5, "done.\n");

    return rp->errmsg;
}

/*
 * Device Management
 */

cudaError_t
cudaSetDeviceFlags(unsigned int flags)
{
    cudaError_t err = cudaSuccess;
    dscudaResult *rp;

    initClient();
    WARN(3, "cudaSetDeviceFlags()...");
    Vdev_t *vdev = Vdev + Vdevid;
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudasetdeviceflagsid_1(flags, Clnt[Vdevid][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
    }
    invalidateModuleCache();

    WARN(3, "done.\n");

    return err;
}


cudaError_t
cudaDriverGetVersion (int *driverVersion)
{
    cudaError_t err = cudaSuccess;
    dscudaDriverGetVersionResult *rp;

    initClient();
    WARN(3, "cudaDriverGetVersionCount(0x%08llx)...", (unsigned long)driverVersion);
    Vdev_t *vdev = Vdev + Vdevid;
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudadrivergetversionid_1(Clnt[Vdevid][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
    }

    *driverVersion = rp->ver;
    WARN(3, "done.\n");

    return err;
}

cudaError_t
cudaRuntimeGetVersion(int *runtimeVersion)
{
    cudaError_t err = cudaSuccess;
    dscudaRuntimeGetVersionResult *rp;

    initClient();
    WARN(3, "cudaRuntimeGetVersion(0x%08llx)...", (unsigned long)runtimeVersion);
    Vdev_t *vdev = Vdev + Vdevid;
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudaruntimegetversionid_1(Clnt[Vdevid][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
    }

    *runtimeVersion = rp->ver;
    WARN(3, "done.\n");

    return err;
}

cudaError_t
cudaDeviceSynchronize(void)
{
    cudaError_t err = cudaSuccess;
    dscudaResult *rp;

    initClient();
    WARN(3, "cudaDeviceSynchronize()...");
    Vdev_t *vdev = Vdev + Vdevid;
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudadevicesynchronize_1(Clnt[Vdevid][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
    }
    WARN(3, "done.\n");

    return err;
}

cudaError_t
cudaDeviceReset(void)
{
    cudaError_t err = cudaSuccess;
    dscudaResult *rp;

    initClient();
    WARN(3, "cudaDeviceReset()...");
    Vdev_t *vdev = Vdev + Vdevid;
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudadevicereset_1(Clnt[Vdevid][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
    }
    WARN(3, "done.\n");

    return err;
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

cudaError_t
dscudaFuncGetAttributesWrapper(int *moduleid, struct cudaFuncAttributes *attr, const char *func)
{
    cudaError_t err = cudaSuccess;
    dscudaFuncGetAttributesResult *rp;

    initClient();
    WARN(3, "dscudaFuncGetAttributesWrapper(%d, 0x%08llx, %s)...",
         moduleid, (unsigned long)attr, func);
    Vdev_t *vdev = Vdev + Vdevid;
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudafuncgetattributesid_1(moduleid[i], (char*)func, Clnt[Vdevid][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
    }

    attr->binaryVersion      = rp->attr.binaryVersion;
    attr->constSizeBytes     = rp->attr.constSizeBytes;
    attr->localSizeBytes     = rp->attr.localSizeBytes;
    attr->maxThreadsPerBlock = rp->attr.maxThreadsPerBlock;
    attr->numRegs            = rp->attr.numRegs;
    attr->ptxVersion         = rp->attr.ptxVersion;
    attr->sharedSizeBytes    = rp->attr.sharedSizeBytes;
    WARN(3, "done.\n");
    WARN(3, "  attr->binaryVersion: %d\n", attr->binaryVersion);
    WARN(3, "  attr->constSizeBytes: %d\n", attr->constSizeBytes);
    WARN(3, "  attr->localSizeBytes: %d\n", attr->localSizeBytes);
    WARN(3, "  attr->maxThreadsPerBlock: %d\n", attr->maxThreadsPerBlock);
    WARN(3, "  attr->numRegs: %d\n", attr->numRegs);
    WARN(3, "  attr->ptxVersion: %d\n", attr->ptxVersion);
    WARN(3, "  attr->sharedSizeBytes: %d\n", attr->sharedSizeBytes);

    return err;
}

/*
 * Memory Management
 */

cudaError_t
cudaMalloc(void **devAdrPtr, size_t size)
{
    cudaError_t err = cudaSuccess;
    dscudaMallocResult *rp;

    initClient();
    WARN(3, "cudaMalloc(0x%08llx, %d)...", (unsigned long)devAdrPtr, size);
    Vdev_t *vdev = Vdev + Vdevid;
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudamallocid_1(size, Clnt[Vdevid][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
    }

    *devAdrPtr = (void*)rp->devAdr;
    WARN(3, "done. *devAdrPtr:0x%08llx\n", *devAdrPtr);

    return err;
}

cudaError_t
cudaFree(void *mem)
{
    cudaError_t err = cudaSuccess;
    dscudaResult *rp;

    initClient();
    WARN(3, "cudaFree(0x%08llx)...", (unsigned long)mem);
    Vdev_t *vdev = Vdev + Vdevid;
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudafreeid_1((RCadr)mem, Clnt[Vdevid][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
    }

    WARN(3, "done.\n");
    return err;
}

cudaError_t
cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind)
{
    cudaError_t err = cudaSuccess;
    dscudaMemcpyD2HResult *d2hrp;
    dscudaResult *h2drp, *d2drp;
    RCbuf srcbuf;
    Vdev_t *vdev;
    RCServer_t *sp;

    initClient();

    WARN(3, "cudaMemcpy(0x%08lx, 0x%08lx, %d, %s)...",
            (unsigned long)dst, (unsigned long)src, count, dscudaMemcpyKindName(kind));

    switch (kind) {
      case cudaMemcpyDeviceToHost:
        vdev = Vdev + Vdevid;
        sp = vdev->server;
        for (int i = 0; i < vdev->nredundancy; i++, sp++) {
            d2hrp = dscudamemcpyd2hid_1((RCadr)src, count, Clnt[Vdevid][sp->id]);
            checkResult(d2hrp, sp);
            err = (cudaError_t)d2hrp->err;
            if (d2hrp->err != cudaSuccess) {
                err = (cudaError_t)d2hrp->err;
            }
            if (i == 0) {
                memcpy(dst, d2hrp->buf.RCbuf_val, d2hrp->buf.RCbuf_len);
            }
            else if (bcmp(dst, d2hrp->buf.RCbuf_val, d2hrp->buf.RCbuf_len) != 0) {
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
        srcbuf.RCbuf_len = count;
        srcbuf.RCbuf_val = (char *)src;
        vdev = Vdev + Vdevid;
        sp = vdev->server;
        for (int i = 0; i < vdev->nredundancy; i++, sp++) {
            h2drp = dscudamemcpyh2did_1((RCadr)dst, srcbuf, count, Clnt[Vdevid][sp->id]);
            checkResult(h2drp, sp);
            if (h2drp->err != cudaSuccess) {
                err = (cudaError_t)h2drp->err;
            }
        }
        break;
      case cudaMemcpyDeviceToDevice:
        vdev = Vdev + Vdevid;
        sp = vdev->server;
        for (int i = 0; i < vdev->nredundancy; i++, sp++) {
            d2drp = dscudamemcpyd2did_1((RCadr)dst, (RCadr)src, count, Clnt[Vdevid][sp->id]);
            checkResult(d2drp, sp);
            if (d2drp->err != cudaSuccess) {
                err = (cudaError_t)d2drp->err;
            }
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
    dscudaGetDevicePropertiesResult *rp;

    initClient();
    WARN(3, "cudaGetDeviceProperties(0x%08lx, %d)...", (unsigned long)prop, device);
    Vdev_t *vdev = Vdev + device;
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudagetdevicepropertiesid_1(device, Clnt[device][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
    }
    memcpy(prop, rp->prop.RCbuf_val, rp->prop.RCbuf_len);
    WARN(3, "done.\n");

    return err;
}

/*
 * launch a kernel function of id 'kid', defined in a module of id 'moduleid'.
 * 'kid' must be unique inside a single module.
 */
void
dscudaLaunchKernelWrapper(int *moduleid, int kid, char *kname,
                         RCdim3 gdim, RCdim3 bdim, RCsize smemsize, RCstream stream,
                         RCargs args)
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
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        void *rp = dscudalaunchkernelid_1(moduleid[i], kid, kname,
                                         gdim, bdim, smemsize, (RCstream)st->s[i],
                                         args, Clnt[Vdevid][sp->id]);
        checkResult(rp, sp);
    }

    mem = RCmappedMemListTop;
    while (mem) {
        cudaMemcpy(mem->pHost, mem->pDevice, mem->size, cudaMemcpyDeviceToHost);
        mem = mem->next;
    }
}

void
ibvDscudaLaunchKernelWrapper(int *moduleid, int kid, char *kname,
                            int *gdim, int *bdim, RCsize smemsize, RCstream stream,
                            int narg, IbvArg *arg)
{
    // a dummy func.
}

cudaError_t
cudaMallocArray(struct cudaArray **array, const struct cudaChannelFormatDesc *desc,
                size_t width, size_t height, unsigned int flags)
{
    cudaError_t err = cudaSuccess;
    dscudaMallocArrayResult *rp;
    RCchanneldesc descbuf;
    cudaArray *ca[RC_NREDUNDANCYMAX];

    initClient();
    WARN(3, "cudaMallocArray(0x%08llx, 0x%08llx, %d, %d, 0x%08x)...",
         (unsigned long)array, desc, width, height, flags);


    descbuf.x = desc->x;
    descbuf.y = desc->y;
    descbuf.z = desc->z;
    descbuf.w = desc->w;
    descbuf.f = desc->f;

    Vdev_t *vdev = Vdev + Vdevid;
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudamallocarrayid_1(descbuf, width, height, flags, Clnt[Vdevid][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
        ca[i] = (cudaArray *)rp->array;
    }

    *array = ca[0];
    RCcuarrayArrayRegister(ca);
    WARN(3, "done. *array:0x%08llx\n", *array);

    return err;
}

cudaError_t
cudaFreeArray(struct cudaArray *array)
{
    cudaError_t err = cudaSuccess;
    dscudaResult *rp;
    RCcuarrayArray *ca;

    initClient();
    WARN(3, "cudaFreeArray(0x%08llx)...", (unsigned long)array);
    ca = RCcuarrayArrayQuery(array);
    if (!ca) {
        WARN(0, "invalid cudaArray : 0x%08llx\n", array);
        exit(1);
    }
    Vdev_t *vdev = Vdev + Vdevid;
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudafreearrayid_1((RCadr)ca->ap[i], Clnt[Vdevid][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
    }
    RCcuarrayArrayUnregister(ca->ap[0]);
    WARN(3, "done.\n");
    return err;
}

cudaError_t
cudaMemcpyToArray(struct cudaArray *dst, size_t wOffset, size_t hOffset, const void *src,
                  size_t count, enum cudaMemcpyKind kind)
{
    cudaError_t err = cudaSuccess;
    dscudaResult *h2drp, *d2drp;
    RCbuf srcbuf;
    RCcuarrayArray *ca;
    Vdev_t *vdev;
    RCServer_t *sp;

    initClient();

    WARN(3, "cudaMemcpyToArray(0x%08llx, %d, %d, 0x%08llx, %d, %s)...",
         (unsigned long)dst, wOffset, hOffset, (unsigned long)src, count, dscudaMemcpyKindName(kind));
    ca = RCcuarrayArrayQuery(dst);
    if (!ca) {
        WARN(0, "invalid cudaArray : 0x%08llx\n", dst);
        exit(1);
    }
    switch (kind) {
      case cudaMemcpyHostToDevice:
        srcbuf.RCbuf_len = count;
        srcbuf.RCbuf_val = (char *)src;

        vdev = Vdev + Vdevid;
        sp = vdev->server;
        for (int i = 0; i < vdev->nredundancy; i++, sp++) {
            h2drp = dscudamemcpytoarrayh2did_1((RCadr)ca->ap[i], wOffset, hOffset, srcbuf, count, Clnt[Vdevid][sp->id]);
            checkResult(h2drp, sp);
            if (h2drp->err != cudaSuccess) {
                err = (cudaError_t)h2drp->err;
            }
        }
        break;
      case cudaMemcpyDeviceToDevice:
        vdev = Vdev + Vdevid;
        sp = vdev->server;
        for (int i = 0; i < vdev->nredundancy; i++, sp++) {
            d2drp = dscudamemcpytoarrayd2did_1((RCadr)ca->ap[i], wOffset, hOffset, (RCadr)src, count, Clnt[Vdevid][sp->id]);
            checkResult(d2drp, sp);
            if (d2drp->err != cudaSuccess) {
                err = (cudaError_t)d2drp->err;
            }
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
dscudaMemcpyToSymbolWrapper(int *moduleid, const char *symbol, const void *src,
                           size_t count, size_t offset, enum cudaMemcpyKind kind)
{
    cudaError_t err = cudaSuccess;
    dscudaResult *rp;
    RCbuf srcbuf;
    RCServer_t *sp;
    Vdev_t *vdev;

    initClient();

    WARN(3, "dscudaMemcpyToSymbolWrapper(%d, 0x%08llx, 0x%08llx, %d, %d, %s)"
         "symbol:%s  ...",
         moduleid, (unsigned long)symbol, (unsigned long)src,
         count, offset, dscudaMemcpyKindName(kind), symbol);

    switch (kind) {
      case cudaMemcpyHostToDevice:
        srcbuf.RCbuf_len = count;
        srcbuf.RCbuf_val = (char *)src;
        vdev = Vdev + Vdevid;
        sp = vdev->server;
        for (int i = 0; i < vdev->nredundancy; i++, sp++) {
            rp = dscudamemcpytosymbolh2did_1(moduleid[i], (char *)symbol, srcbuf, count, offset, Clnt[Vdevid][sp->id]);
            checkResult(rp, sp);
            if (rp->err != cudaSuccess) {
                err = (cudaError_t)rp->err;
            }
        }
        break;
      case cudaMemcpyDeviceToDevice:
        vdev = Vdev + Vdevid;
        sp = vdev->server;
        for (int i = 0; i < vdev->nredundancy; i++, sp++) {
            rp = dscudamemcpytosymbold2did_1(moduleid[i], (char *)symbol, (RCadr)src, count, offset, Clnt[Vdevid][sp->id]);
            checkResult(rp, sp);
            if (rp->err != cudaSuccess) {
                err = (cudaError_t)rp->err;
            }
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
dscudaMemcpyFromSymbolWrapper(int *moduleid, void *dst, const char *symbol,
                             size_t count, size_t offset,
                             enum cudaMemcpyKind kind)
{
    cudaError_t err = cudaSuccess;
    dscudaMemcpyFromSymbolD2HResult *d2hrp;
    dscudaResult *d2drp;
    Vdev_t *vdev;
    RCServer_t *sp;

    initClient();

    WARN(3, "dscudaMemcpyFromSymbolWrapper(0x%08llx, 0x%08llx, 0x%08llx, %d, %d, %s)"
         "symbol:%s  ...",
         moduleid, (unsigned long)dst, (unsigned long)symbol,
         count, offset, dscudaMemcpyKindName(kind), symbol);

    switch (kind) {
      case cudaMemcpyDeviceToDevice:
        vdev = Vdev + Vdevid;
        sp = vdev->server;
        for (int i = 0; i < vdev->nredundancy; i++, sp++) {
            d2drp = dscudamemcpyfromsymbold2did_1(moduleid[i], (RCadr)dst, (char *)symbol, count, offset, Clnt[Vdevid][sp->id]);
            checkResult(d2drp, sp);
            if (d2drp->err != cudaSuccess) {
                err = (cudaError_t)d2drp->err;
            }
        }
        break;
      case cudaMemcpyDeviceToHost:
        vdev = Vdev + Vdevid;
        sp = vdev->server;
        for (int i = 0; i < vdev->nredundancy; i++, sp++) {
            d2hrp = dscudamemcpyfromsymbold2hid_1(moduleid[i], (char *)symbol, count, offset, Clnt[Vdevid][sp->id]);
            checkResult(d2hrp, sp);
            if (d2hrp->err != cudaSuccess) {
                err = (cudaError_t)d2hrp->err;
            }
            if (i == 0) {
                memcpy(dst, d2hrp->buf.RCbuf_val, d2hrp->buf.RCbuf_len);
            }
            else if (bcmp(dst, d2hrp->buf.RCbuf_val, d2hrp->buf.RCbuf_len) != 0) {
                if (errorHandler) {
                    errorHandler(errorHandlerArg);
                }
            }
            else {
                WARN(3, "cudaMemcpyFromSymbol() data copied from device%d matched with that from device0.\n", i);
            }
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
cudaMemset(void *devPtr, int value, size_t count)
{
    cudaError_t err = cudaSuccess;
    dscudaResult *rp;

    initClient();
    WARN(3, "cudaMemset()...");
    Vdev_t *vdev = Vdev + Vdevid;
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudamemsetid_1((RCadr)devPtr, value, count, Clnt[Vdevid][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
    }
    WARN(3, "done.\n");

    return err;
}

cudaError_t
cudaMallocPitch(void **devPtr, size_t *pitch, size_t width, size_t height)
{
    cudaError_t err = cudaSuccess;
    dscudaMallocPitchResult *rp;

    initClient();
    WARN(3, "cudaMallocPitch(0x%08llx, 0x%08llx, %d, %d)...",
         (unsigned long)devPtr, pitch, width, height);
    Vdev_t *vdev = Vdev + Vdevid;
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudamallocpitchid_1(width, height, Clnt[Vdevid][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
    }

    *devPtr = (void*)rp->devPtr;
    *pitch = rp->pitch;
    WARN(3, "done. *devPtr:0x%08llx  *pitch:%d\n", *devPtr, *pitch);

    return err;
}

cudaError_t
cudaMemcpy2DToArray(struct cudaArray *dst, size_t wOffset, size_t hOffset,
                    const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind)
{
    cudaError_t err = cudaSuccess;
    dscudaMemcpy2DToArrayD2HResult *d2hrp;
    dscudaResult *h2drp, *d2drp;
    RCbuf srcbuf;
    RCcuarrayArray *ca;
    Vdev_t *vdev;
    RCServer_t *sp;

    initClient();

    WARN(3, "cudaMemcpy2DToArray(0x%08llx, %d, %d, 0x%08llx, %d, %d, %d, %s)...",
         (unsigned long)dst, wOffset, hOffset,
         (unsigned long)src, spitch, width, height, dscudaMemcpyKindName(kind));
    ca = RCcuarrayArrayQuery(dst);
    if (!ca) {
        WARN(0, "invalid cudaArray : 0x%08llx\n", dst);
        exit(1);
    }
    switch (kind) {
      case cudaMemcpyDeviceToHost:
        vdev = Vdev + Vdevid;
        sp = vdev->server;
        for (int i = 0; i < vdev->nredundancy; i++, sp++) {
            d2hrp = dscudamemcpy2dtoarrayd2hid_1(wOffset, hOffset,
                                                (RCadr)src, spitch, width, height, Clnt[Vdevid][sp->id]);
            checkResult(d2hrp, sp);
            if (d2hrp->err != cudaSuccess) {
                err = (cudaError_t)d2hrp->err;
            }
            if (i == 0) {
                memcpy(dst, d2hrp->buf.RCbuf_val, d2hrp->buf.RCbuf_len);
            }
            else if (bcmp(dst, d2hrp->buf.RCbuf_val, d2hrp->buf.RCbuf_len) != 0) {
                if (errorHandler) {
                    errorHandler(errorHandlerArg);
                }
            }
            else {
                WARN(3, "cudaMemcpy2DToArray() data copied from device%d matched with that from device0.\n", i);
            }
        }
        break;
      case cudaMemcpyHostToDevice:
        srcbuf.RCbuf_len = spitch * height;
        srcbuf.RCbuf_val = (char *)src;
        vdev = Vdev + Vdevid;
        sp = vdev->server;
        for (int i = 0; i < vdev->nredundancy; i++, sp++) {
            h2drp = dscudamemcpy2dtoarrayh2did_1((RCadr)ca->ap[i], wOffset, hOffset,
                                                srcbuf, spitch, width, height, Clnt[Vdevid][sp->id]);
            checkResult(h2drp, sp);
            if (h2drp->err != cudaSuccess) {
                err = (cudaError_t)h2drp->err;
            }
        }
        break;
      case cudaMemcpyDeviceToDevice:
        vdev = Vdev + Vdevid;
        sp = vdev->server;
        for (int i = 0; i < vdev->nredundancy; i++, sp++) {
            d2drp = dscudamemcpy2dtoarrayd2did_1((RCadr)ca->ap[i], wOffset, hOffset,
                                                (RCadr)src, spitch, width, height, Clnt[Vdevid][sp->id]);
            checkResult(d2drp, sp);
            if (d2drp->err != cudaSuccess) {
                err = (cudaError_t)d2drp->err;
            }
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
cudaMemcpy2D(void *dst, size_t dpitch,
             const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind)
{
    cudaError_t err = cudaSuccess;
    dscudaMemcpy2DD2HResult *d2hrp;
    dscudaResult *h2drp, *d2drp;
    RCbuf srcbuf;
    Vdev_t *vdev;
    RCServer_t *sp;

    initClient();

    WARN(3, "cudaMemcpy2D(0x%08llx, %d, 0x%08llx, %d, %d, %d, %s)...",
         (unsigned long)dst, dpitch,
         (unsigned long)src, spitch, width, height, dscudaMemcpyKindName(kind));

    switch (kind) {
      case cudaMemcpyDeviceToHost:
        vdev = Vdev + Vdevid;
        sp = vdev->server;
        for (int i = 0; i < vdev->nredundancy; i++, sp++) {
            d2hrp = dscudamemcpy2dd2hid_1(dpitch,
                                         (RCadr)src, spitch, width, height, Clnt[Vdevid][sp->id]);
            checkResult(d2hrp, sp);
            if (d2hrp->err != cudaSuccess) {
                err = (cudaError_t)d2hrp->err;
            }
            if (i == 0) {
                memcpy(dst, d2hrp->buf.RCbuf_val, d2hrp->buf.RCbuf_len);
            }
            else if (bcmp(dst, d2hrp->buf.RCbuf_val, d2hrp->buf.RCbuf_len) != 0) {
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
        srcbuf.RCbuf_len = spitch * height;
        srcbuf.RCbuf_val = (char *)src;
        vdev = Vdev + Vdevid;
        sp = vdev->server;
        for (int i = 0; i < vdev->nredundancy; i++, sp++) {
            h2drp = dscudamemcpy2dh2did_1((RCadr)dst, dpitch,
                                         srcbuf, spitch, width, height, Clnt[Vdevid][sp->id]);
            checkResult(h2drp, sp);
            if (h2drp->err != cudaSuccess) {
                err = (cudaError_t)h2drp->err;
            }
        }
        break;
      case cudaMemcpyDeviceToDevice:
        vdev = Vdev + Vdevid;
        sp = vdev->server;
        for (int i = 0; i < vdev->nredundancy; i++, sp++) {
            d2drp = dscudamemcpy2dd2did_1((RCadr)dst, dpitch,
                                         (RCadr)src, spitch, width, height, Clnt[Vdevid][sp->id]);
            checkResult(d2drp, sp);
            if (d2drp->err != cudaSuccess) {
                err = (cudaError_t)d2drp->err;
            }
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
cudaMemset2D(void *devPtr, size_t pitch, int value, size_t width, size_t height)
{
    cudaError_t err = cudaSuccess;
    dscudaResult *rp;

    initClient();
    WARN(3, "cudaMemset2D(0x%08llx, %d, %d, %d, %d)...",
         (unsigned long)devPtr, pitch, value, width, height);
    Vdev_t *vdev = Vdev + Vdevid;
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudamemset2did_1((RCadr)devPtr, pitch, value, width, height, Clnt[Vdevid][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
    }
    WARN(3, "done.\n");

    return err;
}

cudaError_t
cudaMallocHost(void **ptr, size_t size)
{
#if RC_SUPPORT_PAGELOCK
    cudaError_t err = cudaSuccess;
    dscudaMallocHostResult *rp;

    initClient();
    WARN(3, "cudaMallocHost(0x%08llx, %d)...", (unsigned long)ptr, size);
    Vdev_t *vdev = Vdev + Vdevid;
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudamallochostid_1(size, Clnt[Vdevid][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
    }

    *ptr = (void*)rp->ptr;

    WARN(3, "done. *ptr:0x%08llx\n", *ptr);
    return err;
#else
    // returned memory is not page locked.
    // it cannot be passed to cudaMemcpyAsync().
    *ptr = malloc(size);
    if (*ptr) {
        return cudaSuccess;
    }
    else {
        return cudaErrorMemoryAllocation;
    }
#endif
}

cudaError_t
cudaHostAlloc(void **pHost, size_t size, unsigned int flags)
{
#if RC_SUPPORT_PAGELOCK
    cudaError_t err = cudaSuccess;
    dscudaHostAllocResult *rp;

    initClient();
    WARN(3, "cudaHostAlloc(0x%08llx, %d, 0x%08x)...", (unsigned long)pHost, size, flags);
    Vdev_t *vdev = Vdev + Vdevid;
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudahostallocid_1(size, flags, Clnt[Vdevid][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
    }

    *pHost = (void*)rp->pHost;
    WARN(3, "done. *pHost:0x%08llx\n", *pHost);

    return err;
#else
    // returned memory is not page locked.
    // it cannot be passed to cudaMemcpyAsync().

    cudaError_t err = cudaSuccess;
    void *devmem;

    initClient();
    WARN(3, "cudaHostAlloc(0x%08llx, %d, 0x%08x)...", (unsigned long)pHost, size, flags);

    *pHost = malloc(size);
    if (!*pHost) return cudaErrorMemoryAllocation;
    if (!(flags & cudaHostAllocMapped)) {
        WARN(3, "done. *pHost:0x%08llx\n", *pHost);
        return cudaSuccess;
    }

    // flags says the host memory must be mapped on to the device memory.
    err = cudaMalloc(&devmem, size);
    if (err == cudaSuccess) {
        RCmappedMemRegister(*pHost, devmem, size);
    }
    WARN(3, "done. host mem:0x%08llx  device mem:0x%08llx\n", *pHost, devmem);

    return err;
#endif
}

cudaError_t
cudaFreeHost(void *ptr)
{
#if RC_SUPPORT_PAGELOCK
    cudaError_t err = cudaSuccess;
    dscudaResult *rp;

    initClient();
    WARN(3, "cudaFreeHost(0x%08llx)...", (unsigned long)ptr);
    Vdev_t *vdev = Vdev + Vdevid;
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudafreehostid_1((RCadr)ptr, Clnt[Vdevid][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
    }
    WARN(3, "done.\n");
    return err;
#else
    cudaError_t err = cudaSuccess;
    RCmappedMem *mem = RCmappedMemQuery(ptr);
    free(ptr);
    if (mem) { // ptr mapped on to a device memory.
        err = cudaFree(mem->pDevice);
        RCmappedMemUnregister(ptr);
        return err;
    }
    else {
        return cudaSuccess;
    }
#endif
}

// flags is not used for now in CUDA3.2. It should always be zero.
cudaError_t
cudaHostGetDevicePointer(void **pDevice, void*pHost, unsigned int flags)
{
#if RC_SUPPORT_PAGELOCK
    cudaError_t err = cudaSuccess;
    dscudaHostGetDevicePointerResult *rp;

    initClient();
    WARN(3, "cudaHostGetDevicePointer(0x%08llx, 0x%08llx, 0x%08x)...",
         (unsigned long)pDevice, (unsigned long)pHost, flags);
    Vdev_t *vdev = Vdev + Vdevid;
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudahostgetdevicepointerid_1((RCadr)pHost, flags, Clnt[Vdevid][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
    }

    *pDevice = (void *)rp->pDevice;
    WARN(3, "done. *pDevice:0x%08llx\n", *pDevice);
    return err;
#else
    RCmappedMem *mem = RCmappedMemQuery(pHost);
    if (!mem) return cudaErrorInvalidValue; // pHost is not registered as RCmappedMem.
    *pDevice = mem->pDevice;
    return cudaSuccess;
#endif
}

cudaError_t
cudaHostGetFlags(unsigned int *pFlags, void *pHost)
{
    cudaError_t err = cudaSuccess;
    dscudaHostGetFlagsResult *rp;

    initClient();
    WARN(3, "cudaHostGetFlags(0x%08x 0x%08llx)...",
         (unsigned long)pFlags, (unsigned long)pHost);
    Vdev_t *vdev = Vdev + Vdevid;
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudahostgetflagsid_1((RCadr)pHost, Clnt[Vdevid][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
    }
    *pFlags = rp->flags;
    WARN(3, "done. flags:0x%08x\n", *pFlags);
    return err;
    
}

cudaError_t
cudaMemcpyAsync(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream)
{
#if RC_SUPPORT_PAGELOCK
    cudaError_t err = cudaSuccess;
    dscudaMemcpyAsyncD2HResult *d2hrp;
    dscudaResult *h2drp, *d2drp;
    RCbuf srcbuf;
    RCstreamArray *st;
    Vdev_t *vdev;
    RCServer_t *sp;

    initClient();

    WARN(3, "cudaMemcpyAsync(0x%08llx, 0x%08llx, %d, %s, 0x%08llx)...",
         (unsigned long)dst, (unsigned long)src, count, dscudaMemcpyKindName(kind), st->s[0]);
    st = RCstreamArrayQuery(stream);
    if (!st) {
        WARN(0, "invalid stream : 0x%08llx\n", stream);
        exit(1);
    }
    switch (kind) {
      case cudaMemcpyDeviceToHost:
        Vdev_t *vdev = Vdev + Vdevid;
        RCServer_t *sp = vdev->server;
        for (int i = 0; i < vdev->nredundancy; i++, sp++) {
            d2hrp = dscudamemcpyasyncd2hid_1((RCadr)src, count, (RCstream)st->s[i], Clnt[Vdevid][sp->id]);
            checkResult(d2hrp, sp);
            if (d2hrp->err != cudaSuccess) {
                err = (cudaError_t)d2hrp->err;
            }
        }
        memcpy(dst, d2hrp->buf.RCbuf_val, d2hrp->buf.RCbuf_len);
        break;
      case cudaMemcpyHostToDevice:
        srcbuf.RCbuf_len = count;
        srcbuf.RCbuf_val = (char *)src;
        Vdev_t *vdev = Vdev + Vdevid;
        RCServer_t *sp = vdev->server;
        for (int i = 0; i < vdev->nredundancy; i++, sp++) {
            h2drp = dscudamemcpyasynch2did_1((RCadr)dst, srcbuf, count, (RCstream)st->s[i], Clnt[Vdevid][sp->id]);
            checkResult(h2drp, sp);
            if (h2drp->err != cudaSuccess) {
                err = (cudaError_t)h2drp->err;
            }
        }
        break;
      case cudaMemcpyDeviceToDevice:
        Vdev_t *vdev = Vdev + Vdevid;
        RCServer_t *sp = vdev->server;
        for (int i = 0; i < vdev->nredundancy; i++, sp++) {
            d2drp = dscudamemcpyasyncd2did_1((RCadr)dst, (RCadr)src, count, (RCstream)st->s[i], Clnt[Vdevid][sp->id]);
            checkResult(d2drp, sp);
            if (d2drp->err != cudaSuccess) {
                err = (cudaError_t)d2drp->err;
            }
        }
        break;
      default:
        WARN(0, "Unsupported value for cudaMemcpyKind : %s\n", dscudaMemcpyKindName(kind));
        exit(1);
    }
    WARN(3, "done.\n");

    return err;

#else
    // this DOES block.
    // this is only for use with a poor implementation of dscudaMallocHost().
    return cudaMemcpy(dst, src, count, kind);
#endif
}

/*
 * Stream Management
 */

cudaError_t
cudaStreamCreate(cudaStream_t *pStream)
{
#if RC_SUPPORT_STREAM
    cudaError_t err = cudaSuccess;
    dscudaStreamCreateResult *rp;
    cudaStream_t st[RC_NREDUNDANCYMAX];

    initClient();
    WARN(3, "cudaStreamCreate(0x%08llx)...", (unsigned long)pStream);
    Vdev_t *vdev = Vdev + Vdevid;
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudastreamcreateid_1(Clnt[Vdevid][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
        st[i] = (cudaStream_t)rp->stream;
    }

    *pStream = st[0];
    RCstreamArrayRegister(st);
    WARN(3, "done. *pStream:0x%08llx\n", *pStream);

    return err;
#else
    *pStream = 0;
    return cudaSuccess;
#endif
}

cudaError_t
cudaStreamDestroy(cudaStream_t stream)
{
#if RC_SUPPORT_STREAM
    cudaError_t err = cudaSuccess;
    dscudaResult *rp;
    RCstreamArray *st;

    initClient();
    WARN(3, "cudaStreamDestroy(0x%08llx)...", (unsigned long)stream);
    st = RCstreamArrayQuery(stream);
    if (!st) {
        WARN(0, "invalid stream : 0x%08llx\n", stream);
        exit(1);
    }
    Vdev_t *vdev = Vdev + Vdevid;
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudastreamdestroyid_1((RCadr)st->s[i], Clnt[Vdevid][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
    }
    RCstreamArrayUnregister(st->s[0]);
    WARN(3, "done.\n");
    return err;
#else
    return cudaSuccess;
#endif
}

cudaError_t
cudaStreamSynchronize(cudaStream_t stream)
{
#if RC_SUPPORT_STREAM
    cudaError_t err = cudaSuccess;
    dscudaResult *rp;
    RCstreamArray *st;

    initClient();
    WARN(3, "cudaStreamSynchronize(0x%08llx)...", (unsigned long)stream);
    st = RCstreamArrayQuery(stream);
    if (!st) {
        WARN(0, "invalid stream : 0x%08llx\n", stream);
        exit(1);
    }
    Vdev_t *vdev = Vdev + Vdevid;
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudastreamsynchronizeid_1((RCadr)st->s[i], Clnt[Vdevid][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
    }
    WARN(3, "done.\n");
    return err;
#else
    return cudaSuccess;
#endif
}

cudaError_t
cudaStreamQuery(cudaStream_t stream)
{
#if RC_SUPPORT_STREAM
    cudaError_t err = cudaSuccess;
    dscudaResult *rp;
    RCstreamArray *st;

    initClient();
    WARN(3, "cudaStreamQuery(0x%08llx)...", (unsigned long)stream);
    st = RCstreamArrayQuery(stream);
    if (!st) {
        WARN(0, "invalid stream : 0x%08llx\n", stream);
        exit(1);
    }
    Vdev_t *vdev = Vdev + Vdevid;
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudastreamqueryid_1((RCadr)st->s[i], Clnt[Vdevid][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
    }
    WARN(3, "done.\n");
    return err;
#else
    return cudaSuccess;
#endif
}

/*
 * Event Management
 */

cudaError_t
cudaEventCreate(cudaEvent_t *event)
{
    cudaError_t err = cudaSuccess;
    dscudaEventCreateResult *rp;
    cudaEvent_t ev[RC_NREDUNDANCYMAX];

    initClient();
    WARN(3, "cudaEventCreate(0x%08llx)...", (unsigned long)event);
    Vdev_t *vdev = Vdev + Vdevid;
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudaeventcreateid_1(Clnt[Vdevid][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
        ev[i] = (cudaEvent_t)rp->event;
    }
    *event = ev[0];
    RCeventArrayRegister(ev);
    WARN(3, "done. *event:0x%08llx\n", *event);

    return err;
}

cudaError_t
cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags)
{
    cudaError_t err = cudaSuccess;
    dscudaEventCreateResult *rp;
    cudaEvent_t ev[RC_NREDUNDANCYMAX];

    initClient();
    WARN(3, "cudaEventCreateWithFlags(0x%08llx, 0x%08x)...", (unsigned long)event, flags);
    Vdev_t *vdev = Vdev + Vdevid;
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudaeventcreatewithflagsid_1(flags, Clnt[Vdevid][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
        ev[i] = (cudaEvent_t)rp->event;
    }
    *event = ev[0];
    RCeventArrayRegister(ev);
    WARN(3, "done. *event:0x%08llx\n", *event);

    return err;
}

cudaError_t
cudaEventDestroy(cudaEvent_t event)
{
    cudaError_t err = cudaSuccess;
    dscudaResult *rp;
    RCeventArray *ev;

    initClient();
    WARN(3, "cudaEventDestroy(0x%08llx)...", (unsigned long)event);
    ev = RCeventArrayQuery(event);
    if (!ev) {
        WARN(0, "invalid event : 0x%08llx\n", event);
        exit(1);
    }
    Vdev_t *vdev = Vdev + Vdevid;
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudaeventdestroyid_1((RCadr)ev->e[i], Clnt[Vdevid][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
    }
    RCeventArrayUnregister(ev->e[0]);
    WARN(3, "done.\n");
    return err;
}

cudaError_t
cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end)
{
    cudaError_t err = cudaSuccess;
    dscudaEventElapsedTimeResult *rp;
    RCeventArray *es, *ee;

    initClient();
    WARN(3, "cudaEventElapsedTime(0x%08llx, 0x%08llx, 0x%08llx)...",
         (unsigned long)ms, (unsigned long)start, (unsigned long)end);
    es = RCeventArrayQuery(start);
    if (!es) {
        WARN(0, "invalid start event : 0x%08llx\n", start);
        exit(1);
    }
    ee = RCeventArrayQuery(end);
    if (!ee) {
        WARN(0, "invalid end event : 0x%08llx\n", end);
        exit(1);
    }
    Vdev_t *vdev = Vdev + Vdevid;
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudaeventelapsedtimeid_1((RCadr)es->e[i], (RCadr)ee->e[i], Clnt[Vdevid][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
    }

    *ms = rp->ms;
    WARN(3, "done.\n");
    return err;
}

cudaError_t
cudaEventRecord(cudaEvent_t event, cudaStream_t stream)
{
    cudaError_t err = cudaSuccess;
    dscudaResult *rp;
    RCstreamArray *st;
    RCeventArray *ev;

    initClient();
    WARN(3, "cudaEventRecord(0x%08llx, 0x%08llx)...", (unsigned long)event, (unsigned long)stream);
    st = RCstreamArrayQuery(stream);
    if (!st) {
        WARN(0, "invalid stream : 0x%08llx\n", stream);
        exit(1);
    }
    ev = RCeventArrayQuery(event);
    if (!ev) {
        WARN(0, "invalid event : 0x%08llx\n", event);
        exit(1);
    }
    Vdev_t *vdev = Vdev + Vdevid;
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudaeventrecordid_1((RCadr)ev->e[i], (RCadr)st->s[i], Clnt[Vdevid][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
    }

    WARN(3, "done.\n");
    return err;
}

cudaError_t
cudaEventSynchronize(cudaEvent_t event)
{
    cudaError_t err = cudaSuccess;
    dscudaResult *rp;
    RCeventArray *ev;

    initClient();
    WARN(3, "cudaEventSynchronize(0x%08llx)...", (unsigned long)event);
    ev = RCeventArrayQuery(event);
    if (!ev) {
        WARN(0, "invalid event : 0x%08llx\n", event);
        exit(1);
    }
    Vdev_t *vdev = Vdev + Vdevid;
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudaeventsynchronizeid_1((RCadr)ev->e[i], Clnt[Vdevid][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
    }
    WARN(3, "done.\n");
    return err;
}

cudaError_t
cudaEventQuery(cudaEvent_t event)
{
    cudaError_t err = cudaSuccess;
    dscudaResult *rp;
    RCeventArray *ev;

    initClient();
    WARN(3, "cudaEventQuery(0x%08llx)...", (unsigned long)event);
    ev = RCeventArrayQuery(event);
    if (!ev) {
        WARN(0, "invalid event : 0x%08llx\n", event);
        exit(1);
    }
    Vdev_t *vdev = Vdev + Vdevid;
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudaeventqueryid_1((RCadr)ev->e[i], Clnt[Vdevid][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
    }

    WARN(3, "done.\n");
    return err;
}

cudaError_t
cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags)
{
    cudaError_t err = cudaSuccess;
    dscudaResult *rp;
    RCstreamArray *st;
    RCeventArray *ev;

    initClient();
    WARN(3, "cudaStreamWaitEvent(0x%08llx, 0x%08llx, 0x%08x)...",
         (unsigned long)stream, (unsigned long)event, flags);
    st = RCstreamArrayQuery(stream);
    if (!st) {
        WARN(0, "invalid stream : 0x%08llx\n", stream);
        exit(1);
    }
    ev = RCeventArrayQuery(event);
    if (!ev) {
        WARN(0, "invalid event : 0x%08llx\n", event);
        exit(1);
    }
    Vdev_t *vdev = Vdev + Vdevid;
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudastreamwaiteventid_1((RCadr)st->s[i], (RCadr)ev->e[i], flags, Clnt[Vdevid][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
    }

    WARN(3, "done.\n");
    return err;
}

/*
 * Texture Reference Management
 */

cudaChannelFormatDesc
cudaCreateChannelDesc(int x, int y, int z, int w, enum cudaChannelFormatKind f)
{
    dscudaCreateChannelDescResult *rp;
    cudaChannelFormatDesc desc;

    initClient();
    WARN(3, "cudaCreateChannelDesc()...");
    Vdev_t *vdev = Vdev + Vdevid;
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudacreatechanneldescid_1(x, y, z, w, f, Clnt[Vdevid][sp->id]);
        checkResult(rp, sp);
    }
    WARN(3, "done.\n");

    desc.x = rp->x;
    desc.y = rp->y;
    desc.z = rp->z;
    desc.w = rp->w;
    desc.f = (enum cudaChannelFormatKind)rp->f;

    return desc;
}

cudaError_t
cudaGetChannelDesc(struct cudaChannelFormatDesc *desc, const struct cudaArray *array)
{
    cudaError_t err = cudaSuccess;
    dscudaGetChannelDescResult *rp;
    RCcuarrayArray *ca;

    initClient();
    WARN(3, "cudaGetChannelDesc()...");
    ca = RCcuarrayArrayQuery((cudaArray *)array);
    if (!ca) {
        WARN(0, "invalid cudaArray : 0x%08llx\n", array);
        exit(1);
    }
    Vdev_t *vdev = Vdev + Vdevid;
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudagetchanneldescid_1((RCadr)ca->ap[i], Clnt[Vdevid][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
    }

    WARN(3, "done.\n");

    desc->x = rp->x;
    desc->y = rp->y;
    desc->z = rp->z;
    desc->w = rp->w;
    desc->f = (enum cudaChannelFormatKind)rp->f;

    WARN(3, "done.\n");
    return err;
}

cudaError_t
dscudaBindTextureWrapper(int *moduleid, char *texname,
                        size_t *offset,
                        const struct textureReference *tex,
                        const void *devPtr,
                        const struct cudaChannelFormatDesc *desc,
                        size_t size)
{
    cudaError_t err = cudaSuccess;
    dscudaBindTextureResult *rp;
    RCtexture texbuf;

    initClient();

    WARN(3, "dscudaBindTextureWrapper(0x%08llx, %s, 0x%08llx, 0x%08llx, 0x%08llx, 0x%08llx, %d)...",
         moduleid, texname,
         offset, tex, devPtr, desc, size);

    setTextureParams(&texbuf, tex, desc);

    Vdev_t *vdev = Vdev + Vdevid;
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudabindtextureid_1(moduleid[i], texname,
                                  (RCadr)devPtr, size, (RCtexture)texbuf, Clnt[Vdevid][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
    }

    if (offset) {
        *offset = rp->offset;
    }

    WARN(3, "done.\n");
    return err;
}

cudaError_t
dscudaBindTexture2DWrapper(int *moduleid, char *texname,
                          size_t *offset,
                          const struct textureReference *tex,
                          const void *devPtr,
                          const struct cudaChannelFormatDesc *desc,
                          size_t width, size_t height, size_t pitch)
{
    cudaError_t err = cudaSuccess;
    dscudaBindTexture2DResult *rp;
    RCtexture texbuf;

    initClient();

    WARN(3, "dscudaBindTexture2DWrapper(0x%08llx, %s, 0x%08llx, 0x%08llx, 0x%08llx, 0x%08llx, %d, %d, %d)...",
         moduleid, texname,
         offset, tex, devPtr, desc, width, height, pitch);

    setTextureParams(&texbuf, tex, desc);

    Vdev_t *vdev = Vdev + Vdevid;
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudabindtexture2did_1(moduleid[i], texname,
                                    (RCadr)devPtr, width, height, pitch, (RCtexture)texbuf, Clnt[Vdevid][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
    }

    if (offset) {
        *offset = rp->offset;
    }

    WARN(3, "done.\n");
    return err;
}

cudaError_t
dscudaBindTextureToArrayWrapper(int *moduleid, char *texname,
                               const struct textureReference *tex,
                               const struct cudaArray *array,
                               const struct cudaChannelFormatDesc *desc)
{
    cudaError_t err = cudaSuccess;
    dscudaResult *rp;
    RCtexture texbuf;
    RCcuarrayArray *ca;

    initClient();

    WARN(3, "dscudaBindTextureToArrayWrapper(0x%08llx, %s, 0x%08llx, 0x%08llx)...",
         moduleid, texname, (unsigned long)array, (unsigned long)desc);

    setTextureParams(&texbuf, tex, desc);

    ca = RCcuarrayArrayQuery((cudaArray *)array);
    if (!ca) {
        WARN(0, "invalid cudaArray : 0x%08llx\n", array);
        exit(1);
    }

    Vdev_t *vdev = Vdev + Vdevid;
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudabindtexturetoarrayid_1(moduleid[i], texname, (RCadr)ca->ap[i], (RCtexture)texbuf, Clnt[Vdevid][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
    }
    WARN(3, "done.\n");
    return err;
}

cudaError_t
cudaUnbindTexture(const struct textureReference * texref)
{
    cudaError_t err = cudaSuccess;

    WARN(4, "Current implementation of cudaUnbindTexture() does nothing "
         "but returning cudaSuccess.\n");

    err = cudaSuccess;

    return err;
}

#include "libdscuda.cu"
