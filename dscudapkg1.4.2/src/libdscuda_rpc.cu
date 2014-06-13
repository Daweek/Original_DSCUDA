#include "libdscuda.h"

static const int UseIbv = 0;

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
    int pgid = DSCUDA_PROG;
    char msg[256];

    struct sockaddr_in sockaddr;
    int ssock = RPC_ANYSOCK; // socket to the server for RPC communication.
                             // automatically created by clnttcp_create().
    int sport; // port number of the server. given by the daemon, or calculated from cid.

    if (UseDaemon) { // access to the server via daemon.
        sport = requestDaemonForDevice(sp->ip, cid, UseIbv);
    }
    else { // directly access to the server.
        sport = RC_SERVER_IP_PORT + cid;
    }
    sockaddr = setupSockaddr(sp->ip, sport);

    Clnt[idev][id] = clnttcp_create(&sockaddr,
                                    pgid,
                                    DSCUDA_VER,
                                    &ssock,
                                    RC_BUFSIZE, RC_BUFSIZE);

    sprintf(msg, "%s:%d (port %d) ", sp->ip, cid, sport);

    if (!Clnt[idev][id]) {
        clnt_pcreateerror(msg);
        if (0 == strcmp(sp->ip, DEFAULT_SVRIP)) {
            WARN(0, "You may need to set an environment variable 'DSCUDA_SERVER'.\n");
        }
        else {
            WARN(0, "DSCUDA server (dscudasrv on %s:%d) may be down.\n", sp->ip, id);
        }
        exit(1);
    }
    WARN(2, "Established a socket connection to %s...\n", msg);
}

static void
checkResult(void *rp, RCServer_t *sp)
{
    if (rp) return;
    clnt_perror(Clnt[Vdevid[vdevidIndex()]][sp->id], sp->ip);
    exit(1);
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
    int vid = vdevidIndex();

    initClient();
    WARN(3, "cudaThreadExit()...");
    Vdev_t *vdev = Vdev + Vdevid[vid];
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudathreadexitid_1(Clnt[Vdevid[vid]][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
        xdr_free((xdrproc_t)xdr_dscudaResult, (char *)rp);
    }
    WARN(3, "done.\n");

    return err;
}

cudaError_t
cudaThreadSynchronize(void)
{
    cudaError_t err = cudaSuccess;
    dscudaResult *rp;
    int vid = vdevidIndex();

    initClient();
    WARN(3, "cudaThreadSynchronize()...");
    Vdev_t *vdev = Vdev + Vdevid[vid];
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudathreadsynchronizeid_1(Clnt[Vdevid[vid]][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
        xdr_free((xdrproc_t)xdr_dscudaResult, (char *)rp);
    }
    WARN(3, "done.\n");

    return err;
}

cudaError_t
cudaThreadSetLimit(enum cudaLimit limit, size_t value)
{
    cudaError_t err = cudaSuccess;
    dscudaResult *rp;
    int vid = vdevidIndex();

    initClient();
    WARN(3, "cudaThreadSetLimit(%d, %d)...", limit, value);
    Vdev_t *vdev = Vdev + Vdevid[vid];
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudathreadsetlimitid_1(limit, value, Clnt[Vdevid[vid]][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
        xdr_free((xdrproc_t)xdr_dscudaResult, (char *)rp);
    }
    WARN(3, "done.\n");

    return err;
}

cudaError_t
cudaThreadGetLimit(size_t *pValue, enum cudaLimit limit)
{
    cudaError_t err = cudaSuccess;
    dscudaThreadGetLimitResult *rp;
    int vid = vdevidIndex();

    initClient();
    WARN(3, "cudaThreadGetLimit(0x%08llx, %d)...", pValue, limit);
    Vdev_t *vdev = Vdev + Vdevid[vid];
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudathreadgetlimitid_1(limit, Clnt[Vdevid[vid]][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
        if (i == 0) {
            *pValue = rp->value;
        }
        xdr_free((xdrproc_t)xdr_dscudaThreadGetLimitResult, (char *)rp);
    }
    WARN(3, "done.  *pValue: %d\n", *pValue);

    return err;
}

cudaError_t
cudaThreadSetCacheConfig(enum cudaFuncCache cacheConfig)
{
    cudaError_t err = cudaSuccess;
    dscudaResult *rp;
    int vid = vdevidIndex();

    initClient();
    WARN(3, "cudaThreadSetCacheConfig(%d)...", cacheConfig);
    Vdev_t *vdev = Vdev + Vdevid[vid];
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudathreadsetcacheconfigid_1(cacheConfig, Clnt[Vdevid[vid]][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
        xdr_free((xdrproc_t)xdr_dscudaResult, (char *)rp);
    }
    WARN(3, "done.\n");

    return err;
}

cudaError_t
cudaThreadGetCacheConfig(enum cudaFuncCache *pCacheConfig)
{
    cudaError_t err = cudaSuccess;
    dscudaThreadGetCacheConfigResult *rp;
    int vid = vdevidIndex();

    initClient();
    WARN(3, "cudaThreadGetCacheConfig(0x%08llx)...", pCacheConfig);
    Vdev_t *vdev = Vdev + Vdevid[vid];
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudathreadgetcacheconfigid_1(Clnt[Vdevid[vid]][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
        if (i == 0) {
            *pCacheConfig = (enum cudaFuncCache)rp->cacheConfig;
        }
        xdr_free((xdrproc_t)xdr_dscudaThreadGetCacheConfigResult, (char *)rp);
    }
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
    int vid = vdevidIndex();

    initClient();
    WARN(5, "cudaGetLastError()...");
    Vdev_t *vdev = Vdev + Vdevid[vid];
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudagetlasterrorid_1(Clnt[Vdevid[vid]][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
        xdr_free((xdrproc_t)xdr_dscudaResult, (char *)rp);
    }
    WARN(5, "done.\n");

    return err;
}

cudaError_t
cudaPeekAtLastError(void)
{
    cudaError_t err = cudaSuccess;
    dscudaResult *rp;
    int vid = vdevidIndex();

    initClient();
    WARN(5, "cudaPeekAtLastError()...");
    Vdev_t *vdev = Vdev + Vdevid[vid];
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudapeekatlasterrorid_1(Clnt[Vdevid[vid]][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
        xdr_free((xdrproc_t)xdr_dscudaResult, (char *)rp);
    }
    WARN(5, "done.\n");

    return err;
}

const char *
cudaGetErrorString(cudaError_t error)
{
    dscudaGetErrorStringResult *rp;
    static char str[4096];
    int vid = vdevidIndex();

    initClient();
    WARN(5, "cudaGetErrorString()...");
    Vdev_t *vdev = Vdev + Vdevid[vid];
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudageterrorstringid_1(error, Clnt[Vdevid[vid]][sp->id]);
        checkResult(rp, sp);
        if (i == 0) {
            strcpy(str, rp->errmsg);
        }
        xdr_free((xdrproc_t)xdr_dscudaGetErrorStringResult, (char *)rp);
    }
    WARN(5, "done.\n");

    return str;
}

/*
 * Device Management
 */

cudaError_t
cudaSetDeviceFlags(unsigned int flags)
{
    cudaError_t err = cudaSuccess;
    dscudaResult *rp;
    int vid = vdevidIndex();

    initClient();
    WARN(3, "cudaSetDeviceFlags()...");
    Vdev_t *vdev = Vdev + Vdevid[vid];
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudasetdeviceflagsid_1(flags, Clnt[Vdevid[vid]][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
        xdr_free((xdrproc_t)xdr_dscudaResult, (char *)rp);
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
    int vid = vdevidIndex();

    initClient();
    WARN(3, "cudaDriverGetVersionCount(0x%08llx)...", (unsigned long)driverVersion);
    Vdev_t *vdev = Vdev + Vdevid[vid];
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudadrivergetversionid_1(Clnt[Vdevid[vid]][sp->id]);
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
    int vid = vdevidIndex();

    initClient();
    WARN(3, "cudaRuntimeGetVersion(0x%08llx)...", (unsigned long)runtimeVersion);
    Vdev_t *vdev = Vdev + Vdevid[vid];
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudaruntimegetversionid_1(Clnt[Vdevid[vid]][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
        if (i == 0) {
            *runtimeVersion = rp->ver;
        }
        xdr_free((xdrproc_t)xdr_dscudaRuntimeGetVersionResult, (char *)rp);
    }

    WARN(3, "done.\n");

    return err;
}

cudaError_t
cudaDeviceSynchronize(void)
{
    cudaError_t err = cudaSuccess;
    dscudaResult *rp;
    int vid = vdevidIndex();

    initClient();
    WARN(3, "cudaDeviceSynchronize()...");
    Vdev_t *vdev = Vdev + Vdevid[vid];
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudadevicesynchronize_1(Clnt[Vdevid[vid]][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
        xdr_free((xdrproc_t)xdr_dscudaResult, (char *)rp);
    }
    WARN(3, "done.\n");

    return err;
}

cudaError_t
cudaDeviceReset(void)
{
    cudaError_t err = cudaSuccess;
    dscudaResult *rp;
    int vid = vdevidIndex();

    initClient();
    WARN(3, "cudaDeviceReset()...");
    Vdev_t *vdev = Vdev + Vdevid[vid];
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudadevicereset_1(Clnt[Vdevid[vid]][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
        xdr_free((xdrproc_t)xdr_dscudaResult, (char *)rp);
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

/*
 * Memory Management
 */

cudaError_t
cudaMalloc(void **devAdrPtr, size_t size)
{
    cudaError_t err = cudaSuccess;
    dscudaMallocResult *rp;
    int vid = vdevidIndex();
    void *adrs[RC_NREDUNDANCYMAX];

    initClient();
    WARN(3, "cudaMalloc(0x%08llx, %d)...", (unsigned long)devAdrPtr, size);
    Vdev_t *vdev = Vdev + Vdevid[vid];
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudamallocid_1(size, Clnt[Vdevid[vid]][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
        adrs[i] = (void*)rp->devAdr;
        xdr_free((xdrproc_t)xdr_dscudaMallocResult, (char *)rp);
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
    dscudaResult *rp;
    int vid = vdevidIndex();

    initClient();
    WARN(3, "cudaFree(0x%08llx)...", (unsigned long)mem);
    Vdev_t *vdev = Vdev + Vdevid[vid];
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudafreeid_1((RCadr)mem, Clnt[Vdevid[vid]][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
        xdr_free((xdrproc_t)xdr_dscudaResult, (char *)rp);
    }

    RCuvaUnregister(mem);

    WARN(3, "done.\n");

    return err;
}

static cudaError_t
cudaMemcpyH2D(void *dst, const void *src, size_t count, Vdev_t *vdev, CLIENT **clnt)
{
    dscudaResult *rp;
    RCServer_t *sp;
    RCbuf srcbuf;
    cudaError_t err = cudaSuccess;

    srcbuf.RCbuf_len = count;
    srcbuf.RCbuf_val = (char *)src;
    sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudamemcpyh2did_1((RCadr)dst, srcbuf, count, clnt[sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
        xdr_free((xdrproc_t)xdr_dscudaResult, (char *)rp);
    }
    return err;
}

static cudaError_t
cudaMemcpyD2H(void *dst, const void *src, size_t count, Vdev_t *vdev, CLIENT **clnt)
{
    dscudaMemcpyD2HResult *rp;
    RCServer_t *sp;
    cudaError_t err = cudaSuccess;

    sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudamemcpyd2hid_1((RCadr)src, count, clnt[sp->id]);
        checkResult(rp, sp);
        err = (cudaError_t)rp->err;
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
        if (i == 0) {
            memcpy(dst, rp->buf.RCbuf_val, rp->buf.RCbuf_len);
        }
        else if (bcmp(dst, rp->buf.RCbuf_val, rp->buf.RCbuf_len) != 0) {
            WARN(2, "\n\ncudaMemcpy() data copied from device%d & device0 UNMATCHED.\n\n\n", i);
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
        xdr_free((xdrproc_t)xdr_dscudaMemcpyD2HResult, (char *)rp);
    }
    return err;
}

static cudaError_t
cudaMemcpyD2D(void *dst, const void *src, size_t count, Vdev_t *vdev, CLIENT **clnt)
{
    dscudaResult *rp;
    RCServer_t *sp;
    cudaError_t err = cudaSuccess;

    sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudamemcpyd2did_1((RCadr)dst, (RCadr)src, count, clnt[sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
        xdr_free((xdrproc_t)xdr_dscudaResult, (char *)rp);
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
    CLIENT **clnt;
    RCuva *suva, *duva;
    int dev0;
    void *lsrc, *ldst;

    initClient();

    WARN(3, "cudaMemcpy(0x%08lx, 0x%08lx, %d, %s)...",
         (unsigned long)dst, (unsigned long)src, count, dscudaMemcpyKindName(kind));

    vdevid = Vdevid[vdevidIndex()];
    vdev = Vdev + vdevid;
    clnt = Clnt[vdevid];

    lsrc = dscudaAdrOfUva((void *)src);
    ldst = dscudaAdrOfUva(dst);
    switch (kind) {
      case cudaMemcpyDeviceToHost:
        err = cudaMemcpyD2H(ldst, lsrc, count, vdev, clnt);
        break;
      case cudaMemcpyHostToDevice:
        err = cudaMemcpyH2D(ldst, lsrc, count, vdev, clnt);
        break;
      case cudaMemcpyDeviceToDevice:
        err = cudaMemcpyD2D(ldst, lsrc, count, vdev, clnt);
        break;
      case cudaMemcpyDefault:
#if !__LP64__
        WARN(0, "cudaMemcpy:In 32-bit environment, cudaMemcpyDefault cannot be given as arg4."
             "UVA is supported for 64-bit environment only.\n");
        exit(1);
#endif

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

    WARN(3, "done.\n");

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
        if (i == 0) {
            memcpy(prop, rp->prop.RCbuf_val, rp->prop.RCbuf_len);
        }
        xdr_free((xdrproc_t)xdr_dscudaGetDevicePropertiesResult, (char *)rp);
    }
    WARN(3, "done.\n");

    return err;
}

static int
dscudaLoadModuleLocal(unsigned int ipaddr, pid_t pid, char *modulename, char *modulebuf, int vdevid, int raidid)
{
    int ret;
    RCServer_t *sp = (Vdev + vdevid)->server + raidid;
    dscudaLoadModuleResult *rp = dscudaloadmoduleid_1(MyIpaddr, getpid(), modulename, modulebuf, Clnt[vdevid][sp->id]);
    checkResult(rp, sp);
    ret = rp->id;
    xdr_free((xdrproc_t)xdr_dscudaLoadModuleResult, (char *)rp);

    return ret;
}

/*
 * launch a kernel function of id 'kid', defined in a module of id 'moduleid'.
 * 'kid' must be unique inside a single module.
 */
void
rpcDscudaLaunchKernelWrapper(int *moduleid, int kid, char *kname,
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

    Vdev_t *vdev = Vdev + Vdevid[vdevidIndex()];
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        void *rp = dscudalaunchkernelid_1(moduleid[i], kid, kname,
                                          gdim, bdim, smemsize, (RCstream)st->s[i],
                                          args, Clnt[Vdevid[vdevidIndex()]][sp->id]);
        checkResult(rp, sp);
    }

    mem = RCmappedMemListTop;
    while (mem) {
        cudaMemcpy(mem->pHost, mem->pDevice, mem->size, cudaMemcpyDeviceToHost);
        mem = mem->next;
    }

    if (autoVerb) {
        cudaRpcLaunchKernelArgs args2;
        args2.moduleid = moduleid;
        args2.kid = kid;
        args2.kname = kname;
        args2.gdim = gdim;
        args2.bdim = bdim;
        args2.smemsize = smemsize;
        args2.stream = stream;
        args2.args = args;
        dscudaVerbAddHist(dscudaLaunchKernelId, (void *)&args2);
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

    Vdev_t *vdev = Vdev + Vdevid[vdevidIndex()];
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudamallocarrayid_1(descbuf, width, height, flags, Clnt[Vdevid[vdevidIndex()]][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
        ca[i] = (cudaArray *)rp->array;
        xdr_free((xdrproc_t)xdr_dscudaMallocArrayResult, (char *)rp);
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
    Vdev_t *vdev = Vdev + Vdevid[vdevidIndex()];
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudafreearrayid_1((RCadr)ca->ap[i], Clnt[Vdevid[vdevidIndex()]][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
        xdr_free((xdrproc_t)xdr_dscudaResult, (char *)rp);
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

        vdev = Vdev + Vdevid[vdevidIndex()];
        sp = vdev->server;
        for (int i = 0; i < vdev->nredundancy; i++, sp++) {
            h2drp = dscudamemcpytoarrayh2did_1((RCadr)ca->ap[i], wOffset, hOffset, srcbuf, count, Clnt[Vdevid[vdevidIndex()]][sp->id]);
            checkResult(h2drp, sp);
            if (h2drp->err != cudaSuccess) {
                err = (cudaError_t)h2drp->err;
            }
            xdr_free((xdrproc_t)xdr_dscudaResult, (char *)h2drp);
        }
        break;
      case cudaMemcpyDeviceToDevice:
        vdev = Vdev + Vdevid[vdevidIndex()];
        sp = vdev->server;
        for (int i = 0; i < vdev->nredundancy; i++, sp++) {
            d2drp = dscudamemcpytoarrayd2did_1((RCadr)ca->ap[i], wOffset, hOffset, (RCadr)src, count, Clnt[Vdevid[vdevidIndex()]][sp->id]);
            checkResult(d2drp, sp);
            if (d2drp->err != cudaSuccess) {
                err = (cudaError_t)d2drp->err;
            }
            xdr_free((xdrproc_t)xdr_dscudaResult, (char *)d2drp);
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
    Vdev_t *vdev = Vdev + Vdevid[vdevidIndex()];
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudamemsetid_1((RCadr)devPtr, value, count, Clnt[Vdevid[vdevidIndex()]][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
        xdr_free((xdrproc_t)xdr_dscudaResult, (char *)rp);
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
    Vdev_t *vdev = Vdev + Vdevid[vdevidIndex()];
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudamallocpitchid_1(width, height, Clnt[Vdevid[vdevidIndex()]][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
        if (i == 0) {
            *devPtr = (void*)rp->devPtr;
            *pitch = rp->pitch;
        }
        xdr_free((xdrproc_t)xdr_dscudaMallocPitchResult, (char *)rp);
    }

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
        vdev = Vdev + Vdevid[vdevidIndex()];
        sp = vdev->server;
        for (int i = 0; i < vdev->nredundancy; i++, sp++) {
            d2hrp = dscudamemcpy2dtoarrayd2hid_1(wOffset, hOffset,
                                                 (RCadr)src, spitch, width, height, Clnt[Vdevid[vdevidIndex()]][sp->id]);
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
            xdr_free((xdrproc_t)xdr_dscudaMemcpy2DToArrayD2HResult, (char *)d2hrp);
        }
        break;
      case cudaMemcpyHostToDevice:
        srcbuf.RCbuf_len = spitch * height;
        srcbuf.RCbuf_val = (char *)src;
        vdev = Vdev + Vdevid[vdevidIndex()];
        sp = vdev->server;
        for (int i = 0; i < vdev->nredundancy; i++, sp++) {
            h2drp = dscudamemcpy2dtoarrayh2did_1((RCadr)ca->ap[i], wOffset, hOffset,
                                                 srcbuf, spitch, width, height, Clnt[Vdevid[vdevidIndex()]][sp->id]);
            checkResult(h2drp, sp);
            if (h2drp->err != cudaSuccess) {
                err = (cudaError_t)h2drp->err;
            }
            xdr_free((xdrproc_t)xdr_dscudaResult, (char *)h2drp);
        }
        break;
      case cudaMemcpyDeviceToDevice:
        vdev = Vdev + Vdevid[vdevidIndex()];
        sp = vdev->server;
        for (int i = 0; i < vdev->nredundancy; i++, sp++) {
            d2drp = dscudamemcpy2dtoarrayd2did_1((RCadr)ca->ap[i], wOffset, hOffset,
                                                 (RCadr)src, spitch, width, height, Clnt[Vdevid[vdevidIndex()]][sp->id]);
            checkResult(d2drp, sp);
            if (d2drp->err != cudaSuccess) {
                err = (cudaError_t)d2drp->err;
            }
            xdr_free((xdrproc_t)xdr_dscudaResult, (char *)d2drp);
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
        vdev = Vdev + Vdevid[vdevidIndex()];
        sp = vdev->server;
        for (int i = 0; i < vdev->nredundancy; i++, sp++) {
            d2hrp = dscudamemcpy2dd2hid_1(dpitch,
                                          (RCadr)src, spitch, width, height, Clnt[Vdevid[vdevidIndex()]][sp->id]);
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
            xdr_free((xdrproc_t)xdr_dscudaMemcpy2DD2HResult, (char *)d2hrp);
        }
        break;
      case cudaMemcpyHostToDevice:
        srcbuf.RCbuf_len = spitch * height;
        srcbuf.RCbuf_val = (char *)src;
        vdev = Vdev + Vdevid[vdevidIndex()];
        sp = vdev->server;
        for (int i = 0; i < vdev->nredundancy; i++, sp++) {
            h2drp = dscudamemcpy2dh2did_1((RCadr)dst, dpitch,
                                          srcbuf, spitch, width, height, Clnt[Vdevid[vdevidIndex()]][sp->id]);
            checkResult(h2drp, sp);
            if (h2drp->err != cudaSuccess) {
                err = (cudaError_t)h2drp->err;
            }
            xdr_free((xdrproc_t)xdr_dscudaResult, (char *)h2drp);
        }
        break;
      case cudaMemcpyDeviceToDevice:
        vdev = Vdev + Vdevid[vdevidIndex()];
        sp = vdev->server;
        for (int i = 0; i < vdev->nredundancy; i++, sp++) {
            d2drp = dscudamemcpy2dd2did_1((RCadr)dst, dpitch,
                                          (RCadr)src, spitch, width, height, Clnt[Vdevid[vdevidIndex()]][sp->id]);
            checkResult(d2drp, sp);
            if (d2drp->err != cudaSuccess) {
                err = (cudaError_t)d2drp->err;
            }
            xdr_free((xdrproc_t)xdr_dscudaResult, (char *)d2drp);
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
    Vdev_t *vdev = Vdev + Vdevid[vdevidIndex()];
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudamemset2did_1((RCadr)devPtr, pitch, value, width, height, Clnt[Vdevid[vdevidIndex()]][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
        xdr_free((xdrproc_t)xdr_dscudaResult, (char *)rp);
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
    Vdev_t *vdev = Vdev + Vdevid[vdevidIndex()];
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudamallochostid_1(size, Clnt[Vdevid[vdevidIndex()]][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
        if (i == 0) {
            *ptr = (void*)rp->ptr;
        }
        xdr_free((xdrproc_t)xdr_dscudaMallocHostResult, (char *)rp);
    }

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
    Vdev_t *vdev = Vdev + Vdevid[vdevidIndex()];
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudahostallocid_1(size, flags, Clnt[Vdevid[vdevidIndex()]][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
        if (i == 0) {
            *pHost = (void*)rp->pHost;
        }
        xdr_free((xdrproc_t)xdr_dscudaHostAllocResult, (char *)rp);
    }

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
    Vdev_t *vdev = Vdev + Vdevid[vdevidIndex()];
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudafreehostid_1((RCadr)ptr, Clnt[Vdevid[vdevidIndex()]][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
        xdr_free((xdrproc_t)xdr_dscudaResult, (char *)rp);
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
    Vdev_t *vdev = Vdev + Vdevid[vdevidIndex()];
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudahostgetdevicepointerid_1((RCadr)pHost, flags, Clnt[Vdevid[vdevidIndex()]][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
        if (i == 0) {
            *pDevice = (void *)rp->pDevice;
        }
        xdr_free((xdrproc_t)xdr_dscudaHostGetDevicePointerResult, (char *)rp);
    }

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
    Vdev_t *vdev = Vdev + Vdevid[vdevidIndex()];
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudahostgetflagsid_1((RCadr)pHost, Clnt[Vdevid[vdevidIndex()]][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
        if (i == 0) {
            *pFlags = rp->flags;
        }
        xdr_free((xdrproc_t)xdr_dscudaHostGetFlagsResult, (char *)rp);
    }
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
        Vdev_t *vdev = Vdev + Vdevid[vdevidIndex()];
        RCServer_t *sp = vdev->server;
        for (int i = 0; i < vdev->nredundancy; i++, sp++) {
            d2hrp = dscudamemcpyasyncd2hid_1((RCadr)src, count, (RCstream)st->s[i], Clnt[Vdevid[vdevidIndex()]][sp->id]);
            checkResult(d2hrp, sp);
            if (d2hrp->err != cudaSuccess) {
                err = (cudaError_t)d2hrp->err;
            }
            if (i == 0) {
                memcpy(dst, d2hrp->buf.RCbuf_val, d2hrp->buf.RCbuf_len);
            }
            xdr_free((xdrproc_t)xdr_dscudaMemcpyAsyncD2HResult, (char *)d2hrp);
        }
        break;
      case cudaMemcpyHostToDevice:
        srcbuf.RCbuf_len = count;
        srcbuf.RCbuf_val = (char *)src;
        Vdev_t *vdev = Vdev + Vdevid[vdevidIndex()];
        RCServer_t *sp = vdev->server;
        for (int i = 0; i < vdev->nredundancy; i++, sp++) {
            h2drp = dscudamemcpyasynch2did_1((RCadr)dst, srcbuf, count, (RCstream)st->s[i], Clnt[Vdevid[vdevidIndex()]][sp->id]);
            checkResult(h2drp, sp);
            if (h2drp->err != cudaSuccess) {
                err = (cudaError_t)h2drp->err;
            }
            xdr_free((xdrproc_t)xdr_dscudaResult, (char *)h2drp);
        }
        break;
      case cudaMemcpyDeviceToDevice:
        Vdev_t *vdev = Vdev + Vdevid[vdevidIndex()];
        RCServer_t *sp = vdev->server;
        for (int i = 0; i < vdev->nredundancy; i++, sp++) {
            d2drp = dscudamemcpyasyncd2did_1((RCadr)dst, (RCadr)src, count, (RCstream)st->s[i], Clnt[Vdevid[vdevidIndex()]][sp->id]);
            checkResult(d2drp, sp);
            if (d2drp->err != cudaSuccess) {
                err = (cudaError_t)d2drp->err;
            }
            xdr_free((xdrproc_t)xdr_dscudaResult, (char *)d2drp);
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

static cudaError_t
dscudaMemcpyToSymbolH2D(int moduleid, char *symbol, const void *src,
                        size_t count, size_t offset, int vdevid, int raidid)
{
    dscudaResult *rp;
    RCServer_t *sp = (Vdev + vdevid)->server + raidid;
    RCbuf srcbuf;
    cudaError_t err;

    srcbuf.RCbuf_len = count;
    srcbuf.RCbuf_val = (char *)src;
    rp = dscudamemcpytosymbolh2did_1(moduleid, symbol, srcbuf, count, offset, Clnt[vdevid][sp->id]);
    checkResult(rp, sp);
    err = (cudaError_t)rp->err;
    xdr_free((xdrproc_t)xdr_dscudaResult, (char *)rp);

    return (cudaError_t)err;
}

static cudaError_t
dscudaMemcpyToSymbolD2D(int moduleid, char *symbol, const void *src,
                        size_t count, size_t offset, int vdevid, int raidid)
{
    dscudaResult *rp;
    RCServer_t *sp = (Vdev + vdevid)->server + raidid;
    cudaError_t err;

    rp = dscudamemcpytosymbold2did_1(moduleid, symbol, (RCadr)src, count, offset, Clnt[vdevid][sp->id]);
    checkResult(rp, sp);
    err = (cudaError_t)rp->err;
    xdr_free((xdrproc_t)xdr_dscudaResult, (char *)rp);

    return (cudaError_t)err;
}

static cudaError_t
dscudaMemcpyFromSymbolD2H(int moduleid, void **dstbuf, char *symbol,
                          size_t count, size_t offset, int vdevid, int raidid)
{
    dscudaMemcpyFromSymbolD2HResult *rp;
    RCServer_t *sp = (Vdev + vdevid)->server + raidid;
    cudaError_t err;

    rp = dscudamemcpyfromsymbold2hid_1(moduleid, (char *)symbol, count, offset, Clnt[vdevid][sp->id]);
    *dstbuf = rp->buf.RCbuf_val;
    checkResult(rp, sp);
    err = (cudaError_t)rp->err;
    xdr_free((xdrproc_t)xdr_dscudaMemcpyFromSymbolD2HResult, (char *)rp);

    return (cudaError_t)err;
}

static cudaError_t
dscudaMemcpyFromSymbolD2D(int moduleid, void *dstadr, char *symbol,
                          size_t count, size_t offset, int vdevid, int raidid)
{
    dscudaResult *rp;
    RCServer_t *sp = (Vdev + vdevid)->server + raidid;
    cudaError_t err;

    rp = dscudamemcpyfromsymbold2did_1(moduleid, (RCadr)dstadr, (char *)symbol, count, offset, Clnt[vdevid][sp->id]);
    checkResult(rp, sp);
    err = (cudaError_t)rp->err;
    xdr_free((xdrproc_t)xdr_dscudaResult, (char *)rp);

    return (cudaError_t)err;
}

static cudaError_t
dscudaMemcpyToSymbolAsyncH2D(int moduleid, char *symbol, const void *src,
                             size_t count, size_t offset, RCstream stream, int vdevid, int raidid)
{
    dscudaResult *rp;
    RCServer_t *sp = (Vdev + vdevid)->server + raidid;
    RCbuf srcbuf;
    cudaError_t err;

    srcbuf.RCbuf_len = count;
    srcbuf.RCbuf_val = (char *)src;
    rp = dscudamemcpytosymbolasynch2did_1(moduleid, symbol, srcbuf, count, offset, stream, Clnt[vdevid][sp->id]);
    checkResult(rp, sp);
    err = (cudaError_t)rp->err;
    xdr_free((xdrproc_t)xdr_dscudaResult, (char *)rp);

    return (cudaError_t)err;
}

static cudaError_t
dscudaMemcpyToSymbolAsyncD2D(int moduleid, char *symbol, const void *src,
                             size_t count, size_t offset, RCstream stream, int vdevid, int raidid)
{
    dscudaResult *rp;
    RCServer_t *sp = (Vdev + vdevid)->server + raidid;
    cudaError_t err;

    rp = dscudamemcpytosymbolasyncd2did_1(moduleid, symbol, (RCadr)src, count, offset, stream,
                                          Clnt[vdevid][sp->id]);
    checkResult(rp, sp);
    err = (cudaError_t)rp->err;
    xdr_free((xdrproc_t)xdr_dscudaResult, (char *)rp);

    return (cudaError_t)err;
}

static cudaError_t
dscudaMemcpyFromSymbolAsyncD2H(int moduleid, void **dstbuf, char *symbol,
                               size_t count, size_t offset, RCstream stream, int vdevid, int raidid)
{
    dscudaMemcpyFromSymbolAsyncD2HResult *rp;
    RCServer_t *sp = (Vdev + vdevid)->server + raidid;
    cudaError_t err;

    rp = dscudamemcpyfromsymbolasyncd2hid_1(moduleid, (char *)symbol, count, offset,
                                            stream, Clnt[vdevid][sp->id]);
    *dstbuf = rp->buf.RCbuf_val;
    checkResult(rp, sp);
    err = (cudaError_t)rp->err;
    xdr_free((xdrproc_t)xdr_dscudaMemcpyFromSymbolAsyncD2HResult, (char *)rp);

    return (cudaError_t)err;
}

static cudaError_t
dscudaMemcpyFromSymbolAsyncD2D(int moduleid, void *dstadr, char *symbol,
                               size_t count, size_t offset, RCstream stream, int vdevid, int raidid)
{
    dscudaResult *rp;
    RCServer_t *sp = (Vdev + vdevid)->server + raidid;
    cudaError_t err;

    rp = dscudamemcpyfromsymbolasyncd2did_1(moduleid, (RCadr)dstadr, (char *)symbol, count, offset,
                                            stream, Clnt[vdevid][sp->id]);
    checkResult(rp, sp);
    err = (cudaError_t)rp->err;
    xdr_free((xdrproc_t)xdr_dscudaResult, (char *)rp);

    return (cudaError_t)err;
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
    Vdev_t *vdev = Vdev + Vdevid[vdevidIndex()];
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudastreamcreateid_1(Clnt[Vdevid[vdevidIndex()]][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
        st[i] = (cudaStream_t)rp->stream;
        xdr_free((xdrproc_t)xdr_dscudaStreamCreateResult, (char *)rp);
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
    Vdev_t *vdev = Vdev + Vdevid[vdevidIndex()];
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudastreamdestroyid_1((RCadr)st->s[i], Clnt[Vdevid[vdevidIndex()]][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
        xdr_free((xdrproc_t)xdr_dscudaResult, (char *)rp);
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
    Vdev_t *vdev = Vdev + Vdevid[vdevidIndex()];
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudastreamsynchronizeid_1((RCadr)st->s[i], Clnt[Vdevid[vdevidIndex()]][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
        xdr_free((xdrproc_t)xdr_dscudaResult, (char *)rp);
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
    Vdev_t *vdev = Vdev + Vdevid[vdevidIndex()];
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudastreamqueryid_1((RCadr)st->s[i], Clnt[Vdevid[vdevidIndex()]][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
        xdr_free((xdrproc_t)xdr_dscudaResult, (char *)rp);
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
    Vdev_t *vdev = Vdev + Vdevid[vdevidIndex()];
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudaeventcreateid_1(Clnt[Vdevid[vdevidIndex()]][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
        ev[i] = (cudaEvent_t)rp->event;
        xdr_free((xdrproc_t)xdr_dscudaEventCreateResult, (char *)rp);
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
    Vdev_t *vdev = Vdev + Vdevid[vdevidIndex()];
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudaeventcreatewithflagsid_1(flags, Clnt[Vdevid[vdevidIndex()]][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
        ev[i] = (cudaEvent_t)rp->event;
        xdr_free((xdrproc_t)xdr_dscudaEventCreateResult, (char *)rp);
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
    Vdev_t *vdev = Vdev + Vdevid[vdevidIndex()];
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudaeventdestroyid_1((RCadr)ev->e[i], Clnt[Vdevid[vdevidIndex()]][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
        xdr_free((xdrproc_t)xdr_dscudaResult, (char *)rp);
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
    Vdev_t *vdev = Vdev + Vdevid[vdevidIndex()];
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudaeventelapsedtimeid_1((RCadr)es->e[i], (RCadr)ee->e[i], Clnt[Vdevid[vdevidIndex()]][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
        xdr_free((xdrproc_t)xdr_dscudaEventElapsedTimeResult, (char *)rp);
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
    Vdev_t *vdev = Vdev + Vdevid[vdevidIndex()];
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudaeventrecordid_1((RCadr)ev->e[i], (RCadr)st->s[i], Clnt[Vdevid[vdevidIndex()]][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
        xdr_free((xdrproc_t)xdr_dscudaResult, (char *)rp);
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
    Vdev_t *vdev = Vdev + Vdevid[vdevidIndex()];
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudaeventsynchronizeid_1((RCadr)ev->e[i], Clnt[Vdevid[vdevidIndex()]][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
        xdr_free((xdrproc_t)xdr_dscudaResult, (char *)rp);
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
    Vdev_t *vdev = Vdev + Vdevid[vdevidIndex()];
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudaeventqueryid_1((RCadr)ev->e[i], Clnt[Vdevid[vdevidIndex()]][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
        xdr_free((xdrproc_t)xdr_dscudaResult, (char *)rp);
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
    Vdev_t *vdev = Vdev + Vdevid[vdevidIndex()];
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudastreamwaiteventid_1((RCadr)st->s[i], (RCadr)ev->e[i], flags, Clnt[Vdevid[vdevidIndex()]][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
        xdr_free((xdrproc_t)xdr_dscudaResult, (char *)rp);
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
    Vdev_t *vdev = Vdev + Vdevid[vdevidIndex()];
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudacreatechanneldescid_1(x, y, z, w, f, Clnt[Vdevid[vdevidIndex()]][sp->id]);
        checkResult(rp, sp);
        if (i == 0) {
            desc.x = rp->x;
            desc.y = rp->y;
            desc.z = rp->z;
            desc.w = rp->w;
            desc.f = (enum cudaChannelFormatKind)rp->f;
        }
        xdr_free((xdrproc_t)xdr_dscudaCreateChannelDescResult, (char *)rp);
    }
    WARN(3, "done.\n");

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
    Vdev_t *vdev = Vdev + Vdevid[vdevidIndex()];
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudagetchanneldescid_1((RCadr)ca->ap[i], Clnt[Vdevid[vdevidIndex()]][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
        if (i == 0) {
            desc->x = rp->x;
            desc->y = rp->y;
            desc->z = rp->z;
            desc->w = rp->w;
            desc->f = (enum cudaChannelFormatKind)rp->f;
        }
        xdr_free((xdrproc_t)xdr_dscudaGetChannelDescResult, (char *)rp);
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

/*
 * CUFFT library
 */
cufftResult CUFFTAPI
cufftPlan3d(cufftHandle *plan, int nx, int ny, int nz, cufftType type)
{
    cufftResult res = CUFFT_SUCCESS;
    dscufftPlanResult *rp;

    initClient();
    WARN(3, "cufftPlan3d()...");
    Vdev_t *vdev = Vdev + Vdevid[vdevidIndex()];
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscufftplan3did_1(nx, ny, nz, (unsigned int)type, Clnt[0][sp->id]);
        checkResult(rp, sp);
        if (rp->err != CUFFT_SUCCESS) {
            res = (cufftResult)rp->err;
        }
        if (i == 0) {
            *plan = rp->plan;
        }
        xdr_free((xdrproc_t)xdr_dscufftPlanResult, (char *)rp);
    }

    WARN(3, "done.\n");

    return res;
}

cufftResult CUFFTAPI
cufftDestroy(cufftHandle plan)
{
    cufftResult res = CUFFT_SUCCESS;
    dscufftResult *rp;

    initClient();
    WARN(3, "cufftDestroy()...");
    Vdev_t *vdev = Vdev + Vdevid[vdevidIndex()];
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscufftdestroyid_1((unsigned int)plan, Clnt[0][sp->id]);
        checkResult(rp, sp);
        if (rp->err != CUFFT_SUCCESS) {
            res = (cufftResult)rp->err;
        }
        xdr_free((xdrproc_t)xdr_dscufftResult, (char *)rp);
    }
    WARN(3, "done.\n");

    return res;
}

cufftResult CUFFTAPI
cufftExecC2C(cufftHandle plan, cufftComplex *idata, cufftComplex *odata, int direction)
{
    cufftResult res = CUFFT_SUCCESS;
    dscufftResult *rp;

    initClient();
    WARN(3, "cufftExecC2C()...");
    Vdev_t *vdev = Vdev + Vdevid[vdevidIndex()];
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscufftexecc2cid_1((unsigned int)plan, (RCadr)idata, (RCadr)odata, direction, Clnt[0][sp->id]);
        checkResult(rp, sp);
        if (rp->err != CUFFT_SUCCESS) {
            res = (cufftResult)rp->err;
        }
        xdr_free((xdrproc_t)xdr_dscufftResult, (char *)rp);
    }

    WARN(3, "done.\n");

    return res;
}

#if 0

/*
 * Interface to CUFFT & CUBLAS written by Yoshikawa for old Remote CUDA.
 * some are already ported to DS-CUDA (see 'dscufftXXXid_1_svc' function defs above),
 * but some are not. Maybe someday, when I have time...
 */

cufftResult CUFFTAPI
cufftPlan1d(cufftHandle *plan, int nx, cufftType type, int batch)
{
    cufftResult res = CUFFT_SUCCESS;
    rcufftPlanResult *rp;

    initClient();
    WARN(3, "cufftPlan1d()...");
    Server *sp = Serverlist;
    for (int i = 0; i < Nredundancy; i++, sp++) {
        rp = rcufftplan1did_1(nx, (unsigned int)type, batch, Clnt[0][sp->id]);
        checkResult(rp, sp);
        if (rp->err != CUFFT_SUCCESS) {
            res = (cufftResult)rp->err;
        }
    }
    *plan = rp->plan;

    WARN(3, "done.\n");

    return res;
}

cufftResult CUFFTAPI
cufftPlan2d(cufftHandle *plan, int nx, int ny, cufftType type, int batch)
{
    cufftResult res = CUFFT_SUCCESS;
    rcufftPlanResult *rp;

    initClient();
    WARN(3, "cufftPlan2d()...");
    Server *sp = Serverlist;
    for (int i = 0; i < Nredundancy; i++, sp++) {
        rp = rcufftplan2did_1(nx, ny, (unsigned int)type, Clnt[0][sp->id]);
        checkResult(rp, sp);
        if (rp->err != CUFFT_SUCCESS) {
            res = (cufftResult)rp->err;
        }
    }
    *plan = rp->plan;

    WARN(3, "done.\n");

    return res;
}

/*
  cufftResult CUFFTAPI
  cufftPlanMany(cufftHandle *plan, int nx, cufftType type, int batch)
  {
  cufftResult res = CUFFT_SUCCESS;
  rcufftPlanResult *rp;

  initClient();
  WARN(3, "cufftPlan1d()...");
  Server *sp = Serverlist;
  for (int i = 0; i < Nredundancy; i++, sp++) {
  rp = rcufftplan1did_1(nx, (unsigned int)type, Clnt[0][sp->id]);
  checkResult(rp, sp);
  if (rp->err != CUFFT_SUCCESS) {
  res = (cufftResult)rp->err;
  }
  }
  *plan = rp->plan;

  WARN(3, "done.\n");

  return res;
  }
*/

cufftResult CUFFTAPI
cufftExecR2C(cufftHandle plan, cufftComplex *idata, cufftComplex *odata)
{
    cufftResult res = CUFFT_SUCCESS;
    rcufftResult *rp;

    initClient();
    WARN(3, "cufftExecR2C()...");
    Server *sp = Serverlist;
    for (int i = 0; i < Nredundancy; i++, sp++) {
        rp = rcufftexecr2cid_1((unsigned int)plan, (RCadr)idata, (RCadr)odata, Clnt[0][sp->id]);
        checkResult(rp, sp);
        if (rp->err != CUFFT_SUCCESS) {
            res = (cufftResult)rp->err;
        }
    }

    WARN(3, "done.\n");

    return res;
}

cufftResult CUFFTAPI
cufftExecC2R(cufftHandle plan, cufftComplex *idata, cufftComplex *odata)
{
    cufftResult res = CUFFT_SUCCESS;
    rcufftResult *rp;

    initClient();
    WARN(3, "cufftExecC2R()...");
    Server *sp = Serverlist;
    for (int i = 0; i < Nredundancy; i++, sp++) {
        rp = rcufftexecc2rid_1((unsigned int)plan, (RCadr)idata, (RCadr)odata, Clnt[0][sp->id]);
        checkResult(rp, sp);
        if (rp->err != CUFFT_SUCCESS) {
            res = (cufftResult)rp->err;
        }
    }

    WARN(3, "done.\n");

    return res;
}

cufftResult CUFFTAPI
cufftExecZ2Z(cufftHandle plan, cufftComplex *idata, cufftComplex *odata, int direction)
{
    cufftResult res = CUFFT_SUCCESS;
    rcufftResult *rp;

    initClient();
    WARN(3, "cufftExecZ2Z()...");
    Server *sp = Serverlist;
    for (int i = 0; i < Nredundancy; i++, sp++) {
        rp = rcufftexecz2zid_1((unsigned int)plan, (RCadr)idata, (RCadr)odata, direction, Clnt[0][sp->id]);
        checkResult(rp, sp);
        if (rp->err != CUFFT_SUCCESS) {
            res = (cufftResult)rp->err;
        }
    }

    WARN(3, "done.\n");

    return res;
}

cufftResult CUFFTAPI
cufftExecD2Z(cufftHandle plan, cufftComplex *idata, cufftComplex *odata)
{
    cufftResult res = CUFFT_SUCCESS;
    rcufftResult *rp;

    initClient();
    WARN(3, "cufftExecD2Z()...");
    Server *sp = Serverlist;
    for (int i = 0; i < Nredundancy; i++, sp++) {
        rp = rcufftexecd2zid_1((unsigned int)plan, (RCadr)idata, (RCadr)odata, Clnt[0][sp->id]);
        checkResult(rp, sp);
        if (rp->err != CUFFT_SUCCESS) {
            res = (cufftResult)rp->err;
        }
    }

    WARN(3, "done.\n");

    return res;
}

cufftResult CUFFTAPI
cufftExecZ2D(cufftHandle plan, cufftComplex *idata, cufftComplex *odata)
{
    cufftResult res = CUFFT_SUCCESS;
    rcufftResult *rp;

    initClient();
    WARN(3, "cufftExecZ2D()...");
    Server *sp = Serverlist;
    for (int i = 0; i < Nredundancy; i++, sp++) {
        rp = rcufftexecz2did_1((unsigned int)plan, (RCadr)idata, (RCadr)odata, Clnt[0][sp->id]);
        checkResult(rp, sp);
        if (rp->err != CUFFT_SUCCESS) {
            res = (cufftResult)rp->err;
        }
    }

    WARN(3, "done.\n");

    return res;
}

/*
  cufftResult CUFFTAPI
  cufftSetStream(cufftHandle plan, cudaStream_t stream)
  {
  }
*/

cufftResult CUFFTAPI
cufftSetCompatibilityMode(cufftHandle plan, cufftCompatibility mode)
{
    cufftResult res = CUFFT_SUCCESS;
    rcufftResult *rp;

    initClient();
    WARN(3, "cufftSetCompatibilityMode()...");
    Server *sp = Serverlist;
    for (int i = 0; i < Nredundancy; i++, sp++) {
        rp = rcufftsetcompatibilitymodeid_1((unsigned int)plan, (unsigned int)mode, Clnt[0][sp->id]);
        checkResult(rp, sp);
        if (rp->err != CUFFT_SUCCESS) {
            res = (cufftResult)rp->err;
        }
    }

    WARN(3, "done.\n");

    return res;
}


/*
 * CUBLAS Library functions
 */
cublasStatus_t CUBLASAPI
cublasCreate_v2(cublasHandle_t *handle)
{
    cublasStatus_t res = CUBLAS_STATUS_SUCCESS;
    rcublasCreateResult *rp;

    initClient();
    WARN(3, "cublasCreate()...");
    Server *sp = Serverlist;
    for (int i = 0; i < Nredundancy; i++, sp++) {
        rp = rcublascreate_v2id_1(Clnt[0][sp->id]);
        checkResult(rp, sp);
        if (rp->stat != CUBLAS_STATUS_SUCCESS) {
            res = (cublasStatus_t)rp->stat;
        }
    }
    *handle = (cublasHandle_t)rp->handle;

    WARN(3, "done.\n");

    return res;
}

cublasStatus_t CUBLASAPI
cublasDestroy_v2(cublasHandle_t handle)
{
    cublasStatus_t res = CUBLAS_STATUS_SUCCESS;
    rcublasResult *rp;

    initClient();
    WARN(3, "cublasDestroy()...");
    Server *sp = Serverlist;
    for (int i = 0; i < Nredundancy; i++, sp++) {
        rp = rcublasdestroy_v2id_1((RCadr)handle, Clnt[0][sp->id]);
        checkResult(rp, sp);
        if (rp->stat != CUBLAS_STATUS_SUCCESS) {
            res = (cublasStatus_t)rp->stat;
        }
    }
    WARN(3, "done.\n");

    return res;
}

cublasStatus_t CUBLASAPI
cublasSetVector(int n, int elemSize, const void *x, int incx, void *devicePtr, int incy)
{
    cublasStatus_t res = CUBLAS_STATUS_SUCCESS;
    rcublasResult *rp;

    RCbuf buf;
    buf.RCbuf_val = (char *)malloc(n * elemSize);
    buf.RCbuf_len = n;
    memcpy(buf.RCbuf_val, x, n);

    initClient();
    WARN(3, "cublasSetVector()...");
    Server *sp = Serverlist;
    for (int i = 0; i < Nredundancy; i++, sp++) {
        rp = rcublassetvectorid_1(n, elemSize, buf, incx, (RCadr)devicePtr, incy, Clnt[0][sp->id]);
        checkResult(rp, sp);
        if (rp->stat != CUBLAS_STATUS_SUCCESS) {
            res = (cublasStatus_t)rp->stat;
        }
    }
    WARN(3, "done.\n");

    return res;
}

cublasStatus_t CUBLASAPI
cublasGetVector(int n, int elemSize, const void *x, int incx, void *y, int incy)
{
    cublasStatus_t res = CUBLAS_STATUS_SUCCESS;
    rcublasGetVectorResult *rp;

    initClient();
    WARN(3, "cublasGetVector()...");
    Server *sp = Serverlist;
    for (int i = 0; i < Nredundancy; i++, sp++) {
        rp = rcublasgetvectorid_1(n, elemSize, (RCadr)x, incx, incy, Clnt[0][sp->id]);
        checkResult(rp, sp);
        if (rp->stat != CUBLAS_STATUS_SUCCESS) {
            res = (cublasStatus_t)rp->stat;
        }
    }

    memcpy(y, rp->y.RCbuf_val, n * elemSize);
    WARN(3, "done.\n");

    return res;
}

cublasStatus_t CUBLASAPI
cublasSgemm_v2(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
               const float *alpha, const float *A, int lda,
               const float *B, int ldb, const float *beta, float *C, int ldc)
{
    cublasStatus_t res = CUBLAS_STATUS_SUCCESS;
    rcublasResult *rp;

    initClient();
    WARN(3, "cublasSgemm()...");
    Server *sp = Serverlist;
    for (int i = 0; i < Nredundancy; i++, sp++) {
        rp = rcublassgemm_v2id_1((RCadr)handle, (unsigned int)transa, (unsigned int)transb, m, n, k,
                                 *alpha, (RCadr)A, lda, (RCadr)B, ldb, *beta, (RCadr)C, ldc, Clnt[0][sp->id]);
        checkResult(rp, sp);
        if (rp->stat != CUBLAS_STATUS_SUCCESS) {
            res = (cublasStatus_t)rp->stat;
        }
    }
    WARN(3, "done.\n");

    return res;
}
#endif // CUFFT

#include "libdscuda.cu"
