#include "libdscuda.h"
#include "ibv_rdma.h"

#define SETUP_IBV_PACKET_BUF(mthd, vid, rid)                            \
        IbvConnection *conn = (IbvConnection *)Cmid[vid][rid]->context; \
        Ibv ## mthd ## InvokeHdr *spkt = (Ibv ## mthd ## InvokeHdr *)conn->rdma_local_region; \
        Ibv ## mthd ## ReturnHdr *rpkt = (Ibv ## mthd ## ReturnHdr *)conn->rdma_remote_region; \
        int spktsize = sizeof(Ibv ## mthd ## InvokeHdr);                \
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
    rdmaBuildConnection(id);
    TEST_NZ(rdma_resolve_route(id, RC_IBV_TIMEOUT));

    return 0;
}

#if 0

struct rdma_ib_addr {
	union ibv_gid	sgid;
	union ibv_gid	dgid;
	uint16_t	pkey;
};

struct rdma_addr {
	union {
		struct sockaddr		src_addr;
		struct sockaddr_in	src_sin;
		struct sockaddr_in6	src_sin6;
		struct sockaddr_storage src_storage;
	};
	union {
		struct sockaddr		dst_addr;
		struct sockaddr_in	dst_sin;
		struct sockaddr_in6	dst_sin6;
		struct sockaddr_storage dst_storage;
	};
	union {
		struct rdma_ib_addr	ibaddr;
	} addr;
};

struct rdma_route {
	struct rdma_addr	 addr;
	struct ibv_sa_path_rec	*path_rec;
	int			 num_paths;
};

struct rdma_cm_id {
	struct ibv_context	*verbs;
	struct rdma_event_channel *channel;
	void			*context;
	struct ibv_qp		*qp;
	struct rdma_route	 route;
	enum rdma_port_space	 ps;
	uint8_t			 port_num;
	struct rdma_cm_event	*event;
	struct ibv_comp_channel *send_cq_channel;
	struct ibv_cq		*send_cq;
	struct ibv_comp_channel *recv_cq_channel;
	struct ibv_cq		*recv_cq;
	struct ibv_srq		*srq;
	struct ibv_pd		*pd;
	enum ibv_qp_type	qp_type;
};

#endif

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
    ((IbvConnection *)id->context)->connected = 1;
    return 0;
}

static void
perform_remote_call(IbvConnection *conn, RCMethod *methodp, int sendsize, RCMethod mthd)
{
    *methodp = RCMethodNone;
    rdmaWaitReadyToKickoff(conn);
    rdmaKickoff(conn, sendsize);
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

    initClient();
    WARN(3, "cudaGetErrorString()...");
    Vdev_t *vdev = Vdev + Vdevid;
    for (int i = 0; i < vdev->nredundancy; i++) {
        SETUP_IBV_PACKET_BUF(GetErrorString, Vdevid, i);

        // pack send data.
        spkt->err = error;
        WARN(3, "spktsize:%d\n", spktsize);
        WARN(3, "Vdevid:%d\n", Vdevid);

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

    initClient();
    WARN(3, "cudaRuntimeGetVersion(0x%08llx)...", (unsigned long)runtimeVersion);
    Vdev_t *vdev = Vdev + Vdevid;
    for (int i = 0; i < vdev->nredundancy; i++) {
        SETUP_IBV_PACKET_BUF(RuntimeGetVersion, Vdevid, i);
        WARN(3, "spktsize:%d\n", spktsize);
        WARN(3, "Vdevid:%d\n", Vdevid);

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

    initClient();
    WARN(3, "cudaDeviceSynchronize()...");
    Vdev_t *vdev = Vdev + Vdevid;
    for (int i = 0; i < vdev->nredundancy; i++) {
        SETUP_IBV_PACKET_BUF(DeviceSynchronize, Vdevid, i);
        WARN(3, "spktsize:%d\n", spktsize);
        WARN(3, "Vdevid:%d\n", Vdevid);

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

    initClient();
    WARN(3, "cudaMalloc(0x%08llx, %d)...", (unsigned long)devAdrPtr, size);
    Vdev_t *vdev = Vdev + Vdevid;
    for (int i = 0; i < vdev->nredundancy; i++) {
        SETUP_IBV_PACKET_BUF(Malloc, Vdevid, i);
        spkt->size = size;
        WARN(3, "spktsize:%d  size:%d\n", spktsize, size);

        perform_remote_call(conn, &rpkt->method, spktsize, spkt->method);

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

    if (autoVerb) {
        cudaArgs args;
        args.cudaMallocArgs.devPtr = devAdrPtr;
        args.cudaMallocArgs.size = size;

        verbAddHist(dscudaMallocId, args);
    }

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
        SETUP_IBV_PACKET_BUF(Free, Vdevid, i);
        spkt->devAdr = (RCadr)mem;
        WARN(3, "spktsize:%d\n", spktsize);

        perform_remote_call(conn, &rpkt->method, spktsize, spkt->method);

        // unpack returned data.
        err = rpkt->err;
        WARN(3, "cudaFree err : %d\n", err);
    }

    WARN(3, "done.\n");

    if (autoVerb) {
        cudaArgs args;
        args.cudaFreeArgs.devPtr = mem;
        verbAddHist(dscudaFreeId, args);
    }

    return err;
}


cudaError_t
cudaMemcpyToAlldev(int ndev, void **dst, const void *src, size_t count, enum cudaMemcpyKind kind)
{
    cudaError_t err = cudaSuccess;
    Vdev_t *vdev;

    initClient();

    WARN(3, "cudaMemcpyToAlldev(%d, 0x%08lx, 0x%08lx, %d, %s)...",
         ndev, (unsigned long)dst, (unsigned long)src, count, dscudaMemcpyKindName(kind));

    if (RDMA_BUFFER_SIZE < count) {
        WARN(0, "count (=%d) exceeds RDMA_BUFFER_SIZE (=%d).\n",
             count, RDMA_BUFFER_SIZE);
        exit(1);
    }

    if (kind != cudaMemcpyHostToDevice) {
        WARN(0, "a transfer direction other than cudaMemcpyHostToDevice makes no sence.\n");
        exit(1);
    }

    for (int idev = 0; idev < ndev; idev++) {
        vdev = Vdev + idev;
        for (int i = 0; i < vdev->nredundancy; i++) {
            SETUP_IBV_PACKET_BUF(MemcpyH2D, idev, i);

            spktsize += count;
            spkt->count = count;
            spkt->dstadr = (RCadr)dst[idev];
            memcpy(&spkt->srcbuf, src, count);
            WARN(3, "spktsize:%d  count:%d\n", spktsize, count);

            rpkt->method = RCMethodNone;
            rdmaWaitReadyToKickoff(conn);
            rdmaKickoff(conn, spktsize);
#if 1
        }
    }

    for (int idev = 0; idev < ndev; idev++) {
        vdev = Vdev + idev;
        for (int i = 0; i < vdev->nredundancy; i++) {
            IbvConnection *conn = (IbvConnection *)Cmid[idev][i]->context;
            IbvMemcpyH2DReturnHdr *rpkt = (IbvMemcpyH2DReturnHdr *)conn->rdma_remote_region;
#endif
            while (rpkt->method == RCMethodNone) {
                // wait the returning packet.
            }
            // unpack returned data.
            err = rpkt->err;
            WARN(3, "cudaMemcpy H2D err : %d\n", err);
        }
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
            SETUP_IBV_PACKET_BUF(MemcpyD2H, Vdevid, i);
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
                    cudaArgs args;
                    args.cudaMemcpyArgs.dst = dst;
                    args.cudaMemcpyArgs.src = src;
                    args.cudaMemcpyArgs.count = count;
                    args.cudaMemcpyArgs.kind = kind;
                    verbAddHist(dscudaMemcpyH2DId, args);
                    verbRecallHist();
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
        break;
      case cudaMemcpyHostToDevice:
        vdev = Vdev + Vdevid;
        for (int i = 0; i < vdev->nredundancy; i++) {
            SETUP_IBV_PACKET_BUF(MemcpyH2D, Vdevid, i);
            spktsize += count;
            spkt->count = count;
            spkt->dstadr = (RCadr)dst;
            memcpy(&spkt->srcbuf, src, count);
            WARN(3, "spktsize:%d  count:%d\n", spktsize, count);

            perform_remote_call(conn, &rpkt->method, spktsize, spkt->method);

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

    if (autoVerb) {
        cudaArgs args;
        switch (kind) {
          case cudaMemcpyHostToDevice:
            args.cudaMemcpyArgs.dst = dst;
            args.cudaMemcpyArgs.src = src;
            args.cudaMemcpyArgs.count = count;
            args.cudaMemcpyArgs.kind = kind;
            verbAddHist(dscudaMemcpyH2DId, args);
            break;

          case cudaMemcpyDeviceToDevice:
            args.cudaMemcpyArgs.dst = dst;
            args.cudaMemcpyArgs.src = src;
            args.cudaMemcpyArgs.count = count;
            args.cudaMemcpyArgs.kind = kind;
            verbAddHist(dscudaMemcpyD2DId, args);
            break;

          case cudaMemcpyDeviceToHost:
            verbClearHist();
            break;
        }
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

    if (autoVerb) {
        cudaArgs args;
        args.cudaGetDevicePropertiesArgs.prop = prop;
        args.cudaGetDevicePropertiesArgs.device = device;

        verbAddHist(dscudaGetDevicePropertiesId, args);
    }

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

    if (autoVerb) {
        cudaArgs args2;
        memcpy(args2.ibvCudaLaunchKernelArgs.moduleid, moduleid, sizeof(int) * RC_NKMODULEMAX);
        args2.ibvCudaLaunchKernelArgs.kid = kid;
        strncpy(args2.ibvCudaLaunchKernelArgs.kname, kname, RC_KMODULENAMELEN);
        memcpy(args2.ibvCudaLaunchKernelArgs.gdim, gdim, sizeof(int) * 3);
        memcpy(args2.ibvCudaLaunchKernelArgs.bdim, bdim, sizeof(int) * 3);
        args2.ibvCudaLaunchKernelArgs.smemsize = smemsize;
        args2.ibvCudaLaunchKernelArgs.stream = stream;
        args2.ibvCudaLaunchKernelArgs.narg = narg;
        args2.ibvCudaLaunchKernelArgs.arg = arg;
        verbAddHist(dscudaLaunchKernelId, args2);



        {
            ibvCudaLaunchKernelArgsType args = args2.ibvCudaLaunchKernelArgs;
            fprintf(stderr, "########## mid[0]:%d   narg:%d  smemsize:%d\n",
                    args.moduleid[0], args.narg, args.smemsize);
            fprintf(stderr, "########## kid:%d     kname:%s\n", args.kid, args.kname);
            fprintf(stderr, "########## gdim:%d %d %d   bdim:%d %d %d\n",
                    args.gdim[0], args.gdim[1], args.gdim[2],
                    args.bdim[0], args.bdim[1], args.bdim[2]);
            for (int iarg = 0; iarg < args.narg; iarg++) {
                IbvArg *a = args.arg + iarg;
                fprintf(stderr, "############ arg[%d] off:%d  size:%d     ",
                        iarg, a->offset, a->size);
                switch (a->type) {
                  case dscudaArgTypeP:
                    fprintf(stderr, "type:ptr  val:%p\n", a->val.pointerval);
                    break;
                  case dscudaArgTypeI:
                    fprintf(stderr, "type:int  val:%d\n", a->val.intval);
                    break;
                  case dscudaArgTypeF:
                    fprintf(stderr, "type:float val:%f\n", a->val.floatval);
                    break;
                  case dscudaArgTypeV:
                    fprintf(stderr, "type:custom val:%d\n", (int *)a->val.customval);
                    break;
                }
            }
        }
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
