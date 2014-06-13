static void *ibvMainLoop(void *arg);
static int on_connection_request(struct rdma_cm_id *id);
static int on_connection(struct rdma_cm_id *id);
static int on_disconnection(struct rdma_cm_id *id);
static void *ibvWatchDisconnectionEvent(void *arg);
static int ibvUnpackKernelParam(CUfunction *kfuncp, int narg, IbvArg *args);
static void setupIbv(void);

static int (*IbvStub[RCMethodEnd])(IbvHdr *, IbvHdr *);
static IbvConnection *IbvConn = NULL;

#define SET_IBV_STUB(mthd) {                            \
        IbvStub[RCMethod ## mthd] = ibv ## mthd;        \
}

static void *
ibvMainLoop(void *arg)
{
    uint16_t port = 0;
    struct sockaddr_in addr;
    struct rdma_cm_id *listener = NULL;
    struct rdma_event_channel *ec = NULL;

    while (true) { // for each connection

        TEST_Z(ec = rdma_create_event_channel());
        TEST_NZ(rdma_create_id(ec, &listener, NULL, RDMA_PS_TCP));

        memset(&addr, 0, sizeof(addr));
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = 0;
        addr.sin_port = htons(TcpPort);

        TEST_NZ(rdma_bind_addr(listener, (struct sockaddr *)&addr));
        TEST_NZ(rdma_listen(listener, 10)); // backlog=10 is arbitrary.
        rdmaSetOnCompletionHandler(rdmaOnCompletionServer);

        port = ntohs(rdma_get_src_port(listener));
        WARN(2, "listening on port %d.\n", port);

        rdmaWaitEvent(ec, RDMA_CM_EVENT_CONNECT_REQUEST, on_connection_request);

        // IbvConn->rdma_local/remote_region are now set.
        volatile IbvConnection *conn = IbvConn;
        IbvHdr *spkt = (IbvHdr *)conn->rdma_local_region;
        IbvHdr *rpkt = (IbvHdr *)conn->rdma_remote_region;
        rpkt->method = RCMethodNone;

        rdmaWaitEvent(ec, RDMA_CM_EVENT_ESTABLISHED, on_connection);

        pthread_t tid;
        TEST_NZ(pthread_create(&tid, NULL, ibvWatchDisconnectionEvent, &ec));

        while (conn->connected) {
            int mtd;
            if (!rpkt->method) continue; // wait a packet arrival.
            mtd = rpkt->method;
            rpkt->method = RCMethodNone; // prepare for the next packet arrival.
            int spktsize = (IbvStub[mtd])(rpkt, spkt);
            rdmaKickoff((IbvConnection *)conn, spktsize);
        }

        //        rdmaDestroyConnection(conn);
        rdma_destroy_id(conn->id);
        rdma_destroy_id(listener);
        rdma_destroy_event_channel(ec);
        WARN(0, "disconnected.\n");

        if (D2Csock >= 0) { // this server is spawned by the daemon.
            exit(0);        // exit on disconnection, then.
        }

    } // for each connection

    return NULL;
}

static int
on_connection_request(struct rdma_cm_id *id)
{
    struct rdma_conn_param cm_params;

    WARN(2, "received a connection request.\n");
    rdmaBuildConnection(id, true);
    rdmaBuildParams(&cm_params);
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
ibvWatchDisconnectionEvent(void *arg)
{
    struct rdma_event_channel *ec = *(struct rdma_event_channel **)arg;
    rdmaWaitEvent(ec, RDMA_CM_EVENT_DISCONNECTED, on_disconnection);

    return NULL;
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
            WARN(0, "ibvUnpackKernelParam: invalid RCargType\n", argp->type);
            fatal_error(1);
        }
    }
    return argp->offset + argp->size;
}

/*
 * CUDA API stubs
 */

#define SETUP_IBV_PACKET_BUF(mthd)                            \
        Ibv ## mthd ## InvokeHdr *rpkt = (Ibv ## mthd ## InvokeHdr *)rpkt0; \
        Ibv ## mthd ## ReturnHdr *spkt = (Ibv ## mthd ## ReturnHdr *)spkt0; \
        int spktsize = sizeof(Ibv ## mthd ## ReturnHdr);                \
        spkt->method = RCMethod ## mthd ;                               \
        WARN(3, "cuda" #mthd "(");                                      \
        if (!dscuContext) createDscuContext();

static int
ibvMemcpyH2D(IbvHdr *rpkt0, IbvHdr *spkt0)
{
    cudaError_t err;
    SETUP_IBV_PACKET_BUF(MemcpyH2D);

    err = cudaMemcpy((void*)rpkt->dstadr, &rpkt->srcbuf, rpkt->count, cudaMemcpyHostToDevice);
    check_cuda_error(err);

    spkt->err = err;
    WARN(3, "0x%08llx, 0x%08lx, %d, %s) done.\n",
         rpkt->dstadr, (unsigned long)&rpkt->srcbuf, rpkt->count,
         dscudaMemcpyKindName(cudaMemcpyHostToDevice));

    rdmaWaitReadyToKickoff(IbvConn);
    return spktsize;
}

static int
ibvMemcpyD2H(IbvHdr *rpkt0, IbvHdr *spkt0)
{
    cudaError_t err;
    SETUP_IBV_PACKET_BUF(MemcpyD2H);

    err = cudaMemcpy(&spkt->dstbuf, (void *)rpkt->srcadr, rpkt->count, cudaMemcpyDeviceToHost);
    check_cuda_error(err);
    spkt->err = err;
    WARN(3, "0x%08lx, 0x%08llx, %d, %s) done.\n",
         (unsigned long)&spkt->dstbuf, rpkt->srcadr, rpkt->count,
         dscudaMemcpyKindName(cudaMemcpyDeviceToHost));

    rdmaWaitReadyToKickoff(IbvConn);
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
ibvMemcpyD2D(IbvHdr *rpkt0, IbvHdr *spkt0)
{
    cudaError_t err;
    SETUP_IBV_PACKET_BUF(MemcpyD2D);

    err = cudaMemcpy((void*)rpkt->dstadr, (void*)rpkt->srcadr, rpkt->count, cudaMemcpyDeviceToDevice);
    check_cuda_error(err);

    spkt->err = err;
    WARN(3, "0x%08llx, 0x%08lx, %d, %s) done.\n",
         rpkt->dstadr, rpkt->srcadr, rpkt->count,
         dscudaMemcpyKindName(cudaMemcpyDeviceToDevice));

    rdmaWaitReadyToKickoff(IbvConn);
    return spktsize;
}

static int
ibvMalloc(IbvHdr *rpkt0, IbvHdr *spkt0)
{
    cudaError_t err;
    SETUP_IBV_PACKET_BUF(Malloc);

    err = cudaMalloc((void**)&spkt->devAdr, rpkt->size);
    check_cuda_error(err);
    spkt->err = err;
    WARN(3, "0x%08llx, %d) done. devadr:0x%08llx\n", &spkt->devAdr, rpkt->size, spkt->devAdr);

    rdmaWaitReadyToKickoff(IbvConn);
    return spktsize;
}

static int
ibvFree(IbvHdr *rpkt0, IbvHdr *spkt0)
{
    cudaError_t err;
    SETUP_IBV_PACKET_BUF(Free);

    err = cudaFree((void*)rpkt->devAdr);
    check_cuda_error(err);
    spkt->err = err;
    WARN(3, "0x%08llx) done.\n", rpkt->devAdr);

    rdmaWaitReadyToKickoff(IbvConn);
    return spktsize;
}

static int
ibvGetErrorString(IbvHdr *rpkt0, IbvHdr *spkt0)
{
    SETUP_IBV_PACKET_BUF(GetErrorString);
    int len;
    const char *str;

    str = cudaGetErrorString(rpkt->err);
    strncpy((char *)&spkt->errmsg, str, 256);
    len = strlen((char *)&spkt->errmsg);
    WARN(3, "%d) errmsg:%s  done.\n", rpkt->err, &spkt->errmsg);

    rdmaWaitReadyToKickoff(IbvConn);
    spktsize += len;
    return spktsize;
}

static int
ibvGetDeviceProperties(IbvHdr *rpkt0, IbvHdr *spkt0)
{
    cudaError_t err;
    SETUP_IBV_PACKET_BUF(GetDeviceProperties);

    if (1 < Ndevice) {
        WARN(0, "ibvGetDeviceProperties() cannot handle multiple devices for now. Ndevice:%d\n",
             Ndevice);
        exit(1);
    }
    err = cudaGetDeviceProperties(&spkt->prop, Devid[0]);
    check_cuda_error(err);
    spkt->err = err;
    WARN(3, "0x%08lx, %d) done.\n", (unsigned long)&spkt->prop, rpkt->device);

    rdmaWaitReadyToKickoff(IbvConn);
    return spktsize;
}

static int
ibvRuntimeGetVersion(IbvHdr *rpkt0, IbvHdr *spkt0)
{
    cudaError_t err;
    SETUP_IBV_PACKET_BUF(RuntimeGetVersion);

    err = cudaRuntimeGetVersion(&spkt->version);
    check_cuda_error(err);
    spkt->err = err;
    WARN(3, "0x%08lx) done.\n", (unsigned long)&spkt->version);

    rdmaWaitReadyToKickoff(IbvConn);
    return spktsize;
}

static int
ibvThreadSynchronize(IbvHdr *rpkt0, IbvHdr *spkt0)
{
    cudaError_t err;
    SETUP_IBV_PACKET_BUF(ThreadSynchronize);

    err = cudaThreadSynchronize();
    check_cuda_error(err);
    spkt->err = err;
    WARN(3, "done.\n");

    rdmaWaitReadyToKickoff(IbvConn);
    return spktsize;
}

static int
ibvThreadExit(IbvHdr *rpkt0, IbvHdr *spkt0)
{
    cudaError_t err;
    SETUP_IBV_PACKET_BUF(ThreadExit);

    err = cudaThreadExit();
    check_cuda_error(err);
    spkt->err = err;
    WARN(3, "done.\n");

    rdmaWaitReadyToKickoff(IbvConn);
    return spktsize;
}

static int
ibvDeviceSynchronize(IbvHdr *rpkt0, IbvHdr *spkt0)
{
    cudaError_t err;
    SETUP_IBV_PACKET_BUF(DeviceSynchronize);

    err = cudaDeviceSynchronize();
    check_cuda_error(err);
    spkt->err = err;
    WARN(3, "done.\n");

    rdmaWaitReadyToKickoff(IbvConn);
    return spktsize;
}


static int
ibvDscudaMemcpyToSymbolH2D(IbvHdr *rpkt0, IbvHdr *spkt0)
{
    SETUP_IBV_PACKET_BUF(DscudaMemcpyToSymbolH2D);

    RCbuf srcbuf;
    srcbuf.RCbuf_len = rpkt->count;
    srcbuf.RCbuf_val = (char *)&rpkt->src;
    dscudaResult *resp = dscudamemcpytosymbolh2did_1_svc(rpkt->moduleid,
                                                            rpkt->symbol,
                                                            srcbuf,
                                                            rpkt->count,
                                                            rpkt->offset,
                                                            NULL);
    spkt->err = (cudaError_t)cudaSuccess;

    rdmaWaitReadyToKickoff(IbvConn);
    return spktsize;
}


static int
ibvDscudaMemcpyToSymbolD2D(IbvHdr *rpkt0, IbvHdr *spkt0)
{
    SETUP_IBV_PACKET_BUF(DscudaMemcpyToSymbolD2D);

    dscudaResult *resp = dscudamemcpytosymbold2did_1_svc(rpkt->moduleid,
                                                            rpkt->symbol,
                                                            rpkt->srcadr,
                                                            rpkt->count,
                                                            rpkt->offset,
                                                            NULL);
    spkt->err = (cudaError_t)cudaSuccess;

    rdmaWaitReadyToKickoff(IbvConn);
    return spktsize;
}


static int
ibvDscudaMemcpyFromSymbolD2H(IbvHdr *rpkt0, IbvHdr *spkt0)
{
    SETUP_IBV_PACKET_BUF(DscudaMemcpyFromSymbolD2H);

    dscudaMemcpyFromSymbolD2HResult *resp = dscudamemcpyfromsymbold2hid_1_svc(rpkt->moduleid,
                                                                                      rpkt->symbol,
                                                                                      rpkt->count,
                                                                                      rpkt->offset,
                                                                                      NULL);
    spkt->err = (cudaError_t)cudaSuccess;

    rdmaWaitReadyToKickoff(IbvConn);
    spktsize += rpkt->count;
    memcpy(&spkt->dst, resp->buf.RCbuf_val, rpkt->count);
    return spktsize;
}

static int
ibvDscudaMemcpyFromSymbolD2D(IbvHdr *rpkt0, IbvHdr *spkt0)
{
    SETUP_IBV_PACKET_BUF(DscudaMemcpyFromSymbolD2D);

    dscudaResult *resp = dscudamemcpyfromsymbold2did_1_svc(rpkt->moduleid,
                                                            rpkt->dstadr,
                                                            rpkt->symbol,
                                                            rpkt->count,
                                                            rpkt->offset,
                                                            NULL);
    spkt->err = (cudaError_t)cudaSuccess;

    rdmaWaitReadyToKickoff(IbvConn);
    return spktsize;
}

static int
ibvDscudaMemcpyToSymbolAsyncH2D(IbvHdr *rpkt0, IbvHdr *spkt0)
{
    SETUP_IBV_PACKET_BUF(DscudaMemcpyToSymbolAsyncH2D);

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

    rdmaWaitReadyToKickoff(IbvConn);
    return spktsize;
}


static int
ibvDscudaMemcpyToSymbolAsyncD2D(IbvHdr *rpkt0, IbvHdr *spkt0)
{
    SETUP_IBV_PACKET_BUF(DscudaMemcpyToSymbolAsyncD2D);

    dscudaResult *resp = dscudamemcpytosymbolasyncd2did_1_svc(rpkt->moduleid,
                                                            rpkt->symbol,
                                                            rpkt->srcadr,
                                                            rpkt->count,
                                                            rpkt->offset,
                                                            rpkt->stream,
                                                            NULL);
    spkt->err = (cudaError_t)cudaSuccess;

    rdmaWaitReadyToKickoff(IbvConn);
    return spktsize;
}


static int
ibvDscudaMemcpyFromSymbolAsyncD2H(IbvHdr *rpkt0, IbvHdr *spkt0)
{
    SETUP_IBV_PACKET_BUF(DscudaMemcpyFromSymbolAsyncD2H);

    dscudaMemcpyFromSymbolAsyncD2HResult *resp = dscudamemcpyfromsymbolasyncd2hid_1_svc(rpkt->moduleid,
                                                                                      rpkt->symbol,
                                                                                      rpkt->count,
                                                                                      rpkt->offset,
                                                                                      rpkt->stream,
                                                                                      NULL);
    spkt->err = (cudaError_t)cudaSuccess;

    rdmaWaitReadyToKickoff(IbvConn);
    spktsize += rpkt->count;
    memcpy(&spkt->dst, resp->buf.RCbuf_val, rpkt->count);
    return spktsize;
}

static int
ibvDscudaMemcpyFromSymbolAsyncD2D(IbvHdr *rpkt0, IbvHdr *spkt0)
{
    SETUP_IBV_PACKET_BUF(DscudaMemcpyFromSymbolAsyncD2D);

    dscudaResult *resp = dscudamemcpyfromsymbolasyncd2did_1_svc(rpkt->moduleid,
                                                            rpkt->dstadr,
                                                            rpkt->symbol,
                                                            rpkt->count,
                                                            rpkt->offset,
                                                            rpkt->stream,
                                                            NULL);
    spkt->err = (cudaError_t)cudaSuccess;

    rdmaWaitReadyToKickoff(IbvConn);
    return spktsize;
}

static int
ibvDscudaLoadModule(IbvHdr *rpkt0, IbvHdr *spkt0)
{
    SETUP_IBV_PACKET_BUF(DscudaLoadModule);
    spkt->moduleid = dscudaLoadModule((RCipaddr)rpkt->ipaddr,
                                      (RCpid)rpkt->pid,
                                      rpkt->modulename,
                                      (char *)&rpkt->moduleimage);
    spkt->err = (cudaError_t)cudaSuccess;

    rdmaWaitReadyToKickoff(IbvConn);
    return spktsize;
}

static int
ibvDscudaLaunchKernel(IbvHdr *rpkt0, IbvHdr *spkt0)
{
    SETUP_IBV_PACKET_BUF(DscudaLaunchKernel);
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
    dscudaLaunchKernel(rpkt->moduleid, rpkt->kernelid, rpkt->kernelname,
                       gdim, bdim, rpkt->smemsize, rpkt->stream, args);
    spkt->err = (cudaError_t)cudaSuccess;

    rdmaWaitReadyToKickoff(IbvConn);
    return spktsize;
}
static void
setupIbv(void)
{
    int i;
    memset(IbvStub, 0, sizeof(RCMethod) * RCMethodEnd);
    SET_IBV_STUB(MemcpyH2D);
    SET_IBV_STUB(MemcpyD2H);
    SET_IBV_STUB(MemcpyD2D);
    SET_IBV_STUB(Malloc);
    SET_IBV_STUB(Free);
    SET_IBV_STUB(GetErrorString);
    SET_IBV_STUB(GetDeviceProperties);
    SET_IBV_STUB(RuntimeGetVersion);
    SET_IBV_STUB(ThreadSynchronize);
    SET_IBV_STUB(ThreadExit);
    SET_IBV_STUB(DeviceSynchronize);
    SET_IBV_STUB(DscudaMemcpyToSymbolH2D);
    SET_IBV_STUB(DscudaMemcpyToSymbolD2D);
    SET_IBV_STUB(DscudaMemcpyFromSymbolD2H);
    SET_IBV_STUB(DscudaMemcpyFromSymbolD2D);
    SET_IBV_STUB(DscudaMemcpyToSymbolAsyncH2D);
    SET_IBV_STUB(DscudaMemcpyToSymbolAsyncD2D);
    SET_IBV_STUB(DscudaMemcpyFromSymbolAsyncD2H);
    SET_IBV_STUB(DscudaMemcpyFromSymbolAsyncD2D);
    SET_IBV_STUB(DscudaLoadModule);
    SET_IBV_STUB(DscudaLaunchKernel);
    for (i = 1; i < RCMethodEnd; i++) {
        if (IbvStub[i]) continue;
        WARN(0, "setupIbv: IbvStub[%d] is not initialized.\n", i);
        exit(1);
    }
}
