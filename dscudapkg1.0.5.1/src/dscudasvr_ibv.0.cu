static int on_connection_request(struct rdma_cm_id *id);
static int on_connection(struct rdma_cm_id *id);
static int on_disconnection(struct rdma_cm_id *id);
static void *ibvWatchDisconnectionEvent(void *arg);

static void setupIbv(void);
static int ibvUnpackKernelParam(CUfunction *kfuncp, int narg, IbvArg *args);
static void *ibvMainLoop(void *arg);

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

static int (*IbvStub[RCMethodEnd])(IbvHdr *, IbvHdr *);
static IbvConnection *IbvConn = NULL;

#define SET_IBV_STUB(mthd) {                            \
        IbvStub[RCMethod ## mthd] = ibv ## mthd;        \
}

static int
on_connection_request(struct rdma_cm_id *id)
{
    struct rdma_conn_param cm_params;

    WARN(2, "received a connection request.\n");
    rdmaBuildConnection(id);
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

static void
setupIbv(void)
{
    int i;
    memset(IbvStub, 0, sizeof(RCMethod) * RCMethodEnd);
    SET_IBV_STUB(MemcpyH2D);
    SET_IBV_STUB(MemcpyD2H);
    SET_IBV_STUB(Malloc);
    SET_IBV_STUB(Free);
    SET_IBV_STUB(GetErrorString);
    SET_IBV_STUB(GetDeviceProperties);
    SET_IBV_STUB(RuntimeGetVersion);
    SET_IBV_STUB(DeviceSynchronize);
    SET_IBV_STUB(DscudaMemcpyToSymbolAsyncH2D);
    SET_IBV_STUB(DscudaMemcpyToSymbolAsyncD2D);
    SET_IBV_STUB(DscudaMemcpyFromSymbolAsyncD2H);
    SET_IBV_STUB(DscudaMemcpyFromSymbolAsyncD2D);
    SET_IBV_STUB(DscudaLoadModule);
    SET_IBV_STUB(DscudaLaunchKernel);
    for (i = 1; i < RCMethodEnd; i++) {
        if (IbvStub[i]) continue;
        fprintf(stderr, "setupIbv: IbvStub[%d] is not initialized.\n", i);
        exit(1);
    }
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
        addr.sin_port = htons(RC_IBV_IP_PORT_BASE + ServerId);

        TEST_NZ(rdma_bind_addr(listener, (struct sockaddr *)&addr));
        TEST_NZ(rdma_listen(listener, 10)); // backlog=10 is arbitrary.
        rdmaSetOnCompletionHandler(rdmaOnCompletionServer);

        port = ntohs(rdma_get_src_port(listener));
        printf("listening on port %d.\n", port);

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
            if (!rpkt->method) continue; // wait a packet arrival.
            int spktsize = (IbvStub[rpkt->method])(rpkt, spkt);
            rdmaKickoffRdma((IbvConnection *)conn, spktsize);
            rpkt->method = RCMethodNone; // prepare for the next packet arrival.
        }

        //        rdmaDestroyConnection(conn);
        rdma_destroy_id(conn->id);
        rdma_destroy_id(listener);
        rdma_destroy_event_channel(ec);
        WARN(0, "disconnected.\n");

    } // for each connection

    return NULL;
}

/*
 * CUDA API stubs
 */

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

    rdmaWaitReadyToRdma(IbvConn);
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

    rdmaWaitReadyToRdma(IbvConn);
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

    rdmaWaitReadyToRdma(IbvConn);
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

    rdmaWaitReadyToRdma(IbvConn);
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

    rdmaWaitReadyToRdma(IbvConn);
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

    rdmaWaitReadyToRdma(IbvConn);
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

    rdmaWaitReadyToRdma(IbvConn);
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

    rdmaWaitReadyToRdma(IbvConn);
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

    rdmaWaitReadyToRdma(IbvConn);
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

    rdmaWaitReadyToRdma(IbvConn);
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

    rdmaWaitReadyToRdma(IbvConn);
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

    rdmaWaitReadyToRdma(IbvConn);
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
    spkt->moduleid = dscudaLoadModule((RCipaddr)rpkt->ipaddr,
                                      (RCpid)rpkt->pid,
                                      rpkt->modulename,
                                      (char *)&rpkt->moduleimage);
    spkt->err = (cudaError_t)cudaSuccess;

    rdmaWaitReadyToRdma(IbvConn);
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
    //    dscudalaunchkernelid_1_svc(rpkt->moduleid, rpkt->kernelid, rpkt->kernelname,
    //                              gdim, bdim, rpkt->smemsize, rpkt->stream, args, NULL);

    dscudaLaunchKernel(rpkt->moduleid, rpkt->kernelid, rpkt->kernelname,
                       gdim, bdim, rpkt->smemsize, rpkt->stream, args);



    spkt->err = (cudaError_t)cudaSuccess;

    rdmaWaitReadyToRdma(IbvConn);
    spktsize = sizeof(IbvDscudaLaunchKernelReturnHdr);
    spkt->method = RCMethodDscudaLaunchKernel;
    return spktsize;
}
