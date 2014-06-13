
typedef struct {
    int tcpport;
    char ip[RC_HOSTNAMELEN];
    struct rdma_event_channel *ec;
    IbvConnection *ibvconn;
} P2PConnection;

static void *ibvMainLoop(void *arg);
static int on_addr_resolved(struct rdma_cm_id *id);
static int on_route_resolved(struct rdma_cm_id *id);
static int on_send_connection(struct rdma_cm_id *id);
static int on_connection_request(struct rdma_cm_id *id);
static int on_recv_connection(struct rdma_cm_id *id);
static int on_disconnection(struct rdma_cm_id *id);
static void *ibvWatchDisconnectionEvent(void *arg);
static int ibvUnpackKernelParam(CUfunction *kfuncp, int narg, IbvArg *args);
static void setupIbv(void);

static void accept_p2p_connection(P2PConnection *p2pconn);
static int ibvDscudaSendP2P(IbvHdr *rpkt0, IbvHdr *spkt0);
static int ibvDscudaRecvP2P(IbvHdr *rpkt0, IbvHdr *spkt0);
static P2PConnection *lookfor_p2p_connection(char *ip, int tcpport);
static P2PConnection *lookfor_send_p2p_connection(char *ip, int tcpport);
static P2PConnection *lookfor_recv_p2p_connection(int tcpport);
static void accept_p2p_connection(P2PConnection *p2pconn);
static void recv_p2p_post_process(void);

static int (*IbvStub[RCMethodEnd])(IbvHdr *, IbvHdr *);
static IbvConnection *IbvConn = NULL;
static IbvConnection *ConnTmp = NULL;
static P2PConnection P2Pconn[RC_NP2PMAX];
static void (*PostProcess)(void) = NULL;
static void *PostProcessCtx = NULL;

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

        port = ntohs(rdma_get_src_port(listener));
        WARN(2, "listening on port %d.\n", port);

        rdmaWaitEvent(ec, RDMA_CM_EVENT_CONNECT_REQUEST, on_connection_request);
        // now a connection is set to ConnTmp.

        IbvConn = ConnTmp;

        // IbvConn->rdma_local/remote_region are now set.
        volatile IbvConnection *conn = IbvConn;
        IbvHdr *spkt = (IbvHdr *)conn->rdma_local_region;
        IbvHdr *rpkt = (IbvHdr *)conn->rdma_remote_region;
        rpkt->method = RCMethodNone;

        rdmaWaitEvent(ec, RDMA_CM_EVENT_ESTABLISHED, on_recv_connection);

        pthread_t tid;
        TEST_NZ(pthread_create(&tid, NULL, ibvWatchDisconnectionEvent, &ec));

        while (conn->connected) {
            int mtd;
            if (!rpkt->method) continue; // wait a packet arrival.
            mtd = rpkt->method;
            rpkt->method = RCMethodNone; // prepare for the next packet arrival.
            int spktsize = (IbvStub[mtd])(rpkt, spkt);
            rdmaKickoff((IbvConnection *)conn, spktsize);
            if (PostProcess) {
                (*PostProcess)();
            }
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
on_send_connection(struct rdma_cm_id *id)
{
    WARN(3, "  IB Verb connection established.\n");
    IbvConnection *conn = ((IbvConnection *)id->context);
    conn->connected = 1;

    *(RCMethod *)conn->rdma_remote_region = RCMethodFree;
    // whatever method other than RCMethodNone will do.

    return 0;
}

static int
on_connection_request(struct rdma_cm_id *id)
{
    struct rdma_conn_param cm_params;

    WARN(2, "received a connection request.\n");
    rdmaBuildConnection(id, true);
    rdmaBuildParams(&cm_params);
    TEST_NZ(rdma_accept(id, &cm_params));
    ConnTmp = (IbvConnection *)id->context;

    return 0;
}

static int
on_recv_connection(struct rdma_cm_id *id)
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

    err = cudaMemcpy((void *)rpkt->dstadr, &rpkt->srcbuf, rpkt->count, cudaMemcpyHostToDevice);
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

static int
ibvDscudaSendP2P(IbvHdr *rpkt0, IbvHdr *spkt0)
{
    cudaError_t err;

    // req from the client.
    SETUP_IBV_PACKET_BUF(DscudaSendP2P);

    // P2P communication.
    P2PConnection *p2pconn = lookfor_send_p2p_connection(rpkt->dstip, rpkt->port);
    P2PInvokeHdr *p2pspkt = (P2PInvokeHdr *)p2pconn->ibvconn->rdma_local_region;
    P2PReturnHdr *p2prpkt = (P2PReturnHdr *)p2pconn->ibvconn->rdma_remote_region;

    p2pspkt->dstadr = rpkt->dstadr;
    p2pspkt->count = rpkt->count;
    p2pspkt->size = sizeof(P2PInvokeHdr) + rpkt->count;
    p2prpkt->size = 0;

    if (!dscuContext) createDscuContext();

    err = cudaMemcpy(&p2pspkt->srcbuf, (void *)rpkt->srcadr, rpkt->count, cudaMemcpyDeviceToHost);
    WARN(3, "ibvDscudaSendP2P:cudaMemcpy(0x%08llx, 0x%08lx, %d, %s) done.\n",
         &p2pspkt->srcbuf, rpkt->srcadr, rpkt->count,
         dscudaMemcpyKindName(cudaMemcpyDeviceToHost));
    check_cuda_error(err);

    rdmaKickoff(p2pconn->ibvconn, p2pspkt->size);
    // wait an acknowledge
    while (p2prpkt->size == 0) {
        // nop
    }

    // ack to the client.
    spkt->err = err;
    WARN(3, "0x%08llx, 0x%08lx, %d, %s) done.\n",
         &p2pspkt->srcbuf, rpkt->srcadr, rpkt->count,
         dscudaMemcpyKindName(cudaMemcpyHostToDevice));

    rdmaWaitReadyToKickoff(IbvConn);
    return spktsize;
}

static int
ibvDscudaRecvP2P(IbvHdr *rpkt0, IbvHdr *spkt0)
{
    SETUP_IBV_PACKET_BUF(DscudaRecvP2P);

    P2PConnection *p2pconn = lookfor_recv_p2p_connection(rpkt->port);
    PostProcess = recv_p2p_post_process;
    PostProcessCtx = p2pconn;

    return spktsize;
}

static void
recv_p2p_post_process(void)
{
    cudaError_t err;
    P2PConnection *p2pconn = (P2PConnection *)(PostProcessCtx);
    PostProcess = NULL;
    PostProcessCtx = NULL;

    if (!p2pconn->ibvconn) {
        accept_p2p_connection(p2pconn);
    }
    P2PReturnHdr *spkt = (P2PReturnHdr *)p2pconn->ibvconn->rdma_local_region;
    P2PInvokeHdr *rpkt = (P2PInvokeHdr *)p2pconn->ibvconn->rdma_remote_region;

    if (!dscuContext) createDscuContext();

    // wait a packet arrival.
    while (rpkt->size == 0) {
        // nop
    }
    err = cudaMemcpy((void *)rpkt->dstadr, &rpkt->srcbuf, rpkt->count, cudaMemcpyHostToDevice);
    WARN(3, "recv_p2p_post_process:cudaMemcpy(0x%08llx, 0x%08lx, %d, %s) done.\n",
         rpkt->dstadr, (unsigned long)&rpkt->srcbuf, rpkt->count,
         dscudaMemcpyKindName(cudaMemcpyHostToDevice));
    check_cuda_error(err);
    spkt->err = err;
    spkt->size = sizeof(P2PReturnHdr);

    rdmaWaitReadyToKickoff(IbvConn);
    usleep(100000);
    rdmaKickoff(p2pconn->ibvconn, spkt->size);
}

/*
 * look for p2p connection to IP "ip" and port 'tcpport' and return it.
 * '->ibvconn' of the returned connection is set to NULL,
 * if no connection established yet.
 */
static P2PConnection *
lookfor_p2p_connection(char *ip, int tcpport)
{
    static int firstcall = 1;
    struct sockaddr_in addr;
    int i;
    P2PConnection *conn = NULL;

    // initial clean up.
    if (firstcall) {
        firstcall = 0;
        for (i = 0; i < RC_NP2PMAX; i++) {
            memset((char *)(P2Pconn + i), 0, sizeof(P2PConnection));
        }
    }

    // look for an established connection.
    WARN(4, "look for an established connection for ip:%s port:%d\n", ip, tcpport);
    for (i = 0; i < RC_NP2PMAX; i++) {

#if 0        // !!!
        fprintf(stderr, "P2Pconn[%d].tcpport: %d  tcpport:%d\n",
                i, P2Pconn[i].tcpport, tcpport);
        fprintf(stderr, "P2Pconn[%d].ip: %s  ip:%s\n",
                i, P2Pconn[i].ip, ip);
#endif

        if (P2Pconn[i].tcpport == tcpport) {
            // dont care ip, or match P2Pconn[i].ip
            if (!ip || !strcmp(ip, P2Pconn[i].ip)) {
                WARN(4, "found P2Pconn[%d].ip: %s  .port:%d\n",
                     i, P2Pconn[i].ip, P2Pconn[i].tcpport);
                return P2Pconn + i;
            }
        }
    }

    // no connection found. alloc a slot for a new connection.
    for (i = 0; i < RC_NP2PMAX; i++) {
        if (P2Pconn[i].tcpport == 0) { // unused slot found.
            WARN(4, "requested P2P connecton not established yet. "
                 "alloc P2Pconn[%d] for a new connection.\n", i);
            conn = P2Pconn + i;
            conn->tcpport = tcpport;
            if (ip) {
                sprintf(conn->ip, "%s", ip);
            }
            conn->ibvconn = NULL;
            break;
        }
    }
    if (!conn) {
        fprintf(stderr, "the number of P2P connections exceeds the limit (=%d).\n", RC_NP2PMAX);
        exit(1);
    }

    return conn;
}

static P2PConnection *
lookfor_send_p2p_connection(char *ip, int tcpport)
{
    struct addrinfo *addr;
    struct rdma_cm_id *cmid= NULL;
    struct rdma_event_channel *ec = NULL;
    IbvConnection *ibvconn;
    char *service;

    P2PConnection *conn = lookfor_p2p_connection(ip, tcpport);
    if (conn->ibvconn) { // a connection found.
        return conn;
    }

    WARN(2, "Requesting IB Verb connection to %s port %d...\n", ip, tcpport);
    asprintf(&service, "%d", tcpport);
    TEST_NZ(getaddrinfo(ip, service, NULL, &addr));
    TEST_Z(ec = rdma_create_event_channel());
    TEST_NZ(rdma_create_id(ec, &cmid, NULL, RDMA_PS_TCP));
    //    Cmid[idev][id] = cmid;
    TEST_NZ(rdma_resolve_addr(cmid, NULL, addr->ai_addr, RC_IBV_TIMEOUT));
    freeaddrinfo(addr);
    rdmaWaitEvent(ec, RDMA_CM_EVENT_ADDR_RESOLVED,  on_addr_resolved);
    rdmaWaitEvent(ec, RDMA_CM_EVENT_ROUTE_RESOLVED, on_route_resolved);
    rdmaWaitEvent(ec, RDMA_CM_EVENT_ESTABLISHED,    on_send_connection);
    ibvconn = (IbvConnection *)cmid->context;
    rdmaSendMr(ibvconn);
    conn->ibvconn = ibvconn;
    usleep(100);
    WARN(2, "connection established\n");

    return conn;
}

static P2PConnection *
lookfor_recv_p2p_connection(int tcpport)
{
    struct sockaddr_in addr;
    struct rdma_cm_id *listener = NULL;
    uint16_t port = 0;

    P2PConnection *conn = lookfor_p2p_connection(NULL, tcpport);
    if (conn->ibvconn) { // a connection found.
        return conn;
    }

    // no established connection found. get ready to accept.
    TEST_Z(conn->ec = rdma_create_event_channel());
    TEST_NZ(rdma_create_id(conn->ec, &listener, NULL, RDMA_PS_TCP));

    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = 0;
    addr.sin_port = htons(tcpport);

    TEST_NZ(rdma_bind_addr(listener, (struct sockaddr *)&addr));
    TEST_NZ(rdma_listen(listener, 10)); // backlog=10 is arbitrary.

    port = ntohs(rdma_get_src_port(listener));
    WARN(2, "listening on port %d.\n", port);

    return conn;
}

static void
accept_p2p_connection(P2PConnection *p2pconn)
{

    rdmaWaitEvent(p2pconn->ec, RDMA_CM_EVENT_CONNECT_REQUEST, on_connection_request);
    p2pconn->ibvconn = ConnTmp;

    rdmaWaitEvent(p2pconn->ec, RDMA_CM_EVENT_ESTABLISHED, on_recv_connection);
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
    SET_IBV_STUB(DscudaSendP2P);
    SET_IBV_STUB(DscudaRecvP2P);
    for (i = 1; i < RCMethodEnd; i++) {
        if (IbvStub[i]) continue;
        WARN(0, "setupIbv: IbvStub[%d] is not initialized.\n", i);
        exit(1);
    }
}
