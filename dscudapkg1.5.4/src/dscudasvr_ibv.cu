typedef struct {
    int tcpport;
    unsigned int ip;
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
static int RCUnpackKernelParam(CUfunction *kfuncp, int narg, RCArg *args);
static void setupIbv(void);

static void accept_p2p_connection(P2PConnection *p2pconn);
static int RCDscudaSendP2P(RCHdr *rpkt0, RCHdr *spkt0);
static int RCDscudaRecvP2P(RCHdr *rpkt0, RCHdr *spkt0);
static P2PConnection *lookfor_p2p_connection(unsigned int ip, int tcpport);
static P2PConnection *lookfor_send_p2p_connection(unsigned int ip, int tcpport);
static P2PConnection *lookfor_recv_p2p_connection(unsigned int ip, int tcpport);
static void accept_p2p_connection(P2PConnection *p2pconn);
static void recv_p2p_post_process(void);

static IbvConnection *IbvConn = NULL;
static IbvConnection *ConnTmp = NULL;
static P2PConnection P2Pconn[RC_NP2PMAX];
static void (*PostProcess)(void) = NULL;
static void *PostProcessCtx = NULL;
struct rdma_cm_id *P2Plistener = NULL;

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
        RCHdr *spkt = (RCHdr *)conn->rdma_local_region;
        RCHdr *rpkt = (RCHdr *)conn->rdma_remote_region;
        rpkt->method = RCMethodNone;

        rdmaWaitEvent(ec, RDMA_CM_EVENT_ESTABLISHED, on_recv_connection);

        pthread_t tid;
        TEST_NZ(pthread_create(&tid, NULL, ibvWatchDisconnectionEvent, &ec));

        while (conn->connected) {
            int mtd;
            if (!rpkt->method) continue; // wait a packet arrival.
            mtd = rpkt->method;
            rpkt->method = RCMethodNone; // prepare for the next packet arrival.
            int spktsize = (RCStub[mtd])(rpkt, spkt);
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
            int off = TcpPort - RC_SERVER_IP_PORT;

            // avoid exit()ing of multiple servers at the same moment.
            // otherwise dscudad may fail to capture singal SIGCHLD,
            // causing the servers become zombies.
            usleep(off * 100000);

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

    usleep(2000); // !!!
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

    WARN(3, "received a connection request.\n");
    rdmaBuildConnection(id, true);
    rdmaBuildParams(&cm_params);
    TEST_NZ(rdma_accept(id, &cm_params));
    ConnTmp = (IbvConnection *)id->context;

    return 0;
}

static int
on_recv_connection(struct rdma_cm_id *id)
{
    WARN(3, "accepted a connection request.\n");
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

#if 0 // not implemented yet

static int
RCDscudaSendP2P(RCHdr *rpkt0, RCHdr *spkt0)
{
    cudaError_t err;

    // req from the client.
    SETUP_PACKET_BUF(DscudaSendP2P);

    // P2P communication.
    P2PConnection *p2pconn = lookfor_send_p2p_connection(rpkt->dstip, rpkt->port);
    usleep(2000); // !!!
    P2PInvokeHdr *p2pspkt = (P2PInvokeHdr *)p2pconn->ibvconn->rdma_local_region;
    P2PReturnHdr *p2prpkt = (P2PReturnHdr *)p2pconn->ibvconn->rdma_remote_region;

    p2pspkt->dstadr = rpkt->dstadr;
    p2pspkt->count = rpkt->count;
    p2pspkt->size = sizeof(P2PInvokeHdr) + rpkt->count;
    p2prpkt->size = 0;

    if (!dscuContext) createDscuContext();


    err = cudaMemcpy(&p2pspkt->srcbuf, (void *)rpkt->srcadr, rpkt->count, cudaMemcpyDeviceToHost);
    WARN(3, "RCDscudaSendP2P:cudaMemcpy(0x%08llx, 0x%08lx, %d, %s) done.\n",
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
RCDscudaRecvP2P(RCHdr *rpkt0, RCHdr *spkt0)
{
    SETUP_PACKET_BUF(DscudaRecvP2P);

    P2PConnection *p2pconn = lookfor_recv_p2p_connection(rpkt->srcip, rpkt->port);
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
    rpkt->size = 0;
    err = cudaMemcpy((void *)rpkt->dstadr, &rpkt->srcbuf, rpkt->count, cudaMemcpyHostToDevice);
    WARN(3, "recv_p2p_post_process:cudaMemcpy(0x%08llx, 0x%08lx, %d, %s) done.\n",
         rpkt->dstadr, (unsigned long)&rpkt->srcbuf, rpkt->count,
         dscudaMemcpyKindName(cudaMemcpyHostToDevice));
    check_cuda_error(err);
    spkt->err = err;
    spkt->size = sizeof(P2PReturnHdr);

    rdmaWaitReadyToKickoff(IbvConn);
    rdmaKickoff(p2pconn->ibvconn, spkt->size);
}

/*
 * look for p2p connection to IP "ip" and port 'tcpport' and return it.
 * '->ibvconn' of the returned connection is set to NULL,
 * if no connection established yet.
 */
static P2PConnection *
lookfor_p2p_connection(unsigned int ip, int tcpport)
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
    WARN(4, "look for an established connection for ip:%s port:%d\n", dscudaAddrToServerIpStr(ip), tcpport);
    for (i = 0; i < RC_NP2PMAX; i++) {

#if 0
        fprintf(stderr, "P2Pconn[%d].tcpport: %d  tcpport:%d\n",
                i, P2Pconn[i].tcpport, tcpport);
        fprintf(stderr, "P2Pconn[%d].ip: %s  ip:%s\n",
                i, dscudaAddrToServerIpStr(P2Pconn[i].ip));
        fprintf(stderr, "ip:%s\n", dscudaAddrToServerIpStr(ip));
#endif

        if (P2Pconn[i].tcpport == tcpport) {
            // dont care ip, or match P2Pconn[i].ip
            if (ip == 0 || ip == P2Pconn[i].ip) {
                WARN(4, "found P2Pconn[%d].ip: %s  .port:%d\n",
                     i, dscudaAddrToServerIpStr(P2Pconn[i].ip), P2Pconn[i].tcpport);
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
                conn->ip = ip;
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
lookfor_send_p2p_connection(unsigned int ip, int tcpport)
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

    WARN(2, "Requesting IB Verb connection to %s port %d (base-%d) ...\n",
         dscudaAddrToServerIpStr(ip), tcpport, RC_SERVER_IP_PORT - 16 - tcpport);
    asprintf(&service, "%d", tcpport);
    TEST_NZ(getaddrinfo(dscudaAddrToServerIpStr(ip), service, NULL, &addr));
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
    WARN(2, "connection established\n");

    return conn;
}

static P2PConnection *
lookfor_recv_p2p_connection(unsigned int ip, int tcpport)
{
    struct sockaddr_in addr;
    //!!!    struct rdma_cm_id *listener = NULL;
    uint16_t port = 0;

    P2PConnection *conn = lookfor_p2p_connection(ip, tcpport);
    if (conn->ibvconn) { // a connection found.
        return conn;
    }

    // no established connection found. get ready to accept.
    TEST_Z(conn->ec = rdma_create_event_channel());
    TEST_NZ(rdma_create_id(conn->ec, &P2Plistener, NULL, RDMA_PS_TCP));


    WARN(3, "bind  ip:%s  port:%d\n", dscudaAddrToServerIpStr(ip), tcpport);
         
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(tcpport);

    TEST_NZ(rdma_bind_addr(P2Plistener, (struct sockaddr *)&addr));
    TEST_NZ(rdma_listen(P2Plistener, 10)); // backlog=10 is arbitrary.

    port = ntohs(rdma_get_src_port(P2Plistener));
    WARN(2, "listening on port %d.\n", port);

    return conn;
}

static void
accept_p2p_connection(P2PConnection *p2pconn)
{

    rdmaWaitEvent(p2pconn->ec, RDMA_CM_EVENT_CONNECT_REQUEST, on_connection_request);
    p2pconn->ibvconn = ConnTmp;

    rdmaWaitEvent(p2pconn->ec, RDMA_CM_EVENT_ESTABLISHED, on_recv_connection);

    if (P2Plistener) {
        rdma_destroy_id(P2Plistener);
        P2Plistener = NULL;
    }

}

#endif // not implemented yet.
