#define _LIBDSCUDA_IBV_CU

#include "libdscuda.h"
#include "ibvdefs.h"

#define SETUP_PACKET_BUF(mthd, vid, rid)                            \
    IbvConnection *conn = (IbvConnection *)Cmid[vid][rid]->context;     \
    RC ## mthd ## InvokeHdr *spkt = (RC ## mthd ## InvokeHdr *)conn->rdma_local_region; \
    RC ## mthd ## ReturnHdr *rpkt = (RC ## mthd ## ReturnHdr *)conn->rdma_remote_region; \
    int spktsize = sizeof(RC ## mthd ## InvokeHdr);                    \
    rdmaWaitReadyToKickoff(conn);                                       \
    spkt->method = RCMethod ## mthd;

static int unused_p2p_port(RCServer_t *srcsvr, RCServer_t *dstsvr);
static void setupConnection(int idev, RCServer_t *sp);
static int on_addr_resolved(struct rdma_cm_id *id);
static int on_route_resolved(struct rdma_cm_id *id);
static int on_connection(struct rdma_cm_id *id);
static void perform_remote_call(IbvConnection *conn, RCMethod *methodp, int sendsize, RCMethod mthd);

static void ibvDscudaLaunchKernel(int moduleid, int kid, char *kname,
                                  int *gdim, int *bdim, RCsize smemsize, RCstream stream,
                                  int narg, RCArg *arg, int vdevid, int raidid);

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
    char *service;

    // port number of the server is given by the daemon, or calculated from cid.
    if (UseDaemon) {
        sp->port = requestDaemonForDevice(sp->ip, cid, UseIbv);
    }
    else {
        sp->port = RC_SERVER_IP_PORT + cid;
    }

    WARN(2, "Requesting IB Verb connection to %s:%d (port %d)...\n", dscudaAddrToServerIpStr(sp->ip), cid, sp->port);
    asprintf(&service, "%d", sp->port);
    TEST_NZ(getaddrinfo(dscudaAddrToServerIpStr(sp->ip), service, NULL, &addr));
    TEST_Z(ec = rdma_create_event_channel());
    TEST_NZ(rdma_create_id(ec, &cmid, NULL, RDMA_PS_TCP));
    Cmid[idev][id] = cmid;
    TEST_NZ(rdma_resolve_addr(cmid, NULL, addr->ai_addr, RC_IBV_TIMEOUT));
    freeaddrinfo(addr);
    rdmaWaitEvent(ec, RDMA_CM_EVENT_ADDR_RESOLVED,  on_addr_resolved);
    rdmaWaitEvent(ec, RDMA_CM_EVENT_ROUTE_RESOLVED, on_route_resolved);
    rdmaWaitEvent(ec, RDMA_CM_EVENT_ESTABLISHED,    on_connection);

    static int firstcall = 1;
    if (firstcall) {
        struct sockaddr_in addrin;
        sockaddr *addrp = rdma_get_local_addr(cmid);
        memcpy(&addrin, addrp, sizeof(addrin));
        MyIpaddr = addrin.sin_addr.s_addr;
        WARN(2, "Client IP address : %s\n", dscudaAddrToServerIpStr(MyIpaddr));
    }
    firstcall = 0;

    rdmaSendMr((IbvConnection *)cmid->context);
    usleep(100); // !!
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
        // wait the returning packet for the previous non-blocking remote call.
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

#include "libdscuda.cu"
