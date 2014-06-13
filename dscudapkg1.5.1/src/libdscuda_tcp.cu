#include "libdscuda.h"

#define SETUP_PACKET_BUF(mthd, vid, rid)                            \
    TcpConnection *conn = &TcpConn[vid][rid];     \
    RC ## mthd ## InvokeHdr *spkt = (RC ## mthd ## InvokeHdr *)conn->sendbuf; \
    RC ## mthd ## ReturnHdr *rpkt = (RC ## mthd ## ReturnHdr *)conn->recvbuf; \
    int spktsize = sizeof(RC ## mthd ## InvokeHdr);                    \
    spkt->method = RCMethod ## mthd;

static const int UseIbv = 0;

int
dscudaRemoteCallType(void)
{
    return RC_REMOTECALL_TYPE_TCP;
}

static void
perform_remote_call(TcpConnection *conn, RCMethod *methodp, int sendsize, RCMethod mthd)
{
    char msg[256];
    int nsent;
    int recvsize;

    ((RCHdr *)conn->sendbuf)->payload = sendsize;
    nsent = send(conn->svrsock, conn->sendbuf, sendsize, 0);
    if (nsent < 0) {
        sprintf(msg, "perform_remote_call():send() sendsize:%d  nsent:%d",
                sendsize, nsent);
        perror(msg);
    }

    // probe just to know the packet size.
    recv(conn->svrsock, msg, sizeof(RCHdr), MSG_PEEK);
    recvsize = ((RCHdr *)msg)->payload;
    recv(conn->svrsock, conn->recvbuf, recvsize, 0);
}

static void
setupConnection(int idev, RCServer_t *sp)
{
    int id = sp->id;
    int cid = sp->cid;
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
    ssock = socket(AF_INET, SOCK_STREAM, 0);
    if (ssock < 0) {
        perror("socket");
        exit(1);
    }
    if (connect(ssock, (struct sockaddr *)&sockaddr, sizeof(sockaddr)) == -1) {
        perror("connect");
        exit(1);
    }
    sprintf(msg, "%s:%d (port %d) ", dscudaAddrToServerIpStr(sp->ip), cid, sport);
    WARN(2, "Established a socket connection to %s...\n", msg);
    TcpConnection *conn = &TcpConn[idev][id];
    conn->svrsock = ssock;
    conn->sendbufsize = RC_RDMA_BUF_SIZE;
    conn->recvbufsize = RC_RDMA_BUF_SIZE;
    conn->sendbuf = (char *)calloc(RC_RDMA_BUF_SIZE, 1);
    conn->recvbuf = (char *)calloc(RC_RDMA_BUF_SIZE, 1);
}

static void
checkResult(void *rp, RCServer_t *sp)
{
    if (rp) return;
}

#include "libdscuda.cu"
