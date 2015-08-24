#define _LIBDSCUDA_TCP_CU

#include "libdscuda.h"

#define SETUP_PACKET_BUF(mthd, vid, rid)                            \
    TcpConnection *conn = &TcpConn[vid][rid];     \
    pthread_mutex_lock(&conn->inuse_mutex);                             \
    RC ## mthd ## InvokeHdr *spkt = (RC ## mthd ## InvokeHdr *)conn->sendbuf; \
    RC ## mthd ## ReturnHdr *rpkt = (RC ## mthd ## ReturnHdr *)conn->recvbuf; \
    int spktsize = sizeof(RC ## mthd ## InvokeHdr);                    \
    rpkt->method = RCMethod ## mthd;                                   \
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
    int nleft, nsent, nsent_sum, nrecvd, nrecvd_sum;
    int recvsize;

    ((RCHdr *)conn->sendbuf)->payload = sendsize;
    nleft = sendsize;
    nsent_sum = 0;
    while (nleft) {
        nsent = send(conn->svrsock, conn->sendbuf + nsent_sum, nleft, 0);
        if (nsent < 0) {
            sprintf(msg, "perform_remote_call():send() nleft:%d  nsent_sum:%d",
                    nleft, nsent_sum);
            perror(msg);
        }
        nsent_sum += nsent;
        nleft -= nsent;
    }

    //    fprintf(stderr, "method:%d nsent_sum:%d  sendsize:%d\n",
    //            *methodp, nsent_sum, sendsize);
    if (nsent_sum != sendsize) {
        fprintf(stderr, "nsent_sum:%d  sendsize:%d\n", nsent_sum, sendsize);
        exit(1);
    }

    // probe just to know the packet size.
    nrecvd = recv(conn->svrsock, msg, sizeof(RCHdr), MSG_PEEK);
    if (nrecvd < 0) {
        perror("perform_remote_call:recv");
        exit(1);
    }
    if (nrecvd != sizeof(RCHdr)) {
        fprintf(stderr, "nrecvd:%d  size:%d\n", nrecvd, sizeof(RCHdr));
        exit(1);
    }

    recvsize = ((RCHdr *)msg)->payload;
    nleft = recvsize;
    nrecvd_sum = 0;
    while (nleft) {
        nrecvd = recv(conn->svrsock, conn->recvbuf + nrecvd_sum, nleft, 0);
        if (nrecvd < 0) {
            perror("perform_remote_call:recv");
            exit(1);
        }
        nrecvd_sum += nrecvd;
        nleft -= nrecvd;
    }
    pthread_mutex_unlock(&conn->inuse_mutex);
    if (nrecvd_sum != recvsize) {
        fprintf(stderr, "nrecvd_sum:%d  recvsize:%d\n", nrecvd_sum, recvsize);
        exit(1);
    }
}

static void
setupConnection(int idev, RCServer_t *sp)
{
    int id = sp->id;
    int cid = sp->cid;
    char msg[256];

    struct sockaddr_in sockaddr;
    int ssock;
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
    conn->sendbufsize = RC_SOCKET_BUF_SIZE;
    conn->recvbufsize = RC_SOCKET_BUF_SIZE;
    conn->sendbuf = (char *)calloc(RC_SOCKET_BUF_SIZE, 1);
    conn->recvbuf = (char *)calloc(RC_SOCKET_BUF_SIZE, 1);
    pthread_mutex_init(&conn->inuse_mutex, NULL);

    static int firstcall = 1;
    if (firstcall) {
        struct sockaddr_in addrin;
        socklen_t addrlen = sizeof(addrin);
        getsockname(ssock, (struct sockaddr *)&addrin, &addrlen);
        MyIpaddr = addrin.sin_addr.s_addr;
        WARN(2, "Client IP address : %s\n", dscudaAddrToServerIpStr(MyIpaddr));
    }
    firstcall = 0;
}

#include "libdscuda.cu"
