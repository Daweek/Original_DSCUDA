static void *tcpWatchDisconnection(void *arg);
static int tcpUnpackKernelParam(CUfunction *kfuncp, int narg, RCArg *args);

static void
handleARemoteCall(int sock)
{
    char msg[256];
    int mtd, nleft, nsent, nsent_sum, nrecvd, nrecvd_sum;
    int pktsize;
    RCHdr *rpkt;
    RCHdr *spkt;
    static unsigned char *recvbuf = NULL;
    static unsigned char *sendbuf = NULL;

    // alloc recv buf if not yet.
    if (!recvbuf) {
        recvbuf = (unsigned char *)calloc(RC_SOCKET_BUF_SIZE, 1);
        sendbuf = (unsigned char *)calloc(RC_SOCKET_BUF_SIZE, 1);
        if (!recvbuf || !sendbuf) {
            perror("handleARemoteCall():realloc()");
            exit(2);
        }
    }

    // probe just to know the packet size.

    //    usleep(1000000);
    nrecvd = recv(sock, msg, sizeof(RCHdr), MSG_PEEK);
    if (nrecvd < 0) {
        perror("dscudasvr_tcp:handleARemoteCall:recv");
        exit(1);
    }
    if (nrecvd != sizeof(RCHdr)) {
        fprintf(stderr, "nrecvd:%d  size:%d\n", nrecvd, sizeof(RCHdr));
        exit(1);
    }


    pktsize = ((RCHdr *)msg)->payload;
    //    fprintf(stderr, "pktsize:%d\n", pktsize);
    if (pktsize < 0) {
        fprintf(stderr, "invalid pktsize:%d\n", pktsize);
        exit(1);
    }

    // now actually recv the packet.
    nleft = pktsize;
    nrecvd_sum = 0;
    while (nleft) {
        nrecvd = recv(sock, recvbuf + nrecvd_sum, nleft, 0);
        if (nrecvd < 0) {
            perror("dscudasvr_tcp:handleARemoteCall:recv");
            exit(1);
        }
        nrecvd_sum += nrecvd;
        nleft -= nrecvd;
    }

    if (nrecvd_sum != pktsize) {
        fprintf(stderr, "nrecvd_sum:%d  pktsize:%d\n", nrecvd_sum, pktsize);
        exit(1);
    }

    rpkt = (RCHdr *)recvbuf;
    spkt = (RCHdr *)sendbuf;

    // perform an API call.
    mtd = rpkt->method;
    int spktsize = (RCStub[mtd])(rpkt, spkt);

    // send results back to the client.
    spkt->payload = spktsize;
    nleft = spktsize;
    nsent_sum = 0;
    while (nleft) {
        nsent = send(sock, sendbuf + nsent_sum, nleft, 0);
        if (nsent < 0) {
            sprintf(msg, "handleAremoteCall():send() nleft:%d  nsent_sum:%d",
                    nleft, nsent_sum);
            perror(msg);
        }
        nsent_sum += nsent;
        nleft -= nsent;
    }
}


static void *
tcpMainLoop(void)
{
    pthread_t tid;
    int listening_sock;
    int sock;
    struct sockaddr_in addr;

    while (true) { // for each connection

        listening_sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        WARN(3, "listening_sock:%d\n", listening_sock);
        if (listening_sock == -1) {
            perror("dscudasvr_tcp:tcpMainLoop:socket");
            exit(1);
        }

        memset(&addr, 0, sizeof(addr));
        addr.sin_addr.s_addr = 0;
        addr.sin_family = AF_INET;
        addr.sin_port = htons(TcpPort);
        //    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK); // listen only on 127.0.0.1
        addr.sin_addr.s_addr = htonl(INADDR_ANY);
        if (bind(listening_sock, (struct sockaddr *) &addr, sizeof addr) == -1) {
            perror("dscudasvr_tcp:bind");
            exit(1);
        }

        const int backlog = 10;
        if (listen(listening_sock, backlog) == -1) {
            perror("dscudasvr_tcp:listen");
            exit(1);
        }
        WARN(2, "listening on port %d.\n", TcpPort);

        sock = accept(listening_sock, NULL, NULL);
        if (sock == -1) {
            WARN(0, "accept() error.\n");
            exit(1);
        }

        if (D2Csock >= 0) {
            pthread_create(&tid, NULL, tcpWatchDisconnection, &D2Csock);
        }

        Connected = 1;
        while (Connected) {
            handleARemoteCall(sock);
        }
        close(sock);
        close(listening_sock);
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



/*
 * A thread to watch over the socket inherited from the daemon,
 * in order to detect disconnection by the client.
 * exit() immediately, if detected.
 */
static void *
tcpWatchDisconnection(void *arg)
{
    int clientsock = *(int *)arg;
    int nrecvd;
    char buf[16];

    sleep(3); // wait long enough so that connection is certainly establised.

    WARN(3, "start socket polling:%d.\n", clientsock);
    while (1) {
        // nrecvd = recv(clientsock, buf, 1, MSG_PEEK | MSG_DONTWAIT);
        nrecvd = recv(clientsock, buf, 1, MSG_PEEK);
        if (nrecvd == 0) {
            Connected = 0;
            WARN(2, "disconnected.\n");
            exit(0);
        }
        else if (nrecvd == -1) {
            Connected = 0;
            if (errno == ENOTCONN) {
                WARN(0, "disconnected by peer.\n");
                exit(1);
            }
            else {
                perror("dscudasvr_tcp:tcpWatchDisconnection:");
                exit(1);
            }
        }
        WARN(2, "got %d-byte side-band message from the client.\n", nrecvd);
    }

    return NULL;
}


static int
tcpUnpackKernelParam(CUfunction *kfuncp, int narg, RCArg *args)
{
    CUresult cuerr;
    CUfunction kfunc = *kfuncp;
    RCArg noarg;
    RCArg *argp = &noarg;
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
