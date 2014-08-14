#ifndef TCPDEFS_H
#define TCPDEFS_H

typedef struct {
    int svrsock;
    int sendbufsize;
    int recvbufsize;
    char *sendbuf;
    char *recvbuf;
} TcpConnection;

#define RC_SOCKET_BUF_SIZE (1024 * 1024 * 512)

#endif // TCPDEFS_H
