#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <unistd.h>

/*
 * utils for communication via TCP socket.
 */

static unsigned int
roundUp(unsigned int src, unsigned int by)
{
    unsigned int dst = ((src - 1) / by + 1) * by;
    return dst;
}

struct sockaddr_in
setupSockaddr(unsigned int ipaddr, int tcpport)
{
#if 1

    struct sockaddr_in sockaddr;
    static char buf[128];
    char *p = (char *)&ipaddr;
    
		sprintf(buf, "%hhu.%hhu.%hhu.%hhu", p[0], p[1], p[2], p[3]);

    memset((char *)&sockaddr, 0, sizeof(sockaddr));
    sockaddr.sin_family = AF_INET;
    sockaddr.sin_addr.s_addr=inet_addr(buf);
		sockaddr.sin_port = htons(tcpport);
    return sockaddr;

#else // original implimentation by AK. remove when the code above is tested enough.

    struct sockaddr_in sockaddr;
    struct hostent *hent;

    hent = gethostbyaddr(&ipaddr, sizeof(unsigned int), AF_INET);
    if (!hent) {
        herror("setupSockaddr:gethostbyaddr");
        exit(1);
    }

    memset((char *)&sockaddr, 0, sizeof(sockaddr));
    sockaddr.sin_family = AF_INET;
    memcpy((caddr_t)&sockaddr.sin_addr, hent->h_addr, hent->h_length);
    sockaddr.sin_port = htons(tcpport);
    return sockaddr;

#endif
}

// send/recv binary data of arbitrary size.
void sendDataBySocket(int sock, char *data, int size)
{
    int nleft, nsent, nsent_sum;

    // pack the body and then send.
    nleft = size;
    nsent_sum = 0;
    while (nleft) {
        nsent = send(sock, data + nsent_sum, nleft, 0);
        if (nsent < 0) {
            char msg[1024];
            sprintf(msg, "sendDataBySocket:send() nleft:%d  nsent_sum:%d",
                    nleft, nsent_sum);
            perror(msg);
        }
        nsent_sum += nsent;
        nleft -= nsent;
    }
}

void
recvDataBySocket(int sock, char **datap, int size)
{
    static char *buf = NULL;
    static int bufsize = 0;
    int nleft, nrecvd, nrecvd_sum;

    // reallock the buf.
    if (bufsize < size) {
        bufsize = roundUp(size, 4096);
        buf = (char *)realloc(buf, bufsize);
    }

    // recv the body.
    nleft = size;
    nrecvd_sum = 0;
    while (nleft) {
        nrecvd = recv(sock, buf + nrecvd_sum, nleft, 0);
        if (nrecvd < 0) {
            perror("recvDataBySocket:recv()");
            exit(1);
        }
        nrecvd_sum += nrecvd;
        nleft -= nrecvd;
    }
    if (nrecvd_sum != size) {
        fprintf(stderr, "recvDataBySocket:nrecvd_sum:%d  size:%d\n", nrecvd_sum, size);
        exit(1);
    }

    *datap = buf;
}

// send/recv string of upto a limited size.

void
sendMsgBySocket(int sock, char *msg)
{
    char buf[1024];
    int len = strlen(msg) + 1;

    if (sizeof buf < len) {
        fprintf(stderr, "sendMsgBySocket:message too long.\n");
        exit(1);
    }

    //    fprintf(stderr, "sendMsgBySocket:len:%d msg:%s\n", len, msg);

    *(int *)buf = htonl(len);
    strcpy(buf + sizeof(int), msg);
    send(sock, buf, sizeof(int) + len, 0);
}

void
recvMsgBySocket(int sock, char *msg, int msgbufsize)
{
    char buf[1024];
    int len = strlen(msg) + 1;

    recv(sock, buf, sizeof(int), 0);
    len = ntohl(*(int *)buf);

    //    fprintf(stderr, "recvMsgBySocket:len:%d\n", len);

    if (msgbufsize < len) {
        fprintf(stderr, "recvMsgBySocket:message too long.\n");
        exit(1);
    }
    recv(sock, msg, len, 0);
}
