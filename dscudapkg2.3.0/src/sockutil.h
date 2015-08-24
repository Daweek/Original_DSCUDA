#ifndef _SOCKUTIL_H
#define _SOCKUTIL_H

#ifdef __cplusplus
extern "C" {
#endif

/*
 * utils for communication via TCP socket.
 */

    struct sockaddr_in setupSockaddr(unsigned int ipaddr, int ipport);
    void sendDataBySocket(int sock, char *data, int size);
    void recvDataBySocket(int sock, char **datap, int size);
    void sendMsgBySocket(int sock, char *msg);
    void recvMsgBySocket(int sock, char *msg, int msgbufsize);

#ifdef __cplusplus
}
#endif

#endif // _SOCKUTIL_H
