#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <unistd.h>
#include <signal.h>
#include <pthread.h>
#include "dscudadefs.h"
#include "sockutil.h"

#define NBACKLOG 1   // max # of servers can be put into the listening queue.

static int WarnLevel = 2;
#undef WARN
#define WARN(lv, fmt, args...) if (lv <= WarnLevel) \
        fprintf(stderr, "dscudad : " fmt, ## args);

typedef struct Server_t {
    pid_t pid;
    int port;
    int sock;
    struct Server_t *prev;
    struct Server_t *next;
} Server;

static void parseArgv(int argc, char **argv);
static void showUsage(char *command);

static int create_daemon_socket(in_port_t port, int backlog);
static int alloc_server_port(void);
static void spawn_server(int listening_sock);
static void signal_from_child(int sig);

static void register_server(pid_t pid, int port, int sock);
static void unregister_server(pid_t pid);
static Server *server_with_pid(pid_t pid);
static int unused_server_port(void);
static void initEnv(void);
static void* response_to_search(void *arg);
static void dump_server_list(void);

static int Daemonize = 0;
static int Nserver = 0;
static Server *ServerListTop = NULL;
static Server *ServerListTail = NULL;
static char LogFileName[1024] = "dscudad.log";
static char Dscudasvrpath[512]; // where DS-CUDA server built on the client side is copied at runtime.

static int
create_daemon_socket(in_port_t port, int backlog)
{
    struct sockaddr_in me;
    int sock;

    memset((char *)&me, 0, sizeof(me));
    me.sin_family = AF_INET;
    me.sin_addr.s_addr = htonl(INADDR_ANY);
    me.sin_port = htons(port);

    sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (sock < 0) {
        perror("dscudad:socket");
        return -1;
    }

    if (bind(sock, (struct sockaddr *)&me, sizeof(me)) == -1) {
        perror("dscudad:bind");
        return -1;
    }

    if (listen(sock, backlog) == -1) {
        perror("dscudad:listen");
        return -1;
    }
    WARN(3, "socket for port %d successfully setup.\n", port);

    return sock;
}


static void
register_server(pid_t pid, int port, int sock)
{
    WARN(2, "register_server(%d, %d).\n", pid, port);
    Server *svr = (Server *)calloc(sizeof(Server), 1);
    if (!svr) {
        perror("dscudad:register_server");
    }

    svr->pid = pid;
    svr->port = port;
    svr->sock = sock;
    svr->prev = ServerListTail;
    svr->next = NULL;
    if (!ServerListTop) { // svr will be the 1st entry.
        ServerListTop = svr;
    }
    else {
        ServerListTail->next = svr;
    }
    ServerListTail = svr;
    WARN(3, "register_server done.\n");
}

static void
unregister_server(pid_t pid)
{
    WARN(3, "unregister_server(%d).\n", pid);

    Server *svr = server_with_pid(pid);
    if (!svr) {
        WARN(0, "server with pid %d not found. "
             "unregister operation not performed.\n", pid);
        return;
    }

    if (svr->prev) { // reconnect the linked list.
        //        fprintf(stderr, "svr->prev:%d  pid:%d\n", svr->prev, svr->prev->pid);

        svr->prev->next = svr->next;
    }
    else { // svr was the 1st entry.
        ServerListTop = svr->next;
        if (svr->next) {
            svr->next->prev = NULL;
        }
    }
    if (svr->next) {
        svr->next->prev = svr->prev;
    }
    else { // svr was the last entry.
        ServerListTail = svr->prev;
    }
    WARN(2, "unregister_server w/pid:%d done. port:%d sock:%d released.\n",
         svr->pid, svr->port, svr->sock);
    //    close(svr->sock); // !!!
    free(svr);
}


static void
dump_server_list(void)
{
    Server *svr = ServerListTop;
    fprintf(stderr, ">>>> ");
    while (svr) {
        fprintf(stderr, "%d", svr->pid);
        svr = svr->next;
        if (svr) {
            fprintf(stderr, "->");
        }
        else {
            fprintf(stderr, ".");
        }
    }
    fprintf(stderr, "\n");
}

static Server *
server_with_pid(pid_t pid)
{
    Server *svr = ServerListTop;
    while (svr) {
        if (svr->pid == pid) {
            return svr;
        }
        svr = svr->next;
    }
    return NULL; // server with pid not found in the list.
}

static int
unused_server_port(void)
{
    int inuse;
    int p;
    Server *s;

    WARN(3, "unused_server_port().\n");

    for (p = RC_SERVER_IP_PORT; p < RC_SERVER_IP_PORT + RC_NSERVERMAX; p++) {
        inuse = 0;
        for (s = ServerListTop; s; s = s->next) {
            if (p == s->port) {
                inuse = 1;
                break;
            }
        }
        if (!inuse) {
            WARN(3, "unused_server_port: port found:%d\n", p);
            return p;
        }
    }

    WARN(3, "unused_server_port: all ports in use.\n");
    return -1;
}

static void
spawn_server(int listening_sock)
{
    int len, dev, sock, sport;
    pid_t pid;
    char *argv[16];
    char msg[256];
    char portstr[128], devstr[128], sockstr[128], svrname[512];

    WARN(3, "listening request from client...\n");
    sock = accept(listening_sock, NULL, NULL);
    if (sock == -1) {
        WARN(0, "accept() error\n");
        exit(1);
    }
    recvMsgBySocket(sock, msg, sizeof(msg));
    sscanf(msg, "deviceid:%d", &dev); // deviceid to be handled by the server.

    sport = unused_server_port();
    sprintf(msg, "sport:%d", sport); // server port to be connected by the client.
    sendMsgBySocket(sock, msg);

    if (sport < 0) {
        WARN(0, "spawn_server: max possible ports already in use.\n");
        close(sock);
        return;
    }

    Nserver++;
    pid = fork();
    if (pid) { // parent
        signal(SIGCHLD, signal_from_child);
        WARN(3, "spawn a server with sock: %d\n", sock);
        register_server(pid, sport, sock);
        close(sock); // !!!
    }
    else { // child
#if TCP_ONLY
        argv[0] = "dscudasvr_tcp";
#else
        argv[0] = "dscudasvr";
#endif
        sprintf(portstr, "-p%d", sport);
        argv[1] = portstr;
        sprintf(devstr, "-d%d", dev);
        argv[2] = devstr;
        sprintf(sockstr, "-S%d", sock);
        argv[3] = sockstr;
        argv[4] = (char *)NULL;
        sprintf(svrname, "%s/%s", Dscudasvrpath, argv[0]);
        WARN(3, "exec %s %s %s %s\n", svrname, argv[1], argv[2], argv[3]);
        execvp(svrname, (char **)argv);
        perror(svrname);
        WARN(0, "execvp() failed.\n");
        WARN(0, "%s may not exist, or not in the PATH?\n", svrname);
        exit(1);
    }
}

static void
signal_from_child(int sig)
{
    int status;
    int pid = waitpid(-1, &status, WNOHANG);

    switch (pid) {
      case -1:
        WARN(0, "signal_from_child:waitpid failed.\n");
        exit(1);
        break;
      case 0:
        WARN(0, "no child has exited.\n");
        break;
      default:
        WARN(2, "exited a child (pid:%d).\n", pid);

        if (WIFEXITED(status)) {
            WARN(2, "exit status:%d\n", WEXITSTATUS(status));
        }
        else if (WIFSIGNALED(status)) {
            WARN(2, "terminated by signal %d.\n", WTERMSIG(status));
        }
        Nserver--;
        unregister_server(pid);
    }
}

static void*
response_to_search(void *arg)
{
    int sock;
    socklen_t sin_size;
    struct sockaddr_in addr, clt;
    char buf[256];

    sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (sock == -1) {
        perror("response_to_search:socket()");
        return NULL;
    }

    addr.sin_family = AF_INET;
    addr.sin_port = htons(RC_DAEMON_IP_PORT - 1);
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    if (bind(sock, (struct sockaddr *)&addr, sizeof(addr)) == -1) {
        perror("response_to_search:bind()");
        return NULL;
    }

    memset(buf, 0, sizeof(buf));
    while (1) {
        sin_size = (sizeof(struct sockaddr_in));
        recvfrom(sock, buf, sizeof(buf) - 1, 0, (struct sockaddr *)&clt, &sin_size);
        if (strcmp(buf, "DSCUDA_SEARCHSERVER") != 0) continue;

        WARN(2, "received %s message from %s\n", buf, inet_ntoa(clt.sin_addr));
        clt.sin_family = AF_INET;
        clt.sin_port = htons(RC_DAEMON_IP_PORT - 2);
        inet_aton(inet_ntoa(clt.sin_addr), &(clt.sin_addr));
        strcpy(buf, "DSCUDA_SERVERRESPONSE");
        sendto(sock, buf, strlen(buf) + 1, 0, (struct sockaddr *)&clt, sizeof(struct sockaddr));
        memset(buf, 0, sizeof(buf));
    }

    /* statement unreachable
       close(sock);
       return NULL;
    */
}

static void
initEnv(void)
{
    static int firstcall = 1;
    char *env;

    if (!firstcall) return;

    firstcall = 0;

    // DSCUDA_WARNLEVEL
    env = getenv("DSCUDA_WARNLEVEL");
    if (env) {
        int tmp;
        tmp = atoi(strtok(env, " "));
        if (0 <= tmp) {
            WarnLevel = tmp;
        }
        WARN(1, "WarnLevel: %d\n", WarnLevel);
    }

    // DSCUDA_SVRPATH
    env = getenv("DSCUDA_SVRPATH");
    if (!env) {
        fprintf(stderr, "An environment variable 'DSCUDA_SVRPATH' not set.\n");
        exit(1);
    }
    strncpy(Dscudasvrpath, env, sizeof(Dscudasvrpath));
}

static void
showUsage(char *command)
{
    fprintf(stderr,
            "usage: %s [-d]\n"
            "  -d: daemonize.\n",
            command);
}

extern char *optarg;
extern int optind;
static void
parseArgv(int argc, char **argv)
{
    int c;
    char *param = "dl:h";

    while ((c = getopt(argc, argv, param)) != EOF) {
        switch (c) {
          case 'd':
            Daemonize = 1;
            break;
          case 'l':
            strncpy(LogFileName, optarg, sizeof(LogFileName));
            break;
          case 'h':
          default:
            showUsage(argv[0]);
            exit(1);
        }
    }
}

int
main(int argc, char **argv)
{
    int sock, nserver0;
    int errfd;
    pthread_t th;

    pthread_create(&th, NULL, response_to_search, NULL);
    parseArgv(argc, argv);
    if (Daemonize) {
        if (fork()) {
            exit(0);
        }
        else {
            close(2);
            errfd = open(LogFileName, O_RDWR | O_CREAT | O_APPEND, S_IRUSR | S_IWUSR);
            if (errfd < 0) {
                perror("open:");
            }
            close(0);
            close(1);
        }
    }

    initEnv();
    sock = create_daemon_socket(RC_DAEMON_IP_PORT, NBACKLOG);
    if (sock == -1) {
        WARN(0, "create_daemon_socket() failed\n");
        exit(1);
    }
    nserver0 = Nserver;
    while (1) {
        if (Nserver < RC_NSERVERMAX) {
            spawn_server(sock);

            if (nserver0 != Nserver) {
                if (Nserver < RC_NSERVERMAX) {
                    WARN(0, "%d servers active (%d max possible).\n", Nserver, RC_NSERVERMAX);
                }
                else {
                    WARN(0, "%d servers active. reached the limit.\n", Nserver);
                }
            }
        }
        usleep(100); // ??? why
        nserver0 = Nserver;
    }
    WARN(0, "%s: cannot be reached.\n", __FILE__);
    pthread_join(th, NULL);
    exit(1);
}
