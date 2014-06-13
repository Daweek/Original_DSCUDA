#include <stdio.h>
#include <string.h>
#include <netdb.h>
#include <sys/socket.h>
#include <rpc/rpc.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>
#include "dscuda.h"
#include "ibv_rdma.h"

static const char *DEFAULT_SVRIP = "localhost";
static char Dscudapath[512];

typedef struct {
    int id;   // index for each redundant server.
    int cid;  // id of a server given by -c option to dscudasvr,
              // a num preceded by an IP address & colon, e.g.,
              // export DSCUDA_SERVER="192.168.1.123:2"
    char ip[512];
} RCServer_t;

typedef struct {
    int nredundancy;
    RCServer_t server[RC_NREDUNDANCYMAX];
} Vdev_t;

static int Nvdev;               // # of virtual devices available.
static int Vdevid = 0;          // the virtual device currently in use.
static Vdev_t Vdev[RC_NVDEVMAX];  // a list of virtual devices.
static CLIENT *Clnt[RC_NVDEVMAX][RC_NREDUNDANCYMAX];
static unsigned int MyIpaddr = 0;

static const int UseIbv = 1; // use IB Verbs if set to 1. use RPC, otherwise.
#define DSCUDAAPI(ret, apiname, args...) ret cuda ## apiname(args) \
{\
    if (UseIbv) {\
      return rpcCuda ## apiname(args);\
    }\
    else {\
      return ibvCuda ## apiname(args);\
    }\
}

DSCUDAAPI(cudaError_t, GetLastError, void)
DSCUDAAPI(cudaError_t, PeekAtLastError, void)
DSCUDAAPI(const char *, GetErrorString, cudaError_t error)
DSCUDAAPI(cudaError_t, RuntimeGetVersion, int *runtimeVersion)
DSCUDAAPI(cudaError_t, DeviceSynchronize, void)
DSCUDAAPI(cudaError_t, DeviceReset, void)
DSCUDAAPI(cudaError_t, Malloc, void **devAdrPtr, size_t size)
DSCUDAAPI(cudaError_t, Free, void *mem)
DSCUDAAPI(cudaError_t, Memcpy, void *dst, const void *src, size_t count, enum cudaMemcpyKind kind)
DSCUDAAPI(cudaError_t, GetDeviceProperties, struct cudaDeviceProp *prop, int device)
DSCUDAAPI(cudaError_t, EventCreate, cudaEvent_t *event)
DSCUDAAPI(cudaError_t, EventCreateWithFlags, cudaEvent_t *event, unsigned int flags)
DSCUDAAPI(cudaError_t, EventDestroy, cudaEvent_t event)
DSCUDAAPI(cudaError_t, EventElapsedTime, float *ms, cudaEvent_t start, cudaEvent_t end)
DSCUDAAPI(cudaError_t, EventRecord, cudaEvent_t event, cudaStream_t stream)
DSCUDAAPI(cudaError_t, EventSynchronize, cudaEvent_t event)
DSCUDAAPI(cudaError_t, EventQuery, cudaEvent_t event)

static struct rdma_cm_id *Cmid[RC_NVDEVMAX][RC_NREDUNDANCYMAX]; // for IB Verb connection.


typedef struct {
    int valid;
    int vdevid;  // the virtual device the module is loaded into.
    int id[RC_NREDUNDANCYMAX]; // id assigned to this module in each real device that consists of the virtual one.
    char name[256];
    time_t sent_time;
} Module;

static Module Modulelist[RC_NKMODULEMAX] = {0};

static void initEnv(void);
static void initClient(void);
static void setupSocketConnection(int idev, RCServer_t *sp);
static void setupIbvConnection(int idev, RCServer_t *sp);
static int on_addr_resolved(struct rdma_cm_id *id);
static int on_route_resolved(struct rdma_cm_id *id);
static int on_connection(struct rdma_cm_id *id);
static int ibvDscudaLoadModule(unsigned int ipaddr, pid_t pid, char *modulename, char *modulebuf, int vdevid, int raidid);
static void ibvDscudaLaunchKernel(int moduleid, int kid, char *kname,
                                 int *gdim, int *bdim, RCsize smemsize, RCstream stream,
                                 int narg, IbvArg *arg, int vdevid, int raidid);

typedef struct RCmappedMem_t {
    void *pHost;
    void *pDevice;
    int   size;
    RCmappedMem_t *prev;
    RCmappedMem_t *next;
} RCmappedMem;

static void RCmappedMemRegister(void *pHost, void* pDevice, size_t size);
static void RCmappedMemUnregister(void *pHost);
static RCmappedMem *RCmappedMemQuery(void *pHost);

RCmappedMem *RCmappedMemListTop  = NULL;
RCmappedMem *RCmappedMemListTail = NULL;

typedef struct RCstreamArray_t {
    cudaStream_t s[RC_NREDUNDANCYMAX];
    RCstreamArray_t *prev;
    RCstreamArray_t *next;
} RCstreamArray;

static void RCstreamArrayRegister(cudaStream_t *streams);
static void RCstreamArrayUnregister(cudaStream_t stream0);
static RCstreamArray *RCstreamArrayQuery(cudaStream_t stream0);

RCstreamArray *RCstreamArrayListTop  = NULL;
RCstreamArray *RCstreamArrayListTail = NULL;

typedef struct RCeventArray_t {
    cudaEvent_t e[RC_NREDUNDANCYMAX];
    RCeventArray_t *prev;
    RCeventArray_t *next;
} RCeventArray;

static void RCeventArrayRegister(cudaEvent_t *events);
static void RCeventArrayUnregister(cudaEvent_t event0);
static RCeventArray *RCeventArrayQuery(cudaEvent_t event0);

RCeventArray *RCeventArrayListTop  = NULL;
RCeventArray *RCeventArrayListTail = NULL;

typedef struct RCcuarrayArray_t {
    cudaArray *ap[RC_NREDUNDANCYMAX];
    RCcuarrayArray_t *prev;
    RCcuarrayArray_t *next;
} RCcuarrayArray;

static void RCcuarrayArrayRegister(cudaArray **cuarrays);
static void RCcuarrayArrayUnregister(cudaArray *cuarray0);
static RCcuarrayArray *RCcuarrayArrayQuery(cudaArray *cuarray0);

RCcuarrayArray *RCcuarrayArrayListTop  = NULL;
RCcuarrayArray *RCcuarrayArrayListTail = NULL;


/*
 * private functions:
 */

static void
RCmappedMemRegister(void *pHost, void* pDevice, size_t size)
{
    RCmappedMem *mem = (RCmappedMem *)malloc(sizeof(RCmappedMem));
    if (!mem) {
        perror("RCmappedMemRegister");
    }
    mem->pHost = pHost;
    mem->pDevice = pDevice;
    mem->size = size;
    mem->prev = RCmappedMemListTail;
    mem->next = NULL;
    if (!RCmappedMemListTop) { // mem will be the 1st entry.
        RCmappedMemListTop = mem;
    }
    else {
        RCmappedMemListTail->next = mem;
    }
    RCmappedMemListTail = mem;
}

static void
RCmappedMemUnregister(void *pHost)
{
    RCmappedMem *mem = RCmappedMemQuery(pHost);
    if (!mem) return;

    if (mem->prev) { // reconnect the linked list.
        mem->prev->next = mem->next;
    }
    else { // mem was the 1st entry.
        RCmappedMemListTop = mem->next;
        if (mem->next) {
            mem->next->prev = NULL;
        }
    }
    if (!mem->next) { // mem was the last entry.
        RCmappedMemListTail = mem->prev;
    }
    free(mem);
}

static RCmappedMem *
RCmappedMemQuery(void *pHost)
{
    RCmappedMem *mem = RCmappedMemListTop;
    while (mem) {
        if (mem->pHost == pHost) {
            return mem;
        }
        mem = mem->next;
    }
    return NULL; // pHost not found in the list.
}

static char*
readServerConf(char *fname)
{
    FILE *fp = fopen(fname, "r");
    char linebuf[1024];
    int len;
    static char buf[1024 * 128];

    buf[0] = NULL;
    if (!fp) {
        WARN(0, "cannot open file '%s'\n", fname);
        exit(1);
    }

    while (!feof(fp)) {
        char *s = fgets(linebuf, sizeof(linebuf), fp);
        if (!s) break;
        len = strlen(linebuf);
        if (linebuf[len-1] == '\n') {
            linebuf[len-1] = NULL;
        }
        if (sizeof(buf) < strlen(buf) + len) {
            WARN(0, "readServerConf:file %s too long.\n", fname);
            exit(1);
        }
        strncat(buf, linebuf, sizeof(linebuf));
        strcat(buf, " ");
    }
    fclose(fp);
    return buf;
}

static void
initEnv(void)
{
    static int firstcall = 1;
    int i, ired;
    char *sconfname, *env, *ip, ips[RC_NVDEVMAX][256];
    char buf[8192];
    RCServer_t *sp;
    Vdev_t *vdev;

    if (!firstcall) return;

    firstcall = 0;

    // DSCUDA_WARNLEVEL
    env = getenv("DSCUDA_WARNLEVEL");
    if (env) {
        int tmp;
        tmp = atoi(strtok(env, " "));
        if (0 <= tmp) {
            dscudaSetWarnLevel(tmp);
        }
        WARN(1, "WarnLevel: %d\n", dscudaWarnLevel());
    }

    // DSCUDA_SERVER
    if (sconfname = getenv("DSCUDA_SERVER_CONF")) {
        env = readServerConf(sconfname);
    }
    else {
        env = getenv("DSCUDA_SERVER");
    }
    if (env) {
	strncpy(buf, env, sizeof(buf));
        Nvdev = 0;
        ip = strtok(buf, " "); // a list of IPs which consist a single vdev.
	while (ip) {
            strcpy(ips[Nvdev], ip);
            Nvdev++;
            ip = strtok(NULL, " ");
        }
        for (i = 0; i < Nvdev; i++) {
            int nred = 0;
            vdev = Vdev + i;
            ip = strtok(ips[i], ","); // an IP (optionally with devid preceded by a colon) of
                                      // a single element of the vdev.
            while (ip) {
                strcpy(vdev->server[nred].ip, ip);
                nred++;
                ip = strtok(NULL, ",");
            }
            vdev->nredundancy = nred;

            sp = vdev->server;
            for (ired = 0; ired < nred; ired++, sp++) {
                strncpy(buf, sp->ip, sizeof(buf));
                ip = strtok(buf, ":");
                strcpy(sp->ip, ip);
                ip = strtok(NULL, ":");
                sp->id = ired;
                sp->cid = ip ? atoi(ip) : 0;
            }
        }
    }
    else {
        Nvdev = 1;
        Vdev[0].nredundancy = 1;
        sp = Vdev[0].server;
        sp->id = 0;
        strncpy(sp->ip, DEFAULT_SVRIP, sizeof(sp->ip));
    }
    WARN(3, "DSCUDA Server\n");
    vdev = Vdev;
    for (i = 0; i < Nvdev; i++) {
        WARN(3, "  virtual device%d\n", i);
        sp = vdev->server;
        for (ired = 0; ired < vdev->nredundancy; ired++) {
            WARN(3, "    %s:%d\n", sp->ip, sp->id);
            sp++;
        }
        vdev++;
    }

    // DSCUDA_PATH
    env = getenv("DSCUDA_PATH");
    if (!env) {
        fprintf(stderr, "An environment variable 'DSCUDA_PATH' not set.\n");
        exit(1);
    }
    strncpy(Dscudapath, env, sizeof(Dscudapath));

    // DSCUDA_REMOTECALL
    env = getenv("DSCUDA_REMOTECALL");
    if (!env) {
        fprintf(stderr, "Set an environment variable 'DSCUDA_REMOTECALL' to 'ibv' or 'rpc'.\n");
        exit(1);
    }
    if (!strcmp(env, "ibv")) {
        UseIbv = 1;
        WARN(2, "method of remote procedure call: InfiniBand Verbs\n");
    }
    else {
        UseIbv = 0;
        WARN(2, "method of remote procedure call: RPC\n");
    }
}

static void
initClient(void)
{
    static int firstcall = 1;

    if (!firstcall) return;

    firstcall = 0;
    initEnv();

    for (int i = 0; i < Nvdev; i++) {
        Vdev_t *vdev = Vdev + i;
        RCServer_t *sp = vdev->server;
        for (int ired = 0; ired < vdev->nredundancy; ired++, sp++) {
            if (UseIbv) {
                setupIbvConnection(i, sp);
            }
            else {
                setupSocketConnection(i, sp);
            }
        } // ired
    } // i
    struct sockaddr_in addrin;
    get_myaddress(&addrin);
    MyIpaddr = addrin.sin_addr.s_addr;
    WARN(2, "Client IP address : %s\n", dscudaGetIpaddrString(MyIpaddr));
}

static void
setupSocketConnection(int idev, RCServer_t *sp)
{
    int id = sp->id;
    int cid = sp->cid;
    int portid = DSCUDA_PROG + cid;

    WARN(2, "Requesting socket connection to %s:%d (port %d)...\n", sp->ip, cid, portid);

#if 0
    Clnt[idev][id] = clnt_create(sp->ip, DSCUDA_PROG, DSCUDA_VER, "tcp");

#elif 1 // TCP

    struct sockaddr_in sockaddr;
    struct hostent *hent;
    int sock = RPC_ANYSOCK;

    hent = gethostbyname(sp->ip);
    memset((char *)&sockaddr, 0, sizeof(sockaddr));
    sockaddr.sin_family = AF_INET;
    bcopy(hent->h_addr, (caddr_t)&sockaddr.sin_addr, hent->h_length);
    sockaddr.sin_port = htons((in_port_t)0);

    Clnt[idev][id] = clnttcp_create(&sockaddr,
                                    portid,
                                    DSCUDA_VER,
                                    &sock,
                                    RC_BUFSIZE, RC_BUFSIZE);

#else // UDP

    struct sockaddr_in sockaddr;
    struct hostent *hent;
    int sock = RPC_ANYSOCK;

    hent = gethostbyname(sp->ip);
    memset((char *)&sockaddr, 0, sizeof(sockaddr));
    sockaddr.sin_family = AF_INET;
    bcopy(hent->h_addr, (caddr_t)&sockaddr.sin_addr, hent->h_length);
    sockaddr.sin_port = htons((in_port_t)0);
    struct timeval wait = {
        1.0, // sec
        0.0, // usec
    };

    Clnt[idev][id] = clntudp_create(&sockaddr,
                                    portid,
                                    DSCUDA_VER,
                                    wait,
                                    &sock);

#endif
    if (!Clnt[idev][id]) {
        char buf[256];
        sprintf(buf, "%s:%d (port %d) ", sp->ip, id, portid);
        clnt_pcreateerror(buf);
        if (0 == strcmp(sp->ip, DEFAULT_SVRIP)) {
            WARN(0, "You may need to set an environment variable 'DSCUDA_SERVER'.\n");
        }
        else {
            WARN(0, "DSCUDA server (dscudasrv on %s:%d) may be down.\n", sp->ip, id);
        }
        exit(1);
    }
}

static void
setupIbvConnection(int idev, RCServer_t *sp)
{
    struct addrinfo *addr;
    struct rdma_cm_id *cmid= NULL;
    struct rdma_event_channel *ec = NULL;
    int id = sp->id;
    int cid = sp->cid;
    int portid = RC_IBV_IP_PORT_BASE + cid;
    char *service;

    WARN(2, "Requesting IB Verb connection to %s:%d (port %d)...\n", sp->ip, cid, portid);

    asprintf(&service, "%d", portid);
    TEST_NZ(getaddrinfo(sp->ip, service, NULL, &addr));
    TEST_Z(ec = rdma_create_event_channel());
    TEST_NZ(rdma_create_id(ec, &cmid, NULL, RDMA_PS_TCP));
    Cmid[idev][id] = cmid;
    TEST_NZ(rdma_resolve_addr(cmid, NULL, addr->ai_addr, RC_IBV_TIMEOUT));
    freeaddrinfo(addr);
    wait_event(ec, RDMA_CM_EVENT_ADDR_RESOLVED,  on_addr_resolved);
    wait_event(ec, RDMA_CM_EVENT_ROUTE_RESOLVED, on_route_resolved);
    set_on_completion_handler(on_completion_client);
    wait_event(ec, RDMA_CM_EVENT_ESTABLISHED,    on_connection);
    send_mr((IbvConnection *)cmid->context);
    sleep(1);
    WARN(2, "connection established\n");
}

static int
on_addr_resolved(struct rdma_cm_id *id)
{
    WARN(3, "  IB Verb address resolved.\n");
    build_connection(id);
    TEST_NZ(rdma_resolve_route(id, RC_IBV_TIMEOUT));

    return 0;
}

static int
on_route_resolved(struct rdma_cm_id *id)
{
    struct rdma_conn_param cm_params;

    WARN(3, "  IB Verb route resolved.\n");
    build_params(&cm_params);
    TEST_NZ(rdma_connect(id, &cm_params));

    return 0;
}

static int
on_connection(struct rdma_cm_id *id)
{
    WARN(3, "  IB Verb connection established.\n");
    ((IbvConnection *)id->context)->connected = 1;
    return 0;
}

/*
 * public functions
 */

static void (*errorHandler)(void *arg) = NULL;
static void *errorHandlerArg = NULL;

void
dscudaSetErrorHandler(void (*handler)(void *), void *handler_arg)
{
    errorHandler = handler;
    errorHandlerArg = handler_arg;
}

void
dscudaGetMangledFunctionName(char *name, const char *info, const char *ptxfile)
{
    static char mangler[256] = {0, };
    char cmd[4096];
    FILE *outpipe;

    if (!mangler[0]) {
        sprintf(mangler, "%s/bin/ptx2symbol", Dscudapath);
    }

    WARN(4, "getMangledFunctionName(%08llx, %08llx)  info:\"%s\"\n",
         name, info, info);

    sprintf(cmd, "%s %s << EOF\n%s\nEOF", mangler, ptxfile, info);
    outpipe = popen(cmd, "r");
    if (!outpipe) {
        perror("getMangledFunctionName()");
        exit(1);
    }
    fgets(name, 256, outpipe);
    pclose(outpipe);
    if (!strlen(name)) {
        WARN(0, "getMangledFunctionName() : %s returned an error. "
             "it could not found any entry, or found multiple candidates. "
             "set DSCUDA_WARNLEVEL 4 or higher and try again to see "
             "error messages from %s.\n", mangler, mangler);
        exit(1);
    }
}

static void
invalidateModuleCache(void)
{
#if RC_CACHE_MODULE
    int i;
    Module *mp;

    for (i = 0, mp = Modulelist; i < RC_NKMODULEMAX; i++, mp++) {
        if (!mp->valid) continue;
        mp->valid = 0; // invalidate the cache.
    }
#endif // RC_CACHE_MODULE
}

static void
checkResult(void *rp, RCServer_t *sp)
{
    if (rp) return;
    clnt_perror(Clnt[Vdevid][sp->id], sp->ip);
    exit(1);
}


/*
 * Load a cuda module from a .ptx file,
 * and then, send it to the server.
 * returns id for the module.
 * the id is cached and send once for a while.
 */
int *
dscudaLoadModule(char *modulename)
{
    int i, j, mid;
    Module *mp;

    WARN(5, "dscudaLoadModule(0x%08llx) modulename:%s  ...", modulename, modulename);

#if RC_CACHE_MODULE
    // look for modulename in the module list.
    for (i = 0, mp = Modulelist; i < RC_NKMODULEMAX; i++, mp++) {
        if (!mp->valid) continue;
        if (mp->vdevid != Vdevid) continue;
        if (!strcmp(modulename, mp->name)) {
            if (time(NULL) - mp->sent_time < RC_CLIENT_CACHE_LIFETIME) {
                WARN(5, "done. found a cached one. id:%d  age:%d  name:%s\n",
                     mp->id[i], time(NULL) - mp->sent_time, mp->name);
                return mp->id; // module found. i.e, it's already loaded.
            }
            WARN(5, "done. found a cached one with id:%d, but not used since it is too old.  age:%d\n",
                 mp->id[i], time(NULL) - mp->sent_time);
            mp->valid = 0; // invalidate the cache.
        }
    }
#endif // RC_CACHE_MODULE

    // module not found in the module list.
    // really need to load it from a file.
    FILE *fp;
    char buf[1024];
    int modulebufoff = 0;
    int len;
    static int modulebufsize = 1024;
    static char *modulebuf = (char *)malloc(modulebufsize);

    fp = fopen(modulename, "r");
    if (!fp) {
        char buf[256];
        sprintf(buf, "dscudaLoadModule() : modulename : %s", modulename);
        perror(buf);
        exit(1);
    }
    while (!feof(fp)) {
        fgets(buf, sizeof(buf), fp);
        len = strlen(buf);
        if (modulebufoff + len > modulebufsize) {
            modulebufsize += 1024;
            modulebuf = (char *)realloc(modulebuf, modulebufsize);
        }
        memcpy(modulebuf + modulebufoff, buf, len);
        modulebufoff += len;
    }
    modulebuf[modulebufoff] = 0;
    fclose(fp);

#if 1
    // module loaded into modulebuf[].
    // now we're going to send it to the server.
    Vdev_t *vdev = Vdev + Vdevid;
    RCServer_t *sp = vdev->server;
    for (i = 0; i < vdev->nredundancy; i++, sp++) {
        if (UseIbv) {
            mid = ibvDscudaLoadModule(MyIpaddr, getpid(), modulename, modulebuf, Vdevid, i);
        }
        else {
            dscudaLoadModuleResult *rp = dscudaloadmoduleid_1(MyIpaddr, getpid(), modulename, modulebuf, Clnt[Vdevid][sp->id]);
            checkResult(rp, sp);
            mid = rp->id;
        }

        // register a new module into the list,
        // and then, return a module id assigned by the server.
        if (i == 0) {
            for (j = 0, mp = Modulelist; j < RC_NKMODULEMAX; j++, mp++) {
                if (!mp->valid) break;
                if (j == RC_NKMODULEMAX) {
                    WARN(0, "module send buffer is full.\n");
                    exit(1);
                }
            }
            mp->valid = 1;
            mp->sent_time = time(NULL);
            strncpy(mp->name, modulename, sizeof(mp->name));
            WARN(5, "done. newly registered. id:%d\n", mid);
        }
        mp->id[i] = mid;
    }
    mp->vdevid = Vdevid;
#else
    // module loaded into modulebuf[].
    // now we're going to send it to the server.
    for (int idev = 0; idev < Nvdev; idev++) {
      Vdev_t *vdev = Vdev + idev;
      RCServer_t *sp = vdev->server;
      for (i = 0; i < vdev->nredundancy; i++, sp++) {
          if (UseIbv) {
              mid = ibvDscudaLoadModule(MyIpaddr, getpid(), modulename, modulebuf, idev, i);
          }
          else {
              dscudaLoadModuleResult *rp = dscudaloadmoduleid_1(MyIpaddr, getpid(), modulename, modulebuf, Clnt[idev][sp->id]);
              checkResult(rp, sp);
              mid = rp->id;
          }

        // register a new module into the list,
        // and then, return a module id assigned by the server.
        if (i == 0) {
	  for (j = 0, mp = Modulelist; j < RC_NKMODULEMAX; j++, mp++) {
	    if (!mp->valid) break;
	    if (j == RC_NKMODULEMAX) {
	      WARN(0, "module send buffer is full.\n");
	      exit(1);
	    }
	  }
	  mp->valid = 1;
	  mp->sent_time = time(NULL);
	  strncpy(mp->name, modulename, sizeof(mp->name));
	  WARN(5, "done. newly registered. id:%d\n", rp->id);
        }
        mp->id[i] = rp->id;
      }
    }
#endif
    return mp->id;
}


/*
 * test API for internal use only.
 */

void
dscudaWrite(size_t size, char *dst, char *src)
{
    dscudaResult *rp;
    RCbuf srcbuf;

    initClient();
    srcbuf.RCbuf_len = size;
    srcbuf.RCbuf_val = (char *)src;
    WARN(3, "cudaWrite(%d, 0x%08llx, 0x%08llx)...", size, (unsigned long)dst, (unsigned long)src);

    Vdev_t *vdev = Vdev + Vdevid;
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudawriteid_1((RCsize)size, (RCadr)dst, srcbuf, Clnt[Vdevid][sp->id]);
        checkResult(rp, sp);
    }
    WARN(3, "done.\n");
}

void
dscudaRead(size_t size, char *dst, char *src)
{
    dscudaReadResult *rp;

    initClient();
    WARN(3, "cudaRead(%d, 0x%08llx, 0x%08llx)...", size, (unsigned long)dst, (unsigned long)src);
    Vdev_t *vdev = Vdev + Vdevid;
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudareadid_1((RCsize)size, (RCadr)src, Clnt[Vdevid][sp->id]);
        checkResult(rp, sp);
    }
    memcpy(dst, rp->buf.RCbuf_val, rp->buf.RCbuf_len);
    WARN(3, "done.\n");
}

/*
 * Dscuda client-side counterpart for CUDA runtime API:
 */

/*
 * Thread Management
 */

cudaError_t
cudaThreadExit(void)
{
    cudaError_t err = cudaSuccess;
    dscudaResult *rp;

    initClient();
    WARN(3, "cudaThreadExit()...");
    Vdev_t *vdev = Vdev + Vdevid;
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudathreadexitid_1(Clnt[Vdevid][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
    }
    WARN(3, "done.\n");

    return err;
}

cudaError_t
cudaThreadSynchronize(void)
{
    cudaError_t err = cudaSuccess;
    dscudaResult *rp;

    initClient();
    WARN(3, "cudaThreadSynchronize()...");
    Vdev_t *vdev = Vdev + Vdevid;
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudathreadsynchronizeid_1(Clnt[Vdevid][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
    }
    WARN(3, "done.\n");

    return err;
}

cudaError_t
cudaThreadSetLimit(enum cudaLimit limit, size_t value)
{
    cudaError_t err = cudaSuccess;
    dscudaResult *rp;

    initClient();
    WARN(3, "cudaThreadSetLimit(%d, %d)...", limit, value);
    Vdev_t *vdev = Vdev + Vdevid;
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudathreadsetlimitid_1(limit, value, Clnt[Vdevid][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
    }
    WARN(3, "done.\n");

    return err;
}

cudaError_t
cudaThreadGetLimit(size_t *pValue, enum cudaLimit limit)
{
    cudaError_t err = cudaSuccess;
    dscudaThreadGetLimitResult *rp;

    initClient();
    WARN(3, "cudaThreadGetLimit(0x%08llx, %d)...", pValue, limit);
    Vdev_t *vdev = Vdev + Vdevid;
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudathreadgetlimitid_1(limit, Clnt[Vdevid][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
    }
    *pValue = rp->value;
    err = (cudaError_t)rp->err;
    WARN(3, "done.  *pValue: %d\n", *pValue);

    return err;
}

cudaError_t
cudaThreadSetCacheConfig(enum cudaFuncCache cacheConfig)
{
    cudaError_t err = cudaSuccess;
    dscudaResult *rp;

    initClient();
    WARN(3, "cudaThreadSetCacheConfig(%d)...", cacheConfig);
    Vdev_t *vdev = Vdev + Vdevid;
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudathreadsetcacheconfigid_1(cacheConfig, Clnt[Vdevid][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
    }
    WARN(3, "done.\n");

    return err;
}

cudaError_t
cudaThreadGetCacheConfig(enum cudaFuncCache *pCacheConfig)
{
    cudaError_t err = cudaSuccess;
    dscudaThreadGetCacheConfigResult *rp;

    initClient();
    WARN(3, "cudaThreadGetCacheConfig(0x%08llx)...", pCacheConfig);
    Vdev_t *vdev = Vdev + Vdevid;
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudathreadgetcacheconfigid_1(Clnt[Vdevid][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
    }
    *pCacheConfig = (enum cudaFuncCache)rp->cacheConfig;
    err = (cudaError_t)rp->err;
    WARN(3, "done.  *pCacheConfig: %d\n", *pCacheConfig);

    return err;
}


/*
 * Error Handling
 */

cudaError_t
ibvCudaGetLastError(void)
{
    WARN(2, "a dummy call to cudaGetLastError()\n");
    return cudaSuccess;
}

cudaError_t
rpcCudaGetLastError(void)
{
    cudaError_t err = cudaSuccess;
    dscudaResult *rp;

    initClient();
    WARN(5, "cudaGetLastError()...");
    Vdev_t *vdev = Vdev + Vdevid;
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudagetlasterrorid_1(Clnt[Vdevid][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
    }
    WARN(5, "done.\n");

    return err;
}

cudaError_t
ibvCudaPeekAtLastError(void)
{
    WARN(2, "a dummy call to cudaPeekAtLastError()\n");
    return cudaSuccess;
}

cudaError_t
rpcCudaPeekAtLastError(void)
{
    cudaError_t err = cudaSuccess;
    dscudaResult *rp;

    initClient();
    WARN(5, "cudaPeekAtLastError()...");
    Vdev_t *vdev = Vdev + Vdevid;
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudapeekatlasterrorid_1(Clnt[Vdevid][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
    }
    WARN(5, "done.\n");

    return err;
}

const char *
ibvCudaGetErrorString(cudaError_t error)
{
    static char str[256];

    initClient();
    WARN(3, "cudaGetErrorString()...");
    Vdev_t *vdev = Vdev + Vdevid;
    for (int i = 0; i < vdev->nredundancy; i++) {
        IbvConnection *conn = (IbvConnection *)Cmid[Vdevid][i]->context;
        IbvGetErrorStringInvokeHdr *spkt = (IbvGetErrorStringInvokeHdr *)conn->rdma_local_region;
        IbvGetErrorStringReturnHdr *rpkt = (IbvGetErrorStringReturnHdr *)conn->rdma_remote_region;

        // pack send data.
        int spktsize = sizeof(IbvGetErrorStringInvokeHdr);
        spkt->method = RCMethodGetErrorString;
        spkt->err = error;
        WARN(3, "spktsize:%d\n", spktsize);
        WARN(3, "Vdevid:%d\n", Vdevid);

        // perform remote call.
        rpkt->method = RCMethodNone;
        wait_ready_to_rdma(conn);
        kickoff_rdma(conn, spktsize);
        while (!rpkt->method) {
            // wait the returning packet.
        }

        // unpack returned data.
        WARN(3, "cudaGetErrorString errmsg : %s\n", &rpkt->errmsg);
        strncpy(str, (char *)&rpkt->errmsg, strlen((char *)&rpkt->errmsg) + 1);
    }
    WARN(3, "done.\n");

    return str;
}

const char *
rpcCudaGetErrorString(cudaError_t error)
{
    dscudaGetErrorStringResult *rp;

    initClient();
    WARN(5, "cudaGetErrorString()...");
    Vdev_t *vdev = Vdev + Vdevid;
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudageterrorstringid_1(error, Clnt[Vdevid][sp->id]);
        checkResult(rp, sp);
    }
    WARN(5, "done.\n");

    return rp->errmsg;
}

/*
 * Device Management
 */

cudaError_t
cudaSetDeviceFlags(unsigned int flags)
{
    cudaError_t err = cudaSuccess;
    dscudaResult *rp;

    initClient();
    WARN(3, "cudaSetDeviceFlags()...");
    Vdev_t *vdev = Vdev + Vdevid;
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudasetdeviceflagsid_1(flags, Clnt[Vdevid][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
    }
    invalidateModuleCache();

    WARN(3, "done.\n");

    return err;
}

cudaError_t
cudaGetDevice(int *device)
{
    cudaError_t err = cudaSuccess;

    initClient();
    WARN(3, "cudaGetDevice(0x%08llx)...", (unsigned long)device);
    *device = Vdevid;
    WARN(3, "done.\n");

    return err;
}

cudaError_t
cudaChooseDevice(int *device, const struct cudaDeviceProp *prop)
{
    cudaError_t err = cudaSuccess;

    initClient();
    WARN(3, "cudaChooseDevice(0x%08llx, 0x%08llx)...",
         (unsigned long)device, (unsigned long)prop);
    *device = 0;
    WARN(3, "done.\n");
    WARN(3, "Note : The current implementation always returns device 0.\n");

    return err;
}

cudaError_t
cudaGetDeviceCount(int *count)
{
    cudaError_t err = cudaSuccess;

    initClient();
    *count = Nvdev;
    WARN(3, "cudaGetDeviceCount(0x%08llx)  count:%d ...",
    (unsigned long)count, *count);
    WARN(3, "done.\n");

    return err;
}


cudaError_t
cudaDriverGetVersion (int *driverVersion)
{
    cudaError_t err = cudaSuccess;
    dscudaDriverGetVersionResult *rp;

    initClient();
    WARN(3, "cudaDriverGetVersionCount(0x%08llx)...", (unsigned long)driverVersion);
    Vdev_t *vdev = Vdev + Vdevid;
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudadrivergetversionid_1(Clnt[Vdevid][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
    }

    *driverVersion = rp->ver;
    WARN(3, "done.\n");

    return err;
}

cudaError_t
ibvCudaRuntimeGetVersion(int *runtimeVersion)
{
    cudaError_t err = cudaSuccess;

    initClient();
    WARN(3, "cudaRuntimeGetVersion(0x%08llx)...", (unsigned long)runtimeVersion);
    Vdev_t *vdev = Vdev + Vdevid;
    for (int i = 0; i < vdev->nredundancy; i++) {
        IbvConnection *conn = (IbvConnection *)Cmid[Vdevid][i]->context;
        IbvRuntimeGetVersionInvokeHdr *spkt = (IbvRuntimeGetVersionInvokeHdr *)conn->rdma_local_region;
        IbvRuntimeGetVersionReturnHdr *rpkt = (IbvRuntimeGetVersionReturnHdr *)conn->rdma_remote_region;

        // pack send data.
        int spktsize = sizeof(IbvRuntimeGetVersionInvokeHdr);
        spkt->method = RCMethodRuntimeGetVersion;
        WARN(3, "spktsize:%d\n", spktsize);
        WARN(3, "Vdevid:%d\n", Vdevid);

        // perform remote call.
        rpkt->method = RCMethodNone;
        wait_ready_to_rdma(conn);
        kickoff_rdma(conn, spktsize);
        while (!rpkt->method) {
            // wait the returning packet.
        }

        // unpack returned data.
        err = rpkt->err;
        WARN(3, "cudaRuntimeGetVersion err : %d\n", err);
        *runtimeVersion = rpkt->version;
    }
    WARN(3, "done.\n");

    return err;
}

cudaError_t
rpcCudaRuntimeGetVersion(int *runtimeVersion)
{
    cudaError_t err = cudaSuccess;
    dscudaRuntimeGetVersionResult *rp;

    initClient();
    WARN(3, "cudaRuntimeGetVersion(0x%08llx)...", (unsigned long)runtimeVersion);
    Vdev_t *vdev = Vdev + Vdevid;
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudaruntimegetversionid_1(Clnt[Vdevid][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
    }

    *runtimeVersion = rp->ver;
    WARN(3, "done.\n");

    return err;
}

cudaError_t
cudaSetDevice(int device)
{
    cudaError_t err = cudaSuccess;

    initClient();
    WARN(3, "cudaSetDevice(%d)...", device);
    Vdevid = device;
    WARN(3, "done.\n");

    return err;
}

cudaError_t
ibvCudaDeviceSynchronize(void)
{
    cudaError_t err = cudaSuccess;

    initClient();
    WARN(3, "cudaDeviceSynchronize()...");
    Vdev_t *vdev = Vdev + Vdevid;
    for (int i = 0; i < vdev->nredundancy; i++) {
        IbvConnection *conn = (IbvConnection *)Cmid[Vdevid][i]->context;
        IbvDeviceSynchronizeInvokeHdr *spkt = (IbvDeviceSynchronizeInvokeHdr *)conn->rdma_local_region;
        IbvDeviceSynchronizeReturnHdr *rpkt = (IbvDeviceSynchronizeReturnHdr *)conn->rdma_remote_region;

        // pack send data.
        int spktsize = sizeof(IbvDeviceSynchronizeInvokeHdr);
        spkt->method = RCMethodDeviceSynchronize;
        WARN(3, "spktsize:%d\n", spktsize);
        WARN(3, "Vdevid:%d\n", Vdevid);

        // perform remote call.
        rpkt->method = RCMethodNone;
        wait_ready_to_rdma(conn);
        kickoff_rdma(conn, spktsize);
        while (!rpkt->method) {
            // wait the returning packet.
        }

        // unpack returned data.
        err = rpkt->err;
        WARN(3, "cudaDeviceSynchronize err : %d\n", err);
    }
    WARN(3, "done.\n");

    return err;
}

cudaError_t
rpcCudaDeviceSynchronize(void)
{
    cudaError_t err = cudaSuccess;
    dscudaResult *rp;

    initClient();
    WARN(3, "cudaDeviceSynchronize()...");
    Vdev_t *vdev = Vdev + Vdevid;
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudadevicesynchronize_1(Clnt[Vdevid][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
    }
    WARN(3, "done.\n");

    return err;
}

cudaError_t
ibvCudaDeviceReset(void)
{
    WARN(3, "a dummy call to cudaDeviceReset()\n");
    return cudaSuccess;
}

cudaError_t
rpcCudaDeviceReset(void)
{
    cudaError_t err = cudaSuccess;
    dscudaResult *rp;

    initClient();
    WARN(3, "cudaDeviceReset()...");
    Vdev_t *vdev = Vdev + Vdevid;
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudadevicereset_1(Clnt[Vdevid][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
    }
    WARN(3, "done.\n");

    return err;
}

/*
 * Execution Control
 */

cudaError_t
cudaFuncSetCacheConfig(const char * func, enum cudaFuncCache cacheConfig)
{
    cudaError_t err = cudaSuccess;
    WARN(1, "Current implementation of cudaFuncSetCacheConfig() does nothing "
         "but returning cudaSuccess.\n");
    err = cudaSuccess;
    return err;
}

cudaError_t
dscudaFuncGetAttributesWrapper(int *moduleid, struct cudaFuncAttributes *attr, const char *func)
{
    cudaError_t err = cudaSuccess;
    dscudaFuncGetAttributesResult *rp;

    initClient();
    WARN(3, "dscudaFuncGetAttributesWrapper(%d, 0x%08llx, %s)...",
         moduleid, (unsigned long)attr, func);
    Vdev_t *vdev = Vdev + Vdevid;
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudafuncgetattributesid_1(moduleid[i], (char*)func, Clnt[Vdevid][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
    }

    attr->binaryVersion      = rp->attr.binaryVersion;
    attr->constSizeBytes     = rp->attr.constSizeBytes;
    attr->localSizeBytes     = rp->attr.localSizeBytes;
    attr->maxThreadsPerBlock = rp->attr.maxThreadsPerBlock;
    attr->numRegs            = rp->attr.numRegs;
    attr->ptxVersion         = rp->attr.ptxVersion;
    attr->sharedSizeBytes    = rp->attr.sharedSizeBytes;
    WARN(3, "done.\n");
    WARN(3, "  attr->binaryVersion: %d\n", attr->binaryVersion);
    WARN(3, "  attr->constSizeBytes: %d\n", attr->constSizeBytes);
    WARN(3, "  attr->localSizeBytes: %d\n", attr->localSizeBytes);
    WARN(3, "  attr->maxThreadsPerBlock: %d\n", attr->maxThreadsPerBlock);
    WARN(3, "  attr->numRegs: %d\n", attr->numRegs);
    WARN(3, "  attr->ptxVersion: %d\n", attr->ptxVersion);
    WARN(3, "  attr->sharedSizeBytes: %d\n", attr->sharedSizeBytes);

    return err;
}

/*
 * Memory Management
 */
cudaError_t
ibvCudaMalloc(void **devAdrPtr, size_t size)
{
    cudaError_t err = cudaSuccess;
    void *devadr;

    initClient();
    WARN(3, "cudaMalloc(0x%08llx, %d)...", (unsigned long)devAdrPtr, size);
    Vdev_t *vdev = Vdev + Vdevid;
    for (int i = 0; i < vdev->nredundancy; i++) {
        IbvConnection *conn = (IbvConnection *)Cmid[Vdevid][i]->context;
        IbvMallocInvokeHdr *spkt = (IbvMallocInvokeHdr *)conn->rdma_local_region;
        IbvMallocReturnHdr *rpkt = (IbvMallocReturnHdr *)conn->rdma_remote_region;

        // pack send data.
        int spktsize = sizeof(IbvMallocInvokeHdr);
        spkt->method = RCMethodMalloc;
        spkt->size = size;
        WARN(3, "spktsize:%d  size:%d\n", spktsize, size);

        // perform remote call.
        rpkt->method = RCMethodNone;
        wait_ready_to_rdma(conn);
        kickoff_rdma(conn, spktsize);
        while (!rpkt->method) {
            // wait the returning packet.
        }

        // unpack returned data.
        err = rpkt->err;
        WARN(3, "cudaMalloc err : %d\n", err);

        if (i == 0) {
            devadr = (void*)rpkt->devAdr;
        }
        else if (devadr != (void*)rpkt->devAdr) {
            WARN(0, "cudaMalloc() on device%d allocated memory address 0x%lx "
                 "different from that of device0, 0x%lx.\n", i, (void*)rpkt->devAdr, devadr);
            exit(1);
        }
    }
    *devAdrPtr = devadr;
    WARN(3, "done. *devAdrPtr:0x%08llx\n", *devAdrPtr);


    return err;
}

cudaError_t
rpcCudaMalloc(void **devAdrPtr, size_t size)
{
    cudaError_t err = cudaSuccess;
    dscudaMallocResult *rp;

    initClient();
    WARN(3, "cudaMalloc(0x%08llx, %d)...", (unsigned long)devAdrPtr, size);
    Vdev_t *vdev = Vdev + Vdevid;
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudamallocid_1(size, Clnt[Vdevid][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
    }

    *devAdrPtr = (void*)rp->devAdr;
    WARN(3, "done. *devAdrPtr:0x%08llx\n", *devAdrPtr);

    return err;
}


cudaError_t
ibvCudaFree(void *mem)
{
    cudaError_t err = cudaSuccess;

    initClient();
    WARN(3, "cudaFree(0x%08llx)...", (unsigned long)mem);
    Vdev_t *vdev = Vdev + Vdevid;
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        IbvConnection *conn = (IbvConnection *)Cmid[Vdevid][i]->context;
        IbvFreeInvokeHdr *spkt = (IbvFreeInvokeHdr *)conn->rdma_local_region;
        IbvFreeReturnHdr *rpkt = (IbvFreeReturnHdr *)conn->rdma_remote_region;

        // pack send data.
        int spktsize = sizeof(IbvFreeInvokeHdr);
        spkt->method = RCMethodFree;
        spkt->devAdr = (RCadr)mem;
        WARN(3, "spktsize:%d\n", spktsize);

        // perform remote call.
        rpkt->method = RCMethodNone;
        wait_ready_to_rdma(conn);
        kickoff_rdma(conn, spktsize);
        while (!rpkt->method) {
            // wait the returning packet.
        }

        // unpack returned data.
        err = rpkt->err;
        WARN(3, "cudaFree err : %d\n", err);
    }

    WARN(3, "done.\n");
    return err;
}

cudaError_t
rpcCudaFree(void *mem)
{
    cudaError_t err = cudaSuccess;
    dscudaResult *rp;

    initClient();
    WARN(3, "cudaFree(0x%08llx)...", (unsigned long)mem);
    Vdev_t *vdev = Vdev + Vdevid;
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudafreeid_1((RCadr)mem, Clnt[Vdevid][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
    }

    WARN(3, "done.\n");
    return err;
}

cudaError_t
ibvCudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind)
{
    cudaError_t err = cudaSuccess;
    Vdev_t *vdev;

    initClient();

    WARN(3, "cudaMemcpy(0x%08lx, 0x%08lx, %d, %s)...",
            (unsigned long)dst, (unsigned long)src, count, dscudaMemcpyKindName(kind));

    if (RDMA_BUFFER_SIZE < count) {
        WARN(0, "count (=%d) exceeds RDMA_BUFFER_SIZE (=%d).\n",
             count, RDMA_BUFFER_SIZE);
        exit(1);
    }


    switch (kind) {
      case cudaMemcpyDeviceToHost:
        vdev = Vdev + Vdevid;
        for (int i = 0; i < vdev->nredundancy; i++) {
            IbvConnection *conn = (IbvConnection *)Cmid[Vdevid][i]->context;
            IbvMemcpyD2HInvokeHdr *spkt = (IbvMemcpyD2HInvokeHdr *)conn->rdma_local_region;
            IbvMemcpyD2HReturnHdr *rpkt = (IbvMemcpyD2HReturnHdr *)conn->rdma_remote_region;

            // pack send data.
            int spktsize = sizeof(IbvMemcpyD2HInvokeHdr);
            spkt->method = RCMethodMemcpyD2H;
            spkt->count = count;
            spkt->srcadr = (RCadr)src;
            WARN(3, "spktsize:%d  count:%d\n", spktsize, count);

            // perform remote call.
            rpkt->method = RCMethodNone;
            wait_ready_to_rdma(conn);
            kickoff_rdma(conn, spktsize);
            while (!rpkt->method) {
                // wait the returning packet.
            }

            // unpack returned data.
            err = rpkt->err;
            WARN(3, "cudaMemcpy D2H err : %d\n", err);

            if (i == 0) {
                memcpy(dst, &rpkt->dstbuf, count);
            }
            else if (bcmp(dst, &rpkt->dstbuf, count) != 0) {
                if (errorHandler) {
                    errorHandler(errorHandlerArg);
                }
            }
            else {
                WARN(3, "cudaMemcpy() data copied from device%d matched with that from device0.\n", i);
            }
        }
        break;
      case cudaMemcpyHostToDevice:
        vdev = Vdev + Vdevid;
        for (int i = 0; i < vdev->nredundancy; i++) {
            IbvConnection *conn = (IbvConnection *)Cmid[Vdevid][i]->context;
            IbvMemcpyH2DInvokeHdr *spkt = (IbvMemcpyH2DInvokeHdr *)conn->rdma_local_region;
            IbvMemcpyH2DReturnHdr *rpkt = (IbvMemcpyH2DReturnHdr *)conn->rdma_remote_region;

            // pack send data.
            int spktsize = sizeof(IbvMemcpyH2DInvokeHdr) + count;
            spkt->method = RCMethodMemcpyH2D;
            spkt->count = count;
            spkt->dstadr = (RCadr)dst;
            memcpy(&spkt->srcbuf, src, count);
            WARN(3, "spktsize:%d  count:%d\n", spktsize, count);

            // perform remote call.
            rpkt->method = RCMethodNone;
            wait_ready_to_rdma(conn);
            kickoff_rdma(conn, spktsize);
            while (!rpkt->method) {
                // wait the returning packet.
            }

            // unpack returned data.
            err = rpkt->err;
            WARN(3, "cudaMemcpy H2D err : %d\n", err);
        }
        break;
      case cudaMemcpyDeviceToDevice:
        vdev = Vdev + Vdevid;
        for (int i = 0; i < vdev->nredundancy; i++) {
            WARN(0, "Unsupported value for cudaMemcpyKind : %s\n", dscudaMemcpyKindName(kind));
            exit(1);
        }
        break;
      default:
        WARN(0, "Unsupported value for cudaMemcpyKind : %s\n", dscudaMemcpyKindName(kind));
        exit(1);
    }

    WARN(3, "done.\n");

    return err;
}

cudaError_t
rpcCudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind)
{
    cudaError_t err = cudaSuccess;
    dscudaMemcpyD2HResult *d2hrp;
    dscudaResult *h2drp, *d2drp;
    RCbuf srcbuf;
    Vdev_t *vdev;
    RCServer_t *sp;

    initClient();

    WARN(3, "cudaMemcpy(0x%08lx, 0x%08lx, %d, %s)...",
            (unsigned long)dst, (unsigned long)src, count, dscudaMemcpyKindName(kind));

    switch (kind) {
      case cudaMemcpyDeviceToHost:
        vdev = Vdev + Vdevid;
        sp = vdev->server;
        for (int i = 0; i < vdev->nredundancy; i++, sp++) {
            d2hrp = dscudamemcpyd2hid_1((RCadr)src, count, Clnt[Vdevid][sp->id]);
            checkResult(d2hrp, sp);
            err = (cudaError_t)d2hrp->err;
            if (d2hrp->err != cudaSuccess) {
                err = (cudaError_t)d2hrp->err;
            }
            if (i == 0) {
                memcpy(dst, d2hrp->buf.RCbuf_val, d2hrp->buf.RCbuf_len);
            }
            else if (bcmp(dst, d2hrp->buf.RCbuf_val, d2hrp->buf.RCbuf_len) != 0) {
                if (errorHandler) {
                    errorHandler(errorHandlerArg);
                }
            }
            else {
                WARN(3, "cudaMemcpy() data copied from device%d matched with that from device0.\n", i);
            }
        }
        break;
      case cudaMemcpyHostToDevice:
        srcbuf.RCbuf_len = count;
        srcbuf.RCbuf_val = (char *)src;
        vdev = Vdev + Vdevid;
        sp = vdev->server;
        for (int i = 0; i < vdev->nredundancy; i++, sp++) {
            h2drp = dscudamemcpyh2did_1((RCadr)dst, srcbuf, count, Clnt[Vdevid][sp->id]);
            checkResult(h2drp, sp);
            if (h2drp->err != cudaSuccess) {
                err = (cudaError_t)h2drp->err;
            }
        }
        break;
      case cudaMemcpyDeviceToDevice:
        vdev = Vdev + Vdevid;
        sp = vdev->server;
        for (int i = 0; i < vdev->nredundancy; i++, sp++) {
            d2drp = dscudamemcpyd2did_1((RCadr)dst, (RCadr)src, count, Clnt[Vdevid][sp->id]);
            checkResult(d2drp, sp);
            if (d2drp->err != cudaSuccess) {
                err = (cudaError_t)d2drp->err;
            }
        }
        break;
      default:
        WARN(0, "Unsupported value for cudaMemcpyKind : %s\n", dscudaMemcpyKindName(kind));
        exit(1);
    }

    WARN(3, "done.\n");

    return err;
}

cudaError_t
ibvCudaGetDeviceProperties(struct cudaDeviceProp *prop, int device)
{
    cudaError_t err = cudaSuccess;

    initClient();
    WARN(3, "cudaGetDeviceProperties(0x%08lx, %d)...", (unsigned long)prop, device);

    // Vdev_t *vdev = Vdev + device;
    //    for (int i = 0; i < vdev->nredundancy; i++) {
    for (int i = 0; i < 1; i++) { // performs no redundant call for now.
        IbvConnection *conn = (IbvConnection *)Cmid[device][i]->context;
        IbvGetDevicePropertiesInvokeHdr *spkt = (IbvGetDevicePropertiesInvokeHdr *)conn->rdma_local_region;
        IbvGetDevicePropertiesReturnHdr *rpkt = (IbvGetDevicePropertiesReturnHdr *)conn->rdma_remote_region;

        // pack send data.
        int spktsize = sizeof(IbvGetDevicePropertiesInvokeHdr);
        spkt->method = RCMethodGetDeviceProperties;
        spkt->device = device;
        WARN(3, "spktsize:%d  device:%d\n", spktsize, device);

        // perform remote call.
        rpkt->method = RCMethodNone;
        wait_ready_to_rdma(conn);
        kickoff_rdma(conn, spktsize);
        while (!rpkt->method) {
            // wait the returning packet.
        }

        // unpack returned data.
        err = rpkt->err;
        WARN(3, "cudaGetDeviceProperties err : %d\n", err);
        memcpy(prop, &rpkt->prop, sizeof(cudaDeviceProp));
    }
    WARN(3, "done.\n");

    return err;
}

cudaError_t
rpcCudaGetDeviceProperties(struct cudaDeviceProp *prop, int device)
{
    cudaError_t err = cudaSuccess;
    dscudaGetDevicePropertiesResult *rp;

    initClient();
    WARN(3, "cudaGetDeviceProperties(0x%08lx, %d)...", (unsigned long)prop, device);
    Vdev_t *vdev = Vdev + device;
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudagetdevicepropertiesid_1(device, Clnt[device][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
    }
    memcpy(prop, rp->prop.RCbuf_val, rp->prop.RCbuf_len);
    WARN(3, "done.\n");

    return err;
}

static int
ibvDscudaLoadModule(unsigned int ipaddr, pid_t pid, char *modulename, char *modulebuf, int vdevid, int raidid)
{
    cudaError_t err = cudaSuccess;
    IbvConnection *conn = (IbvConnection *)Cmid[vdevid][raidid]->context;
    IbvDscudaLoadModuleInvokeHdr *spkt = (IbvDscudaLoadModuleInvokeHdr *)conn->rdma_local_region;
    IbvDscudaLoadModuleReturnHdr *rpkt = (IbvDscudaLoadModuleReturnHdr *)conn->rdma_remote_region;
    int moduleid;
    int namelen = strlen(modulename);
    int imagelen = strlen(modulebuf);

    if (RC_KMODULENAMELEN <= namelen) {
        WARN(0, "ibvDscudaLoadModule:modulename too long (%d byte).\n", namelen);
        exit(1);
    }
    if (RC_KMODULEIMAGELEN <= imagelen) {
        WARN(0, "ibvDscudaLoadModule:modulebuf too long (%d byte).\n", imagelen);
        exit(1);
    }

    // pack send data.
    int spktsize = sizeof(IbvDscudaLoadModuleInvokeHdr) + imagelen + 1;
    spkt->method = RCMethodDscudaLoadModule;
    spkt->ipaddr = ipaddr;
    spkt->pid = pid;
    strncpy(spkt->modulename, modulename, RC_KMODULENAMELEN);
    strncpy((char *)&spkt->moduleimage, modulebuf, RC_KMODULEIMAGELEN);
    WARN(3, "spktsize:%d\n", spktsize);

    // perform remote call.
    rpkt->method = RCMethodNone;
    wait_ready_to_rdma(conn);
    kickoff_rdma(conn, spktsize);
    while (!rpkt->method) {
        // wait the returning packet.
    }

    // unpack returned data.
    err = rpkt->err;
    WARN(3, "ibvDscudaLoadModule err : %d\n", err);
    moduleid = rpkt->moduleid;

    return moduleid;
}

/*
 * launch a kernel function of id 'kid', defined in a module of id 'moduleid'.
 * 'kid' must be unique inside a single module.
 */

static void
ibvDscudaLaunchKernel(int moduleid, int kid, char *kname,
                     int *gdim, int *bdim, RCsize smemsize, RCstream stream,
                     int narg, IbvArg *arg, int vdevid, int raidid)
{
    cudaError_t err = cudaSuccess;
    IbvConnection *conn = (IbvConnection *)Cmid[vdevid][raidid]->context;
    IbvDscudaLaunchKernelInvokeHdr *spkt = (IbvDscudaLaunchKernelInvokeHdr *)conn->rdma_local_region;
    IbvDscudaLaunchKernelReturnHdr *rpkt = (IbvDscudaLaunchKernelReturnHdr *)conn->rdma_remote_region;
    int k;

    // pack send data.
    int spktsize = sizeof(IbvDscudaLaunchKernelInvokeHdr) + sizeof(IbvArg) * narg;
    spkt->method = RCMethodDscudaLaunchKernel;
    spkt->moduleid = moduleid;
    spkt->kernelid = kid;
    strncpy(spkt->kernelname, kname, RC_KNAMELEN);
    for (k = 0; k < 3; k++) {
        spkt->gdim[k] = gdim[k];
        spkt->bdim[k] = bdim[k];
    }
    spkt->smemsize = smemsize;
    spkt->stream = stream;
    spkt->narg = narg;
    memcpy((char *)&spkt->args, arg, sizeof(IbvArg) * narg);
    WARN(3, "spktsize:%d\n", spktsize);

    // perform remote call.
    rpkt->method = RCMethodNone;
    wait_ready_to_rdma(conn);
    kickoff_rdma(conn, spktsize);
    while (!rpkt->method) {
        // wait the returning packet.
    }

    // unpack returned data.
    err = rpkt->err;
    WARN(3, "ibvDscudaLaunchKernel err : %d\n", err);
}

void
ibvDscudaLaunchKernelWrapper(int *moduleid, int kid, char *kname,
                            int *gdim, int *bdim, RCsize smemsize, RCstream stream,
                            int narg, IbvArg *arg)
{
    RCmappedMem *mem;
    RCstreamArray *st;

    st = RCstreamArrayQuery((cudaStream_t)stream);
    if (!st) {
        WARN(0, "invalid stream : 0x%08llx\n", stream);
        exit(1);
    }

    mem = RCmappedMemListTop;
    while (mem) {
        cudaMemcpy(mem->pDevice, mem->pHost, mem->size, cudaMemcpyHostToDevice);
        mem = mem->next;
    }

    Vdev_t *vdev = Vdev + Vdevid;
    for (int i = 0; i < vdev->nredundancy; i++) {
        ibvDscudaLaunchKernel(moduleid[i], kid, kname,
                             gdim, bdim, smemsize, (RCstream)st->s[i],
                             narg, arg, Vdevid, i);
    }

    mem = RCmappedMemListTop;
    while (mem) {
        cudaMemcpy(mem->pHost, mem->pDevice, mem->size, cudaMemcpyDeviceToHost);
        mem = mem->next;
    }
}

/*
 * launch a kernel function of id 'kid', defined in a module of id 'moduleid'.
 * 'kid' must be unique inside a single module.
 */
void
dscudaLaunchKernelWrapper(int *moduleid, int kid, char *kname,
                         RCdim3 gdim, RCdim3 bdim, RCsize smemsize, RCstream stream,
                         RCargs args)
{
    RCmappedMem *mem;
    RCstreamArray *st;

    st = RCstreamArrayQuery((cudaStream_t)stream);
    if (!st) {
        WARN(0, "invalid stream : 0x%08llx\n", stream);
        exit(1);
    }

    mem = RCmappedMemListTop;
    while (mem) {
        cudaMemcpy(mem->pDevice, mem->pHost, mem->size, cudaMemcpyHostToDevice);
        mem = mem->next;
    }

    Vdev_t *vdev = Vdev + Vdevid;
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        void *rp = dscudalaunchkernelid_1(moduleid[i], kid, kname,
                                         gdim, bdim, smemsize, (RCstream)st->s[i],
                                         args, Clnt[Vdevid][sp->id]);
        checkResult(rp, sp);
    }

    mem = RCmappedMemListTop;
    while (mem) {
        cudaMemcpy(mem->pHost, mem->pDevice, mem->size, cudaMemcpyDeviceToHost);
        mem = mem->next;
    }
}

/*
 * Register a cudaArray array. each component is associated to a cudaArray
 * on each Server[]. User see only the 1st element, cuarrays[0].
 * Others, i.e., cuarrays[1..Nredunddancy-1], are used by this library
 * to handle redundant calculation mechanism.
 */
static void
RCcuarrayArrayRegister(cudaArray **cuarrays)
{
    RCcuarrayArray *ca = (RCcuarrayArray *)malloc(sizeof(RCcuarrayArray));
    if (!ca) {
        perror("RCcuarrayArrayRegister");
    }
    for (int i = 0; i < RC_NREDUNDANCYMAX; i++) {
        ca->ap[i] = cuarrays[i];
    }
    ca->prev = RCcuarrayArrayListTail;
    ca->next = NULL;
    if (!RCcuarrayArrayListTop) { // ca will be the 1st entry.
        RCcuarrayArrayListTop = ca;
    }
    else {
        RCcuarrayArrayListTail->next = ca;
    }
    RCcuarrayArrayListTail = ca;
}

static void
RCcuarrayArrayUnregister(cudaArray *cuarray0)
{
    RCcuarrayArray *ca = RCcuarrayArrayQuery(cuarray0);
    if (!ca) return;

    if (ca->prev) { // reconnect the linked list.
        ca->prev->next = ca->next;
    }
    else { // ca was the 1st entry.
        RCcuarrayArrayListTop = ca->next;
        if (ca->next) {
            ca->next->prev = NULL;
        }
    }
    if (!ca->next) { // ca was the last entry.
        RCcuarrayArrayListTail = ca->prev;
    }
    free(ca);
}

static RCcuarrayArray *
RCcuarrayArrayQuery(cudaArray *cuarray0)
{
    RCcuarrayArray *ca = RCcuarrayArrayListTop;
    while (ca) {
        if (ca->ap[0] == cuarray0) {
            return ca;
        }
        ca = ca->next;
    }
    return NULL;
}

cudaError_t
cudaMallocArray(struct cudaArray **array, const struct cudaChannelFormatDesc *desc,
                size_t width, size_t height, unsigned int flags)
{
    cudaError_t err = cudaSuccess;
    dscudaMallocArrayResult *rp;
    RCchanneldesc descbuf;
    cudaArray *ca[RC_NREDUNDANCYMAX];

    initClient();
    WARN(3, "cudaMallocArray(0x%08llx, 0x%08llx, %d, %d, 0x%08x)...",
         (unsigned long)array, desc, width, height, flags);


    descbuf.x = desc->x;
    descbuf.y = desc->y;
    descbuf.z = desc->z;
    descbuf.w = desc->w;
    descbuf.f = desc->f;

    Vdev_t *vdev = Vdev + Vdevid;
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudamallocarrayid_1(descbuf, width, height, flags, Clnt[Vdevid][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
        ca[i] = (cudaArray *)rp->array;
    }

    *array = ca[0];
    RCcuarrayArrayRegister(ca);
    WARN(3, "done. *array:0x%08llx\n", *array);

    return err;
}

cudaError_t
cudaFreeArray(struct cudaArray *array)
{
    cudaError_t err = cudaSuccess;
    dscudaResult *rp;
    RCcuarrayArray *ca;

    initClient();
    WARN(3, "cudaFreeArray(0x%08llx)...", (unsigned long)array);
    ca = RCcuarrayArrayQuery(array);
    if (!ca) {
        WARN(0, "invalid cudaArray : 0x%08llx\n", array);
        exit(1);
    }
    Vdev_t *vdev = Vdev + Vdevid;
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudafreearrayid_1((RCadr)ca->ap[i], Clnt[Vdevid][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
    }
    RCcuarrayArrayUnregister(ca->ap[0]);
    WARN(3, "done.\n");
    return err;
}

cudaError_t
cudaMemcpyToArray(struct cudaArray *dst, size_t wOffset, size_t hOffset, const void *src,
                  size_t count, enum cudaMemcpyKind kind)
{
    cudaError_t err = cudaSuccess;
    dscudaResult *h2drp, *d2drp;
    RCbuf srcbuf;
    RCcuarrayArray *ca;
    Vdev_t *vdev;
    RCServer_t *sp;

    initClient();

    WARN(3, "cudaMemcpyToArray(0x%08llx, %d, %d, 0x%08llx, %d, %s)...",
         (unsigned long)dst, wOffset, hOffset, (unsigned long)src, count, dscudaMemcpyKindName(kind));
    ca = RCcuarrayArrayQuery(dst);
    if (!ca) {
        WARN(0, "invalid cudaArray : 0x%08llx\n", dst);
        exit(1);
    }
    switch (kind) {
      case cudaMemcpyHostToDevice:
        srcbuf.RCbuf_len = count;
        srcbuf.RCbuf_val = (char *)src;

        vdev = Vdev + Vdevid;
        sp = vdev->server;
        for (int i = 0; i < vdev->nredundancy; i++, sp++) {
            h2drp = dscudamemcpytoarrayh2did_1((RCadr)ca->ap[i], wOffset, hOffset, srcbuf, count, Clnt[Vdevid][sp->id]);
            checkResult(h2drp, sp);
            if (h2drp->err != cudaSuccess) {
                err = (cudaError_t)h2drp->err;
            }
        }
        break;
      case cudaMemcpyDeviceToDevice:
        vdev = Vdev + Vdevid;
        sp = vdev->server;
        for (int i = 0; i < vdev->nredundancy; i++, sp++) {
            d2drp = dscudamemcpytoarrayd2did_1((RCadr)ca->ap[i], wOffset, hOffset, (RCadr)src, count, Clnt[Vdevid][sp->id]);
            checkResult(d2drp, sp);
            if (d2drp->err != cudaSuccess) {
                err = (cudaError_t)d2drp->err;
            }
        }
        break;
      default:
        WARN(0, "Unsupported value for cudaMemcpyKind : %s\n", dscudaMemcpyKindName(kind));
        exit(1);
    }
    WARN(3, "done.\n");

    return err;
}

cudaError_t
dscudaMemcpyToSymbolWrapper(int *moduleid, const char *symbol, const void *src,
                           size_t count, size_t offset, enum cudaMemcpyKind kind)
{
    cudaError_t err = cudaSuccess;
    dscudaResult *rp;
    RCbuf srcbuf;
    RCServer_t *sp;
    Vdev_t *vdev;

    initClient();

    WARN(3, "dscudaMemcpyToSymbolWrapper(%d, 0x%08llx, 0x%08llx, %d, %d, %s)"
         "symbol:%s  ...",
         moduleid, (unsigned long)symbol, (unsigned long)src,
         count, offset, dscudaMemcpyKindName(kind), symbol);

    switch (kind) {
      case cudaMemcpyHostToDevice:
        srcbuf.RCbuf_len = count;
        srcbuf.RCbuf_val = (char *)src;
        vdev = Vdev + Vdevid;
        sp = vdev->server;
        for (int i = 0; i < vdev->nredundancy; i++, sp++) {
            rp = dscudamemcpytosymbolh2did_1(moduleid[i], (char *)symbol, srcbuf, count, offset, Clnt[Vdevid][sp->id]);
            checkResult(rp, sp);
            if (rp->err != cudaSuccess) {
                err = (cudaError_t)rp->err;
            }
        }
        break;
      case cudaMemcpyDeviceToDevice:
        vdev = Vdev + Vdevid;
        sp = vdev->server;
        for (int i = 0; i < vdev->nredundancy; i++, sp++) {
            rp = dscudamemcpytosymbold2did_1(moduleid[i], (char *)symbol, (RCadr)src, count, offset, Clnt[Vdevid][sp->id]);
            checkResult(rp, sp);
            if (rp->err != cudaSuccess) {
                err = (cudaError_t)rp->err;
            }
        }
        break;
      default:
        WARN(0, "Unsupported value for cudaMemcpyKind : %s\n", dscudaMemcpyKindName(kind));
        exit(1);
    }
    WARN(3, "done.\n");

    return err;
}

cudaError_t
dscudaMemcpyFromSymbolWrapper(int *moduleid, void *dst, const char *symbol,
                             size_t count, size_t offset,
                             enum cudaMemcpyKind kind)
{
    cudaError_t err = cudaSuccess;
    dscudaMemcpyFromSymbolD2HResult *d2hrp;
    dscudaResult *d2drp;
    Vdev_t *vdev;
    RCServer_t *sp;

    initClient();

    WARN(3, "dscudaMemcpyFromSymbolWrapper(0x%08llx, 0x%08llx, 0x%08llx, %d, %d, %s)"
         "symbol:%s  ...",
         moduleid, (unsigned long)dst, (unsigned long)symbol,
         count, offset, dscudaMemcpyKindName(kind), symbol);

    switch (kind) {
      case cudaMemcpyDeviceToDevice:
        vdev = Vdev + Vdevid;
        sp = vdev->server;
        for (int i = 0; i < vdev->nredundancy; i++, sp++) {
            d2drp = dscudamemcpyfromsymbold2did_1(moduleid[i], (RCadr)dst, (char *)symbol, count, offset, Clnt[Vdevid][sp->id]);
            checkResult(d2drp, sp);
            if (d2drp->err != cudaSuccess) {
                err = (cudaError_t)d2drp->err;
            }
        }
        break;
      case cudaMemcpyDeviceToHost:
        vdev = Vdev + Vdevid;
        sp = vdev->server;
        for (int i = 0; i < vdev->nredundancy; i++, sp++) {
            d2hrp = dscudamemcpyfromsymbold2hid_1(moduleid[i], (char *)symbol, count, offset, Clnt[Vdevid][sp->id]);
            checkResult(d2hrp, sp);
            if (d2hrp->err != cudaSuccess) {
                err = (cudaError_t)d2hrp->err;
            }
            if (i == 0) {
                memcpy(dst, d2hrp->buf.RCbuf_val, d2hrp->buf.RCbuf_len);
            }
            else if (bcmp(dst, d2hrp->buf.RCbuf_val, d2hrp->buf.RCbuf_len) != 0) {
                if (errorHandler) {
                    errorHandler(errorHandlerArg);
                }
            }
            else {
                WARN(3, "cudaMemcpyFromSymbol() data copied from device%d matched with that from device0.\n", i);
            }
        }
        break;
      default:
        WARN(0, "Unsupported value for cudaMemcpyKind : %s\n", dscudaMemcpyKindName(kind));
        exit(1);
    }
    WARN(3, "done.\n");

    return err;
}


static cudaError_t
ibvDscudaMemcpyToSymbolAsyncH2D(int moduleid, char *symbol, const void *src,
                               size_t count, size_t offset, RCstream stream, int vdevid, int raidid)
{
    cudaError_t err = cudaSuccess;
    IbvConnection *conn = (IbvConnection *)Cmid[vdevid][raidid]->context;
    IbvDscudaMemcpyToSymbolAsyncH2DInvokeHdr *spkt = (IbvDscudaMemcpyToSymbolAsyncH2DInvokeHdr *)conn->rdma_local_region;
    IbvDscudaMemcpyToSymbolAsyncH2DReturnHdr *rpkt = (IbvDscudaMemcpyToSymbolAsyncH2DReturnHdr *)conn->rdma_remote_region;

    int snamelen = strlen(symbol);
    if (RC_SNAMELEN <= snamelen) {
        WARN(0, "ibvDscudaMemcpyToSymbolAsyncH2D:symbol name too long (%d byte).\n", snamelen);
        exit(1);
    }

    // pack send data.
    int spktsize = sizeof(IbvDscudaMemcpyToSymbolAsyncH2DInvokeHdr) + count;
    spkt->method = RCMethodDscudaMemcpyToSymbolAsyncH2D;
    spkt->moduleid = moduleid;
    strncpy(spkt->symbol, symbol, RC_SNAMELEN);
    memcpy((char *)&spkt->src, src, count);
    spkt->count = count;
    spkt->offset = offset;
    spkt->stream = stream;
    WARN(3, "spktsize:%d\n", spktsize);

    // perform remote call.
    rpkt->method = RCMethodNone;
    wait_ready_to_rdma(conn);
    kickoff_rdma(conn, spktsize);
    while (!rpkt->method) {
        // wait the returning packet.
    }

    // unpack returned data.
    err = rpkt->err;
    WARN(3, "ibvDscudaMemcpyToSymbolAsyncH2D err : %d\n", err);
    return err;
}

static cudaError_t
ibvDscudaMemcpyToSymbolAsyncD2D(int moduleid, char *symbol, const void *src,
                               size_t count, size_t offset, RCstream stream, int vdevid, int raidid)
{
    cudaError_t err = cudaSuccess;
    IbvConnection *conn = (IbvConnection *)Cmid[vdevid][raidid]->context;
    IbvDscudaMemcpyToSymbolAsyncD2DInvokeHdr *spkt = (IbvDscudaMemcpyToSymbolAsyncD2DInvokeHdr *)conn->rdma_local_region;
    IbvDscudaMemcpyToSymbolAsyncD2DReturnHdr *rpkt = (IbvDscudaMemcpyToSymbolAsyncD2DReturnHdr *)conn->rdma_remote_region;

    int snamelen = strlen(symbol);
    if (RC_SNAMELEN <= snamelen) {
        WARN(0, "ibvDscudaMemcpyToSymbolAsyncD2D:symbol name too long (%d byte).\n", snamelen);
        exit(1);
    }

    // pack send data.
    int spktsize = sizeof(IbvDscudaMemcpyToSymbolAsyncD2DInvokeHdr);
    spkt->method = RCMethodDscudaMemcpyToSymbolAsyncD2D;
    spkt->moduleid = moduleid;
    strncpy(spkt->symbol, symbol, RC_SNAMELEN);
    spkt->srcadr = (RCadr)src;
    spkt->count = count;
    spkt->offset = offset;
    spkt->stream = stream;
    WARN(3, "spktsize:%d\n", spktsize);

    // perform remote call.
    rpkt->method = RCMethodNone;
    wait_ready_to_rdma(conn);
    kickoff_rdma(conn, spktsize);
    while (!rpkt->method) {
        // wait the returning packet.
    }

    // unpack returned data.
    err = rpkt->err;
    WARN(3, "ibvDscudaMemcpyToSymbolAsyncD2D err : %d\n", err);
    return err;
}

cudaError_t
dscudaMemcpyToSymbolAsyncWrapper(int *moduleid, const char *symbol, const void *src,
				size_t count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream)
{
    cudaError_t err = cudaSuccess;
    dscudaResult *rp;
    RCbuf srcbuf;
    RCServer_t *sp;
    Vdev_t *vdev;
    RCstreamArray *st;

    initClient();

    WARN(3, "sym:%s\n", symbol);

    WARN(3, "dscudaMemcpyToSymbolAsyncWrapper(%d, 0x%08lx, 0x%08lx, %d, %d, %s, 0x%08lx) "
         "symbol:%s  ...",
         moduleid, (unsigned long)symbol, (unsigned long)src,
         count, offset, dscudaMemcpyKindName(kind), (unsigned long)stream, symbol);
    st = RCstreamArrayQuery(stream);
    if (!st) {
        WARN(0, "invalid stream : 0x%08llx\n", stream);
        exit(1);
    }
    switch (kind) {
      case cudaMemcpyHostToDevice:
        srcbuf.RCbuf_len = count;
        srcbuf.RCbuf_val = (char *)src;
        vdev = Vdev + Vdevid;
        sp = vdev->server;
        for (int i = 0; i < vdev->nredundancy; i++, sp++) {
            if (UseIbv) {
                err = ibvDscudaMemcpyToSymbolAsyncH2D(moduleid[i], (char *)symbol, src, count, offset, (RCstream)st->s[i],
                                                     Vdevid, i);
            }
            else {
                rp = dscudamemcpytosymbolasynch2did_1(moduleid[i], (char *)symbol, srcbuf, count, offset, (RCstream)st->s[i],
                                                     Clnt[Vdevid][sp->id]);
                checkResult(rp, sp);
                err = (cudaError_t)rp->err;
            }
        }
        break;
      case cudaMemcpyDeviceToDevice:
        vdev = Vdev + Vdevid;
        sp = vdev->server;
        for (int i = 0; i < vdev->nredundancy; i++, sp++) {
            if (UseIbv) {
                err = ibvDscudaMemcpyToSymbolAsyncD2D(moduleid[i], (char *)symbol, src, count, offset, (RCstream)st->s[i], 
                                                     Vdevid, i);
            }
            else {
                rp = dscudamemcpytosymbolasyncd2did_1(moduleid[i], (char *)symbol, (RCadr)src, count, offset, (RCstream)st->s[i], 
                                                     Clnt[Vdevid][sp->id]);
                checkResult(rp, sp);
                err = (cudaError_t)rp->err;
            }
        }
        break;
      default:
        WARN(0, "Unsupported value for cudaMemcpyKind : %s\n", dscudaMemcpyKindName(kind));
        exit(1);
    }
    WARN(3, "done.\n");

    return err;
}


static cudaError_t
ibvDscudaMemcpyFromSymbolAsyncD2H(int moduleid, void *dstbuf, char *symbol,
                               size_t count, size_t offset, RCstream stream, int vdevid, int raidid)
{
    cudaError_t err = cudaSuccess;
    IbvConnection *conn = (IbvConnection *)Cmid[vdevid][raidid]->context;
    IbvDscudaMemcpyFromSymbolAsyncD2HInvokeHdr *spkt = (IbvDscudaMemcpyFromSymbolAsyncD2HInvokeHdr *)conn->rdma_local_region;
    IbvDscudaMemcpyFromSymbolAsyncD2HReturnHdr *rpkt = (IbvDscudaMemcpyFromSymbolAsyncD2HReturnHdr *)conn->rdma_remote_region;

    int snamelen = strlen(symbol);
    if (RC_SNAMELEN <= snamelen) {
        WARN(0, "ibvDscudaMemcpyFromSymbolAsyncD2H:symbol name too long (%d byte).\n", snamelen);
        exit(1);
    }

    // pack send data.
    int spktsize = sizeof(IbvDscudaMemcpyFromSymbolAsyncD2HInvokeHdr);
    spkt->method = RCMethodDscudaMemcpyFromSymbolAsyncD2H;
    spkt->moduleid = moduleid;
    strncpy(spkt->symbol, symbol, RC_SNAMELEN);
    spkt->count = count;
    spkt->offset = offset;
    spkt->stream = stream;
    WARN(3, "spktsize:%d\n", spktsize);

    // perform remote call.
    rpkt->method = RCMethodNone;
    wait_ready_to_rdma(conn);
    kickoff_rdma(conn, spktsize);
    while (!rpkt->method) {
        // wait the returning packet.
    }

    // unpack returned data.
    err = rpkt->err;
    memcpy(dstbuf, (char *)&rpkt->dst, count);

    WARN(3, "ibvDscudaMemcpyFromSymbolAsyncD2H err : %d\n", err);
    return err;
}

static cudaError_t
ibvDscudaMemcpyFromSymbolAsyncD2D(int moduleid, void *dstadr, char *symbol,
                               size_t count, size_t offset, RCstream stream, int vdevid, int raidid)
{
    cudaError_t err = cudaSuccess;
    IbvConnection *conn = (IbvConnection *)Cmid[vdevid][raidid]->context;
    IbvDscudaMemcpyFromSymbolAsyncD2DInvokeHdr *spkt = (IbvDscudaMemcpyFromSymbolAsyncD2DInvokeHdr *)conn->rdma_local_region;
    IbvDscudaMemcpyFromSymbolAsyncD2DReturnHdr *rpkt = (IbvDscudaMemcpyFromSymbolAsyncD2DReturnHdr *)conn->rdma_remote_region;

    int snamelen = strlen(symbol);
    if (RC_SNAMELEN <= snamelen) {
        WARN(0, "ibvDscudaMemcpyFromSymbolAsyncD2D:symbol name too long (%d byte).\n", snamelen);
        exit(1);
    }

    // pack send data.
    int spktsize = sizeof(IbvDscudaMemcpyFromSymbolAsyncD2DInvokeHdr);
    spkt->method = RCMethodDscudaMemcpyFromSymbolAsyncD2D;
    spkt->moduleid = moduleid;
    spkt->dstadr = (RCadr)dstadr;
    strncpy(spkt->symbol, symbol, RC_SNAMELEN);
    spkt->count = count;
    spkt->offset = offset;
    spkt->stream = stream;
    WARN(3, "spktsize:%d\n", spktsize);

    // perform remote call.
    rpkt->method = RCMethodNone;
    wait_ready_to_rdma(conn);
    kickoff_rdma(conn, spktsize);
    while (!rpkt->method) {
        // wait the returning packet.
    }

    // unpack returned data.
    err = rpkt->err;
    WARN(3, "ibvDscudaMemcpyFromSymbolAsyncD2D err : %d\n", err);
    return err;
}

cudaError_t
dscudaMemcpyFromSymbolAsyncWrapper(int *moduleid, void *dst, const char *symbol,
				  size_t count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream)
{
    cudaError_t err = cudaSuccess;
    dscudaMemcpyFromSymbolAsyncD2HResult *d2hrp;
    dscudaResult *d2drp;
    Vdev_t *vdev;
    RCServer_t *sp;
    RCstreamArray *st;

    initClient();

    WARN(3, "dscudaMemcpyFromSymbolAsyncWrapper(%d, 0x%08lx, 0x%08lx, %d, %d, %s, 0x%08lx)"
         " symbol:%s  ...",
         moduleid, (unsigned long)dst, (unsigned long)symbol,
         count, offset, dscudaMemcpyKindName(kind), (unsigned long)stream, symbol);
    st = RCstreamArrayQuery(stream);
    if (!st) {
        WARN(0, "invalid stream : 0x%08llx\n", stream);
        exit(1);
    }
    switch (kind) {
      case cudaMemcpyDeviceToDevice:
        vdev = Vdev + Vdevid;
        sp = vdev->server;
        for (int i = 0; i < vdev->nredundancy; i++, sp++) {
            if (UseIbv) {
                err = ibvDscudaMemcpyFromSymbolAsyncD2D(moduleid[i], dst, (char *)symbol, count, offset, (RCstream)st->s[i],
                                                       Vdevid, i);
            }
            else {
                d2drp = dscudamemcpyfromsymbolasyncd2did_1(moduleid[i], (RCadr)dst, (char *)symbol, count, offset,
                                                          (RCstream)st->s[i], Clnt[Vdevid][sp->id]);
                checkResult(d2drp, sp);
                err = (cudaError_t)d2drp->err;
            }
        }
        break;
      case cudaMemcpyDeviceToHost:
        vdev = Vdev + Vdevid;
        sp = vdev->server;
        for (int i = 0; i < vdev->nredundancy; i++, sp++) {
            void *dstbuf;
            if (UseIbv) {
                dstbuf = calloc(1, count);
                if (!dstbuf) {
                    WARN(0, "dscudaMemcpyFromSymbolAsyncWrapper:calloc() failed.\n");
                    exit(1);
                }
                err = ibvDscudaMemcpyFromSymbolAsyncD2H(moduleid[i], dstbuf, (char *)symbol, count, offset, (RCstream)st->s[i],
                                                       Vdevid, i);
            }
            else {
                d2hrp = dscudamemcpyfromsymbolasyncd2hid_1(moduleid[i], (char *)symbol, count, offset,
                                                          (RCstream)st->s[i], Clnt[Vdevid][sp->id]);
                checkResult(d2hrp, sp);
                err = (cudaError_t)d2hrp->err;
                dstbuf = d2hrp->buf.RCbuf_val;
            }
            if (i == 0) {
                memcpy(dst, dstbuf, count);
            }
            else if (bcmp(dst, dstbuf, count) != 0) {
                if (errorHandler) {
                    errorHandler(errorHandlerArg);
                }
            }
            else {
                WARN(3, "cudaMemcpyFromSymbol() data copied from device%d matched with that from device0.\n", i);
            }
        }
        break;
      default:
        WARN(0, "Unsupported value for cudaMemcpyKind : %s\n", dscudaMemcpyKindName(kind));
        exit(1);
    }
    WARN(3, "done.\n");

    return err;
}


cudaError_t
cudaMemset(void *devPtr, int value, size_t count)
{
    cudaError_t err = cudaSuccess;
    dscudaResult *rp;

    initClient();
    WARN(3, "cudaMemset()...");
    Vdev_t *vdev = Vdev + Vdevid;
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudamemsetid_1((RCadr)devPtr, value, count, Clnt[Vdevid][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
    }
    WARN(3, "done.\n");

    return err;
}

cudaError_t
cudaMallocPitch(void **devPtr, size_t *pitch, size_t width, size_t height)
{
    cudaError_t err = cudaSuccess;
    dscudaMallocPitchResult *rp;

    initClient();
    WARN(3, "cudaMallocPitch(0x%08llx, 0x%08llx, %d, %d)...",
         (unsigned long)devPtr, pitch, width, height);
    Vdev_t *vdev = Vdev + Vdevid;
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudamallocpitchid_1(width, height, Clnt[Vdevid][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
    }

    *devPtr = (void*)rp->devPtr;
    *pitch = rp->pitch;
    WARN(3, "done. *devPtr:0x%08llx  *pitch:%d\n", *devPtr, *pitch);

    return err;
}

cudaError_t
cudaMemcpy2DToArray(struct cudaArray *dst, size_t wOffset, size_t hOffset,
                    const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind)
{
    cudaError_t err = cudaSuccess;
    dscudaMemcpy2DToArrayD2HResult *d2hrp;
    dscudaResult *h2drp, *d2drp;
    RCbuf srcbuf;
    RCcuarrayArray *ca;
    Vdev_t *vdev;
    RCServer_t *sp;

    initClient();

    WARN(3, "cudaMemcpy2DToArray(0x%08llx, %d, %d, 0x%08llx, %d, %d, %d, %s)...",
         (unsigned long)dst, wOffset, hOffset,
         (unsigned long)src, spitch, width, height, dscudaMemcpyKindName(kind));
    ca = RCcuarrayArrayQuery(dst);
    if (!ca) {
        WARN(0, "invalid cudaArray : 0x%08llx\n", dst);
        exit(1);
    }
    switch (kind) {
      case cudaMemcpyDeviceToHost:
        vdev = Vdev + Vdevid;
        sp = vdev->server;
        for (int i = 0; i < vdev->nredundancy; i++, sp++) {
            d2hrp = dscudamemcpy2dtoarrayd2hid_1(wOffset, hOffset,
                                                (RCadr)src, spitch, width, height, Clnt[Vdevid][sp->id]);
            checkResult(d2hrp, sp);
            if (d2hrp->err != cudaSuccess) {
                err = (cudaError_t)d2hrp->err;
            }
            if (i == 0) {
                memcpy(dst, d2hrp->buf.RCbuf_val, d2hrp->buf.RCbuf_len);
            }
            else if (bcmp(dst, d2hrp->buf.RCbuf_val, d2hrp->buf.RCbuf_len) != 0) {
                if (errorHandler) {
                    errorHandler(errorHandlerArg);
                }
            }
            else {
                WARN(3, "cudaMemcpy2DToArray() data copied from device%d matched with that from device0.\n", i);
            }
        }
        break;
      case cudaMemcpyHostToDevice:
        srcbuf.RCbuf_len = spitch * height;
        srcbuf.RCbuf_val = (char *)src;
        vdev = Vdev + Vdevid;
        sp = vdev->server;
        for (int i = 0; i < vdev->nredundancy; i++, sp++) {
            h2drp = dscudamemcpy2dtoarrayh2did_1((RCadr)ca->ap[i], wOffset, hOffset,
                                                srcbuf, spitch, width, height, Clnt[Vdevid][sp->id]);
            checkResult(h2drp, sp);
            if (h2drp->err != cudaSuccess) {
                err = (cudaError_t)h2drp->err;
            }
        }
        break;
      case cudaMemcpyDeviceToDevice:
        vdev = Vdev + Vdevid;
        sp = vdev->server;
        for (int i = 0; i < vdev->nredundancy; i++, sp++) {
            d2drp = dscudamemcpy2dtoarrayd2did_1((RCadr)ca->ap[i], wOffset, hOffset,
                                                (RCadr)src, spitch, width, height, Clnt[Vdevid][sp->id]);
            checkResult(d2drp, sp);
            if (d2drp->err != cudaSuccess) {
                err = (cudaError_t)d2drp->err;
            }
        }
        break;
      default:
        WARN(0, "Unsupported value for cudaMemcpyKind : %s\n", dscudaMemcpyKindName(kind));
        exit(1);
    }
    WARN(3, "done.\n");

    return err;
}

cudaError_t
cudaMemcpy2D(void *dst, size_t dpitch,
             const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind)
{
    cudaError_t err = cudaSuccess;
    dscudaMemcpy2DD2HResult *d2hrp;
    dscudaResult *h2drp, *d2drp;
    RCbuf srcbuf;
    Vdev_t *vdev;
    RCServer_t *sp;

    initClient();

    WARN(3, "cudaMemcpy2D(0x%08llx, %d, 0x%08llx, %d, %d, %d, %s)...",
         (unsigned long)dst, dpitch,
         (unsigned long)src, spitch, width, height, dscudaMemcpyKindName(kind));

    switch (kind) {
      case cudaMemcpyDeviceToHost:
        vdev = Vdev + Vdevid;
        sp = vdev->server;
        for (int i = 0; i < vdev->nredundancy; i++, sp++) {
            d2hrp = dscudamemcpy2dd2hid_1(dpitch,
                                         (RCadr)src, spitch, width, height, Clnt[Vdevid][sp->id]);
            checkResult(d2hrp, sp);
            if (d2hrp->err != cudaSuccess) {
                err = (cudaError_t)d2hrp->err;
            }
            if (i == 0) {
                memcpy(dst, d2hrp->buf.RCbuf_val, d2hrp->buf.RCbuf_len);
            }
            else if (bcmp(dst, d2hrp->buf.RCbuf_val, d2hrp->buf.RCbuf_len) != 0) {
                if (errorHandler) {
                    errorHandler(errorHandlerArg);
                }
            }
            else {
                WARN(3, "cudaMemcpy() data copied from device%d matched with that from device0.\n", i);
            }
        }
        break;
      case cudaMemcpyHostToDevice:
        srcbuf.RCbuf_len = spitch * height;
        srcbuf.RCbuf_val = (char *)src;
        vdev = Vdev + Vdevid;
        sp = vdev->server;
        for (int i = 0; i < vdev->nredundancy; i++, sp++) {
            h2drp = dscudamemcpy2dh2did_1((RCadr)dst, dpitch,
                                         srcbuf, spitch, width, height, Clnt[Vdevid][sp->id]);
            checkResult(h2drp, sp);
            if (h2drp->err != cudaSuccess) {
                err = (cudaError_t)h2drp->err;
            }
        }
        break;
      case cudaMemcpyDeviceToDevice:
        vdev = Vdev + Vdevid;
        sp = vdev->server;
        for (int i = 0; i < vdev->nredundancy; i++, sp++) {
            d2drp = dscudamemcpy2dd2did_1((RCadr)dst, dpitch,
                                         (RCadr)src, spitch, width, height, Clnt[Vdevid][sp->id]);
            checkResult(d2drp, sp);
            if (d2drp->err != cudaSuccess) {
                err = (cudaError_t)d2drp->err;
            }
        }
        break;
      default:
        WARN(0, "Unsupported value for cudaMemcpyKind : %s\n", dscudaMemcpyKindName(kind));
        exit(1);
    }
    WARN(3, "done.\n");

    return err;
}

cudaError_t
cudaMemset2D(void *devPtr, size_t pitch, int value, size_t width, size_t height)
{
    cudaError_t err = cudaSuccess;
    dscudaResult *rp;

    initClient();
    WARN(3, "cudaMemset2D(0x%08llx, %d, %d, %d, %d)...",
         (unsigned long)devPtr, pitch, value, width, height);
    Vdev_t *vdev = Vdev + Vdevid;
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudamemset2did_1((RCadr)devPtr, pitch, value, width, height, Clnt[Vdevid][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
    }
    WARN(3, "done.\n");

    return err;
}

cudaError_t
cudaMallocHost(void **ptr, size_t size)
{
#if RC_SUPPORT_PAGELOCK
    cudaError_t err = cudaSuccess;
    dscudaMallocHostResult *rp;

    initClient();
    WARN(3, "cudaMallocHost(0x%08llx, %d)...", (unsigned long)ptr, size);
    Vdev_t *vdev = Vdev + Vdevid;
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudamallochostid_1(size, Clnt[Vdevid][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
    }

    *ptr = (void*)rp->ptr;

    WARN(3, "done. *ptr:0x%08llx\n", *ptr);
    return err;
#else
    // returned memory is not page locked.
    // it cannot be passed to cudaMemcpyAsync().
    *ptr = malloc(size);
    if (*ptr) {
        return cudaSuccess;
    }
    else {
        return cudaErrorMemoryAllocation;
    }
#endif
}

cudaError_t
cudaHostAlloc(void **pHost, size_t size, unsigned int flags)
{
#if RC_SUPPORT_PAGELOCK
    cudaError_t err = cudaSuccess;
    dscudaHostAllocResult *rp;

    initClient();
    WARN(3, "cudaHostAlloc(0x%08llx, %d, 0x%08x)...", (unsigned long)pHost, size, flags);
    Vdev_t *vdev = Vdev + Vdevid;
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudahostallocid_1(size, flags, Clnt[Vdevid][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
    }

    *pHost = (void*)rp->pHost;
    WARN(3, "done. *pHost:0x%08llx\n", *pHost);

    return err;
#else
    // returned memory is not page locked.
    // it cannot be passed to cudaMemcpyAsync().

    cudaError_t err = cudaSuccess;
    void *devmem;

    initClient();
    WARN(3, "cudaHostAlloc(0x%08llx, %d, 0x%08x)...", (unsigned long)pHost, size, flags);

    *pHost = malloc(size);
    if (!*pHost) return cudaErrorMemoryAllocation;
    if (!(flags & cudaHostAllocMapped)) {
        WARN(3, "done. *pHost:0x%08llx\n", *pHost);
        return cudaSuccess;
    }

    // flags says the host memory must be mapped on to the device memory.
    err = cudaMalloc(&devmem, size);
    if (err == cudaSuccess) {
        RCmappedMemRegister(*pHost, devmem, size);
    }
    WARN(3, "done. host mem:0x%08llx  device mem:0x%08llx\n", *pHost, devmem);

    return err;
#endif
}

cudaError_t
cudaFreeHost(void *ptr)
{
#if RC_SUPPORT_PAGELOCK
    cudaError_t err = cudaSuccess;
    dscudaResult *rp;

    initClient();
    WARN(3, "cudaFreeHost(0x%08llx)...", (unsigned long)ptr);
    Vdev_t *vdev = Vdev + Vdevid;
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudafreehostid_1((RCadr)ptr, Clnt[Vdevid][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
    }
    WARN(3, "done.\n");
    return err;
#else
    cudaError_t err = cudaSuccess;
    RCmappedMem *mem = RCmappedMemQuery(ptr);
    free(ptr);
    if (mem) { // ptr mapped on to a device memory.
        err = cudaFree(mem->pDevice);
        RCmappedMemUnregister(ptr);
        return err;
    }
    else {
        return cudaSuccess;
    }
#endif
}

// flags is not used for now in CUDA3.2. It should always be zero.
cudaError_t
cudaHostGetDevicePointer(void **pDevice, void*pHost, unsigned int flags)
{
#if RC_SUPPORT_PAGELOCK
    cudaError_t err = cudaSuccess;
    dscudaHostGetDevicePointerResult *rp;

    initClient();
    WARN(3, "cudaHostGetDevicePointer(0x%08llx, 0x%08llx, 0x%08x)...",
         (unsigned long)pDevice, (unsigned long)pHost, flags);
    Vdev_t *vdev = Vdev + Vdevid;
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudahostgetdevicepointerid_1((RCadr)pHost, flags, Clnt[Vdevid][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
    }

    *pDevice = (void *)rp->pDevice;
    WARN(3, "done. *pDevice:0x%08llx\n", *pDevice);
    return err;
#else
    RCmappedMem *mem = RCmappedMemQuery(pHost);
    if (!mem) return cudaErrorInvalidValue; // pHost is not registered as RCmappedMem.
    *pDevice = mem->pDevice;
    return cudaSuccess;
#endif
}

cudaError_t
cudaHostGetFlags(unsigned int *pFlags, void *pHost)
{
    cudaError_t err = cudaSuccess;
    dscudaHostGetFlagsResult *rp;

    initClient();
    WARN(3, "cudaHostGetFlags(0x%08x 0x%08llx)...",
         (unsigned long)pFlags, (unsigned long)pHost);
    Vdev_t *vdev = Vdev + Vdevid;
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudahostgetflagsid_1((RCadr)pHost, Clnt[Vdevid][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
    }
    *pFlags = rp->flags;
    WARN(3, "done. flags:0x%08x\n", *pFlags);
    return err;
    
}

cudaError_t
cudaMemcpyAsync(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream)
{
#if RC_SUPPORT_PAGELOCK
    cudaError_t err = cudaSuccess;
    dscudaMemcpyAsyncD2HResult *d2hrp;
    dscudaResult *h2drp, *d2drp;
    RCbuf srcbuf;
    RCstreamArray *st;
    Vdev_t *vdev;
    RCServer_t *sp;

    initClient();

    WARN(3, "cudaMemcpyAsync(0x%08llx, 0x%08llx, %d, %s, 0x%08llx)...",
         (unsigned long)dst, (unsigned long)src, count, dscudaMemcpyKindName(kind), st->s[0]);
    st = RCstreamArrayQuery(stream);
    if (!st) {
        WARN(0, "invalid stream : 0x%08llx\n", stream);
        exit(1);
    }
    switch (kind) {
      case cudaMemcpyDeviceToHost:
        Vdev_t *vdev = Vdev + Vdevid;
        RCServer_t *sp = vdev->server;
        for (int i = 0; i < vdev->nredundancy; i++, sp++) {
            d2hrp = dscudamemcpyasyncd2hid_1((RCadr)src, count, (RCstream)st->s[i], Clnt[Vdevid][sp->id]);
            checkResult(d2hrp, sp);
            if (d2hrp->err != cudaSuccess) {
                err = (cudaError_t)d2hrp->err;
            }
        }
        memcpy(dst, d2hrp->buf.RCbuf_val, d2hrp->buf.RCbuf_len);
        break;
      case cudaMemcpyHostToDevice:
        srcbuf.RCbuf_len = count;
        srcbuf.RCbuf_val = (char *)src;
        Vdev_t *vdev = Vdev + Vdevid;
        RCServer_t *sp = vdev->server;
        for (int i = 0; i < vdev->nredundancy; i++, sp++) {
            h2drp = dscudamemcpyasynch2did_1((RCadr)dst, srcbuf, count, (RCstream)st->s[i], Clnt[Vdevid][sp->id]);
            checkResult(h2drp, sp);
            if (h2drp->err != cudaSuccess) {
                err = (cudaError_t)h2drp->err;
            }
        }
        break;
      case cudaMemcpyDeviceToDevice:
        Vdev_t *vdev = Vdev + Vdevid;
        RCServer_t *sp = vdev->server;
        for (int i = 0; i < vdev->nredundancy; i++, sp++) {
            d2drp = dscudamemcpyasyncd2did_1((RCadr)dst, (RCadr)src, count, (RCstream)st->s[i], Clnt[Vdevid][sp->id]);
            checkResult(d2drp, sp);
            if (d2drp->err != cudaSuccess) {
                err = (cudaError_t)d2drp->err;
            }
        }
        break;
      default:
        WARN(0, "Unsupported value for cudaMemcpyKind : %s\n", dscudaMemcpyKindName(kind));
        exit(1);
    }
    WARN(3, "done.\n");

    return err;

#else
    // this DOES block.
    // this is only for use with a poor implementation of dscudaMallocHost().
    return cudaMemcpy(dst, src, count, kind);
#endif
}

/*
 * Stream Management
 */

/*
 * Register a stream array. each component is associated to a stream
 * on each Server[]. User see only the 1st element, streams[0].
 * Others, i.e., streams[1..Nredunddancy-1], are used by this library
 * to handle redundant calculation mechanism.
 */
static void
RCstreamArrayRegister(cudaStream_t *streams)
{
    RCstreamArray *st = (RCstreamArray *)malloc(sizeof(RCstreamArray));
    if (!st) {
        perror("RCstreamArrayRegister");
    }
    for (int i = 0; i < RC_NREDUNDANCYMAX; i++) {
        st->s[i] = streams[i];
    }
    st->prev = RCstreamArrayListTail;
    st->next = NULL;
    if (!RCstreamArrayListTop) { // st will be the 1st entry.
        RCstreamArrayListTop = st;
    }
    else {
        RCstreamArrayListTail->next = st;
    }
    RCstreamArrayListTail = st;
}

#if 0
static void
showsta(void)
{
    RCstreamArray *st = RCstreamArrayListTop;
    while (st) {
        fprintf(stderr, ">>> 0x%08llx    prev:%p  next:%p\n", st, st->prev, st->next);
        st = st->next;
    }
}
#endif

static void
RCstreamArrayUnregister(cudaStream_t stream0)
{
    RCstreamArray *st = RCstreamArrayQuery(stream0);
    if (!st) return;

    if (st->prev) { // reconnect the linked list.
        st->prev->next = st->next;
    }
    else { // st was the 1st entry.
        RCstreamArrayListTop = st->next;
        if (st->next) {
            st->next->prev = NULL;
        }
    }
    if (!st->next) { // st was the last entry.
        RCstreamArrayListTail = st->prev;
    }
    free(st);
    //    showsta();
}

static RCstreamArray *
RCstreamArrayQuery(cudaStream_t stream0)
{
    static RCstreamArray default_stream = { 0,};

    if (stream0 == 0) {
        return &default_stream;
    }

    RCstreamArray *st = RCstreamArrayListTop;
    while (st) {
        if (st->s[0] == stream0) {
            return st;
        }
        st = st->next;
    }
    return NULL;
}

cudaError_t
cudaStreamCreate(cudaStream_t *pStream)
{
#if RC_SUPPORT_STREAM
    cudaError_t err = cudaSuccess;
    dscudaStreamCreateResult *rp;
    cudaStream_t st[RC_NREDUNDANCYMAX];

    initClient();
    WARN(3, "cudaStreamCreate(0x%08llx)...", (unsigned long)pStream);
    Vdev_t *vdev = Vdev + Vdevid;
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudastreamcreateid_1(Clnt[Vdevid][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
        st[i] = (cudaStream_t)rp->stream;
    }

    *pStream = st[0];
    RCstreamArrayRegister(st);
    WARN(3, "done. *pStream:0x%08llx\n", *pStream);

    return err;
#else
    *pStream = 0;
    return cudaSuccess;
#endif
}

cudaError_t
cudaStreamDestroy(cudaStream_t stream)
{
#if RC_SUPPORT_STREAM
    cudaError_t err = cudaSuccess;
    dscudaResult *rp;
    RCstreamArray *st;

    initClient();
    WARN(3, "cudaStreamDestroy(0x%08llx)...", (unsigned long)stream);
    st = RCstreamArrayQuery(stream);
    if (!st) {
        WARN(0, "invalid stream : 0x%08llx\n", stream);
        exit(1);
    }
    Vdev_t *vdev = Vdev + Vdevid;
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudastreamdestroyid_1((RCadr)st->s[i], Clnt[Vdevid][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
    }
    RCstreamArrayUnregister(st->s[0]);
    WARN(3, "done.\n");
    return err;
#else
    return cudaSuccess;
#endif
}

cudaError_t
cudaStreamSynchronize(cudaStream_t stream)
{
#if RC_SUPPORT_STREAM
    cudaError_t err = cudaSuccess;
    dscudaResult *rp;
    RCstreamArray *st;

    initClient();
    WARN(3, "cudaStreamSynchronize(0x%08llx)...", (unsigned long)stream);
    st = RCstreamArrayQuery(stream);
    if (!st) {
        WARN(0, "invalid stream : 0x%08llx\n", stream);
        exit(1);
    }
    Vdev_t *vdev = Vdev + Vdevid;
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudastreamsynchronizeid_1((RCadr)st->s[i], Clnt[Vdevid][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
    }
    WARN(3, "done.\n");
    return err;
#else
    return cudaSuccess;
#endif
}

cudaError_t
cudaStreamQuery(cudaStream_t stream)
{
#if RC_SUPPORT_STREAM
    cudaError_t err = cudaSuccess;
    dscudaResult *rp;
    RCstreamArray *st;

    initClient();
    WARN(3, "cudaStreamQuery(0x%08llx)...", (unsigned long)stream);
    st = RCstreamArrayQuery(stream);
    if (!st) {
        WARN(0, "invalid stream : 0x%08llx\n", stream);
        exit(1);
    }
    Vdev_t *vdev = Vdev + Vdevid;
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudastreamqueryid_1((RCadr)st->s[i], Clnt[Vdevid][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
    }
    WARN(3, "done.\n");
    return err;
#else
    return cudaSuccess;
#endif
}

/*
 * Event Management
 */

/*
 * Register an event array. each component is associated to an event
 * on each Server[]. User see only the 1st element, events[0].
 * Others, i.e., events[1..Nredunddancy-1], are used by this library
 * to handle redundant calculation mechanism.
 */
static void
RCeventArrayRegister(cudaEvent_t *events)
{
    RCeventArray *ev = (RCeventArray *)malloc(sizeof(RCeventArray));
    if (!ev) {
        perror("RCeventArrayRegister");
    }
    for (int i = 0; i < RC_NREDUNDANCYMAX; i++) {
        ev->e[i] = events[i];
    }
    ev->prev = RCeventArrayListTail;
    ev->next = NULL;
    if (!RCeventArrayListTop) { // ev will be the 1st entry.
        RCeventArrayListTop = ev;
    }
    else {
        RCeventArrayListTail->next = ev;
    }
    RCeventArrayListTail = ev;
}

static void
RCeventArrayUnregister(cudaEvent_t event0)
{
    RCeventArray *ev = RCeventArrayQuery(event0);
    if (!ev) return;

    if (ev->prev) { // reconnect the linked list.
        ev->prev->next = ev->next;
    }
    else { // ev was the 1st entry.
        RCeventArrayListTop = ev->next;
        if (ev->next) {
            ev->next->prev = NULL;
        }
    }
    if (!ev->next) { // ev was the last entry.
        RCeventArrayListTail = ev->prev;
    }
    free(ev);
}

static RCeventArray *
RCeventArrayQuery(cudaEvent_t event0)
{
    RCeventArray *ev = RCeventArrayListTop;
    while (ev) {
        if (ev->e[0] == event0) {
            return ev;
        }
        ev = ev->next;
    }
    return NULL;
}

cudaError_t
ibvCudaEventCreate(cudaEvent_t *event)
{
    static cudaEvent_t e;
    *event = e;
    WARN(3, "a dummy call to cudaEventCreate()\n");
    return cudaSuccess;
}

cudaError_t
rpcCudaEventCreate(cudaEvent_t *event)
{
    cudaError_t err = cudaSuccess;
    dscudaEventCreateResult *rp;
    cudaEvent_t ev[RC_NREDUNDANCYMAX];

    initClient();
    WARN(3, "cudaEventCreate(0x%08llx)...", (unsigned long)event);
    Vdev_t *vdev = Vdev + Vdevid;
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudaeventcreateid_1(Clnt[Vdevid][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
        ev[i] = (cudaEvent_t)rp->event;
    }
    *event = ev[0];
    RCeventArrayRegister(ev);
    WARN(3, "done. *event:0x%08llx\n", *event);

    return err;
}

cudaError_t
ibvCudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags)
{
    static cudaEvent_t e;
    *event = e;
    WARN(3, "a dummy call to cudaEventCreateWithFlags()\n");
    return cudaSuccess;
}

cudaError_t
rpcCudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags)
{
    cudaError_t err = cudaSuccess;
    dscudaEventCreateResult *rp;
    cudaEvent_t ev[RC_NREDUNDANCYMAX];

    initClient();
    WARN(3, "cudaEventCreateWithFlags(0x%08llx, 0x%08x)...", (unsigned long)event, flags);
    Vdev_t *vdev = Vdev + Vdevid;
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudaeventcreatewithflagsid_1(flags, Clnt[Vdevid][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
        ev[i] = (cudaEvent_t)rp->event;
    }
    *event = ev[0];
    RCeventArrayRegister(ev);
    WARN(3, "done. *event:0x%08llx\n", *event);

    return err;
}

cudaError_t
ibvCudaEventDestroy(cudaEvent_t event)
{
    WARN(3, "a dummy call to cudaEventDestroy()\n");
    return cudaSuccess;
}

cudaError_t
rpcCudaEventDestroy(cudaEvent_t event)
{
    cudaError_t err = cudaSuccess;
    dscudaResult *rp;
    RCeventArray *ev;

    initClient();
    WARN(3, "cudaEventDestroy(0x%08llx)...", (unsigned long)event);
    ev = RCeventArrayQuery(event);
    if (!ev) {
        WARN(0, "invalid event : 0x%08llx\n", event);
        exit(1);
    }
    Vdev_t *vdev = Vdev + Vdevid;
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudaeventdestroyid_1((RCadr)ev->e[i], Clnt[Vdevid][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
    }
    RCeventArrayUnregister(ev->e[0]);
    WARN(3, "done.\n");
    return err;
}

cudaError_t
ibvCudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end)
{
    *ms = 123.0;
    WARN(3, "a dummy call to cudaEventElapsedTime()\n");
    return cudaSuccess;
}

cudaError_t
rpcCudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end)
{
    cudaError_t err = cudaSuccess;
    dscudaEventElapsedTimeResult *rp;
    RCeventArray *es, *ee;

    initClient();
    WARN(3, "cudaEventElapsedTime(0x%08llx, 0x%08llx, 0x%08llx)...",
         (unsigned long)ms, (unsigned long)start, (unsigned long)end);
    es = RCeventArrayQuery(start);
    if (!es) {
        WARN(0, "invalid start event : 0x%08llx\n", start);
        exit(1);
    }
    ee = RCeventArrayQuery(end);
    if (!ee) {
        WARN(0, "invalid end event : 0x%08llx\n", end);
        exit(1);
    }
    Vdev_t *vdev = Vdev + Vdevid;
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudaeventelapsedtimeid_1((RCadr)es->e[i], (RCadr)ee->e[i], Clnt[Vdevid][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
    }

    *ms = rp->ms;
    WARN(3, "done.\n");
    return err;
}

cudaError_t
ibvCudaEventRecord(cudaEvent_t event, cudaStream_t stream)
{
    WARN(3, "a dummy call to cudaEventRecord()\n");
    return cudaSuccess;
}

cudaError_t
rpcCudaEventRecord(cudaEvent_t event, cudaStream_t stream)
{
    cudaError_t err = cudaSuccess;
    dscudaResult *rp;
    RCstreamArray *st;
    RCeventArray *ev;

    initClient();
    WARN(3, "cudaEventRecord(0x%08llx, 0x%08llx)...", (unsigned long)event, (unsigned long)stream);
    st = RCstreamArrayQuery(stream);
    if (!st) {
        WARN(0, "invalid stream : 0x%08llx\n", stream);
        exit(1);
    }
    ev = RCeventArrayQuery(event);
    if (!ev) {
        WARN(0, "invalid event : 0x%08llx\n", event);
        exit(1);
    }
    Vdev_t *vdev = Vdev + Vdevid;
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudaeventrecordid_1((RCadr)ev->e[i], (RCadr)st->s[i], Clnt[Vdevid][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
    }

    WARN(3, "done.\n");
    return err;
}

cudaError_t
ibvCudaEventSynchronize(cudaEvent_t event)
{
    WARN(3, "a dummy call to cudaEventSynchronize()\n");
    return cudaSuccess;
}

cudaError_t
rpcCudaEventSynchronize(cudaEvent_t event)
{
    cudaError_t err = cudaSuccess;
    dscudaResult *rp;
    RCeventArray *ev;

    initClient();
    WARN(3, "cudaEventSynchronize(0x%08llx)...", (unsigned long)event);
    ev = RCeventArrayQuery(event);
    if (!ev) {
        WARN(0, "invalid event : 0x%08llx\n", event);
        exit(1);
    }
    Vdev_t *vdev = Vdev + Vdevid;
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudaeventsynchronizeid_1((RCadr)ev->e[i], Clnt[Vdevid][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
    }
    WARN(3, "done.\n");
    return err;
}

cudaError_t
ibvCudaEventQuery(cudaEvent_t event)
{
    WARN(3, "a dummy call to cudaEventQuery()\n");
    return cudaSuccess;
}

cudaError_t
rpcCudaEventQuery(cudaEvent_t event)
{
    cudaError_t err = cudaSuccess;
    dscudaResult *rp;
    RCeventArray *ev;

    initClient();
    WARN(3, "cudaEventQuery(0x%08llx)...", (unsigned long)event);
    ev = RCeventArrayQuery(event);
    if (!ev) {
        WARN(0, "invalid event : 0x%08llx\n", event);
        exit(1);
    }
    Vdev_t *vdev = Vdev + Vdevid;
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudaeventqueryid_1((RCadr)ev->e[i], Clnt[Vdevid][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
    }

    WARN(3, "done.\n");
    return err;
}

cudaError_t
cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags)
{
    cudaError_t err = cudaSuccess;
    dscudaResult *rp;
    RCstreamArray *st;
    RCeventArray *ev;

    initClient();
    WARN(3, "cudaStreamWaitEvent(0x%08llx, 0x%08llx, 0x%08x)...",
         (unsigned long)stream, (unsigned long)event, flags);
    st = RCstreamArrayQuery(stream);
    if (!st) {
        WARN(0, "invalid stream : 0x%08llx\n", stream);
        exit(1);
    }
    ev = RCeventArrayQuery(event);
    if (!ev) {
        WARN(0, "invalid event : 0x%08llx\n", event);
        exit(1);
    }
    Vdev_t *vdev = Vdev + Vdevid;
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudastreamwaiteventid_1((RCadr)st->s[i], (RCadr)ev->e[i], flags, Clnt[Vdevid][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
    }

    WARN(3, "done.\n");
    return err;
}

/*
 * Texture Reference Management
 */

cudaChannelFormatDesc
cudaCreateChannelDesc(int x, int y, int z, int w, enum cudaChannelFormatKind f)
{
    dscudaCreateChannelDescResult *rp;
    cudaChannelFormatDesc desc;

    initClient();
    WARN(3, "cudaCreateChannelDesc()...");
    Vdev_t *vdev = Vdev + Vdevid;
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudacreatechanneldescid_1(x, y, z, w, f, Clnt[Vdevid][sp->id]);
        checkResult(rp, sp);
    }
    WARN(3, "done.\n");

    desc.x = rp->x;
    desc.y = rp->y;
    desc.z = rp->z;
    desc.w = rp->w;
    desc.f = (enum cudaChannelFormatKind)rp->f;

    return desc;
}

cudaError_t
cudaGetChannelDesc(struct cudaChannelFormatDesc *desc, const struct cudaArray *array)
{
    cudaError_t err = cudaSuccess;
    dscudaGetChannelDescResult *rp;
    RCcuarrayArray *ca;

    initClient();
    WARN(3, "cudaGetChannelDesc()...");
    ca = RCcuarrayArrayQuery((cudaArray *)array);
    if (!ca) {
        WARN(0, "invalid cudaArray : 0x%08llx\n", array);
        exit(1);
    }
    Vdev_t *vdev = Vdev + Vdevid;
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudagetchanneldescid_1((RCadr)ca->ap[i], Clnt[Vdevid][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
    }

    WARN(3, "done.\n");

    desc->x = rp->x;
    desc->y = rp->y;
    desc->z = rp->z;
    desc->w = rp->w;
    desc->f = (enum cudaChannelFormatKind)rp->f;

    WARN(3, "done.\n");
    return err;
}

static void
setTextureParams(RCtexture *texbufp, const struct textureReference *tex, const struct cudaChannelFormatDesc *desc)
{
    texbufp->normalized = tex->normalized;
    texbufp->filterMode = tex->filterMode;
    texbufp->addressMode[0] = tex->addressMode[0];
    texbufp->addressMode[1] = tex->addressMode[1];
    texbufp->addressMode[2] = tex->addressMode[2];
    if (desc) {
        texbufp->x = desc->x;
        texbufp->y = desc->y;
        texbufp->z = desc->z;
        texbufp->w = desc->w;
        texbufp->f = desc->f;
    }
    else {
        texbufp->x = tex->channelDesc.x;
        texbufp->y = tex->channelDesc.y;
        texbufp->z = tex->channelDesc.z;
        texbufp->w = tex->channelDesc.w;
        texbufp->f = tex->channelDesc.f;
    }
}

cudaError_t
dscudaBindTextureWrapper(int *moduleid, char *texname,
                        size_t *offset,
                        const struct textureReference *tex,
                        const void *devPtr,
                        const struct cudaChannelFormatDesc *desc,
                        size_t size)
{
    cudaError_t err = cudaSuccess;
    dscudaBindTextureResult *rp;
    RCtexture texbuf;

    initClient();

    WARN(3, "dscudaBindTextureWrapper(0x%08llx, %s, 0x%08llx, 0x%08llx, 0x%08llx, 0x%08llx, %d)...",
         moduleid, texname,
         offset, tex, devPtr, desc, size);

    setTextureParams(&texbuf, tex, desc);

    Vdev_t *vdev = Vdev + Vdevid;
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudabindtextureid_1(moduleid[i], texname,
                                  (RCadr)devPtr, size, (RCtexture)texbuf, Clnt[Vdevid][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
    }

    if (offset) {
        *offset = rp->offset;
    }

    WARN(3, "done.\n");
    return err;
}

cudaError_t
dscudaBindTexture2DWrapper(int *moduleid, char *texname,
                          size_t *offset,
                          const struct textureReference *tex,
                          const void *devPtr,
                          const struct cudaChannelFormatDesc *desc,
                          size_t width, size_t height, size_t pitch)
{
    cudaError_t err = cudaSuccess;
    dscudaBindTexture2DResult *rp;
    RCtexture texbuf;

    initClient();

    WARN(3, "dscudaBindTexture2DWrapper(0x%08llx, %s, 0x%08llx, 0x%08llx, 0x%08llx, 0x%08llx, %d, %d, %d)...",
         moduleid, texname,
         offset, tex, devPtr, desc, width, height, pitch);

    setTextureParams(&texbuf, tex, desc);

    Vdev_t *vdev = Vdev + Vdevid;
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudabindtexture2did_1(moduleid[i], texname,
                                    (RCadr)devPtr, width, height, pitch, (RCtexture)texbuf, Clnt[Vdevid][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
    }

    if (offset) {
        *offset = rp->offset;
    }

    WARN(3, "done.\n");
    return err;
}

cudaError_t
dscudaBindTextureToArrayWrapper(int *moduleid, char *texname,
                               const struct textureReference *tex,
                               const struct cudaArray *array,
                               const struct cudaChannelFormatDesc *desc)
{
    cudaError_t err = cudaSuccess;
    dscudaResult *rp;
    RCtexture texbuf;
    RCcuarrayArray *ca;

    initClient();

    WARN(3, "dscudaBindTextureToArrayWrapper(0x%08llx, %s, 0x%08llx, 0x%08llx)...",
         moduleid, texname, (unsigned long)array, (unsigned long)desc);

    setTextureParams(&texbuf, tex, desc);

    ca = RCcuarrayArrayQuery((cudaArray *)array);
    if (!ca) {
        WARN(0, "invalid cudaArray : 0x%08llx\n", array);
        exit(1);
    }

    Vdev_t *vdev = Vdev + Vdevid;
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        rp = dscudabindtexturetoarrayid_1(moduleid[i], texname, (RCadr)ca->ap[i], (RCtexture)texbuf, Clnt[Vdevid][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
    }
    WARN(3, "done.\n");
    return err;
}

cudaError_t
cudaUnbindTexture(const struct textureReference * texref)
{
    cudaError_t err = cudaSuccess;

    WARN(4, "Current implementation of cudaUnbindTexture() does nothing "
         "but returning cudaSuccess.\n");

    err = cudaSuccess;

    return err;
}

int
dscudaRemoteCallType(void)
{
    int rctype = RC_REMOTECALL_TYPE_RPC;

    if (UseIbv) {
        rctype = RC_REMOTECALL_TYPE_IBV;
    }

    return rctype;
}
