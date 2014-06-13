#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <cuda.h>

#if CUDA_VERSION >= 5000
#include <helper_cuda_drvapi.h>
#else
#include <cutil.h>
// remove definition of some macros which will be redefined in \"cutil_inline.h\".

#ifdef MIN
#undef MIN
#endif
#ifdef MAX
#undef MAX
#endif
#include <cutil_inline.h>

#endif

#include <cufft.h>
#include <pthread.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <poll.h>
#include <errno.h>

#include "dscuda.h"
#include "sockutil.h"
#include "ibvdefs.h"
#include "tcpdefs.h"

typedef struct {
    int valid;
    unsigned int id;
    unsigned int ipaddr;
    unsigned int pid;
    time_t loaded_time;
    char name[256];
    CUmodule handle;
    CUfunction kfunc[RC_NKFUNCMAX]; // this is not used for now.
} Module;

static int D2Csock = -1; // socket for sideband communication to the client. inherited from the daemon.
static int TcpPort = RC_SERVER_IP_PORT;
static int Connected = 0;
static int UseIbv = 0; // use IB Verbs if set to 1. use socket by default.
static int Ndevice = 1;                 // # of devices in the system.
static int Devid[RC_NSERVERMAX] = {0,}; // real device ids of the ones in the system.
static int dscuDevice;                   // virtual device id of the one used in the current context.
static CUcontext dscuContext = NULL;
static int Devid2Vdevid[RC_NDEVICEMAX]; // device id conversion table from real to virtual.
static Module Modulelist[RC_NKMODULEMAX] = {0};
static int (*RCStub[RCMethodEnd])(RCHdr *, RCHdr *);

static void notifyIamReady(void);
static void showUsage(char *command);
static void showConf(void);
static void parseArgv(int argc, char **argv);
static cudaError_t initDscuda(void);
static cudaError_t createDscuContext(void);
static cudaError_t destroyDscuContext(void);
static void initEnv(void);
static void releaseModules(bool releaseall);
static CUresult getFunctionByName(CUfunction *kfuncp, char *kname, int moduleid);
static void getGlobalSymbol(int moduleid, char *symbolname, CUdeviceptr *dptr, size_t *size);
static int dscudaLoadModule(RCipaddr ipaddr, RCpid pid, char *mname, char *image);
static void * dscudaLaunchKernel(int moduleid, int kid, char *kname,
                                 RCdim3 gdim, RCdim3 bdim, RCsize smemsize, RCstream stream, int narg, RCArg *args);
static cudaError_t setTextureParams(CUtexref texref, RCtexture texbuf, char *texname, CUDA_ARRAY_DESCRIPTOR *descp = NULL);

#undef WARN
#define WARN(lv, fmt, args...) if (lv <= dscudaWarnLevel()) \
        fprintf(stderr, "dscudasvr[%d] : " fmt, dscudaMyServerId(), ## args);

#define SET_STUB(mthd) {                            \
        RCStub[RCMethod ## mthd] = RC ## mthd;        \
}

int
dscudaMyServerId(void)
{
    return TcpPort - RC_SERVER_IP_PORT;
}

#if 0
inline void
fatal_error(int exitcode)
{
    fprintf(stderr,
            "%s(%i) : fatal_error().\n"
            "Probably you need to restart dscudasvr.\n",
            __FILE__, __LINE__);
    exit(exitcode);
}

inline void
check_cuda_error(cudaError err)
{
    if (cudaSuccess != err) {
        fprintf(stderr,
                "%s(%i) : check_cuda_error() Runtime API error : %s.\n"
                "You may need to restart dscudasvr.\n",
                __FILE__, __LINE__, cudaGetErrorString(err));
    }
}
#else

#define fatal_error(exitcode)\
{\
    fprintf(stderr,\
            "%s(%i) : fatal_error().\n"\
            "Probably you need to restart dscudasvr.\n",\
            __FILE__, __LINE__);\
    exit(exitcode);\
}\

#define check_cuda_error(err)\
{\
    if (cudaSuccess != err) {\
        fprintf(stderr,\
                "%s(%i) : check_cuda_error() Runtime API error : %s.\n"\
                "You may need to restart dscudasvr.\n",\
                __FILE__, __LINE__, cudaGetErrorString(err));\
    }\
}
#endif


#define SETUP_PACKET_BUF(mthd)                            \
    RC ## mthd ## InvokeHdr *rpkt = (RC ## mthd ## InvokeHdr *)rpkt0; \
    RC ## mthd ## ReturnHdr *spkt = (RC ## mthd ## ReturnHdr *)spkt0; \
    int spktsize = sizeof(RC ## mthd ## ReturnHdr);                    \
    spkt->method = RCMethod ## mthd ;                                   \
    WARN(3, "cuda" #mthd "(");                                          \
    if (!dscuContext) createDscuContext();


#if TCP_ONLY
#define WAIT_READY_TO_KICKOFF(conn) /* nop */
#else
#define WAIT_READY_TO_KICKOFF(conn)             \
    if (UseIbv) {                               \
        rdmaWaitReadyToKickoff(conn);           \
    }                                           \
    else {                                      \
        /* nop */                               \
    }
#include "dscudasvr_ibv.cu"
#endif

#include "dscudasvr_tcp.cu"

/*
 * CUDA API stubs
 */

static int
RCMemcpyH2D(RCHdr *rpkt0, RCHdr *spkt0)
{
    cudaError_t err;
    SETUP_PACKET_BUF(MemcpyH2D);

    err = cudaMemcpy((void *)rpkt->dstadr, &rpkt->srcbuf, rpkt->count, cudaMemcpyHostToDevice);
    check_cuda_error(err);

    spkt->err = err;
    WARN(3, "0x%08llx, 0x%08lx, %d, %s) done.\n",
         rpkt->dstadr, (unsigned long)&rpkt->srcbuf, rpkt->count,
         dscudaMemcpyKindName(cudaMemcpyHostToDevice));

    WAIT_READY_TO_KICKOFF(IbvConn);
    return spktsize;
}

static int
RCMemcpyD2H(RCHdr *rpkt0, RCHdr *spkt0)
{
    cudaError_t err;
    SETUP_PACKET_BUF(MemcpyD2H);

    err = cudaMemcpy(&spkt->dstbuf, (void *)rpkt->srcadr, rpkt->count, cudaMemcpyDeviceToHost);
    check_cuda_error(err);
    spkt->err = err;
    WARN(3, "0x%08lx, 0x%08llx, %d, %s) done.\n",
         (unsigned long)&spkt->dstbuf, rpkt->srcadr, rpkt->count,
         dscudaMemcpyKindName(cudaMemcpyDeviceToHost));

    WAIT_READY_TO_KICKOFF(IbvConn);
    spktsize += rpkt->count;


#if 0 // !!! [debugging purpose only] destroy some part of the returning data
      // !!! in order to emulate a malfunctional GPU.
    {
        static int firstcall = 1;
        static int err_in_prev_call = 0; // avoid bad data generation in adjacent calls.
        if (firstcall) {
            firstcall = 0;
            srand48(time(NULL));
        }
        if (drand48() < 1.0/10.0 && !err_in_prev_call) {
            WARN(2, "################ bad data generatad.\n\n");
            ((int *)&spkt->dstbuf)[0] = 123;
            err_in_prev_call = 1;
        }
        else {
            err_in_prev_call = 0;
        }
    }
#endif

    return spktsize;
}

static int
RCMemcpyD2D(RCHdr *rpkt0, RCHdr *spkt0)
{
    cudaError_t err;
    SETUP_PACKET_BUF(MemcpyD2D);

    err = cudaMemcpy((void*)rpkt->dstadr, (void*)rpkt->srcadr, rpkt->count, cudaMemcpyDeviceToDevice);
    check_cuda_error(err);

    spkt->err = err;
    WARN(3, "0x%08llx, 0x%08lx, %d, %s) done.\n",
         rpkt->dstadr, rpkt->srcadr, rpkt->count,
         dscudaMemcpyKindName(cudaMemcpyDeviceToDevice));

    WAIT_READY_TO_KICKOFF(IbvConn);
    return spktsize;
}

static int
RCMalloc(RCHdr *rpkt0, RCHdr *spkt0)
{
    cudaError_t err;
    SETUP_PACKET_BUF(Malloc);

    err = cudaMalloc((void**)&spkt->devAdr, rpkt->size);
    check_cuda_error(err);
    spkt->err = err;
    WARN(3, "0x%08llx, %d) done. devadr:0x%08llx\n", &spkt->devAdr, rpkt->size, spkt->devAdr);

    WAIT_READY_TO_KICKOFF(IbvConn);
    return spktsize;
}

static int
RCFree(RCHdr *rpkt0, RCHdr *spkt0)
{
    cudaError_t err;
    SETUP_PACKET_BUF(Free);

    err = cudaFree((void*)rpkt->devAdr);
    check_cuda_error(err);
    spkt->err = err;
    WARN(3, "0x%08llx) done.\n", rpkt->devAdr);

    WAIT_READY_TO_KICKOFF(IbvConn);
    return spktsize;
}

static int
RCGetErrorString(RCHdr *rpkt0, RCHdr *spkt0)
{
    SETUP_PACKET_BUF(GetErrorString);
    int len;
    const char *str;

    str = cudaGetErrorString(rpkt->err);
    strncpy((char *)&spkt->errmsg, str, 256);
    len = strlen((char *)&spkt->errmsg);
    WARN(3, "%d) errmsg:%s  done.\n", rpkt->err, &spkt->errmsg);

    WAIT_READY_TO_KICKOFF(IbvConn);
    spktsize += len;
    return spktsize;
}

static int
RCGetDeviceProperties(RCHdr *rpkt0, RCHdr *spkt0)
{
    cudaError_t err;
    SETUP_PACKET_BUF(GetDeviceProperties);

    if (1 < Ndevice) {
        WARN(0, "ibvGetDeviceProperties() cannot handle multiple devices for now. Ndevice:%d\n",
             Ndevice);
        exit(1);
    }
    err = cudaGetDeviceProperties(&spkt->prop, Devid[0]);
    check_cuda_error(err);
    spkt->err = err;
    WARN(3, "0x%08lx, %d) done.\n", (unsigned long)&spkt->prop, rpkt->device);

    WAIT_READY_TO_KICKOFF(IbvConn);
    return spktsize;
}

static int
RCRuntimeGetVersion(RCHdr *rpkt0, RCHdr *spkt0)
{
    cudaError_t err;
    SETUP_PACKET_BUF(RuntimeGetVersion);

    err = cudaRuntimeGetVersion(&spkt->version);
    check_cuda_error(err);
    spkt->err = err;
    WARN(3, "0x%08lx) done.\n", (unsigned long)&spkt->version);

    WAIT_READY_TO_KICKOFF(IbvConn);
    return spktsize;
}

static int
RCThreadSynchronize(RCHdr *rpkt0, RCHdr *spkt0)
{
    cudaError_t err;
    SETUP_PACKET_BUF(ThreadSynchronize);

    err = cudaThreadSynchronize();
    check_cuda_error(err);
    spkt->err = err;
    WARN(3, "done.\n");

    WAIT_READY_TO_KICKOFF(IbvConn);
    return spktsize;
}

static int
RCThreadExit(RCHdr *rpkt0, RCHdr *spkt0)
{
    cudaError_t err;
    SETUP_PACKET_BUF(ThreadExit);

    err = cudaThreadExit();
    check_cuda_error(err);
    spkt->err = err;
    WARN(3, "done.\n");

    WAIT_READY_TO_KICKOFF(IbvConn);
    return spktsize;
}

static int
RCDeviceSynchronize(RCHdr *rpkt0, RCHdr *spkt0)
{
    cudaError_t err;
    SETUP_PACKET_BUF(DeviceSynchronize);

    err = cudaDeviceSynchronize();
    check_cuda_error(err);
    spkt->err = err;
    WARN(3, "done.\n");

    WAIT_READY_TO_KICKOFF(IbvConn);
    return spktsize;
}


static int
RCDscudaMemcpyToSymbolH2D(RCHdr *rpkt0, RCHdr *spkt0)
{
#warning fill this part in ibvDscudaMemcpyToSymbolH2D

    SETUP_PACKET_BUF(DscudaMemcpyToSymbolH2D);

#if 0
    RCbuf srcbuf;
    srcbuf.RCbuf_len = rpkt->count;
    srcbuf.RCbuf_val = (char *)&rpkt->src;

    dscudaResult *resp = dscudamemcpytosymbolh2did_1_svc(rpkt->moduleid,
                                                            rpkt->symbol,
                                                            srcbuf,
                                                            rpkt->count,
                                                            rpkt->offset,
                                                            NULL);
    spkt->err = (cudaError_t)cudaSuccess;

    WAIT_READY_TO_KICKOFF(IbvConn);

#endif

    return spktsize;
}


static int
RCDscudaMemcpyToSymbolD2D(RCHdr *rpkt0, RCHdr *spkt0)
{
#warning fill this part in ibvDscudaMemcpyToSymbolD2D

    SETUP_PACKET_BUF(DscudaMemcpyToSymbolD2D);

#if 0
    dscudaResult *resp = dscudamemcpytosymbold2did_1_svc(rpkt->moduleid,
                                                            rpkt->symbol,
                                                            rpkt->srcadr,
                                                            rpkt->count,
                                                            rpkt->offset,
                                                            NULL);
    spkt->err = (cudaError_t)cudaSuccess;

    WAIT_READY_TO_KICKOFF(IbvConn);
#endif

    return spktsize;
}


static int
RCDscudaMemcpyFromSymbolD2H(RCHdr *rpkt0, RCHdr *spkt0)
{
#warning fill this part in ibvDscudaMemcpyFromSymbolD2H

    SETUP_PACKET_BUF(DscudaMemcpyFromSymbolD2H);

#if 0
    dscudaMemcpyFromSymbolD2HResult *resp = dscudamemcpyfromsymbold2hid_1_svc(rpkt->moduleid,
                                                                                      rpkt->symbol,
                                                                                      rpkt->count,
                                                                                      rpkt->offset,
                                                                                      NULL);
    spkt->err = (cudaError_t)cudaSuccess;

    WAIT_READY_TO_KICKOFF(IbvConn);
    spktsize += rpkt->count;
    memcpy(&spkt->dst, resp->buf.RCbuf_val, rpkt->count);
#endif

    return spktsize;
}

static int
RCDscudaMemcpyFromSymbolD2D(RCHdr *rpkt0, RCHdr *spkt0)
{
#warning fill this part in ibvDscudaMemcpyFromSymbolD2D

    SETUP_PACKET_BUF(DscudaMemcpyFromSymbolD2D);

#if 0
    dscudaResult *resp = dscudamemcpyfromsymbold2did_1_svc(rpkt->moduleid,
                                                            rpkt->dstadr,
                                                            rpkt->symbol,
                                                            rpkt->count,
                                                            rpkt->offset,
                                                            NULL);
    spkt->err = (cudaError_t)cudaSuccess;

    WAIT_READY_TO_KICKOFF(IbvConn);

#endif

    return spktsize;
}

static int
RCDscudaMemcpyToSymbolAsyncH2D(RCHdr *rpkt0, RCHdr *spkt0)
{
#warning fill this part in ibvDscudaMemcpyToSymbolAsyncH2D

    SETUP_PACKET_BUF(DscudaMemcpyToSymbolAsyncH2D);

#if 0
    RCbuf srcbuf;
    srcbuf.RCbuf_len = rpkt->count;
    srcbuf.RCbuf_val = (char *)&rpkt->src;
    dscudaResult *resp = dscudamemcpytosymbolasynch2did_1_svc(rpkt->moduleid,
                                                            rpkt->symbol,
                                                            srcbuf,
                                                            rpkt->count,
                                                            rpkt->offset,
                                                            rpkt->stream,
                                                            NULL);
    spkt->err = (cudaError_t)cudaSuccess;

    WAIT_READY_TO_KICKOFF(IbvConn);

#endif

    return spktsize;
}


static int
RCDscudaMemcpyToSymbolAsyncD2D(RCHdr *rpkt0, RCHdr *spkt0)
{
#warning fill this part in ibvDscudaMemcpyToSymbolAsyncD2D

    SETUP_PACKET_BUF(DscudaMemcpyToSymbolAsyncD2D);

#if 0
    dscudaResult *resp = dscudamemcpytosymbolasyncd2did_1_svc(rpkt->moduleid,
                                                            rpkt->symbol,
                                                            rpkt->srcadr,
                                                            rpkt->count,
                                                            rpkt->offset,
                                                            rpkt->stream,
                                                            NULL);
    spkt->err = (cudaError_t)cudaSuccess;

    WAIT_READY_TO_KICKOFF(IbvConn);

#endif

    return spktsize;
}


static int
RCDscudaMemcpyFromSymbolAsyncD2H(RCHdr *rpkt0, RCHdr *spkt0)
{
#warning fill this part in ibvDscudaMemcpyFromSymbolAsyncD2H

    SETUP_PACKET_BUF(DscudaMemcpyFromSymbolAsyncD2H);

#if 0
    dscudaMemcpyFromSymbolAsyncD2HResult *resp = dscudamemcpyfromsymbolasyncd2hid_1_svc(rpkt->moduleid,
                                                                                      rpkt->symbol,
                                                                                      rpkt->count,
                                                                                      rpkt->offset,
                                                                                      rpkt->stream,
                                                                                      NULL);
    spkt->err = (cudaError_t)cudaSuccess;

    WAIT_READY_TO_KICKOFF(IbvConn);
    spktsize += rpkt->count;
    memcpy(&spkt->dst, resp->buf.RCbuf_val, rpkt->count);

#endif

    return spktsize;
}

static int
RCDscudaMemcpyFromSymbolAsyncD2D(RCHdr *rpkt0, RCHdr *spkt0)
{
#warning fill this part in ibvDscudaMemcpyFromSymbolAsyncD2D

    SETUP_PACKET_BUF(DscudaMemcpyFromSymbolAsyncD2D);

#if 0
    dscudaResult *resp = dscudamemcpyfromsymbolasyncd2did_1_svc(rpkt->moduleid,
                                                            rpkt->dstadr,
                                                            rpkt->symbol,
                                                            rpkt->count,
                                                            rpkt->offset,
                                                            rpkt->stream,
                                                            NULL);
    spkt->err = (cudaError_t)cudaSuccess;

    WAIT_READY_TO_KICKOFF(IbvConn);

#endif

    return spktsize;
}

static int
RCDscudaLoadModule(RCHdr *rpkt0, RCHdr *spkt0)
{
    SETUP_PACKET_BUF(DscudaLoadModule);
    spkt->moduleid = dscudaLoadModule((RCipaddr)rpkt->ipaddr,
                                      (RCpid)rpkt->pid,
                                      rpkt->modulename,
                                      (char *)&rpkt->moduleimage);
    spkt->err = (cudaError_t)cudaSuccess;

    WAIT_READY_TO_KICKOFF(IbvConn);
    return spktsize;
}

static int
RCDscudaLaunchKernel(RCHdr *rpkt0, RCHdr *spkt0)
{
#warning fill this part in ibvDscudaLaunchKernel

    SETUP_PACKET_BUF(DscudaLaunchKernel);

    RCdim3 gdim, bdim;

    gdim.x = rpkt->gdim[0];
    gdim.y = rpkt->gdim[1];
    gdim.z = rpkt->gdim[2];

    bdim.x = rpkt->bdim[0];
    bdim.y = rpkt->bdim[1];
    bdim.z = rpkt->bdim[2];

    dscudaLaunchKernel(rpkt->moduleid, rpkt->kernelid, rpkt->kernelname,
                       gdim, bdim, rpkt->smemsize, rpkt->stream,
                       rpkt->narg, (RCArg *)&rpkt->args);

    spkt->err = (cudaError_t)cudaSuccess;

    WAIT_READY_TO_KICKOFF(IbvConn);

    return spktsize;
}

static void
setupStub(void)
{
    int i;
    memset(RCStub, 0, sizeof(RCMethod) * RCMethodEnd);
    SET_STUB(MemcpyH2D);
    SET_STUB(MemcpyD2H);
    SET_STUB(MemcpyD2D);
    SET_STUB(Malloc);
    SET_STUB(Free);
    SET_STUB(GetErrorString);
    SET_STUB(GetDeviceProperties);
    SET_STUB(RuntimeGetVersion);
    SET_STUB(ThreadSynchronize);
    SET_STUB(ThreadExit);
    SET_STUB(DeviceSynchronize);
    SET_STUB(DscudaMemcpyToSymbolH2D);
    SET_STUB(DscudaMemcpyToSymbolD2D);
    SET_STUB(DscudaMemcpyFromSymbolD2H);
    SET_STUB(DscudaMemcpyFromSymbolD2D);
    SET_STUB(DscudaMemcpyToSymbolAsyncH2D);
    SET_STUB(DscudaMemcpyToSymbolAsyncD2D);
    SET_STUB(DscudaMemcpyFromSymbolAsyncD2H);
    SET_STUB(DscudaMemcpyFromSymbolAsyncD2D);
    SET_STUB(DscudaLoadModule);
    SET_STUB(DscudaLaunchKernel);
    SET_STUB(DscudaSendP2P);
    SET_STUB(DscudaRecvP2P);
    for (i = 1; i < RCMethodEnd; i++) {
        if (RCStub[i]) continue;
        WARN(0, "setupStub: RCStub[%d] is not initialized.\n", i);
        exit(1);
    }
}

static void
notifyIamReady(void)
{
    char msg[] = "ready";
    if (D2Csock >= 0) {
        WARN(3, "send \"ready\" to the client.\n");
        sendMsgBySocket(D2Csock, msg);
    }
}

static int
receiveProtocolPreference(void)
{
    char msg[256], rc[64];

    if (D2Csock >= 0) {
        WARN(3, "wait for remotecall preference (\"sock\" or \"ibv\") from the client.\n");
        recvMsgBySocket(D2Csock, msg, sizeof(msg));
        sscanf(msg, "remotecall:%s", rc);
        WARN(2, "method of remote procedure call: %s\n", rc);
        if (!strncmp("ibv", rc, strlen("ibv"))) {
            return 1;
        }
        else {
            return 0;
        }
    }
    else {
        return UseIbv; // do not modify the preference.
    }
}

int
main(int argc, char **argv)
{
    parseArgv(argc, argv);
    initEnv();
    initDscuda();
    showConf();

    UseIbv = receiveProtocolPreference();

    setupStub();
    if (UseIbv) {
#if !TCP_ONLY
        notifyIamReady();
        ibvMainLoop(NULL);
#endif
    }
    else {
        notifyIamReady();
        tcpMainLoop();
    }
    fprintf (stderr, "main loop returned.\n"); // never reached.
    exit (1);
}

static void
showUsage(char *command)
{
    fprintf(stderr,
            "usage: %s [-s server_id] [-d 'deviceid'] [-p port] [-S socket]\n"
            "       (-p & -S are used by the daemon only.)\n",
            command);
}

static void
showConf(void)
{
    int i;
    char str[1024], str0[1024];

    WARN(2, "TCP port : %d (base + %d)\n", TcpPort, TcpPort - RC_SERVER_IP_PORT);
    WARN(2, "ndevice : %d\n", Ndevice);
    sprintf(str, "real device%s      :", Ndevice > 1 ? "s" : " ");
    for (i = 0; i < Ndevice; i++) {
        sprintf(str0, " %d", Devid[i]);
        strcat(str, str0);
    }
    WARN(2, "%s\n", str);
    sprintf(str, "virtual device%s   :", Ndevice > 1 ? "s" : " ");
    for (i = 0; i < Ndevice; i++) {
        sprintf(str0, " %d", Devid2Vdevid[Devid[i]]);
        strcat(str, str0);
    }
    WARN(2, "%s\n", str);
}

extern char *optarg;
extern int optind;
static void
parseArgv(int argc, char **argv)
{
    int c, ic;
    char *param = "d:hp:s:S:";
    char *num;
    char buf[256];
    int device_used[RC_NDEVICEMAX] = {0,};
    int tcpport_set = 0;
    int serverid = 0;

    while ((c = getopt(argc, argv, param)) != EOF) {
        switch (c) {
          case 'p':
            TcpPort = atoi(optarg);
            tcpport_set = 1;
            break;

          case 's':
            serverid = atoi(optarg);
            break;

          case 'S':
            D2Csock = atoi(optarg);
            break;

          case 'd':
            Ndevice = 0;
            strncpy(buf, optarg, sizeof(buf));
            num = strtok(buf, " ");
            while (num) {
                ic = atoi(num);
                if (ic < 0 || RC_NDEVICEMAX <= ic ) {
                    fprintf(stderr, "device id out of range: %d\n", ic);
                    exit(2);
                }
                if (!device_used[ic]) { // care for malformed optarg value such as "0 1 2 2 3".
                    device_used[ic] = 1;
                    Devid[Ndevice] = ic;
                    Ndevice++;
                }
                num = strtok(NULL, " ");
            }
            break;

          case 'h':
          default:
            showUsage(argv[0]);
            exit(1);
        }
    }
    if (!tcpport_set) {
        TcpPort = RC_SERVER_IP_PORT + serverid;
        WARN(3, "TCP port number not given by '-p' option. Use default (%d).\n", TcpPort);
    }
}

// should be called only once in a run.
static cudaError_t
initDscuda(void)
{
    int i;
    unsigned int flags = 0; // should always be 0.
    CUresult err;

    WARN(4, "initDscuda...\n");

    for (i = 0; i < Ndevice; i++) {
        Devid2Vdevid[Devid[i]] = i;
    }

    err = cuInit(flags);
    if (err != CUDA_SUCCESS) {
        WARN(0, "cuInit(%d) failed.\n", flags);
        exit(1);
    }
    err = (CUresult)cudaSetValidDevices(Devid, Ndevice);
    if (err != CUDA_SUCCESS) {
        WARN(0, "cudaSetValidDevices(0x%08llx, %d) failed.\n", Devid, Ndevice);
        exit(1);
    }
    dscuDevice = Devid[0];
    WARN(3, "cudaSetValidDevices(0x%08llx, %d). dscuDevice:%d\n",
         Devid, Ndevice, dscuDevice);
    WARN(4, "initDscuda done.\n");
    return (cudaError_t)err;
}

static cudaError_t
createDscuContext(void)
{
    //    unsigned int flags = 0; // should always be 0.
    CUdevice dev = 0;
    CUresult err;

    err = cuDeviceGet(&dev, dscuDevice);
    if (err != CUDA_SUCCESS) {
        WARN(0, "cuDeviceGet() failed.\n");
        return (cudaError_t)err;
    }

#if 0
    err = cuCtxCreate(&dscuContext, flags, dev);
    if (err != CUDA_SUCCESS) {
        WARN(0, "cuCtxCreate() failed.\n");
        return (cudaError_t)err;
    }
#else // not used. set a dummy value not to be called repeatedly.
    dscuContext = (CUcontext)-1;
#endif

    return (cudaError_t)err;
}

static cudaError_t
destroyDscuContext(void)
{
#if 0

    CUresult cuerr;
    bool all = true;

    WARN(3, "destroyDscuContext(");
    releaseModules(all);

    cuerr = cuCtxDestroy(dscuContext);
    WARN(4, "cuCtxDestroy(0x%08llx", dscuContext);
    if (cuerr != CUDA_SUCCESS) {
        WARN(0, "cuCtxDestroy() failed.\n");
        fatal_error(1);
        return (cudaError_t)cuerr;
    }
    dscuContext = NULL;
    WARN(4, ") done.\n");
    WARN(3, ") done.\n");

#else

    dscuContext = NULL;

#endif
    return cudaSuccess;
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
            dscudaSetWarnLevel(tmp);
        }
        WARN(1, "WarnLevel: %d\n", dscudaWarnLevel());
    }

    // DSCUDA_REMOTECALL
    env = getenv("DSCUDA_REMOTECALL");
#if TCP_ONLY
    UseIbv = 0;
    WARN(2, "method of remote procedure call: SOCK\n");
#else
    if (D2Csock >= 0) { // launched by daemon.
        WARN(3, "A server launched by the daemon "
             "does not use the evironment variable 'DSCUDA_REMOTECALL'.\n");
    }
    else { // launched by hand.
        if (!env) {
            fprintf(stderr, "Set an environment variable 'DSCUDA_REMOTECALL' to 'ibv' or 'sock'.\n");
            exit(1);
        }
        if (!strcmp(env, "ibv")) {
            UseIbv = 1;
            WARN(2, "method of remote procedure call: InfiniBand Verbs\n");
        }
        else if (!strcmp(env, "tcp")) {
            UseIbv = 0;
            WARN(2, "method of remote procedure call: TCP\n");
        }
        else {
            UseIbv = 0;
            WARN(2, "method of remote procedure call '%s' is not available. use TCP.\n", env);
        }
    }
#endif
}

/*
 * Unload Modules never been used for a long time.
 */
static void
releaseModules(bool releaseall = false)
{
    Module *mp;
    int i;

    for (i = 0, mp = Modulelist; i < RC_NKMODULEMAX; i++, mp++) {
        if (!mp->valid) continue;
        if (releaseall || time(NULL) - mp->loaded_time > RC_SERVER_CACHE_LIFETIME) {
            cuModuleUnload((CUmodule)mp->handle);
            mp->valid = 0;
            mp->handle = NULL;
            for (i = 0; i < RC_NKFUNCMAX; i++) {
                mp->kfunc[i] = NULL;
            }
            WARN(3, "releaseModules() unloaded a module. name:%s pid:%d ip:%s age:%d\n",
                 mp->name, mp->pid, dscudaAddrToServerIpStr(mp->ipaddr),
                 time(NULL) - mp->loaded_time);
        }
    }
}

static CUresult
getFunctionByName(CUfunction *kfuncp, char *kname, int moduleid)
{
    CUresult cuerr;
    Module *mp = Modulelist + moduleid;

    cuerr = cuModuleGetFunction(kfuncp, mp->handle, kname);
    if (cuerr == CUDA_SUCCESS) {
        WARN(3, "cuModuleGetFunction() : function '%s' found.\n", kname);
    }
    else {
        WARN(0, "cuModuleGetFunction() : function:'%s'. %s\n",
             kname, cudaGetErrorString((cudaError_t)cuerr));
	WARN(0, "moduleid:%d module valid:%d id:%d name:%s\n",
	     moduleid, mp->valid, mp->id, mp->name);
        fatal_error(1);
    }
    return cuerr;
}

static void
getGlobalSymbol(int moduleid, char *symbolname, CUdeviceptr *dptr, size_t *size)
{
    CUresult cuerr;
    Module *mp;

    if (moduleid < 0 || RC_NKMODULEMAX <= moduleid) {
        WARN(0, "getGlobalSymbol() : invalid module id:%d.\n", moduleid);
        fatal_error(1);
    }
    mp = Modulelist + moduleid;
    cuerr = cuModuleGetGlobal(dptr, size, mp->handle, symbolname);
    if (cuerr == CUDA_SUCCESS) {
    WARN(3, "cuModuleGetGlobal(0x%08lx, 0x%08lx, 0x%08lx, %s) done."
	 " modulename:%s  symbolname:%s  *dptr:0x%08lx\n",
	 dptr, size, mp->handle, symbolname,
	 mp->name, symbolname, *dptr);
    }
    else {
        WARN(0, "cuModuleGetGlobal(0x%08llx, 0x%08llx, 0x%08llx, 0x%08llx) failed."
             " modulename:%s  symbolname:%s  %s\n",
             dptr, size, mp->handle, symbolname,
             mp->name, symbolname, cudaGetErrorString((cudaError_t)cuerr));
        fatal_error(1);
    }
}

static int
dscudaLoadModule(RCipaddr ipaddr, RCpid pid, char *mname, char *image)
{
    CUresult cuerr;
    Module   *mp;
    int      i;

#if RC_CACHE_MODULE
    // look for mname in the module list, which may found if the client
    // resent multiple requests for the same mname:pid:ipaddr.
    int found = 0;
    for (i = 0, mp = Modulelist; i < RC_NKMODULEMAX; i++, mp++) {
        if (!mp->valid) continue;
        if ((unsigned int)ipaddr == mp->ipaddr &&
            pid    == mp->pid &&
            !strcmp(mname, mp->name)) {
            found = 1;
            break;
        }
	WARN(4, "ip:%x  %x    pid:%d  %d    name:%s  %s\n",
	     (unsigned int)ipaddr, mp->ipaddr, pid, mp->pid, mname, mp->name);
    }

    if (found) { // module found. i.e, it's already loaded.
        WARN(3, "\n\n------------------------------------------------------------------\n"
             "dscudaloadmoduleid_1_svc() got multiple requests for\n"
             "  the same module name : %s,\n"
             "  the same process id  : %d, and\n"
             "  the same IP address  : %s,\n"
             "which means a client resent the same module twice or more.\n"
             "If you see this message too often, you may want to increase\n"
             "$dscuda/include/dscudadefs.h:RC_CLIENT_CACHE_LIFETIME\n"
             "for better performance.\n"
             "------------------------------------------------------------------\n\n",
             mname, pid, dscudaAddrToServerIpStr(ipaddr));
        WARN(3, "cuModuleLoadData() : a module found in the cache. id:%d  name:%s  age:%d\n",
             mp->id, mname, time(NULL) - mp->loaded_time);
    }
    else  // module not found in the cache. load it from image.
#endif // RC_CACHE_MODULE

    {
        for (i = 0, mp = Modulelist; i < RC_NKMODULEMAX; i++, mp++) {
            if (!mp->valid) break;
            if (i == RC_NKMODULEMAX) {
                WARN(0, "module cache is full.\n");
                fatal_error(1);
            }
        }
        mp->id = i;
        cuerr = cuModuleLoadData(&mp->handle, image);
        if (cuerr != CUDA_SUCCESS) {
            WARN(0, "cuModuleLoadData() failed. %s\n", cudaGetErrorString((cudaError_t)cuerr));
            fatal_error(1);
        }
        mp->valid = 1;
        mp->ipaddr = ipaddr;
        mp->pid = pid;
        strncpy(mp->name, mname, sizeof(Modulelist[0].name));
        for (i = 0; i < RC_NKFUNCMAX; i++) {
            mp->kfunc[i] = NULL;
        }
        WARN(3, "cuModuleLoadData() : a module loaded. id:%d  name:%s\n", mp->id, mname);
    }
    mp->loaded_time = time(NULL); // (re)set the lifetime of the cache.
    releaseModules();
    return mp->id;
}

static void *
dscudaLaunchKernel(int moduleid, int kid, char *kname,
                   RCdim3 gdim, RCdim3 bdim, RCsize smemsize, RCstream stream,
                   int narg, RCArg *args)
                   
{
    static int dummyres = 123;
    int paramsize;
    CUresult cuerr;

#if !RC_SUPPORT_CONCURRENT_EXEC
    stream = 0;
#endif

    if (!dscuContext) createDscuContext();

    // load a kernel function into Module[moduleid].kfunc[kid]
    // form Module[moduleid].handle.
    if (moduleid < 0 || RC_NKMODULEMAX <= moduleid) {
        WARN(0, "dscudalaunchkernelid_1_svc() : invalid module id:%d.\n", moduleid);
        fatal_error(1);
    }

#if 1 // look inside a module for a function by name.
    CUfunction kfunc;
    getFunctionByName(&kfunc, kname, moduleid);
#else // look for a function by its ID.
    // this is faster, but not used since it would cause a problem
    // when called from a kernel function that uses C++ template.
    // in that case kid might not be unique for each instance of the template.
    Module *mp = Modulelist + moduleid;
    CUfunction kfunc = mp->kfunc[kid];
    if (!kfunc) {
        getFunctionByName(&kfunc, kname, moduleid);
        mp->kfunc[kid] = kfunc;
    }
#endif

    // a kernel function found.
    // now make it run.
    paramsize = ibvUnpackKernelParam(&kfunc, narg, (RCArg *)args);
    cuerr = cuParamSetSize(kfunc, paramsize);
    if (cuerr != CUDA_SUCCESS) {
        WARN(0, "cuParamSetSize() failed. size:%d %s\n",
             paramsize, cudaGetErrorString((cudaError_t)cuerr));
        fatal_error(1);
    }
    WARN(5, "cuParamSetSize() done.\n");

    cuerr = cuFuncSetBlockShape(kfunc, bdim.x, bdim.y, bdim.z);
    if (cuerr != CUDA_SUCCESS) {
        WARN(0, "cuFuncSetBlockShape() failed. %s\n", cudaGetErrorString((cudaError_t)cuerr));
        fatal_error(1);
    }
    WARN(5, "cuFuncSetBlockShape() done.\n");

    if (smemsize != 0) {
        cuerr = cuFuncSetSharedSize(kfunc, smemsize);
        if (cuerr != CUDA_SUCCESS) {
            WARN(0, "cuFuncSetSharedSize() failed. %s\n", cudaGetErrorString((cudaError_t)cuerr));
            fatal_error(1);
        }
        WARN(5, "cuFuncSetSharedSize() done.\n");
    }

    if ((cudaStream_t)stream == NULL) {
        cuerr = cuLaunchGrid(kfunc, gdim.x, gdim.y);
        if (cuerr != CUDA_SUCCESS) {
            WARN(0, "cuLaunchGrid() failed. kname:%s %s\n",
                 kname, cudaGetErrorString((cudaError_t)cuerr));
            fatal_error(1);
        }
        WARN(3, "cuLaunchGrid() done. kname:%s\n", kname);
    }
    else {
        cuerr = cuLaunchGridAsync(kfunc, gdim.x, gdim.y, (cudaStream_t)stream);
        if (cuerr != CUDA_SUCCESS) {
            WARN(0, "cuLaunchGridAsync() failed. kname:%s  %s\n",
                 kname, cudaGetErrorString((cudaError_t)cuerr));
            fatal_error(1);
        }
        WARN(3, "cuLaunchGridAsync() done.  kname:%s  stream:0x%08llx\n", kname, stream);
    }

    return &dummyres; // seems necessary to return something even if it's not used by the client.
}

static cudaError_t
setTextureParams(CUtexref texref, RCtexture texbuf, char *texname, CUDA_ARRAY_DESCRIPTOR *descp)
{
    cudaError_t err;
    int ncomponent, i;
    unsigned int texref_flags = 0;
    unsigned int fmt_high, fmt_low, fmt_index;

    CUarray_format fmt[] = {
        CU_AD_FORMAT_UNSIGNED_INT8,    // 00_00
        CU_AD_FORMAT_UNSIGNED_INT16,   // 00_01
        CU_AD_FORMAT_UNSIGNED_INT32,   // 00_10
        (CUarray_format)0,
        CU_AD_FORMAT_SIGNED_INT8,      // 01_00
        CU_AD_FORMAT_SIGNED_INT16,     // 01_01
        CU_AD_FORMAT_SIGNED_INT32,     // 01_10
        (CUarray_format)0,
        (CUarray_format)0,
        CU_AD_FORMAT_HALF,             // 10_01
        (CUarray_format)0,
        (CUarray_format)0,
        (CUarray_format)0,
        (CUarray_format)0,
        CU_AD_FORMAT_FLOAT,            // 11_10
        (CUarray_format)0,
    };

    // set addressmode (wrap/clamp/mirror/border)
    //
    for (i = 0; i < 3; i++) {
        err = (cudaError_t)cuTexRefSetAddressMode(texref, i, (CUaddress_mode_enum)texbuf.addressMode[i]);
        if (err != cudaSuccess) {
            check_cuda_error(err);
            return err;
        }
    }

    // set filtermode (point/linear)
    //
    err = (cudaError_t)cuTexRefSetFilterMode(texref, (CUfilter_mode_enum)texbuf.filterMode);
    if (err != cudaSuccess) {
        check_cuda_error(err);
        return err;
    }

    // set flags (integer/normalized)
    //
    if (texbuf.normalized) {
        texref_flags |= CU_TRSF_NORMALIZED_COORDINATES;
    }
    else {
        texref_flags |= CU_TRSF_READ_AS_INTEGER;
    }
    err = (cudaError_t)cuTexRefSetFlags(texref, texref_flags);
    if (err != cudaSuccess) {
        check_cuda_error(err);
        return err;
    }

    // set format (unsigned/signed/float, 32/16/8-bit)
    //
    switch (texbuf.x) {
      case 8:
        fmt_low = 0;
        break;
      case 16:
        fmt_low = 1;
        break;
      case 32:
        fmt_low = 2;
        break;
      default:
        WARN(0, "cuModuleGetTexRef() invalid channel format. texture name:%s descriptor.x:%d\n",
             texname, texbuf.x);
        err = cudaErrorInvalidValue;
        return err;
    }
    switch (texbuf.f) {
      case cudaChannelFormatKindUnsigned:
        fmt_high = 0;
        break;

      case cudaChannelFormatKindSigned:
        fmt_high = 1;
        break;

      case cudaChannelFormatKindFloat:
        fmt_high = 3;
        break;

      case cudaChannelFormatKindNone:
        WARN(0, "cuModuleGetTexRef() invalid channel format. texture name:%s descriptor.f:%s\n",
             texname, "cudaChannelFormatKindNone");
        err = cudaErrorInvalidValue;
        return err;

      default:
        WARN(0, "cuModuleGetTexRef() invalid channel format. texture name:%s descriptor.f:%s\n",
             texname, texbuf.f);
        err = cudaErrorInvalidValue;
        return err;
    }
    fmt_index = fmt_high << 2 | fmt_low;
    ncomponent = 1;
    if (texbuf.y) ncomponent = 2;
    if (texbuf.z) ncomponent = 3;
    if (texbuf.w) ncomponent = 4;
    if (descp) {
        descp->Format = fmt[fmt_index];
        descp->NumChannels = ncomponent;
    }
    WARN(4, "cuTexRefSetFormat(0x%08llx, %d, %d)\n", texref, fmt[fmt_index], ncomponent);
    err = (cudaError_t)cuTexRefSetFormat(texref, fmt[fmt_index], ncomponent);
    if (err != cudaSuccess) {
        check_cuda_error(err);
        return err;
    }

    return cudaSuccess;
}
