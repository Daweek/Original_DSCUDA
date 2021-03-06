#include <stdio.h>
#include <string.h>
#include <netdb.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <net/if.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>
#include <cufft.h>
#include <cublas_v2.h>
#include <pthread.h>
#include <libgen.h>
#include "sockutil.h"
#include "dscuda.h"
#include "dscudaverb.h"

#ifdef _LIBDSCUDA_IBV_CU
#include "ibvdefs.h"
#endif

#ifdef _LIBDSCUDA_TCP_CU
#include "tcpdefs.h"
#endif


typedef struct {
    int id;   // index for each redundant server.
    int cid;  // id of a server given by -s option to dscudasvr.
              // clients specify the server using this num preceded
              // by an IP address & colon, e.g.,
              // export DSCUDA_SERVER="192.168.1.123:2"
    int port;
    unsigned int ip;
} RCServer_t;

typedef struct {
    int nredundancy;
    RCServer_t server[RC_NREDUNDANCYMAX];
} Vdev_t;

typedef struct {
    int valid;
    int vdevid;  // the virtual device the module is loaded into.
    int id[RC_NREDUNDANCYMAX]; // id assigned to this module in each real device that consists of the virtual one.
    char name[RC_KMODULENAMELEN];
    time_t sent_time;
} Module;

static Module Modulelist[RC_NKMODULEMAX] = {0};

typedef struct RCmappedMem_t {
    void *pHost;
    void *pDevice;
    int   size;
    RCmappedMem_t *prev;
    RCmappedMem_t *next;
} RCmappedMem;

typedef struct RCstreamArray_t {
    cudaStream_t s[RC_NREDUNDANCYMAX];
    RCstreamArray_t *prev;
    RCstreamArray_t *next;
} RCstreamArray;

typedef struct RCeventArray_t {
    cudaEvent_t e[RC_NREDUNDANCYMAX];
    RCeventArray_t *prev;
    RCeventArray_t *next;
} RCeventArray;

typedef struct RCcuarrayArray_t {
    cudaArray *ap[RC_NREDUNDANCYMAX];
    RCcuarrayArray_t *prev;
    RCcuarrayArray_t *next;
} RCcuarrayArray;

typedef struct RCuva_t {
    void    *kadr; // address seen from the host.
    void    *adr[RC_NREDUNDANCYMAX];
    int      devid;
    int      size;
    char    *snapshot; // image saved at the checkpoint.
    RCuva_t *prev;
    RCuva_t *next;
} RCuva;

typedef struct RCp2pconn_t {
    int            ddevid;
    int            sdevid;
    struct ibv_pd *send_pd[RC_NREDUNDANCYMAX];
    struct ibv_pd *recv_pd[RC_NREDUNDANCYMAX];
    uint16_t       ports[RC_NREDUNDANCYMAX]; // IP port[0..nredundancy] for P2P communication.
    RCp2pconn_t   *prev;
    RCp2pconn_t   *next;
} RCp2pconn;

typedef struct RCmr_t {
    void    *adr[RC_NREDUNDANCYMAX];
    int      ddevid;
    int      sdevid;
    int      is_send;
    int      lkey[RC_NREDUNDANCYMAX];
    int      rkey[RC_NREDUNDANCYMAX];
    RCmr_t  *prev;
    RCmr_t  *next;
} RCmr;

static int requestDaemonForDevice(unsigned int ipaddr, int devid, int useibv);
static int vdevidIndex(void);

static void RCmappedMemRegister(void *pHost, void *pDevice, size_t size);
static void RCmappedMemUnregister(void *pHost);
static RCmappedMem *RCmappedMemQuery(void *pHost);

static void RCstreamArrayRegister(cudaStream_t *streams);
static void RCstreamArrayUnregister(cudaStream_t stream0);
static RCstreamArray *RCstreamArrayQuery(cudaStream_t stream0);

static void RCeventArrayRegister(cudaEvent_t *events);
static void RCeventArrayUnregister(cudaEvent_t event0);
static RCeventArray *RCeventArrayQuery(cudaEvent_t event0);

static void RCcuarrayArrayRegister(cudaArray **cuarrays);
static void RCcuarrayArrayUnregister(cudaArray *cuarray0);
static RCcuarrayArray *RCcuarrayArrayQuery(cudaArray *cuarray0);

static RCuva *RCuvaRegister(int devid, void *adr[], size_t size);
static void RCuvaUnregister(void *adr);
static RCuva *RCuvaQuery(void *adr);

static void initEnv(void);
static void initClient(void);
static void invalidateModuleCache(void);
static void setTextureParams(RCtexture *texbufp, const struct textureReference *tex, const struct cudaChannelFormatDesc *desc);

static int dscudaLoadModuleLocal(unsigned int ipaddr, pid_t pid, char *modulename, char *modulebuf, int vdevid, int raidid);
static cudaError_t dscudaMemcpyToSymbolAsyncH2D(int moduleid, char *symbol, const void *src,
                                                size_t count, size_t offset, RCstream stream, int vdevid, int raidid);
static cudaError_t dscudaMemcpyToSymbolAsyncD2D(int moduleid, char *symbol, const void *src,
                                                size_t count, size_t offset, RCstream stream, int vdevid, int raidid);
static cudaError_t dscudaMemcpyFromSymbolAsyncD2H(int moduleid, void **dstbuf, char *symbol,
                                                  size_t count, size_t offset, RCstream stream, int vdevid, int raidid);
static cudaError_t dscudaMemcpyFromSymbolAsyncD2D(int moduleid, void *dstadr, char *symbol,
                                                  size_t count, size_t offset, RCstream stream, int vdevid, int raidid);

static RCmappedMem *RCmappedMemListTop  = NULL;
static RCmappedMem *RCmappedMemListTail = NULL;

static RCstreamArray *RCstreamArrayListTop  = NULL;
static RCstreamArray *RCstreamArrayListTail = NULL;

static RCeventArray *RCeventArrayListTop  = NULL;
static RCeventArray *RCeventArrayListTail = NULL;

static RCcuarrayArray *RCcuarrayArrayListTop  = NULL;
static RCcuarrayArray *RCcuarrayArrayListTail = NULL;

static RCuva *RCuvaListTop  = NULL;
static RCuva *RCuvaListTail = NULL;

static RCp2pconn *RCp2pconnListTop  = NULL;
static RCp2pconn *RCp2pconnListTail = NULL;

static RCmr *RCmrListTop  = NULL;
static RCmr *RCmrListTail = NULL;

static const char *DEFAULT_SVRIP = "localhost";
static char Dscudapath[512];
static int Nvdev;                          // # of virtual devices available.
static int VdevidIndexMax = 0;            // # of pthreads which utilize virtual devices.
static pthread_t VdevidIndex2ptid[RC_NPTHREADMAX];   // convert an Vdevid index into pthread id.
static int Vdevid[RC_NPTHREADMAX] = { 0 };           // the virtual device currently in use.
static Vdev_t Vdev[RC_NVDEVMAX];           // a list of virtual devices.
static int NSpareServer = 0;
static int NSpareServerLeft = 0;
static RCServer_t SpareServer[RC_NSPAREMAX];
static unsigned int MyIpaddr = 0;
static void (*errorHandler)(void *arg) = NULL;
static void *errorHandlerArg = NULL;
static int autoVerb = 0;
static int CpPeriod = 60;
static time_t CpSavedAt;
//static CLIENT *Clnt[RC_NVDEVMAX][RC_NREDUNDANCYMAX];
static struct rdma_cm_id *Cmid[RC_NVDEVMAX][RC_NREDUNDANCYMAX];
static TcpConnection TcpConn[RC_NVDEVMAX][RC_NREDUNDANCYMAX];
static int UseDaemon = 0;
static int UseGD2 = 0;
static int UseGD3 = 0;
static char Dscudasvrpath[512]; // where DS-CUDA server resides.

#define check_cuda_error(err, msg)                  \
{\
    if (cudaSuccess != err) {\
        fprintf(stderr,\
                "%s(%i) : %s : %s.\n"\
                __FILE__, __LINE__, cudaGetErrorString(err));\
        exit(1);\
    }\
}
