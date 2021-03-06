#include <stdio.h>
#include <string.h>
#include <netdb.h>
#include <sys/socket.h>
#include <rpc/rpc.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>
#include "dscuda.h"

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

typedef struct {
    int valid;
    int vdevid;  // the virtual device the module is loaded into.
    int id[RC_NREDUNDANCYMAX]; // id assigned to this module in each real device that consists of the virtual one.
    char name[256];
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

static void RCmappedMemRegister(void *pHost, void* pDevice, size_t size);
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

static void initEnv(void);
static void initClient(void);
static void invalidateModuleCache(void);
static void setTextureParams(RCtexture *texbufp, const struct textureReference *tex, const struct cudaChannelFormatDesc *desc);

static RCmappedMem *RCmappedMemListTop  = NULL;
static RCmappedMem *RCmappedMemListTail = NULL;

static RCstreamArray *RCstreamArrayListTop  = NULL;
static RCstreamArray *RCstreamArrayListTail = NULL;

static RCeventArray *RCeventArrayListTop  = NULL;
static RCeventArray *RCeventArrayListTail = NULL;

static RCcuarrayArray *RCcuarrayArrayListTop  = NULL;
static RCcuarrayArray *RCcuarrayArrayListTail = NULL;

static const char *DEFAULT_SVRIP = "localhost";
static char Dscudapath[512];
static int Nvdev;               // # of virtual devices available.
static int Vdevid = 0;          // the virtual device currently in use.
static Vdev_t Vdev[RC_NVDEVMAX];  // a list of virtual devices.
static unsigned int MyIpaddr = 0;
static void (*errorHandler)(void *arg) = NULL;
static void *errorHandlerArg = NULL;
