#include "dscudadefs.h"

#if __LP64__
typedef unsigned hyper int RCadr;
typedef unsigned hyper int RCstream;
typedef unsigned hyper int RCevent;
typedef unsigned hyper int RCipaddr;
#else
typedef unsigned long int RCadr;
typedef unsigned long int RCstream;
typedef unsigned long int RCevent;
typedef unsigned long int RCipaddr;
#endif
typedef unsigned int RCsize;
typedef unsigned int RCerror;
typedef opaque RCbuf<>;
typedef unsigned int RCchannelformat;
typedef unsigned long int RCpid;

struct RCchanneldesc_t {
    RCchannelformat f;
    int w;
    int x;
    int y;
    int z;
};
typedef RCchanneldesc_t RCchanneldesc;

struct RCtexture_t {
    int normalized;
    int filterMode;
    int addressMode[3];
    RCchannelformat f;
    int w;
    int x;
    int y;
    int z;
};
typedef RCtexture_t RCtexture;

struct RCfuncattr_t {
    int binaryVersion;
    RCsize constSizeBytes;
    RCsize localSizeBytes;
    int maxThreadsPerBlock;
    int numRegs;
    int ptxVersion;
    RCsize sharedSizeBytes;
};
typedef RCfuncattr_t RCfuncattr;

enum RCargType {
    dscudaArgTypeP = 0,
    dscudaArgTypeI = 1,
    dscudaArgTypeF = 2,
    dscudaArgTypeV = 3
};

union RCargVal switch (RCargType type) {
  case dscudaArgTypeP:
    RCadr address;
  case dscudaArgTypeI:
    unsigned int valuei;
  case dscudaArgTypeF:
    float valuef;
  case dscudaArgTypeV:
    char valuev[RC_KARGMAX];
  default:
    void;
};

struct RCarg {
    RCargVal val;
    unsigned int offset;
    unsigned int size;
};

typedef RCarg RCargs<>;

struct dscudaResult {
    RCerror err;
};

struct dscudaThreadGetLimitResult {
    RCerror err;
    RCsize value;
};

struct dscudaThreadGetCacheConfigResult {
    RCerror err;
    int cacheConfig;
};

struct dscudaMallocResult {
    RCerror err;
    RCadr devAdr;
};

struct dscudaHostAllocResult {
    RCerror err;
    RCadr pHost;
};

struct dscudaMallocHostResult {
    RCerror err;
    RCadr ptr;
};

struct dscudaMallocArrayResult {
    RCerror err;
    RCadr array;
};

struct dscudaMallocPitchResult {
    RCerror err;
    RCadr devPtr;
    RCsize pitch;
};

struct dscudaMemcpyD2HResult {
    RCerror err;
    RCbuf buf;
};

struct dscudaMemcpyH2HResult {
    RCerror err;
    RCbuf buf;
};


struct dscudaMemcpyToArrayD2HResult {
    RCerror err;
    RCbuf buf;
};

struct dscudaMemcpyToArrayH2HResult {
    RCerror err;
    RCbuf buf;
};

struct dscudaMemcpy2DToArrayD2HResult {
    RCerror err;
    RCbuf buf;
};

struct dscudaMemcpy2DToArrayH2HResult {
    RCerror err;
    RCbuf buf;
};

struct dscudaMemcpy2DD2HResult {
    RCerror err;
    RCbuf buf;
};

struct dscudaMemcpy2DH2HResult {
    RCerror err;
    RCbuf buf;
};

struct dscudaGetDeviceResult {
    RCerror err;
    int device;
};

struct dscudaGetDeviceCountResult {
    RCerror err;
    int count;
};

struct dscudaGetDevicePropertiesResult {
    RCerror err;
    RCbuf prop;
};

struct dscudaDriverGetVersionResult {
    RCerror err;
    int ver;
};

struct dscudaRuntimeGetVersionResult {
    RCerror err;
    int ver;
};

struct dscudaGetErrorStringResult {
    string errmsg<>;
};

struct dscudaCreateChannelDescResult {
    int x;
    int y;
    int z;
    int w;
    RCchannelformat f;
};

struct dscudaGetChannelDescResult {
    RCerror err;
    int x;
    int y;
    int z;
    int w;
    RCchannelformat f;
};

struct dscudaChooseDeviceResult {
    RCerror err;
    int device;
};

struct dscudaMemcpyAsyncD2HResult {
    RCerror err;
    RCbuf buf;
};

struct dscudaMemcpyAsyncH2HResult {
    RCerror err;
    RCbuf buf;
};

struct dscudaMemcpyFromSymbolD2HResult {
    RCerror err;
    RCbuf buf;
};

struct dscudaMemcpyFromSymbolAsyncD2HResult {
    RCerror err;
    RCbuf buf;
};

struct dscudaStreamCreateResult {
    RCerror err;
    RCadr stream;
};

struct dscudaEventCreateResult {
    RCerror err;
    RCadr event;
};

struct dscudaEventElapsedTimeResult {
    RCerror err;
    float ms;
};

struct dscudaHostGetDevicePointerResult {
    RCerror err;
    RCadr pDevice;
};

struct dscudaHostGetFlagsResult {
    RCerror err;
    unsigned int flags;
};

struct dscudaLoadModuleResult {
    unsigned int id;
};

struct dscudaFuncGetAttributesResult {
    RCerror err;
    RCfuncattr attr;
};


struct dscudaBindTextureResult {
    RCerror err;
    RCsize offset;
};

struct dscudaBindTexture2DResult {
    RCerror err;
    RCsize offset;
};

struct dscufftResult {
    RCerror err;
};

struct dscufftPlanResult {
    RCerror err;
    unsigned int plan;
};

struct dscublasResult {
    RCerror err;
    unsigned int stat;
};

struct dscublasCreateResult {
    RCerror err;
    unsigned int stat;
    RCadr handle;
};

struct dscublasGetVectorResult {
    RCerror err;
    unsigned int stat;
    RCbuf y;
};

struct RCdim3 {
    unsigned int x;
    unsigned int y;
    unsigned int z;
};

program DSCUDA_PROG {
    version DSCUDA_VER {

        /* Thread Management */
        dscudaResult                     dscudaThreadExitId(void) = 100;
        dscudaResult                     dscudaThreadSynchronizeId(void) = 101;
        dscudaResult                     dscudaThreadSetLimitId(int limit, RCsize value) = 102;
        dscudaThreadGetLimitResult       dscudaThreadGetLimitId(int limit) = 103;
        dscudaResult                     dscudaThreadSetCacheConfigId(int chacheConfig) = 104;
        dscudaThreadGetCacheConfigResult dscudaThreadGetCacheConfigId(void) = 105;

        /* Error Handling */
        dscudaResult                     dscudaGetLastErrorId(void) = 200;
        dscudaResult                     dscudaPeekAtLastErrorId(void) = 201;
        dscudaGetErrorStringResult       dscudaGetErrorStringId(int err) = 202;

        /* Device Management */
        dscudaGetDeviceResult            dscudaGetDeviceId(void) = 300;
        dscudaGetDeviceCountResult       dscudaGetDeviceCountId(void) = 301;
        dscudaGetDevicePropertiesResult  dscudaGetDevicePropertiesId(int device) = 302;
        dscudaDriverGetVersionResult     dscudaDriverGetVersionId(void) = 303;
        dscudaRuntimeGetVersionResult    dscudaRuntimeGetVersionId(void) = 304;
        dscudaResult                     dscudaSetDeviceId(int device) = 305;
        dscudaResult                     dscudaSetDeviceFlagsId(unsigned int flags) = 306;
        dscudaChooseDeviceResult         dscudaChooseDeviceId(RCbuf prop) = 307;
	dscudaResult                     dscudaDeviceSynchronize(void) = 308;
	dscudaResult                     dscudaDeviceReset(void) = 309;

        /* Stream Management */
        dscudaStreamCreateResult         dscudaStreamCreateId(void) = 400;
        dscudaResult                     dscudaStreamDestroyId(RCstream stream) = 401;
        dscudaResult                     dscudaStreamSynchronizeId(RCstream stream) = 402;
        dscudaResult                     dscudaStreamQueryId(RCstream stream) = 403;
        dscudaResult                     dscudaStreamWaitEventId(RCstream stream, RCevent event, unsigned int flags) = 404;

        /* Event Management */
        dscudaEventCreateResult          dscudaEventCreateId(void) = 500;
        dscudaEventCreateResult          dscudaEventCreateWithFlagsId(unsigned int flags) = 501;
        dscudaResult                     dscudaEventDestroyId(RCevent event) = 502;
        dscudaEventElapsedTimeResult     dscudaEventElapsedTimeId(RCevent start, RCevent end) = 503;
        dscudaResult                     dscudaEventRecordId(RCevent event, RCstream stream) = 504;
        dscudaResult                     dscudaEventSynchronizeId(RCevent event) = 505;
        dscudaResult                     dscudaEventQueryId(RCevent event) = 506;

        /* Execution Control */
        void                            dscudaLaunchKernelId(int moduleid, int kid, string kname, RCdim3 gdim, RCdim3 bdim, RCsize smemsize, RCstream stream, RCargs args) = 600;
        dscudaLoadModuleResult           dscudaLoadModuleId(RCipaddr ipaddr, RCpid pid, string mname, string image) = 601;
        dscudaFuncGetAttributesResult    dscudaFuncGetAttributesId(int moduleid, string kname) = 602;

        /* Memory Management */
        dscudaMallocResult               dscudaMallocId(RCsize size) = 700;
        dscudaResult                     dscudaFreeId(RCadr mem) = 701;

        dscudaMemcpyH2HResult            dscudaMemcpyH2HId(RCadr dst, RCbuf src, RCsize count) = 702;
        dscudaResult                     dscudaMemcpyH2DId(RCadr dst, RCbuf src, RCsize count) = 703;
        dscudaMemcpyD2HResult            dscudaMemcpyD2HId(RCadr src, RCsize count) = 704;
        dscudaResult                     dscudaMemcpyD2DId(RCadr dst, RCadr src, RCsize count) = 705;

        dscudaMemcpyAsyncH2HResult       dscudaMemcpyAsyncH2HId(RCadr dst, RCbuf src, RCsize count, RCstream stream) = 706;
        dscudaResult                     dscudaMemcpyAsyncH2DId(RCadr dst, RCbuf src, RCsize count, RCstream stream) = 707;
        dscudaMemcpyAsyncD2HResult       dscudaMemcpyAsyncD2HId(RCadr src, RCsize count, RCstream stream) = 708;
        dscudaResult                     dscudaMemcpyAsyncD2DId(RCadr dst, RCadr src, RCsize count, RCstream stream) = 709;

        dscudaResult                     dscudaMemcpyToSymbolH2DId(int moduleid, string symbol, RCbuf src, RCsize count, RCsize offset) = 710;
        dscudaResult                     dscudaMemcpyToSymbolD2DId(int moduleid, string symbol, RCadr src, RCsize count, RCsize offset) = 711;
        dscudaMemcpyFromSymbolD2HResult  dscudaMemcpyFromSymbolD2HId(int moduleid, string symbol, RCsize count, RCsize offset) = 712;
        dscudaResult                     dscudaMemcpyFromSymbolD2DId(int moduleid, RCadr dst, string symbol, RCsize count, RCsize offset) = 713;
        dscudaResult                     dscudaMemsetId(RCadr dst, int value, RCsize count) = 714;

        dscudaHostAllocResult            dscudaHostAllocId(RCsize size, unsigned int flags) = 715;
        dscudaMallocHostResult           dscudaMallocHostId(RCsize size) = 716;
        dscudaResult                     dscudaFreeHostId(RCadr ptr) = 717;
        dscudaHostGetDevicePointerResult dscudaHostGetDevicePointerId(RCadr pHost, unsigned int flags) = 718;
        dscudaHostGetFlagsResult         dscudaHostGetFlagsID(RCadr pHost) = 719;

        dscudaMallocArrayResult          dscudaMallocArrayId(RCchanneldesc desc, RCsize width, RCsize height, unsigned int flags) = 720;
        dscudaResult                     dscudaFreeArrayId(RCadr array) = 721;

        dscudaMemcpyToArrayH2HResult     dscudaMemcpyToArrayH2HId(RCadr dst, RCsize wOffset, RCsize hOffset, RCbuf src, RCsize count) = 722;
        dscudaResult                     dscudaMemcpyToArrayH2DId(RCadr dst, RCsize wOffset, RCsize hOffset, RCbuf src, RCsize count) = 723;
        dscudaMemcpyToArrayD2HResult     dscudaMemcpyToArrayD2HId(RCsize wOffset, RCsize hOffset, RCadr src, RCsize count) = 724;
        dscudaResult                     dscudaMemcpyToArrayD2DId(RCadr dst, RCsize wOffset, RCsize hOffset, RCadr src, RCsize count) = 725;

        dscudaMallocPitchResult          dscudaMallocPitchId(RCsize width, RCsize height) = 726;
        dscudaMemcpy2DToArrayH2HResult   dscudaMemcpy2DToArrayH2HId(RCadr dst, RCsize wOffset, RCsize hOffset, RCbuf src, RCsize spitch, RCsize width, RCsize height) = 727;
        dscudaResult                     dscudaMemcpy2DToArrayH2DId(RCadr dst, RCsize wOffset, RCsize hOffset, RCbuf srcbuf, RCsize spitch, RCsize width, RCsize height) = 728;
        dscudaMemcpy2DToArrayD2HResult   dscudaMemcpy2DToArrayD2HId(RCsize wOffset, RCsize hOffset, RCadr src, RCsize spitch, RCsize width, RCsize height) = 729;
        dscudaResult                     dscudaMemcpy2DToArrayD2DId(RCadr dst, RCsize wOffset, RCsize hOffset, RCadr src, RCsize spitch, RCsize width, RCsize height) = 730;

        dscudaMemcpy2DH2HResult          dscudaMemcpy2DH2HId(RCadr dst, RCsize dpitch, RCbuf src, RCsize spitch, RCsize width, RCsize height) = 731;
        dscudaResult                     dscudaMemcpy2DH2DId(RCadr dst, RCsize dpitch, RCbuf src, RCsize spitch, RCsize width, RCsize height) = 732;
        dscudaMemcpy2DD2HResult          dscudaMemcpy2DD2HId(RCsize dpitch, RCadr src, RCsize spitch, RCsize width, RCsize height) = 733;
        dscudaResult                     dscudaMemcpy2DD2DId(RCadr dst, RCsize dpitch, RCadr src, RCsize spitch, RCsize width, RCsize height) = 734;
        dscudaResult                     dscudaMemset2DId(RCadr dst, RCsize pitch, int value, RCsize width, RCsize height) = 735;

        dscudaResult                     dscudaMemcpyToSymbolAsyncH2DId(int moduleid, string symbol, RCbuf src, RCsize count, RCsize offset, RCstream stream) = 736;
        dscudaResult                     dscudaMemcpyToSymbolAsyncD2DId(int moduleid, string symbol, RCadr src, RCsize count, RCsize offset, RCstream stream) = 737;
        dscudaMemcpyFromSymbolAsyncD2HResult  dscudaMemcpyFromSymbolAsyncD2HId(int moduleid, string symbol, RCsize count, RCsize offset, RCstream stream) = 738;
        dscudaResult                     dscudaMemcpyFromSymbolAsyncD2DId(int moduleid, RCadr dst, string symbol, RCsize count, RCsize offset, RCstream stream) = 739;

        /* Texture Reference Management */
        dscudaCreateChannelDescResult    dscudaCreateChannelDescId(int x, int y, int z, int w, RCchannelformat f) = 1400;
        dscudaGetChannelDescResult       dscudaGetChannelDescId(RCadr array) = 1401;
        dscudaBindTextureResult          dscudaBindTextureId(int moduleid, string texname, RCadr devPtr, RCsize size, RCtexture texbuf) = 1402;
        dscudaBindTexture2DResult        dscudaBindTexture2DId(int moduleid, string texname, RCadr devPtr, RCsize width, RCsize height, RCsize pitch, RCtexture texbuf) = 1403;
        dscudaResult                     dscudaBindTextureToArrayId(int moduleid, string texname, RCadr array, RCtexture texbuf) = 1404;
        dscudaResult                     dscudaUnbindTextureId(RCtexture texbuf) = 1405;

        /* CUFFT Library */
        /*
        dscufftPlanResult                dscufftPlan1dId(int nx, unsigned int type, int batch) = 2000;
        dscufftPlanResult                dscufftPlan2dId(int nx, int ny, unsigned int type) = 2001;
        */
        dscufftPlanResult                dscufftPlan3dId(int nx, int ny, int nz, unsigned int type) = 2002;

        dscufftResult                    dscufftDestroyId(unsigned int plan) = 2004;
        dscufftResult                    dscufftExecC2CId(unsigned int plan, RCadr idata, RCadr odata, int direction) = 2005;
        /*
        dscufftResult                    dscufftExecR2CId(unsigned int plan, RCadr idata, RCadr odata) = 2006;
        dscufftResult                    dscufftExecC2RId(unsigned int plan, RCadr idata, RCadr odata) = 2007;
        dscufftResult                    dscufftExecZ2ZId(unsigned int plan, RCadr idata, RCadr odata, int direction) = 2008;
        dscufftResult                    dscufftExecD2ZId(unsigned int plan, RCadr idata, RCadr odata) = 2009;
        dscufftResult                    dscufftExecZ2DId(unsigned int plan, RCadr idata, RCadr odata) = 2010;
        dscufftResult                    dscufftSetCompatibilityModeId(unsigned int plan, unsigned int mode) = 2012;
        */

	/* CUBLAS Library */
        /*
        rcublasCreateResult             rcublasCreate_v2Id(void) = 3000;
        rcublasResult                   rcublasDestroy_v2Id(RCadr handle) = 3001;
        rcublasResult                   rcublasSetVectorId(int n, int elemSize, RCbuf x, int incx, RCadr y, int incy) = 3002;
        rcublasGetVectorResult          rcublasGetVectorId(int n, int elemSize, RCadr x, int incx, int incy) = 3003;
        rcublasResult                   rcublasSgemm_v2Id(RCadr handle, unsigned int transa, unsigned int transb, int m, int n, int k, float alpha,
                                                          RCadr A, int lda, RCadr B, int ldb, float beta, RCadr C, int ldc) = 3004;
        */

    } = 1;
} = 60000;
