#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <rpc/rpc.h>
#include <rpc/pmap_clnt.h>
#include <cutil.h>
// remove definition of some macros which will be redefined in \"cutil_inline.h\".
#ifdef MIN
#undef MIN
#endif
#ifdef MAX
#undef MAX
#endif
#include <cutil_inline.h>
#include <cufft.h>
#include <pthread.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <poll.h>
#include <errno.h>

#include "dscudarpc.h"
#include "dscuda.h"
#include "sockutil.h"
#include "ibv_rdma.h"

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
static int UseIbv = 0; // use IB Verbs if set to 1. use RPC by default.
static int Ndevice = 1;                 // # of devices in the system.
static int Devid[RC_NDEVICEMAX] = {0,}; // real device ids of the ones in the system.
static int dscuDevice;                   // virtual device id of the one used in the current context.
static CUcontext dscuContext = NULL;
static int Devid2Vdevid[RC_NDEVICEMAX]; // device id conversion table from real to virtual.
static Module Modulelist[RC_NKMODULEMAX] = {0};

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
static void *dscudaLaunchKernel(int moduleid, int kid, char *kname, RCdim3 gdim, RCdim3 bdim, RCsize smemsize, RCstream stream, RCargs args);
static cudaError_t setTextureParams(CUtexref texref, RCtexture texbuf, char *texname, CUDA_ARRAY_DESCRIPTOR *descp = NULL);

#undef WARN
#define WARN(lv, fmt, args...) if (lv <= dscudaWarnLevel()) \
        fprintf(stderr, "dscudasvr[%d] : " fmt, TcpPort - RC_SERVER_IP_PORT, ## args);

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

#if !RPC_ONLY
#include "dscudasvr_ibv.cu"
#endif
#include "dscudasvr_rpc.cu"


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
        WARN(3, "wait for remotecall preference (\"rpc\" or \"ibv\") from the client.\n");
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

    if (UseIbv) {
#if !RPC_ONLY
        setupIbv();
        notifyIamReady();
        ibvMainLoop(NULL);
#endif
    }
    else {
        setupRpc();
        notifyIamReady();
        svc_run(); // RPC main loop.
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
#if RPC_ONLY
    UseIbv = 0;
    WARN(2, "method of remote procedure call: RPC\n");
#else
    if (D2Csock >= 0) { // launched by daemon.
        WARN(3, "A server launched by the daemon "
             "does not use the evironment variable 'DSCUDA_REMOTECALL'.\n");
    }
    else { // launched by hand.
        if (!env) {
            fprintf(stderr, "Set an environment variable 'DSCUDA_REMOTECALL' to 'ibv' or 'rpc'.\n");
            exit(1);
        }
        if (!strcmp(env, "ibv")) {
            UseIbv = 1;
            WARN(2, "method of remote procedure call: InfiniBand Verbs\n");
        }
        else if (!strcmp(env, "rpc")) {
            UseIbv = 0;
            WARN(2, "method of remote procedure call: RPC\n");
        }
        else {
            UseIbv = 0;
            WARN(2, "method of remote procedure call '%s' is not available. use RPC.\n", env);
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
                 mp->name, mp->pid, dscudaGetIpaddrString(mp->ipaddr),
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
             mname, pid, dscudaGetIpaddrString(ipaddr));
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
                   RCdim3 gdim, RCdim3 bdim, RCsize smemsize, RCstream stream, RCargs args)
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
    if (UseIbv) {
#if !RPC_ONLY
        paramsize = ibvUnpackKernelParam(&kfunc, args.RCargs_len, (IbvArg *)args.RCargs_val);
#endif
    }
    else {
        paramsize = rpcUnpackKernelParam(&kfunc, &args);
    }
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
