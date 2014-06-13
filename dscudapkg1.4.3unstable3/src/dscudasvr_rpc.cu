extern void dscuda_prog_1(struct svc_req *rqstp, register SVCXPRT *transp);

static int rpcUnpackKernelParam(CUfunction *kfuncp, RCargs *argsp);
static void setupRpc(void);


/*
 * A thread to watch over the socket inherited from the daemon,
 * in order to detect disconnection by the client.
 * exit() immediately, if detected.
 */
static void *
rpcWatchDisconnection(void *arg)
{
    int clientsock = *(int *)arg;
    int nrecvd;
    char buf[16];

    sleep(3); // wait long enough so that connection is certainly establised.

    WARN(3, "start socket polling:%d.\n", clientsock);
    while (1) {
        // nrecvd = recv(clientsock, buf, 1, MSG_PEEK | MSG_DONTWAIT);
        nrecvd = recv(clientsock, buf, 1, MSG_PEEK);
        if (nrecvd == 0) {
            WARN(2, "disconnected.\n");
            exit(0);
        }
        else if (nrecvd == -1) {
            if (errno == ENOTCONN) {
                WARN(0, "disconnected by peer.\n");
                exit(1);
            }
            else {
                perror("dscudasvr_rpc:rpcWatchDisconnection:");
                exit(1);
            }
        }
        WARN(2, "got %d-byte side-band message from the client.\n", nrecvd);
    }

    return NULL;
}


static int
rpcUnpackKernelParam(CUfunction *kfuncp, RCargs *argsp)
{
    CUresult cuerr;
    CUfunction kfunc = *kfuncp;
    int ival;
    float fval;
    void *pval;
    RCarg noarg;
    RCarg *argp = &noarg;

    noarg.offset = 0;
    noarg.size = 0;

    for (int i = 0; i < argsp->RCargs_len; i++) {
        argp = &(argsp->RCargs_val[i]);

        switch (argp->val.type) {
          case dscudaArgTypeP:
            pval = (void*)&(argp->val.RCargVal_u.address);
            cuerr = cuParamSetv(kfunc, argp->offset, pval, argp->size);
            if (cuerr != CUDA_SUCCESS) {
                WARN(0, "cuParamSetv(0x%08llx, %d, 0x%08llx, %d) failed. %s\n",
                     kfunc, argp->offset, pval, argp->size,
                     cudaGetErrorString((cudaError_t)cuerr));
                fatal_error(1);
            }
            break;

          case dscudaArgTypeI:
            ival = argp->val.RCargVal_u.valuei;
            cuerr = cuParamSeti(kfunc, argp->offset, ival);
            if (cuerr != CUDA_SUCCESS) {
                WARN(0, "cuParamSeti(0x%08llx, %d, %d) failed. %s\n",
                     kfunc, argp->offset, ival,
                     cudaGetErrorString((cudaError_t)cuerr));
                fatal_error(1);
            }
            break;

          case dscudaArgTypeF:
            fval = argp->val.RCargVal_u.valuef;
            cuerr = cuParamSetf(kfunc, argp->offset, fval);
            if (cuerr != CUDA_SUCCESS) {
                WARN(0, "cuParamSetv(0x%08llx, %d, %f) failed. %s\n",
                     kfunc, argp->offset, fval,
                     cudaGetErrorString((cudaError_t)cuerr));
                fatal_error(1);
            }
            break;

          case dscudaArgTypeV:
            pval = argp->val.RCargVal_u.valuev;
            cuerr = cuParamSetv(kfunc, argp->offset, pval, argp->size);
            if (cuerr != CUDA_SUCCESS) {
                WARN(0, "cuParamSetv(0x%08llx, %d, 0x%08llx, %d) failed. %s\n",
                     kfunc, argp->offset, pval, argp->size,
                     cudaGetErrorString((cudaError_t)cuerr));
                fatal_error(1);
            }
            break;

          default:
            WARN(0, "rpcUnpackKernelParam: invalid RCargType\n", argp->val.type);
            fatal_error(1);
        }
    }
    return argp->offset + argp->size;
}

static void
setupRpc(void)
{
    register SVCXPRT *transp;
    unsigned long int prog = DSCUDA_PROG;
    pthread_t tid;
    static int sock; // must be static since refered by the poller.

    // TCP w/o portmapper, i.e., a fixed port.
    pmap_unset (prog, DSCUDA_VER);
    sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);

    WARN(3, "rpc sock:%d\n", sock);

    if (sock == -1) {
        perror("dscudasvr_rpc:setupRpc:socket()");
        exit(1);
    }

    struct sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_port = htons(TcpPort);
    //    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK); // listen only on 127.0.0.1
    addr.sin_addr.s_addr = htonl(INADDR_ANY);

    if (bind(sock, (struct sockaddr *) &addr, sizeof addr) == -1) {
        perror("dscudasvr_rpc:bind");
        exit(1);
    }
    transp = svctcp_create(sock, RC_BUFSIZE, RC_BUFSIZE);
    if (transp == NULL) {
        fprintf (stderr, "%s", "cannot create tcp service.");
        exit(1);
    }
    if (!svc_register(transp, prog, DSCUDA_VER, dscuda_prog_1, 0)) {
        fprintf (stderr, "unable to register (prog:0x%x DSCUDA_VER:%d, TCP).\n",
                 prog, DSCUDA_VER);
        exit(1);
    }
    if (D2Csock >= 0) {
        pthread_create(&tid, NULL, rpcWatchDisconnection, &D2Csock);
    }
}

/*
 * CUDA API stubs
 */

/*
 * Thread Management
 */

dscudaResult *
dscudathreadexitid_1_svc(struct svc_req *sr)
{
    cudaError_t err;
    static dscudaResult res;

    WARN(3, "cudaThreadExit(\n");
    if (!dscuContext) createDscuContext();

    err = cudaThreadExit();
    check_cuda_error(err);
    res.err = err;
    WARN(3, ") done.\n");
    return &res;
}

dscudaResult *
dscudathreadsynchronizeid_1_svc(struct svc_req *sr)
{
    cudaError_t err;
    static dscudaResult res;

    WARN(3, "cudaThreadSynchronize(");
    if (!dscuContext) createDscuContext();

    err = cudaThreadSynchronize();
    check_cuda_error(err);
    res.err = err;
    WARN(3, ") done.\n");
    return &res;
}

dscudaResult *
dscudathreadsetlimitid_1_svc(int limit, RCsize value, struct svc_req *sr)
{
    cudaError_t err;
    static dscudaResult res;

    WARN(3, "cudaThreadSetLimit(");
    if (!dscuContext) createDscuContext();

    err = cudaThreadSetLimit((enum cudaLimit)limit, value);
    check_cuda_error(err);
    res.err = err;
    WARN(3, "%d, %d) done.\n", limit, value);
    return &res;
}

dscudaThreadGetLimitResult *
dscudathreadgetlimitid_1_svc(int limit, struct svc_req *sr)
{
    cudaError_t err;
    static dscudaThreadGetLimitResult res;
    size_t value;

    WARN(3, "cudaThreadGetLimit(");
    if (!dscuContext) createDscuContext();

    err = cudaThreadGetLimit(&value, (enum cudaLimit)limit);
    check_cuda_error(err);
    res.err = err;
    res.value = value;
    WARN(3, "0x%08llx, %d) done.  value:%d\n", &value, limit, value);
    return &res;
}

dscudaResult *
dscudathreadsetcacheconfigid_1_svc(int cacheConfig, struct svc_req *sr)
{
    cudaError_t err;
    static dscudaResult res;

    WARN(3, "cudaThreadSetCacheConfig(");
    if (!dscuContext) createDscuContext();

    err = cudaThreadSetCacheConfig((enum cudaFuncCache)cacheConfig);
    check_cuda_error(err);
    res.err = err;
    WARN(3, "%d) done.\n", cacheConfig);
    return &res;
}

dscudaThreadGetCacheConfigResult *
dscudathreadgetcacheconfigid_1_svc(struct svc_req *sr)
{
    cudaError_t err;
    static dscudaThreadGetCacheConfigResult res;
    int cacheConfig;

    WARN(3, "cudaThreadGetCacheConfig(");
    if (!dscuContext) createDscuContext();

    err = cudaThreadGetCacheConfig((enum cudaFuncCache *)&cacheConfig);
    check_cuda_error(err);
    res.err = err;
    res.cacheConfig = cacheConfig;
    WARN(3, "0x%08llx) done.  cacheConfig:%d\n", &cacheConfig, cacheConfig);
    return &res;
}


/*
 * Error Handling
 */

dscudaResult *
dscudagetlasterrorid_1_svc(struct svc_req *sr)
{
    cudaError_t err;
    static dscudaResult res;

    WARN(5, "cudaGetLastError(");
    if (!dscuContext) createDscuContext();

    err = cudaGetLastError();
    check_cuda_error(err);
    res.err = err;
    WARN(5, ") done.\n");
    return &res;
}

dscudaResult *
dscudapeekatlasterrorid_1_svc(struct svc_req *sr)
{
    cudaError_t err;
    static dscudaResult res;

    WARN(5, "cudaPeekAtLastError(");
    if (!dscuContext) createDscuContext();

    err = cudaPeekAtLastError();
    check_cuda_error(err);
    res.err = err;
    WARN(5, ") done.\n");
    return &res;
}

dscudaGetErrorStringResult *
dscudageterrorstringid_1_svc(int err, struct svc_req *sr)
{
    static dscudaGetErrorStringResult res;

    WARN(3, "cudaGetErrorString(");
    if (!dscuContext) createDscuContext();

    res.errmsg = (char *)cudaGetErrorString((cudaError_t)err);
    WARN(3, "%d) done.\n", err);
    return &res;
}


/*
 * Device Management
 */

dscudaGetDeviceResult *
dscudagetdeviceid_1_svc(struct svc_req *sr)
{
    cudaError_t err;
    int device;
    static dscudaGetDeviceResult res;

    WARN(3, "cudaGetDevice(");
    if (!dscuContext) createDscuContext();

    err = cudaGetDevice(&device);
    check_cuda_error(err);
    res.device = Devid2Vdevid[device];
    res.err = err;
    WARN(3, "0x%08llx) done. device:%d  virtual device:%d\n",
         (unsigned long)&device, device, res.device);
    return &res;
}

dscudaGetDeviceCountResult *
dscudagetdevicecountid_1_svc(struct svc_req *sr)
{
    int count;
    static dscudaGetDeviceCountResult res;

    WARN(3, "cudaGetDeviceCount(");

#if 0
// this returns # of devices in the system, even if the number of
// valid devices set by cudaSetValidDevices() is smaller.
    cudaError_t err;
    err = cudaGetDeviceCount(&count);
    check_cuda_error(err);
    res.count = count;
    res.err = err;
#else
    res.count = count = Ndevice;
    res.err = cudaSuccess;
#endif
    WARN(3, "0x%08llx) done. count:%d\n", (unsigned long)&count, count);
    return &res;
}

dscudaGetDevicePropertiesResult *
dscudagetdevicepropertiesid_1_svc(int device, struct svc_req *sr)
{
    cudaError_t err;
    static int firstcall = 1;
    static dscudaGetDevicePropertiesResult res;

    WARN(3, "cudaGetDeviceProperties(");

    if (firstcall) {
        firstcall = 0;
        res.prop.RCbuf_val = (char*)malloc(sizeof(cudaDeviceProp));
        res.prop.RCbuf_len = sizeof(cudaDeviceProp);
    }
    if (1 < Ndevice) {
        WARN(0, "dscudagetdevicepropertiesid_1_svc() cannot handle multiple devices for now. Ndevice:%d\n",
             Ndevice);
        exit(1);
    }
    err = cudaGetDeviceProperties((cudaDeviceProp *)res.prop.RCbuf_val, Devid[0]);
    check_cuda_error(err);
    res.err = err;
    WARN(3, "0x%08llx, %d) done.\n", (unsigned long)res.prop.RCbuf_val, Devid[0]);
    return &res;
}

dscudaDriverGetVersionResult *
dscudadrivergetversionid_1_svc(struct svc_req *sr)
{
    cudaError_t err;
    int ver;
    static dscudaDriverGetVersionResult res;

    WARN(3, "cudaDriverGetVersion(");

    if (!dscuContext) createDscuContext();

    err = cudaDriverGetVersion(&ver);
    check_cuda_error(err);
    res.ver = ver;
    res.err = err;
    WARN(3, "0x%08llx) done.\n", (unsigned long)&ver);
    return &res;
}

dscudaRuntimeGetVersionResult *
dscudaruntimegetversionid_1_svc(struct svc_req *sr)
{
    cudaError_t err;
    int ver;
    static dscudaRuntimeGetVersionResult res;

    WARN(3, "cudaRuntimeGetVersion(");

    if (!dscuContext) createDscuContext();

    err = cudaRuntimeGetVersion(&ver);
    check_cuda_error(err);
    res.ver = ver;
    res.err = err;
    WARN(3, "0x%08llx) done.\n", (unsigned long)&ver);
    return &res;
}

dscudaResult *
dscudasetdeviceid_1_svc(int device, struct svc_req *sr)
{
    cudaError_t err;
    static dscudaResult res;

    WARN(3, "cudaSetDevice(");

    if (dscuContext) destroyDscuContext();

    dscuDevice = Devid[device];
    err = createDscuContext();
    res.err = err;
    WARN(3, "%d) done.  dscuDevice: %d\n",
         device, dscuDevice);
    return &res;
}

dscudaResult *
dscudasetdeviceflagsid_1_svc(unsigned int flags, struct svc_req *sr)
{
    cudaError_t err;
    static dscudaResult res;

    WARN(3, "cudaSetDeviceFlags(");

    /* cudaSetDeviceFlags() API should be called only when
     * the device is not active, i.e., dscuContext does not exist.
     * Before invoking the API, destroy the context if valid. */
    if (dscuContext) destroyDscuContext();

    err = cudaSetDeviceFlags(flags);
    check_cuda_error(err);
    res.err = err;
    WARN(3, "0x%08x)\n", flags);

    return &res;
}

dscudaChooseDeviceResult *
dscudachoosedeviceid_1_svc(RCbuf prop, struct svc_req *sr)
{
    cudaError_t err;
    int device;
    static dscudaChooseDeviceResult res;

    WARN(3, "cudaGetDevice(");
    if (!dscuContext) createDscuContext();

    err = cudaChooseDevice(&device, (const struct cudaDeviceProp *)&prop.RCbuf_val);
    check_cuda_error(err);
    res.device = Devid2Vdevid[device];
    res.err = err;
    WARN(3, "0x%08llx) done. device:%d  virtual device:%d\n",
         (unsigned long)&device, device, res.device);
    return &res;
}


dscudaResult *
dscudadevicesynchronize_1_svc(struct svc_req *sr)
{
    cudaError_t err;
    static dscudaResult res;

    WARN(3, "cudaDeviceSynchronize(");
    if (!dscuContext) createDscuContext();

    err = cudaDeviceSynchronize();
    check_cuda_error(err);
    res.err = err;
    WARN(3, ") done.\n");

    return &res;
}

dscudaResult *
dscudadevicereset_1_svc(struct svc_req *sr)
{
    cudaError_t err;
    bool all = true;
    static dscudaResult res;

    WARN(3, "cudaDeviceReset(");
    if (!dscuContext) createDscuContext();

    err = cudaDeviceReset();
    check_cuda_error(err);
    res.err = err;
    releaseModules(all);
    WARN(3, ") done.\n");

    return &res;
}

/*
 * Stream Management
 */

dscudaStreamCreateResult *
dscudastreamcreateid_1_svc(struct svc_req *sr)
{
    static dscudaStreamCreateResult res;
    cudaError_t err;
    cudaStream_t stream;

    WARN(3, "cudaStreamCreate(");
    if (!dscuContext) createDscuContext();
    err = cudaStreamCreate(&stream);
    res.stream = (RCadr)stream;
    check_cuda_error(err);
    res.err = err;
    WARN(3, "0x%08llx) done. stream:0x%08llx\n", &stream, stream);

    return &res;
}

dscudaResult *
dscudastreamdestroyid_1_svc(RCstream stream, struct svc_req *sr)
{
    static dscudaResult res;
    cudaError_t err;

    WARN(3, "cudaStreamDestroy(");
    if (!dscuContext) createDscuContext();
    err = cudaStreamDestroy((cudaStream_t)stream);
    check_cuda_error(err);
    res.err = err;
    WARN(3, "0x%08llx) done.\n", stream);

    return &res;
}

dscudaResult *
dscudastreamwaiteventid_1_svc(RCstream stream, RCevent event, unsigned int flags, struct svc_req *sr)
{
    static dscudaResult res;
    cudaError_t err;

    WARN(3, "cudaStreamWaitEvent(");
    if (!dscuContext) createDscuContext();
    err = cudaStreamWaitEvent((cudaStream_t)stream, (cudaEvent_t)event, flags);
    check_cuda_error(err);
    res.err = err;
    WARN(3, "0x%08llx 0x%08llx, 0x%08x) done.\n", stream, event, flags);

    return &res;
}

dscudaResult *
dscudastreamsynchronizeid_1_svc(RCstream stream, struct svc_req *sr)
{
    static dscudaResult res;
    cudaError_t err;

    WARN(3, "cudaStreamSynchronize(");
    if (!dscuContext) createDscuContext();
    err = cudaStreamSynchronize((cudaStream_t)stream);
    check_cuda_error(err);
    res.err = err;
    WARN(3, "0x%08llx) done.\n", stream);

    return &res;
}

dscudaResult *
dscudastreamqueryid_1_svc(RCstream stream, struct svc_req *sr)
{
    static dscudaResult res;
    cudaError_t err;

    WARN(3, "cudaStreamQuery(");
    if (!dscuContext) createDscuContext();
    err = cudaStreamQuery((cudaStream_t)stream);
    // should not check error due to the nature of this API.
    // check_cuda_error(err);
    res.err = err;
    WARN(3, "0x%08llx) done.\n", stream);

    return &res;
}

/*
 * Event Management
 */

dscudaEventCreateResult *
dscudaeventcreateid_1_svc(struct svc_req *sr)
{
    static dscudaEventCreateResult res;
    cudaError_t err;
    cudaEvent_t event;

    WARN(3, "cudaEventCreate(");
    if (!dscuContext) createDscuContext();
    err = cudaEventCreate(&event);
    res.event = (RCadr)event;
    check_cuda_error(err);
    res.err = err;
    WARN(3, "0x%08llx) done. event:0x%08llx\n", &event, event);

    return &res;
}

dscudaEventCreateResult *
dscudaeventcreatewithflagsid_1_svc(unsigned int flags, struct svc_req *sr)
{
    static dscudaEventCreateResult res;
    cudaError_t err;
    cudaEvent_t event;

    WARN(3, "cudaEventCreateWithFlags(");
    if (!dscuContext) createDscuContext();
    err = cudaEventCreateWithFlags(&event, flags);
    res.event = (RCadr)event;
    check_cuda_error(err);
    res.err = err;
    WARN(3, "0x%08llx, 0x%08x) done. event:0x%08llx\n", &event, flags, event);

    return &res;
}

dscudaResult *
dscudaeventdestroyid_1_svc(RCevent event, struct svc_req *sr)
{
    static dscudaResult res;
    cudaError_t err;

    WARN(3, "cudaEventDestroy(");
    if (!dscuContext) createDscuContext();
    err = cudaEventDestroy((cudaEvent_t)event);
    check_cuda_error(err);
    res.err = err;
    WARN(3, "0x%08llx) done.\n", event);

    return &res;
}

dscudaEventElapsedTimeResult *
dscudaeventelapsedtimeid_1_svc(RCevent start, RCevent end, struct svc_req *sr)
{
    static dscudaEventElapsedTimeResult res;
    cudaError_t err;
    float millisecond;

    WARN(3, "cudaEventElapsedTime(");
    if (!dscuContext) createDscuContext();
    err = cudaEventElapsedTime(&millisecond, (cudaEvent_t)start, (cudaEvent_t)end);
    check_cuda_error(err);
    res.ms = millisecond;
    res.err = err;
    WARN(3, "%5.3f 0x%08llx 0x%08llx) done.\n", millisecond, start, end);

    return &res;
}

dscudaResult *
dscudaeventrecordid_1_svc(RCevent event, RCstream stream, struct svc_req *sr)
{
    static dscudaResult res;
    cudaError_t err;

    WARN(3, "cudaEventRecord(");
    if (!dscuContext) createDscuContext();
    err = cudaEventRecord((cudaEvent_t)event, (cudaStream_t)stream);
    check_cuda_error(err);
    res.err = err;
    WARN(3, "0x%08llx 0x%08llx) done.\n", event, stream);

    return &res;
}

dscudaResult *
dscudaeventsynchronizeid_1_svc(RCevent event, struct svc_req *sr)
{
    static dscudaResult res;
    cudaError_t err;

    WARN(3, "cudaEventSynchronize(");
    if (!dscuContext) createDscuContext();
    err = cudaEventSynchronize((cudaEvent_t)event);
    check_cuda_error(err);
    res.err = err;
    WARN(3, "0x%08llx) done.\n", event);

    return &res;
}

dscudaResult *
dscudaeventqueryid_1_svc(RCevent event, struct svc_req *sr)
{
    static dscudaResult res;
    cudaError_t err;

    WARN(3, "cudaEventQuery(");
    if (!dscuContext) createDscuContext();
    err = cudaEventQuery((cudaEvent_t)event);
    // should not check error due to the nature of this API.
    // check_cuda_error(err);
    res.err = err;
    WARN(3, "0x%08llx) done.\n", event);

    return &res;
}


dscudaFuncGetAttributesResult *
dscudafuncgetattributesid_1_svc(int moduleid, char *kname, struct svc_req *sr)
{
    static dscudaFuncGetAttributesResult res;
    CUresult err;
    CUfunction kfunc;

    if (!dscuContext) createDscuContext();

    err = getFunctionByName(&kfunc, kname, moduleid);
    check_cuda_error((cudaError_t)err);

    WARN(3, "cuFuncGetAttribute(");
    err = cuFuncGetAttribute(&res.attr.binaryVersion, CU_FUNC_ATTRIBUTE_BINARY_VERSION, kfunc);
    check_cuda_error((cudaError_t)err);
    WARN(3, "0x%08llx, %d, 0x%08llx) done.\n", &res.attr.binaryVersion, CU_FUNC_ATTRIBUTE_BINARY_VERSION, kfunc);

    err = cuFuncGetAttribute((int *)&res.attr.constSizeBytes, CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES, kfunc);
    check_cuda_error((cudaError_t)err);
    WARN(3, "0x%08llx, %d, 0x%08llx) done.\n", &res.attr.constSizeBytes, CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES, kfunc);

    err = cuFuncGetAttribute((int *)&res.attr.localSizeBytes, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, kfunc);
    check_cuda_error((cudaError_t)err);
    WARN(3, "0x%08llx, %d, 0x%08llx) done.\n", &res.attr.localSizeBytes, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, kfunc);

    err = cuFuncGetAttribute(&res.attr.maxThreadsPerBlock, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, kfunc);
    check_cuda_error((cudaError_t)err);
    WARN(3, "0x%08llx, %d, 0x%08llx) done.\n", &res.attr.maxThreadsPerBlock, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, kfunc);

    err = cuFuncGetAttribute(&res.attr.numRegs, CU_FUNC_ATTRIBUTE_NUM_REGS, kfunc);
    check_cuda_error((cudaError_t)err);
    WARN(3, "0x%08llx, %d, 0x%08llx) done.\n", &res.attr.numRegs, CU_FUNC_ATTRIBUTE_NUM_REGS, kfunc);

    err = cuFuncGetAttribute(&res.attr.ptxVersion, CU_FUNC_ATTRIBUTE_PTX_VERSION, kfunc);
    check_cuda_error((cudaError_t)err);
    WARN(3, "0x%08llx, %d, 0x%08llx) done.\n", &res.attr.ptxVersion, CU_FUNC_ATTRIBUTE_PTX_VERSION, kfunc);

    err = cuFuncGetAttribute((int *)&res.attr.sharedSizeBytes, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, kfunc);
    check_cuda_error((cudaError_t)err);
    WARN(3, "0x%08llx, %d, 0x%08llx) done.\n", &res.attr.sharedSizeBytes, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, kfunc);

    res.err = err;

    return &res;
}

/*
 * Memory Management
 */

dscudaMallocResult * 
dscudamallocid_1_svc(RCsize size, struct svc_req *sr)
{
    static dscudaMallocResult res;
    cudaError_t err;
    int *devadr;

    WARN(3, "cudaMalloc(");
    if (!dscuContext) createDscuContext();
    err = cudaMalloc((void**)&devadr, size);
    res.devAdr = (RCadr)devadr;
    check_cuda_error(err);
    res.err = err;
    WARN(3, "0x%08llx, %d) done. devadr:0x%08llx\n", &devadr, size, devadr);

    return &res;
}

dscudaResult *
dscudafreeid_1_svc(RCadr mem, struct svc_req *)
{
    static dscudaResult res;
    cudaError_t err;

    WARN(3, "cudaFree(");
    if (!dscuContext) createDscuContext();
    err = cudaFree((void*)mem);
    check_cuda_error(err);
    res.err = err;
    WARN(3, "0x%08llx) done.\n", mem);

    return &res;
}

dscudaMemcpyH2HResult *
dscudamemcpyh2hid_1_svc(RCadr dst, RCbuf srcbuf, RCsize count, struct svc_req *sr)
{
    static dscudaMemcpyH2HResult res;
    WARN(0, "dscudaMemcpy() does not support cudaMemcpyHostToHost transfer yet.\n");
    return &res;
}

dscudaResult *
dscudamemcpyh2did_1_svc(RCadr dst, RCbuf srcbuf, RCsize count, struct svc_req *sr)
{
    static dscudaResult res;
    cudaError_t err;

    WARN(3, "cudaMemcpy(");
    if (!dscuContext) createDscuContext();
    err = cudaMemcpy((void*)dst, srcbuf.RCbuf_val, count, cudaMemcpyHostToDevice);
    check_cuda_error(err);
    res.err = err;
    WARN(3, "0x%08llx, 0x%08lx, %d, %s) done.\n",
            dst, (unsigned long)srcbuf.RCbuf_val, count, dscudaMemcpyKindName(cudaMemcpyHostToDevice));
    return &res;
}

dscudaMemcpyD2HResult *
dscudamemcpyd2hid_1_svc(RCadr src, RCsize count, struct svc_req *sr)
{
    static RCsize maxcount = 0;
    static dscudaMemcpyD2HResult res;
    cudaError_t err;

    WARN(3, "cudaMemcpy(");
    if (!dscuContext) createDscuContext();
    if (maxcount == 0) {
        res.buf.RCbuf_val = NULL;
    }
    if (maxcount < count) {
        res.buf.RCbuf_val = (char*)realloc(res.buf.RCbuf_val, count);
        maxcount = count;
    }
    res.buf.RCbuf_len = count;
    err = cudaMemcpy(res.buf.RCbuf_val, (const void*)src, count, cudaMemcpyDeviceToHost);
    WARN(3, "0x%08lx, 0x%08llx, %d, %s) done.\n",
         (unsigned long)res.buf.RCbuf_val, src, count, dscudaMemcpyKindName(cudaMemcpyDeviceToHost));
    check_cuda_error(err);
    res.err = err;

#if 0 // !!! [debugging purpose only] destroy some part of the returning data
      // !!! in order to emulate a malfunctional GPU.
    {
        static int firstcall = 1;
        static int err_in_prev_call = 0; // avoid bad data generation in adjacent calls.
        if (firstcall) {
            firstcall = 0;
            srand48(time(NULL));
        }
        if (drand48() < 1.0/30.0 && !err_in_prev_call) {
            WARN(2, "################ bad data generatad.\n\n");
            res.buf.RCbuf_val[0] = 123;
            err_in_prev_call = 1;
        }
        else {
            err_in_prev_call = 0;
        }
    }
#endif


    return &res;
}

dscudaResult *
dscudamemcpyd2did_1_svc(RCadr dst, RCadr src, RCsize count, struct svc_req *sr)
{
    cudaError_t err;
    static dscudaResult res;
    WARN(3, "cudaMemcpy(");
    err = cudaMemcpy((void *)dst, (void *)src, count, cudaMemcpyDeviceToDevice);
    check_cuda_error(err);
    res.err = err;
    WARN(3, "0x%08llx, 0x%08llx, %d, %s) done.\n",
            dst, src, count, dscudaMemcpyKindName(cudaMemcpyDeviceToDevice));
    return &res;
}

dscudaMallocArrayResult *
dscudamallocarrayid_1_svc(RCchanneldesc desc, RCsize width, RCsize height, unsigned int flags, struct svc_req *sr)
{
    static dscudaMallocArrayResult res;
    cudaError_t err;
    cudaArray *devadr;
    cudaChannelFormatDesc descbuf = cudaCreateChannelDesc(desc.x, desc.y, desc.z, desc.w, (enum cudaChannelFormatKind)desc.f);

    WARN(3, "cudaMallocArray(");
    if (!dscuContext) createDscuContext();
    err = cudaMallocArray((cudaArray**)&devadr, &descbuf, width, height, flags);
    res.array = (RCadr)devadr;
    check_cuda_error(err);
    res.err = err;
    WARN(3, "0x%08llx, 0x%08llx, %d, %d, 0x%08x) done. devadr:0x%08llx\n",
         &devadr, &descbuf, width, height, flags, devadr)

    return &res;
}

dscudaResult *
dscudafreearrayid_1_svc(RCadr array, struct svc_req *sr)
{
    static dscudaResult res;
    cudaError_t err;

    WARN(3, "cudaFreeArray(");
    if (!dscuContext) createDscuContext();
    err = cudaFreeArray((cudaArray*)array);
    check_cuda_error(err);
    res.err = err;
    WARN(3, "0x%08llx) done.\n", array);

    return &res;
}

dscudaMemcpyToArrayH2HResult *
dscudamemcpytoarrayh2hid_1_svc(RCadr dst, RCsize wOffset, RCsize hOffset, RCbuf src, RCsize count, struct svc_req *sr)
{
    static dscudaMemcpyToArrayH2HResult res;
    WARN(0, "dscudaMemcpyToArray() does not support cudaMemcpyHostToHost transfer yet.\n");
    return &res;
}

dscudaResult *
dscudamemcpytoarrayh2did_1_svc(RCadr dst, RCsize wOffset, RCsize hOffset, RCbuf src, RCsize count, struct svc_req *sr)
{
    static dscudaResult res;
    cudaError_t err;

    WARN(3, "cudaMemcpyToArray(");
    if (!dscuContext) createDscuContext();
    err = cudaMemcpyToArray((cudaArray *)dst, wOffset, hOffset, src.RCbuf_val, count, cudaMemcpyHostToDevice);
    check_cuda_error(err);
    res.err = err;
    WARN(3, "0x%08llx, %d, %d, 0x%08llx, %d, %s) done.\n",
         dst, wOffset, hOffset, (unsigned long)src.RCbuf_val, count, dscudaMemcpyKindName(cudaMemcpyHostToDevice));
    return &res;
}

dscudaMemcpyToArrayD2HResult *
dscudamemcpytoarrayd2hid_1_svc(RCsize wOffset, RCsize hOffset, RCadr src, RCsize count, struct svc_req *sr)
{
    static dscudaMemcpyToArrayD2HResult res;
    WARN(0, "dscudaMemcpyToArray() does not support cudaMemcpyDeviceToHost transfer yet.\n");
    return &res;
}

dscudaResult *
dscudamemcpytoarrayd2did_1_svc(RCadr dst, RCsize wOffset, RCsize hOffset, RCadr src, RCsize count, struct svc_req *sr)
{
    cudaError_t err;
    static dscudaResult res;
    WARN(3, "cudaMemcpyToArray(");
    err = cudaMemcpyToArray((cudaArray *)dst, wOffset, hOffset, (void *)src, count, cudaMemcpyDeviceToDevice);
    check_cuda_error(err);
    res.err = err;
    WARN(3, "0x%08llx, %d, %d, 0x%08llx, %d, %s) done.\n",
         dst, wOffset, hOffset, src, count, dscudaMemcpyKindName(cudaMemcpyDeviceToDevice));
    return &res;
}

dscudaResult *
dscudamemcpytosymbolh2did_1_svc(int moduleid, char *symbol, RCbuf src, RCsize count, RCsize offset, struct svc_req *sr)
{
    static dscudaResult res;
    cudaError_t err;
    CUdeviceptr gsptr;
    size_t gssize;

    WARN(3, "cudaMemcpyToSymbol(");
    if (!dscuContext) createDscuContext();

    getGlobalSymbol(moduleid, symbol, &gsptr, &gssize);
    err = cudaMemcpy((char *)gsptr + offset, src.RCbuf_val, count, cudaMemcpyHostToDevice);
                             
    WARN(3, "0x%08lx, 0x%08lx, %d, %d, %s) done. module name:%s  symbol:%s\n",
         gsptr, (unsigned long)src.RCbuf_val, count, offset,
         dscudaMemcpyKindName(cudaMemcpyHostToDevice),
         Modulelist[moduleid].name, symbol);
    check_cuda_error(err);
    res.err = err;
    return &res;
}

dscudaResult *
dscudamemcpytosymbold2did_1_svc(int moduleid, char *symbol, RCadr src, RCsize count, RCsize offset, struct svc_req *sr)
{
    static dscudaResult res;
    cudaError_t err;
    CUdeviceptr gsptr;
    size_t gssize;

    WARN(3, "cudaMemcpyToSymbol(");
    if (!dscuContext) createDscuContext();

    getGlobalSymbol(moduleid, symbol, &gsptr, &gssize);

    err = cudaMemcpy((char *)gsptr + offset, (void*)src, count, cudaMemcpyDeviceToDevice);
    check_cuda_error(err);
    res.err = err;
    WARN(3, "0x%08llx, 0x%08llx, %d, %d, %s) done.\n",
         gsptr, (unsigned long)src, count, offset,
         dscudaMemcpyKindName(cudaMemcpyDeviceToDevice));

    return &res;
}

dscudaMemcpyFromSymbolD2HResult *
dscudamemcpyfromsymbold2hid_1_svc(int moduleid, char *symbol, RCsize count, RCsize offset, struct svc_req *sr)
{
    static RCsize maxcount = 0;
    static dscudaMemcpyFromSymbolD2HResult res;
    cudaError_t err;
    CUdeviceptr gsptr;
    size_t gssize;

    WARN(3, "cudaMemcpyFromSymbol(");
    if (!dscuContext) createDscuContext();
    if (maxcount == 0) {
        res.buf.RCbuf_val = NULL;
    }
    if (maxcount < count) {
        res.buf.RCbuf_val = (char*)realloc(res.buf.RCbuf_val, count);
        maxcount = count;
    }
    res.buf.RCbuf_len = count;

    getGlobalSymbol(moduleid, symbol, &gsptr, &gssize);
    err = cudaMemcpy(res.buf.RCbuf_val, (char *)gsptr + offset, count, cudaMemcpyDeviceToHost);
                             
    WARN(3, "0x%08llx, 0x%08llx, %d, %d, %s) done. module name:%s  symbol:%s\n",
         (unsigned long)res.buf.RCbuf_val, gsptr, count, offset,
         dscudaMemcpyKindName(cudaMemcpyDeviceToHost),
         Modulelist[moduleid].name, symbol);
    check_cuda_error(err);
    res.err = err;
    return &res;
}

dscudaResult *
dscudamemcpyfromsymbold2did_1_svc(int moduleid, RCadr dst, char *symbol, RCsize count, RCsize offset, struct svc_req *sr)
{
    static dscudaResult res;
    cudaError_t err;
    CUdeviceptr gsptr;
    size_t gssize;

    WARN(3, "cudaMemcpyFromSymbol(");
    if (!dscuContext) createDscuContext();

    getGlobalSymbol(moduleid, symbol, &gsptr, &gssize);

    err = cudaMemcpy((void*)dst, (char *)gsptr + offset, count, cudaMemcpyDeviceToDevice);
    check_cuda_error(err);
    res.err = err;
    WARN(3, "0x%08llx, 0x%08llx, %d, %d, %s) done.\n",
         (unsigned long)dst, gsptr, count, offset,
         dscudaMemcpyKindName(cudaMemcpyDeviceToDevice));

    return &res;
}


dscudaResult *
dscudamemcpytosymbolasynch2did_1_svc(int moduleid, char *symbol, RCbuf src, RCsize count, RCsize offset, RCstream stream, struct svc_req *sr)
{
    static dscudaResult res;
    cudaError_t err;
    CUdeviceptr gsptr;
    size_t gssize;

    WARN(3, "cudaMemcpyToSymbolAsync(");
    if (!dscuContext) createDscuContext();
    getGlobalSymbol(moduleid, symbol, &gsptr, &gssize);
    err = cudaMemcpyAsync((char *)gsptr + offset, src.RCbuf_val, count, cudaMemcpyHostToDevice, (cudaStream_t)stream);
                             
    WARN(3, "0x%08lx, 0x%08lx, %d, %d, %s, 0x%08llx) done. module name:%s  symbol:%s\n",
         gsptr, (unsigned long)src.RCbuf_val, count, offset,
         dscudaMemcpyKindName(cudaMemcpyHostToDevice), stream,
         Modulelist[moduleid].name, symbol);

    check_cuda_error(err);
    res.err = err;
    return &res;
}

dscudaResult *
dscudamemcpytosymbolasyncd2did_1_svc(int moduleid, char *symbol, RCadr src, RCsize count, RCsize offset, RCstream stream, struct svc_req *sr)
{
    static dscudaResult res;
    cudaError_t err;
    CUdeviceptr gsptr;
    size_t gssize;

    WARN(3, "cudaMemcpyToSymbolAsync(");
    if (!dscuContext) createDscuContext();

    getGlobalSymbol(moduleid, symbol, &gsptr, &gssize);

    err = cudaMemcpyAsync((char *)gsptr + offset, (void*)src, count, cudaMemcpyDeviceToDevice, (cudaStream_t)stream);
    check_cuda_error(err);
    res.err = err;
    WARN(3, "0x%08lx, 0x%08lx, %d, %d, %s, 0x%08llx) done.\n",
         gsptr, (unsigned long)src, count, offset, stream,
         dscudaMemcpyKindName(cudaMemcpyDeviceToDevice));

    return &res;
}

dscudaMemcpyFromSymbolAsyncD2HResult *
dscudamemcpyfromsymbolasyncd2hid_1_svc(int moduleid, char *symbol, RCsize count, RCsize offset, RCstream stream, struct svc_req *sr)
{
    static RCsize maxcount = 0;
    static dscudaMemcpyFromSymbolAsyncD2HResult res;
    cudaError_t err;
    CUdeviceptr gsptr;
    size_t gssize;

    WARN(3, "cudaMemcpyFromSymbolAsync(");
    if (!dscuContext) createDscuContext();
    if (maxcount == 0) {
        res.buf.RCbuf_val = NULL;
    }
    if (maxcount < count) {
        res.buf.RCbuf_val = (char*)realloc(res.buf.RCbuf_val, count);
        maxcount = count;
    }
    res.buf.RCbuf_len = count;

    getGlobalSymbol(moduleid, symbol, &gsptr, &gssize);
    err = cudaMemcpyAsync(res.buf.RCbuf_val, (char *)gsptr + offset, count, cudaMemcpyDeviceToHost, (cudaStream_t)stream);
                             
    WARN(3, "0x%08lx, 0x%08lx, %d, %d, %s, 0x%08llx) done. module name:%s  symbol:%s\n",
         (unsigned long)res.buf.RCbuf_val, gsptr, count, offset, stream,
         dscudaMemcpyKindName(cudaMemcpyDeviceToHost),
         Modulelist[moduleid].name, symbol);
    check_cuda_error(err);
    res.err = err;
    return &res;
}

dscudaResult *
dscudamemcpyfromsymbolasyncd2did_1_svc(int moduleid, RCadr dst, char *symbol, RCsize count, RCsize offset, RCstream stream, struct svc_req *sr)
{
    static dscudaResult res;
    cudaError_t err;
    CUdeviceptr gsptr;
    size_t gssize;

    WARN(3, "cudaMemcpyFromSymbolAsync(");
    if (!dscuContext) createDscuContext();

    getGlobalSymbol(moduleid, symbol, &gsptr, &gssize);

    err = cudaMemcpyAsync((void*)dst, (char *)gsptr + offset, count, cudaMemcpyDeviceToDevice, (cudaStream_t)stream);
    check_cuda_error(err);
    res.err = err;
    WARN(3, "0x%08lx, 0x%08lx, %d, %d, %s, 0x%08llx) done.\n",
         (unsigned long)dst, gsptr, count, offset, stream,
         dscudaMemcpyKindName(cudaMemcpyDeviceToDevice));

    return &res;
}


dscudaResult *
dscudamemsetid_1_svc(RCadr dst, int value, RCsize count, struct svc_req *sq)
{
    static dscudaResult res;
    cudaError_t err;

    WARN(3, "cudaMemset(");
    if (!dscuContext) createDscuContext();
    err = cudaMemset((void *)dst, value, count);
    check_cuda_error(err);
    res.err = err;
    WARN(3, "0x%08llx, %d, %d) done.\n", dst, value, count);
    return &res;
}

dscudaHostAllocResult *
dscudahostallocid_1_svc(RCsize size, unsigned int flags, struct svc_req *sr)
{
    static dscudaHostAllocResult res;
    cudaError_t err;
    int *devadr;

    WARN(3, "cudaHostAlloc(");
    if (!dscuContext) createDscuContext();
    err = cudaHostAlloc((void**)&devadr, size, flags);
    res.pHost = (RCadr)devadr;
    check_cuda_error(err);
    res.err = err;
    WARN(3, "0x%08llx, %d, 0x%08x) done.\n", res.pHost, size, flags);

    return &res;
}

dscudaMallocHostResult *
dscudamallochostid_1_svc(RCsize size, struct svc_req *sr)
{
    static dscudaMallocHostResult res;
    cudaError_t err;
    int *devadr;

    WARN(3, "cudaMallocHost(");
    if (!dscuContext) createDscuContext();
    err = cudaMallocHost((void**)&devadr, size);
    res.ptr = (RCadr)devadr;
    check_cuda_error(err);
    res.err = err;
    WARN(3, "0x%08llx, %d) done. devadr:0x%08llx\n", &devadr, size, devadr);

    return &res;
}

dscudaResult *
dscudafreehostid_1_svc(RCadr ptr, struct svc_req *sr)
{
    static dscudaResult res;
    cudaError_t err;

    WARN(3, "cudaFreeHost(");
    if (!dscuContext) createDscuContext();
    err = cudaFreeHost((void*)ptr);
    check_cuda_error(err);
    res.err = err;
    WARN(3, "0x%08llx) done.\n", ptr);

    return &res;
}

dscudaHostGetDevicePointerResult *
dscudahostgetdevicepointerid_1_svc(RCadr pHost, unsigned int flags , struct svc_req *sr)
{
    cudaError_t err;
    static dscudaHostGetDevicePointerResult res;
    RCadr pDevice;

    WARN(3, "cudaHostGetDevicePointer(");
    if (!dscuContext) createDscuContext();

    err = cudaHostGetDevicePointer((void **)&pDevice, (void *)pHost, flags);
    check_cuda_error(err);
    res.pDevice = pDevice;
    res.err = err;
    WARN(3, ") done.\n");
    return &res;
}

dscudaHostGetFlagsResult *
dscudahostgetflagsid_1_svc(RCadr pHost, struct svc_req *sr)
{
    cudaError_t err;
    static dscudaHostGetFlagsResult res;
    unsigned int flags;

    WARN(3, "cudaHostGetFlags(");
    if (!dscuContext) createDscuContext();

    err = cudaHostGetFlags(&flags, (void *)pHost);
    check_cuda_error(err);
    res.err = err;
    res.flags = flags;
    WARN(3, ") done.\n");
    return &res;
}

dscudaMemcpyAsyncH2HResult *
dscudamemcpyasynch2hid_1_svc(RCadr dst, RCbuf src, RCsize count, RCstream stream, struct svc_req *sr)
{
    static dscudaMemcpyAsyncH2HResult res;
    WARN(0, "dscudaMemcpyAsync() does not support cudaMemcpyHostToHost transfer yet.\n");
    return &res;
}

dscudaResult *
dscudamemcpyasynch2did_1_svc(RCadr dst, RCbuf src, RCsize count, RCstream stream, struct svc_req *sr)
{
    static dscudaResult res;
    cudaError_t err;

    WARN(3, "cudaMemcpyAsync(");
    if (!dscuContext) createDscuContext();
    err = cudaMemcpyAsync((void*)dst, src.RCbuf_val, count, cudaMemcpyHostToDevice, (cudaStream_t)stream);
    check_cuda_error(err);
    res.err = err;
    WARN(3, "0x%08lx, 0x%08lx, %d, %s, 0x%08lx) done.\n",
         dst, (unsigned long)src.RCbuf_val, count, dscudaMemcpyKindName(cudaMemcpyHostToDevice), stream);
    return &res;
}

dscudaMemcpyAsyncD2HResult *
dscudamemcpyasyncd2hid_1_svc(RCadr src, RCsize count, RCstream stream, struct svc_req *sr)
{
    static RCsize maxcount = 0;
    cudaError_t err;
    static dscudaMemcpyAsyncD2HResult res;

    WARN(3, "cudaMemcpyAsync(");
    if (!dscuContext) createDscuContext();
    if (maxcount == 0) {
        res.buf.RCbuf_val = NULL;
    }
    if (maxcount < count) {
        res.buf.RCbuf_val = (char*)realloc(res.buf.RCbuf_val, count);
        maxcount = count;
    }
    res.buf.RCbuf_len = count;
    err = cudaMemcpyAsync(res.buf.RCbuf_val, (const void*)src, count, cudaMemcpyDeviceToHost, (cudaStream_t)stream);
    check_cuda_error(err);
    res.err = err;
    WARN(3, "0x%08llx, 0x%08llx, %d, %s, 0x%08llx) done.\n",
         (unsigned long)res.buf.RCbuf_val, src, count, dscudaMemcpyKindName(cudaMemcpyDeviceToHost), stream);
    return &res;
}

dscudaResult *
dscudamemcpyasyncd2did_1_svc(RCadr dst, RCadr src, RCsize count, RCstream stream, struct svc_req *sr)
{
    cudaError_t err;
    static dscudaResult res;
    WARN(3, "cudaMemcpyAsync(");
    err = cudaMemcpyAsync((void *)dst, (void *)src, count, cudaMemcpyDeviceToDevice, (cudaStream_t)stream);
    check_cuda_error(err);
    res.err = err;
    WARN(3, "0x%08llx, 0x%08llx, %d, %s, 0x%08llx) done.\n",
         dst, src, count, dscudaMemcpyKindName(cudaMemcpyDeviceToDevice), stream);
    return &res;
}


dscudaMallocPitchResult *
dscudamallocpitchid_1_svc(RCsize width, RCsize height, struct svc_req *sr)
{
    static dscudaMallocPitchResult res;
    cudaError_t err;
    int *devadr;
    size_t pitch;

    WARN(3, "cudaMallocPitch(");
    if (!dscuContext) createDscuContext();
    err = cudaMallocPitch((void**)&devadr, &pitch, width, height);
    res.devPtr = (RCadr)devadr;
    res.pitch = pitch;
    check_cuda_error(err);
    res.err = err;
    WARN(3, "0x%08llx, %d, %d) done. devadr:0x%08llx\n", &devadr, width, height, devadr);

    return &res;
}

dscudaMemcpy2DToArrayH2HResult *
dscudamemcpy2dtoarrayh2hid_1_svc(RCadr dst, RCsize wOffset, RCsize hOffset, RCbuf src, RCsize spitch, RCsize width, RCsize height, struct svc_req *sr)
{
    static dscudaMemcpy2DToArrayH2HResult res;
    WARN(0, "dscudaMemcpy2DToArray() does not support cudaMemcpyHostToHost transfer yet.\n");
    return &res;
}

dscudaResult *
dscudamemcpy2dtoarrayh2did_1_svc(RCadr dst, RCsize wOffset, RCsize hOffset, RCbuf srcbuf, RCsize spitch, RCsize width, RCsize height, struct svc_req *sr)
{
    static dscudaResult res;
    cudaError_t err;

    WARN(3, "cudaMemcpy2DToArray(");
    if (!dscuContext) createDscuContext();
    err = cudaMemcpy2DToArray((cudaArray*)dst, wOffset, hOffset, srcbuf.RCbuf_val, spitch, width, height, cudaMemcpyHostToDevice);
    check_cuda_error(err);
    res.err = err;
    WARN(3, "0x%08lx, %d, %d, 0x%08llx, %d, %d, %d, %s) done.\n",
         dst, wOffset, hOffset, (unsigned long)srcbuf.RCbuf_val, spitch, width, height, dscudaMemcpyKindName(cudaMemcpyHostToDevice));
    return &res;
}

dscudaMemcpy2DToArrayD2HResult *
dscudamemcpy2dtoarrayd2hid_1_svc(RCsize wOffset, RCsize hOffset, RCadr src, RCsize spitch, RCsize width, RCsize height, struct svc_req *sr)
{
    static RCsize maxcount = 0;
    cudaError_t err;
    static dscudaMemcpy2DToArrayD2HResult res;
    int count = spitch * height;

    WARN(3, "cudaMemcpy2DToArray(");
    if (!dscuContext) createDscuContext();
    if (maxcount == 0) {
        res.buf.RCbuf_val = NULL;
    }
    if (maxcount < count) {
        res.buf.RCbuf_val = (char*)realloc(res.buf.RCbuf_val, count);
        maxcount = count;
    }
    res.buf.RCbuf_len = count;
    err = cudaMemcpy2DToArray((cudaArray *)res.buf.RCbuf_val, wOffset, hOffset, (void *)src, spitch, width, height, cudaMemcpyDeviceToHost);
    check_cuda_error(err);
    res.err = err;
    WARN(3, "0x%08llx, %d, %d, 0x%08llx, %d, %d, %d, %s) done. 2D buf size : %d\n",
         (unsigned long)res.buf.RCbuf_val, wOffset, hOffset, src, spitch, width, height, dscudaMemcpyKindName(cudaMemcpyDeviceToHost), count);
    return &res;
}

dscudaResult *
dscudamemcpy2dtoarrayd2did_1_svc(RCadr dst, RCsize wOffset, RCsize hOffset, RCadr src, RCsize spitch, RCsize width, RCsize height, struct svc_req *sr)
{
    cudaError_t err;
    static dscudaResult res;
    WARN(3, "cudaMemcpy2DToArray(");
    err = cudaMemcpy2DToArray((cudaArray *)dst, wOffset, hOffset, (void *)src, spitch, width, height, cudaMemcpyDeviceToDevice);
    check_cuda_error(err);
    res.err = err;
    WARN(3, "0x%08llx, %d, %d, 0x%08llx, %d, %d, %d, %s) done.\n",
         dst, wOffset, hOffset, src, spitch, width, height, dscudaMemcpyKindName(cudaMemcpyDeviceToDevice));
    return &res;
}

dscudaMemcpy2DH2HResult *
dscudamemcpy2dh2hid_1_svc(RCadr dst, RCsize dpitch, RCbuf src, RCsize spitch, RCsize width, RCsize height, struct svc_req *sr)
{
    static dscudaMemcpy2DH2HResult res;
    WARN(0, "dscudaMemcpy2D() does not support cudaMemcpyHostToHost transfer yet.\n");
    return &res;
}

dscudaResult *
dscudamemcpy2dh2did_1_svc(RCadr dst, RCsize dpitch, RCbuf srcbuf, RCsize spitch, RCsize width, RCsize height, struct svc_req *sr)
{
    static dscudaResult res;
    cudaError_t err;

    WARN(3, "cudaMemcpy2D(");
    if (!dscuContext) createDscuContext();
    err = cudaMemcpy2D((void*)dst, dpitch, srcbuf.RCbuf_val, spitch, width, height, cudaMemcpyHostToDevice);
    check_cuda_error(err);
    res.err = err;
    WARN(3, "0x%08lx, %d, 0x%08lx, %d, %d, %d, %s) done.\n",
         dst, dpitch, (unsigned long)srcbuf.RCbuf_val, spitch, width, height, dscudaMemcpyKindName(cudaMemcpyHostToDevice));
    return &res;
}

dscudaMemcpy2DD2HResult *
dscudamemcpy2dd2hid_1_svc(RCsize dpitch, RCadr src, RCsize spitch, RCsize width, RCsize height, struct svc_req *sr)
{
    static RCsize maxcount = 0;
    cudaError_t err;
    static dscudaMemcpy2DD2HResult res;
    int count = spitch * height;

    WARN(3, "cudaMemcpy2D(");
    if (!dscuContext) createDscuContext();
    if (maxcount == 0) {
        res.buf.RCbuf_val = NULL;
    }
    if (maxcount < count) {
        res.buf.RCbuf_val = (char*)realloc(res.buf.RCbuf_val, count);
        maxcount = count;
    }
    res.buf.RCbuf_len = count;
    err = cudaMemcpy2D(res.buf.RCbuf_val, dpitch, (void *)src, spitch, width, height, cudaMemcpyDeviceToHost);
    check_cuda_error(err);
    res.err = err;
    WARN(3, "0x%08llx, %d, 0x%08llx, %d, %d, %d, %s) done. 2D buf size : %d\n",
         (unsigned long)res.buf.RCbuf_val, dpitch, src, spitch, width, height, dscudaMemcpyKindName(cudaMemcpyDeviceToHost), count);
    return &res;
}

dscudaResult *
dscudamemcpy2dd2did_1_svc(RCadr dst, RCsize dpitch, RCadr src, RCsize spitch, RCsize width, RCsize height, struct svc_req *sr)
{
    cudaError_t err;
    static dscudaResult res;
    WARN(3, "cudaMemcpy2D(");
    err = cudaMemcpy2D((void *)dst, dpitch, (void *)src, spitch, width, height, cudaMemcpyDeviceToDevice);
    check_cuda_error(err);
    res.err = err;
    WARN(3, "0x%08llx, %d, 0x%08llx, %d, %d, %d, %s) done.\n",
         dst, dpitch, src, spitch, width, height, dscudaMemcpyKindName(cudaMemcpyDeviceToDevice));
    return &res;
}

dscudaResult *
dscudamemset2did_1_svc(RCadr dst, RCsize pitch, int value, RCsize width, RCsize height, struct svc_req *sq)
{
    static dscudaResult res;
    cudaError_t err;

    WARN(3, "cudaMemset2D(");
    if (!dscuContext) createDscuContext();
    err = cudaMemset2D((void *)dst, pitch, value, width, height);
    check_cuda_error(err);
    res.err = err;
    WARN(3, "0x%08llx, %d, %d, %d, %d) done.\n", dst, pitch, value, width, height);
    return &res;
}


/*
 * Texture Reference Management
 */

dscudaCreateChannelDescResult *
dscudacreatechanneldescid_1_svc(int x, int y, int z, int w, RCchannelformat f, struct svc_req *sr)
{
    static dscudaCreateChannelDescResult res;
    cudaChannelFormatDesc desc;

    WARN(3, "cudaCreateChannelDesc(");
    if (!dscuContext) createDscuContext();
    desc = cudaCreateChannelDesc(x, y, z, w, (enum cudaChannelFormatKind)f);
    res.x = desc.x;
    res.y = desc.y;
    res.z = desc.z;
    res.w = desc.w;
    res.f = desc.f;
    WARN(3, "%d, %d, %d, %d, %d) done.\n", x, y, z, w, f)
    return &res;
}

dscudaGetChannelDescResult *
dscudagetchanneldescid_1_svc(RCadr array, struct svc_req *sr)
{
    static dscudaGetChannelDescResult res;
    cudaError_t err;
    cudaChannelFormatDesc desc;

    WARN(3, "cudaGetChannelDesc(");
    if (!dscuContext) createDscuContext();
    err = cudaGetChannelDesc(&desc, (const struct cudaArray*)array);
    res.err = err;
    res.x = desc.x;
    res.y = desc.y;
    res.z = desc.z;
    res.w = desc.w;
    res.f = desc.f;
    WARN(3, "0x%08llx, 0x&08llx) done.\n", &desc, array)
    return &res;
}

dscudaBindTextureResult *
dscudabindtextureid_1_svc(int moduleid, char *texname, RCadr devPtr, RCsize size, RCtexture texbuf, struct svc_req *sr)
{
    static dscudaBindTextureResult res;
    cudaError_t err;
    CUtexref texref;
    Module *mp = Modulelist + moduleid;

    if (!dscuContext) createDscuContext();

    err = (cudaError_t)cuModuleGetTexRef(&texref, mp->handle, texname);
    WARN(3, "cuModuleGetTexRef(0x%08llx, 0x%08llx, %s) : module: %s\n",
         &texref, mp->handle, texname, mp->name);
    if (err != cudaSuccess) {
        check_cuda_error(err);
        res.err = err;
        return &res;
    }

    err = setTextureParams(texref, texbuf, texname);
    if (err != cudaSuccess) {
        res.err = err;
        return &res;
    }

    WARN(4, "cuTexRefSetAddress(0x%08llx, 0x%08llx, 0x%08llx, %d)\n", &res.offset, texref, devPtr, size);
    err = (cudaError_t)cuTexRefSetAddress((size_t *)&res.offset, texref, (CUdeviceptr)devPtr, size);
    if (err != cudaSuccess) {
        check_cuda_error(err);
        res.err = err;
        return &res;
    }
    res.err = err;

    return &res;
}

dscudaBindTexture2DResult *
dscudabindtexture2did_1_svc(int moduleid, char *texname, RCadr devPtr, RCsize width, RCsize height, RCsize pitch, RCtexture texbuf, struct svc_req *sr)
{
    static dscudaBindTexture2DResult res;
    cudaError_t err;
    CUtexref texref;
    Module *mp = Modulelist + moduleid;
    CUDA_ARRAY_DESCRIPTOR desc;

    if (!dscuContext) createDscuContext();

    err = (cudaError_t)cuModuleGetTexRef(&texref, mp->handle, texname);
    WARN(3, "cuModuleGetTexRef(0x%08llx, 0x%08llx, %s) : module: %s\n",
         &texref, mp->handle, texname, mp->name);
    if (err != cudaSuccess) {
        check_cuda_error(err);
        res.err = err;
        return &res;
    }

    err = setTextureParams(texref, texbuf, texname, &desc);
    if (err != cudaSuccess) {
        res.err = err;
        return &res;
    }
    desc.Height = height;
    desc.Width  = width;

    WARN(4, "cuTexRefSetAddress2D(0x%08llx, 0x%08llx, 0x%08llx, %d)\n", texref, desc, devPtr, pitch);
    err = (cudaError_t)cuTexRefSetAddress2D(texref, &desc, (CUdeviceptr)devPtr, pitch);
    if (err != cudaSuccess) {
        check_cuda_error(err);
        res.err = err;
        return &res;
    }
    res.err = err;

    unsigned int align = CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT;
    unsigned long int roundup_adr = ((devPtr - 1) / align + 1) * align;
    res.offset = roundup_adr - devPtr;
    WARN(4, "align:0x%x  roundup_adr:0x%08llx  devPtr:0x%08llx  offset:0x%08llx\n",
         align, roundup_adr, devPtr, res.offset);
    return &res;
}

dscudaResult *
dscudabindtexturetoarrayid_1_svc(int moduleid, char *texname, RCadr array, RCtexture texbuf, struct svc_req *sr)
{
    static dscudaResult res;
    cudaError_t err;
    CUtexref texref;
    Module *mp = Modulelist + moduleid;

    if (!dscuContext) createDscuContext();

    err = (cudaError_t)cuModuleGetTexRef(&texref, mp->handle, texname);
    WARN(3, "cuModuleGetTexRef(0x%08llx, 0x%08llx, %s) : module: %s  moduleid:%d\n",
         &texref, mp->handle, texname, mp->name, moduleid);
    if (err != cudaSuccess) {
        check_cuda_error(err);
        res.err = err;
        return &res;
    }

    err = setTextureParams(texref, texbuf, texname);
    if (err != cudaSuccess) {
        res.err = err;
        return &res;
    }

    WARN(4, "cuTexRefSetArray(0x%08llx, 0x%08llx, %d)\n", texref, array, CU_TRSA_OVERRIDE_FORMAT);
    err = (cudaError_t)cuTexRefSetArray(texref, (CUarray)array, CU_TRSA_OVERRIDE_FORMAT);
    if (err != cudaSuccess) {
        check_cuda_error(err);
        res.err = err;
        return &res;
    }
    res.err = err;
    return &res;
}

dscudaResult *
dscudaunbindtextureid_1_svc(RCtexture texrefbuf, struct svc_req *sr)
{
    static dscudaResult res;
    cudaError_t err = cudaSuccess;

    WARN(4, "Current implementation of cudaUnbindTexture() does nothing "
         "but returning cudaSuccess.\n");

    res.err = err;
    return &res;
}

dscudaLoadModuleResult *
dscudaloadmoduleid_1_svc(RCipaddr ipaddr, RCpid pid, char *mname, char *image, struct svc_req *sr)
{
    static dscudaLoadModuleResult res;
    res.id = dscudaLoadModule(ipaddr, pid, mname, image);
    return &res;
}

/*
 * launch a kernel function of id 'kid' (or name 'kname', if it's not loaded yet),
 * defined in a module of id 'moduleid'.
 */
void *
dscudalaunchkernelid_1_svc(int moduleid, int kid, char *kname,
                          RCdim3 gdim, RCdim3 bdim, RCsize smemsize, RCstream stream, RCargs args, struct svc_req *sr)
{
    static int dummyres = 0;
    dscudaLaunchKernel(moduleid, kid, kname, gdim, bdim, smemsize, stream, args);
    return &dummyres; // seems necessary to return something even if it's not used by the client.
}


/*
 * CUFFT library
 */
dscufftPlanResult *
dscufftplan3did_1_svc(int nx, int ny, int nz, unsigned int type, struct svc_req *sr)
{
    static dscufftPlanResult res;
    cufftResult err = CUFFT_SUCCESS;
    cufftHandle plan;

    WARN(3, "cufftplan1d(");
    err = cufftPlan3d(&plan, nx, ny, nz, (cufftType)type);
    WARN(3, "%d, %d, %d, %d, %d) done.\n", plan, nx, ny, nz, type);

    check_cuda_error((cudaError_t)err);
    res.err = err;
    res.plan = (unsigned int)plan;
    return &res;
}

dscufftResult *
dscufftdestroyid_1_svc(unsigned int plan, struct svc_req *sr)
{
    static dscufftResult res;
    cufftResult err = CUFFT_SUCCESS;

    WARN(3, "cufftDestroy(")
    err = cufftDestroy((cufftHandle)plan);
    WARN(3, "%d) done.\n", plan);

    res.err = err;
    return &res;
}

dscufftResult *
dscufftexecc2cid_1_svc(unsigned int plan, RCadr idata, RCadr odata, int direction, struct svc_req *sr)
{
    static dscufftResult res;
    cufftResult err = CUFFT_SUCCESS;

    WARN(3, "cufftExecC2C(");
    err = cufftExecC2C((cufftHandle)plan, (cufftComplex *)idata, (cufftComplex *)odata, direction);
    WARN(3, "%d, %p, %p, %d) done.\n", plan, idata, odata, direction);

    res.err = err;
    return &res;
}

#if 0 // CUFFT

/*
 * Interface to CUFFT & CUBLAS written by Yoshikawa for old Remote CUDA.
 * some are already ported to DS-CUDA (see 'dscufftXXXid_1_svc' function defs above),
 * but some are not. Maybe someday, when I have time...
 */

/*
 * cufft library functions
 */

rcufftPlanResult *
rcufftplan1did_1_svc(int nx, unsigned int type, int batch, struct svc_req *sr)
{
    static rcufftPlanResult res;
    cufftResult err = CUFFT_SUCCESS;
        cufftHandle plan;

    WARN(3, "cufftplan1d(");
    cufftPlan1d(&plan, nx, (cufftType)type, batch);
    WARN(3, "%d, %d, %d, %d) done.\n", plan, nx, type, batch);

    check_cuda_error((cudaError_t)err);
    res.err = err;
        res.plan = (unsigned int)plan;
    return &res;
}

rcufftPlanResult *
rcufftplan2did_1_svc(int nx, int ny, unsigned int type, struct svc_req *sr)
{
    static rcufftPlanResult res;
    cufftResult err = CUFFT_SUCCESS;
        cufftHandle plan;

    WARN(3, "cufftplan1d(");
    cufftPlan2d(&plan, nx, ny, (cufftType)type);
    WARN(3, "%d, %d, %d, %d) done.\n", plan, nx, ny, type);

    check_cuda_error((cudaError_t)err);
    res.err = err;
        res.plan = (unsigned int)plan;
    return &res;
}

/*
rcufftplanresult *
rcufftplanmanyid_1_svc(int nx, int ny, int nz, unsigned int type, int batch, struct svc_req *sr)
{
    static rcufftPlan1dResult res;
    cufftResult err = CUFFT_SUCCESS;
        cufftHandle plan;

    WARN(3, "cufftplan1d(");
    cufftPlanMany(&plan, nx, ny, nz, (cufftType)type, batch);
    WARN(3, "%d, %d, %d, %d, %d, %d) done.", plan, nx, ny, nz, type, batch);

    check_cuda_error((cudaError_t)err);
    res.err = err;
        res.plan = (unsigned int)plan;
    return &res;
}
*/

rcufftResult *
rcufftexecr2cid_1_svc(unsigned int plan, RCadr idata, RCadr odata, struct svc_req *sr)
{
    static rcufftResult res;
    cufftResult err = CUFFT_SUCCESS;

    WARN(3, "cufftExecR2C(");
    cufftExecR2C((cufftHandle)plan, (cufftReal *)idata, (cufftComplex *)odata);
        WARN(3, "%d, %p, %p) done.\n", plan, idata, odata);

    res.err = err;
    return &res;
}

rcufftResult *
rcufftexecc2rid_1_svc(unsigned int plan, RCadr idata, RCadr odata, struct svc_req *sr)
{
    static rcufftResult res;
    cufftResult err = CUFFT_SUCCESS;

    WARN(3, "cufftExecC2R(");
    cufftExecC2R((cufftHandle)plan, (cufftComplex *)idata, (cufftReal *)odata);
        WARN(3, "%d, %p, %p) done.\n", plan, idata, odata);

    res.err = err;
    return &res;
}

rcufftResult *
rcufftexecz2zid_1_svc(unsigned int plan, RCadr idata, RCadr odata, int direction, struct svc_req *sr)
{
    static rcufftResult res;
    cufftResult err = CUFFT_SUCCESS;

    WARN(3, "cufftExecZ2Z(");
    cufftExecZ2Z((cufftHandle)plan, (cufftDoubleComplex *)idata, (cufftDoubleComplex *)odata, direction);
        WARN(3, "%d, %p, %p, %d) done.\n", plan, idata, odata, direction);

    res.err = err;
    return &res;
}

rcufftResult *
rcufftexecd2zid_1_svc(unsigned int plan, RCadr idata, RCadr odata, struct svc_req *sr)
{
    static rcufftResult res;
    cufftResult err = CUFFT_SUCCESS;

    WARN(3, "cufftExecD2Z(");
    cufftExecD2Z((cufftHandle)plan, (cufftDoubleReal *)idata, (cufftDoubleComplex *)odata);
        WARN(3, "%d, %p, %p) done.\n", plan, idata, odata);

    res.err = err;
    return &res;
}

rcufftResult *
rcufftexecz2did_1_svc(unsigned int plan, RCadr idata, RCadr odata, struct svc_req *sr)
{
    static rcufftResult res;
    cufftResult err = CUFFT_SUCCESS;

    WARN(3, "cufftExecZ2D(");
    cufftExecZ2D((cufftHandle)plan, (cufftDoubleComplex *)idata, (cufftDoubleReal *)odata);
        WARN(3, "%d, %p, %p) done.\n", plan, idata, odata);

    res.err = err;
    return &res;
}

/*
rcufftResult *
rcufftSetStreamId()
*/

rcufftResult *
rcufftsetcompatibilitymodeid_1_svc(unsigned int plan, unsigned int mode, struct svc_req *sr)
{
    static rcufftResult res;
    cufftResult err = CUFFT_SUCCESS;

    WARN(3, "cufftSetCompatibilityMode(");
    cufftSetCompatibilityMode((cufftHandle)plan, (cufftCompatibility)mode);
    WARN(3, "%d, %d) done.\n", plan, mode);

    res.err = err;
    return &res;
}


/*
 * cublas library functions
 */

rcublasCreateResult *
rcublascreate_v2id_1_svc(struct svc_req *sr)
{
    static rcublasCreateResult res;
    cudaError_t err = cudaSuccess;
    cublasStatus_t stat = CUBLAS_STATUS_SUCCESS;
    cublasHandle_t handle;

    WARN(3, "cublasCreate(");
    stat = cublasCreate(&handle);
    WARN(3, "%p) done.\n", &handle);

    res.err = err;
    res.stat = stat;
    res.handle = (RCadr)handle;

    return &res;
}

rcublasResult *
rcublasdestroy_v2id_1_svc(RCadr handle, struct svc_req *sr)
{
    static rcublasResult res;
    cudaError_t err = cudaSuccess;
    cublasStatus_t stat = CUBLAS_STATUS_SUCCESS;

    WARN(3, "cublasDestroy(");
    stat = cublasDestroy((cublasHandle_t)handle);
    WARN(3, "%d) done.\n", handle);

    res.err = err;
    res.stat = stat;

    return &res;
}

rcublasResult *
rcublassetvectorid_1_svc(int n, int elemSize, RCbuf x, int incx, RCadr y, int incy, struct svc_req *sr)
{
    static rcublasResult res;
    cudaError_t err = cudaSuccess;
    cublasStatus_t stat = CUBLAS_STATUS_SUCCESS;

    WARN(3, "cublasSetVector(");
    stat = cublasSetVector(n, elemSize, (const void *)x.RCbuf_val, incx, (void *)y, incy);
    WARN(3, "%d, %d, %p, %d, %p, %d) done.\n", n, elemSize, x.RCbuf_val, incx, y, incy);

    res.err = err;
    res.stat = stat;

    return &res;
}

rcublasGetVectorResult *
rcublasgetvectorid_1_svc(int n, int elemSize, RCadr x, int incx, int incy, struct svc_req *sr)
{
    static rcublasGetVectorResult res;
    cudaError_t err = cudaSuccess;
    cublasStatus_t stat = CUBLAS_STATUS_SUCCESS;

    res.y.RCbuf_val = (char *)malloc(n * elemSize);
    res.y.RCbuf_len = n * elemSize;

    WARN(3, "cublasGetVector(");
    stat = cublasGetVector(n, elemSize, (const void *)x, incx, (void *)res.y.RCbuf_val, incy);
    WARN(3, "%d, %d, %p, %d, %p, %d) done.\n", n, elemSize, x, incx, res.y.RCbuf_val, incy);

    res.err = err;
    res.stat = stat;

    return &res;
}

rcublasResult *
rcublassgemm_v2id_1_svc(RCadr handle, unsigned int transa, unsigned int transb, int m, int n, int k, float alpha, RCadr A, int lda, RCadr B, int ldb, float beta, RCadr C, int ldc, struct svc_req *sr)
{
    static rcublasResult res;
    cudaError_t err = cudaSuccess;
    cublasStatus_t stat = CUBLAS_STATUS_SUCCESS;

    WARN(3, "cublasSgemm(");
    stat = cublasSgemm((cublasHandle_t)handle, (cublasOperation_t)transa, (cublasOperation_t)transb, m, n, k,
                           (const float *)&alpha, (const float *)A, lda, (const float *)B, ldb, (const float *)&beta, (float *)C, ldc);
    WARN(3, "%p, %d, %d, %d, %d, %d, %f, %p, %d, %p, %d, %f, %p, %d) done.\n", handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);

    res.err = err;
    res.stat = stat;

    return &res;
}

#endif // CUFFT
