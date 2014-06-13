/*
 * this file is included into the bottom of libdscuda_ibv.cu & libdscuda_rpc.cu.
 */

static int
requestDaemonForDevice(char *ipaddr, int devid, int useibv)
{
    int dsock; // socket for side-band communication with the daemon & server.
    int sport; // port number of the server. given by the daemon.
    char msg[256];
    struct sockaddr_in sockaddr;

    sockaddr = setupSockaddr(ipaddr, RC_DAEMON_IP_PORT);
    dsock = socket(AF_INET, SOCK_STREAM, 0);
    if (dsock < 0) {
        perror("socket");
        exit(1);
    }
    if (connect(dsock, (struct sockaddr *)&sockaddr, sizeof(sockaddr)) == -1) {
        perror("connect");
        exit(1);
    }
    sprintf(msg, "deviceid:%d", devid);
    sendMsgBySocket(dsock, msg);

    memset(msg, 0, strlen(msg));
    recvMsgBySocket(dsock, msg, sizeof(msg));
    sscanf(msg, "sport:%d", &sport);

    if (sport < 0) {
        WARN(0, "max possible ports on %s already in use.\n", ipaddr);
        exit(1);
    }

    WARN(3, "server port: %d  daemon socket: %d\n", sport, dsock);

    if (useibv) {
        sprintf(msg, "remotecall:ibv");
    }
    else {
        sprintf(msg, "remotecall:rpc");
    }
    WARN(3, "send \"%s\" to the server.\n", msg);
    sendMsgBySocket(dsock, msg);

    WARN(2, "waiting for the server to be set up...\n");
    memset(msg, 0, strlen(msg));
    recvMsgBySocket(dsock, msg, sizeof(msg)); // wait for "ready" from the server.
    if (strncmp("ready", msg, strlen("ready"))) {
        WARN(0, "unexpected message (\"%s\") from the server. abort.\n", msg);
        exit(1);
    }
    return sport;
}

/*
 * Obtain a small integer unique for each thread.
 * The integer is used as an index to 'Vdevid[]'.
 */
static pthread_mutex_t VdevidMutex = PTHREAD_MUTEX_INITIALIZER;
static int
vdevidIndex(void)
{
    int i;
    pthread_t ptid = pthread_self();

    for (i = 0; i < VdevidIndexMax; i++) {
        if (VdevidIndex2ptid[i] == ptid) {
            return i;
        }
    }

    pthread_mutex_lock(&VdevidMutex);
    i = VdevidIndexMax;
    VdevidIndex2ptid[i] = ptid;
    VdevidIndexMax++;
    pthread_mutex_unlock(&VdevidMutex);

    if (RC_NPTHREADMAX <= VdevidIndexMax) {
        fprintf(stderr, "vdevidIndex():device requests from too many (more than %d) pthreads.\n", RC_NPTHREADMAX);
        exit(1);
    }

    return i;
}

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

/*
 * Register an array of UVA. each component is associated to an UVA
 * on each Server[i]. User see only the 1st element, [0].
 * Others, i.e., Server[1..Nredunddancy-1], are hidden to the user,
 * and used by this library to handle redundant calculation mechanism.
 */
static void
RCuvaRegister(int devid, void *adr[], size_t size)
{
    int i;
    int nredundancy = dscudaNredundancy();
    RCuva *uva = (RCuva *)malloc(sizeof(RCuva));

    if (!uva) {
        perror("RCuvaRegister");
    }
    for (i = 0; i < nredundancy; i++) {
        uva->adr[i] = adr[i];
    }
    uva->devid = devid;
    uva->size = size;
    uva->prev = RCuvaListTail;
    uva->next = NULL;
    if (!RCuvaListTop) { // uva will be the 1st entry.
        RCuvaListTop = uva;
    }
    else {
        RCuvaListTail->next = uva;
    }
    RCuvaListTail = uva;
}

static void
RCuvaUnregister(void *adr)
{
    RCuva *uva = RCuvaQuery(adr);

    if (!uva) return;

    if (uva->prev) { // reconnect the linked list.
        uva->prev->next = uva->next;
    }
    else { // uva was the 1st entry.
        RCuvaListTop = uva->next;
        if (uva->next) {
            uva->next->prev = NULL;
        }
    }
    if (!uva->next) { // uva was the last entry.
        RCuvaListTail = uva->prev;
    }
    free(uva);
}

static RCuva *
RCuvaQuery(void *adr)
{
    RCuva *uva = RCuvaListTop;
    while (uva) {
        if (uva->adr[0] <= adr && adr < (char *)uva->adr[0] + uva->size) {
            return uva;
        }
        uva = uva->next;
    }
    return NULL; // uva not found in the list.
}


static char*
readServerConf(char *fname)
{
    FILE *fp = fopen(fname, "r");
    char linebuf[1024];
    int len;
    static char buf[1024 * RC_NVDEVMAX];

    buf[0] = 0;
    if (!fp) {
        WARN(0, "cannot open file '%s'\n", fname);
        exit(1);
    }

    while (!feof(fp)) {
        char *s = fgets(linebuf, sizeof(linebuf), fp);
        if (!s) break;
        len = strlen(linebuf);
        if (linebuf[len-1] == '\n') {
            linebuf[len-1] = 0;
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
    char buf[1024 * RC_NVDEVMAX];
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
        if (sizeof(buf) < strlen(env)) {
            WARN(0, "initEnv:evironment variable DSCUDA_SERVER too long.\n");
            exit(1);
        }
        strncpy(buf, env, sizeof(buf));
        Nvdev = 0;
        ip = strtok(buf, " "); // a list of IPs which consist a single vdev.
        while (ip) {
            strcpy(ips[Nvdev], ip);
            Nvdev++;
            if (RC_NVDEVMAX < Nvdev) {
                WARN(0, "initEnv:number of devices exceeds the limit, RC_NVDEVMAX (=%d).\n",
                     RC_NVDEVMAX);
                exit(1);
            }
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
            WARN(3, "    %s:%d\n", sp->ip, sp->cid);
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

    switch (dscudaRemoteCallType()) {
      case RC_REMOTECALL_TYPE_RPC:
        WARN(2, "method of remote procedure call: RPC\n");
        break;
      case RC_REMOTECALL_TYPE_IBV:
        WARN(2, "method of remote procedure call: InfiniBand Verbs\n");
    }

    // DSCUDA_AUTOVERB
    env = getenv("DSCUDA_AUTOVERB");
    WARN(2, "automatic data recovery: ");
    if (env) {
        dscudaSetAutoVerb(1);
        dscudaVerbInit();
        WARN(2, "on");
    }
    else {
        WARN(2, "off");
    }
    WARN(2, "\n");

    // DSCUDA_USEDAEMON
    env = getenv("DSCUDA_USEDAEMON");
    if (env && atoi(env)) {
        WARN(3, "connect to the server via daemon.\n");
        UseDaemon = 1;
    }
    else {
        WARN(3, "do not use daemon. connect to the server directly.\n");
        UseDaemon = 0;
    }
}

static pthread_mutex_t InitClientMutex = PTHREAD_MUTEX_INITIALIZER;
static void
initClient(void)
{
    static int firstcall = 1;

    pthread_mutex_lock(&InitClientMutex);

    if (!firstcall) {
        pthread_mutex_unlock(&InitClientMutex);
        return;
    }

    initEnv();

    for (int idev = 0; idev < Nvdev; idev++) {
        Vdev_t *vdev = Vdev + idev;
        RCServer_t *sp = vdev->server;
        for (int ired = 0; ired < vdev->nredundancy; ired++, sp++) {
            setupConnection(idev, sp);
        }
    }
    struct sockaddr_in addrin;
    get_myaddress(&addrin);
    MyIpaddr = addrin.sin_addr.s_addr;
    WARN(2, "Client IP address : %s\n", dscudaGetIpaddrString(MyIpaddr));
    firstcall = 0;
    pthread_mutex_unlock(&InitClientMutex);
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

/*
 * public functions
 */

int
dscudaNredundancy(void)
{
    Vdev_t *vdev = Vdev + Vdevid[vdevidIndex()];
    return vdev->nredundancy;
}

void
dscudaSetAutoVerb(int verb)
{
    autoVerb = verb;
    return;
}

void
dscudaSetErrorHandler(void (*handler)(void *), void *handler_arg)
{
    errorHandler = handler;
    errorHandlerArg = handler_arg;
}


/*
 * Obtain a mangled symbol name of a function, whose
 * interface is given by 'funcif' and is defined somewhere in 'ptxdata'.
 * The obtained symbol name is returned to 'name'.
 *
 * eg) funcif  : void dscudavecAdd(dim3, dim3, size_t, CUstream_st*, float*, float*, float*)
 *     ptxdata : .version 1.4
 *               .target sm_10, map_f64_to_f32
 *               ...
 *               .entry _Z6vecAddPfS_S_ (
 *               ...
 *               } // _Z6vecMulPfS_fS_iPi
 */
void
dscudaGetMangledFunctionName(char *name, const char *funcif, const char *ptxdata)
{
    static char mangler[256] = {0, };
    char cmd[4096];
    FILE *outpipe;
    FILE *tmpfp;
    char ptxfile[1024];

    WARN(4, "getMangledFunctionName(%08llx, %08llx, %08llx)  funcif:\"%s\"\n",
         name, funcif, ptxdata, funcif);

    // create a tmporary file that contains 'ptxdata'.
    system("/bin/mkdir /tmp/dscuda >& /dev/null");
    sprintf(ptxfile, "/tmp/dscuda/mgl%d", getpid());
    tmpfp = fopen(ptxfile, "w");
    fprintf(tmpfp, "%s", ptxdata);
    fclose(tmpfp);

    // exec 'ptx2symbol' to obtain the mangled name.
    // command output is stored to name.
    if (!mangler[0]) {
        sprintf(mangler, "%s/bin/ptx2symbol", Dscudapath);
    }
    sprintf(cmd, "%s %s << EOF\n%s\nEOF", mangler, ptxfile, funcif);
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


static pthread_mutex_t LoadModuleMutex = PTHREAD_MUTEX_INITIALIZER;
/*
 * Load a cuda module from a .ptx file, and then, send it to the server.
 * returns id for the module.
 * the module is cached and sent only once for a certain period.
 */
int *
dscudaLoadModule(char *name, char *strdata) // 'strdata' must be NULL terminated.
{
    int i, j, mid;
    Module *mp;

    WARN(5, "dscudaLoadModule(0x%08llx) modulename:%s  ...", name, name);

#if RC_CACHE_MODULE
    // look for modulename in the module list.
    for (i = 0, mp = Modulelist; i < RC_NKMODULEMAX; i++, mp++) {
        if (!mp->valid) continue;
        if (mp->vdevid != Vdevid[vdevidIndex()]) continue;
        if (!strcmp(name, mp->name)) {
            if (time(NULL) - mp->sent_time < RC_CLIENT_CACHE_LIFETIME) {
                WARN(5, "done. found a cached one. id:%d  age:%d  name:%s\n",
                     mp->id[i], time(NULL) - mp->sent_time, mp->name);
                return mp->id; // module found. i.e, it's already loaded.
            }
            WARN(5, "found a cached one with id:%d, but it is too old (age:%d). resend it.\n",
                 mp->id[i], time(NULL) - mp->sent_time);
            mp->valid = 0; // invalidate the cache.
        }
    }
#endif // RC_CACHE_MODULE

    // module not found in the module list.
    // really need to send it to the server.
    int vi = vdevidIndex();
    Vdev_t *vdev = Vdev + Vdevid[vi];
    for (i = 0; i < vdev->nredundancy; i++) {

        mid = dscudaLoadModuleLocal(MyIpaddr, getpid(), name, strdata, Vdevid[vi], i);

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
            strncpy(mp->name, name, sizeof(mp->name));
            WARN(5, "done. newly registered. id:%d\n", mid);
        }
        mp->id[i] = mid;
    }
    mp->vdevid = Vdevid[vi];

    return mp->id;
}

cudaError_t
dscudaFuncGetAttributesWrapper(int *moduleid, struct cudaFuncAttributes *attr, const char *func)
{
    cudaError_t err = cudaSuccess;
    dscudaFuncGetAttributesResult *rp;

    initClient();
    WARN(3, "dscudaFuncGetAttributesWrapper(%d, 0x%08llx, %s)...",
         moduleid, (unsigned long)attr, func);
    Vdev_t *vdev = Vdev + Vdevid[vdevidIndex()];
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        if (UseIbv) {
#warning fill this part in dscudaFuncGetAttributesWrapper().
        }
        else {
            rp = dscudafuncgetattributesid_1(moduleid[i], (char*)func, Clnt[Vdevid[vdevidIndex()]][sp->id]);
            checkResult(rp, sp);
            if (rp->err != cudaSuccess) {
                err = (cudaError_t)rp->err;
            }
            if (i == 0) {
                attr->binaryVersion      = rp->attr.binaryVersion;
                attr->constSizeBytes     = rp->attr.constSizeBytes;
                attr->localSizeBytes     = rp->attr.localSizeBytes;
                attr->maxThreadsPerBlock = rp->attr.maxThreadsPerBlock;
                attr->numRegs            = rp->attr.numRegs;
                attr->ptxVersion         = rp->attr.ptxVersion;
                attr->sharedSizeBytes    = rp->attr.sharedSizeBytes;
            }
            xdr_free((xdrproc_t)xdr_dscudaFuncGetAttributesResult, (char *)rp);
        }
    }

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

cudaError_t
dscudaMemcpyToSymbolWrapper(int *moduleid, const char *symbol, const void *src,
                           size_t count, size_t offset, enum cudaMemcpyKind kind)
{
    cudaError_t err = cudaSuccess;
    int nredundancy;

    initClient();

    WARN(3, "dscudaMemcpyToSymbolWrapper(%d, 0x%08llx, 0x%08llx, %d, %d, %s)"
         "symbol:%s  ...",
         moduleid, (unsigned long)symbol, (unsigned long)src,
         count, offset, dscudaMemcpyKindName(kind), symbol);

    nredundancy = (Vdev + Vdevid[vdevidIndex()])->nredundancy;
    switch (kind) {
      case cudaMemcpyHostToDevice:
        for (int i = 0; i < nredundancy; i++) {
            err = dscudaMemcpyToSymbolH2D(moduleid[i], (char *)symbol, src, count, offset, Vdevid[vdevidIndex()], i);
        }
        break;
      case cudaMemcpyDeviceToDevice:
        for (int i = 0; i < nredundancy; i++) {
            err = dscudaMemcpyToSymbolD2D(moduleid[i], (char *)symbol, src, count, offset, Vdevid[vdevidIndex()], i);
        }
        break;
      default:
        WARN(0, "Unsupported value for cudaMemcpyKind : %s\n", dscudaMemcpyKindName(kind));
        exit(1);
    }
    WARN(3, "done.\n");

    if (autoVerb && (kind == cudaMemcpyHostToDevice || kind == cudaMemcpyDeviceToDevice)) {
        cudaMemcpyToSymbolArgs args;
        args.moduleid = moduleid;
        args.symbol = (char *)symbol;
        args.src = (void *)src;
        args.count = count;
        args.offset = offset;
        args.kind = kind;
        dscudaVerbAddHist(dscudaMemcpyToSymbolH2DId, (void *)&args);
    }

    return err;
}

cudaError_t
dscudaMemcpyFromSymbolWrapper(int *moduleid, void *dst, const char *symbol,
                             size_t count, size_t offset,
                             enum cudaMemcpyKind kind)
{
    cudaError_t err = cudaSuccess;
    int nredundancy;
    void *dstbuf;

    initClient();

    WARN(3, "dscudaMemcpyFromSymbolWrapper(0x%08llx, 0x%08llx, 0x%08llx, %d, %d, %s)"
         "symbol:%s  ...",
         moduleid, (unsigned long)dst, (unsigned long)symbol,
         count, offset, dscudaMemcpyKindName(kind), symbol);

    nredundancy = (Vdev + Vdevid[vdevidIndex()])->nredundancy;
    switch (kind) {
      case cudaMemcpyDeviceToHost:
        if (UseIbv) {
            dstbuf = calloc(1, count);
            if (!dstbuf) {
                WARN(0, "dscudaMemcpyFromSymbolWrapper:calloc() failed.\n");
                exit(1);
            }
        }

        for (int i = 0; i < nredundancy; i++) {
            err = dscudaMemcpyFromSymbolD2H(moduleid[i], &dstbuf, (char *)symbol, count, offset, Vdevid[vdevidIndex()], i);
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
        if (UseIbv) {
            free(dstbuf);
        }
        break;
      case cudaMemcpyDeviceToDevice:
        for (int i = 0; i < nredundancy; i++) {
            err = dscudaMemcpyFromSymbolD2D(moduleid[i], dst, (char *)symbol, count, offset, Vdevid[vdevidIndex()], i);
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
dscudaMemcpyToSymbolAsyncWrapper(int *moduleid, const char *symbol, const void *src,
                                 size_t count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream)
{
    cudaError_t err = cudaSuccess;
    RCstreamArray *st;
    int nredundancy;

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
    nredundancy = (Vdev + Vdevid[vdevidIndex()])->nredundancy;
    switch (kind) {
      case cudaMemcpyHostToDevice:
        for (int i = 0; i < nredundancy; i++) {
            err = dscudaMemcpyToSymbolAsyncH2D(moduleid[i], (char *)symbol, src, count, offset,
                                               (RCstream)st->s[i], Vdevid[vdevidIndex()], i);
        }
        break;
      case cudaMemcpyDeviceToDevice:
        for (int i = 0; i < nredundancy; i++) {
            err = dscudaMemcpyToSymbolAsyncD2D(moduleid[i], (char *)symbol, src, count, offset,
                                               (RCstream)st->s[i], Vdevid[vdevidIndex()], i);
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
dscudaMemcpyFromSymbolAsyncWrapper(int *moduleid, void *dst, const char *symbol,
                                   size_t count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream)
{
    cudaError_t err = cudaSuccess;
    RCstreamArray *st;
    int nredundancy;
    void *dstbuf;

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
    nredundancy = (Vdev + Vdevid[vdevidIndex()])->nredundancy;
    switch (kind) {
      case cudaMemcpyDeviceToHost:
        if (UseIbv) {
            dstbuf = calloc(1, count);
            if (!dstbuf) {
                WARN(0, "dscudaMemcpyFromSymbolAsyncWrapper:calloc() failed.\n");
                exit(1);
            }
        }
        for (int i = 0; i < nredundancy; i++) {
            err = dscudaMemcpyFromSymbolAsyncD2H(moduleid[i], &dstbuf, (char *)symbol, count, offset,
                                                 (RCstream)st->s[i], Vdevid[vdevidIndex()], i);
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
        if (UseIbv) {
            free(dstbuf);
        }
        break;
      case cudaMemcpyDeviceToDevice:
        for (int i = 0; i < nredundancy; i++) {
            err = dscudaMemcpyFromSymbolAsyncD2D(moduleid[i], dst, (char *)symbol, count, offset,
                                                 (RCstream)st->s[i], Vdevid[vdevidIndex()], i);
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

    Vdev_t *vdev = Vdev + Vdevid[vdevidIndex()];
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        if (UseIbv) {

#warning fill this part in dscudaBindTextureWrapper().
        }
        else {
            rp = dscudabindtextureid_1(moduleid[i], texname,
                                       (RCadr)devPtr, size, (RCtexture)texbuf, Clnt[Vdevid[vdevidIndex()]][sp->id]);
            checkResult(rp, sp);
            if (rp->err != cudaSuccess) {
                err = (cudaError_t)rp->err;
            }
            if (i == 0) {
                if (offset) {
                    *offset = rp->offset;
                }
            }
            xdr_free((xdrproc_t)xdr_dscudaBindTextureResult, (char *)rp);
        }
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

    Vdev_t *vdev = Vdev + Vdevid[vdevidIndex()];
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        if (UseIbv) {

#warning fill this part in dscudaBindTexture2DWrapper().
        }
        else {

            rp = dscudabindtexture2did_1(moduleid[i], texname,
                                         (RCadr)devPtr, width, height, pitch, (RCtexture)texbuf, Clnt[Vdevid[vdevidIndex()]][sp->id]);
            checkResult(rp, sp);
            if (rp->err != cudaSuccess) {
                err = (cudaError_t)rp->err;
            }
            if (i == 0) {
                if (offset) {
                    *offset = rp->offset;
                }
            }
            xdr_free((xdrproc_t)xdr_dscudaBindTexture2DResult, (char *)rp);
        }
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

    Vdev_t *vdev = Vdev + Vdevid[vdevidIndex()];
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        if (UseIbv) {

#warning fill this part in dscudaBindTextureToArrayWrapper().
        }
        else {

            rp = dscudabindtexturetoarrayid_1(moduleid[i], texname, (RCadr)ca->ap[i], (RCtexture)texbuf, Clnt[Vdevid[vdevidIndex()]][sp->id]);
            checkResult(rp, sp);
            if (rp->err != cudaSuccess) {
                err = (cudaError_t)rp->err;
            }
            xdr_free((xdrproc_t)xdr_dscudaResult, (char *)rp);
        }
    }
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
cudaGetDevice(int *device)
{
    cudaError_t err = cudaSuccess;

    initClient();
    WARN(3, "cudaGetDevice(0x%08llx)...", (unsigned long)device);
    *device = Vdevid[vdevidIndex()];
    WARN(3, "done.\n");

    return err;
}

cudaError_t
cudaSetDevice(int device)
{
    cudaError_t err = cudaSuccess;
    int vi = vdevidIndex();

    initClient();
    WARN(3, "cudaSetDevice(%d)...", device);

    if (0 <= device && device < Nvdev) {
        Vdevid[vi] = device;
    }
    else {
        err = cudaErrorInvalidDevice;
    }
    WARN(3, "done.\n");

    /*
    WARN(3, "vdevidIndex:%d  Vdevid[]: %d  %d  %d  %d    VdevidIndex2ptid[]: %d  %d  %d  %d\n",
         vdevidIndex(), Vdevid[0], Vdevid[1], Vdevid[2], Vdevid[3],
         VdevidIndex2ptid[0], VdevidIndex2ptid[1], VdevidIndex2ptid[2], VdevidIndex2ptid[3]);
    */

    if (autoVerb) {
        cudaSetDeviceArgs args;
        args.device = device;
        dscudaVerbAddHist(dscudaSetDeviceId, (void *)&args);
    }

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
cudaDeviceCanAccessPeer(int *canAccessPeer, int device, int peerDevice)
{
    cudaError_t err = cudaSuccess;

    WARN(3, "cudaDeviceCanAccessPeer(0x%08lx, %d, %d)...",
         canAccessPeer, device, peerDevice);
    if (device < 0 || Nvdev <= device) {
        err = cudaErrorInvalidDevice;
    }
    if (peerDevice < 0 || Nvdev <= peerDevice) {
        err = cudaErrorInvalidDevice;
    }
    WARN(3, "done.\n");

    return err;
}

cudaError_t
cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags)
{
    cudaError_t err = cudaSuccess;

    WARN(3, "cudaDeviceEnablePeer(%d, %d)...", peerDevice, flags);
    if (peerDevice < 0 || Nvdev <= peerDevice) {
        err = cudaErrorInvalidDevice;
    }
    WARN(3, "done.\n");

    return err;
}

cudaError_t
cudaDeviceDisablePeerAccess(int peerDevice)
{
    cudaError_t err = cudaSuccess;

    WARN(3, "cudaDeviceDisablePeer(%d)...", peerDevice);
    if (peerDevice < 0 || Nvdev <= peerDevice) {
        err = cudaErrorInvalidDevice;
    }
    WARN(3, "done.\n");

    return err;
}
