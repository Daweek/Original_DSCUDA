/*
 * this file is included into the bottom of libdscuda_ibv.cu & libdscuda_tcp.cu.
 */

static cudaError_t cudaMemcpyP2P(void *dst, int ddev, const void *src, int sdev, size_t count);

static void
checkResult(void *rp, RCServer_t *sp)
{
    // a dummy func.
}

static int
requestDaemonForDevice(unsigned int ipaddr, int devid, int useibv)
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
        WARN(0, "max possible ports on %s already in use.\n", dscudaAddrToServerIpStr(ipaddr));
        exit(1);
    }

    WARN(3, "server port: %d  daemon socket: %d\n", sport, dsock);

    if (useibv) {
        sprintf(msg, "remotecall:ibv");
    }
    else {
        sprintf(msg, "remotecall:tcp");
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
    if (mem->next) {
        mem->next->prev = mem->prev;
    }
    else { // mem was the last entry.
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
    if (st->next) {
        st->next->prev = st->prev;
    }
    else { // st was the last entry.
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

    if (ca->next) {
        ca->next->prev = ca->prev;
    }
    else{ // ca was the last entry.
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
    if (ev->next) {
        ev->next->prev = ev->prev;
    }
    else { // ev was the last entry.
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

void *
dscudaUvaOfAdr(void *adr, int devid)
{
    unsigned long adri = (unsigned long)adr;
#if __LP64__
    adri |= ((unsigned long)devid << 48);
#endif
    return (void *)adri;
}

int
dscudaDevidOfUva(void *adr)
{
#if __LP64__
    unsigned long adri = (unsigned long)adr;
    int devid = adri >> 48;
    return devid;
#else
    return 0;
#endif
}

void *
dscudaAdrOfUva(void *adr)
{
    unsigned long adri = (unsigned long)adr;
#if __LP64__
    adri &= 0x0000ffffffffffffLL;
#endif
    return (void *)adri;
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

#if 1
static void
showuva(void)
{
    RCuva *st = RCuvaListTop;
    while (st) {
        fprintf(stderr, ">>> 0x%08llx    prev:%p  next:%p\n", st, st->prev, st->next);
        st = st->next;
    }
}
#endif

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

    if (uva->next) {
        uva->next->prev = uva->prev;
    }
    else { // uva was the last entry.
        RCuvaListTail = uva->prev;
    }
    free(uva);
}

static RCuva *
RCuvaQuery(void *adr)
{
    RCuva *uva = RCuvaListTop;
    unsigned long ladr = (unsigned long)dscudaAdrOfUva(adr);
    int devid = dscudaDevidOfUva(adr);

    while (uva) {
        if ((unsigned long)uva->adr[0] <= ladr &&
            ladr < (unsigned long)uva->adr[0] + uva->size &&
            uva->devid == devid) {
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

static int
dscudaSearchServer(char *ips, int size)
{
    int sock, rcvsock, nsvr, val = 1;
    unsigned int adr, mask;
    socklen_t sin_size;
    char rcvbuf[256], buf[256];
    struct sockaddr_in addr, svr;
    struct ifreq ifr[2];
    struct ifconf ifc;

    WARN(2, "searching DSCUDA servers...\n");
    sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    rcvsock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (sock == -1 || rcvsock == -1) {
        perror("dscudaSearchServer: socket()");
        return -1;
    }
    setsockopt(sock, SOL_SOCKET, SO_BROADCAST, &val, sizeof(val));

    ifc.ifc_len = sizeof(ifr) * 2;
    ifc.ifc_ifcu.ifcu_buf = (char *)ifr;
    ioctl(sock, SIOCGIFCONF, &ifc);

    ifr[1].ifr_addr.sa_family = AF_INET;
    ioctl(sock, SIOCGIFADDR, &ifr[1]);
    adr = ((struct sockaddr_in *)(&ifr[1].ifr_addr))->sin_addr.s_addr;
    ioctl(sock, SIOCGIFNETMASK, &ifr[1]);
    mask = ((struct sockaddr_in *)(&ifr[1].ifr_netmask))->sin_addr.s_addr;

    addr.sin_family = AF_INET;
    addr.sin_port = htons(RC_DAEMON_IP_PORT - 1);
    addr.sin_addr.s_addr = adr | ~mask;

    strcpy(buf, "DSCUDA_SEARCHSERVER");
    sendto(sock, buf, strlen(buf) + 1, 0, (struct sockaddr *)&addr, sizeof(addr));

    sin_size = sizeof(struct sockaddr_in);
    memset(ips, 0, size);
    nsvr = 0;

    svr.sin_family = AF_INET;
    svr.sin_port = htons(RC_DAEMON_IP_PORT - 2);
    svr.sin_addr.s_addr = htonl(INADDR_ANY);
    ioctl(rcvsock, FIONBIO, &val);

    if (bind(rcvsock, (struct sockaddr *)&svr, sizeof(svr)) != 0) {
        perror("dscudaSearchServer: bind()");
        return -1;
    }

    sleep(1);
    memset(rcvbuf, 0, 256);
    while (0 < recvfrom(rcvsock, rcvbuf, 256 - 1, 0, (struct sockaddr *)&svr, &sin_size)) {
        WARN(2, "recvfrom ");
        if (strcmp(rcvbuf, "DSCUDA_SERVERRESPONSE") == 0) {
            WARN(2, "found server: \"%s\"\n", inet_ntoa(svr.sin_addr));
            strcat(ips, " ");
            strcat(ips, inet_ntoa(svr.sin_addr));
            nsvr++;
        }
        memset(rcvbuf, 0, 256);
    }

    close(sock);
    close(rcvsock);

    return nsvr;
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

    if (!env && 0 < dscudaSearchServer(buf, 1024 * RC_NVDEVMAX)) {
        setenv("DSCUDA_SERVER", buf, 1);
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
                vdev->server[nred].ip = dscudaServerNameToAddr(ip);
                vdev->server[nred].cid = dscudaServerNameToDevid(ip);
                vdev->server[nred].id = nred;
                nred++;
                ip = strtok(NULL, ",");
            }
            vdev->nredundancy = nred;
        }
    }
    else {
        Nvdev = 1;
        Vdev[0].nredundancy = 1;
        sp = Vdev[0].server;
        sp->id = 0;
        sp->ip = dscudaServerNameToAddr((char *)DEFAULT_SVRIP);
    }
    WARN(3, "DSCUDA Server\n");
    vdev = Vdev;
    for (i = 0; i < Nvdev; i++) {
        WARN(3, "  virtual device%d\n", i);
        sp = vdev->server;
        for (ired = 0; ired < vdev->nredundancy; ired++) {
            WARN(3, "    %s:%d\n", dscudaAddrToServerIpStr(sp->ip), sp->cid);
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
      case RC_REMOTECALL_TYPE_TCP:
        WARN(2, "method of remote procedure call: TCP Socket\n");
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
    //    get_myaddress(&addrin);
    firstcall = 0;
    pthread_mutex_unlock(&InitClientMutex);

    usleep(1000000); // !!!
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
    system("/bin/mkdir /tmp/dscuda 1> /dev/null  2> /dev/null");
    // do not use >& since /bin/sh on some distro does not recognize it.

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
#warning fill this part in dscudaFuncGetAttributesWrapper().
    cudaError_t err = cudaSuccess;

#if 0
    dscudaFuncGetAttributesResult *rp;

    initClient();
    WARN(3, "dscudaFuncGetAttributesWrapper(%d, 0x%08llx, %s)...",
         moduleid, (unsigned long)attr, func);
    Vdev_t *vdev = Vdev + Vdevid[vdevidIndex()];
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        if (UseIbv) {


            // !!!


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
#endif

    return err;
}

static cudaError_t
dscudaMemcpyToSymbolH2D(int moduleid, char *symbol, const void *src,
                        size_t count, size_t offset, int vdevid, int raidid)
{
    cudaError_t err = cudaSuccess;
    SETUP_PACKET_BUF(DscudaMemcpyToSymbolH2D, vdevid, raidid);

    int snamelen = strlen(symbol);
    if (RC_SNAMELEN <= snamelen) {
        WARN(0, "dscudaMemcpyToSymbolH2D:symbol name too long (%d byte).\n", snamelen);
        exit(1);
    }

    WARN(3, "dscudaMemcpyToSymbolH2D(%d, %s, 0x%08llx, %d, %d, %d, %d)...",
         moduleid, symbol, src, count, offset, vdevid, raidid);
    WARN(3, "method:%d\n", spkt->method);

    // pack send data.
    spktsize += count;
    spkt->moduleid = moduleid;
    strncpy(spkt->symbol, symbol, RC_SNAMELEN);
    memcpy((char *)&spkt->srcbuf, src, count);
    spkt->count = count;
    spkt->offset = offset;
    WARN(3, "spktsize:%d\n", spktsize);

    perform_remote_call(conn, &rpkt->method, spktsize, spkt->method);

    // unpack returned data.
    err = rpkt->err;
    WARN(3, "dscudaMemcpyToSymbolH2D err : %d\n", err);
    return err;
}

static cudaError_t
dscudaMemcpyToSymbolD2D(int moduleid, char *symbol, const void *src,
                        size_t count, size_t offset, int vdevid, int raidid)
{
    cudaError_t err = cudaSuccess;
    SETUP_PACKET_BUF(DscudaMemcpyToSymbolD2D, vdevid, raidid);

    int snamelen = strlen(symbol);
    if (RC_SNAMELEN <= snamelen) {
        WARN(0, "dscudaMemcpyToSymbolD2D:symbol name too long (%d byte).\n", snamelen);
        exit(1);
    }

    WARN(3, "dscudaMemcpyToSymbolD2D(%d, %s, 0x%08llx, %d, %d, %d, %d)...",
         moduleid, symbol, src, count, offset, vdevid, raidid);
    WARN(3, "method:%d\n", spkt->method);

    spkt->moduleid = moduleid;
    strncpy(spkt->symbol, symbol, RC_SNAMELEN);
    spkt->srcadr = (RCadr)src;
    spkt->count = count;
    spkt->offset = offset;
    WARN(3, "spktsize:%d\n", spktsize);

    perform_remote_call(conn, &rpkt->method, spktsize, spkt->method);

    // unpack returned data.
    err = rpkt->err;
    WARN(3, "dscudaMemcpyToSymbolD2D err : %d\n", err);
    return err;
}

static cudaError_t
dscudaMemcpyFromSymbolD2H(int moduleid, void **dstbuf, char *symbol,
                          size_t count, size_t offset, int vdevid, int raidid)
{
    cudaError_t err = cudaSuccess;
    SETUP_PACKET_BUF(DscudaMemcpyFromSymbolD2H, vdevid, raidid);

    int snamelen = strlen(symbol);
    if (RC_SNAMELEN <= snamelen) {
        WARN(0, "dscudaMemcpyFromSymbolD2H:symbol name too long (%d byte).\n", snamelen);
        exit(1);
    }

    spkt->moduleid = moduleid;
    strncpy(spkt->symbol, symbol, RC_SNAMELEN);
    spkt->count = count;
    spkt->offset = offset;
    WARN(3, "spktsize:%d\n", spktsize);

    perform_remote_call(conn, &rpkt->method, spktsize, spkt->method);
        
    // unpack returned data.
    err = rpkt->err;
    memcpy(*dstbuf, (char *)&rpkt->dstbuf, count);

    WARN(3, "dscudaMemcpyFromSymbolD2H err : %d\n", err);
    return err;
}

static cudaError_t
dscudaMemcpyFromSymbolD2D(int moduleid, void *dstadr, char *symbol,
                          size_t count, size_t offset, int vdevid, int raidid)
{
    cudaError_t err = cudaSuccess;
    SETUP_PACKET_BUF(DscudaMemcpyFromSymbolD2D, vdevid, raidid);

    int snamelen = strlen(symbol);
    if (RC_SNAMELEN <= snamelen) {
        WARN(0, "dscudaMemcpyFromSymbolD2D:symbol name too long (%d byte).\n", snamelen);
        exit(1);
    }

    spkt->moduleid = moduleid;
    spkt->dstadr = (RCadr)dstadr;
    strncpy(spkt->symbol, symbol, RC_SNAMELEN);
    spkt->count = count;
    spkt->offset = offset;
    WARN(3, "spktsize:%d\n", spktsize);

    perform_remote_call(conn, &rpkt->method, spktsize, spkt->method);

    // unpack returned data.
    err = rpkt->err;
    WARN(3, "dscudaMemcpyFromSymbolD2D err : %d\n", err);
    return err;
}



static cudaError_t
dscudaMemcpyToSymbolAsyncH2D(int moduleid, char *symbol, const void *src,
                             size_t count, size_t offset, RCstream stream, int vdevid, int raidid)
{
    cudaError_t err = cudaSuccess;
    SETUP_PACKET_BUF(DscudaMemcpyToSymbolAsyncH2D, vdevid, raidid);

    int snamelen = strlen(symbol);
    if (RC_SNAMELEN <= snamelen) {
        WARN(0, "dscudaMemcpyToSymbolAsyncH2D:symbol name too long (%d byte).\n", snamelen);
        exit(1);
    }

    // pack send data.
    spktsize += count;
    spkt->moduleid = moduleid;
    strncpy(spkt->symbol, symbol, RC_SNAMELEN);
    memcpy((char *)&spkt->src, src, count);
    spkt->count = count;
    spkt->offset = offset;
    spkt->stream = stream;
    WARN(3, "spktsize:%d\n", spktsize);

    perform_remote_call(conn, &rpkt->method, spktsize, spkt->method);

    // unpack returned data.
    err = rpkt->err;
    WARN(3, "dscudaMemcpyToSymbolAsyncH2D err : %d\n", err);
    return err;
}

static cudaError_t
dscudaMemcpyToSymbolAsyncD2D(int moduleid, char *symbol, const void *src,
                             size_t count, size_t offset, RCstream stream, int vdevid, int raidid)
{
    cudaError_t err = cudaSuccess;
    SETUP_PACKET_BUF(DscudaMemcpyToSymbolAsyncD2D, vdevid, raidid);

    int snamelen = strlen(symbol);
    if (RC_SNAMELEN <= snamelen) {
        WARN(0, "dscudaMemcpyToSymbolAsyncD2D:symbol name too long (%d byte).\n", snamelen);
        exit(1);
    }

    spkt->moduleid = moduleid;
    strncpy(spkt->symbol, symbol, RC_SNAMELEN);
    spkt->srcadr = (RCadr)src;
    spkt->count = count;
    spkt->offset = offset;
    spkt->stream = stream;
    WARN(3, "spktsize:%d\n", spktsize);

    perform_remote_call(conn, &rpkt->method, spktsize, spkt->method);

    // unpack returned data.
    err = rpkt->err;
    WARN(3, "dscudaMemcpyToSymbolAsyncD2D err : %d\n", err);
    return err;
}

static cudaError_t
dscudaMemcpyFromSymbolAsyncD2H(int moduleid, void **dstbuf, char *symbol,
                               size_t count, size_t offset, RCstream stream, int vdevid, int raidid)
{
    cudaError_t err = cudaSuccess;
    SETUP_PACKET_BUF(DscudaMemcpyFromSymbolAsyncD2H, vdevid, raidid);

    int snamelen = strlen(symbol);
    if (RC_SNAMELEN <= snamelen) {
        WARN(0, "dscudaMemcpyFromSymbolAsyncD2H:symbol name too long (%d byte).\n", snamelen);
        exit(1);
    }

    spkt->moduleid = moduleid;
    strncpy(spkt->symbol, symbol, RC_SNAMELEN);
    spkt->count = count;
    spkt->offset = offset;
    spkt->stream = stream;
    WARN(3, "spktsize:%d\n", spktsize);

    perform_remote_call(conn, &rpkt->method, spktsize, spkt->method);
        
    // unpack returned data.
    err = rpkt->err;
    memcpy(*dstbuf, (char *)&rpkt->dst, count);

    WARN(3, "dscudaMemcpyFromSymbolAsyncD2H err : %d\n", err);
    return err;
}

static cudaError_t
dscudaMemcpyFromSymbolAsyncD2D(int moduleid, void *dstadr, char *symbol,
                               size_t count, size_t offset, RCstream stream, int vdevid, int raidid)
{
    cudaError_t err = cudaSuccess;
    SETUP_PACKET_BUF(DscudaMemcpyFromSymbolAsyncD2D, vdevid, raidid);

    int snamelen = strlen(symbol);
    if (RC_SNAMELEN <= snamelen) {
        WARN(0, "dscudaMemcpyFromSymbolAsyncD2D:symbol name too long (%d byte).\n", snamelen);
        exit(1);
    }

    spkt->moduleid = moduleid;
    spkt->dstadr = (RCadr)dstadr;
    strncpy(spkt->symbol, symbol, RC_SNAMELEN);
    spkt->count = count;
    spkt->offset = offset;
    spkt->stream = stream;
    WARN(3, "spktsize:%d\n", spktsize);

    perform_remote_call(conn, &rpkt->method, spktsize, spkt->method);

    // unpack returned data.
    err = rpkt->err;
    WARN(3, "dscudaMemcpyFromSymbolAsyncD2D err : %d\n", err);
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
         " symbol:%s  ...",
         moduleid, (unsigned long)symbol, (unsigned long)src,
         count, offset, dscudaMemcpyKindName(kind), symbol);

    nredundancy = (Vdev + Vdevid[vdevidIndex()])->nredundancy;
    switch (kind) {
      case cudaMemcpyHostToDevice:
        for (int i = 0; i < nredundancy; i++) {
            err = dscudaMemcpyToSymbolH2D(moduleid ? moduleid[i] : 0,
                                          (char *)symbol, src, count, offset,
                                          Vdevid[vdevidIndex()], i);
        }
        break;
      case cudaMemcpyDeviceToDevice:
        for (int i = 0; i < nredundancy; i++) {
            err = dscudaMemcpyToSymbolD2D(moduleid ? moduleid[i] : 0,
                                          (char *)symbol, src, count, offset,
                                          Vdevid[vdevidIndex()], i);
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
        dscudaVerbAddHist(RCMethodDscudaMemcpyToSymbolH2D, (void *)&args);
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
        dstbuf = calloc(1, count);
        if (!dstbuf) {
            WARN(0, "dscudaMemcpyFromSymbolWrapper:calloc() failed.\n");
            exit(1);
        }

        for (int i = 0; i < nredundancy; i++) {
            err = dscudaMemcpyFromSymbolD2H(moduleid ? moduleid[i] : 0,
                                            &dstbuf, (char *)symbol, count, offset,
                                            Vdevid[vdevidIndex()], i);
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
        free(dstbuf);
        break;
      case cudaMemcpyDeviceToDevice:
        for (int i = 0; i < nredundancy; i++) {
            err = dscudaMemcpyFromSymbolD2D(moduleid ? moduleid[i] : 0,
                                            dst, (char *)symbol, count, offset,
                                            Vdevid[vdevidIndex()], i);
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
            err = dscudaMemcpyToSymbolAsyncH2D(moduleid ? moduleid[i] : 0,
                                               (char *)symbol, src, count, offset,
                                               (RCstream)st->s[i], Vdevid[vdevidIndex()], i);
        }
        break;
      case cudaMemcpyDeviceToDevice:
        for (int i = 0; i < nredundancy; i++) {
            err = dscudaMemcpyToSymbolAsyncD2D(moduleid ? moduleid[i] : 0,
                                               (char *)symbol, src, count, offset,
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
        dstbuf = calloc(1, count);
        if (!dstbuf) {
            WARN(0, "dscudaMemcpyFromSymbolAsyncWrapper:calloc() failed.\n");
            exit(1);
        }
        for (int i = 0; i < nredundancy; i++) {
            err = dscudaMemcpyFromSymbolAsyncD2H(moduleid ? moduleid[i] : 0,
                                                 &dstbuf, (char *)symbol, count, offset,
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
        free(dstbuf);
        break;
      case cudaMemcpyDeviceToDevice:
        for (int i = 0; i < nredundancy; i++) {
            err = dscudaMemcpyFromSymbolAsyncD2D(moduleid ? moduleid[i] : 0,
                                                 dst, (char *)symbol, count, offset,
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
dscudaBindTexture(int moduleid, char *texname, void*devptr, size_t size,
                  RCtexture texbuf, size_t *offsetp, int vdevid, int raidid)
{
    cudaError_t err = cudaSuccess;
    SETUP_PACKET_BUF(DscudaBindTexture, vdevid, raidid);

    int tnamelen = strlen(texname);
    if (RC_SNAMELEN <= tnamelen) {
        WARN(0, "dscudaBindTexture:texture name too long (%d byte).\n", tnamelen);
        exit(1);
    }

    spkt->moduleid = moduleid;
    strncpy(spkt->texname, texname, RC_SNAMELEN);
    spkt->devptr = (RCadr)devptr;
    spkt->size = size;
    spkt->texbuf = texbuf;

    WARN(3, "spktsize:%d\n", spktsize);

    perform_remote_call(conn, &rpkt->method, spktsize, spkt->method);
        
    // unpack returned data.
    err = rpkt->err;
    if (offsetp) {
        *offsetp = rpkt->offset;
    }

    WARN(3, "dscudaBindTexture err : %d\n", err);

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
    int nredundancy;

    RCtexture texbuf;

    initClient();

    WARN(3, "dscudaBindTextureWrapper(0x%08llx, %s, 0x%08llx, 0x%08llx, 0x%08llx, 0x%08llx, %d)...",
         moduleid, texname, offset, tex, devPtr, desc, size);

    setTextureParams(&texbuf, tex, desc);

    nredundancy = (Vdev + Vdevid[vdevidIndex()])->nredundancy;

    for (int i = 0; i < nredundancy; i++) {
        err = dscudaBindTexture(moduleid ? moduleid[i] : 0,
                                texname, (void *)devPtr, size, (RCtexture)texbuf, offset,
                                Vdevid[vdevidIndex()], i);
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
#warning fill this part in dscudaBindTexture2DWrapper().
    cudaError_t err = cudaSuccess;

#if 0
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

            // !!!

        }
        else {

            rp = dscudabindtexture2did_1(moduleid ? moduleid[i] : 0, texname,
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

#endif
    return err;
}

cudaError_t
dscudaBindTextureToArrayWrapper(int *moduleid, char *texname,
                               const struct textureReference *tex,
                               const struct cudaArray *array,
                               const struct cudaChannelFormatDesc *desc)
{
#warning fill this part in dscudaBindTextureToArrayWrapper().
    cudaError_t err = cudaSuccess;

#if 0
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

            // !!!

        }
        else {

            rp = dscudabindtexturetoarrayid_1(moduleid ? moduleid[i] : 0,
                                              texname, (RCadr)ca->ap[i], (RCtexture)texbuf,
                                              Clnt[Vdevid[vdevidIndex()]][sp->id]);
            checkResult(rp, sp);
            if (rp->err != cudaSuccess) {
                err = (cudaError_t)rp->err;
            }
            xdr_free((xdrproc_t)xdr_dscudaResult, (char *)rp);
        }
    }
    WARN(3, "done.\n");

#endif
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
cudaUnbindTexture(const struct textureReference *texref)
{
    cudaError_t err = cudaSuccess;

    WARNONCE(3, "cudaUnbindTexture does nothing but returns cudaSuccess.\n");

    return err;
}

cudaChannelFormatDesc
cudaCreateChannelDesc(int x, int y, int z, int w, enum cudaChannelFormatKind f)
{
    cudaChannelFormatDesc desc;
    Vdev_t *vdev;
    int vdevid;

    initClient();

    WARN(3, "cudaCreateChannelDesc(%d, %d, %d, %d, %d)...", x, y, z, w, f);

    vdevid = Vdevid[vdevidIndex()];
    vdev = Vdev + vdevid;

    for (int i = 0; i < vdev->nredundancy; i++) {
        SETUP_PACKET_BUF(CreateChannelDesc, vdevid, i);
        spkt->x = x;
        spkt->y = y;
        spkt->z = z;
        spkt->w = w;
        spkt->f = f;
        WARN(3, "spktsize:%d\n", spktsize);

        perform_remote_call(conn, &rpkt->method, spktsize, spkt->method);

        // unpack returned data.
        if (i == 0) {
            memcpy(&desc, &rpkt->desc, sizeof(desc));
        }
    }
    WARN(3, "done.\n");

    return desc;
}

cudaError_t
cudaMemcpyH2D(void *dst, const void *src, size_t count, int vdevid)
{
    cudaError_t err = cudaSuccess;
    Vdev_t *vdev;

    vdev = Vdev + vdevid;
    for (int i = 0; i < vdev->nredundancy; i++) {
        SETUP_PACKET_BUF(MemcpyH2D, vdevid, i);
        spktsize += count;
        spkt->count = count;
        spkt->dstadr = (RCadr)dst;

        WARN(3, "spktsize:%d  count:%d\n", spktsize, count);

#if 1
        memcpy(&spkt->srcbuf, src, count);
        perform_remote_call(conn, &rpkt->method, spktsize, spkt->method);
#else
        while (rpkt->method == RCMethodNone) {
            // wait the returning packet for the previous non-blocking remote call.
        }
        rdmaWaitReadyToKickoff(conn);
        rpkt->method = RCMethodNone;
        rdmaPipelinedKickoff(conn, spktsize, (char *)&spkt->srcbuf, (char *)src, count);
#endif

        // unpack returned data.
        err = rpkt->err;
        WARN(3, "cudaMemcpy H2D err : %d\n", err);
    }

    return err;
}

cudaError_t
cudaMemcpyD2H(void *dst, const void *src, size_t count, int vdevid)
{
    cudaError_t err = cudaSuccess;
    Vdev_t *vdev;

    vdev = Vdev + vdevid;
    for (int i = 0; i < vdev->nredundancy; i++) {
        SETUP_PACKET_BUF(MemcpyD2H, vdevid, i);
        spkt->count = count;
        spkt->srcadr = (RCadr)src;
        WARN(3, "spktsize:%d  count:%d\n", spktsize, count);

        perform_remote_call(conn, &rpkt->method, spktsize, spkt->method);

        // unpack returned data.
        err = rpkt->err;
        WARN(3, "cudaMemcpy D2H err : %d\n", err);

        if (i == 0) {
            memcpy(dst, &rpkt->dstbuf, count);
        }
        else if (bcmp(dst, &rpkt->dstbuf, count) != 0) {
            WARN(1, "\n\ncudaMemcpy() data copied from device%d & device0 UNMATCHED.\n\n\n", i);
            if (autoVerb) {
                cudaMemcpyArgs args;
                args.dst = dst;
                args.src = (void *)src;
                args.count = count;
                args.kind = cudaMemcpyDeviceToHost;
                dscudaVerbAddHist(RCMethodMemcpyD2H, (void *)&args);
                dscudaVerbRecallHist();
                break;
            }
            else if (errorHandler) {
                errorHandler(errorHandlerArg);
            }
        }
        else {
            WARN(3, "cudaMemcpy() data copied from device%d & device0 matched.\n", i);
        }
    }

    return err;
}

cudaError_t
cudaMemcpyD2D(void *dst, const void *src, size_t count, int vdevid)
{
    cudaError_t err = cudaSuccess;
    Vdev_t *vdev;

    vdev = Vdev + vdevid;
    for (int i = 0; i < vdev->nredundancy; i++) {
        SETUP_PACKET_BUF(MemcpyD2D, vdevid, i);
        spkt->count = count;
        spkt->srcadr = (RCadr)src;
        spkt->dstadr = (RCadr)dst;

        WARN(3, "spktsize:%d  count:%d\n", spktsize, count);

        perform_remote_call(conn, &rpkt->method, spktsize, spkt->method);

        // unpack returned data.
        err = rpkt->err;
        WARN(3, "cudaMemcpy D2D err : %d\n", err);
    }

    return err;
}

cudaError_t
cudaMemset(void *devptr, int value, size_t count)
{
    cudaError_t err = cudaSuccess;
    int vid = vdevidIndex();

    initClient();
    WARN(3, "cudaMemset(0x%016llx, %d, %d)...", devptr, value, count);
    Vdev_t *vdev = Vdev + Vdevid[vid];
    for (int i = 0; i < vdev->nredundancy; i++) {
        SETUP_PACKET_BUF(Memset, Vdevid[vid], i);
        spkt->devptr = (RCadr)devptr;
        spkt->value = value;
        spkt->count = count;

        WARN(3, "spktsize:%d\n", spktsize);
        perform_remote_call(conn, &rpkt->method, spktsize, spkt->method);

        // unpack returned data.
        err = rpkt->err;
    }
    WARN(3, "done.\n");

    return err;
}


static int
unused_p2p_port(RCServer_t *dstsvr, RCServer_t *srcsvr)
{
    uint16_t port;

#if 0
    // !!! this is not practical for huge system with > 100 GPUs.
    // !!! current implementation returns a unique value for each pair of ddevid & sdevid,
    // !!! requiring so many (nnode^2) ports that cannot fit in 16 bits.
    int off = 35432;
    port = off + ddevid * RC_NVDEVMAX + sdevid;
    WARN(3, "ddevid:%d  sdevid:%d  off:%d\n", ddevid, sdevid, off);
#else
    int off = RC_SERVER_IP_PORT - 16;
    int sid = srcsvr->port - RC_SERVER_IP_PORT;
    int did = dstsvr->port - RC_SERVER_IP_PORT;
    port = off - (sid * RC_NSERVERMAX + did);
    WARN(3, "\n");
    WARN(3, "p2p port: %d (base-%d)\n", port, off - port);
    WARN(3, "srcsvr:%s  sid:%d\n", dscudaAddrToServerIpStr(srcsvr->ip), sid);
    WARN(3, "dstsvr:%s  did:%d\n", dscudaAddrToServerIpStr(dstsvr->ip), did);
#endif

    return port;
}


#if 0 // !!! not implemented yet.

static cudaError_t
dscudaSendP2P(void *sadr, int sdevid, void *dadr, int ddevid, size_t count, uint16_t *p2pports)
{
    cudaError_t err = cudaSuccess;
    Vdev_t *dstdev, *srcdev;
    RCServer_t *dstsvr, *srcsvr;

    srcdev = Vdev + sdevid;
    srcsvr = srcdev->server;

    dstdev = Vdev + ddevid;
    dstsvr = dstdev->server;

    // request srcdev[0] to send to all dstdevs.
    for (int i = 0; i < dstdev->nredundancy; i++, dstsvr++) {
        SETUP_PACKET_BUF(DscudaSendP2P, sdevid, 0);
        spkt->count = count;
        spkt->srcadr = (RCadr)sadr;
        spkt->dstadr = (RCadr)dadr;

        spkt->dstip = dstsvr->ip;
        spkt->port = p2pports[i];

        WARN(3, "spktsize:%d  count:%d\n", spktsize, count);

        perform_remote_call(conn, &rpkt->method, spktsize, spkt->method);

        // unpack returned data.
        err = rpkt->err;
        WARN(3, "dscudaSendP2P err : %d\n", err);
    }

    return err;
}

static cudaError_t
dscudaRecvP2P(int ddevid, int sdevid, uint16_t *p2pports)
{
    cudaError_t err = cudaSuccess;
    Vdev_t *dstdev, *srcdev;
    RCServer_t *dstsvr, *srcsvr;

    srcdev = Vdev + sdevid;
    srcsvr = srcdev->server;

    dstdev = Vdev + ddevid;
    dstsvr = dstdev->server;

    // request all dstdevs to receive from srcdev[0].
    for (int i = 0; i < dstdev->nredundancy; i++, dstsvr++) {
        SETUP_PACKET_BUF(DscudaRecvP2P, ddevid, i);
        spkt->port = p2pports[i] = unused_p2p_port(dstsvr, srcsvr);
        spkt->srcip = srcsvr->ip;

        WARN(3, "port:%d\n", spkt->port);
        WARN(3, "spktsize:%d\n", spktsize);

        perform_remote_call(conn, &rpkt->method, spktsize, spkt->method);

        // unpack returned data.
        err = rpkt->err;
        WARN(3, "dscudaRecvP2P err : %d\n", err);
    }

    return err;
}

#endif // !!! not implemented yet.


static cudaError_t
cudaMemcpyP2P(void *dst, int ddev, const void *src, int sdev, size_t count)
{
#if 0 // dev0 -> dev1 direct transfer.

    cudaError_t err = cudaSuccess;
    uint16_t ports[RC_NREDUNDANCYMAX]; // IP port[0..nredundancy] for P2P communication.

    err = dscudaRecvP2P(ddev, sdev, ports);
    if (err != cudaSuccess) {
        fprintf(stderr, "!!! err:\n", err);
        return err;
    }

    err = dscudaSendP2P(dscudaAdrOfUva((void*)src), sdev,
                        dscudaAdrOfUva(dst), ddev, count, ports);
    return err;

#else // dev0 -> client -> dev1 indirect transfer.
    cudaError_t err = cudaSuccess;
    int dev0;
    int pgsz = 4096;
    static int bufsize = 0;
    static char *buf = NULL;

    if (bufsize < count) {
        bufsize = ((count - 1) / pgsz + 1) * pgsz;
        buf = (char *)realloc(buf, bufsize);
        if (!buf) {
            perror("cudaMemcpyP2P");
            exit(1);
        }
    }

    cudaGetDevice(&dev0);

    if (sdev != dev0) {
        cudaSetDevice(sdev);
    }
    err = cudaMemcpy(buf, src, count, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        if (sdev != dev0) {
            cudaSetDevice(dev0);
        }
        return err;
    }
    if (ddev != sdev) {
        cudaSetDevice(ddev);
    }
    err = cudaMemcpy(dst, buf, count, cudaMemcpyHostToDevice);
    if (ddev != dev0) {
        cudaSetDevice(dev0);
    }
    return err;
#endif
}

static int
dscudaLoadModuleLocal(unsigned int ipaddr, pid_t pid, char *modulename, char *modulebuf, int vdevid, int raidid)
{
    cudaError_t err = cudaSuccess;
    SETUP_PACKET_BUF(DscudaLoadModule, vdevid, raidid);

    int moduleid;
    int namelen = strlen(modulename);
    int imagelen = strlen(modulebuf);

    if (RC_KMODULENAMELEN <= namelen) {
        WARN(0, "dscudaLoadModuleLocal:modulename too long (%d byte).\n", namelen);
        exit(1);
    }
    if (RC_KMODULEIMAGELEN <= imagelen) {
        WARN(0, "dscudaLoadModuleLocal:modulebuf too long (%d byte).\n", imagelen);
        exit(1);
    }

    spktsize += imagelen + 1;
    spkt->ipaddr = ipaddr;
    spkt->pid = pid;
    strncpy(spkt->modulename, modulename, RC_KMODULENAMELEN);
    strncpy((char *)&spkt->moduleimage, modulebuf, RC_KMODULEIMAGELEN);
    WARN(5, "spktsize:%d\n", spktsize);

    perform_remote_call(conn, &rpkt->method, spktsize, spkt->method);

    // unpack returned data.
    err = rpkt->err;
    WARN(5, "dscudaLoadModuleLocal err : %d\n", err);
    moduleid = rpkt->moduleid;

    return moduleid;
}

/*
 * launch a kernel function of __PRETTY_FUNCTION__ 'key'.
 */

static void
dscudaLaunch(void **kadrp, char *key, int vdevid, int raidid)
{
    cudaError_t err = cudaSuccess;
    int k;

    SETUP_PACKET_BUF(DscudaLaunch, vdevid, raidid);
    spkt->kadr = (RCadr)*kadrp;
    strncpy(spkt->prettyname, key, RC_SNAMELEN);
    perform_remote_call(conn, &rpkt->method, spktsize, spkt->method);

    // unpack returned data.
    *kadrp = (void *)rpkt->kadr;
    err = rpkt->err;
    WARN(3, "dscudaLaunch kadr : %016llx\n", *kadrp);
    WARN(5, "dscudaLaunch err  : %d\n", err);
}

void
dscudaLaunchWrapper(void **kadrp, char *key)
{
    int vid = vdevidIndex();
    Vdev_t *vdev = Vdev + Vdevid[vid];
    RCmappedMem *mem;

    for (int i = 0; i < vdev->nredundancy; i++) {
        dscudaLaunch(kadrp, key, Vdevid[vid], i);
    }

    mem = RCmappedMemListTop;
    while (mem) {
        cudaMemcpy(mem->pHost, mem->pDevice, mem->size, cudaMemcpyDeviceToHost);
        mem = mem->next;
    }

#warning add some autoVerb handling here.

}

/*
 * launch a kernel function of id 'kid', defined in a module of id 'moduleid'.
 * 'kid' must be unique inside a single module.
 */

static void
dscudaLaunchKernel(int moduleid, int kid, char *kname,
                      int *gdim, int *bdim, RCsize smemsize, RCstream stream,
                      int narg, RCArg *arg, int vdevid, int raidid)
{
    cudaError_t err = cudaSuccess;
    int k;

    SETUP_PACKET_BUF(DscudaLaunchKernel, vdevid, raidid);

    spktsize += sizeof(RCArg) * narg;
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
    memcpy((char *)&spkt->args, arg, sizeof(RCArg) * narg);
    WARN(5, "spktsize:%d\n", spktsize);

    perform_remote_call(conn, &rpkt->method, spktsize, spkt->method);

    // unpack returned data.
    err = rpkt->err;
    WARN(5, "dscudaLaunchKernel err : %d\n", err);
}

void
dscudaLaunchKernelWrapper(int *moduleid, int kid, char *kname,
                          int *gdim, int *bdim, RCsize smemsize, RCstream stream,
                          int narg, RCArg *arg)
{
    RCmappedMem *mem;
    RCstreamArray *st;
    int vid = vdevidIndex();

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

    Vdev_t *vdev = Vdev + Vdevid[vid];
    for (int i = 0; i < vdev->nredundancy; i++) {
        dscudaLaunchKernel(moduleid[i], kid, kname,
                           gdim, bdim, smemsize, (RCstream)st->s[i],
                           narg, arg, Vdevid[vid], i);
    }

    mem = RCmappedMemListTop;
    while (mem) {
        cudaMemcpy(mem->pHost, mem->pDevice, mem->size, cudaMemcpyDeviceToHost);
        mem = mem->next;
    }

    if (autoVerb) {
        cudaLaunchKernelArgs args2;
        args2.moduleid = moduleid;
        args2.kid = kid;
        args2.kname = kname;
        args2.gdim = gdim;
        args2.bdim = bdim;
        args2.smemsize = smemsize;
        args2.stream = stream;
        args2.narg = narg;
        args2.arg = arg;
        dscudaVerbAddHist(RCMethodDscudaLaunchKernel, (void *)&args2);
    }
}

/*
 * Dscuda client-side counterpart for CUDA runtime API:
 */

/*
 * Thread Management
 */

cudaError_t
cudaThreadSynchronize(void)
{
    cudaError_t err = cudaSuccess;
    int vid = vdevidIndex();

    initClient();
    WARN(3, "cudaThreadSynchronize()...");
    Vdev_t *vdev = Vdev + Vdevid[vid];
    for (int i = 0; i < vdev->nredundancy; i++) {
        SETUP_PACKET_BUF(ThreadSynchronize, Vdevid[vid], i);
        WARN(3, "spktsize:%d\n", spktsize);
        WARN(3, "Vdevid[vid]:%d\n", Vdevid[vid]);

        perform_remote_call(conn, &rpkt->method, spktsize, spkt->method);

        // unpack returned data.
        err = rpkt->err;
        WARN(3, "cudaThreadSynchronize err : %d\n", err);
    }
    WARN(3, "done.\n");

    return err;
}

cudaError_t
cudaThreadExit(void)
{
    cudaError_t err = cudaSuccess;
    int vid = vdevidIndex();

    initClient();
    WARN(3, "cudaThreadExit()...");
    Vdev_t *vdev = Vdev + Vdevid[vid];
    for (int i = 0; i < vdev->nredundancy; i++) {
        SETUP_PACKET_BUF(ThreadExit, Vdevid[vid], i);
        WARN(3, "spktsize:%d\n", spktsize);
        WARN(3, "Vdevid[vid]:%d\n", Vdevid[vid]);

        perform_remote_call(conn, &rpkt->method, spktsize, spkt->method);

        // unpack returned data.
        err = rpkt->err;
        WARN(3, "cudaThreadExit err : %d\n", err);
    }
    WARN(3, "done.\n");

    return err;
}

/*
 * Error Handling
 */

cudaError_t
cudaGetLastError(void)
{
    cudaError_t err;
    int vid = vdevidIndex();

    initClient();
    WARN(3, "cudaGetLastError()...");
    Vdev_t *vdev = Vdev + Vdevid[vid];
    for (int i = 0; i < vdev->nredundancy; i++) {
        SETUP_PACKET_BUF(GetLastError, Vdevid[vid], i);

        // pack send data.
        WARN(3, "spktsize:%d\n", spktsize);
        WARN(3, "Vdevid[vid]:%d\n", Vdevid[vid]);

        perform_remote_call(conn, &rpkt->method, spktsize, spkt->method);

        // unpack returned data.
        WARN(3, "cudaGetLastError : %d\n", rpkt->err);
        err = rpkt->err;
    }
    WARN(3, "done.\n");

    return err;
}

cudaError_t
cudaPeekAtLastError(void)
{
    //    WARNONCE(2, "a dummy call to cudaPeekAtLastError()\n"); // !!!
    return cudaSuccess;
}

const char *
cudaGetErrorString(cudaError_t error)
{
    static char str[256];
    int vid = vdevidIndex();

    initClient();
    WARN(3, "cudaGetErrorString()...");
    Vdev_t *vdev = Vdev + Vdevid[vid];
    for (int i = 0; i < vdev->nredundancy; i++) {
        SETUP_PACKET_BUF(GetErrorString, Vdevid[vid], i);

        // pack send data.
        spkt->err = error;
        WARN(3, "spktsize:%d\n", spktsize);
        WARN(3, "Vdevid[vid]:%d\n", Vdevid[vid]);

        perform_remote_call(conn, &rpkt->method, spktsize, spkt->method);

        // unpack returned data.
        WARN(3, "cudaGetErrorString errmsg : %s\n", &rpkt->errmsg);
        strncpy(str, (char *)&rpkt->errmsg, strlen((char *)&rpkt->errmsg) + 1);
    }
    WARN(3, "done.\n");

    return str;
}

/*
 * Device Management
 */

cudaError_t
cudaDeviceSetLimit(cudaLimit limit, size_t value)
{
    cudaError_t err = cudaSuccess;
    int vid = vdevidIndex();

    initClient();
    WARN(3, "cudaDeviceSetLimit()...");
    Vdev_t *vdev = Vdev + Vdevid[vid];
    for (int i = 0; i < vdev->nredundancy; i++) {
        SETUP_PACKET_BUF(DeviceSetLimit, Vdevid[vid], i);

        WARN(3, "spktsize:%d\n", spktsize);
        spkt->limit = limit;
        spkt->value = value;
        perform_remote_call(conn, &rpkt->method, spktsize, spkt->method);

        // unpack returned data.
        err = rpkt->err;
    }
    WARN(3, "done.\n");

    return err;
}

cudaError_t
dscudaDeviceSetSharedMemConfig(cudaSharedMemConfig config)
{
    cudaError_t err = cudaSuccess;

    int vid = vdevidIndex();

    initClient();
    WARN(3, "cudaDeviceSetSharedMemConfig()...");
    Vdev_t *vdev = Vdev + Vdevid[vid];
    for (int i = 0; i < vdev->nredundancy; i++) {
        SETUP_PACKET_BUF(DeviceSetSharedMemConfig, Vdevid[vid], i);

        WARN(3, "spktsize:%d\n", spktsize);
        spkt->config = config;
        perform_remote_call(conn, &rpkt->method, spktsize, spkt->method);

        // unpack returned data.
        err = rpkt->err;
    }
    WARN(3, "done.\n");

    return err;
}

cudaError_t
cudaRuntimeGetVersion(int *runtimeVersion)
{
    cudaError_t err = cudaSuccess;
    int vid = vdevidIndex();

    initClient();
    WARN(3, "cudaRuntimeGetVersion(0x%08llx)...", (unsigned long)runtimeVersion);
    Vdev_t *vdev = Vdev + Vdevid[vid];
    for (int i = 0; i < vdev->nredundancy; i++) {
        SETUP_PACKET_BUF(RuntimeGetVersion, Vdevid[vid], i);
        WARN(3, "spktsize:%d\n", spktsize);
        WARN(3, "Vdevid[vid]:%d\n", Vdevid[vid]);

        perform_remote_call(conn, &rpkt->method, spktsize, spkt->method);

        // unpack returned data.
        err = rpkt->err;
        WARN(3, "cudaRuntimeGetVersion err : %d\n", err);
        *runtimeVersion = rpkt->version;
    }
    WARN(3, "done.\n");

    return err;
}

cudaError_t
cudaDeviceSynchronize(void)
{
    cudaError_t err = cudaSuccess;
    int vid = vdevidIndex();

    initClient();
    WARN(3, "cudaDeviceSynchronize()...");
    Vdev_t *vdev = Vdev + Vdevid[vid];
    for (int i = 0; i < vdev->nredundancy; i++) {
        SETUP_PACKET_BUF(DeviceSynchronize, Vdevid[vid], i);
        WARN(4, "spktsize:%d\n", spktsize);
        WARN(4, "Vdevid[vid]:%d\n", Vdevid[vid]);

        perform_remote_call(conn, &rpkt->method, spktsize, spkt->method);

        // unpack returned data.
        err = rpkt->err;
        WARN(4, "cudaDeviceSynchronize err : %d\n", err);
    }
    WARN(3, "done.\n");

    return err;
}

cudaError_t
cudaDeviceReset(void)
{
    WARN(3, "a dummy call to cudaDeviceReset()\n");
    return cudaSuccess;
}

/*
 * Execution Control
 */

#ifndef CUDA_VERSION
#include <cuda.h>
#endif

cudaError_t
#if CUDA_VERSION >= 5000
cudaLaunch(const void *func)
#else
cudaLaunch(const char *func)
#endif
{
    cudaError_t err = cudaSuccess;
    int vi = vdevidIndex();
    int vdevid = Vdevid[vi];
    Vdev_t *vdev = Vdev + Vdevid[vi];

    initClient();
    WARN(3, "cudaLaunch(0x%08llx)...", func);

    for (int i = 0; i < vdev->nredundancy; i++) {
        SETUP_PACKET_BUF(Launch, vdevid, i);
        spkt->func = (RCadr)func;
        perform_remote_call(conn, &rpkt->method, spktsize, spkt->method);
        if (rpkt->err != cudaSuccess) {
            err = rpkt->err;
        }
    }

    WARN(3, "done.\n");

    return err;
}

cudaError_t
cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem, cudaStream_t stream)
{
    cudaError_t err = cudaSuccess;
    int vi = vdevidIndex();
    int vdevid = Vdevid[vi];
    Vdev_t *vdev = Vdev + Vdevid[vi];
    RCdim3 gdim, bdim;

    initClient();
    WARN(3, "cudaConfigureCall([%d, %d, %d], [%d, %d, %d], %d, %d)\n",
         gridDim.x, gridDim.y, gridDim.z,
         blockDim.x, blockDim.y, blockDim.z,
         sharedMem, stream);

    for (int i = 0; i < vdev->nredundancy; i++) {
        SETUP_PACKET_BUF(ConfigureCall, vdevid, i);

        spkt->gdim[0] = gridDim.x;
        spkt->gdim[1] = gridDim.y;
        spkt->gdim[2] = gridDim.z;
        spkt->bdim[0] = blockDim.x;
        spkt->bdim[1] = blockDim.y;
        spkt->bdim[2] = blockDim.z;
        spkt->smemsize = sharedMem;
        spkt->stream = (RCstream)stream;

        perform_remote_call(conn, &rpkt->method, spktsize, spkt->method);
        if (rpkt->err != cudaSuccess) {
            err = rpkt->err;
        }
    }

    WARN(3, "done.  err:%d\n", err);

    return err;
}

cudaError_t
cudaSetupArgument(const void *arg, size_t size, size_t offset)
{
    cudaError_t err = cudaSuccess;
    int vi = vdevidIndex();
    int vdevid = Vdevid[vi];
    Vdev_t *vdev = Vdev + Vdevid[vi];
    RCbuf argbuf;

    initClient();
    WARN(3, "cudaSetupArgument(0x%llx, %d, %d)...", arg, size, offset);

    for (int i = 0; i < vdev->nredundancy; i++) {
        SETUP_PACKET_BUF(SetupArgument, vdevid, i);

        spkt->size = size;
        spkt->offset = offset;
        memcpy(&spkt->argbuf, arg, size);
        spktsize += size;
        WARN(3, "spktsize:%d\n", spktsize);

        perform_remote_call(conn, &rpkt->method, spktsize, spkt->method);
        if (rpkt->err != cudaSuccess) {
            err = rpkt->err;
        }
    }

    WARN(3, "done.\n");

    return err;
}

cudaError_t
cudaFuncSetCacheConfig(const void *func, enum cudaFuncCache cacheConfig)
{
    cudaError_t err = cudaSuccess;
    static bool firstcall = true;
    if (firstcall) {
        WARN(1, "Current implementation of cudaFuncSetCacheConfig() does nothing "
             "but returning cudaSuccess.\n");
        firstcall = false;
    }
    err = cudaSuccess;
    return err;
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
        dscudaVerbAddHist(RCMethodSetDevice, (void *)&args);
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


/*
 * Memory Management
 */

cudaError_t
cudaHostAlloc(void **devAdrPtr, size_t size, unsigned int flags)
{
    static int firstcall = true;
    if (!firstcall) return cudaSuccess;
    firstcall = false;
    WARN(2, "cudaHostAlloc() not implemented in DS-CUDA.\n");
}

cudaError_t
cudaMallocHost(void **devAdrPtr, size_t size)
{
    static int firstcall = true;
    if (!firstcall) return cudaSuccess;
    firstcall = false;
    WARN(2, "cudaMallocHost() not implemented in DS-CUDA.\n");
}

cudaError_t
cudaFreeHost(void *ptr)
{
    static int firstcall = true;
    if (!firstcall) return cudaSuccess;
    firstcall = false;
    WARN(2, "cudaFreeHost() not implemented in DS-CUDA.\n");
}

cudaError_t
cudaMalloc(void **devAdrPtr, size_t size)
{
    cudaError_t err = cudaSuccess;
    int vid = vdevidIndex();
    void *adrs[RC_NREDUNDANCYMAX];

    initClient();
    WARN(3, "cudaMalloc(0x%08llx, %d)...", (unsigned long)devAdrPtr, size);
    Vdev_t *vdev = Vdev + Vdevid[vid];
    for (int i = 0; i < vdev->nredundancy; i++) {
        SETUP_PACKET_BUF(Malloc, Vdevid[vid], i);
        spkt->size = size;
        WARN(3, "spktsize:%d  size:%d\n", spktsize, size);

        perform_remote_call(conn, &rpkt->method, spktsize, spkt->method);

        // unpack returned data.
        if (rpkt->err != cudaSuccess) {
            err = (cudaError_t)rpkt->err;
        }
        adrs[i] = (void*)rpkt->devAdr;
    }
    RCuvaRegister(Vdevid[vid], adrs, size);
    *devAdrPtr = dscudaUvaOfAdr(adrs[0], Vdevid[vid]);

    WARN(3, "done. *devAdrPtr:0x%08llx\n", *devAdrPtr);

    return err;
}

cudaError_t
cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind)
{
    cudaError_t err = cudaSuccess;
    int vdevid;
    int vi = vdevidIndex();
    RCuva *suva, *duva;
    int dev0;
    void *lsrc, *ldst;
    int combufsize;

    initClient();

    WARN(3, "cudaMemcpy(0x%08lx, 0x%08lx, %d, %s)...",
         (unsigned long)dst, (unsigned long)src, count, dscudaMemcpyKindName(kind));

#ifdef _LIBDSCUDA_IBV_CU
    combufsize = RC_RDMA_BUF_SIZE;
#endif

#ifdef _LIBDSCUDA_TCP_CU
    combufsize = RC_SOCKET_BUF_SIZE;
#endif

    if (combufsize < count) {
        WARN(0, "count(=%d) exceeds the size of send/recv buffer(=%d).\n",
             count, combufsize);
        exit(1);
    }

    vdevid = Vdevid[vi];

    lsrc = dscudaAdrOfUva((void *)src);
    ldst = dscudaAdrOfUva(dst);


    //    fprintf(stderr, ">>>> src:0x%016llx  lsrc:0x%016llx\n", src, lsrc);
    //    fprintf(stderr, ">>>> dst:0x%016llx  ldst:0x%016llx\n", dst, ldst);

    switch (kind) {
      case cudaMemcpyDeviceToHost:
        err = cudaMemcpyD2H(ldst, lsrc, count, vdevid);
        break;
      case cudaMemcpyHostToDevice:
        err = cudaMemcpyH2D(ldst, lsrc, count, vdevid);
        break;
      case cudaMemcpyDeviceToDevice:
        err = cudaMemcpyD2D(ldst, lsrc, count, vdevid);
        break;
      case cudaMemcpyDefault:
        cudaGetDevice(&dev0);
        suva = RCuvaQuery((void *)src);
        duva = RCuvaQuery(dst);
        if (!suva && !duva) {
            WARN(0, "cudaMemcpy:invalid argument.\n");
            exit(1);
        }
        else if (!suva) { // sbuf resides in the client.
            if (duva->devid != dev0) {
                cudaSetDevice(duva->devid);
            }
            err = cudaMemcpy(dst, src, count, cudaMemcpyHostToDevice);
            if (duva->devid != dev0) {
                cudaSetDevice(dev0);
            }
        }
        else if (!duva) { // dbuf resides in the client.
            if (suva->devid != dev0) {
                cudaSetDevice(suva->devid);
            }
            err = cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost);
            if (suva->devid != dev0) {
                cudaSetDevice(dev0);
            }
        }
        else {
            err = cudaMemcpyP2P(dst, duva->devid, src, suva->devid, count);
        }
        break;
      default:
        WARN(0, "Unsupported value for cudaMemcpyKind : %s\n", dscudaMemcpyKindName(kind));
        exit(1);
    }

    if (autoVerb) {
        cudaMemcpyArgs args;
        switch (kind) {
          case cudaMemcpyHostToDevice:
            args.dst = dst;
            args.src = (void *)src;
            args.count = count;
            args.kind = kind;
            dscudaVerbAddHist(RCMethodMemcpyH2D, (void *)&args);
            break;

          case cudaMemcpyDeviceToDevice:
            args.dst = dst;
            args.src = (void *)src;
            args.count = count;
            args.kind = kind;
            dscudaVerbAddHist(RCMethodMemcpyD2D, (void *)&args);
            break;

          case cudaMemcpyDeviceToHost:
            dscudaVerbClearHist();
            break;
        }
    }
    WARN(3, "done.\n");

    return err;
}

cudaError_t
cudaFree(void *mem)
{
    cudaError_t err = cudaSuccess;
    int vid = vdevidIndex();

    initClient();
    WARN(3, "cudaFree(0x%08llx)...", (unsigned long)mem);
    Vdev_t *vdev = Vdev + Vdevid[vid];
    for (int i = 0; i < vdev->nredundancy; i++) {
        SETUP_PACKET_BUF(Free, Vdevid[vid], i);
        spkt->devAdr = (RCadr)dscudaAdrOfUva(mem);
        WARN(3, "spktsize:%d\n", spktsize);

        perform_remote_call(conn, &rpkt->method, spktsize, spkt->method);

        // unpack returned data.
        if (rpkt->err != cudaSuccess) {
            err = rpkt->err;
        }
    }

    RCuvaUnregister(mem);

    WARN(3, "done.\n");

    return err;
}

cudaError_t
cudaMemcpyPeer(void *dst, int ddev, const void *src, int sdev, size_t count)
{
    cudaError_t err;

    WARN(3, "cudaMemcpyPeer(0x%08lx, %d, 0x%08lx, %d, %d)...",
         (unsigned long)dst, ddev, (unsigned long)src, sdev, count);

    err = cudaMemcpyP2P(dst, ddev, src, sdev, count);

    WARN(3, "done.\n");

    return err;
}

cudaError_t
cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device)
{
    cudaError_t err = cudaSuccess;

    initClient();
    WARN(3, "cudaGetDeviceProperties(0x%08lx, %d)...", (unsigned long)prop, device);

    // Vdev_t *vdev = Vdev + device;
    //    for (int i = 0; i < vdev->nredundancy; i++) {
    for (int i = 0; i < 1; i++) { // performs no redundant call for now.
        SETUP_PACKET_BUF(GetDeviceProperties, device, i);
        spkt->device = device;
        WARN(3, "spktsize:%d  device:%d\n", spktsize, device);

        perform_remote_call(conn, &rpkt->method, spktsize, spkt->method);

        // unpack returned data.
        err = rpkt->err;
        WARN(3, "cudaGetDeviceProperties err : %d\n", err);
        memcpy(prop, &rpkt->prop, sizeof(cudaDeviceProp));
    }
    WARN(3, "done.\n");

    return err;
}
/*
 * Stream Management
 */

/*
 * Event Management
 */


cudaError_t
cudaEventCreate(cudaEvent_t *event)
{
    static cudaEvent_t e;
    *event = e;
    WARN(3, "a dummy call to cudaEventCreate()\n");
    return cudaSuccess;
}

cudaError_t
cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags)
{
    static cudaEvent_t e;
    *event = e;
    WARN(3, "a dummy call to cudaEventCreateWithFlags()\n");
    return cudaSuccess;
}

cudaError_t
cudaEventDestroy(cudaEvent_t event)
{
    WARN(3, "a dummy call to cudaEventDestroy()\n");
    return cudaSuccess;
}

cudaError_t
cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end)
{
    *ms = 123.0;
    WARN(3, "a dummy call to cudaEventElapsedTime()\n");
    return cudaSuccess;
}

cudaError_t
cudaEventRecord(cudaEvent_t event, cudaStream_t stream)
{
    WARN(3, "a dummy call to cudaEventRecord()\n");
    return cudaSuccess;
}

cudaError_t
cudaEventSynchronize(cudaEvent_t event)
{
    WARN(3, "a dummy call to cudaEventSynchronize()\n");
    return cudaSuccess;
}

cudaError_t
cudaEventQuery(cudaEvent_t event)
{
    WARN(3, "a dummy call to cudaEventQuery()\n");
    return cudaSuccess;
}

/*
 * Non CUDA official (i.e., DS-CUDA specific) APIs
 */

cudaError_t
dscudaSortIntBy32BitKey(const int size, int *key, int *value)
{
    cudaError_t err = cudaSuccess;
    int vi = vdevidIndex();
    int vdevid = Vdevid[vi];
    Vdev_t *vdev = Vdev + Vdevid[vi];
    RCbuf argbuf;

    initClient();
    WARN(3, "dscudaSortIntBy32BitKey(%d, 0x%08llx, 0x%08llx)...", size, key, value);

    for (int i = 0; i < vdev->nredundancy; i++) {
        SETUP_PACKET_BUF(DscudaSortIntBy32BitKey, vdevid, i);

        spkt->nitems = size;
        spkt->key = (RCadr)key;
        spkt->value = (RCadr)value;

        perform_remote_call(conn, &rpkt->method, spktsize, spkt->method);
        if (rpkt->err != cudaSuccess) {
            err = rpkt->err;
        }
    }

    WARN(3, "done.\n");

    return err;
}

cudaError_t
dscudaSortIntBy64BitKey(const int size, uint64_t *key, int *value)
{
    cudaError_t err = cudaSuccess;
    int vi = vdevidIndex();
    int vdevid = Vdevid[vi];
    Vdev_t *vdev = Vdev + Vdevid[vi];
    RCbuf argbuf;

    initClient();
    WARN(3, "dscudaSortIntBy64BitKey(%d, 0x%08llx, 0x%08llx)...", size, key, value);

    for (int i = 0; i < vdev->nredundancy; i++) {
        SETUP_PACKET_BUF(DscudaSortIntBy64BitKey, vdevid, i);

        spkt->nitems = size;
        spkt->key = (RCadr)key;
        spkt->value = (RCadr)value;

        perform_remote_call(conn, &rpkt->method, spktsize, spkt->method);
        if (rpkt->err != cudaSuccess) {
            err = rpkt->err;
        }
    }

    WARN(3, "done.\n");

    return err;
}

cudaError_t
dscudaScanIntBy64BitKey(const int size, uint64_t *key, int *value)
{
    cudaError_t err = cudaSuccess;
    int vi = vdevidIndex();
    int vdevid = Vdevid[vi];
    Vdev_t *vdev = Vdev + Vdevid[vi];
    RCbuf argbuf;

    initClient();
    WARN(3, "dscudaScanIntBy64BitKey(%d, 0x%08llx, 0x%08llx)...", size, key, value);

    for (int i = 0; i < vdev->nredundancy; i++) {
        SETUP_PACKET_BUF(DscudaScanIntBy64BitKey, vdevid, i);

        spkt->nitems = size;
        spkt->key = (RCadr)key;
        spkt->value = (RCadr)value;

        perform_remote_call(conn, &rpkt->method, spktsize, spkt->method);
        if (rpkt->err != cudaSuccess) {
            err = rpkt->err;
        }
    }

    WARN(3, "done.\n");

    return err;
}


#if BINCOMPATIBLE

/*
 * undocumented CUDA API
 */
extern "C" {

void**
__cudaRegisterFatBinary(void *fatCubin)
{
    cudaError_t err = cudaSuccess;
    int vid = vdevidIndex();
    int i, imgbyte, imgword;

    initClient();

    WARN(3, "__cudaRegisterFatBinary(0x%016llx).\n", fatCubin);

    fatDeviceText_t *fdtp = (fatDeviceText_t *)fatCubin;
    unsigned long long int *img = fdtp->d;
    imgbyte = img[1];
    imgword = imgbyte / 8 + 0;

    for (i = 0; i < 3 * 4; i += 4) {
        WARN(3, "%d: %016llx %016llx %016llx %016llx\n",
             i, img[i + 0], img[i + 1], img[i + 2], img[i + 3]);
    }

    RCbuf imgbuf;
    int count = ((imgbyte - 1) / 8 + 1) * 8;
    void **handle;

    initClient();
    WARN(3, "dscudaRegisterFatBinary()...");
    Vdev_t *vdev = Vdev + Vdevid[vid];


#if 1 // this is necessary. I don't know why.
    static int cnt = 0;
    if (cnt == 3) {
        count += 16;
    }
    cnt++;
    WARN(3, "count:%d\n\n", count);
#endif


    for (int i = 0; i < vdev->nredundancy; i++) {
        SETUP_PACKET_BUF(DscudaRegisterFatBinary, Vdevid[vid], i);

       // pack send data.
        spktsize += count;
        spkt->m = fdtp->m;
        spkt->v = fdtp->v;
        if (fdtp->f) {
            memcpy(spkt->f, fdtp->f, strlen(fdtp->f));
        }
        else {
            spkt->f[0] = 0;
        }
        memcpy(&spkt->fatbinbuf, img, count);
        spkt->count = count;
        WARN(3, "spktsize:%d\n", spktsize);

        perform_remote_call(conn, &rpkt->method, spktsize, spkt->method);

        // unpack returned data.
        if (i == 0) {
            err = rpkt->err;
            handle = (void **)rpkt->handle;
        }

    }
    WARN(3, "dscudaRegisterFatBinary: %d\n", err);
    return handle;
}

void
__cudaUnregisterFatBinary(void **fatCubinHandle)
{
    cudaError_t err = cudaSuccess;
    int vid = vdevidIndex();
    Vdev_t *vdev = Vdev + Vdevid[vid];
    void **handle;

    initClient();
    WARN(3, "__cudaUnregisterFatBinary(0x%016llx).\n", fatCubinHandle);

    for (int i = 0; i < vdev->nredundancy; i++) {
        SETUP_PACKET_BUF(DscudaUnregisterFatBinary, Vdevid[vid], i);

       // pack send data.
        spkt->handle = (RCadr)handle;
        perform_remote_call(conn, &rpkt->method, spktsize, spkt->method);
        err = rpkt->err;
        WARN(3, "__cudaUnregisterFatBinary() err : %d\n", err);
    }
    WARN(3, "done.\n");
}

void
__cudaRegisterFunction(void       **fatCubinHandle,
                       const char *hostFun,
                       char       *deviceFun,
                       const char *deviceName,
                       int        thread_limit,
                       uint3      *tid,
                       uint3      *bid,
                       dim3       *bDim,
                       dim3       *gDim,
                       int        *wSize)
{
    cudaError_t err = cudaSuccess;
    int vid = vdevidIndex();
    Vdev_t *vdev = Vdev + Vdevid[vid];

    initClient();
    WARN(3, "__cudaRegisterFunction(0x%llx, 0x%llx, %s, %s, %d, ...).\n",
         fatCubinHandle, hostFun, deviceFun, deviceName, thread_limit);

    if (tid || bid || bDim || gDim || wSize) {
        fprintf(stderr, "One or more of tid, bid, bDim, gDim, wSize has non-zero value, "
                "which is not supported by DS-CUDA yet.\n");
        exit(1);
    }

    for (int i = 0; i < vdev->nredundancy; i++) {
        SETUP_PACKET_BUF(DscudaRegisterFunction, Vdevid[vid], i);

       // pack send data.
        spkt->handle = (RCadr)fatCubinHandle;
        spkt->hfunc = (RCadr)hostFun;
        strcpy(spkt->dfunc, deviceFun);
        strcpy(spkt->dname, deviceName);
        spkt->tlimit = thread_limit;
        perform_remote_call(conn, &rpkt->method, spktsize, spkt->method);
        err = rpkt->err;
        WARN(3, "__cudaRegisterFunction() err : %d\n", err);
    }
    WARN(3, "done.\n");
}

void
__cudaRegisterVar(void **fatCubinHandle,
                  char  *hostVar,
                  char  *deviceAddress,
                  const char  *deviceName,
                  int    ext,
                  int    size,
                  int    constant,
                  int    global)
{
    cudaError_t err = cudaSuccess;
    int vid = vdevidIndex();
    Vdev_t *vdev = Vdev + Vdevid[vid];

    initClient();
    WARN(3, "__cudaRegisterVar(0x%llx, 0x%llx, %s, %s, %d, %d, %d, %d).\n",
         fatCubinHandle, hostVar, deviceAddress, deviceName, ext, size, constant, global);

    for (int i = 0; i < vdev->nredundancy; i++) {
        SETUP_PACKET_BUF(DscudaRegisterVar, Vdevid[vid], i);

       // pack send data.
        spkt->handle = (RCadr)fatCubinHandle;
        spkt->hvar = (RCadr)hostVar;
        strcpy(spkt->dvar, deviceAddress);
        strcpy(spkt->dname, deviceName);
        spkt->ext = ext;
        spkt->size = size;
        spkt->constant = constant;
        spkt->global = global;
        perform_remote_call(conn, &rpkt->method, spktsize, spkt->method);
        err = rpkt->err;
        WARN(3, "__cudaRegisterVar() err : %d\n", err);
    }
    WARN(3, "done.\n");
}

void
__cudaRegisterTexture(void **fatCubinHandle,
                      const struct textureReference *hostVar,
                      const void **deviceAddress,
                      const char *deviceName,
                      int dim,       
                      int norm,      
                      int ext)

{
    fprintf(stderr, "__cudaRegisterTexture.\n");
}

} // "C"


#endif // BINCOMPAT
