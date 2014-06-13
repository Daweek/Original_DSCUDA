/*
 * this file is included into the bottom of libdscuda_ibv.cu & libdscuda_rpc.cu.
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

static char*
readServerConf(char *fname)
{
    FILE *fp = fopen(fname, "r");
    char linebuf[1024];
    int len;
    static char buf[1024 * 128];

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

    switch (dscudaRemoteCallType()) {
      case RC_REMOTECALL_TYPE_RPC:
        WARN(2, "method of remote procedure call: RPC\n");
        break;
      case RC_REMOTECALL_TYPE_IBV:
        WARN(2, "method of remote procedure call: InfiniBand Verbs\n");
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
            setupConnection(i, sp);
        } // ired
    } // i
    struct sockaddr_in addrin;
    get_myaddress(&addrin);
    MyIpaddr = addrin.sin_addr.s_addr;
    WARN(2, "Client IP address : %s\n", dscudaGetIpaddrString(MyIpaddr));
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
#if USE_IBV
            mid = ibvDscudaLoadModule(MyIpaddr, getpid(), modulename, modulebuf, Vdevid, i);
#else
            dscudaLoadModuleResult *rp = dscudaloadmoduleid_1(MyIpaddr, getpid(), modulename, modulebuf, Clnt[Vdevid][sp->id]);
            checkResult(rp, sp);
            mid = rp->id;
#endif

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
#if USE_IBV
              mid = ibvDscudaLoadModule(MyIpaddr, getpid(), modulename, modulebuf, idev, i);
#else
              dscudaLoadModuleResult *rp = dscudaloadmoduleid_1(MyIpaddr, getpid(), modulename, modulebuf, Clnt[idev][sp->id]);
              checkResult(rp, sp);
              mid = rp->id;
#endif

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
#if USE_IBV
#warning fill this part
#else
        rp = dscudafuncgetattributesid_1(moduleid[i], (char*)func, Clnt[Vdevid][sp->id]);
        checkResult(rp, sp);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
#endif
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
#if USE_IBV
#warning fill this part
#else
            rp = dscudamemcpytosymbolh2did_1(moduleid[i], (char *)symbol, srcbuf, count, offset, Clnt[Vdevid][sp->id]);
            checkResult(rp, sp);
            if (rp->err != cudaSuccess) {
                err = (cudaError_t)rp->err;
            }
#endif
        }
        break;
      case cudaMemcpyDeviceToDevice:
        vdev = Vdev + Vdevid;
        sp = vdev->server;
        for (int i = 0; i < vdev->nredundancy; i++, sp++) {
#if USE_IBV
#warning fill this part
#else
            rp = dscudamemcpytosymbold2did_1(moduleid[i], (char *)symbol, (RCadr)src, count, offset, Clnt[Vdevid][sp->id]);
            checkResult(rp, sp);
            if (rp->err != cudaSuccess) {
                err = (cudaError_t)rp->err;
            }
#endif
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
#if USE_IBV
#warning fill this part
#else
            d2drp = dscudamemcpyfromsymbold2did_1(moduleid[i], (RCadr)dst, (char *)symbol, count, offset, Clnt[Vdevid][sp->id]);
            checkResult(d2drp, sp);
            if (d2drp->err != cudaSuccess) {
                err = (cudaError_t)d2drp->err;
            }
#endif
        }
        break;
      case cudaMemcpyDeviceToHost:
        vdev = Vdev + Vdevid;
        sp = vdev->server;
        for (int i = 0; i < vdev->nredundancy; i++, sp++) {
#if USE_IBV
#warning fill this part
#else
            d2hrp = dscudamemcpyfromsymbold2hid_1(moduleid[i], (char *)symbol, count, offset, Clnt[Vdevid][sp->id]);
            checkResult(d2hrp, sp);
#endif
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
#if USE_IBV
                err = ibvDscudaMemcpyToSymbolAsyncH2D(moduleid[i], (char *)symbol, src, count, offset, (RCstream)st->s[i],
                                                     Vdevid, i);
#else
                rp = dscudamemcpytosymbolasynch2did_1(moduleid[i], (char *)symbol, srcbuf, count, offset, (RCstream)st->s[i],
                                                     Clnt[Vdevid][sp->id]);
                checkResult(rp, sp);
                err = (cudaError_t)rp->err;
#endif
        }
        break;
      case cudaMemcpyDeviceToDevice:
        vdev = Vdev + Vdevid;
        sp = vdev->server;
        for (int i = 0; i < vdev->nredundancy; i++, sp++) {
#if USE_IBV
                err = ibvDscudaMemcpyToSymbolAsyncD2D(moduleid[i], (char *)symbol, src, count, offset, (RCstream)st->s[i], 
                                                     Vdevid, i);
#else
                rp = dscudamemcpytosymbolasyncd2did_1(moduleid[i], (char *)symbol, (RCadr)src, count, offset, (RCstream)st->s[i], 
                                                     Clnt[Vdevid][sp->id]);
                checkResult(rp, sp);
                err = (cudaError_t)rp->err;
#endif
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
#if USE_IBV
                err = ibvDscudaMemcpyFromSymbolAsyncD2D(moduleid[i], dst, (char *)symbol, count, offset, (RCstream)st->s[i],
                                                       Vdevid, i);
#else
                d2drp = dscudamemcpyfromsymbolasyncd2did_1(moduleid[i], (RCadr)dst, (char *)symbol, count, offset,
                                                          (RCstream)st->s[i], Clnt[Vdevid][sp->id]);
                checkResult(d2drp, sp);
                err = (cudaError_t)d2drp->err;
#endif
        }
        break;
      case cudaMemcpyDeviceToHost:
        vdev = Vdev + Vdevid;
        sp = vdev->server;
        for (int i = 0; i < vdev->nredundancy; i++, sp++) {
            void *dstbuf;
#if USE_IBV
                dstbuf = calloc(1, count);
                if (!dstbuf) {
                    WARN(0, "dscudaMemcpyFromSymbolAsyncWrapper:calloc() failed.\n");
                    exit(1);
                }
                err = ibvDscudaMemcpyFromSymbolAsyncD2H(moduleid[i], dstbuf, (char *)symbol, count, offset, (RCstream)st->s[i],
                                                       Vdevid, i);
#else
                d2hrp = dscudamemcpyfromsymbolasyncd2hid_1(moduleid[i], (char *)symbol, count, offset,
                                                          (RCstream)st->s[i], Clnt[Vdevid][sp->id]);
                checkResult(d2hrp, sp);
                err = (cudaError_t)d2hrp->err;
                dstbuf = d2hrp->buf.RCbuf_val;
#endif
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
#if USE_IBV
#warning fill this part
#else
        rp = dscudabindtextureid_1(moduleid[i], texname,
                                  (RCadr)devPtr, size, (RCtexture)texbuf, Clnt[Vdevid][sp->id]);
        checkResult(rp, sp);
#endif
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
#if USE_IBV
#warning fill this part
#else
        rp = dscudabindtexture2did_1(moduleid[i], texname,
                                    (RCadr)devPtr, width, height, pitch, (RCtexture)texbuf, Clnt[Vdevid][sp->id]);
        checkResult(rp, sp);
#endif
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
#if USE_IBV
#warning fill this part
#else
        rp = dscudabindtexturetoarrayid_1(moduleid[i], texname, (RCadr)ca->ap[i], (RCtexture)texbuf, Clnt[Vdevid][sp->id]);
        checkResult(rp, sp);
#endif
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
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
    *device = Vdevid;
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
