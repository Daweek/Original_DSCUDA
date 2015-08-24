#include <stdio.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <sys/time.h>
#include <driver_types.h>

static int WarnLevel = 2; /* warning message output level. the higher the more verbose.
                             0: no warning (may cause wrong result with g7pkg/scripts/check.csh)
                             1: minimum
                             2: default
                             3: for debugging purpose
                          */
int
dscudaWarnLevel(void)
{
    return WarnLevel;
}

void
dscudaSetWarnLevel(int level)
{
    WarnLevel = level;
}

char *
dscudaMemcpyKindName(cudaMemcpyKind kind)
{
    static char *name;

    switch (kind) {
      case cudaMemcpyHostToHost:
        name = "cudaMemcpyHostToHost";
        break;
      case cudaMemcpyHostToDevice:
        name = "cudaMemcpyHostToDevice";
        break;
      case cudaMemcpyDeviceToHost:
        name = "cudaMemcpyDeviceToHost";
        break;
      case cudaMemcpyDeviceToDevice:
        name = "cudaMemcpyDeviceToDevice";
        break;
      case cudaMemcpyDefault:
        name = "cudaMemcpyDefault";
        break;
      default:
        name = "Invalid cudaMemcpyKind";
    }
    return name;
}

unsigned int
dscudaServerNameToDevid(char *svrname)
{
    char buf[256];
    char *token, *savep = NULL;

    strncpy(buf, svrname, sizeof(buf));
    token = strtok_r(buf, ":", &savep);
    token = strtok_r(NULL, ":", &savep);
    return token ? atoi(token) : 0;
}

unsigned int
dscudaServerIpStrToAddr(char *ipstr)
{
    unsigned int addr;
    struct hostent *host;
    host = gethostbyname(ipstr);
    if (!host) {
        fprintf(stderr, "unknown host name: %s\n", ipstr);
        exit(1);
    }
    addr = *(unsigned int *)host->h_addr_list[0];

    return addr;
}

unsigned int
dscudaServerNameToAddr(char *svrname)
{
    char buf[256];
    char *token, *savep = NULL;

    strncpy(buf, svrname, sizeof(buf));
    token = strtok_r(buf, ":", &savep);
    return dscudaServerIpStrToAddr(token);
}

char *
dscudaAddrToServerIpStr(unsigned int addr)
{
    struct in_addr ia;

    ia.s_addr = addr;
    return inet_ntoa(ia);
}

/*
 *
 * t0 : time of day (in second) the last time this function is called.
 * returns the number of seconds passed since *t0.
 */
double
RCgetCputime(double *t0)
{
    struct timeval t;
    double tnow, dt;

    gettimeofday(&t, NULL);
    tnow = t.tv_sec + t.tv_usec/1000000.0;
    dt = tnow - *t0;
    *t0 = tnow;
    return dt;
}


int
dscudaAlignUp(int off, int align)
{
    return ((off) + (align) - 1) & ~((align) - 1);
}

unsigned int
dscudaRoundUp(unsigned int src, unsigned int by)
{
    unsigned int dst = ((src - 1) / by + 1) * by;
    return dst;
}

