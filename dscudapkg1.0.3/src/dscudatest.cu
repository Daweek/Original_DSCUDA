#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include "dscuda.h"
#include "ibv_rdma.h"
// remove definition of some macros which will be redefined in \"cutil_inline.h\".
#ifdef MIN
#undef MIN
#endif
#ifdef MAX
#undef MAX
#endif
#include <cutil_inline.h>
#include <rpc/rpc.h>

static void showusage(int argc, char **argv);
static void showstatus(int argc, char **argv);
static void devwperf(int argc, char **argv);
static void devrperf(int argc, char **argv);
static void get_cputime(double *laptime, double *sprittime);

static const double MEGA = 1e6;

typedef struct {
    void (*func)(int argc, char **argv);
    char *usage;
} TestMode;

static TestMode testmode[] =
    {
	showstatus,  "shows GPU status.",
        devwperf,    "measure device-memory write (host->GPU) performance.",
        devrperf,    "measure device-memory read (host<-GPU) performance.",
    };

int
main(int argc, char **argv)
{
    int mode;

    if (argc < 2) {
	showusage(argc, argv);
	exit (1);
    }
    mode = atoi(argv[1]); argc--; argv++;
    dscudaSetWarnLevel(2);
    if (0 <= mode && mode < sizeof(testmode)/sizeof(testmode[0])) {
	testmode[mode].func(argc, argv);
    }
    else {
	showusage(argc, argv);
	exit (1);
    }
    exit (0);
}

static void
showusage(int argc, char **argv)
{
    int i;
    int nitems = sizeof(testmode)/sizeof(testmode[0]);

    fprintf(stderr, "usage: %s <test_program_ID> [destination_IP_address]\n", argv[0]);
    for (i = 0; i < nitems; i++) {
	fprintf(stderr, "  %2d) %s\n", i, testmode[i].usage);
    }
}

static void
showstatus(int argc, char **argv)
{
    fprintf(stderr, "this command is not implemented yet.\n"
            "should be designed to show status of GPU(s).\n");
}

static void
devwperf(int argc, char **argv)
{
    int maxsize = RDMA_BUFFER_SIZE;
    int j;
    size_t size;
    double sized;
    char *src = (char*)malloc(sizeof(char) * maxsize);
    char *dst;
    double lt = 0.0, st = 0.0;
    double ratio = 2.5;
    double nloop = 2e8;

    cutilSafeCall(cudaMalloc((void**)&dst, sizeof(char) * maxsize));

    printf("\n#\n# Raw device-memory write (local host -> remote GPU)\n#\n");
    for (sized = 4096; sized < maxsize; sized *= ratio) {
        size = (size_t)sized;

	get_cputime(&lt, &st);
        for (j = 0; j < nloop/size; j++) {
            cutilSafeCall(cudaMemcpy(dst, src, size,
                                     cudaMemcpyHostToDevice));
	}
	get_cputime(&lt, &st);
	printf("%d byte    %f sec    %f MB/s\n",
               size, lt, nloop/MEGA/lt);
	fflush(stdout);
    }

    cutilSafeCall(cudaFree(dst));
    free(src);
}

static void
devrperf(int argc, char **argv)
{
    int maxsize = RDMA_BUFFER_SIZE;
    int j;
    size_t size;
    double sized;
    char *src;
    char *dst = (char*)malloc(sizeof(char) * maxsize);
    double lt = 0.0, st = 0.0;
    double ratio = 2.5;
    double nloop = 2e9;

    cutilSafeCall(cudaMalloc((void**)&src, sizeof(char) * maxsize));

    printf("\n#\n# Raw device-memory read (local host <- remote GPU)\n#\n");
    for (sized = 4096; sized < maxsize; sized *= ratio) {
        size = (size_t)sized;

	get_cputime(&lt, &st);
	for (j = 0; j < nloop/size; j++) {
            cutilSafeCall(cudaMemcpy(dst, src, size,
                                     cudaMemcpyDeviceToHost));
	}
	get_cputime(&lt, &st);
	printf("%d byte    %f sec    %f MB/s\n",
               size, lt, nloop/MEGA/lt);
	fflush(stdout);
    }

    cutilSafeCall(cudaFree(src));
    free(dst);
}

static void
get_cputime(double *splittime, double *laptime)
{
    struct timeval x;

    gettimeofday(&x, NULL);

    *splittime = x.tv_sec + x.tv_usec/1000000.0 - *laptime;
    *laptime = x.tv_sec + x.tv_usec/1000000.0;
}
