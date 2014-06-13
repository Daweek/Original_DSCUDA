#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <cutil.h>
#include <cutil_inline.h>
#include <sys/time.h>
#ifdef _OPENMP
#include <omp.h>
#endif // _OPENMP

#define MAXDEV 32
static const double MEGA  = 1e6;

cudaError_t cudaMemcpyToAlldev(int ndev, void **dst, const void *src, size_t count, enum cudaMemcpyKind kind);

static void
get_cputime(double *splittime, double *laptime)
{
    struct timeval x;

    gettimeofday(&x, NULL);

    *splittime = x.tv_sec + x.tv_usec/1000000.0 - *laptime;
    *laptime = x.tv_sec + x.tv_usec/1000000.0;
}

static void
bcastperf(int argc, char **argv)
{
    int maxsize = 1024 * 1024 * 10.0;
    int i, j;
    size_t size0 = 4096, size;
    double sized;
    double lt = 0.0, st = 0.0;
    double ratio = 2.0;
    double nloop = 2e8;
    char *src = (char *)malloc(sizeof(char) * maxsize);
    char *dst[MAXDEV];
    int ndev0 = 1, ndev, ndevmax;
    static int nthread = 0;

    if (1 < argc) {
        ndev0 = atoi(argv[1]);
    }
    if (2 < argc) {
        size0 = atoi(argv[2]);
    }
    cutilSafeCall(cudaGetDeviceCount(&ndevmax));
    printf("# %d device%s found.\n", ndevmax, ndevmax > 1 ? "s" : "");

    for (i = 0; i < ndevmax; i++) {
        cudaSetDevice(i);
        cutilSafeCall(cudaMalloc((void**) &dst[i], sizeof(char) * maxsize));
    }
    printf("\n#\n# cudaMemcpy (HostToDevice)\n");
    printf("# broadcast to %d..%d servers.\n#\n", ndev0, ndevmax);

    for (sized = size0; sized < maxsize; sized *= ratio) {
        //    for ( nloop = 2e8, sized = 4096 * 1; ; ) { // !!!
        size = (size_t)sized;

        for (ndev = ndev0; ndev <= ndevmax; ndev++) { // # of devices broadcast to.
            get_cputime(&lt, &st);
#pragma omp parallel for private(j)
            for (i = 0; i < ndev; i++) {
#ifdef _OPENMP
                if (nthread == 0) {
                    nthread = omp_get_num_threads();
                    fprintf(stderr, "nthread:%d\n", nthread);
                }
#endif // _OPENMP
                for (j = 0; j < nloop/size; j++) { // # of iterations.
                    cudaSetDevice(i);
                    cudaMemcpy(dst[i], src, size, cudaMemcpyHostToDevice);
                } // i
                cudaDeviceSynchronize();
            } // j
            get_cputime(&lt, &st);
            printf("%d devices %d byte    %f sec    %f MB/s   %f MB/s\n",
                   ndev, size, lt, nloop/MEGA/lt, nloop/MEGA/lt*ndev);
            fflush(stdout);
	} // ndev
    } // sized
}

int
main(int argc, char **argv)
{
    bcastperf(argc, argv);
    fprintf(stderr, "going to quit...\n");
    sleep(1);
    exit(0);
}
