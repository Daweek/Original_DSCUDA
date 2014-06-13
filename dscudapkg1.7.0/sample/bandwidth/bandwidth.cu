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
static const double MICRO = 1e-6;

cudaError_t cudaMemcpyToAlldev(int ndev, void **dst, const void *src, size_t count, enum cudaMemcpyKind kind);

static void
get_cputime(double *nowp, double *deltap)
{
    struct timeval t;
    double now0;

    gettimeofday(&t, NULL);
    now0 = t.tv_sec + t.tv_usec/1000000.0;
    *deltap = now0 - *nowp;
    *nowp   = now0;
}

static void
bcastperf(int argc, char **argv)
{
    int maxsize = 1024 * 1024 * 10.0;
    int i, j;
    size_t size;
    double sized;
    double now = 0.0, dt = 0.0;
    double ratio = 2.5;
    double nloop = 2e8;
    char *src = (char *)malloc(sizeof(char) * maxsize);
    char *dst[MAXDEV];
    int ndev;
    static int nthread = 0;

    cutilSafeCall(cudaGetDeviceCount(&ndev));
    printf("# %d device%s found.\n", ndev, ndev > 1 ? "s" : "");

    for (i = 0; i < ndev; i++) {
        cudaSetDevice(i);
        cutilSafeCall(cudaMalloc((void**) &dst[i], sizeof(char) * maxsize));
    }
    printf("\n#\n# cudaMemcpy (HostToDevice)\n");
    printf("# broadcast to %d servers.\n#\n", ndev);

    for (sized = 4096; sized < maxsize; sized *= ratio) {
        //    for ( nloop = 2e8, sized = 4096 * 1; ; ) { // !!!
        size = (size_t)sized;

	get_cputime(&now, &dt);
	for (j = 0; j < nloop/size; j++) {
#if 1
#pragma omp parallel for
	    for (i = 0; i < ndev; i++) {
#ifdef _OPENMP
                if (nthread == 0) {
                    nthread = omp_get_num_threads();
                    fprintf(stderr, "nthread:%d\n", nthread);
                }
#endif // _OPENMP
  	        cudaSetDevice(i);
                cudaMemcpy(dst[i], src, size, cudaMemcpyHostToDevice);
	    }
#else
            cudaMemcpyToAlldev(ndev, (void **)dst, src, size, cudaMemcpyHostToDevice);
#endif
	}
        cudaDeviceSynchronize();
	get_cputime(&now, &dt);
	printf("%d byte    %f sec    %f MB/s\n",
               size, dt, nloop/MEGA/dt);
	fflush(stdout);
    }
}


static void
sendperf(int argc, char **argv)
{
    int maxsize = 1024 * 1024 * 10.0;
    int i, j;
    size_t size;
    double sized;
    double now = 0.0, dt = 0.0;
    double ratio = 2.5;
    double nloop = 2e8;
    char *src[MAXDEV];
    char *dst[MAXDEV];
    int ndev;
    cutilSafeCall(cudaGetDeviceCount(&ndev));

    ndev = 1; // !!!

    printf("# %d device%s found.\n", ndev, ndev > 1 ? "s" : "");
    for (i = 0; i < ndev; i++) {
        cudaSetDevice(i);
        cutilSafeCall(cudaMalloc((void**) &dst[i], sizeof(char) * maxsize));
	src[i] = (char *)malloc(sizeof(char) * maxsize);
    }
    printf("\n#\n# cudaMemcpy (HostToDevice)\n#\n");

#if 1
    nloop = 2e8;
    for (sized = 4096; sized < maxsize; sized *= ratio) {

        size = (size_t)sized;

	get_cputime(&now, &dt);
	for (j = 0; j < nloop/size; j++) {
	    for (i = 0; i < ndev; i++) {
  	        cudaSetDevice(i);
                cudaMemcpy(dst[i], src[i], size, cudaMemcpyHostToDevice);
	    }
	}
        cudaDeviceSynchronize();
	get_cputime(&now, &dt);

#if 0 // with estimated RPC overhead.
	double throughput = 1700.0; // MB/s
	double latency    = 60.0; // us
	double ibsec = nloop / (throughput * MEGA) + latency * MICRO * nloop / size;
	printf("%d byte    %f sec    %f MB/s    %f ib_sec  %f MB/s\n",
	       size, lt, nloop/MEGA/lt, ibsec, nloop/MEGA/(lt + ibsec));
#else
	  printf("%d byte    %f sec    %f MB/s\n",
	  size, dt, nloop/MEGA/dt);
#endif
	fflush(stdout);
    }

#else
          size = 40;
          for (i = 0; i < ndev; i++) {
              for (j = 0; j < size; j++) {
                  src[i][j] = j;
              }
          }
          for (i = 0; i < ndev; i++) {
              cudaSetDevice(i);
              cudaMemcpy(dst[i], src[i], size, cudaMemcpyHostToDevice);
          }
#endif
}

static void
receiveperf(int argc, char **argv)
{
    int maxsize = 1024 * 1024 * 10.0;
    int i, j;
    size_t size;
    double sized;
    double now = 0.0, dt = 0.0;
    double ratio = 2.5;
    double nloop = 2e8;
    char *src[MAXDEV];
    char *dst[MAXDEV];
    int ndev;
    cutilSafeCall(cudaGetDeviceCount(&ndev));

    ndev = 1; // !!!

    printf("# %d device%s found.\n", ndev, ndev > 1 ? "s" : "");
    for (i = 0; i < ndev; i++) {
        cudaSetDevice(i);
        cutilSafeCall(cudaMalloc((void**) &src[i], sizeof(char) * maxsize));
	dst[i] = (char *)malloc(sizeof(char) * maxsize);
    }
    printf("\n#\n# cudaMemcpy (DeviceToHost)\n#\n");


    nloop = 2e9;
    for (sized = 4096; sized < maxsize; sized *= ratio) {

        size = (size_t)sized;

	get_cputime(&now, &dt);
	for (j = 0; j < nloop/size; j++) {
	    for (i = 0; i < ndev; i++) {
  	        cudaSetDevice(i);
                cudaMemcpy(dst[i], src[i], size, cudaMemcpyDeviceToHost);
	    }
	}
        cudaDeviceSynchronize();
	get_cputime(&now, &dt);
	printf("%d byte    %f sec    %f MB/s\n",
               size, dt, nloop/MEGA/dt);
	fflush(stdout);
    }
}

int
main(int argc, char **argv)
{
    int ndev;
    cutilSafeCall(cudaGetDeviceCount(&ndev));

    if (1 < ndev) {
        bcastperf(argc, argv);
    }
    sendperf(argc, argv);
    receiveperf(argc, argv);

    fprintf(stderr, "going to quit...\n");
    sleep(1);
    exit(0);
}
