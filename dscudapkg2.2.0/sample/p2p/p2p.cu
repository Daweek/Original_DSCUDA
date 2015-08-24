#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

static const double MEGA  = 1e6;

enum {
    COPY_BY_MEMCPY,
    COPY_BY_MEMCPYPEER,
};

#define safeCall(err)             __safeCall   (err, __FILE__, __LINE__)
// warn CUDA API errors, but do not exit.
#define unsafeCall(err)           __unsafeCall   (err, __FILE__, __LINE__)

static inline void
__unsafeCall(cudaError err, const char *file, const int line)
{
    if (cudaSuccess != err) {
        fprintf(stderr, "%s(%i) : __unsafeCall() Runtime API error %d: %s.\n",
                file, line, (int)err, cudaGetErrorString(err));
    }
}

static inline void
__safeCall(cudaError err, const char *file, const int line)
{
    if (cudaSuccess != err) {
        fprintf(stderr, "%s(%i) : __unsafeCall() Runtime API error %d: %s.\n",
                file, line, (int)err, cudaGetErrorString(err));
        exit(1);
    }
}

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

int
main(int argc, char **argv)
{
    double ratio = 2.5;
    double nloop = 2e8;
    double sized;
    double now, dt;
    int maxsize = 1024 * 1024 * 10.0;
    int i, j, size;
    int *bufA, *bufB;
    int *dbufA, *dbufB;
    int srcdev, dstdev;
    int copyapi;
    char copyapistr[128];

    if (argc < 4) {
        fprintf(stderr,
                "copies an array of int from one device to another.\n"
                "usage: %s <c|p> <src_dev> <dst_dev> [size]\n"
                "           'c' for cudaMemcpy()\n"
                "           'p' for cudaMemcpyPeer()\n",
                argv[0]);
        exit(1);
    }
    switch (argv[1][0]) {
      case 'c':
        copyapi = COPY_BY_MEMCPY;
        sprintf(copyapistr, "cudaMemcpy() ");
        break;
      case 'p':
        copyapi = COPY_BY_MEMCPYPEER;
        sprintf(copyapistr, "cudaMemcpyPeer() ");
        break;
      default:
        fprintf(stderr, "arg1 should be 'c' or 'p'.\n");
        exit(1);
    }

    srcdev = atoi(argv[2]);
    dstdev = atoi(argv[3]);
    if (4 < argc) {
        maxsize = atoi(argv[4]);
    }

    fprintf(stderr, "%s %dB from device %d to device %d.\n",
            copyapistr, maxsize, srcdev, dstdev);

    bufA = (int *)malloc(maxsize);
    bufB = (int *)malloc(maxsize);

    safeCall(cudaSetDevice(srcdev));
    cudaMalloc((void**) &dbufA, maxsize);
    safeCall(cudaSetDevice(dstdev));
    cudaMalloc((void**) &dbufB, maxsize);

    // set randomly generated data to the source device.
    for (i = 0; i < maxsize / sizeof(int); i++) {
        bufA[i] = rand() % 64;
        bufB[i] = rand() % 64;
    }
    safeCall(cudaSetDevice(srcdev));
    safeCall(cudaMemcpy(dbufA, bufA, maxsize, cudaMemcpyHostToDevice));
    safeCall(cudaSetDevice(dstdev));

    // copy data from the source device to the destination device.
    if (copyapi == COPY_BY_MEMCPY) {
        safeCall(cudaMemcpy(dbufB, dbufA, maxsize, cudaMemcpyDefault));
    }
    else {
        unsafeCall(cudaMemcpyPeer(dbufB, dstdev, dbufA, srcdev, maxsize));
    }

    // send the data back from the destination device to the host.
    safeCall(cudaMemcpy(bufB, dbufB, maxsize, cudaMemcpyDeviceToHost));

    // comparing the result with the original to check the correctness of
    // cudaMemcpy() / cudaMemcpyPeer()
    for (i = 0; i < maxsize / sizeof(int); i++) {
        if (bufA[i] != bufB[i]) {
            fprintf(stderr, "NG\n");
            fprintf(stderr, "bufA[%d]:0x%08x\n", i, bufA[i]);
            fprintf(stderr, "bufB[%d]:0x%08x\n", i, bufB[i]);
            //            exit(1);
        }
    }
    fprintf(stderr, "OK\n");

    // measure the bandwidth.
    for (sized = 4096; sized < maxsize; sized *= ratio) {
        //    for ( nloop = 2e8, sized = 4096 * 1; ; ) { // !!!
        size = (size_t)sized;

	get_cputime(&now, &dt);
	for (j = 0; j < nloop/size; j++) {

            if (copyapi == COPY_BY_MEMCPY) {
                cudaMemcpy(dbufB, dbufA, size, cudaMemcpyDefault);
            }
            else {
                cudaMemcpyPeer(dbufB, dstdev, dbufA, srcdev, size);
            }
            //            fprintf(stderr, "j:%d\n", j);
        }
        cudaDeviceSynchronize();
	get_cputime(&now, &dt);
	printf("%d byte    %f sec    %f MB/s\n",
               size, dt, nloop/MEGA/dt);
	fflush(stdout);
    }

    exit(0);
}
