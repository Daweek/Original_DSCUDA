#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <cutil.h>
#include <cutil_inline.h>

#include "reduceMultiGPU.cuh"

#define PERF_MEASUREMENT_ONLY (0) // set 1 for perf measurement w/o cudaMemcpy() host -> device.

#define NTHREAD    (64)
#define NBLOCKMAX  (65535)
#define NDEVMAX    (64)

typedef struct {
    int id;
    int nelem, nelem0;
    int nblock, nblock0;
    int *idata;
    int *odata;
    int *tmpdata;
} Device_t;

static int prepareDevices(int ndev_req, int nelem, int nblock);
static int reduceWithMultiGPU(int ndev, int nelem, int *h_idata);

static Device_t Device[NDEVMAX];


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
    int i, ndev_req, nelem;
    long long int sumCPU, sumGPU;
    int *idata;

    if (argc < 2) {
        fprintf(stderr, "reduces an array of int.\n"
                "usage: %s <# of elements> [# of devices]\n", argv[0]);
        exit(1);
    }
    nelem = atoi(argv[1]);
    ndev_req = 1;
    if (2 < argc) {
        ndev_req = atoi(argv[2]);
    }

    idata = (int *)malloc(sizeof(int) * nelem);
    srand48(time(NULL));
    sumCPU = 0;
    for (i = 0; i < nelem; i++) {
        idata[i] = lrand48() % (1 << 8);
        //        idata[i] = 1; // !!!
        sumCPU += idata[i];
    }

    int ntry = 100;
    double tdelta, tnow = 0;
    int bytes = sizeof(int) * nelem;

    sumGPU = reduceWithMultiGPU(ndev_req, nelem, idata); // warm up

    get_cputime(&tnow, &tdelta);
    for (i = 0; i < ntry; i++) {
        sumGPU = reduceWithMultiGPU(ndev_req, nelem, idata);
    }
    get_cputime(&tnow, &tdelta);

    printf("%lld bytes  %f sec  %f GB/s\n", bytes, tdelta, bytes / tdelta * ntry / 1e9);
    printf("sumCPU: %lld\n", sumCPU);
    printf("sumGPU: %lld\n", sumGPU);
}

static int
prepareDevices(int ndev_req, int nelem, int nblock)
{
    int off, idev, ndev;
    Device_t *dev;
    static int firstcall = 1;

    cutilSafeCall(cudaGetDeviceCount(&ndev));
    if (ndev < 1) {
        fprintf(stderr, "No CUDA device found.\n");
        exit(1);
    }
    if (NDEVMAX < ndev) {
        fprintf(stderr, "Too many CUDA devices (%d) found.\n", ndev);
        exit(1);
    }
    if (ndev < ndev_req) {
        fprintf(stderr, "too many devices (=%d) requested.\n", ndev_req);
        exit(1);
    }
    ndev = ndev_req;

    if (firstcall) {
        firstcall = 0;
        fprintf(stderr, "%d CUDA devices found.\n", ndev);
        for (idev = 0; idev < ndev; idev++) {
            dev = Device + idev;
            dev->nblock0 = 0;
            dev->nelem0 = 0;
        }
    }

    off = 0;
    for (idev = 0; idev < ndev; idev++) {
        dev = Device + idev;

        dev->id = idev;

        dev->nblock = nblock / ndev;
        if (idev < nblock % ndev) {
            dev->nblock++;
        }
        if (dev->nblock == 0) break; // too small nelem to use all ndev devices.

        dev->nelem = dev->nblock * NTHREAD;
        if (nelem < off + dev->nelem) {
            dev->nelem = nelem - off;
        }

        // update the max nblock & nelem used so far.
        cudaSetDevice(idev);
        if (dev->nelem0 < dev->nelem) {
            if (dev->nelem0) {
                cutilSafeCall(cudaFree(dev->idata));
            }
            dev->nelem0 = dev->nelem;
            cutilSafeCall(cudaMalloc((void**)&dev->idata, sizeof(int) * dev->nelem));
        }
        if (dev->nblock0 < dev->nblock) {
            if (dev->nblock0) {
                cutilSafeCall(cudaFree((void *)dev->nblock));
            }
            dev->nblock0 = dev->nblock;
            cutilSafeCall(cudaMalloc((void**)&dev->odata, sizeof(int) * dev->nblock));
        }

        off += dev->nelem;
    }
    ndev = idev;

#if 0
    fprintf(stderr, "use %d CUDA devices.\n", ndev);
    for (idev = 0; idev < ndev; idev++) {
        dev = Device + idev;
        fprintf(stderr, "dev:%d  nblock:%d  nelem:%d\n",
                idev, dev->nblock, dev->nelem);
    }
#endif

    return ndev;
}

static void
reduceInEachDevice(int ndev, int *h_idata)
{
    int off, size, idev;
    Device_t *dev;

    off = 0;
    for (idev = 0; idev < ndev; idev++) {
        dev = Device + idev;
        cudaSetDevice(idev);

        size = sizeof(int) * dev->nelem;
#if !PERF_MEASUREMENT_ONLY
        cutilSafeCall(cudaMemcpy(dev->idata, h_idata + off, size, cudaMemcpyHostToDevice)); // !!!
#endif
        off += dev->nelem;
    }

    for (idev = 0; idev < ndev; idev++) {
        dev = Device + idev;
        cudaSetDevice(idev);

        dim3 threads(NTHREAD, 1, 1);
        dim3 grids(dev->nblock, 1, 1);
        int smemsize = sizeof(int) * NTHREAD;

        reduceVec3<<<grids, threads, smemsize>>>(dev->nelem, dev->idata, dev->odata);
    }
}

static void
retrieveResultsFromAllDevices(int ndev, int *h_odata)
{
    int off, size, idev;
    Device_t *dev;

    off = 0;
    for (idev = 0; idev < ndev; idev++) {
        dev = Device + idev;
        cudaSetDevice(idev);

        size = sizeof(int) * dev->nblock;
        cutilSafeCall(cudaMemcpy(h_odata + off, dev->odata, size, cudaMemcpyDeviceToHost));
        off += dev->nblock;
    }
}

static void
retrieveResultFromDevice(Device_t *dev, int *h_odata)
{
    int size;

    cudaSetDevice(dev->id);
    size = sizeof(int) * dev->nblock;
    cutilSafeCall(cudaMemcpy(h_odata, dev->odata, size, cudaMemcpyDeviceToHost));
}

static void
reduceDevicePair(Device_t *dev0, Device_t *dev1)
{
    int size = sizeof(int) * dev1->nblock;

    cudaSetDevice(dev0->id);
    if (!dev0->tmpdata) {
        cutilSafeCall(cudaMalloc((void**)&dev0->tmpdata, size));
    }
    cutilSafeCall(cudaMemcpy(dev0->tmpdata, dev1->odata, size, cudaMemcpyDefault));

    int nblock = dev0->nblock > dev1->nblock ? dev0->nblock : dev1->nblock;
    nblock = (nblock - 1) / NTHREAD + 1;
    dim3 threads(NTHREAD, 1, 1);
    dim3 grids(nblock, 1, 1);
    int smemsize = sizeof(int) * NTHREAD * 2;
    //    fprintf(stderr, "nblock:%d  dev0->nblock:%d  dev1->nblock:%d\n", nblock, dev0->nblock, dev1->nblock);
    addVec<<<grids, threads, smemsize>>>(dev0->nblock, dev0->odata, dev1->nblock, dev0->tmpdata);    
}

static int
reduceWithMultiGPU(int ndev_req, int nelem, int *h_idata)
{
    int ndev, nblock, i, j, stride, nres;
    long long int sum;
    static int *h_odata = NULL;
    static int nblock0 = 0;

    nblock = (nelem - 1) / NTHREAD + 1;

    // realloc buffers.
    if (nblock0 < nblock) {
        if (NBLOCKMAX * ndev_req < nblock) {
            fprintf(stderr, "# of elements exceeds the limit (=%d).\n", NTHREAD * NBLOCKMAX * ndev_req);
            exit(1);
        }
        fprintf(stderr, "nelem:%d  nthread per block:%d  nblock:%d\n", nelem, NTHREAD, nblock);
        h_odata = (int *)realloc(h_odata, sizeof(int) * nblock);
        nblock0 = nblock;
    }

    // # of device (ndev) might be smaller than
    // that requested (ndev_req) if nelem is small.
    ndev = prepareDevices(ndev_req, nelem, nblock);

    reduceInEachDevice(ndev, h_idata);

#if 0
    retrieveResultsFromAllDevices(ndev, h_odata);
    nres = nblock;
#else

    for (stride = 1; stride < ndev; stride *= 2) {
        for (i = 0; i < ndev; i += stride * 2) {
            j = i + stride;
            if (j < ndev) {
                // fprintf(stderr, "%d + %d  ", i, j); // !!!
                reduceDevicePair(Device + i, Device + j);
            }
            else {
                // fprintf(stderr, "%d  ", i); // !!!
            }
        }
        // fprintf(stderr, "\n"); // !!!
    }
    retrieveResultFromDevice(Device + 0, h_odata); // retrieve from the 1st device.
    cudaDeviceSynchronize();
    nres = (Device + 0)->nblock;
#endif

    sum = 0;

#if !PERF_MEASUREMENT_ONLY
    for (i = 0; i < nres; i++) {
        sum += h_odata[i];
    }
#endif

    return sum;
}
