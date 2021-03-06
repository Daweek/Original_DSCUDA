#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "reduce.cuh"

#define safeCall(err)             __safeCall   (err, __FILE__, __LINE__)

static inline void
__safeCall(cudaError err, const char *file, const int line)
{
    if (cudaSuccess != err) {
        fprintf(stderr, "%s(%i) : __unsafeCall() Runtime API error %d: %s.\n",
                file, line, (int)err, cudaGetErrorString(err));
        exit(1);
    }
}

#define NTHREAD    (64)
#define NBLOCKMAX  (65536)

int
main(int argc, char **argv)
{
    int i, nelem, nblock;
    int sum, h_sum;
    int *h_idata, *h_odata;
    int *d_idata, *d_odata;

    if (argc < 2) {
        fprintf(stderr, "usage: %s <# of elements>\n", argv[0]);
        exit(1);
    }
    nelem = atoi(argv[1]);
    nblock = (nelem - 1) / NTHREAD + 1;
    if (NBLOCKMAX < nblock) {
        fprintf(stderr, "# of elements exceeds the limit (=%d).\n", NTHREAD * NBLOCKMAX);
        exit(1);
    }
    fprintf(stderr, "nelem:%d  nthread:%d  nblock:%d\n", nelem, NTHREAD, nblock);

    h_idata = (int *)malloc(sizeof(int) * nelem);
    h_odata = (int *)malloc(sizeof(int) * nblock);
    safeCall(cudaMalloc((void**) &d_idata, sizeof(int) * nelem));
    safeCall(cudaMalloc((void**) &d_odata, sizeof(int) * nblock));

    h_sum = 0;
    for (i = 0; i < nelem; i++) {
        h_idata[i] = lrand48() % (1 << 8);
        h_sum += h_idata[i];
    }
    safeCall(cudaMemcpy(d_idata, h_idata, sizeof(int) * nelem, cudaMemcpyHostToDevice));

    for (i = 0; i < nblock; i++) {
        h_odata[i] = 0;
    }
    safeCall(cudaMemcpy(d_odata, h_odata, sizeof(int) * nblock, cudaMemcpyHostToDevice));

    dim3 threads(NTHREAD, 1, 1);
    dim3 grids(nblock, 1, 1);
    int smemsize = sizeof(int) * NTHREAD;

    reduce<<<grids, threads, smemsize>>>(nelem, d_idata, d_odata);

    safeCall(cudaMemcpy(h_odata, d_odata, sizeof(int) * nblock, cudaMemcpyDeviceToHost));    

    sum = 0;
    for (i = 0; i < nblock; i++) {
        fprintf(stderr, "block[%d]:%d\n", i, h_odata[i]);
        sum += h_odata[i];
    }
    printf("  sum: %d\n", sum);
    printf("h_sum: %d\n", h_sum);
}
