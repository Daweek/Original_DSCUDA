#include <stdio.h>
#include <stdlib.h>
#include "userapp.cuh"

#define N (8)

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

template <typename T0, typename T1> __global__ void
vecAddT(T1 *a, T1 *b, T0 *c)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    c[i] = a[i] + b[i];
}

static int
local_func(void)
{
    int i, t;
    float a[N], b[N], c[N];
    float *d_a, *d_b, *d_c;
    double *dd_a, *dd_b, *dd_c;
    void (*func)(float *, float *, float *);

    safeCall(cudaMalloc((void**) &d_a, sizeof(float) * N));
    safeCall(cudaMalloc((void**) &d_b, sizeof(float) * N));
    safeCall(cudaMalloc((void**) &d_c, sizeof(float) * N));

    for (t = 0; t < 3; t++) {
        printf("try %d\n", t);
        for (i = 0; i < N; i++) {
            a[i] = rand()%64;
            b[i] = rand()%64;
        }
        safeCall(cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice));
        safeCall(cudaMemcpy(d_b, b, sizeof(float) * N, cudaMemcpyHostToDevice));
        int nth = 4;
        dim3 threads(nth, 1, 1);
        dim3 grids((N+nth-1)/nth, 1, 1);
        vecAdd<<<grids, threads>>>(d_a, d_b, d_c);
        //        vecAddT<float, float><<<grids, threads>>>(d_a, d_b, d_c);
        //        vecAddT<float, double><<<grids, threads>>>(dd_a, dd_b, d_c);

        func = vecAdd;
        //        fprintf(stderr, ">>>>%x\n", func);
        safeCall(cudaMemcpy(c, d_c, sizeof(float) * N, cudaMemcpyDeviceToHost));
        for (i = 0; i < N; i++) {
            printf("% 6.2f + % 6.2f = % 7.2f",
                   a[i], b[i], c[i]);
            if (a[i] + b[i] != c[i]) printf("   NG");
            printf("\n");
        }
        printf("\n");
    }

    exit(0);
}

int main(void)
{
    local_func();
    exit(0);
}
