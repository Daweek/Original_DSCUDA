#include <stdio.h>
#include <stdlib.h>
#include <helper_cuda.h>
#include "userapp.cuh"

#define cutilSafeCall checkCudaErrors
#define N (8)

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

    cutilSafeCall(cudaMalloc((void**) &d_a, sizeof(float) * N));
    cutilSafeCall(cudaMalloc((void**) &d_b, sizeof(float) * N));
    cutilSafeCall(cudaMalloc((void**) &d_c, sizeof(float) * N));

    for (t = 0; t < 3; t++) {
        printf("try %d\n", t);
        for (i = 0; i < N; i++) {
            a[i] = rand()%64;
            b[i] = rand()%64;
        }
        cutilSafeCall(cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice));
        cutilSafeCall(cudaMemcpy(d_b, b, sizeof(float) * N, cudaMemcpyHostToDevice));
        int nth = 4;
        dim3 threads(nth, 1, 1);
        dim3 grids((N+nth-1)/nth, 1, 1);
        vecAdd<<<grids, threads>>>(d_a, d_b, d_c);
        //        vecAddT<float, float><<<grids, threads>>>(d_a, d_b, d_c);
        //        vecAddT<float, double><<<grids, threads>>>(dd_a, dd_b, d_c);

        func = vecAdd;
        //        fprintf(stderr, ">>>>%x\n", func);
        cutilSafeCall(cudaMemcpy(c, d_c, sizeof(float) * N, cudaMemcpyDeviceToHost));
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

#ifdef NOMAIN
int _unused_main(void)
#else
int main(void)
#endif
{
    local_func();
    exit(0);
}
