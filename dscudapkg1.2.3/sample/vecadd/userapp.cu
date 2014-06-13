#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cutil.h>
#include <cutil_inline.h>

#include "userapp.cuh"

#define N (8)

int
main(void)
{
    int i, t;
    float a[N], b[N], c[N];

    float *d_a, *d_b, *d_c;
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
