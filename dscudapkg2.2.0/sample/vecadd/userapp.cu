#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

__global__ void
vecAdd(float *a, float *b, float *c)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    c[i] = a[i] + b[i];
}

__global__ void
vecMul(float *a, float *b, float *c)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    c[i] = a[i] + b[i];
}

