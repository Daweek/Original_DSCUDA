__global__ void
vecAdd(float *a, float *b, float *c)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    c[i] = a[i] + b[i];
}

__global__ void
vecMul(float *a, float *b, float c, float *d, int e, int * f)
{
    int i = threadIdx.x;
    d[i] = a[i] * b[i] + c + e + f[i];
}

