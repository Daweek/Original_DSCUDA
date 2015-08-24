// kernel 0. a naive one.
__global__ void
reduceVec0(int n, int *g_idata, int *g_odata)
{
    extern __shared__ int __smem[];
    int *sdata = __smem;

    // load shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (i < n) ? g_idata[i] : 0;
    
    __syncthreads();

    // do reduction in shared mem
    //    for (unsigned int s = 1; s < blockDim.x; s *= 2) { // !!!
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        // modulo arithmetic is slow!
        if ((tid % (2 * s)) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}


// kernel 3. a tuned one.
__global__ void
reduceVec3(int n, int *g_idata, int *g_odata)
{
    extern __shared__ int __smem[];
    int *sdata = __smem;

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    int mySum = (i < n) ? g_idata[i] : 0;

    if (i + blockDim.x < n) {
        mySum += g_idata[i + blockDim.x];
    }

    sdata[tid] = mySum;
    __syncthreads();

    // do reduction in shared mem
    for (unsigned int s = blockDim.x/2; 0 < s; s >>= 1) {
        if (tid < s) {
            sdata[tid] = mySum = mySum + sdata[tid + s];
        }
        __syncthreads();
    }

    // write result for this block to global mem 
    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}

__global__ void
addVec(int n0, int *data0, int n1, int *data1)
{
    extern __shared__ int __smem[];

    unsigned int i, tid = threadIdx.x;
    int *sdata0 = __smem;
    int *sdata1 = __smem + blockDim.x;

    // load shared mem
    i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata0[tid] = (i < n0) ? data0[i] : 0;
    __syncthreads();

    sdata1[tid] = (i < n1) ? data1[i] : 0;

    __syncthreads();

    // write result for this block to global mem
    data0[i] = sdata0[tid] + sdata1[tid];
}
