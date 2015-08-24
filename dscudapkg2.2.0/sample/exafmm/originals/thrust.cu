#include <iostream>
#include <stdint.h>
#include <cstdio>
#include <helper_cuda.h>
#include <helper_string.h>

static void show(int size, uint64_t *key, int *value)
{
    int i;
    uint64_t *keyh = (uint64_t *)malloc(size * sizeof(uint64_t));
    int *valh = (int *)malloc(size * sizeof(int));

    cudaMemcpy(keyh, key, size * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(valh, value, size * sizeof(int), cudaMemcpyDeviceToHost);
    fprintf(stderr, "---- gpu\n");
    for (i = 0; i < size; i++) {
        fprintf(stderr, "%5d  %5lld  %5d\n", i, keyh[i], valh[i]);
    }
    fprintf(stderr, "----\n");

    free(valh);
    free(keyh);
}

#if __DSCUDA__

#include <dscuda.h>

void sort(const int size, int * key, int * value) {
    dscudaSortIntBy32BitKey(size, key, value);
}

void sort(const int size, uint64_t *key, int *value)
{
    //    show(10, key, value);
    dscudaSortIntBy64BitKey(size, key, value);
    //    show(10, key, value);
}

void scan(const int size, uint64_t * key, int * value) {
    dscudaScanIntBy64BitKey(size, key, value);
}

#else

#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/scan.h>

void sort(const int size, int * key, int * value) {
  thrust::device_ptr<int> keyBegin(key);
  thrust::device_ptr<int> keyEnd(key+size);
  thrust::device_ptr<int> valueBegin(value);
  thrust::sort_by_key(keyBegin, keyEnd, valueBegin);
}

void sort(const int size, uint64_t * key, int * value) {
  thrust::device_ptr<uint64_t> keyBegin(key);
  thrust::device_ptr<uint64_t> keyEnd(key+size);
  thrust::device_ptr<int> valueBegin(value);
  thrust::sort_by_key(keyBegin, keyEnd, valueBegin);
}

void scan(const int size, uint64_t * key, int * value) {
  thrust::device_ptr<uint64_t> keyBegin(key);
  thrust::device_ptr<uint64_t> keyEnd(key+size);
  thrust::device_ptr<int> valueBegin(value);
  thrust::inclusive_scan_by_key(keyBegin, keyEnd, valueBegin, valueBegin);
}

#endif
