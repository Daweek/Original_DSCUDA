# 1 "/tmp/tmpxft_00001cd2_00000000-1_mr3.rcu.cudafe1.cpp"
# 1 "<built-in>"
# 1 "<command-line>"
# 1 "/tmp/tmpxft_00001cd2_00000000-1_mr3.rcu.cudafe1.cpp"
# 1 "./rcudatmp/mr3.rcu.cu"
# 1642 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/x86_64-redhat-linux/bits/c++config.h" 3
namespace std __attribute__((visibility("default"))) {
# 1654 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/x86_64-redhat-linux/bits/c++config.h" 3
}
# 152 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/include/stddef.h" 3
typedef long ptrdiff_t;
# 214 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/include/stddef.h" 3
typedef unsigned long size_t;
# 1 "/usr/local/cuda/bin/../include/crt/host_runtime.h" 1 3
# 69 "/usr/local/cuda/bin/../include/crt/host_runtime.h" 3
# 1 "/usr/local/cuda/bin/../include/cuda_runtime_api.h" 1 3
# 54 "/usr/local/cuda/bin/../include/cuda_runtime_api.h" 3
# 1 "/usr/local/cuda/bin/../include/host_defines.h" 1 3
# 55 "/usr/local/cuda/bin/../include/cuda_runtime_api.h" 2 3
# 1 "/usr/local/cuda/bin/../include/builtin_types.h" 1 3
# 42 "/usr/local/cuda/bin/../include/builtin_types.h" 3
# 1 "/usr/local/cuda/bin/../include/device_types.h" 1 3
# 46 "/usr/local/cuda/bin/../include/device_types.h" 3
enum cudaRoundMode
{
  cudaRoundNearest,
  cudaRoundZero,
  cudaRoundPosInf,
  cudaRoundMinInf
};
# 43 "/usr/local/cuda/bin/../include/builtin_types.h" 2 3
# 1 "/usr/local/cuda/bin/../include/driver_types.h" 1 3
# 59 "/usr/local/cuda/bin/../include/driver_types.h" 3
enum cudaError
{
  cudaSuccess = 0,
  cudaErrorMissingConfiguration,
  cudaErrorMemoryAllocation,
  cudaErrorInitializationError,
  cudaErrorLaunchFailure,
  cudaErrorPriorLaunchFailure,
  cudaErrorLaunchTimeout,
  cudaErrorLaunchOutOfResources,
  cudaErrorInvalidDeviceFunction,
  cudaErrorInvalidConfiguration,
  cudaErrorInvalidDevice,
  cudaErrorInvalidValue,
  cudaErrorInvalidPitchValue,
  cudaErrorInvalidSymbol,
  cudaErrorMapBufferObjectFailed,
  cudaErrorUnmapBufferObjectFailed,
  cudaErrorInvalidHostPointer,
  cudaErrorInvalidDevicePointer,
  cudaErrorInvalidTexture,
  cudaErrorInvalidTextureBinding,
  cudaErrorInvalidChannelDescriptor,
  cudaErrorInvalidMemcpyDirection,
  cudaErrorAddressOfConstant,
  cudaErrorTextureFetchFailed,
  cudaErrorTextureNotBound,
  cudaErrorSynchronizationError,
  cudaErrorInvalidFilterSetting,
  cudaErrorInvalidNormSetting,
  cudaErrorMixedDeviceExecution,
  cudaErrorCudartUnloading,
  cudaErrorUnknown,
  cudaErrorNotYetImplemented,
  cudaErrorMemoryValueTooLarge,
  cudaErrorInvalidResourceHandle,
  cudaErrorNotReady,
  cudaErrorInsufficientDriver,
  cudaErrorSetOnActiveProcess,
  cudaErrorStartupFailure = 0x7f,
  cudaErrorApiFailureBase = 10000
};


enum cudaChannelFormatKind
{
  cudaChannelFormatKindSigned,
  cudaChannelFormatKindUnsigned,
  cudaChannelFormatKindFloat,
  cudaChannelFormatKindNone
};


struct cudaChannelFormatDesc
{
  int x;
  int y;
  int z;
  int w;
  enum cudaChannelFormatKind f;
};


struct cudaArray;


enum cudaMemcpyKind
{
  cudaMemcpyHostToHost = 0,
  cudaMemcpyHostToDevice,
  cudaMemcpyDeviceToHost,
  cudaMemcpyDeviceToDevice
};


struct cudaPitchedPtr
{
  void *ptr;
  size_t pitch;
  size_t xsize;
  size_t ysize;
};


struct cudaExtent
{
  size_t width;
  size_t height;
  size_t depth;
};


struct cudaPos
{
  size_t x;
  size_t y;
  size_t z;
};


struct cudaMemcpy3DParms
{
  struct cudaArray *srcArray;
  struct cudaPos srcPos;
  struct cudaPitchedPtr srcPtr;

  struct cudaArray *dstArray;
  struct cudaPos dstPos;
  struct cudaPitchedPtr dstPtr;

  struct cudaExtent extent;
  enum cudaMemcpyKind kind;
};


struct cudaDeviceProp
{
  char name[256];
  size_t totalGlobalMem;
  size_t sharedMemPerBlock;
  int regsPerBlock;
  int warpSize;
  size_t memPitch;
  int maxThreadsPerBlock;
  int maxThreadsDim[3];
  int maxGridSize[3];
  int clockRate;
  size_t totalConstMem;
  int major;
  int minor;
  size_t textureAlignment;
  int deviceOverlap;
  int multiProcessorCount;
  int kernelExecTimeoutEnabled;
  int __cudaReserved[39];
};
# 224 "/usr/local/cuda/bin/../include/driver_types.h" 3
typedef enum cudaError cudaError_t;


typedef int cudaStream_t;


typedef int cudaEvent_t;
# 44 "/usr/local/cuda/bin/../include/builtin_types.h" 2 3
# 1 "/usr/local/cuda/bin/../include/texture_types.h" 1 3
# 54 "/usr/local/cuda/bin/../include/texture_types.h" 3
enum cudaTextureAddressMode
{
  cudaAddressModeWrap,
  cudaAddressModeClamp
};


enum cudaTextureFilterMode
{
  cudaFilterModePoint,
  cudaFilterModeLinear
};


enum cudaTextureReadMode
{
  cudaReadModeElementType,
  cudaReadModeNormalizedFloat
};


struct textureReference
{
  int normalized;
  enum cudaTextureFilterMode filterMode;
  enum cudaTextureAddressMode addressMode[3];
  struct cudaChannelFormatDesc channelDesc;
  int __cudaReserved[16];
};
# 45 "/usr/local/cuda/bin/../include/builtin_types.h" 2 3
# 1 "/usr/local/cuda/bin/../include/vector_types.h" 1 3
# 54 "/usr/local/cuda/bin/../include/vector_types.h" 3
struct char1
{
  signed char x;
};


struct uchar1
{
  unsigned char x;
};


struct char2
{
  signed char x, y;
};


struct uchar2
{
  unsigned char x, y;
};


struct char3
{
  signed char x, y, z;
};


struct uchar3
{
  unsigned char x, y, z;
};


struct char4
{
  signed char x, y, z, w;
};


struct uchar4
{
  unsigned char x, y, z, w;
};


struct short1
{
  short x;
};


struct ushort1
{
  unsigned short x;
};


struct short2
{
  short x, y;
};


struct ushort2
{
  unsigned short x, y;
};


struct short3
{
  short x, y, z;
};


struct ushort3
{
  unsigned short x, y, z;
};


struct short4
{
  short x, y, z, w;
};


struct ushort4
{
  unsigned short x, y, z, w;
};


struct int1
{
  int x;
};


struct uint1
{
  unsigned int x;
};


struct int2
{
  int x, y;
};


struct uint2
{
  unsigned int x, y;
};


struct int3
{
  int x, y, z;
};


struct uint3
{
  unsigned int x, y, z;
};


struct int4
{
  int x, y, z, w;
};


struct uint4
{
  unsigned int x, y, z, w;
};


struct long1
{
  long int x;
};


struct ulong1
{
  unsigned long x;
};


struct



      

                                             long2
{
  long int x, y;
};


struct



      

                                                      ulong2
{
  unsigned long int x, y;
};
# 262 "/usr/local/cuda/bin/../include/vector_types.h" 3
struct float1
{
  float x;
};


struct float2
{
  float x, y;
};


struct float3
{
  float x, y, z;
};


struct float4
{
  float x, y, z, w;
};


struct longlong1
{
  long long int x;
};


struct ulonglong1
{
  unsigned long long int x;
};


struct longlong2
{
  long long int x, y;
};


struct ulonglong2
{
  unsigned long long int x, y;
};


struct double1
{
  double x;
};


struct double2
{
  double x, y;
};
# 328 "/usr/local/cuda/bin/../include/vector_types.h" 3
typedef struct char1 char1;

typedef struct uchar1 uchar1;

typedef struct char2 char2;

typedef struct uchar2 uchar2;

typedef struct char3 char3;

typedef struct uchar3 uchar3;

typedef struct char4 char4;

typedef struct uchar4 uchar4;

typedef struct short1 short1;

typedef struct ushort1 ushort1;

typedef struct short2 short2;

typedef struct ushort2 ushort2;

typedef struct short3 short3;

typedef struct ushort3 ushort3;

typedef struct short4 short4;

typedef struct ushort4 ushort4;

typedef struct int1 int1;

typedef struct uint1 uint1;

typedef struct int2 int2;

typedef struct uint2 uint2;

typedef struct int3 int3;

typedef struct uint3 uint3;

typedef struct int4 int4;

typedef struct uint4 uint4;

typedef struct long1 long1;

typedef struct ulong1 ulong1;

typedef struct long2 long2;

typedef struct ulong2 ulong2;

typedef struct long3 long3;

typedef struct ulong3 ulong3;

typedef struct long4 long4;

typedef struct ulong4 ulong4;

typedef struct float1 float1;

typedef struct float2 float2;

typedef struct float3 float3;

typedef struct float4 float4;

typedef struct longlong1 longlong1;

typedef struct ulonglong1 ulonglong1;

typedef struct longlong2 longlong2;

typedef struct ulonglong2 ulonglong2;

typedef struct double1 double1;

typedef struct double2 double2;
# 419 "/usr/local/cuda/bin/../include/vector_types.h" 3
typedef struct dim3 dim3;


struct dim3
{
    unsigned int x, y, z;

    dim3(unsigned int x = 1, unsigned int y = 1, unsigned int z = 1) : x(x), y(y), z(z) {}
    dim3(uint3 v) : x(v.x), y(v.y), z(v.z) {}
    operator uint3(void) { uint3 t; t.x = x; t.y = y; t.z = z; return t; }

};
# 45 "/usr/local/cuda/bin/../include/builtin_types.h" 2 3
# 56 "/usr/local/cuda/bin/../include/cuda_runtime_api.h" 2 3
# 79 "/usr/local/cuda/bin/../include/cuda_runtime_api.h" 3
extern "C" {
# 88 "/usr/local/cuda/bin/../include/cuda_runtime_api.h" 3
extern cudaError_t cudaMalloc3D(struct cudaPitchedPtr* pitchDevPtr, struct cudaExtent extent);
extern cudaError_t cudaMalloc3DArray(struct cudaArray** arrayPtr, const struct cudaChannelFormatDesc* desc, struct cudaExtent extent);
extern cudaError_t cudaMemset3D(struct cudaPitchedPtr pitchDevPtr, int value, struct cudaExtent extent);
extern cudaError_t cudaMemcpy3D(const struct cudaMemcpy3DParms *p);
extern cudaError_t cudaMemcpy3DAsync(const struct cudaMemcpy3DParms *p, cudaStream_t stream);
# 101 "/usr/local/cuda/bin/../include/cuda_runtime_api.h" 3
extern cudaError_t cudaMalloc(void **devPtr, size_t size);
extern cudaError_t cudaMallocHost(void **ptr, size_t size);
extern cudaError_t cudaMallocPitch(void **devPtr, size_t *pitch, size_t width, size_t height);
extern cudaError_t cudaMallocArray(struct cudaArray **array, const struct cudaChannelFormatDesc *desc, size_t width, size_t height );
extern cudaError_t cudaFree(void *devPtr);
extern cudaError_t cudaFreeHost(void *ptr);
extern cudaError_t cudaFreeArray(struct cudaArray *array);
# 116 "/usr/local/cuda/bin/../include/cuda_runtime_api.h" 3
extern cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind);
extern cudaError_t cudaMemcpyToArray(struct cudaArray *dst, size_t wOffset, size_t hOffset, const void *src, size_t count, enum cudaMemcpyKind kind);
extern cudaError_t cudaMemcpyFromArray(void *dst, const struct cudaArray *src, size_t wOffset, size_t hOffset, size_t count, enum cudaMemcpyKind kind);
extern cudaError_t cudaMemcpyArrayToArray(struct cudaArray *dst, size_t wOffsetDst, size_t hOffsetDst, const struct cudaArray *src, size_t wOffsetSrc, size_t hOffsetSrc, size_t count, enum cudaMemcpyKind kind );
extern cudaError_t cudaMemcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind);
extern cudaError_t cudaMemcpy2DToArray(struct cudaArray *dst, size_t wOffset, size_t hOffset, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind);
extern cudaError_t cudaMemcpy2DFromArray(void *dst, size_t dpitch, const struct cudaArray *src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind);
extern cudaError_t cudaMemcpy2DArrayToArray(struct cudaArray *dst, size_t wOffsetDst, size_t hOffsetDst, const struct cudaArray *src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, enum cudaMemcpyKind kind );
extern cudaError_t cudaMemcpyToSymbol(const char *symbol, const void *src, size_t count, size_t offset , enum cudaMemcpyKind kind );
extern cudaError_t cudaMemcpyFromSymbol(void *dst, const char *symbol, size_t count, size_t offset , enum cudaMemcpyKind kind );







extern cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream);
extern cudaError_t cudaMemcpyToArrayAsync(struct cudaArray *dst, size_t wOffset, size_t hOffset, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream);
extern cudaError_t cudaMemcpyFromArrayAsync(void *dst, const struct cudaArray *src, size_t wOffset, size_t hOffset, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream);
extern cudaError_t cudaMemcpy2DAsync(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream);
extern cudaError_t cudaMemcpy2DToArrayAsync(struct cudaArray *dst, size_t wOffset, size_t hOffset, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream);
extern cudaError_t cudaMemcpy2DFromArrayAsync(void *dst, size_t dpitch, const struct cudaArray *src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream);
extern cudaError_t cudaMemcpyToSymbolAsync(const char *symbol, const void *src, size_t count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream);
extern cudaError_t cudaMemcpyFromSymbolAsync(void *dst, const char *symbol, size_t count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream);







extern cudaError_t cudaMemset(void *mem, int c, size_t count);
extern cudaError_t cudaMemset2D(void *mem, size_t pitch, int c, size_t width, size_t height);







extern cudaError_t cudaGetSymbolAddress(void **devPtr, const char *symbol);
extern cudaError_t cudaGetSymbolSize(size_t *size, const char *symbol);







extern cudaError_t cudaGetDeviceCount(int *count);
extern cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device);
extern cudaError_t cudaChooseDevice(int *device, const struct cudaDeviceProp *prop);
extern cudaError_t cudaSetDevice(int device);
extern cudaError_t cudaGetDevice(int *device);







extern cudaError_t cudaBindTexture(size_t *offset, const struct textureReference *texref, const void *devPtr, const struct cudaChannelFormatDesc *desc, size_t size );
extern cudaError_t cudaBindTextureToArray(const struct textureReference *texref, const struct cudaArray *array, const struct cudaChannelFormatDesc *desc);
extern cudaError_t cudaUnbindTexture(const struct textureReference *texref);
extern cudaError_t cudaGetTextureAlignmentOffset(size_t *offset, const struct textureReference *texref);
extern cudaError_t cudaGetTextureReference(const struct textureReference **texref, const char *symbol);







extern cudaError_t cudaGetChannelDesc(struct cudaChannelFormatDesc *desc, const struct cudaArray *array);
extern struct cudaChannelFormatDesc cudaCreateChannelDesc(int x, int y, int z, int w, enum cudaChannelFormatKind f);







extern cudaError_t cudaGetLastError(void);
extern const char* cudaGetErrorString(cudaError_t error);







extern cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem , cudaStream_t stream );
extern cudaError_t cudaSetupArgument(const void *arg, size_t size, size_t offset);
extern cudaError_t cudaLaunch(const char *symbol);







extern cudaError_t cudaStreamCreate(cudaStream_t *stream);
extern cudaError_t cudaStreamDestroy(cudaStream_t stream);
extern cudaError_t cudaStreamSynchronize(cudaStream_t stream);
extern cudaError_t cudaStreamQuery(cudaStream_t stream);







extern cudaError_t cudaEventCreate(cudaEvent_t *event);
extern cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream);
extern cudaError_t cudaEventQuery(cudaEvent_t event);
extern cudaError_t cudaEventSynchronize(cudaEvent_t event);
extern cudaError_t cudaEventDestroy(cudaEvent_t event);
extern cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end);







extern cudaError_t cudaSetDoubleForDevice(double *d);
extern cudaError_t cudaSetDoubleForHost(double *d);







extern cudaError_t cudaThreadExit(void);
extern cudaError_t cudaThreadSynchronize(void);


}
# 70 "/usr/local/cuda/bin/../include/crt/host_runtime.h" 2 3
# 1 "/usr/local/cuda/bin/../include/crt/storage_class.h" 1 3
# 71 "/usr/local/cuda/bin/../include/crt/host_runtime.h" 2 3
# 216 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/include/stddef.h" 2 3
# 88 "/usr/local/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaMalloc3D(cudaPitchedPtr *, cudaExtent);
# 89 "/usr/local/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaMalloc3DArray(cudaArray **, const cudaChannelFormatDesc *, cudaExtent);
# 90 "/usr/local/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaMemset3D(cudaPitchedPtr, int, cudaExtent);
# 91 "/usr/local/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaMemcpy3D(const cudaMemcpy3DParms *);
# 92 "/usr/local/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaMemcpy3DAsync(const cudaMemcpy3DParms *, cudaStream_t);
# 101 "/usr/local/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaMalloc(void **, size_t);
# 102 "/usr/local/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaMallocHost(void **, size_t);
# 103 "/usr/local/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaMallocPitch(void **, size_t *, size_t, size_t);
# 104 "/usr/local/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaMallocArray(cudaArray **, const cudaChannelFormatDesc *, size_t, size_t = (1));
# 105 "/usr/local/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaFree(void *);
# 106 "/usr/local/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaFreeHost(void *);
# 107 "/usr/local/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaFreeArray(cudaArray *);
# 116 "/usr/local/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaMemcpy(void *, const void *, size_t, cudaMemcpyKind);
# 117 "/usr/local/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaMemcpyToArray(cudaArray *, size_t, size_t, const void *, size_t, cudaMemcpyKind);
# 118 "/usr/local/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaMemcpyFromArray(void *, const cudaArray *, size_t, size_t, size_t, cudaMemcpyKind);
# 119 "/usr/local/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaMemcpyArrayToArray(cudaArray *, size_t, size_t, const cudaArray *, size_t, size_t, size_t, cudaMemcpyKind = cudaMemcpyDeviceToDevice);
# 120 "/usr/local/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaMemcpy2D(void *, size_t, const void *, size_t, size_t, size_t, cudaMemcpyKind);
# 121 "/usr/local/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaMemcpy2DToArray(cudaArray *, size_t, size_t, const void *, size_t, size_t, size_t, cudaMemcpyKind);
# 122 "/usr/local/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaMemcpy2DFromArray(void *, size_t, const cudaArray *, size_t, size_t, size_t, size_t, cudaMemcpyKind);
# 123 "/usr/local/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaMemcpy2DArrayToArray(cudaArray *, size_t, size_t, const cudaArray *, size_t, size_t, size_t, size_t, cudaMemcpyKind = cudaMemcpyDeviceToDevice);
# 124 "/usr/local/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaMemcpyToSymbol(const char *, const void *, size_t, size_t = (0), cudaMemcpyKind = cudaMemcpyHostToDevice);
# 125 "/usr/local/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaMemcpyFromSymbol(void *, const char *, size_t, size_t = (0), cudaMemcpyKind = cudaMemcpyDeviceToHost);
# 133 "/usr/local/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaMemcpyAsync(void *, const void *, size_t, cudaMemcpyKind, cudaStream_t);
# 134 "/usr/local/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaMemcpyToArrayAsync(cudaArray *, size_t, size_t, const void *, size_t, cudaMemcpyKind, cudaStream_t);
# 135 "/usr/local/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaMemcpyFromArrayAsync(void *, const cudaArray *, size_t, size_t, size_t, cudaMemcpyKind, cudaStream_t);
# 136 "/usr/local/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaMemcpy2DAsync(void *, size_t, const void *, size_t, size_t, size_t, cudaMemcpyKind, cudaStream_t);
# 137 "/usr/local/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaMemcpy2DToArrayAsync(cudaArray *, size_t, size_t, const void *, size_t, size_t, size_t, cudaMemcpyKind, cudaStream_t);
# 138 "/usr/local/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaMemcpy2DFromArrayAsync(void *, size_t, const cudaArray *, size_t, size_t, size_t, size_t, cudaMemcpyKind, cudaStream_t);
# 139 "/usr/local/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaMemcpyToSymbolAsync(const char *, const void *, size_t, size_t, cudaMemcpyKind, cudaStream_t);
# 140 "/usr/local/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaMemcpyFromSymbolAsync(void *, const char *, size_t, size_t, cudaMemcpyKind, cudaStream_t);
# 148 "/usr/local/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaMemset(void *, int, size_t);
# 149 "/usr/local/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaMemset2D(void *, size_t, int, size_t, size_t);
# 157 "/usr/local/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaGetSymbolAddress(void **, const char *);
# 158 "/usr/local/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaGetSymbolSize(size_t *, const char *);
# 166 "/usr/local/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaGetDeviceCount(int *);
# 167 "/usr/local/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaGetDeviceProperties(cudaDeviceProp *, int);
# 168 "/usr/local/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaChooseDevice(int *, const cudaDeviceProp *);
# 169 "/usr/local/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaSetDevice(int);
# 170 "/usr/local/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaGetDevice(int *);
# 178 "/usr/local/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaBindTexture(size_t *, const textureReference *, const void *, const cudaChannelFormatDesc *, size_t = (((2147483647) * 2U) + 1U));
# 179 "/usr/local/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaBindTextureToArray(const textureReference *, const cudaArray *, const cudaChannelFormatDesc *);
# 180 "/usr/local/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaUnbindTexture(const textureReference *);
# 181 "/usr/local/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaGetTextureAlignmentOffset(size_t *, const textureReference *);
# 182 "/usr/local/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaGetTextureReference(const textureReference **, const char *);
# 190 "/usr/local/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaGetChannelDesc(cudaChannelFormatDesc *, const cudaArray *);
# 191 "/usr/local/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaChannelFormatDesc cudaCreateChannelDesc(int, int, int, int, cudaChannelFormatKind);
# 199 "/usr/local/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaGetLastError();
# 200 "/usr/local/cuda/bin/../include/cuda_runtime_api.h"
extern "C" const char *cudaGetErrorString(cudaError_t);
# 208 "/usr/local/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaConfigureCall(dim3, dim3, size_t = (0), cudaStream_t = (0));
# 209 "/usr/local/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaSetupArgument(const void *, size_t, size_t);
# 210 "/usr/local/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaLaunch(const char *);
# 218 "/usr/local/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaStreamCreate(cudaStream_t *);
# 219 "/usr/local/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaStreamDestroy(cudaStream_t);
# 220 "/usr/local/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaStreamSynchronize(cudaStream_t);
# 221 "/usr/local/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaStreamQuery(cudaStream_t);
# 229 "/usr/local/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaEventCreate(cudaEvent_t *);
# 230 "/usr/local/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t);
# 231 "/usr/local/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaEventQuery(cudaEvent_t);
# 232 "/usr/local/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaEventSynchronize(cudaEvent_t);
# 233 "/usr/local/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaEventDestroy(cudaEvent_t);
# 234 "/usr/local/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaEventElapsedTime(float *, cudaEvent_t, cudaEvent_t);
# 242 "/usr/local/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaSetDoubleForDevice(double *);
# 243 "/usr/local/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaSetDoubleForHost(double *);
# 251 "/usr/local/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaThreadExit();
# 252 "/usr/local/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaThreadSynchronize();
# 58 "/usr/local/cuda/bin/../include/channel_descriptor.h"
template<class T> inline cudaChannelFormatDesc cudaCreateChannelDesc()
# 59 "/usr/local/cuda/bin/../include/channel_descriptor.h"
{
# 60 "/usr/local/cuda/bin/../include/channel_descriptor.h"
return cudaCreateChannelDesc(0, 0, 0, 0, cudaChannelFormatKindNone);
# 61 "/usr/local/cuda/bin/../include/channel_descriptor.h"
}
# 63 "/usr/local/cuda/bin/../include/channel_descriptor.h"
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< char> ()
# 64 "/usr/local/cuda/bin/../include/channel_descriptor.h"
{
# 65 "/usr/local/cuda/bin/../include/channel_descriptor.h"
auto int e = (((int)sizeof(char)) * 8);
# 70 "/usr/local/cuda/bin/../include/channel_descriptor.h"
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned);
# 72 "/usr/local/cuda/bin/../include/channel_descriptor.h"
}
# 74 "/usr/local/cuda/bin/../include/channel_descriptor.h"
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< signed char> ()
# 75 "/usr/local/cuda/bin/../include/channel_descriptor.h"
{
# 76 "/usr/local/cuda/bin/../include/channel_descriptor.h"
auto int e = (((int)sizeof(signed char)) * 8);
# 78 "/usr/local/cuda/bin/../include/channel_descriptor.h"
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned);
# 79 "/usr/local/cuda/bin/../include/channel_descriptor.h"
}
# 81 "/usr/local/cuda/bin/../include/channel_descriptor.h"
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< unsigned char> ()
# 82 "/usr/local/cuda/bin/../include/channel_descriptor.h"
{
# 83 "/usr/local/cuda/bin/../include/channel_descriptor.h"
auto int e = (((int)sizeof(unsigned char)) * 8);
# 85 "/usr/local/cuda/bin/../include/channel_descriptor.h"
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned);
# 86 "/usr/local/cuda/bin/../include/channel_descriptor.h"
}
# 88 "/usr/local/cuda/bin/../include/channel_descriptor.h"
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< char1> ()
# 89 "/usr/local/cuda/bin/../include/channel_descriptor.h"
{
# 90 "/usr/local/cuda/bin/../include/channel_descriptor.h"
auto int e = (((int)sizeof(signed char)) * 8);
# 92 "/usr/local/cuda/bin/../include/channel_descriptor.h"
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned);
# 93 "/usr/local/cuda/bin/../include/channel_descriptor.h"
}
# 95 "/usr/local/cuda/bin/../include/channel_descriptor.h"
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< uchar1> ()
# 96 "/usr/local/cuda/bin/../include/channel_descriptor.h"
{
# 97 "/usr/local/cuda/bin/../include/channel_descriptor.h"
auto int e = (((int)sizeof(unsigned char)) * 8);
# 99 "/usr/local/cuda/bin/../include/channel_descriptor.h"
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned);
# 100 "/usr/local/cuda/bin/../include/channel_descriptor.h"
}
# 102 "/usr/local/cuda/bin/../include/channel_descriptor.h"
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< char2> ()
# 103 "/usr/local/cuda/bin/../include/channel_descriptor.h"
{
# 104 "/usr/local/cuda/bin/../include/channel_descriptor.h"
auto int e = (((int)sizeof(signed char)) * 8);
# 106 "/usr/local/cuda/bin/../include/channel_descriptor.h"
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindSigned);
# 107 "/usr/local/cuda/bin/../include/channel_descriptor.h"
}
# 109 "/usr/local/cuda/bin/../include/channel_descriptor.h"
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< uchar2> ()
# 110 "/usr/local/cuda/bin/../include/channel_descriptor.h"
{
# 111 "/usr/local/cuda/bin/../include/channel_descriptor.h"
auto int e = (((int)sizeof(unsigned char)) * 8);
# 113 "/usr/local/cuda/bin/../include/channel_descriptor.h"
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindUnsigned);
# 114 "/usr/local/cuda/bin/../include/channel_descriptor.h"
}
# 116 "/usr/local/cuda/bin/../include/channel_descriptor.h"
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< char4> ()
# 117 "/usr/local/cuda/bin/../include/channel_descriptor.h"
{
# 118 "/usr/local/cuda/bin/../include/channel_descriptor.h"
auto int e = (((int)sizeof(signed char)) * 8);
# 120 "/usr/local/cuda/bin/../include/channel_descriptor.h"
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindSigned);
# 121 "/usr/local/cuda/bin/../include/channel_descriptor.h"
}
# 123 "/usr/local/cuda/bin/../include/channel_descriptor.h"
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< uchar4> ()
# 124 "/usr/local/cuda/bin/../include/channel_descriptor.h"
{
# 125 "/usr/local/cuda/bin/../include/channel_descriptor.h"
auto int e = (((int)sizeof(unsigned char)) * 8);
# 127 "/usr/local/cuda/bin/../include/channel_descriptor.h"
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindUnsigned);
# 128 "/usr/local/cuda/bin/../include/channel_descriptor.h"
}
# 130 "/usr/local/cuda/bin/../include/channel_descriptor.h"
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< short> ()
# 131 "/usr/local/cuda/bin/../include/channel_descriptor.h"
{
# 132 "/usr/local/cuda/bin/../include/channel_descriptor.h"
auto int e = (((int)sizeof(short)) * 8);
# 134 "/usr/local/cuda/bin/../include/channel_descriptor.h"
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned);
# 135 "/usr/local/cuda/bin/../include/channel_descriptor.h"
}
# 137 "/usr/local/cuda/bin/../include/channel_descriptor.h"
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< unsigned short> ()
# 138 "/usr/local/cuda/bin/../include/channel_descriptor.h"
{
# 139 "/usr/local/cuda/bin/../include/channel_descriptor.h"
auto int e = (((int)sizeof(unsigned short)) * 8);
# 141 "/usr/local/cuda/bin/../include/channel_descriptor.h"
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned);
# 142 "/usr/local/cuda/bin/../include/channel_descriptor.h"
}
# 144 "/usr/local/cuda/bin/../include/channel_descriptor.h"
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< short1> ()
# 145 "/usr/local/cuda/bin/../include/channel_descriptor.h"
{
# 146 "/usr/local/cuda/bin/../include/channel_descriptor.h"
auto int e = (((int)sizeof(short)) * 8);
# 148 "/usr/local/cuda/bin/../include/channel_descriptor.h"
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned);
# 149 "/usr/local/cuda/bin/../include/channel_descriptor.h"
}
# 151 "/usr/local/cuda/bin/../include/channel_descriptor.h"
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< ushort1> ()
# 152 "/usr/local/cuda/bin/../include/channel_descriptor.h"
{
# 153 "/usr/local/cuda/bin/../include/channel_descriptor.h"
auto int e = (((int)sizeof(unsigned short)) * 8);
# 155 "/usr/local/cuda/bin/../include/channel_descriptor.h"
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned);
# 156 "/usr/local/cuda/bin/../include/channel_descriptor.h"
}
# 158 "/usr/local/cuda/bin/../include/channel_descriptor.h"
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< short2> ()
# 159 "/usr/local/cuda/bin/../include/channel_descriptor.h"
{
# 160 "/usr/local/cuda/bin/../include/channel_descriptor.h"
auto int e = (((int)sizeof(short)) * 8);
# 162 "/usr/local/cuda/bin/../include/channel_descriptor.h"
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindSigned);
# 163 "/usr/local/cuda/bin/../include/channel_descriptor.h"
}
# 165 "/usr/local/cuda/bin/../include/channel_descriptor.h"
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< ushort2> ()
# 166 "/usr/local/cuda/bin/../include/channel_descriptor.h"
{
# 167 "/usr/local/cuda/bin/../include/channel_descriptor.h"
auto int e = (((int)sizeof(unsigned short)) * 8);
# 169 "/usr/local/cuda/bin/../include/channel_descriptor.h"
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindUnsigned);
# 170 "/usr/local/cuda/bin/../include/channel_descriptor.h"
}
# 172 "/usr/local/cuda/bin/../include/channel_descriptor.h"
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< short4> ()
# 173 "/usr/local/cuda/bin/../include/channel_descriptor.h"
{
# 174 "/usr/local/cuda/bin/../include/channel_descriptor.h"
auto int e = (((int)sizeof(short)) * 8);
# 176 "/usr/local/cuda/bin/../include/channel_descriptor.h"
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindSigned);
# 177 "/usr/local/cuda/bin/../include/channel_descriptor.h"
}
# 179 "/usr/local/cuda/bin/../include/channel_descriptor.h"
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< ushort4> ()
# 180 "/usr/local/cuda/bin/../include/channel_descriptor.h"
{
# 181 "/usr/local/cuda/bin/../include/channel_descriptor.h"
auto int e = (((int)sizeof(unsigned short)) * 8);
# 183 "/usr/local/cuda/bin/../include/channel_descriptor.h"
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindUnsigned);
# 184 "/usr/local/cuda/bin/../include/channel_descriptor.h"
}
# 186 "/usr/local/cuda/bin/../include/channel_descriptor.h"
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< int> ()
# 187 "/usr/local/cuda/bin/../include/channel_descriptor.h"
{
# 188 "/usr/local/cuda/bin/../include/channel_descriptor.h"
auto int e = (((int)sizeof(int)) * 8);
# 190 "/usr/local/cuda/bin/../include/channel_descriptor.h"
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned);
# 191 "/usr/local/cuda/bin/../include/channel_descriptor.h"
}
# 193 "/usr/local/cuda/bin/../include/channel_descriptor.h"
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< unsigned> ()
# 194 "/usr/local/cuda/bin/../include/channel_descriptor.h"
{
# 195 "/usr/local/cuda/bin/../include/channel_descriptor.h"
auto int e = (((int)sizeof(unsigned)) * 8);
# 197 "/usr/local/cuda/bin/../include/channel_descriptor.h"
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned);
# 198 "/usr/local/cuda/bin/../include/channel_descriptor.h"
}
# 200 "/usr/local/cuda/bin/../include/channel_descriptor.h"
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< int1> ()
# 201 "/usr/local/cuda/bin/../include/channel_descriptor.h"
{
# 202 "/usr/local/cuda/bin/../include/channel_descriptor.h"
auto int e = (((int)sizeof(int)) * 8);
# 204 "/usr/local/cuda/bin/../include/channel_descriptor.h"
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned);
# 205 "/usr/local/cuda/bin/../include/channel_descriptor.h"
}
# 207 "/usr/local/cuda/bin/../include/channel_descriptor.h"
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< uint1> ()
# 208 "/usr/local/cuda/bin/../include/channel_descriptor.h"
{
# 209 "/usr/local/cuda/bin/../include/channel_descriptor.h"
auto int e = (((int)sizeof(unsigned)) * 8);
# 211 "/usr/local/cuda/bin/../include/channel_descriptor.h"
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned);
# 212 "/usr/local/cuda/bin/../include/channel_descriptor.h"
}
# 214 "/usr/local/cuda/bin/../include/channel_descriptor.h"
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< int2> ()
# 215 "/usr/local/cuda/bin/../include/channel_descriptor.h"
{
# 216 "/usr/local/cuda/bin/../include/channel_descriptor.h"
auto int e = (((int)sizeof(int)) * 8);
# 218 "/usr/local/cuda/bin/../include/channel_descriptor.h"
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindSigned);
# 219 "/usr/local/cuda/bin/../include/channel_descriptor.h"
}
# 221 "/usr/local/cuda/bin/../include/channel_descriptor.h"
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< uint2> ()
# 222 "/usr/local/cuda/bin/../include/channel_descriptor.h"
{
# 223 "/usr/local/cuda/bin/../include/channel_descriptor.h"
auto int e = (((int)sizeof(unsigned)) * 8);
# 225 "/usr/local/cuda/bin/../include/channel_descriptor.h"
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindUnsigned);
# 226 "/usr/local/cuda/bin/../include/channel_descriptor.h"
}
# 228 "/usr/local/cuda/bin/../include/channel_descriptor.h"
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< int4> ()
# 229 "/usr/local/cuda/bin/../include/channel_descriptor.h"
{
# 230 "/usr/local/cuda/bin/../include/channel_descriptor.h"
auto int e = (((int)sizeof(int)) * 8);
# 232 "/usr/local/cuda/bin/../include/channel_descriptor.h"
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindSigned);
# 233 "/usr/local/cuda/bin/../include/channel_descriptor.h"
}
# 235 "/usr/local/cuda/bin/../include/channel_descriptor.h"
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< uint4> ()
# 236 "/usr/local/cuda/bin/../include/channel_descriptor.h"
{
# 237 "/usr/local/cuda/bin/../include/channel_descriptor.h"
auto int e = (((int)sizeof(unsigned)) * 8);
# 239 "/usr/local/cuda/bin/../include/channel_descriptor.h"
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindUnsigned);
# 240 "/usr/local/cuda/bin/../include/channel_descriptor.h"
}
# 302 "/usr/local/cuda/bin/../include/channel_descriptor.h"
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< float> ()
# 303 "/usr/local/cuda/bin/../include/channel_descriptor.h"
{
# 304 "/usr/local/cuda/bin/../include/channel_descriptor.h"
auto int e = (((int)sizeof(float)) * 8);
# 306 "/usr/local/cuda/bin/../include/channel_descriptor.h"
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindFloat);
# 307 "/usr/local/cuda/bin/../include/channel_descriptor.h"
}
# 309 "/usr/local/cuda/bin/../include/channel_descriptor.h"
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< float1> ()
# 310 "/usr/local/cuda/bin/../include/channel_descriptor.h"
{
# 311 "/usr/local/cuda/bin/../include/channel_descriptor.h"
auto int e = (((int)sizeof(float)) * 8);
# 313 "/usr/local/cuda/bin/../include/channel_descriptor.h"
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindFloat);
# 314 "/usr/local/cuda/bin/../include/channel_descriptor.h"
}
# 316 "/usr/local/cuda/bin/../include/channel_descriptor.h"
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< float2> ()
# 317 "/usr/local/cuda/bin/../include/channel_descriptor.h"
{
# 318 "/usr/local/cuda/bin/../include/channel_descriptor.h"
auto int e = (((int)sizeof(float)) * 8);
# 320 "/usr/local/cuda/bin/../include/channel_descriptor.h"
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindFloat);
# 321 "/usr/local/cuda/bin/../include/channel_descriptor.h"
}
# 323 "/usr/local/cuda/bin/../include/channel_descriptor.h"
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< float4> ()
# 324 "/usr/local/cuda/bin/../include/channel_descriptor.h"
{
# 325 "/usr/local/cuda/bin/../include/channel_descriptor.h"
auto int e = (((int)sizeof(float)) * 8);
# 327 "/usr/local/cuda/bin/../include/channel_descriptor.h"
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindFloat);
# 328 "/usr/local/cuda/bin/../include/channel_descriptor.h"
}
# 54 "/usr/local/cuda/bin/../include/driver_functions.h"
static inline cudaPitchedPtr make_cudaPitchedPtr(void *d, size_t p, size_t xsz, size_t ysz)
# 55 "/usr/local/cuda/bin/../include/driver_functions.h"
{
# 56 "/usr/local/cuda/bin/../include/driver_functions.h"
auto cudaPitchedPtr s;
# 58 "/usr/local/cuda/bin/../include/driver_functions.h"
(s.ptr) = d;
# 59 "/usr/local/cuda/bin/../include/driver_functions.h"
(s.pitch) = p;
# 60 "/usr/local/cuda/bin/../include/driver_functions.h"
(s.xsize) = xsz;
# 61 "/usr/local/cuda/bin/../include/driver_functions.h"
(s.ysize) = ysz;
# 63 "/usr/local/cuda/bin/../include/driver_functions.h"
return s;
# 64 "/usr/local/cuda/bin/../include/driver_functions.h"
}
# 66 "/usr/local/cuda/bin/../include/driver_functions.h"
static inline cudaPos make_cudaPos(size_t x, size_t y, size_t z)
# 67 "/usr/local/cuda/bin/../include/driver_functions.h"
{
# 68 "/usr/local/cuda/bin/../include/driver_functions.h"
auto cudaPos p;
# 70 "/usr/local/cuda/bin/../include/driver_functions.h"
(p.x) = x;
# 71 "/usr/local/cuda/bin/../include/driver_functions.h"
(p.y) = y;
# 72 "/usr/local/cuda/bin/../include/driver_functions.h"
(p.z) = z;
# 74 "/usr/local/cuda/bin/../include/driver_functions.h"
return p;
# 75 "/usr/local/cuda/bin/../include/driver_functions.h"
}
# 77 "/usr/local/cuda/bin/../include/driver_functions.h"
static inline cudaExtent make_cudaExtent(size_t w, size_t h, size_t d)
# 78 "/usr/local/cuda/bin/../include/driver_functions.h"
{
# 79 "/usr/local/cuda/bin/../include/driver_functions.h"
auto cudaExtent e;
# 81 "/usr/local/cuda/bin/../include/driver_functions.h"
(e.width) = w;
# 82 "/usr/local/cuda/bin/../include/driver_functions.h"
(e.height) = h;
# 83 "/usr/local/cuda/bin/../include/driver_functions.h"
(e.depth) = d;
# 85 "/usr/local/cuda/bin/../include/driver_functions.h"
return e;
# 86 "/usr/local/cuda/bin/../include/driver_functions.h"
}
# 54 "/usr/local/cuda/bin/../include/vector_functions.h"
static inline char1 make_char1(signed char x)
# 55 "/usr/local/cuda/bin/../include/vector_functions.h"
{
# 56 "/usr/local/cuda/bin/../include/vector_functions.h"
auto char1 t; (t.x) = x; return t;
# 57 "/usr/local/cuda/bin/../include/vector_functions.h"
}
# 59 "/usr/local/cuda/bin/../include/vector_functions.h"
static inline uchar1 make_uchar1(unsigned char x)
# 60 "/usr/local/cuda/bin/../include/vector_functions.h"
{
# 61 "/usr/local/cuda/bin/../include/vector_functions.h"
auto uchar1 t; (t.x) = x; return t;
# 62 "/usr/local/cuda/bin/../include/vector_functions.h"
}
# 64 "/usr/local/cuda/bin/../include/vector_functions.h"
static inline char2 make_char2(signed char x, signed char y)
# 65 "/usr/local/cuda/bin/../include/vector_functions.h"
{
# 66 "/usr/local/cuda/bin/../include/vector_functions.h"
auto char2 t; (t.x) = x; (t.y) = y; return t;
# 67 "/usr/local/cuda/bin/../include/vector_functions.h"
}
# 69 "/usr/local/cuda/bin/../include/vector_functions.h"
static inline uchar2 make_uchar2(unsigned char x, unsigned char y)
# 70 "/usr/local/cuda/bin/../include/vector_functions.h"
{
# 71 "/usr/local/cuda/bin/../include/vector_functions.h"
auto uchar2 t; (t.x) = x; (t.y) = y; return t;
# 72 "/usr/local/cuda/bin/../include/vector_functions.h"
}
# 74 "/usr/local/cuda/bin/../include/vector_functions.h"
static inline char3 make_char3(signed char x, signed char y, signed char z)
# 75 "/usr/local/cuda/bin/../include/vector_functions.h"
{
# 76 "/usr/local/cuda/bin/../include/vector_functions.h"
auto char3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t;
# 77 "/usr/local/cuda/bin/../include/vector_functions.h"
}
# 79 "/usr/local/cuda/bin/../include/vector_functions.h"
static inline uchar3 make_uchar3(unsigned char x, unsigned char y, unsigned char z)
# 80 "/usr/local/cuda/bin/../include/vector_functions.h"
{
# 81 "/usr/local/cuda/bin/../include/vector_functions.h"
auto uchar3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t;
# 82 "/usr/local/cuda/bin/../include/vector_functions.h"
}
# 84 "/usr/local/cuda/bin/../include/vector_functions.h"
static inline char4 make_char4(signed char x, signed char y, signed char z, signed char w)
# 85 "/usr/local/cuda/bin/../include/vector_functions.h"
{
# 86 "/usr/local/cuda/bin/../include/vector_functions.h"
auto char4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t;
# 87 "/usr/local/cuda/bin/../include/vector_functions.h"
}
# 89 "/usr/local/cuda/bin/../include/vector_functions.h"
static inline uchar4 make_uchar4(unsigned char x, unsigned char y, unsigned char z, unsigned char w)
# 90 "/usr/local/cuda/bin/../include/vector_functions.h"
{
# 91 "/usr/local/cuda/bin/../include/vector_functions.h"
auto uchar4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t;
# 92 "/usr/local/cuda/bin/../include/vector_functions.h"
}
# 94 "/usr/local/cuda/bin/../include/vector_functions.h"
static inline short1 make_short1(short x)
# 95 "/usr/local/cuda/bin/../include/vector_functions.h"
{
# 96 "/usr/local/cuda/bin/../include/vector_functions.h"
auto short1 t; (t.x) = x; return t;
# 97 "/usr/local/cuda/bin/../include/vector_functions.h"
}
# 99 "/usr/local/cuda/bin/../include/vector_functions.h"
static inline ushort1 make_ushort1(unsigned short x)
# 100 "/usr/local/cuda/bin/../include/vector_functions.h"
{
# 101 "/usr/local/cuda/bin/../include/vector_functions.h"
auto ushort1 t; (t.x) = x; return t;
# 102 "/usr/local/cuda/bin/../include/vector_functions.h"
}
# 104 "/usr/local/cuda/bin/../include/vector_functions.h"
static inline short2 make_short2(short x, short y)
# 105 "/usr/local/cuda/bin/../include/vector_functions.h"
{
# 106 "/usr/local/cuda/bin/../include/vector_functions.h"
auto short2 t; (t.x) = x; (t.y) = y; return t;
# 107 "/usr/local/cuda/bin/../include/vector_functions.h"
}
# 109 "/usr/local/cuda/bin/../include/vector_functions.h"
static inline ushort2 make_ushort2(unsigned short x, unsigned short y)
# 110 "/usr/local/cuda/bin/../include/vector_functions.h"
{
# 111 "/usr/local/cuda/bin/../include/vector_functions.h"
auto ushort2 t; (t.x) = x; (t.y) = y; return t;
# 112 "/usr/local/cuda/bin/../include/vector_functions.h"
}
# 114 "/usr/local/cuda/bin/../include/vector_functions.h"
static inline short3 make_short3(short x, short y, short z)
# 115 "/usr/local/cuda/bin/../include/vector_functions.h"
{
# 116 "/usr/local/cuda/bin/../include/vector_functions.h"
auto short3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t;
# 117 "/usr/local/cuda/bin/../include/vector_functions.h"
}
# 119 "/usr/local/cuda/bin/../include/vector_functions.h"
static inline ushort3 make_ushort3(unsigned short x, unsigned short y, unsigned short z)
# 120 "/usr/local/cuda/bin/../include/vector_functions.h"
{
# 121 "/usr/local/cuda/bin/../include/vector_functions.h"
auto ushort3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t;
# 122 "/usr/local/cuda/bin/../include/vector_functions.h"
}
# 124 "/usr/local/cuda/bin/../include/vector_functions.h"
static inline short4 make_short4(short x, short y, short z, short w)
# 125 "/usr/local/cuda/bin/../include/vector_functions.h"
{
# 126 "/usr/local/cuda/bin/../include/vector_functions.h"
auto short4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t;
# 127 "/usr/local/cuda/bin/../include/vector_functions.h"
}
# 129 "/usr/local/cuda/bin/../include/vector_functions.h"
static inline ushort4 make_ushort4(unsigned short x, unsigned short y, unsigned short z, unsigned short w)
# 130 "/usr/local/cuda/bin/../include/vector_functions.h"
{
# 131 "/usr/local/cuda/bin/../include/vector_functions.h"
auto ushort4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t;
# 132 "/usr/local/cuda/bin/../include/vector_functions.h"
}
# 134 "/usr/local/cuda/bin/../include/vector_functions.h"
static inline int1 make_int1(int x)
# 135 "/usr/local/cuda/bin/../include/vector_functions.h"
{
# 136 "/usr/local/cuda/bin/../include/vector_functions.h"
auto int1 t; (t.x) = x; return t;
# 137 "/usr/local/cuda/bin/../include/vector_functions.h"
}
# 139 "/usr/local/cuda/bin/../include/vector_functions.h"
static inline uint1 make_uint1(unsigned x)
# 140 "/usr/local/cuda/bin/../include/vector_functions.h"
{
# 141 "/usr/local/cuda/bin/../include/vector_functions.h"
auto uint1 t; (t.x) = x; return t;
# 142 "/usr/local/cuda/bin/../include/vector_functions.h"
}
# 144 "/usr/local/cuda/bin/../include/vector_functions.h"
static inline int2 make_int2(int x, int y)
# 145 "/usr/local/cuda/bin/../include/vector_functions.h"
{
# 146 "/usr/local/cuda/bin/../include/vector_functions.h"
auto int2 t; (t.x) = x; (t.y) = y; return t;
# 147 "/usr/local/cuda/bin/../include/vector_functions.h"
}
# 149 "/usr/local/cuda/bin/../include/vector_functions.h"
static inline uint2 make_uint2(unsigned x, unsigned y)
# 150 "/usr/local/cuda/bin/../include/vector_functions.h"
{
# 151 "/usr/local/cuda/bin/../include/vector_functions.h"
auto uint2 t; (t.x) = x; (t.y) = y; return t;
# 152 "/usr/local/cuda/bin/../include/vector_functions.h"
}
# 154 "/usr/local/cuda/bin/../include/vector_functions.h"
static inline int3 make_int3(int x, int y, int z)
# 155 "/usr/local/cuda/bin/../include/vector_functions.h"
{
# 156 "/usr/local/cuda/bin/../include/vector_functions.h"
auto int3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t;
# 157 "/usr/local/cuda/bin/../include/vector_functions.h"
}
# 159 "/usr/local/cuda/bin/../include/vector_functions.h"
static inline uint3 make_uint3(unsigned x, unsigned y, unsigned z)
# 160 "/usr/local/cuda/bin/../include/vector_functions.h"
{
# 161 "/usr/local/cuda/bin/../include/vector_functions.h"
auto uint3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t;
# 162 "/usr/local/cuda/bin/../include/vector_functions.h"
}
# 164 "/usr/local/cuda/bin/../include/vector_functions.h"
static inline int4 make_int4(int x, int y, int z, int w)
# 165 "/usr/local/cuda/bin/../include/vector_functions.h"
{
# 166 "/usr/local/cuda/bin/../include/vector_functions.h"
auto int4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t;
# 167 "/usr/local/cuda/bin/../include/vector_functions.h"
}
# 169 "/usr/local/cuda/bin/../include/vector_functions.h"
static inline uint4 make_uint4(unsigned x, unsigned y, unsigned z, unsigned w)
# 170 "/usr/local/cuda/bin/../include/vector_functions.h"
{
# 171 "/usr/local/cuda/bin/../include/vector_functions.h"
auto uint4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t;
# 172 "/usr/local/cuda/bin/../include/vector_functions.h"
}
# 174 "/usr/local/cuda/bin/../include/vector_functions.h"
static inline long1 make_long1(long x)
# 175 "/usr/local/cuda/bin/../include/vector_functions.h"
{
# 176 "/usr/local/cuda/bin/../include/vector_functions.h"
auto long1 t; (t.x) = x; return t;
# 177 "/usr/local/cuda/bin/../include/vector_functions.h"
}
# 179 "/usr/local/cuda/bin/../include/vector_functions.h"
static inline ulong1 make_ulong1(unsigned long x)
# 180 "/usr/local/cuda/bin/../include/vector_functions.h"
{
# 181 "/usr/local/cuda/bin/../include/vector_functions.h"
auto ulong1 t; (t.x) = x; return t;
# 182 "/usr/local/cuda/bin/../include/vector_functions.h"
}
# 184 "/usr/local/cuda/bin/../include/vector_functions.h"
static inline long2 make_long2(long x, long y)
# 185 "/usr/local/cuda/bin/../include/vector_functions.h"
{
# 186 "/usr/local/cuda/bin/../include/vector_functions.h"
auto long2 t; (t.x) = x; (t.y) = y; return t;
# 187 "/usr/local/cuda/bin/../include/vector_functions.h"
}
# 189 "/usr/local/cuda/bin/../include/vector_functions.h"
static inline ulong2 make_ulong2(unsigned long x, unsigned long y)
# 190 "/usr/local/cuda/bin/../include/vector_functions.h"
{
# 191 "/usr/local/cuda/bin/../include/vector_functions.h"
auto ulong2 t; (t.x) = x; (t.y) = y; return t;
# 192 "/usr/local/cuda/bin/../include/vector_functions.h"
}
# 218 "/usr/local/cuda/bin/../include/vector_functions.h"
static inline float1 make_float1(float x)
# 219 "/usr/local/cuda/bin/../include/vector_functions.h"
{
# 220 "/usr/local/cuda/bin/../include/vector_functions.h"
auto float1 t; (t.x) = x; return t;
# 221 "/usr/local/cuda/bin/../include/vector_functions.h"
}
# 223 "/usr/local/cuda/bin/../include/vector_functions.h"
static inline float2 make_float2(float x, float y)
# 224 "/usr/local/cuda/bin/../include/vector_functions.h"
{
# 225 "/usr/local/cuda/bin/../include/vector_functions.h"
auto float2 t; (t.x) = x; (t.y) = y; return t;
# 226 "/usr/local/cuda/bin/../include/vector_functions.h"
}
# 228 "/usr/local/cuda/bin/../include/vector_functions.h"
static inline float3 make_float3(float x, float y, float z)
# 229 "/usr/local/cuda/bin/../include/vector_functions.h"
{
# 230 "/usr/local/cuda/bin/../include/vector_functions.h"
auto float3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t;
# 231 "/usr/local/cuda/bin/../include/vector_functions.h"
}
# 233 "/usr/local/cuda/bin/../include/vector_functions.h"
static inline float4 make_float4(float x, float y, float z, float w)
# 234 "/usr/local/cuda/bin/../include/vector_functions.h"
{
# 235 "/usr/local/cuda/bin/../include/vector_functions.h"
auto float4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t;
# 236 "/usr/local/cuda/bin/../include/vector_functions.h"
}
# 238 "/usr/local/cuda/bin/../include/vector_functions.h"
static inline longlong1 make_longlong1(long long x)
# 239 "/usr/local/cuda/bin/../include/vector_functions.h"
{
# 240 "/usr/local/cuda/bin/../include/vector_functions.h"
auto longlong1 t; (t.x) = x; return t;
# 241 "/usr/local/cuda/bin/../include/vector_functions.h"
}
# 243 "/usr/local/cuda/bin/../include/vector_functions.h"
static inline ulonglong1 make_ulonglong1(unsigned long long x)
# 244 "/usr/local/cuda/bin/../include/vector_functions.h"
{
# 245 "/usr/local/cuda/bin/../include/vector_functions.h"
auto ulonglong1 t; (t.x) = x; return t;
# 246 "/usr/local/cuda/bin/../include/vector_functions.h"
}
# 248 "/usr/local/cuda/bin/../include/vector_functions.h"
static inline longlong2 make_longlong2(long long x, long long y)
# 249 "/usr/local/cuda/bin/../include/vector_functions.h"
{
# 250 "/usr/local/cuda/bin/../include/vector_functions.h"
auto longlong2 t; (t.x) = x; (t.y) = y; return t;
# 251 "/usr/local/cuda/bin/../include/vector_functions.h"
}
# 253 "/usr/local/cuda/bin/../include/vector_functions.h"
static inline ulonglong2 make_ulonglong2(unsigned long long x, unsigned long long y)
# 254 "/usr/local/cuda/bin/../include/vector_functions.h"
{
# 255 "/usr/local/cuda/bin/../include/vector_functions.h"
auto ulonglong2 t; (t.x) = x; (t.y) = y; return t;
# 256 "/usr/local/cuda/bin/../include/vector_functions.h"
}
# 258 "/usr/local/cuda/bin/../include/vector_functions.h"
static inline double1 make_double1(double x)
# 259 "/usr/local/cuda/bin/../include/vector_functions.h"
{
# 260 "/usr/local/cuda/bin/../include/vector_functions.h"
auto double1 t; (t.x) = x; return t;
# 261 "/usr/local/cuda/bin/../include/vector_functions.h"
}
# 263 "/usr/local/cuda/bin/../include/vector_functions.h"
static inline double2 make_double2(double x, double y)
# 264 "/usr/local/cuda/bin/../include/vector_functions.h"
{
# 265 "/usr/local/cuda/bin/../include/vector_functions.h"
auto double2 t; (t.x) = x; (t.y) = y; return t;
# 266 "/usr/local/cuda/bin/../include/vector_functions.h"
}
# 31 "/usr/include/bits/types.h" 3
extern "C" { typedef unsigned char __u_char; }
# 32 "/usr/include/bits/types.h" 3
extern "C" { typedef unsigned short __u_short; }
# 33 "/usr/include/bits/types.h" 3
extern "C" { typedef unsigned __u_int; }
# 34 "/usr/include/bits/types.h" 3
extern "C" { typedef unsigned long __u_long; }
# 37 "/usr/include/bits/types.h" 3
extern "C" { typedef signed char __int8_t; }
# 38 "/usr/include/bits/types.h" 3
extern "C" { typedef unsigned char __uint8_t; }
# 39 "/usr/include/bits/types.h" 3
extern "C" { typedef signed short __int16_t; }
# 40 "/usr/include/bits/types.h" 3
extern "C" { typedef unsigned short __uint16_t; }
# 41 "/usr/include/bits/types.h" 3
extern "C" { typedef signed int __int32_t; }
# 42 "/usr/include/bits/types.h" 3
extern "C" { typedef unsigned __uint32_t; }
# 44 "/usr/include/bits/types.h" 3
extern "C" { typedef signed long __int64_t; }
# 45 "/usr/include/bits/types.h" 3
extern "C" { typedef unsigned long __uint64_t; }
# 53 "/usr/include/bits/types.h" 3
extern "C" { typedef long __quad_t; }
# 54 "/usr/include/bits/types.h" 3
extern "C" { typedef unsigned long __u_quad_t; }
# 134 "/usr/include/bits/types.h" 3
extern "C" { typedef unsigned long __dev_t; }
# 135 "/usr/include/bits/types.h" 3
extern "C" { typedef unsigned __uid_t; }
# 136 "/usr/include/bits/types.h" 3
extern "C" { typedef unsigned __gid_t; }
# 137 "/usr/include/bits/types.h" 3
extern "C" { typedef unsigned long __ino_t; }
# 138 "/usr/include/bits/types.h" 3
extern "C" { typedef unsigned long __ino64_t; }
# 139 "/usr/include/bits/types.h" 3
extern "C" { typedef unsigned __mode_t; }
# 140 "/usr/include/bits/types.h" 3
extern "C" { typedef unsigned long __nlink_t; }
# 141 "/usr/include/bits/types.h" 3
extern "C" { typedef long __off_t; }
# 142 "/usr/include/bits/types.h" 3
extern "C" { typedef long __off64_t; }
# 143 "/usr/include/bits/types.h" 3
extern "C" { typedef int __pid_t; }
# 144 "/usr/include/bits/types.h" 3
extern "C" { typedef struct __fsid_t { int __val[2]; } __fsid_t; }
# 145 "/usr/include/bits/types.h" 3
extern "C" { typedef long __clock_t; }
# 146 "/usr/include/bits/types.h" 3
extern "C" { typedef unsigned long __rlim_t; }
# 147 "/usr/include/bits/types.h" 3
extern "C" { typedef unsigned long __rlim64_t; }
# 148 "/usr/include/bits/types.h" 3
extern "C" { typedef unsigned __id_t; }
# 149 "/usr/include/bits/types.h" 3
extern "C" { typedef long __time_t; }
# 150 "/usr/include/bits/types.h" 3
extern "C" { typedef unsigned __useconds_t; }
# 151 "/usr/include/bits/types.h" 3
extern "C" { typedef long __suseconds_t; }
# 153 "/usr/include/bits/types.h" 3
extern "C" { typedef int __daddr_t; }
# 154 "/usr/include/bits/types.h" 3
extern "C" { typedef long __swblk_t; }
# 155 "/usr/include/bits/types.h" 3
extern "C" { typedef int __key_t; }
# 158 "/usr/include/bits/types.h" 3
extern "C" { typedef int __clockid_t; }
# 161 "/usr/include/bits/types.h" 3
extern "C" { typedef void *__timer_t; }
# 164 "/usr/include/bits/types.h" 3
extern "C" { typedef long __blksize_t; }
# 169 "/usr/include/bits/types.h" 3
extern "C" { typedef long __blkcnt_t; }
# 170 "/usr/include/bits/types.h" 3
extern "C" { typedef long __blkcnt64_t; }
# 173 "/usr/include/bits/types.h" 3
extern "C" { typedef unsigned long __fsblkcnt_t; }
# 174 "/usr/include/bits/types.h" 3
extern "C" { typedef unsigned long __fsblkcnt64_t; }
# 177 "/usr/include/bits/types.h" 3
extern "C" { typedef unsigned long __fsfilcnt_t; }
# 178 "/usr/include/bits/types.h" 3
extern "C" { typedef unsigned long __fsfilcnt64_t; }
# 180 "/usr/include/bits/types.h" 3
extern "C" { typedef long __ssize_t; }
# 184 "/usr/include/bits/types.h" 3
extern "C" { typedef __off64_t __loff_t; }
# 185 "/usr/include/bits/types.h" 3
extern "C" { typedef __quad_t *__qaddr_t; }
# 186 "/usr/include/bits/types.h" 3
extern "C" { typedef char *__caddr_t; }
# 189 "/usr/include/bits/types.h" 3
extern "C" { typedef long __intptr_t; }
# 192 "/usr/include/bits/types.h" 3
extern "C" { typedef unsigned __socklen_t; }
# 61 "/usr/include/time.h" 3
extern "C" { typedef __clock_t clock_t; }
# 77 "/usr/include/time.h" 3
extern "C" { typedef __time_t time_t; }
# 93 "/usr/include/time.h" 3
extern "C" { typedef __clockid_t clockid_t; }
# 105 "/usr/include/time.h" 3
extern "C" { typedef __timer_t timer_t; }
# 121 "/usr/include/time.h" 3
extern "C" { struct timespec {
# 123 "/usr/include/time.h" 3
__time_t tv_sec;
# 124 "/usr/include/time.h" 3
long tv_nsec;
# 125 "/usr/include/time.h" 3
}; }
# 134 "/usr/include/time.h" 3
extern "C" { struct tm {
# 136 "/usr/include/time.h" 3
int tm_sec;
# 137 "/usr/include/time.h" 3
int tm_min;
# 138 "/usr/include/time.h" 3
int tm_hour;
# 139 "/usr/include/time.h" 3
int tm_mday;
# 140 "/usr/include/time.h" 3
int tm_mon;
# 141 "/usr/include/time.h" 3
int tm_year;
# 142 "/usr/include/time.h" 3
int tm_wday;
# 143 "/usr/include/time.h" 3
int tm_yday;
# 144 "/usr/include/time.h" 3
int tm_isdst;
# 147 "/usr/include/time.h" 3
long tm_gmtoff;
# 148 "/usr/include/time.h" 3
const char *tm_zone;
# 153 "/usr/include/time.h" 3
}; }
# 162 "/usr/include/time.h" 3
extern "C" { struct itimerspec {
# 164 "/usr/include/time.h" 3
timespec it_interval;
# 165 "/usr/include/time.h" 3
timespec it_value;
# 166 "/usr/include/time.h" 3
}; }
# 169 "/usr/include/time.h" 3
struct sigevent;
# 175 "/usr/include/time.h" 3
extern "C" { typedef __pid_t pid_t; }
# 184 "/usr/include/time.h" 3
extern "C" __attribute__((__weak__)) clock_t clock() throw();
# 187 "/usr/include/time.h" 3
extern "C" time_t time(time_t *) throw();
# 190 "/usr/include/time.h" 3
extern "C" double difftime(time_t, time_t) throw() __attribute__((__const__));
# 194 "/usr/include/time.h" 3
extern "C" time_t mktime(tm *) throw();
# 200 "/usr/include/time.h" 3
extern "C" size_t strftime(char *__restrict__, size_t, const char *__restrict__, const tm *__restrict__) throw();
# 208 "/usr/include/time.h" 3
extern "C" char *strptime(const char *__restrict__, const char *__restrict__, tm *) throw();
# 40 "/usr/include/xlocale.h" 3
extern "C" { typedef
# 28 "/usr/include/xlocale.h" 3
struct __locale_struct {
# 31 "/usr/include/xlocale.h" 3
struct locale_data *__locales[13];
# 34 "/usr/include/xlocale.h" 3
const unsigned short *__ctype_b;
# 35 "/usr/include/xlocale.h" 3
const int *__ctype_tolower;
# 36 "/usr/include/xlocale.h" 3
const int *__ctype_toupper;
# 39 "/usr/include/xlocale.h" 3
const char *__names[13];
# 40 "/usr/include/xlocale.h" 3
} *__locale_t; }
# 218 "/usr/include/time.h" 3
extern "C" size_t strftime_l(char *__restrict__, size_t, const char *__restrict__, const tm *__restrict__, __locale_t) throw();
# 223 "/usr/include/time.h" 3
extern "C" char *strptime_l(const char *__restrict__, const char *__restrict__, tm *, __locale_t) throw();
# 232 "/usr/include/time.h" 3
extern "C" tm *gmtime(const time_t *) throw();
# 236 "/usr/include/time.h" 3
extern "C" tm *localtime(const time_t *) throw();
# 242 "/usr/include/time.h" 3
extern "C" tm *gmtime_r(const time_t *__restrict__, tm *__restrict__) throw();
# 247 "/usr/include/time.h" 3
extern "C" tm *localtime_r(const time_t *__restrict__, tm *__restrict__) throw();
# 254 "/usr/include/time.h" 3
extern "C" char *asctime(const tm *) throw();
# 257 "/usr/include/time.h" 3
extern "C" char *ctime(const time_t *) throw();
# 265 "/usr/include/time.h" 3
extern "C" char *asctime_r(const tm *__restrict__, char *__restrict__) throw();
# 269 "/usr/include/time.h" 3
extern "C" char *ctime_r(const time_t *__restrict__, char *__restrict__) throw();
# 275 "/usr/include/time.h" 3
extern "C" { extern char *__tzname[2]; }
# 276 "/usr/include/time.h" 3
extern "C" { extern int __daylight; }
# 277 "/usr/include/time.h" 3
extern "C" { extern long __timezone; }
# 282 "/usr/include/time.h" 3
extern "C" { extern char *tzname[2]; }
# 286 "/usr/include/time.h" 3
extern "C" void tzset() throw();
# 290 "/usr/include/time.h" 3
extern "C" { extern int daylight; }
# 291 "/usr/include/time.h" 3
extern "C" { extern long timezone; }
# 297 "/usr/include/time.h" 3
extern "C" int stime(const time_t *) throw();
# 312 "/usr/include/time.h" 3
extern "C" time_t timegm(tm *) throw();
# 315 "/usr/include/time.h" 3
extern "C" time_t timelocal(tm *) throw();
# 318 "/usr/include/time.h" 3
extern "C" int dysize(int) throw() __attribute__((__const__));
# 327 "/usr/include/time.h" 3
extern "C" int nanosleep(const timespec *, timespec *);
# 332 "/usr/include/time.h" 3
extern "C" int clock_getres(clockid_t, timespec *) throw();
# 335 "/usr/include/time.h" 3
extern "C" int clock_gettime(clockid_t, timespec *) throw();
# 338 "/usr/include/time.h" 3
extern "C" int clock_settime(clockid_t, const timespec *) throw();
# 346 "/usr/include/time.h" 3
extern "C" int clock_nanosleep(clockid_t, int, const timespec *, timespec *);
# 351 "/usr/include/time.h" 3
extern "C" int clock_getcpuclockid(pid_t, clockid_t *) throw();
# 356 "/usr/include/time.h" 3
extern "C" int timer_create(clockid_t, sigevent *__restrict__, timer_t *__restrict__) throw();
# 361 "/usr/include/time.h" 3
extern "C" int timer_delete(timer_t) throw();
# 364 "/usr/include/time.h" 3
extern "C" int timer_settime(timer_t, int, const itimerspec *__restrict__, itimerspec *__restrict__) throw();
# 369 "/usr/include/time.h" 3
extern "C" int timer_gettime(timer_t, itimerspec *) throw();
# 373 "/usr/include/time.h" 3
extern "C" int timer_getoverrun(timer_t) throw();
# 389 "/usr/include/time.h" 3
extern "C" { extern int getdate_err; }
# 398 "/usr/include/time.h" 3
extern "C" tm *getdate(const char *);
# 412 "/usr/include/time.h" 3
extern "C" int getdate_r(const char *__restrict__, tm *__restrict__);
# 38 "/usr/include/string.h" 3
extern "C" __attribute__((__weak__)) void *memcpy(void *__restrict__, const void *__restrict__, size_t) throw();
# 43 "/usr/include/string.h" 3
extern "C" void *memmove(void *, const void *, size_t) throw();
# 51 "/usr/include/string.h" 3
extern "C" void *memccpy(void *__restrict__, const void *__restrict__, int, size_t) throw();
# 59 "/usr/include/string.h" 3
extern "C" __attribute__((__weak__)) void *memset(void *, int, size_t) throw();
# 62 "/usr/include/string.h" 3
extern "C" int memcmp(const void *, const void *, size_t) throw() __attribute__((__pure__));
# 66 "/usr/include/string.h" 3
extern "C" void *memchr(const void *, int, size_t) throw() __attribute__((__pure__));
# 73 "/usr/include/string.h" 3
extern "C" void *rawmemchr(const void *, int) throw() __attribute__((__pure__));
# 77 "/usr/include/string.h" 3
extern "C" void *memrchr(const void *, int, size_t) throw() __attribute__((__pure__));
# 84 "/usr/include/string.h" 3
extern "C" char *strcpy(char *__restrict__, const char *__restrict__) throw();
# 87 "/usr/include/string.h" 3
extern "C" char *strncpy(char *__restrict__, const char *__restrict__, size_t) throw();
# 92 "/usr/include/string.h" 3
extern "C" char *strcat(char *__restrict__, const char *__restrict__) throw();
# 95 "/usr/include/string.h" 3
extern "C" char *strncat(char *__restrict__, const char *__restrict__, size_t) throw();
# 99 "/usr/include/string.h" 3
extern "C" int strcmp(const char *, const char *) throw() __attribute__((__pure__));
# 102 "/usr/include/string.h" 3
extern "C" int strncmp(const char *, const char *, size_t) throw() __attribute__((__pure__));
# 106 "/usr/include/string.h" 3
extern "C" int strcoll(const char *, const char *) throw() __attribute__((__pure__));
# 109 "/usr/include/string.h" 3
extern "C" size_t strxfrm(char *__restrict__, const char *__restrict__, size_t) throw();
# 121 "/usr/include/string.h" 3
extern "C" int strcoll_l(const char *, const char *, __locale_t) throw() __attribute__((__pure__));
# 124 "/usr/include/string.h" 3
extern "C" size_t strxfrm_l(char *, const char *, size_t, __locale_t) throw();
# 130 "/usr/include/string.h" 3
extern "C" char *strdup(const char *) throw() __attribute__((__malloc__));
# 138 "/usr/include/string.h" 3
extern "C" char *strndup(const char *, size_t) throw() __attribute__((__malloc__));
# 167 "/usr/include/string.h" 3
extern "C" char *strchr(const char *, int) throw() __attribute__((__pure__));
# 170 "/usr/include/string.h" 3
extern "C" char *strrchr(const char *, int) throw() __attribute__((__pure__));
# 177 "/usr/include/string.h" 3
extern "C" char *strchrnul(const char *, int) throw() __attribute__((__pure__));
# 184 "/usr/include/string.h" 3
extern "C" size_t strcspn(const char *, const char *) throw() __attribute__((__pure__));
# 188 "/usr/include/string.h" 3
extern "C" size_t strspn(const char *, const char *) throw() __attribute__((__pure__));
# 191 "/usr/include/string.h" 3
extern "C" char *strpbrk(const char *, const char *) throw() __attribute__((__pure__));
# 194 "/usr/include/string.h" 3
extern "C" char *strstr(const char *, const char *) throw() __attribute__((__pure__));
# 199 "/usr/include/string.h" 3
extern "C" char *strtok(char *__restrict__, const char *__restrict__) throw();
# 205 "/usr/include/string.h" 3
extern "C" char *__strtok_r(char *__restrict__, const char *__restrict__, char **__restrict__) throw();
# 210 "/usr/include/string.h" 3
extern "C" char *strtok_r(char *__restrict__, const char *__restrict__, char **__restrict__) throw();
# 217 "/usr/include/string.h" 3
extern "C" char *strcasestr(const char *, const char *) throw() __attribute__((__pure__));
# 225 "/usr/include/string.h" 3
extern "C" void *memmem(const void *, size_t, const void *, size_t) throw() __attribute__((__pure__));
# 231 "/usr/include/string.h" 3
extern "C" void *__mempcpy(void *__restrict__, const void *__restrict__, size_t) throw();
# 234 "/usr/include/string.h" 3
extern "C" void *mempcpy(void *__restrict__, const void *__restrict__, size_t) throw();
# 242 "/usr/include/string.h" 3
extern "C" size_t strlen(const char *) throw() __attribute__((__pure__));
# 249 "/usr/include/string.h" 3
extern "C" size_t strnlen(const char *, size_t) throw() __attribute__((__pure__));
# 256 "/usr/include/string.h" 3
extern "C" char *strerror(int) throw();
# 281 "/usr/include/string.h" 3
extern "C" char *strerror_r(int, char *, size_t) throw();
# 288 "/usr/include/string.h" 3
extern "C" char *strerror_l(int, __locale_t) throw();
# 294 "/usr/include/string.h" 3
extern "C" void __bzero(void *, size_t) throw();
# 298 "/usr/include/string.h" 3
extern "C" void bcopy(const void *, void *, size_t) throw();
# 302 "/usr/include/string.h" 3
extern "C" void bzero(void *, size_t) throw();
# 305 "/usr/include/string.h" 3
extern "C" int bcmp(const void *, const void *, size_t) throw() __attribute__((__pure__));
# 309 "/usr/include/string.h" 3
extern "C" char *index(const char *, int) throw() __attribute__((__pure__));
# 313 "/usr/include/string.h" 3
extern "C" char *rindex(const char *, int) throw() __attribute__((__pure__));
# 318 "/usr/include/string.h" 3
extern "C" int ffs(int) throw() __attribute__((__const__));
# 323 "/usr/include/string.h" 3
extern "C" int ffsl(long) throw() __attribute__((__const__));
# 325 "/usr/include/string.h" 3
extern "C" int ffsll(long long) throw() __attribute__((__const__));
# 331 "/usr/include/string.h" 3
extern "C" int strcasecmp(const char *, const char *) throw() __attribute__((__pure__));
# 335 "/usr/include/string.h" 3
extern "C" int strncasecmp(const char *, const char *, size_t) throw() __attribute__((__pure__));
# 342 "/usr/include/string.h" 3
extern "C" int strcasecmp_l(const char *, const char *, __locale_t) throw() __attribute__((__pure__));
# 346 "/usr/include/string.h" 3
extern "C" int strncasecmp_l(const char *, const char *, size_t, __locale_t) throw() __attribute__((__pure__));
# 354 "/usr/include/string.h" 3
extern "C" char *strsep(char **__restrict__, const char *__restrict__) throw();
# 361 "/usr/include/string.h" 3
extern "C" int strverscmp(const char *, const char *) throw() __attribute__((__pure__));
# 365 "/usr/include/string.h" 3
extern "C" char *strsignal(int) throw();
# 368 "/usr/include/string.h" 3
extern "C" char *__stpcpy(char *__restrict__, const char *__restrict__) throw();
# 370 "/usr/include/string.h" 3
extern "C" char *stpcpy(char *__restrict__, const char *__restrict__) throw();
# 375 "/usr/include/string.h" 3
extern "C" char *__stpncpy(char *__restrict__, const char *__restrict__, size_t) throw();
# 378 "/usr/include/string.h" 3
extern "C" char *stpncpy(char *__restrict__, const char *__restrict__, size_t) throw();
# 383 "/usr/include/string.h" 3
extern "C" char *strfry(char *) throw();
# 386 "/usr/include/string.h" 3
extern "C" void *memfrob(void *, size_t) throw();
# 393 "/usr/include/string.h" 3
extern "C" char *basename(const char *) throw();
# 56 "/usr/local/cuda/bin/../include/common_functions.h"
extern "C" __attribute__((__weak__)) clock_t clock() throw();
# 59 "/usr/local/cuda/bin/../include/common_functions.h"
extern "C" __attribute__((__weak__)) void *memset(void *, int, size_t) throw();
# 62 "/usr/local/cuda/bin/../include/common_functions.h"
extern "C" __attribute__((__weak__)) void *memcpy(void *, const void *, size_t) throw();
# 65 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) int abs(int) throw() __attribute__((__const__));
# 67 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) long labs(long) throw() __attribute__((__const__));
# 69 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) long long llabs(long long) throw() __attribute__((__const__));
# 71 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) double fabs(double) throw() __attribute__((__const__));
# 73 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) float fabsf(float) throw() __attribute__((__const__));
# 76 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) int min(int, int);
# 78 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) unsigned umin(unsigned, unsigned);
# 80 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) float fminf(float, float) throw();
# 82 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) double fmin(double, double) throw();
# 85 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) int max(int, int);
# 87 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) unsigned umax(unsigned, unsigned);
# 89 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) float fmaxf(float, float) throw();
# 91 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) double fmax(double, double) throw();
# 94 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) double sin(double) throw();
# 96 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) float sinf(float) throw();
# 99 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) double cos(double) throw();
# 101 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) float cosf(float) throw();
# 104 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) void sincos(double, double *, double *) throw();
# 106 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) void sincosf(float, float *, float *) throw();
# 109 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) double tan(double) throw();
# 111 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) float tanf(float) throw();
# 114 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) double sqrt(double) throw();
# 116 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) float sqrtf(float) throw();
# 119 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) double rsqrt(double);
# 121 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) float rsqrtf(float);
# 124 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) double exp2(double) throw();
# 126 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) float exp2f(float) throw();
# 129 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) double exp10(double) throw();
# 131 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) float exp10f(float) throw();
# 134 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) double expm1(double) throw();
# 136 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) float expm1f(float) throw();
# 139 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) double log2(double) throw();
# 141 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) float log2f(float) throw();
# 144 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) double log10(double) throw();
# 146 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) float log10f(float) throw();
# 149 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) double log(double) throw();
# 151 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) float logf(float) throw();
# 154 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) double log1p(double) throw();
# 156 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) float log1pf(float) throw();
# 159 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) double floor(double) throw() __attribute__((__const__));
# 161 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) float floorf(float) throw() __attribute__((__const__));
# 164 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) double exp(double) throw();
# 166 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) float expf(float) throw();
# 169 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) double cosh(double) throw();
# 171 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) float coshf(float) throw();
# 174 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) double sinh(double) throw();
# 176 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) float sinhf(float) throw();
# 179 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) double tanh(double) throw();
# 181 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) float tanhf(float) throw();
# 184 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) double acosh(double) throw();
# 186 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) float acoshf(float) throw();
# 189 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) double asinh(double) throw();
# 191 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) float asinhf(float) throw();
# 194 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) double atanh(double) throw();
# 196 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) float atanhf(float) throw();
# 199 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) double ldexp(double, int) throw();
# 201 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) float ldexpf(float, int) throw();
# 204 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) double logb(double) throw();
# 206 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) float logbf(float) throw();
# 209 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) int ilogb(double) throw();
# 211 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) int ilogbf(float) throw();
# 214 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) double scalbn(double, int) throw();
# 216 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) float scalbnf(float, int) throw();
# 219 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) double scalbln(double, long) throw();
# 221 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) float scalblnf(float, long) throw();
# 224 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) double frexp(double, int *) throw();
# 226 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) float frexpf(float, int *) throw();
# 229 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) double round(double) throw() __attribute__((__const__));
# 231 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) float roundf(float) throw() __attribute__((__const__));
# 234 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) long lround(double) throw();
# 236 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) long lroundf(float) throw();
# 239 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) long long llround(double) throw();
# 241 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) long long llroundf(float) throw();
# 244 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) double rint(double) throw();
# 246 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) float rintf(float) throw();
# 249 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) long lrint(double) throw();
# 251 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) long lrintf(float) throw();
# 254 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) long long llrint(double) throw();
# 256 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) long long llrintf(float) throw();
# 259 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) double nearbyint(double) throw();
# 261 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) float nearbyintf(float) throw();
# 264 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) double ceil(double) throw() __attribute__((__const__));
# 266 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) float ceilf(float) throw() __attribute__((__const__));
# 269 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) double trunc(double) throw() __attribute__((__const__));
# 271 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) float truncf(float) throw() __attribute__((__const__));
# 274 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) double fdim(double, double) throw();
# 276 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) float fdimf(float, float) throw();
# 279 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) double atan2(double, double) throw();
# 281 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) float atan2f(float, float) throw();
# 284 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) double atan(double) throw();
# 286 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) float atanf(float) throw();
# 289 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) double asin(double) throw();
# 291 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) float asinf(float) throw();
# 294 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) double acos(double) throw();
# 296 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) float acosf(float) throw();
# 299 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) double hypot(double, double) throw();
# 301 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) float hypotf(float, float) throw();
# 304 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) double cbrt(double) throw();
# 306 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) float cbrtf(float) throw();
# 309 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) double pow(double, double) throw();
# 311 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) float powf(float, float) throw();
# 314 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) double modf(double, double *) throw();
# 316 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) float modff(float, float *) throw();
# 319 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) double fmod(double, double) throw();
# 321 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) float fmodf(float, float) throw();
# 324 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) double remainder(double, double) throw();
# 326 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) float remainderf(float, float) throw();
# 329 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) double remquo(double, double, int *) throw();
# 331 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) float remquof(float, float, int *) throw();
# 334 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) double erf(double) throw();
# 336 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) float erff(float) throw();
# 339 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) double erfc(double) throw();
# 341 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) float erfcf(float) throw();
# 344 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) double lgamma(double) throw();
# 346 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) float lgammaf(float) throw();
# 349 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) double tgamma(double) throw();
# 351 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) float tgammaf(float) throw();
# 354 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) double copysign(double, double) throw() __attribute__((__const__));
# 356 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) float copysignf(float, float) throw() __attribute__((__const__));
# 359 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) double nextafter(double, double) throw() __attribute__((__const__));
# 361 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) float nextafterf(float, float) throw() __attribute__((__const__));
# 364 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) double nan(const char *) throw() __attribute__((__const__));
# 366 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) float nanf(const char *) throw() __attribute__((__const__));
# 369 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) int __isinf(double) throw() __attribute__((__const__));
# 371 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) int __isinff(float) throw() __attribute__((__const__));
# 374 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) int __isnan(double) throw() __attribute__((__const__));
# 376 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) int __isnanf(float) throw() __attribute__((__const__));
# 390 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) int __finite(double) throw() __attribute__((__const__));
# 392 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) int __finitef(float) throw() __attribute__((__const__));
# 394 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) int __signbit(double) throw() __attribute__((__const__));
# 399 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) int __signbitf(float) throw() __attribute__((__const__));
# 402 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) double fma(double, double, double) throw();
# 404 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) float fmaf(float, float, float) throw();
# 31 "/usr/include/bits/mathdef.h" 3
extern "C" { typedef float float_t; }
# 32 "/usr/include/bits/mathdef.h" 3
extern "C" { typedef double double_t; }
# 55 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) double acos(double) throw(); extern "C" double __acos(double) throw();
# 57 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) double asin(double) throw(); extern "C" double __asin(double) throw();
# 59 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) double atan(double) throw(); extern "C" double __atan(double) throw();
# 61 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) double atan2(double, double) throw(); extern "C" double __atan2(double, double) throw();
# 64 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) double cos(double) throw(); extern "C" double __cos(double) throw();
# 66 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) double sin(double) throw(); extern "C" double __sin(double) throw();
# 68 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) double tan(double) throw(); extern "C" double __tan(double) throw();
# 73 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) double cosh(double) throw(); extern "C" double __cosh(double) throw();
# 75 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) double sinh(double) throw(); extern "C" double __sinh(double) throw();
# 77 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) double tanh(double) throw(); extern "C" double __tanh(double) throw();
# 82 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) void sincos(double, double *, double *) throw(); extern "C" void __sincos(double, double *, double *) throw();
# 89 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) double acosh(double) throw(); extern "C" double __acosh(double) throw();
# 91 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) double asinh(double) throw(); extern "C" double __asinh(double) throw();
# 93 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) double atanh(double) throw(); extern "C" double __atanh(double) throw();
# 101 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) double exp(double) throw(); extern "C" double __exp(double) throw();
# 104 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) double frexp(double, int *) throw(); extern "C" double __frexp(double, int *) throw();
# 107 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) double ldexp(double, int) throw(); extern "C" double __ldexp(double, int) throw();
# 110 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) double log(double) throw(); extern "C" double __log(double) throw();
# 113 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) double log10(double) throw(); extern "C" double __log10(double) throw();
# 116 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) double modf(double, double *) throw(); extern "C" double __modf(double, double *) throw();
# 121 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) double exp10(double) throw(); extern "C" double __exp10(double) throw();
# 123 "/usr/include/bits/mathcalls.h" 3
extern "C" double pow10(double) throw(); extern "C" double __pow10(double) throw();
# 129 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) double expm1(double) throw(); extern "C" double __expm1(double) throw();
# 132 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) double log1p(double) throw(); extern "C" double __log1p(double) throw();
# 135 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) double logb(double) throw(); extern "C" double __logb(double) throw();
# 142 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) double exp2(double) throw(); extern "C" double __exp2(double) throw();
# 145 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) double log2(double) throw(); extern "C" double __log2(double) throw();
# 154 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) double pow(double, double) throw(); extern "C" double __pow(double, double) throw();
# 157 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) double sqrt(double) throw(); extern "C" double __sqrt(double) throw();
# 163 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) double hypot(double, double) throw(); extern "C" double __hypot(double, double) throw();
# 170 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) double cbrt(double) throw(); extern "C" double __cbrt(double) throw();
# 179 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) double ceil(double) throw() __attribute__((__const__)); extern "C" double __ceil(double) throw() __attribute__((__const__));
# 182 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) double fabs(double) throw() __attribute__((__const__)); extern "C" double __fabs(double) throw() __attribute__((__const__));
# 185 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) double floor(double) throw() __attribute__((__const__)); extern "C" double __floor(double) throw() __attribute__((__const__));
# 188 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) double fmod(double, double) throw(); extern "C" double __fmod(double, double) throw();
# 193 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) int __isinf(double) throw() __attribute__((__const__));
# 196 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) int __finite(double) throw() __attribute__((__const__));
# 202 "/usr/include/bits/mathcalls.h" 3
extern "C" int isinf(double) throw() __attribute__((__const__));
# 205 "/usr/include/bits/mathcalls.h" 3
extern "C" int finite(double) throw() __attribute__((__const__));
# 208 "/usr/include/bits/mathcalls.h" 3
extern "C" double drem(double, double) throw(); extern "C" double __drem(double, double) throw();
# 212 "/usr/include/bits/mathcalls.h" 3
extern "C" double significand(double) throw(); extern "C" double __significand(double) throw();
# 218 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) double copysign(double, double) throw() __attribute__((__const__)); extern "C" double __copysign(double, double) throw() __attribute__((__const__));
# 225 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) double nan(const char *) throw() __attribute__((__const__)); extern "C" double __nan(const char *) throw() __attribute__((__const__));
# 231 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) int __isnan(double) throw() __attribute__((__const__));
# 235 "/usr/include/bits/mathcalls.h" 3
extern "C" int isnan(double) throw() __attribute__((__const__));
# 238 "/usr/include/bits/mathcalls.h" 3
extern "C" double j0(double) throw(); extern "C" double __j0(double) throw();
# 239 "/usr/include/bits/mathcalls.h" 3
extern "C" double j1(double) throw(); extern "C" double __j1(double) throw();
# 240 "/usr/include/bits/mathcalls.h" 3
extern "C" double jn(int, double) throw(); extern "C" double __jn(int, double) throw();
# 241 "/usr/include/bits/mathcalls.h" 3
extern "C" double y0(double) throw(); extern "C" double __y0(double) throw();
# 242 "/usr/include/bits/mathcalls.h" 3
extern "C" double y1(double) throw(); extern "C" double __y1(double) throw();
# 243 "/usr/include/bits/mathcalls.h" 3
extern "C" double yn(int, double) throw(); extern "C" double __yn(int, double) throw();
# 250 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) double erf(double) throw(); extern "C" double __erf(double) throw();
# 251 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) double erfc(double) throw(); extern "C" double __erfc(double) throw();
# 252 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) double lgamma(double) throw(); extern "C" double __lgamma(double) throw();
# 259 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) double tgamma(double) throw(); extern "C" double __tgamma(double) throw();
# 265 "/usr/include/bits/mathcalls.h" 3
extern "C" double gamma(double) throw(); extern "C" double __gamma(double) throw();
# 272 "/usr/include/bits/mathcalls.h" 3
extern "C" double lgamma_r(double, int *) throw(); extern "C" double __lgamma_r(double, int *) throw();
# 280 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) double rint(double) throw(); extern "C" double __rint(double) throw();
# 283 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) double nextafter(double, double) throw() __attribute__((__const__)); extern "C" double __nextafter(double, double) throw() __attribute__((__const__));
# 285 "/usr/include/bits/mathcalls.h" 3
extern "C" double nexttoward(double, long double) throw() __attribute__((__const__)); extern "C" double __nexttoward(double, long double) throw() __attribute__((__const__));
# 289 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) double remainder(double, double) throw(); extern "C" double __remainder(double, double) throw();
# 293 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) double scalbn(double, int) throw(); extern "C" double __scalbn(double, int) throw();
# 297 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) int ilogb(double) throw(); extern "C" int __ilogb(double) throw();
# 302 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) double scalbln(double, long) throw(); extern "C" double __scalbln(double, long) throw();
# 306 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) double nearbyint(double) throw(); extern "C" double __nearbyint(double) throw();
# 310 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) double round(double) throw() __attribute__((__const__)); extern "C" double __round(double) throw() __attribute__((__const__));
# 314 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) double trunc(double) throw() __attribute__((__const__)); extern "C" double __trunc(double) throw() __attribute__((__const__));
# 319 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) double remquo(double, double, int *) throw(); extern "C" double __remquo(double, double, int *) throw();
# 326 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) long lrint(double) throw(); extern "C" long __lrint(double) throw();
# 327 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) long long llrint(double) throw(); extern "C" long long __llrint(double) throw();
# 331 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) long lround(double) throw(); extern "C" long __lround(double) throw();
# 332 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) long long llround(double) throw(); extern "C" long long __llround(double) throw();
# 336 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) double fdim(double, double) throw(); extern "C" double __fdim(double, double) throw();
# 339 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) double fmax(double, double) throw(); extern "C" double __fmax(double, double) throw();
# 342 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) double fmin(double, double) throw(); extern "C" double __fmin(double, double) throw();
# 346 "/usr/include/bits/mathcalls.h" 3
extern "C" int __fpclassify(double) throw() __attribute__((__const__));
# 350 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) int __signbit(double) throw() __attribute__((__const__));
# 355 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) double fma(double, double, double) throw(); extern "C" double __fma(double, double, double) throw();
# 364 "/usr/include/bits/mathcalls.h" 3
extern "C" double scalb(double, double) throw(); extern "C" double __scalb(double, double) throw();
# 55 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) float acosf(float) throw(); extern "C" float __acosf(float) throw();
# 57 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) float asinf(float) throw(); extern "C" float __asinf(float) throw();
# 59 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) float atanf(float) throw(); extern "C" float __atanf(float) throw();
# 61 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) float atan2f(float, float) throw(); extern "C" float __atan2f(float, float) throw();
# 64 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) float cosf(float) throw();
# 66 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) float sinf(float) throw();
# 68 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) float tanf(float) throw();
# 73 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) float coshf(float) throw(); extern "C" float __coshf(float) throw();
# 75 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) float sinhf(float) throw(); extern "C" float __sinhf(float) throw();
# 77 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) float tanhf(float) throw(); extern "C" float __tanhf(float) throw();
# 82 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) void sincosf(float, float *, float *) throw();
# 89 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) float acoshf(float) throw(); extern "C" float __acoshf(float) throw();
# 91 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) float asinhf(float) throw(); extern "C" float __asinhf(float) throw();
# 93 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) float atanhf(float) throw(); extern "C" float __atanhf(float) throw();
# 101 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) float expf(float) throw();
# 104 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) float frexpf(float, int *) throw(); extern "C" float __frexpf(float, int *) throw();
# 107 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) float ldexpf(float, int) throw(); extern "C" float __ldexpf(float, int) throw();
# 110 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) float logf(float) throw();
# 113 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) float log10f(float) throw();
# 116 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) float modff(float, float *) throw(); extern "C" float __modff(float, float *) throw();
# 121 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) float exp10f(float) throw();
# 123 "/usr/include/bits/mathcalls.h" 3
extern "C" float pow10f(float) throw(); extern "C" float __pow10f(float) throw();
# 129 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) float expm1f(float) throw(); extern "C" float __expm1f(float) throw();
# 132 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) float log1pf(float) throw(); extern "C" float __log1pf(float) throw();
# 135 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) float logbf(float) throw(); extern "C" float __logbf(float) throw();
# 142 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) float exp2f(float) throw(); extern "C" float __exp2f(float) throw();
# 145 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) float log2f(float) throw();
# 154 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) float powf(float, float) throw();
# 157 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) float sqrtf(float) throw(); extern "C" float __sqrtf(float) throw();
# 163 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) float hypotf(float, float) throw(); extern "C" float __hypotf(float, float) throw();
# 170 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) float cbrtf(float) throw(); extern "C" float __cbrtf(float) throw();
# 179 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) float ceilf(float) throw() __attribute__((__const__)); extern "C" float __ceilf(float) throw() __attribute__((__const__));
# 182 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) float fabsf(float) throw() __attribute__((__const__)); extern "C" float __fabsf(float) throw() __attribute__((__const__));
# 185 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) float floorf(float) throw() __attribute__((__const__)); extern "C" float __floorf(float) throw() __attribute__((__const__));
# 188 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) float fmodf(float, float) throw(); extern "C" float __fmodf(float, float) throw();
# 193 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) int __isinff(float) throw() __attribute__((__const__));
# 196 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) int __finitef(float) throw() __attribute__((__const__));
# 202 "/usr/include/bits/mathcalls.h" 3
extern "C" int isinff(float) throw() __attribute__((__const__));
# 205 "/usr/include/bits/mathcalls.h" 3
extern "C" int finitef(float) throw() __attribute__((__const__));
# 208 "/usr/include/bits/mathcalls.h" 3
extern "C" float dremf(float, float) throw(); extern "C" float __dremf(float, float) throw();
# 212 "/usr/include/bits/mathcalls.h" 3
extern "C" float significandf(float) throw(); extern "C" float __significandf(float) throw();
# 218 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) float copysignf(float, float) throw() __attribute__((__const__)); extern "C" float __copysignf(float, float) throw() __attribute__((__const__));
# 225 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) float nanf(const char *) throw() __attribute__((__const__)); extern "C" float __nanf(const char *) throw() __attribute__((__const__));
# 231 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) int __isnanf(float) throw() __attribute__((__const__));
# 235 "/usr/include/bits/mathcalls.h" 3
extern "C" int isnanf(float) throw() __attribute__((__const__));
# 238 "/usr/include/bits/mathcalls.h" 3
extern "C" float j0f(float) throw(); extern "C" float __j0f(float) throw();
# 239 "/usr/include/bits/mathcalls.h" 3
extern "C" float j1f(float) throw(); extern "C" float __j1f(float) throw();
# 240 "/usr/include/bits/mathcalls.h" 3
extern "C" float jnf(int, float) throw(); extern "C" float __jnf(int, float) throw();
# 241 "/usr/include/bits/mathcalls.h" 3
extern "C" float y0f(float) throw(); extern "C" float __y0f(float) throw();
# 242 "/usr/include/bits/mathcalls.h" 3
extern "C" float y1f(float) throw(); extern "C" float __y1f(float) throw();
# 243 "/usr/include/bits/mathcalls.h" 3
extern "C" float ynf(int, float) throw(); extern "C" float __ynf(int, float) throw();
# 250 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) float erff(float) throw(); extern "C" float __erff(float) throw();
# 251 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) float erfcf(float) throw(); extern "C" float __erfcf(float) throw();
# 252 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) float lgammaf(float) throw(); extern "C" float __lgammaf(float) throw();
# 259 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) float tgammaf(float) throw(); extern "C" float __tgammaf(float) throw();
# 265 "/usr/include/bits/mathcalls.h" 3
extern "C" float gammaf(float) throw(); extern "C" float __gammaf(float) throw();
# 272 "/usr/include/bits/mathcalls.h" 3
extern "C" float lgammaf_r(float, int *) throw(); extern "C" float __lgammaf_r(float, int *) throw();
# 280 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) float rintf(float) throw(); extern "C" float __rintf(float) throw();
# 283 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) float nextafterf(float, float) throw() __attribute__((__const__)); extern "C" float __nextafterf(float, float) throw() __attribute__((__const__));
# 285 "/usr/include/bits/mathcalls.h" 3
extern "C" float nexttowardf(float, long double) throw() __attribute__((__const__)); extern "C" float __nexttowardf(float, long double) throw() __attribute__((__const__));
# 289 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) float remainderf(float, float) throw(); extern "C" float __remainderf(float, float) throw();
# 293 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) float scalbnf(float, int) throw(); extern "C" float __scalbnf(float, int) throw();
# 297 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) int ilogbf(float) throw(); extern "C" int __ilogbf(float) throw();
# 302 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) float scalblnf(float, long) throw(); extern "C" float __scalblnf(float, long) throw();
# 306 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) float nearbyintf(float) throw(); extern "C" float __nearbyintf(float) throw();
# 310 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) float roundf(float) throw() __attribute__((__const__)); extern "C" float __roundf(float) throw() __attribute__((__const__));
# 314 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) float truncf(float) throw() __attribute__((__const__)); extern "C" float __truncf(float) throw() __attribute__((__const__));
# 319 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) float remquof(float, float, int *) throw(); extern "C" float __remquof(float, float, int *) throw();
# 326 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) long lrintf(float) throw(); extern "C" long __lrintf(float) throw();
# 327 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) long long llrintf(float) throw(); extern "C" long long __llrintf(float) throw();
# 331 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) long lroundf(float) throw(); extern "C" long __lroundf(float) throw();
# 332 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) long long llroundf(float) throw(); extern "C" long long __llroundf(float) throw();
# 336 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) float fdimf(float, float) throw(); extern "C" float __fdimf(float, float) throw();
# 339 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) float fmaxf(float, float) throw(); extern "C" float __fmaxf(float, float) throw();
# 342 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) float fminf(float, float) throw(); extern "C" float __fminf(float, float) throw();
# 346 "/usr/include/bits/mathcalls.h" 3
extern "C" int __fpclassifyf(float) throw() __attribute__((__const__));
# 350 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) int __signbitf(float) throw() __attribute__((__const__));
# 355 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) float fmaf(float, float, float) throw(); extern "C" float __fmaf(float, float, float) throw();
# 364 "/usr/include/bits/mathcalls.h" 3
extern "C" float scalbf(float, float) throw(); extern "C" float __scalbf(float, float) throw();
# 55 "/usr/include/bits/mathcalls.h" 3
extern "C" long double acosl(long double) throw(); extern "C" long double __acosl(long double) throw();
# 57 "/usr/include/bits/mathcalls.h" 3
extern "C" long double asinl(long double) throw(); extern "C" long double __asinl(long double) throw();
# 59 "/usr/include/bits/mathcalls.h" 3
extern "C" long double atanl(long double) throw(); extern "C" long double __atanl(long double) throw();
# 61 "/usr/include/bits/mathcalls.h" 3
extern "C" long double atan2l(long double, long double) throw(); extern "C" long double __atan2l(long double, long double) throw();
# 64 "/usr/include/bits/mathcalls.h" 3
extern "C" long double cosl(long double) throw(); extern "C" long double __cosl(long double) throw();
# 66 "/usr/include/bits/mathcalls.h" 3
extern "C" long double sinl(long double) throw(); extern "C" long double __sinl(long double) throw();
# 68 "/usr/include/bits/mathcalls.h" 3
extern "C" long double tanl(long double) throw(); extern "C" long double __tanl(long double) throw();
# 73 "/usr/include/bits/mathcalls.h" 3
extern "C" long double coshl(long double) throw(); extern "C" long double __coshl(long double) throw();
# 75 "/usr/include/bits/mathcalls.h" 3
extern "C" long double sinhl(long double) throw(); extern "C" long double __sinhl(long double) throw();
# 77 "/usr/include/bits/mathcalls.h" 3
extern "C" long double tanhl(long double) throw(); extern "C" long double __tanhl(long double) throw();
# 82 "/usr/include/bits/mathcalls.h" 3
extern "C" void sincosl(long double, long double *, long double *) throw(); extern "C" void __sincosl(long double, long double *, long double *) throw();
# 89 "/usr/include/bits/mathcalls.h" 3
extern "C" long double acoshl(long double) throw(); extern "C" long double __acoshl(long double) throw();
# 91 "/usr/include/bits/mathcalls.h" 3
extern "C" long double asinhl(long double) throw(); extern "C" long double __asinhl(long double) throw();
# 93 "/usr/include/bits/mathcalls.h" 3
extern "C" long double atanhl(long double) throw(); extern "C" long double __atanhl(long double) throw();
# 101 "/usr/include/bits/mathcalls.h" 3
extern "C" long double expl(long double) throw(); extern "C" long double __expl(long double) throw();
# 104 "/usr/include/bits/mathcalls.h" 3
extern "C" long double frexpl(long double, int *) throw(); extern "C" long double __frexpl(long double, int *) throw();
# 107 "/usr/include/bits/mathcalls.h" 3
extern "C" long double ldexpl(long double, int) throw(); extern "C" long double __ldexpl(long double, int) throw();
# 110 "/usr/include/bits/mathcalls.h" 3
extern "C" long double logl(long double) throw(); extern "C" long double __logl(long double) throw();
# 113 "/usr/include/bits/mathcalls.h" 3
extern "C" long double log10l(long double) throw(); extern "C" long double __log10l(long double) throw();
# 116 "/usr/include/bits/mathcalls.h" 3
extern "C" long double modfl(long double, long double *) throw(); extern "C" long double __modfl(long double, long double *) throw();
# 121 "/usr/include/bits/mathcalls.h" 3
extern "C" long double exp10l(long double) throw(); extern "C" long double __exp10l(long double) throw();
# 123 "/usr/include/bits/mathcalls.h" 3
extern "C" long double pow10l(long double) throw(); extern "C" long double __pow10l(long double) throw();
# 129 "/usr/include/bits/mathcalls.h" 3
extern "C" long double expm1l(long double) throw(); extern "C" long double __expm1l(long double) throw();
# 132 "/usr/include/bits/mathcalls.h" 3
extern "C" long double log1pl(long double) throw(); extern "C" long double __log1pl(long double) throw();
# 135 "/usr/include/bits/mathcalls.h" 3
extern "C" long double logbl(long double) throw(); extern "C" long double __logbl(long double) throw();
# 142 "/usr/include/bits/mathcalls.h" 3
extern "C" long double exp2l(long double) throw(); extern "C" long double __exp2l(long double) throw();
# 145 "/usr/include/bits/mathcalls.h" 3
extern "C" long double log2l(long double) throw(); extern "C" long double __log2l(long double) throw();
# 154 "/usr/include/bits/mathcalls.h" 3
extern "C" long double powl(long double, long double) throw(); extern "C" long double __powl(long double, long double) throw();
# 157 "/usr/include/bits/mathcalls.h" 3
extern "C" long double sqrtl(long double) throw(); extern "C" long double __sqrtl(long double) throw();
# 163 "/usr/include/bits/mathcalls.h" 3
extern "C" long double hypotl(long double, long double) throw(); extern "C" long double __hypotl(long double, long double) throw();
# 170 "/usr/include/bits/mathcalls.h" 3
extern "C" long double cbrtl(long double) throw(); extern "C" long double __cbrtl(long double) throw();
# 179 "/usr/include/bits/mathcalls.h" 3
extern "C" long double ceill(long double) throw() __attribute__((__const__)); extern "C" long double __ceill(long double) throw() __attribute__((__const__));
# 182 "/usr/include/bits/mathcalls.h" 3
extern "C" long double fabsl(long double) throw() __attribute__((__const__)); extern "C" long double __fabsl(long double) throw() __attribute__((__const__));
# 185 "/usr/include/bits/mathcalls.h" 3
extern "C" long double floorl(long double) throw() __attribute__((__const__)); extern "C" long double __floorl(long double) throw() __attribute__((__const__));
# 188 "/usr/include/bits/mathcalls.h" 3
extern "C" long double fmodl(long double, long double) throw(); extern "C" long double __fmodl(long double, long double) throw();
# 193 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) int __isinfl(long double) throw() __attribute__((__const__));
# 196 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) int __finitel(long double) throw() __attribute__((__const__));
# 202 "/usr/include/bits/mathcalls.h" 3
extern "C" int isinfl(long double) throw() __attribute__((__const__));
# 205 "/usr/include/bits/mathcalls.h" 3
extern "C" int finitel(long double) throw() __attribute__((__const__));
# 208 "/usr/include/bits/mathcalls.h" 3
extern "C" long double dreml(long double, long double) throw(); extern "C" long double __dreml(long double, long double) throw();
# 212 "/usr/include/bits/mathcalls.h" 3
extern "C" long double significandl(long double) throw(); extern "C" long double __significandl(long double) throw();
# 218 "/usr/include/bits/mathcalls.h" 3
extern "C" long double copysignl(long double, long double) throw() __attribute__((__const__)); extern "C" long double __copysignl(long double, long double) throw() __attribute__((__const__));
# 225 "/usr/include/bits/mathcalls.h" 3
extern "C" long double nanl(const char *) throw() __attribute__((__const__)); extern "C" long double __nanl(const char *) throw() __attribute__((__const__));
# 231 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) int __isnanl(long double) throw() __attribute__((__const__));
# 235 "/usr/include/bits/mathcalls.h" 3
extern "C" int isnanl(long double) throw() __attribute__((__const__));
# 238 "/usr/include/bits/mathcalls.h" 3
extern "C" long double j0l(long double) throw(); extern "C" long double __j0l(long double) throw();
# 239 "/usr/include/bits/mathcalls.h" 3
extern "C" long double j1l(long double) throw(); extern "C" long double __j1l(long double) throw();
# 240 "/usr/include/bits/mathcalls.h" 3
extern "C" long double jnl(int, long double) throw(); extern "C" long double __jnl(int, long double) throw();
# 241 "/usr/include/bits/mathcalls.h" 3
extern "C" long double y0l(long double) throw(); extern "C" long double __y0l(long double) throw();
# 242 "/usr/include/bits/mathcalls.h" 3
extern "C" long double y1l(long double) throw(); extern "C" long double __y1l(long double) throw();
# 243 "/usr/include/bits/mathcalls.h" 3
extern "C" long double ynl(int, long double) throw(); extern "C" long double __ynl(int, long double) throw();
# 250 "/usr/include/bits/mathcalls.h" 3
extern "C" long double erfl(long double) throw(); extern "C" long double __erfl(long double) throw();
# 251 "/usr/include/bits/mathcalls.h" 3
extern "C" long double erfcl(long double) throw(); extern "C" long double __erfcl(long double) throw();
# 252 "/usr/include/bits/mathcalls.h" 3
extern "C" long double lgammal(long double) throw(); extern "C" long double __lgammal(long double) throw();
# 259 "/usr/include/bits/mathcalls.h" 3
extern "C" long double tgammal(long double) throw(); extern "C" long double __tgammal(long double) throw();
# 265 "/usr/include/bits/mathcalls.h" 3
extern "C" long double gammal(long double) throw(); extern "C" long double __gammal(long double) throw();
# 272 "/usr/include/bits/mathcalls.h" 3
extern "C" long double lgammal_r(long double, int *) throw(); extern "C" long double __lgammal_r(long double, int *) throw();
# 280 "/usr/include/bits/mathcalls.h" 3
extern "C" long double rintl(long double) throw(); extern "C" long double __rintl(long double) throw();
# 283 "/usr/include/bits/mathcalls.h" 3
extern "C" long double nextafterl(long double, long double) throw() __attribute__((__const__)); extern "C" long double __nextafterl(long double, long double) throw() __attribute__((__const__));
# 285 "/usr/include/bits/mathcalls.h" 3
extern "C" long double nexttowardl(long double, long double) throw() __attribute__((__const__)); extern "C" long double __nexttowardl(long double, long double) throw() __attribute__((__const__));
# 289 "/usr/include/bits/mathcalls.h" 3
extern "C" long double remainderl(long double, long double) throw(); extern "C" long double __remainderl(long double, long double) throw();
# 293 "/usr/include/bits/mathcalls.h" 3
extern "C" long double scalbnl(long double, int) throw(); extern "C" long double __scalbnl(long double, int) throw();
# 297 "/usr/include/bits/mathcalls.h" 3
extern "C" int ilogbl(long double) throw(); extern "C" int __ilogbl(long double) throw();
# 302 "/usr/include/bits/mathcalls.h" 3
extern "C" long double scalblnl(long double, long) throw(); extern "C" long double __scalblnl(long double, long) throw();
# 306 "/usr/include/bits/mathcalls.h" 3
extern "C" long double nearbyintl(long double) throw(); extern "C" long double __nearbyintl(long double) throw();
# 310 "/usr/include/bits/mathcalls.h" 3
extern "C" long double roundl(long double) throw() __attribute__((__const__)); extern "C" long double __roundl(long double) throw() __attribute__((__const__));
# 314 "/usr/include/bits/mathcalls.h" 3
extern "C" long double truncl(long double) throw() __attribute__((__const__)); extern "C" long double __truncl(long double) throw() __attribute__((__const__));
# 319 "/usr/include/bits/mathcalls.h" 3
extern "C" long double remquol(long double, long double, int *) throw(); extern "C" long double __remquol(long double, long double, int *) throw();
# 326 "/usr/include/bits/mathcalls.h" 3
extern "C" long lrintl(long double) throw(); extern "C" long __lrintl(long double) throw();
# 327 "/usr/include/bits/mathcalls.h" 3
extern "C" long long llrintl(long double) throw(); extern "C" long long __llrintl(long double) throw();
# 331 "/usr/include/bits/mathcalls.h" 3
extern "C" long lroundl(long double) throw(); extern "C" long __lroundl(long double) throw();
# 332 "/usr/include/bits/mathcalls.h" 3
extern "C" long long llroundl(long double) throw(); extern "C" long long __llroundl(long double) throw();
# 336 "/usr/include/bits/mathcalls.h" 3
extern "C" long double fdiml(long double, long double) throw(); extern "C" long double __fdiml(long double, long double) throw();
# 339 "/usr/include/bits/mathcalls.h" 3
extern "C" long double fmaxl(long double, long double) throw(); extern "C" long double __fmaxl(long double, long double) throw();
# 342 "/usr/include/bits/mathcalls.h" 3
extern "C" long double fminl(long double, long double) throw(); extern "C" long double __fminl(long double, long double) throw();
# 346 "/usr/include/bits/mathcalls.h" 3
extern "C" int __fpclassifyl(long double) throw() __attribute__((__const__));
# 350 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((__weak__)) int __signbitl(long double) throw() __attribute__((__const__));
# 355 "/usr/include/bits/mathcalls.h" 3
extern "C" long double fmal(long double, long double, long double) throw(); extern "C" long double __fmal(long double, long double, long double) throw();
# 364 "/usr/include/bits/mathcalls.h" 3
extern "C" long double scalbl(long double, long double) throw(); extern "C" long double __scalbl(long double, long double) throw();
# 157 "/usr/include/math.h" 3
extern "C" { extern int signgam; }
# 199 "/usr/include/math.h" 3
enum __cuda_FP_NAN {
# 200 "/usr/include/math.h" 3
FP_NAN,
# 202 "/usr/include/math.h" 3
FP_INFINITE,
# 204 "/usr/include/math.h" 3
FP_ZERO,
# 206 "/usr/include/math.h" 3
FP_SUBNORMAL,
# 208 "/usr/include/math.h" 3
FP_NORMAL
# 210 "/usr/include/math.h" 3
};
# 291 "/usr/include/math.h" 3
extern "C" { typedef
# 285 "/usr/include/math.h" 3
enum {
# 286 "/usr/include/math.h" 3
_IEEE_ = (-1),
# 287 "/usr/include/math.h" 3
_SVID_,
# 288 "/usr/include/math.h" 3
_XOPEN_,
# 289 "/usr/include/math.h" 3
_POSIX_,
# 290 "/usr/include/math.h" 3
_ISOC_
# 291 "/usr/include/math.h" 3
} _LIB_VERSION_TYPE; }
# 296 "/usr/include/math.h" 3
extern "C" { extern _LIB_VERSION_TYPE _LIB_VERSION; }
# 307 "/usr/include/math.h" 3
extern "C" { struct __exception {
# 312 "/usr/include/math.h" 3
int type;
# 313 "/usr/include/math.h" 3
char *name;
# 314 "/usr/include/math.h" 3
double arg1;
# 315 "/usr/include/math.h" 3
double arg2;
# 316 "/usr/include/math.h" 3
double retval;
# 317 "/usr/include/math.h" 3
}; }
# 320 "/usr/include/math.h" 3
extern "C" int matherr(__exception *) throw();
# 67 "/usr/include/bits/waitstatus.h" 3
extern "C" { union wait {
# 69 "/usr/include/bits/waitstatus.h" 3
int w_status;
# 71 "/usr/include/bits/waitstatus.h" 3
struct {
# 73 "/usr/include/bits/waitstatus.h" 3
unsigned __w_termsig:7;
# 74 "/usr/include/bits/waitstatus.h" 3
unsigned __w_coredump:1;
# 75 "/usr/include/bits/waitstatus.h" 3
unsigned __w_retcode:8;
# 76 "/usr/include/bits/waitstatus.h" 3
unsigned:16;
# 84 "/usr/include/bits/waitstatus.h" 3
} __wait_terminated;
# 86 "/usr/include/bits/waitstatus.h" 3
struct {
# 88 "/usr/include/bits/waitstatus.h" 3
unsigned __w_stopval:8;
# 89 "/usr/include/bits/waitstatus.h" 3
unsigned __w_stopsig:8;
# 90 "/usr/include/bits/waitstatus.h" 3
unsigned:16;
# 97 "/usr/include/bits/waitstatus.h" 3
} __wait_stopped;
# 98 "/usr/include/bits/waitstatus.h" 3
}; }
# 102 "/usr/include/stdlib.h" 3
extern "C" { typedef
# 99 "/usr/include/stdlib.h" 3
struct div_t {
# 100 "/usr/include/stdlib.h" 3
int quot;
# 101 "/usr/include/stdlib.h" 3
int rem;
# 102 "/usr/include/stdlib.h" 3
} div_t; }
# 110 "/usr/include/stdlib.h" 3
extern "C" { typedef
# 107 "/usr/include/stdlib.h" 3
struct ldiv_t {
# 108 "/usr/include/stdlib.h" 3
long quot;
# 109 "/usr/include/stdlib.h" 3
long rem;
# 110 "/usr/include/stdlib.h" 3
} ldiv_t; }
# 122 "/usr/include/stdlib.h" 3
extern "C" { typedef
# 119 "/usr/include/stdlib.h" 3
struct lldiv_t {
# 120 "/usr/include/stdlib.h" 3
long long quot;
# 121 "/usr/include/stdlib.h" 3
long long rem;
# 122 "/usr/include/stdlib.h" 3
} lldiv_t; }
# 140 "/usr/include/stdlib.h" 3
extern "C" size_t __ctype_get_mb_cur_max() throw();
# 145 "/usr/include/stdlib.h" 3
extern "C" double atof(const char *) throw() __attribute__((__pure__));
# 148 "/usr/include/stdlib.h" 3
extern "C" int atoi(const char *) throw() __attribute__((__pure__));
# 151 "/usr/include/stdlib.h" 3
extern "C" long atol(const char *) throw() __attribute__((__pure__));
# 158 "/usr/include/stdlib.h" 3
extern "C" long long atoll(const char *) throw() __attribute__((__pure__));
# 165 "/usr/include/stdlib.h" 3
extern "C" double strtod(const char *__restrict__, char **__restrict__) throw();
# 173 "/usr/include/stdlib.h" 3
extern "C" float strtof(const char *__restrict__, char **__restrict__) throw();
# 176 "/usr/include/stdlib.h" 3
extern "C" long double strtold(const char *__restrict__, char **__restrict__) throw();
# 184 "/usr/include/stdlib.h" 3
extern "C" long strtol(const char *__restrict__, char **__restrict__, int) throw();
# 188 "/usr/include/stdlib.h" 3
extern "C" unsigned long strtoul(const char *__restrict__, char **__restrict__, int) throw();
# 196 "/usr/include/stdlib.h" 3
extern "C" long long strtoq(const char *__restrict__, char **__restrict__, int) throw();
# 201 "/usr/include/stdlib.h" 3
extern "C" unsigned long long strtouq(const char *__restrict__, char **__restrict__, int) throw();
# 210 "/usr/include/stdlib.h" 3
extern "C" long long strtoll(const char *__restrict__, char **__restrict__, int) throw();
# 215 "/usr/include/stdlib.h" 3
extern "C" unsigned long long strtoull(const char *__restrict__, char **__restrict__, int) throw();
# 240 "/usr/include/stdlib.h" 3
extern "C" long strtol_l(const char *__restrict__, char **__restrict__, int, __locale_t) throw();
# 244 "/usr/include/stdlib.h" 3
extern "C" unsigned long strtoul_l(const char *__restrict__, char **__restrict__, int, __locale_t) throw();
# 250 "/usr/include/stdlib.h" 3
extern "C" long long strtoll_l(const char *__restrict__, char **__restrict__, int, __locale_t) throw();
# 256 "/usr/include/stdlib.h" 3
extern "C" unsigned long long strtoull_l(const char *__restrict__, char **__restrict__, int, __locale_t) throw();
# 261 "/usr/include/stdlib.h" 3
extern "C" double strtod_l(const char *__restrict__, char **__restrict__, __locale_t) throw();
# 265 "/usr/include/stdlib.h" 3
extern "C" float strtof_l(const char *__restrict__, char **__restrict__, __locale_t) throw();
# 269 "/usr/include/stdlib.h" 3
extern "C" long double strtold_l(const char *__restrict__, char **__restrict__, __locale_t) throw();
# 311 "/usr/include/stdlib.h" 3
extern "C" char *l64a(long) throw();
# 314 "/usr/include/stdlib.h" 3
extern "C" long a64l(const char *) throw() __attribute__((__pure__));
# 35 "/usr/include/sys/types.h" 3
extern "C" { typedef __u_char u_char; }
# 36 "/usr/include/sys/types.h" 3
extern "C" { typedef __u_short u_short; }
# 37 "/usr/include/sys/types.h" 3
extern "C" { typedef __u_int u_int; }
# 38 "/usr/include/sys/types.h" 3
extern "C" { typedef __u_long u_long; }
# 39 "/usr/include/sys/types.h" 3
extern "C" { typedef __quad_t quad_t; }
# 40 "/usr/include/sys/types.h" 3
extern "C" { typedef __u_quad_t u_quad_t; }
# 41 "/usr/include/sys/types.h" 3
extern "C" { typedef __fsid_t fsid_t; }
# 46 "/usr/include/sys/types.h" 3
extern "C" { typedef __loff_t loff_t; }
# 50 "/usr/include/sys/types.h" 3
extern "C" { typedef __ino_t ino_t; }
# 57 "/usr/include/sys/types.h" 3
extern "C" { typedef __ino64_t ino64_t; }
# 62 "/usr/include/sys/types.h" 3
extern "C" { typedef __dev_t dev_t; }
# 67 "/usr/include/sys/types.h" 3
extern "C" { typedef __gid_t gid_t; }
# 72 "/usr/include/sys/types.h" 3
extern "C" { typedef __mode_t mode_t; }
# 77 "/usr/include/sys/types.h" 3
extern "C" { typedef __nlink_t nlink_t; }
# 82 "/usr/include/sys/types.h" 3
extern "C" { typedef __uid_t uid_t; }
# 88 "/usr/include/sys/types.h" 3
extern "C" { typedef __off_t off_t; }
# 95 "/usr/include/sys/types.h" 3
extern "C" { typedef __off64_t off64_t; }
# 105 "/usr/include/sys/types.h" 3
extern "C" { typedef __id_t id_t; }
# 110 "/usr/include/sys/types.h" 3
extern "C" { typedef __ssize_t ssize_t; }
# 116 "/usr/include/sys/types.h" 3
extern "C" { typedef __daddr_t daddr_t; }
# 117 "/usr/include/sys/types.h" 3
extern "C" { typedef __caddr_t caddr_t; }
# 123 "/usr/include/sys/types.h" 3
extern "C" { typedef __key_t key_t; }
# 137 "/usr/include/sys/types.h" 3
extern "C" { typedef __useconds_t useconds_t; }
# 141 "/usr/include/sys/types.h" 3
extern "C" { typedef __suseconds_t suseconds_t; }
# 151 "/usr/include/sys/types.h" 3
extern "C" { typedef unsigned long ulong; }
# 152 "/usr/include/sys/types.h" 3
extern "C" { typedef unsigned short ushort; }
# 153 "/usr/include/sys/types.h" 3
extern "C" { typedef unsigned uint; }
# 195 "/usr/include/sys/types.h" 3
extern "C" { typedef signed char int8_t; }
# 196 "/usr/include/sys/types.h" 3
extern "C" { typedef short int16_t; }
# 197 "/usr/include/sys/types.h" 3
extern "C" { typedef int int32_t; }
# 198 "/usr/include/sys/types.h" 3
extern "C" { typedef long int64_t; }
# 201 "/usr/include/sys/types.h" 3
extern "C" { typedef unsigned char u_int8_t; }
# 202 "/usr/include/sys/types.h" 3
extern "C" { typedef unsigned short u_int16_t; }
# 203 "/usr/include/sys/types.h" 3
extern "C" { typedef unsigned u_int32_t; }
# 204 "/usr/include/sys/types.h" 3
extern "C" { typedef unsigned long u_int64_t; }
# 206 "/usr/include/sys/types.h" 3
extern "C" { typedef int register_t; }
# 24 "/usr/include/bits/sigset.h" 3
extern "C" { typedef int __sig_atomic_t; }
# 32 "/usr/include/bits/sigset.h" 3
extern "C" { typedef
# 30 "/usr/include/bits/sigset.h" 3
struct __sigset_t {
# 31 "/usr/include/bits/sigset.h" 3
unsigned long __val[((1024) / ((8) * sizeof(unsigned long)))];
# 32 "/usr/include/bits/sigset.h" 3
} __sigset_t; }
# 38 "/usr/include/sys/select.h" 3
extern "C" { typedef __sigset_t sigset_t; }
# 69 "/usr/include/bits/time.h" 3
extern "C" { struct timeval {
# 71 "/usr/include/bits/time.h" 3
__time_t tv_sec;
# 72 "/usr/include/bits/time.h" 3
__suseconds_t tv_usec;
# 73 "/usr/include/bits/time.h" 3
}; }
# 55 "/usr/include/sys/select.h" 3
extern "C" { typedef long __fd_mask; }
# 78 "/usr/include/sys/select.h" 3
extern "C" { typedef
# 68 "/usr/include/sys/select.h" 3
struct fd_set {
# 72 "/usr/include/sys/select.h" 3
__fd_mask fds_bits[((1024) / ((8) * sizeof(__fd_mask)))];
# 78 "/usr/include/sys/select.h" 3
} fd_set; }
# 85 "/usr/include/sys/select.h" 3
extern "C" { typedef __fd_mask fd_mask; }
# 109 "/usr/include/sys/select.h" 3
extern "C" int select(int, fd_set *__restrict__, fd_set *__restrict__, fd_set *__restrict__, timeval *__restrict__);
# 121 "/usr/include/sys/select.h" 3
extern "C" int pselect(int, fd_set *__restrict__, fd_set *__restrict__, fd_set *__restrict__, const timespec *__restrict__, const __sigset_t *__restrict__);
# 31 "/usr/include/sys/sysmacros.h" 3
extern "C" unsigned gnu_dev_major(unsigned long long) throw();
# 34 "/usr/include/sys/sysmacros.h" 3
extern "C" unsigned gnu_dev_minor(unsigned long long) throw();
# 37 "/usr/include/sys/sysmacros.h" 3
extern "C" unsigned long long gnu_dev_makedev(unsigned, unsigned) throw();
# 228 "/usr/include/sys/types.h" 3
extern "C" { typedef __blksize_t blksize_t; }
# 235 "/usr/include/sys/types.h" 3
extern "C" { typedef __blkcnt_t blkcnt_t; }
# 239 "/usr/include/sys/types.h" 3
extern "C" { typedef __fsblkcnt_t fsblkcnt_t; }
# 243 "/usr/include/sys/types.h" 3
extern "C" { typedef __fsfilcnt_t fsfilcnt_t; }
# 262 "/usr/include/sys/types.h" 3
extern "C" { typedef __blkcnt64_t blkcnt64_t; }
# 263 "/usr/include/sys/types.h" 3
extern "C" { typedef __fsblkcnt64_t fsblkcnt64_t; }
# 264 "/usr/include/sys/types.h" 3
extern "C" { typedef __fsfilcnt64_t fsfilcnt64_t; }
# 50 "/usr/include/bits/pthreadtypes.h" 3
extern "C" { typedef unsigned long pthread_t; }
# 57 "/usr/include/bits/pthreadtypes.h" 3
extern "C" { typedef
# 54 "/usr/include/bits/pthreadtypes.h" 3
union pthread_attr_t {
# 55 "/usr/include/bits/pthreadtypes.h" 3
char __size[56];
# 56 "/usr/include/bits/pthreadtypes.h" 3
long __align;
# 57 "/usr/include/bits/pthreadtypes.h" 3
} pthread_attr_t; }
# 65 "/usr/include/bits/pthreadtypes.h" 3
extern "C" { typedef
# 61 "/usr/include/bits/pthreadtypes.h" 3
struct __pthread_internal_list {
# 63 "/usr/include/bits/pthreadtypes.h" 3
__pthread_internal_list *__prev;
# 64 "/usr/include/bits/pthreadtypes.h" 3
__pthread_internal_list *__next;
# 65 "/usr/include/bits/pthreadtypes.h" 3
} __pthread_list_t; }
# 104 "/usr/include/bits/pthreadtypes.h" 3
extern "C" { typedef
# 77 "/usr/include/bits/pthreadtypes.h" 3
union pthread_mutex_t {
# 78 "/usr/include/bits/pthreadtypes.h" 3
struct __pthread_mutex_s {
# 80 "/usr/include/bits/pthreadtypes.h" 3
int __lock;
# 81 "/usr/include/bits/pthreadtypes.h" 3
unsigned __count;
# 82 "/usr/include/bits/pthreadtypes.h" 3
int __owner;
# 84 "/usr/include/bits/pthreadtypes.h" 3
unsigned __nusers;
# 88 "/usr/include/bits/pthreadtypes.h" 3
int __kind;
# 90 "/usr/include/bits/pthreadtypes.h" 3
int __spins;
# 91 "/usr/include/bits/pthreadtypes.h" 3
__pthread_list_t __list;
# 101 "/usr/include/bits/pthreadtypes.h" 3
} __data;
# 102 "/usr/include/bits/pthreadtypes.h" 3
char __size[40];
# 103 "/usr/include/bits/pthreadtypes.h" 3
long __align;
# 104 "/usr/include/bits/pthreadtypes.h" 3
} pthread_mutex_t; }
# 110 "/usr/include/bits/pthreadtypes.h" 3
extern "C" { typedef
# 107 "/usr/include/bits/pthreadtypes.h" 3
union pthread_mutexattr_t {
# 108 "/usr/include/bits/pthreadtypes.h" 3
char __size[4];
# 109 "/usr/include/bits/pthreadtypes.h" 3
int __align;
# 110 "/usr/include/bits/pthreadtypes.h" 3
} pthread_mutexattr_t; }
# 130 "/usr/include/bits/pthreadtypes.h" 3
extern "C" { typedef
# 116 "/usr/include/bits/pthreadtypes.h" 3
union pthread_cond_t {
# 118 "/usr/include/bits/pthreadtypes.h" 3
struct {
# 119 "/usr/include/bits/pthreadtypes.h" 3
int __lock;
# 120 "/usr/include/bits/pthreadtypes.h" 3
unsigned __futex;
# 121 "/usr/include/bits/pthreadtypes.h" 3
unsigned long long __total_seq;
# 122 "/usr/include/bits/pthreadtypes.h" 3
unsigned long long __wakeup_seq;
# 123 "/usr/include/bits/pthreadtypes.h" 3
unsigned long long __woken_seq;
# 124 "/usr/include/bits/pthreadtypes.h" 3
void *__mutex;
# 125 "/usr/include/bits/pthreadtypes.h" 3
unsigned __nwaiters;
# 126 "/usr/include/bits/pthreadtypes.h" 3
unsigned __broadcast_seq;
# 127 "/usr/include/bits/pthreadtypes.h" 3
} __data;
# 128 "/usr/include/bits/pthreadtypes.h" 3
char __size[48];
# 129 "/usr/include/bits/pthreadtypes.h" 3
long long __align;
# 130 "/usr/include/bits/pthreadtypes.h" 3
} pthread_cond_t; }
# 136 "/usr/include/bits/pthreadtypes.h" 3
extern "C" { typedef
# 133 "/usr/include/bits/pthreadtypes.h" 3
union pthread_condattr_t {
# 134 "/usr/include/bits/pthreadtypes.h" 3
char __size[4];
# 135 "/usr/include/bits/pthreadtypes.h" 3
int __align;
# 136 "/usr/include/bits/pthreadtypes.h" 3
} pthread_condattr_t; }
# 140 "/usr/include/bits/pthreadtypes.h" 3
extern "C" { typedef unsigned pthread_key_t; }
# 144 "/usr/include/bits/pthreadtypes.h" 3
extern "C" { typedef int pthread_once_t; }
# 189 "/usr/include/bits/pthreadtypes.h" 3
extern "C" { typedef
# 151 "/usr/include/bits/pthreadtypes.h" 3
union pthread_rwlock_t {
# 154 "/usr/include/bits/pthreadtypes.h" 3
struct {
# 155 "/usr/include/bits/pthreadtypes.h" 3
int __lock;
# 156 "/usr/include/bits/pthreadtypes.h" 3
unsigned __nr_readers;
# 157 "/usr/include/bits/pthreadtypes.h" 3
unsigned __readers_wakeup;
# 158 "/usr/include/bits/pthreadtypes.h" 3
unsigned __writer_wakeup;
# 159 "/usr/include/bits/pthreadtypes.h" 3
unsigned __nr_readers_queued;
# 160 "/usr/include/bits/pthreadtypes.h" 3
unsigned __nr_writers_queued;
# 161 "/usr/include/bits/pthreadtypes.h" 3
int __writer;
# 162 "/usr/include/bits/pthreadtypes.h" 3
int __shared;
# 163 "/usr/include/bits/pthreadtypes.h" 3
unsigned long __pad1;
# 164 "/usr/include/bits/pthreadtypes.h" 3
unsigned long __pad2;
# 167 "/usr/include/bits/pthreadtypes.h" 3
unsigned __flags;
# 168 "/usr/include/bits/pthreadtypes.h" 3
} __data;
# 187 "/usr/include/bits/pthreadtypes.h" 3
char __size[56];
# 188 "/usr/include/bits/pthreadtypes.h" 3
long __align;
# 189 "/usr/include/bits/pthreadtypes.h" 3
} pthread_rwlock_t; }
# 195 "/usr/include/bits/pthreadtypes.h" 3
extern "C" { typedef
# 192 "/usr/include/bits/pthreadtypes.h" 3
union pthread_rwlockattr_t {
# 193 "/usr/include/bits/pthreadtypes.h" 3
char __size[8];
# 194 "/usr/include/bits/pthreadtypes.h" 3
long __align;
# 195 "/usr/include/bits/pthreadtypes.h" 3
} pthread_rwlockattr_t; }
# 201 "/usr/include/bits/pthreadtypes.h" 3
extern "C" { typedef volatile int pthread_spinlock_t; }
# 210 "/usr/include/bits/pthreadtypes.h" 3
extern "C" { typedef
# 207 "/usr/include/bits/pthreadtypes.h" 3
union pthread_barrier_t {
# 208 "/usr/include/bits/pthreadtypes.h" 3
char __size[32];
# 209 "/usr/include/bits/pthreadtypes.h" 3
long __align;
# 210 "/usr/include/bits/pthreadtypes.h" 3
} pthread_barrier_t; }
# 216 "/usr/include/bits/pthreadtypes.h" 3
extern "C" { typedef
# 213 "/usr/include/bits/pthreadtypes.h" 3
union pthread_barrierattr_t {
# 214 "/usr/include/bits/pthreadtypes.h" 3
char __size[4];
# 215 "/usr/include/bits/pthreadtypes.h" 3
int __align;
# 216 "/usr/include/bits/pthreadtypes.h" 3
} pthread_barrierattr_t; }
# 327 "/usr/include/stdlib.h" 3
extern "C" long random() throw();
# 330 "/usr/include/stdlib.h" 3
extern "C" void srandom(unsigned) throw();
# 336 "/usr/include/stdlib.h" 3
extern "C" char *initstate(unsigned, char *, size_t) throw();
# 341 "/usr/include/stdlib.h" 3
extern "C" char *setstate(char *) throw();
# 349 "/usr/include/stdlib.h" 3
extern "C" { struct random_data {
# 351 "/usr/include/stdlib.h" 3
int32_t *fptr;
# 352 "/usr/include/stdlib.h" 3
int32_t *rptr;
# 353 "/usr/include/stdlib.h" 3
int32_t *state;
# 354 "/usr/include/stdlib.h" 3
int rand_type;
# 355 "/usr/include/stdlib.h" 3
int rand_deg;
# 356 "/usr/include/stdlib.h" 3
int rand_sep;
# 357 "/usr/include/stdlib.h" 3
int32_t *end_ptr;
# 358 "/usr/include/stdlib.h" 3
}; }
# 360 "/usr/include/stdlib.h" 3
extern "C" int random_r(random_data *__restrict__, int32_t *__restrict__) throw();
# 363 "/usr/include/stdlib.h" 3
extern "C" int srandom_r(unsigned, random_data *) throw();
# 366 "/usr/include/stdlib.h" 3
extern "C" int initstate_r(unsigned, char *__restrict__, size_t, random_data *__restrict__) throw();
# 371 "/usr/include/stdlib.h" 3
extern "C" int setstate_r(char *__restrict__, random_data *__restrict__) throw();
# 380 "/usr/include/stdlib.h" 3
extern "C" int rand() throw();
# 382 "/usr/include/stdlib.h" 3
extern "C" void srand(unsigned) throw();
# 387 "/usr/include/stdlib.h" 3
extern "C" int rand_r(unsigned *) throw();
# 395 "/usr/include/stdlib.h" 3
extern "C" double drand48() throw();
# 396 "/usr/include/stdlib.h" 3
extern "C" double erand48(unsigned short [3]) throw();
# 399 "/usr/include/stdlib.h" 3
extern "C" long lrand48() throw();
# 400 "/usr/include/stdlib.h" 3
extern "C" long nrand48(unsigned short [3]) throw();
# 404 "/usr/include/stdlib.h" 3
extern "C" long mrand48() throw();
# 405 "/usr/include/stdlib.h" 3
extern "C" long jrand48(unsigned short [3]) throw();
# 409 "/usr/include/stdlib.h" 3
extern "C" void srand48(long) throw();
# 410 "/usr/include/stdlib.h" 3
extern "C" unsigned short *seed48(unsigned short [3]) throw();
# 412 "/usr/include/stdlib.h" 3
extern "C" void lcong48(unsigned short [7]) throw();
# 418 "/usr/include/stdlib.h" 3
extern "C" { struct drand48_data {
# 420 "/usr/include/stdlib.h" 3
unsigned short __x[3];
# 421 "/usr/include/stdlib.h" 3
unsigned short __old_x[3];
# 422 "/usr/include/stdlib.h" 3
unsigned short __c;
# 423 "/usr/include/stdlib.h" 3
unsigned short __init;
# 424 "/usr/include/stdlib.h" 3
unsigned long long __a;
# 425 "/usr/include/stdlib.h" 3
}; }
# 428 "/usr/include/stdlib.h" 3
extern "C" int drand48_r(drand48_data *__restrict__, double *__restrict__) throw();
# 430 "/usr/include/stdlib.h" 3
extern "C" int erand48_r(unsigned short [3], drand48_data *__restrict__, double *__restrict__) throw();
# 435 "/usr/include/stdlib.h" 3
extern "C" int lrand48_r(drand48_data *__restrict__, long *__restrict__) throw();
# 438 "/usr/include/stdlib.h" 3
extern "C" int nrand48_r(unsigned short [3], drand48_data *__restrict__, long *__restrict__) throw();
# 444 "/usr/include/stdlib.h" 3
extern "C" int mrand48_r(drand48_data *__restrict__, long *__restrict__) throw();
# 447 "/usr/include/stdlib.h" 3
extern "C" int jrand48_r(unsigned short [3], drand48_data *__restrict__, long *__restrict__) throw();
# 453 "/usr/include/stdlib.h" 3
extern "C" int srand48_r(long, drand48_data *) throw();
# 456 "/usr/include/stdlib.h" 3
extern "C" int seed48_r(unsigned short [3], drand48_data *) throw();
# 459 "/usr/include/stdlib.h" 3
extern "C" int lcong48_r(unsigned short [7], drand48_data *) throw();
# 471 "/usr/include/stdlib.h" 3
extern "C" void *malloc(size_t) throw() __attribute__((__malloc__));
# 473 "/usr/include/stdlib.h" 3
extern "C" void *calloc(size_t, size_t) throw() __attribute__((__malloc__));
# 485 "/usr/include/stdlib.h" 3
extern "C" void *realloc(void *, size_t) throw();
# 488 "/usr/include/stdlib.h" 3
extern "C" void free(void *) throw();
# 493 "/usr/include/stdlib.h" 3
extern "C" void cfree(void *) throw();
# 33 "/usr/include/alloca.h" 3
extern "C" void *alloca(size_t) throw();
# 502 "/usr/include/stdlib.h" 3
extern "C" void *valloc(size_t) throw() __attribute__((__malloc__));
# 507 "/usr/include/stdlib.h" 3
extern "C" int posix_memalign(void **, size_t, size_t) throw();
# 513 "/usr/include/stdlib.h" 3
extern "C" void abort() throw() __attribute__((__noreturn__));
# 517 "/usr/include/stdlib.h" 3
extern "C" int atexit(void (*)(void)) throw();
# 523 "/usr/include/stdlib.h" 3
extern "C" int on_exit(void (*)(int, void *), void *) throw();
# 531 "/usr/include/stdlib.h" 3
extern "C" void exit(int) throw() __attribute__((__noreturn__));
# 538 "/usr/include/stdlib.h" 3
extern "C" void _Exit(int) throw() __attribute__((__noreturn__));
# 545 "/usr/include/stdlib.h" 3
extern "C" char *getenv(const char *) throw();
# 550 "/usr/include/stdlib.h" 3
extern "C" char *__secure_getenv(const char *) throw();
# 557 "/usr/include/stdlib.h" 3
extern "C" int putenv(char *) throw();
# 563 "/usr/include/stdlib.h" 3
extern "C" int setenv(const char *, const char *, int) throw();
# 567 "/usr/include/stdlib.h" 3
extern "C" int unsetenv(const char *) throw();
# 574 "/usr/include/stdlib.h" 3
extern "C" int clearenv() throw();
# 583 "/usr/include/stdlib.h" 3
extern "C" char *mktemp(char *) throw();
# 594 "/usr/include/stdlib.h" 3
extern "C" int mkstemp(char *);
# 604 "/usr/include/stdlib.h" 3
extern "C" int mkstemp64(char *);
# 614 "/usr/include/stdlib.h" 3
extern "C" char *mkdtemp(char *) throw();
# 625 "/usr/include/stdlib.h" 3
extern "C" int mkostemp(char *, int);
# 635 "/usr/include/stdlib.h" 3
extern "C" int mkostemp64(char *, int);
# 645 "/usr/include/stdlib.h" 3
extern "C" int system(const char *);
# 652 "/usr/include/stdlib.h" 3
extern "C" char *canonicalize_file_name(const char *) throw();
# 662 "/usr/include/stdlib.h" 3
extern "C" char *realpath(const char *__restrict__, char *__restrict__) throw();
# 670 "/usr/include/stdlib.h" 3
extern "C" { typedef int (*__compar_fn_t)(const void *, const void *); }
# 673 "/usr/include/stdlib.h" 3
extern "C" { typedef __compar_fn_t comparison_fn_t; }
# 677 "/usr/include/stdlib.h" 3
extern "C" { typedef int (*__compar_d_fn_t)(const void *, const void *, void *); }
# 683 "/usr/include/stdlib.h" 3
extern "C" void *bsearch(const void *, const void *, size_t, size_t, __compar_fn_t);
# 689 "/usr/include/stdlib.h" 3
extern "C" void qsort(void *, size_t, size_t, __compar_fn_t);
# 692 "/usr/include/stdlib.h" 3
extern "C" void qsort_r(void *, size_t, size_t, __compar_d_fn_t, void *);
# 699 "/usr/include/stdlib.h" 3
extern "C" __attribute__((__weak__)) int abs(int) throw() __attribute__((__const__));
# 700 "/usr/include/stdlib.h" 3
extern "C" __attribute__((__weak__)) long labs(long) throw() __attribute__((__const__));
# 704 "/usr/include/stdlib.h" 3
extern "C" __attribute__((__weak__)) long long llabs(long long) throw() __attribute__((__const__));
# 713 "/usr/include/stdlib.h" 3
extern "C" div_t div(int, int) throw() __attribute__((__const__));
# 715 "/usr/include/stdlib.h" 3
extern "C" ldiv_t ldiv(long, long) throw() __attribute__((__const__));
# 721 "/usr/include/stdlib.h" 3
extern "C" lldiv_t lldiv(long long, long long) throw() __attribute__((__const__));
# 735 "/usr/include/stdlib.h" 3
extern "C" char *ecvt(double, int, int *__restrict__, int *__restrict__) throw();
# 741 "/usr/include/stdlib.h" 3
extern "C" char *fcvt(double, int, int *__restrict__, int *__restrict__) throw();
# 747 "/usr/include/stdlib.h" 3
extern "C" char *gcvt(double, int, char *) throw();
# 753 "/usr/include/stdlib.h" 3
extern "C" char *qecvt(long double, int, int *__restrict__, int *__restrict__) throw();
# 756 "/usr/include/stdlib.h" 3
extern "C" char *qfcvt(long double, int, int *__restrict__, int *__restrict__) throw();
# 759 "/usr/include/stdlib.h" 3
extern "C" char *qgcvt(long double, int, char *) throw();
# 765 "/usr/include/stdlib.h" 3
extern "C" int ecvt_r(double, int, int *__restrict__, int *__restrict__, char *__restrict__, size_t) throw();
# 768 "/usr/include/stdlib.h" 3
extern "C" int fcvt_r(double, int, int *__restrict__, int *__restrict__, char *__restrict__, size_t) throw();
# 772 "/usr/include/stdlib.h" 3
extern "C" int qecvt_r(long double, int, int *__restrict__, int *__restrict__, char *__restrict__, size_t) throw();
# 776 "/usr/include/stdlib.h" 3
extern "C" int qfcvt_r(long double, int, int *__restrict__, int *__restrict__, char *__restrict__, size_t) throw();
# 787 "/usr/include/stdlib.h" 3
extern "C" int mblen(const char *, size_t) throw();
# 790 "/usr/include/stdlib.h" 3
extern "C" int mbtowc(wchar_t *__restrict__, const char *__restrict__, size_t) throw();
# 794 "/usr/include/stdlib.h" 3
extern "C" int wctomb(char *, wchar_t) throw();
# 798 "/usr/include/stdlib.h" 3
extern "C" size_t mbstowcs(wchar_t *__restrict__, const char *__restrict__, size_t) throw();
# 801 "/usr/include/stdlib.h" 3
extern "C" size_t wcstombs(char *__restrict__, const wchar_t *__restrict__, size_t) throw();
# 812 "/usr/include/stdlib.h" 3
extern "C" int rpmatch(const char *) throw();
# 823 "/usr/include/stdlib.h" 3
extern "C" int getsubopt(char **__restrict__, char *const *__restrict__, char **__restrict__) throw();
# 832 "/usr/include/stdlib.h" 3
extern "C" void setkey(const char *) throw();
# 840 "/usr/include/stdlib.h" 3
extern "C" int posix_openpt(int);
# 848 "/usr/include/stdlib.h" 3
extern "C" int grantpt(int) throw();
# 852 "/usr/include/stdlib.h" 3
extern "C" int unlockpt(int) throw();
# 857 "/usr/include/stdlib.h" 3
extern "C" char *ptsname(int) throw();
# 864 "/usr/include/stdlib.h" 3
extern "C" int ptsname_r(int, char *, size_t) throw();
# 868 "/usr/include/stdlib.h" 3
extern "C" int getpt();
# 875 "/usr/include/stdlib.h" 3
extern "C" int getloadavg(double [], int) throw();
# 74 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
namespace __gnu_cxx __attribute__((visibility("default"))) {
# 76 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
template<class _Iterator, class _Container> class __normal_iterator;
# 79 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
}
# 81 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
namespace std __attribute__((visibility("default"))) {
# 83 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
struct __true_type { };
# 84 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
struct __false_type { };
# 86 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
template<bool >
# 87 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
struct __truth_type {
# 88 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
typedef __false_type __type; };
# 91 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
template<> struct __truth_type< true> {
# 92 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
typedef __true_type __type; };
# 96 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
template<class _Sp, class _Tp>
# 97 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
struct __traitor {
# 99 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
enum __cuda___value { __value = (((bool)_Sp::__value) || ((bool)_Tp::__value))};
# 100 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
typedef typename __truth_type< (((bool)_Sp::__value) || ((bool)_Tp::__value))> ::__type __type;
# 101 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
};
# 104 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
template<class , class >
# 105 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
struct __are_same {
# 107 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
enum __cuda___value { __value};
# 108 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
typedef __false_type __type;
# 109 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
};
# 111 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
template<class _Tp>
# 112 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
struct __are_same< _Tp, _Tp> {
# 114 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
enum __cuda___value { __value = 1};
# 115 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
typedef __true_type __type;
# 116 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
};
# 119 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
template<class _Tp>
# 120 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
struct __is_void {
# 122 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
enum __cuda___value { __value};
# 123 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
typedef __false_type __type;
# 124 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
};
# 127 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
template<> struct __is_void< void> {
# 129 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
enum __cuda___value { __value = 1};
# 130 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
typedef __true_type __type;
# 131 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
};
# 136 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
template<class _Tp>
# 137 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
struct __is_integer {
# 139 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
enum __cuda___value { __value};
# 140 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
typedef __false_type __type;
# 141 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
};
# 147 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
template<> struct __is_integer< bool> {
# 149 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
enum __cuda___value { __value = 1};
# 150 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
typedef __true_type __type;
# 151 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
};
# 154 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
template<> struct __is_integer< char> {
# 156 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
enum __cuda___value { __value = 1};
# 157 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
typedef __true_type __type;
# 158 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
};
# 161 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
template<> struct __is_integer< signed char> {
# 163 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
enum __cuda___value { __value = 1};
# 164 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
typedef __true_type __type;
# 165 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
};
# 168 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
template<> struct __is_integer< unsigned char> {
# 170 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
enum __cuda___value { __value = 1};
# 171 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
typedef __true_type __type;
# 172 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
};
# 176 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
template<> struct __is_integer< wchar_t> {
# 178 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
enum __cuda___value { __value = 1};
# 179 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
typedef __true_type __type;
# 180 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
};
# 184 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
template<> struct __is_integer< short> {
# 186 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
enum __cuda___value { __value = 1};
# 187 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
typedef __true_type __type;
# 188 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
};
# 191 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
template<> struct __is_integer< unsigned short> {
# 193 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
enum __cuda___value { __value = 1};
# 194 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
typedef __true_type __type;
# 195 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
};
# 198 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
template<> struct __is_integer< int> {
# 200 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
enum __cuda___value { __value = 1};
# 201 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
typedef __true_type __type;
# 202 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
};
# 205 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
template<> struct __is_integer< unsigned> {
# 207 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
enum __cuda___value { __value = 1};
# 208 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
typedef __true_type __type;
# 209 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
};
# 212 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
template<> struct __is_integer< long> {
# 214 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
enum __cuda___value { __value = 1};
# 215 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
typedef __true_type __type;
# 216 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
};
# 219 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
template<> struct __is_integer< unsigned long> {
# 221 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
enum __cuda___value { __value = 1};
# 222 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
typedef __true_type __type;
# 223 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
};
# 226 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
template<> struct __is_integer< long long> {
# 228 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
enum __cuda___value { __value = 1};
# 229 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
typedef __true_type __type;
# 230 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
};
# 233 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
template<> struct __is_integer< unsigned long long> {
# 235 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
enum __cuda___value { __value = 1};
# 236 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
typedef __true_type __type;
# 237 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
};
# 242 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
template<class _Tp>
# 243 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
struct __is_floating {
# 245 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
enum __cuda___value { __value};
# 246 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
typedef __false_type __type;
# 247 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
};
# 251 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
template<> struct __is_floating< float> {
# 253 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
enum __cuda___value { __value = 1};
# 254 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
typedef __true_type __type;
# 255 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
};
# 258 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
template<> struct __is_floating< double> {
# 260 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
enum __cuda___value { __value = 1};
# 261 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
typedef __true_type __type;
# 262 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
};
# 265 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
template<> struct __is_floating< long double> {
# 267 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
enum __cuda___value { __value = 1};
# 268 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
typedef __true_type __type;
# 269 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
};
# 274 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
template<class _Tp>
# 275 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
struct __is_pointer {
# 277 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
enum __cuda___value { __value};
# 278 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
typedef __false_type __type;
# 279 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
};
# 281 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
template<class _Tp>
# 282 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
struct __is_pointer< _Tp *> {
# 284 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
enum __cuda___value { __value = 1};
# 285 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
typedef __true_type __type;
# 286 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
};
# 291 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
template<class _Tp>
# 292 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
struct __is_normal_iterator {
# 294 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
enum __cuda___value { __value};
# 295 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
typedef __false_type __type;
# 296 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
};
# 298 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
template<class _Iterator, class _Container>
# 299 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
struct __is_normal_iterator< __gnu_cxx::__normal_iterator< _Iterator, _Container> > {
# 302 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
enum __cuda___value { __value = 1};
# 303 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
typedef __true_type __type;
# 304 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
};
# 309 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
template<class _Tp>
# 310 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
struct __is_arithmetic : public __traitor< __is_integer< _Tp> , __is_floating< _Tp> > {
# 312 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
};
# 317 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
template<class _Tp>
# 318 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
struct __is_fundamental : public __traitor< __is_void< _Tp> , __is_arithmetic< _Tp> > {
# 320 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
};
# 325 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
template<class _Tp>
# 326 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
struct __is_scalar : public __traitor< __is_arithmetic< _Tp> , __is_pointer< _Tp> > {
# 328 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
};
# 333 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
template<class _Tp>
# 334 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
struct __is_char {
# 336 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
enum __cuda___value { __value};
# 337 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
typedef __false_type __type;
# 338 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
};
# 341 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
template<> struct __is_char< char> {
# 343 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
enum __cuda___value { __value = 1};
# 344 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
typedef __true_type __type;
# 345 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
};
# 349 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
template<> struct __is_char< wchar_t> {
# 351 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
enum __cuda___value { __value = 1};
# 352 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
typedef __true_type __type;
# 353 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
};
# 356 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
template<class _Tp>
# 357 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
struct __is_byte {
# 359 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
enum __cuda___value { __value};
# 360 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
typedef __false_type __type;
# 361 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
};
# 364 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
template<> struct __is_byte< char> {
# 366 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
enum __cuda___value { __value = 1};
# 367 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
typedef __true_type __type;
# 368 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
};
# 371 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
template<> struct __is_byte< signed char> {
# 373 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
enum __cuda___value { __value = 1};
# 374 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
typedef __true_type __type;
# 375 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
};
# 378 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
template<> struct __is_byte< unsigned char> {
# 380 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
enum __cuda___value { __value = 1};
# 381 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
typedef __true_type __type;
# 382 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
};
# 387 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
template<class _Tp>
# 388 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
struct __is_move_iterator {
# 390 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
enum __cuda___value { __value};
# 391 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
typedef __false_type __type;
# 392 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
};
# 406 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cpp_type_traits.h" 3
}
# 43 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/ext/type_traits.h" 3
namespace __gnu_cxx __attribute__((visibility("default"))) {
# 46 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/ext/type_traits.h" 3
template<bool , class >
# 47 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/ext/type_traits.h" 3
struct __enable_if {
# 48 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/ext/type_traits.h" 3
};
# 50 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/ext/type_traits.h" 3
template<class _Tp>
# 51 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/ext/type_traits.h" 3
struct __enable_if< true, _Tp> {
# 52 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/ext/type_traits.h" 3
typedef _Tp __type; };
# 56 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/ext/type_traits.h" 3
template<bool _Cond, class _Iftrue, class _Iffalse>
# 57 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/ext/type_traits.h" 3
struct __conditional_type {
# 58 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/ext/type_traits.h" 3
typedef _Iftrue __type; };
# 60 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/ext/type_traits.h" 3
template<class _Iftrue, class _Iffalse>
# 61 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/ext/type_traits.h" 3
struct __conditional_type< false, _Iftrue, _Iffalse> {
# 62 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/ext/type_traits.h" 3
typedef _Iffalse __type; };
# 66 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/ext/type_traits.h" 3
template<class _Tp>
# 67 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/ext/type_traits.h" 3
struct __add_unsigned {
# 70 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/ext/type_traits.h" 3
private: typedef __enable_if< std::__is_integer< _Tp> ::__value, _Tp> __if_type;
# 73 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/ext/type_traits.h" 3
public: typedef typename __enable_if< std::__is_integer< _Tp> ::__value, _Tp> ::__type __type;
# 74 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/ext/type_traits.h" 3
};
# 77 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/ext/type_traits.h" 3
template<> struct __add_unsigned< char> {
# 78 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/ext/type_traits.h" 3
typedef unsigned char __type; };
# 81 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/ext/type_traits.h" 3
template<> struct __add_unsigned< signed char> {
# 82 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/ext/type_traits.h" 3
typedef unsigned char __type; };
# 85 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/ext/type_traits.h" 3
template<> struct __add_unsigned< short> {
# 86 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/ext/type_traits.h" 3
typedef unsigned short __type; };
# 89 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/ext/type_traits.h" 3
template<> struct __add_unsigned< int> {
# 90 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/ext/type_traits.h" 3
typedef unsigned __type; };
# 93 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/ext/type_traits.h" 3
template<> struct __add_unsigned< long> {
# 94 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/ext/type_traits.h" 3
typedef unsigned long __type; };
# 97 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/ext/type_traits.h" 3
template<> struct __add_unsigned< long long> {
# 98 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/ext/type_traits.h" 3
typedef unsigned long long __type; };
# 102 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/ext/type_traits.h" 3
template<> struct __add_unsigned< bool> ;
# 105 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/ext/type_traits.h" 3
template<> struct __add_unsigned< wchar_t> ;
# 109 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/ext/type_traits.h" 3
template<class _Tp>
# 110 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/ext/type_traits.h" 3
struct __remove_unsigned {
# 113 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/ext/type_traits.h" 3
private: typedef __enable_if< std::__is_integer< _Tp> ::__value, _Tp> __if_type;
# 116 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/ext/type_traits.h" 3
public: typedef typename __enable_if< std::__is_integer< _Tp> ::__value, _Tp> ::__type __type;
# 117 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/ext/type_traits.h" 3
};
# 120 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/ext/type_traits.h" 3
template<> struct __remove_unsigned< char> {
# 121 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/ext/type_traits.h" 3
typedef signed char __type; };
# 124 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/ext/type_traits.h" 3
template<> struct __remove_unsigned< unsigned char> {
# 125 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/ext/type_traits.h" 3
typedef signed char __type; };
# 128 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/ext/type_traits.h" 3
template<> struct __remove_unsigned< unsigned short> {
# 129 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/ext/type_traits.h" 3
typedef short __type; };
# 132 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/ext/type_traits.h" 3
template<> struct __remove_unsigned< unsigned> {
# 133 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/ext/type_traits.h" 3
typedef int __type; };
# 136 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/ext/type_traits.h" 3
template<> struct __remove_unsigned< unsigned long> {
# 137 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/ext/type_traits.h" 3
typedef long __type; };
# 140 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/ext/type_traits.h" 3
template<> struct __remove_unsigned< unsigned long long> {
# 141 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/ext/type_traits.h" 3
typedef long long __type; };
# 145 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/ext/type_traits.h" 3
template<> struct __remove_unsigned< bool> ;
# 148 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/ext/type_traits.h" 3
template<> struct __remove_unsigned< wchar_t> ;
# 152 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/ext/type_traits.h" 3
template<class _Type> inline bool
# 154 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/ext/type_traits.h" 3
__is_null_pointer(_Type *__ptr)
# 155 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/ext/type_traits.h" 3
{ return __ptr == 0; }
# 157 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/ext/type_traits.h" 3
template<class _Type> inline bool
# 159 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/ext/type_traits.h" 3
__is_null_pointer(_Type)
# 160 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/ext/type_traits.h" 3
{ return false; }
# 164 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/ext/type_traits.h" 3
template<class _Tp, bool = std::__is_integer< _Tp> ::__value>
# 165 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/ext/type_traits.h" 3
struct __promote {
# 166 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/ext/type_traits.h" 3
typedef double __type; };
# 168 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/ext/type_traits.h" 3
template<class _Tp>
# 169 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/ext/type_traits.h" 3
struct __promote< _Tp, false> {
# 170 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/ext/type_traits.h" 3
typedef _Tp __type; };
# 172 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/ext/type_traits.h" 3
template<class _Tp, class _Up>
# 173 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/ext/type_traits.h" 3
struct __promote_2 {
# 176 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/ext/type_traits.h" 3
private: typedef typename __promote< _Tp, std::__is_integer< _Tp> ::__value> ::__type __type1;
# 177 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/ext/type_traits.h" 3
typedef typename __promote< _Up, std::__is_integer< _Up> ::__value> ::__type __type2;
# 180 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/ext/type_traits.h" 3
public: typedef __typeof__((__type1() + __type2())) __type;
# 181 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/ext/type_traits.h" 3
};
# 183 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/ext/type_traits.h" 3
template<class _Tp, class _Up, class _Vp>
# 184 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/ext/type_traits.h" 3
struct __promote_3 {
# 187 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/ext/type_traits.h" 3
private: typedef typename __promote< _Tp, std::__is_integer< _Tp> ::__value> ::__type __type1;
# 188 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/ext/type_traits.h" 3
typedef typename __promote< _Up, std::__is_integer< _Up> ::__value> ::__type __type2;
# 189 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/ext/type_traits.h" 3
typedef typename __promote< _Vp, std::__is_integer< _Vp> ::__value> ::__type __type3;
# 192 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/ext/type_traits.h" 3
public: typedef __typeof__(((__type1() + __type2()) + __type3())) __type;
# 193 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/ext/type_traits.h" 3
};
# 195 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/ext/type_traits.h" 3
template<class _Tp, class _Up, class _Vp, class _Wp>
# 196 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/ext/type_traits.h" 3
struct __promote_4 {
# 199 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/ext/type_traits.h" 3
private: typedef typename __promote< _Tp, std::__is_integer< _Tp> ::__value> ::__type __type1;
# 200 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/ext/type_traits.h" 3
typedef typename __promote< _Up, std::__is_integer< _Up> ::__value> ::__type __type2;
# 201 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/ext/type_traits.h" 3
typedef typename __promote< _Vp, std::__is_integer< _Vp> ::__value> ::__type __type3;
# 202 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/ext/type_traits.h" 3
typedef typename __promote< _Wp, std::__is_integer< _Wp> ::__value> ::__type __type4;
# 205 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/ext/type_traits.h" 3
public: typedef __typeof__((((__type1() + __type2()) + __type3()) + __type4())) __type;
# 206 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/ext/type_traits.h" 3
};
# 208 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/ext/type_traits.h" 3
}
# 82 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
namespace std __attribute__((visibility("default"))) {
# 86 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
template<class _Tp> extern inline _Tp __cmath_power(_Tp, unsigned);
# 89 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
template<class _Tp> inline _Tp
# 91 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
__pow_helper(_Tp __x, int __n)
# 92 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
{
# 93 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
return (__n < 0) ? (((_Tp)(1)) / __cmath_power(__x, -__n)) : (__cmath_power(__x, __n));
# 96 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
}
# 99 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
inline double abs(double __x)
# 100 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
{ return __builtin_fabs(__x); }
# 103 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
inline float abs(float __x)
# 104 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
{ return __builtin_fabsf(__x); }
# 107 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
inline long double abs(long double __x)
# 108 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
{ return __builtin_fabsl(__x); }
# 110 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
using ::acos;
# 113 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
inline float acos(float __x)
# 114 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
{ return __builtin_acosf(__x); }
# 117 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
inline long double acos(long double __x)
# 118 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
{ return __builtin_acosl(__x); }
# 120 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
template<class _Tp> inline typename __gnu_cxx::__enable_if< (__is_integer< _Tp> ::__value), double> ::__type
# 123 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
acos(_Tp __x)
# 124 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
{ return __builtin_acos(__x); }
# 126 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
using ::asin;
# 129 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
inline float asin(float __x)
# 130 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
{ return __builtin_asinf(__x); }
# 133 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
inline long double asin(long double __x)
# 134 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
{ return __builtin_asinl(__x); }
# 136 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
template<class _Tp> inline typename __gnu_cxx::__enable_if< (__is_integer< _Tp> ::__value), double> ::__type
# 139 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
asin(_Tp __x)
# 140 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
{ return __builtin_asin(__x); }
# 142 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
using ::atan;
# 145 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
inline float atan(float __x)
# 146 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
{ return __builtin_atanf(__x); }
# 149 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
inline long double atan(long double __x)
# 150 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
{ return __builtin_atanl(__x); }
# 152 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
template<class _Tp> inline typename __gnu_cxx::__enable_if< (__is_integer< _Tp> ::__value), double> ::__type
# 155 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
atan(_Tp __x)
# 156 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
{ return __builtin_atan(__x); }
# 158 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
using ::atan2;
# 161 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
inline float atan2(float __y, float __x)
# 162 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
{ return __builtin_atan2f(__y, __x); }
# 165 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
inline long double atan2(long double __y, long double __x)
# 166 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
{ return __builtin_atan2l(__y, __x); }
# 168 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
template<class _Tp, class _Up> inline typename __gnu_cxx::__promote_2< typename __gnu_cxx::__enable_if< (__is_arithmetic< _Tp> ::__value && __is_arithmetic< _Up> ::__value), _Tp> ::__type, _Up> ::__type
# 174 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
atan2(_Tp __y, _Up __x)
# 175 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
{
# 176 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type;
# 177 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
return atan2(((__type)(__y)), ((__type)(__x)));
# 178 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
}
# 180 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
using ::ceil;
# 183 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
inline float ceil(float __x)
# 184 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
{ return __builtin_ceilf(__x); }
# 187 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
inline long double ceil(long double __x)
# 188 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
{ return __builtin_ceill(__x); }
# 190 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
template<class _Tp> inline typename __gnu_cxx::__enable_if< (__is_integer< _Tp> ::__value), double> ::__type
# 193 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
ceil(_Tp __x)
# 194 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
{ return __builtin_ceil(__x); }
# 196 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
using ::cos;
# 199 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
inline float cos(float __x)
# 200 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
{ return __builtin_cosf(__x); }
# 203 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
inline long double cos(long double __x)
# 204 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
{ return __builtin_cosl(__x); }
# 206 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
template<class _Tp> inline typename __gnu_cxx::__enable_if< (__is_integer< _Tp> ::__value), double> ::__type
# 209 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
cos(_Tp __x)
# 210 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
{ return __builtin_cos(__x); }
# 212 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
using ::cosh;
# 215 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
inline float cosh(float __x)
# 216 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
{ return __builtin_coshf(__x); }
# 219 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
inline long double cosh(long double __x)
# 220 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
{ return __builtin_coshl(__x); }
# 222 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
template<class _Tp> inline typename __gnu_cxx::__enable_if< (__is_integer< _Tp> ::__value), double> ::__type
# 225 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
cosh(_Tp __x)
# 226 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
{ return __builtin_cosh(__x); }
# 228 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
using ::exp;
# 231 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
inline float exp(float __x)
# 232 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
{ return __builtin_expf(__x); }
# 235 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
inline long double exp(long double __x)
# 236 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
{ return __builtin_expl(__x); }
# 238 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
template<class _Tp> inline typename __gnu_cxx::__enable_if< (__is_integer< _Tp> ::__value), double> ::__type
# 241 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
exp(_Tp __x)
# 242 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
{ return __builtin_exp(__x); }
# 244 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
using ::fabs;
# 247 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
inline float fabs(float __x)
# 248 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
{ return __builtin_fabsf(__x); }
# 251 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
inline long double fabs(long double __x)
# 252 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
{ return __builtin_fabsl(__x); }
# 254 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
template<class _Tp> inline typename __gnu_cxx::__enable_if< (__is_integer< _Tp> ::__value), double> ::__type
# 257 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
fabs(_Tp __x)
# 258 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
{ return __builtin_fabs(__x); }
# 260 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
using ::floor;
# 263 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
inline float floor(float __x)
# 264 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
{ return __builtin_floorf(__x); }
# 267 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
inline long double floor(long double __x)
# 268 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
{ return __builtin_floorl(__x); }
# 270 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
template<class _Tp> inline typename __gnu_cxx::__enable_if< (__is_integer< _Tp> ::__value), double> ::__type
# 273 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
floor(_Tp __x)
# 274 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
{ return __builtin_floor(__x); }
# 276 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
using ::fmod;
# 279 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
inline float fmod(float __x, float __y)
# 280 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
{ return __builtin_fmodf(__x, __y); }
# 283 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
inline long double fmod(long double __x, long double __y)
# 284 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
{ return __builtin_fmodl(__x, __y); }
# 286 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
using ::frexp;
# 289 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
inline float frexp(float __x, int *__exp)
# 290 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
{ return __builtin_frexpf(__x, __exp); }
# 293 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
inline long double frexp(long double __x, int *__exp)
# 294 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
{ return __builtin_frexpl(__x, __exp); }
# 296 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
template<class _Tp> inline typename __gnu_cxx::__enable_if< (__is_integer< _Tp> ::__value), double> ::__type
# 299 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
frexp(_Tp __x, int *__exp)
# 300 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
{ return __builtin_frexp(__x, __exp); }
# 302 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
using ::ldexp;
# 305 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
inline float ldexp(float __x, int __exp)
# 306 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
{ return __builtin_ldexpf(__x, __exp); }
# 309 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
inline long double ldexp(long double __x, int __exp)
# 310 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
{ return __builtin_ldexpl(__x, __exp); }
# 312 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
template<class _Tp> inline typename __gnu_cxx::__enable_if< (__is_integer< _Tp> ::__value), double> ::__type
# 315 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
ldexp(_Tp __x, int __exp)
# 316 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
{ return __builtin_ldexp(__x, __exp); }
# 318 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
using ::log;
# 321 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
inline float log(float __x)
# 322 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
{ return __builtin_logf(__x); }
# 325 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
inline long double log(long double __x)
# 326 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
{ return __builtin_logl(__x); }
# 328 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
template<class _Tp> inline typename __gnu_cxx::__enable_if< (__is_integer< _Tp> ::__value), double> ::__type
# 331 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
log(_Tp __x)
# 332 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
{ return __builtin_log(__x); }
# 334 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
using ::log10;
# 337 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
inline float log10(float __x)
# 338 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
{ return __builtin_log10f(__x); }
# 341 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
inline long double log10(long double __x)
# 342 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
{ return __builtin_log10l(__x); }
# 344 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
template<class _Tp> inline typename __gnu_cxx::__enable_if< (__is_integer< _Tp> ::__value), double> ::__type
# 347 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
log10(_Tp __x)
# 348 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
{ return __builtin_log10(__x); }
# 350 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
using ::modf;
# 353 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
inline float modf(float __x, float *__iptr)
# 354 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
{ return __builtin_modff(__x, __iptr); }
# 357 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
inline long double modf(long double __x, long double *__iptr)
# 358 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
{ return __builtin_modfl(__x, __iptr); }
# 360 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
using ::pow;
# 363 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
inline float pow(float __x, float __y)
# 364 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
{ return __builtin_powf(__x, __y); }
# 367 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
inline long double pow(long double __x, long double __y)
# 368 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
{ return __builtin_powl(__x, __y); }
# 372 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
inline double pow(double __x, int __i)
# 373 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
{ return __builtin_powi(__x, __i); }
# 376 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
inline float pow(float __x, int __n)
# 377 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
{ return __builtin_powif(__x, __n); }
# 380 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
inline long double pow(long double __x, int __n)
# 381 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
{ return __builtin_powil(__x, __n); }
# 383 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
template<class _Tp, class _Up> inline typename __gnu_cxx::__promote_2< typename __gnu_cxx::__enable_if< (__is_arithmetic< _Tp> ::__value && __is_arithmetic< _Up> ::__value), _Tp> ::__type, _Up> ::__type
# 389 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
pow(_Tp __x, _Up __y)
# 390 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
{
# 391 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type;
# 392 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
return pow(((__type)(__x)), ((__type)(__y)));
# 393 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
}
# 395 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
using ::sin;
# 398 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
inline float sin(float __x)
# 399 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
{ return __builtin_sinf(__x); }
# 402 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
inline long double sin(long double __x)
# 403 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
{ return __builtin_sinl(__x); }
# 405 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
template<class _Tp> inline typename __gnu_cxx::__enable_if< (__is_integer< _Tp> ::__value), double> ::__type
# 408 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
sin(_Tp __x)
# 409 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
{ return __builtin_sin(__x); }
# 411 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
using ::sinh;
# 414 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
inline float sinh(float __x)
# 415 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
{ return __builtin_sinhf(__x); }
# 418 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
inline long double sinh(long double __x)
# 419 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
{ return __builtin_sinhl(__x); }
# 421 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
template<class _Tp> inline typename __gnu_cxx::__enable_if< (__is_integer< _Tp> ::__value), double> ::__type
# 424 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
sinh(_Tp __x)
# 425 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
{ return __builtin_sinh(__x); }
# 427 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
using ::sqrt;
# 430 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
inline float sqrt(float __x)
# 431 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
{ return __builtin_sqrtf(__x); }
# 434 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
inline long double sqrt(long double __x)
# 435 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
{ return __builtin_sqrtl(__x); }
# 437 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
template<class _Tp> inline typename __gnu_cxx::__enable_if< (__is_integer< _Tp> ::__value), double> ::__type
# 440 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
sqrt(_Tp __x)
# 441 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
{ return __builtin_sqrt(__x); }
# 443 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
using ::tan;
# 446 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
inline float tan(float __x)
# 447 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
{ return __builtin_tanf(__x); }
# 450 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
inline long double tan(long double __x)
# 451 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
{ return __builtin_tanl(__x); }
# 453 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
template<class _Tp> inline typename __gnu_cxx::__enable_if< (__is_integer< _Tp> ::__value), double> ::__type
# 456 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
tan(_Tp __x)
# 457 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
{ return __builtin_tan(__x); }
# 459 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
using ::tanh;
# 462 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
inline float tanh(float __x)
# 463 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
{ return __builtin_tanhf(__x); }
# 466 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
inline long double tanh(long double __x)
# 467 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
{ return __builtin_tanhl(__x); }
# 469 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
template<class _Tp> inline typename __gnu_cxx::__enable_if< (__is_integer< _Tp> ::__value), double> ::__type
# 472 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
tanh(_Tp __x)
# 473 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
{ return __builtin_tanh(__x); }
# 475 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
}
# 483 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
namespace __gnu_cxx __attribute__((visibility("default"))) {
# 485 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
template<class _Tp> inline int
# 487 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
__capture_fpclassify(_Tp __f) { return (sizeof(__f) == sizeof(float)) ? (__fpclassifyf(__f)) : ((sizeof(__f) == sizeof(double)) ? (__fpclassify(__f)) : (__fpclassifyl(__f))); }
# 489 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
}
# 505 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
namespace std __attribute__((visibility("default"))) {
# 507 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
template<class _Tp> inline typename __gnu_cxx::__enable_if< (__is_arithmetic< _Tp> ::__value), int> ::__type
# 510 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
fpclassify(_Tp __f)
# 511 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
{
# 512 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
typedef typename __gnu_cxx::__promote< _Tp, __is_integer< _Tp> ::__value> ::__type __type;
# 513 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
return __gnu_cxx::__capture_fpclassify(((__type)(__f)));
# 514 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
}
# 516 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
template<class _Tp> inline typename __gnu_cxx::__enable_if< (__is_arithmetic< _Tp> ::__value), int> ::__type
# 519 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
isfinite(_Tp __f)
# 520 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
{
# 521 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
typedef typename __gnu_cxx::__promote< _Tp, __is_integer< _Tp> ::__value> ::__type __type;
# 522 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
return __builtin_isfinite(((__type)(__f)));
# 523 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
}
# 525 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
template<class _Tp> inline typename __gnu_cxx::__enable_if< (__is_arithmetic< _Tp> ::__value), int> ::__type
# 528 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
isinf(_Tp __f)
# 529 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
{
# 530 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
typedef typename __gnu_cxx::__promote< _Tp, __is_integer< _Tp> ::__value> ::__type __type;
# 531 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
return __builtin_isinf(((__type)(__f)));
# 532 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
}
# 534 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
template<class _Tp> inline typename __gnu_cxx::__enable_if< (__is_arithmetic< _Tp> ::__value), int> ::__type
# 537 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
isnan(_Tp __f)
# 538 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
{
# 539 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
typedef typename __gnu_cxx::__promote< _Tp, __is_integer< _Tp> ::__value> ::__type __type;
# 540 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
return __builtin_isnan(((__type)(__f)));
# 541 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
}
# 543 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
template<class _Tp> inline typename __gnu_cxx::__enable_if< (__is_arithmetic< _Tp> ::__value), int> ::__type
# 546 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
isnormal(_Tp __f)
# 547 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
{
# 548 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
typedef typename __gnu_cxx::__promote< _Tp, __is_integer< _Tp> ::__value> ::__type __type;
# 549 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
return __builtin_isnormal(((__type)(__f)));
# 550 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
}
# 552 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
template<class _Tp> inline typename __gnu_cxx::__enable_if< (__is_arithmetic< _Tp> ::__value), int> ::__type
# 555 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
signbit(_Tp __f)
# 556 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
{
# 557 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
typedef typename __gnu_cxx::__promote< _Tp, __is_integer< _Tp> ::__value> ::__type __type;
# 558 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
return __builtin_signbit(((__type)(__f)));
# 559 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
}
# 561 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
template<class _Tp> inline typename __gnu_cxx::__enable_if< (__is_arithmetic< _Tp> ::__value), int> ::__type
# 564 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
isgreater(_Tp __f1, _Tp __f2)
# 565 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
{
# 566 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
typedef typename __gnu_cxx::__promote< _Tp, __is_integer< _Tp> ::__value> ::__type __type;
# 567 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
return __builtin_isgreater(((__type)(__f1)), ((__type)(__f2)));
# 568 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
}
# 570 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
template<class _Tp> inline typename __gnu_cxx::__enable_if< (__is_arithmetic< _Tp> ::__value), int> ::__type
# 573 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
isgreaterequal(_Tp __f1, _Tp __f2)
# 574 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
{
# 575 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
typedef typename __gnu_cxx::__promote< _Tp, __is_integer< _Tp> ::__value> ::__type __type;
# 576 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
return __builtin_isgreaterequal(((__type)(__f1)), ((__type)(__f2)));
# 577 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
}
# 579 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
template<class _Tp> inline typename __gnu_cxx::__enable_if< (__is_arithmetic< _Tp> ::__value), int> ::__type
# 582 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
isless(_Tp __f1, _Tp __f2)
# 583 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
{
# 584 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
typedef typename __gnu_cxx::__promote< _Tp, __is_integer< _Tp> ::__value> ::__type __type;
# 585 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
return __builtin_isless(((__type)(__f1)), ((__type)(__f2)));
# 586 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
}
# 588 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
template<class _Tp> inline typename __gnu_cxx::__enable_if< (__is_arithmetic< _Tp> ::__value), int> ::__type
# 591 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
islessequal(_Tp __f1, _Tp __f2)
# 592 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
{
# 593 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
typedef typename __gnu_cxx::__promote< _Tp, __is_integer< _Tp> ::__value> ::__type __type;
# 594 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
return __builtin_islessequal(((__type)(__f1)), ((__type)(__f2)));
# 595 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
}
# 597 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
template<class _Tp> inline typename __gnu_cxx::__enable_if< (__is_arithmetic< _Tp> ::__value), int> ::__type
# 600 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
islessgreater(_Tp __f1, _Tp __f2)
# 601 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
{
# 602 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
typedef typename __gnu_cxx::__promote< _Tp, __is_integer< _Tp> ::__value> ::__type __type;
# 603 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
return __builtin_islessgreater(((__type)(__f1)), ((__type)(__f2)));
# 604 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
}
# 606 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
template<class _Tp> inline typename __gnu_cxx::__enable_if< (__is_arithmetic< _Tp> ::__value), int> ::__type
# 609 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
isunordered(_Tp __f1, _Tp __f2)
# 610 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
{
# 611 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
typedef typename __gnu_cxx::__promote< _Tp, __is_integer< _Tp> ::__value> ::__type __type;
# 612 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
return __builtin_isunordered(((__type)(__f1)), ((__type)(__f2)));
# 613 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
}
# 615 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cmath" 3
}
# 40 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cmath.tcc" 3
namespace std __attribute__((visibility("default"))) {
# 42 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cmath.tcc" 3
template<class _Tp> inline _Tp
# 44 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cmath.tcc" 3
__cmath_power(_Tp __x, unsigned __n)
# 45 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cmath.tcc" 3
{
# 46 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cmath.tcc" 3
auto _Tp __y = ((__n % (2)) ? __x : ((_Tp)(1)));
# 48 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cmath.tcc" 3
while (__n >>= 1)
# 49 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cmath.tcc" 3
{
# 50 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cmath.tcc" 3
__x = __x * __x;
# 51 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cmath.tcc" 3
if (__n % (2)) {
# 52 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cmath.tcc" 3
__y = __y * __x; }
# 53 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cmath.tcc" 3
}
# 55 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cmath.tcc" 3
return __y;
# 56 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cmath.tcc" 3
}
# 58 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/bits/cmath.tcc" 3
}
# 53 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cstddef" 3
namespace std __attribute__((visibility("default"))) {
# 55 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cstddef" 3
using ::ptrdiff_t;
# 56 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cstddef" 3
using ::size_t;
# 58 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cstddef" 3
}
# 105 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cstdlib" 3
namespace std __attribute__((visibility("default"))) {
# 107 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cstdlib" 3
using ::div_t;
# 108 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cstdlib" 3
using ::ldiv_t;
# 110 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cstdlib" 3
using ::abort;
# 111 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cstdlib" 3
using ::abs;
# 112 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cstdlib" 3
using ::atexit;
# 113 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cstdlib" 3
using ::atof;
# 114 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cstdlib" 3
using ::atoi;
# 115 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cstdlib" 3
using ::atol;
# 116 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cstdlib" 3
using ::bsearch;
# 117 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cstdlib" 3
using ::calloc;
# 118 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cstdlib" 3
using ::div;
# 119 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cstdlib" 3
using ::exit;
# 120 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cstdlib" 3
using ::free;
# 121 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cstdlib" 3
using ::getenv;
# 122 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cstdlib" 3
using ::labs;
# 123 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cstdlib" 3
using ::ldiv;
# 124 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cstdlib" 3
using ::malloc;
# 126 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cstdlib" 3
using ::mblen;
# 127 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cstdlib" 3
using ::mbstowcs;
# 128 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cstdlib" 3
using ::mbtowc;
# 130 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cstdlib" 3
using ::qsort;
# 131 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cstdlib" 3
using ::rand;
# 132 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cstdlib" 3
using ::realloc;
# 133 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cstdlib" 3
using ::srand;
# 134 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cstdlib" 3
using ::strtod;
# 135 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cstdlib" 3
using ::strtol;
# 136 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cstdlib" 3
using ::strtoul;
# 137 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cstdlib" 3
using ::system;
# 139 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cstdlib" 3
using ::wcstombs;
# 140 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cstdlib" 3
using ::wctomb;
# 144 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cstdlib" 3
inline long abs(long __i) { return labs(__i); }
# 147 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cstdlib" 3
inline ldiv_t div(long __i, long __j) { return ldiv(__i, __j); }
# 149 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cstdlib" 3
}
# 162 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cstdlib" 3
namespace __gnu_cxx __attribute__((visibility("default"))) {
# 165 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cstdlib" 3
using ::lldiv_t;
# 171 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cstdlib" 3
using ::_Exit;
# 175 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cstdlib" 3
inline long long abs(long long __x) { return (__x >= (0)) ? __x : (-__x); }
# 178 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cstdlib" 3
using ::llabs;
# 181 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cstdlib" 3
inline lldiv_t div(long long __n, long long __d)
# 182 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cstdlib" 3
{ auto lldiv_t __q; (__q.quot) = __n / __d; (__q.rem) = __n % __d; return __q; }
# 184 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cstdlib" 3
using ::lldiv;
# 195 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cstdlib" 3
using ::atoll;
# 196 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cstdlib" 3
using ::strtoll;
# 197 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cstdlib" 3
using ::strtoull;
# 199 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cstdlib" 3
using ::strtof;
# 200 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cstdlib" 3
using ::strtold;
# 202 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cstdlib" 3
}
# 204 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cstdlib" 3
namespace std __attribute__((visibility("default"))) {
# 207 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cstdlib" 3
using __gnu_cxx::lldiv_t;
# 209 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cstdlib" 3
using __gnu_cxx::_Exit;
# 210 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cstdlib" 3
using __gnu_cxx::abs;
# 212 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cstdlib" 3
using __gnu_cxx::llabs;
# 213 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cstdlib" 3
using __gnu_cxx::div;
# 214 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cstdlib" 3
using __gnu_cxx::lldiv;
# 216 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cstdlib" 3
using __gnu_cxx::atoll;
# 217 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cstdlib" 3
using __gnu_cxx::strtof;
# 218 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cstdlib" 3
using __gnu_cxx::strtoll;
# 219 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cstdlib" 3
using __gnu_cxx::strtoull;
# 220 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cstdlib" 3
using __gnu_cxx::strtold;
# 222 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/../../../../include/c++/4.3.2/cstdlib" 3
}
# 424 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) int __signbitl(long double) throw() __attribute__((__const__));
# 426 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) int __isinfl(long double) throw() __attribute__((__const__));
# 428 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) int __isnanl(long double) throw() __attribute__((__const__));
# 438 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((__weak__)) int __finitel(long double) throw() __attribute__((__const__));
# 463 "/usr/local/cuda/bin/../include/math_functions.h"
namespace __gnu_cxx {
# 465 "/usr/local/cuda/bin/../include/math_functions.h"
extern inline long long abs(long long) __attribute__((visibility("default")));
# 466 "/usr/local/cuda/bin/../include/math_functions.h"
}
# 468 "/usr/local/cuda/bin/../include/math_functions.h"
namespace std {
# 470 "/usr/local/cuda/bin/../include/math_functions.h"
template<class T> extern inline T __pow_helper(T, int);
# 471 "/usr/local/cuda/bin/../include/math_functions.h"
template<class T> extern inline T __cmath_power(T, unsigned);
# 472 "/usr/local/cuda/bin/../include/math_functions.h"
}
# 474 "/usr/local/cuda/bin/../include/math_functions.h"
using std::abs;
# 475 "/usr/local/cuda/bin/../include/math_functions.h"
using std::fabs;
# 476 "/usr/local/cuda/bin/../include/math_functions.h"
using std::ceil;
# 477 "/usr/local/cuda/bin/../include/math_functions.h"
using std::floor;
# 478 "/usr/local/cuda/bin/../include/math_functions.h"
using std::sqrt;
# 479 "/usr/local/cuda/bin/../include/math_functions.h"
using std::pow;
# 480 "/usr/local/cuda/bin/../include/math_functions.h"
using std::log;
# 481 "/usr/local/cuda/bin/../include/math_functions.h"
using std::log10;
# 482 "/usr/local/cuda/bin/../include/math_functions.h"
using std::fmod;
# 483 "/usr/local/cuda/bin/../include/math_functions.h"
using std::modf;
# 484 "/usr/local/cuda/bin/../include/math_functions.h"
using std::exp;
# 485 "/usr/local/cuda/bin/../include/math_functions.h"
using std::frexp;
# 486 "/usr/local/cuda/bin/../include/math_functions.h"
using std::ldexp;
# 487 "/usr/local/cuda/bin/../include/math_functions.h"
using std::asin;
# 488 "/usr/local/cuda/bin/../include/math_functions.h"
using std::sin;
# 489 "/usr/local/cuda/bin/../include/math_functions.h"
using std::sinh;
# 490 "/usr/local/cuda/bin/../include/math_functions.h"
using std::acos;
# 491 "/usr/local/cuda/bin/../include/math_functions.h"
using std::cos;
# 492 "/usr/local/cuda/bin/../include/math_functions.h"
using std::cosh;
# 493 "/usr/local/cuda/bin/../include/math_functions.h"
using std::atan;
# 494 "/usr/local/cuda/bin/../include/math_functions.h"
using std::atan2;
# 495 "/usr/local/cuda/bin/../include/math_functions.h"
using std::tan;
# 496 "/usr/local/cuda/bin/../include/math_functions.h"
using std::tanh;
# 550 "/usr/local/cuda/bin/../include/math_functions.h"
namespace std {
# 553 "/usr/local/cuda/bin/../include/math_functions.h"
extern inline long abs(long) __attribute__((visibility("default")));
# 554 "/usr/local/cuda/bin/../include/math_functions.h"
extern inline float abs(float) __attribute__((visibility("default")));
# 555 "/usr/local/cuda/bin/../include/math_functions.h"
extern inline double abs(double) __attribute__((visibility("default")));
# 556 "/usr/local/cuda/bin/../include/math_functions.h"
extern inline float fabs(float) __attribute__((visibility("default")));
# 557 "/usr/local/cuda/bin/../include/math_functions.h"
extern inline float ceil(float) __attribute__((visibility("default")));
# 558 "/usr/local/cuda/bin/../include/math_functions.h"
extern inline float floor(float) __attribute__((visibility("default")));
# 559 "/usr/local/cuda/bin/../include/math_functions.h"
extern inline float sqrt(float) __attribute__((visibility("default")));
# 560 "/usr/local/cuda/bin/../include/math_functions.h"
extern inline float pow(float, float) __attribute__((visibility("default")));
# 561 "/usr/local/cuda/bin/../include/math_functions.h"
extern inline float pow(float, int) __attribute__((visibility("default")));
# 562 "/usr/local/cuda/bin/../include/math_functions.h"
extern inline double pow(double, int) __attribute__((visibility("default")));
# 563 "/usr/local/cuda/bin/../include/math_functions.h"
extern inline float log(float) __attribute__((visibility("default")));
# 564 "/usr/local/cuda/bin/../include/math_functions.h"
extern inline float log10(float) __attribute__((visibility("default")));
# 565 "/usr/local/cuda/bin/../include/math_functions.h"
extern inline float fmod(float, float) __attribute__((visibility("default")));
# 566 "/usr/local/cuda/bin/../include/math_functions.h"
extern inline float modf(float, float *) __attribute__((visibility("default")));
# 567 "/usr/local/cuda/bin/../include/math_functions.h"
extern inline float exp(float) __attribute__((visibility("default")));
# 568 "/usr/local/cuda/bin/../include/math_functions.h"
extern inline float frexp(float, int *) __attribute__((visibility("default")));
# 569 "/usr/local/cuda/bin/../include/math_functions.h"
extern inline float ldexp(float, int) __attribute__((visibility("default")));
# 570 "/usr/local/cuda/bin/../include/math_functions.h"
extern inline float asin(float) __attribute__((visibility("default")));
# 571 "/usr/local/cuda/bin/../include/math_functions.h"
extern inline float sin(float) __attribute__((visibility("default")));
# 572 "/usr/local/cuda/bin/../include/math_functions.h"
extern inline float sinh(float) __attribute__((visibility("default")));
# 573 "/usr/local/cuda/bin/../include/math_functions.h"
extern inline float acos(float) __attribute__((visibility("default")));
# 574 "/usr/local/cuda/bin/../include/math_functions.h"
extern inline float cos(float) __attribute__((visibility("default")));
# 575 "/usr/local/cuda/bin/../include/math_functions.h"
extern inline float cosh(float) __attribute__((visibility("default")));
# 576 "/usr/local/cuda/bin/../include/math_functions.h"
extern inline float atan(float) __attribute__((visibility("default")));
# 577 "/usr/local/cuda/bin/../include/math_functions.h"
extern inline float atan2(float, float) __attribute__((visibility("default")));
# 578 "/usr/local/cuda/bin/../include/math_functions.h"
extern inline float tan(float) __attribute__((visibility("default")));
# 579 "/usr/local/cuda/bin/../include/math_functions.h"
extern inline float tanh(float) __attribute__((visibility("default")));
# 582 "/usr/local/cuda/bin/../include/math_functions.h"
}
# 585 "/usr/local/cuda/bin/../include/math_functions.h"
static inline float logb(float a)
# 586 "/usr/local/cuda/bin/../include/math_functions.h"
{
# 587 "/usr/local/cuda/bin/../include/math_functions.h"
return logbf(a);
# 588 "/usr/local/cuda/bin/../include/math_functions.h"
}
# 590 "/usr/local/cuda/bin/../include/math_functions.h"
static inline int ilogb(float a)
# 591 "/usr/local/cuda/bin/../include/math_functions.h"
{
# 592 "/usr/local/cuda/bin/../include/math_functions.h"
return ilogbf(a);
# 593 "/usr/local/cuda/bin/../include/math_functions.h"
}
# 595 "/usr/local/cuda/bin/../include/math_functions.h"
static inline float scalbn(float a, int b)
# 596 "/usr/local/cuda/bin/../include/math_functions.h"
{
# 597 "/usr/local/cuda/bin/../include/math_functions.h"
return scalbnf(a, b);
# 598 "/usr/local/cuda/bin/../include/math_functions.h"
}
# 600 "/usr/local/cuda/bin/../include/math_functions.h"
static inline float scalbln(float a, long b)
# 601 "/usr/local/cuda/bin/../include/math_functions.h"
{
# 602 "/usr/local/cuda/bin/../include/math_functions.h"
return scalblnf(a, b);
# 603 "/usr/local/cuda/bin/../include/math_functions.h"
}
# 605 "/usr/local/cuda/bin/../include/math_functions.h"
static inline float exp2(float a)
# 606 "/usr/local/cuda/bin/../include/math_functions.h"
{
# 607 "/usr/local/cuda/bin/../include/math_functions.h"
return exp2f(a);
# 608 "/usr/local/cuda/bin/../include/math_functions.h"
}
# 610 "/usr/local/cuda/bin/../include/math_functions.h"
static inline float exp10(float a)
# 611 "/usr/local/cuda/bin/../include/math_functions.h"
{
# 612 "/usr/local/cuda/bin/../include/math_functions.h"
return exp10f(a);
# 613 "/usr/local/cuda/bin/../include/math_functions.h"
}
# 615 "/usr/local/cuda/bin/../include/math_functions.h"
static inline float expm1(float a)
# 616 "/usr/local/cuda/bin/../include/math_functions.h"
{
# 617 "/usr/local/cuda/bin/../include/math_functions.h"
return expm1f(a);
# 618 "/usr/local/cuda/bin/../include/math_functions.h"
}
# 620 "/usr/local/cuda/bin/../include/math_functions.h"
static inline float log2(float a)
# 621 "/usr/local/cuda/bin/../include/math_functions.h"
{
# 622 "/usr/local/cuda/bin/../include/math_functions.h"
return log2f(a);
# 623 "/usr/local/cuda/bin/../include/math_functions.h"
}
# 625 "/usr/local/cuda/bin/../include/math_functions.h"
static inline float log1p(float a)
# 626 "/usr/local/cuda/bin/../include/math_functions.h"
{
# 627 "/usr/local/cuda/bin/../include/math_functions.h"
return log1pf(a);
# 628 "/usr/local/cuda/bin/../include/math_functions.h"
}
# 630 "/usr/local/cuda/bin/../include/math_functions.h"
static inline float rsqrt(float a)
# 631 "/usr/local/cuda/bin/../include/math_functions.h"
{
# 632 "/usr/local/cuda/bin/../include/math_functions.h"
return rsqrtf(a);
# 633 "/usr/local/cuda/bin/../include/math_functions.h"
}
# 635 "/usr/local/cuda/bin/../include/math_functions.h"
static inline float acosh(float a)
# 636 "/usr/local/cuda/bin/../include/math_functions.h"
{
# 637 "/usr/local/cuda/bin/../include/math_functions.h"
return acoshf(a);
# 638 "/usr/local/cuda/bin/../include/math_functions.h"
}
# 640 "/usr/local/cuda/bin/../include/math_functions.h"
static inline float asinh(float a)
# 641 "/usr/local/cuda/bin/../include/math_functions.h"
{
# 642 "/usr/local/cuda/bin/../include/math_functions.h"
return asinhf(a);
# 643 "/usr/local/cuda/bin/../include/math_functions.h"
}
# 645 "/usr/local/cuda/bin/../include/math_functions.h"
static inline float atanh(float a)
# 646 "/usr/local/cuda/bin/../include/math_functions.h"
{
# 647 "/usr/local/cuda/bin/../include/math_functions.h"
return atanhf(a);
# 648 "/usr/local/cuda/bin/../include/math_functions.h"
}
# 650 "/usr/local/cuda/bin/../include/math_functions.h"
static inline float hypot(float a, float b)
# 651 "/usr/local/cuda/bin/../include/math_functions.h"
{
# 652 "/usr/local/cuda/bin/../include/math_functions.h"
return hypotf(a, b);
# 653 "/usr/local/cuda/bin/../include/math_functions.h"
}
# 655 "/usr/local/cuda/bin/../include/math_functions.h"
static inline float cbrt(float a)
# 656 "/usr/local/cuda/bin/../include/math_functions.h"
{
# 657 "/usr/local/cuda/bin/../include/math_functions.h"
return cbrtf(a);
# 658 "/usr/local/cuda/bin/../include/math_functions.h"
}
# 660 "/usr/local/cuda/bin/../include/math_functions.h"
static inline void sincos(float a, float *sptr, float *cptr)
# 661 "/usr/local/cuda/bin/../include/math_functions.h"
{
# 662 "/usr/local/cuda/bin/../include/math_functions.h"
sincosf(a, sptr, cptr);
# 663 "/usr/local/cuda/bin/../include/math_functions.h"
}
# 665 "/usr/local/cuda/bin/../include/math_functions.h"
static inline float erf(float a)
# 666 "/usr/local/cuda/bin/../include/math_functions.h"
{
# 667 "/usr/local/cuda/bin/../include/math_functions.h"
return erff(a);
# 668 "/usr/local/cuda/bin/../include/math_functions.h"
}
# 670 "/usr/local/cuda/bin/../include/math_functions.h"
static inline float erfc(float a)
# 671 "/usr/local/cuda/bin/../include/math_functions.h"
{
# 672 "/usr/local/cuda/bin/../include/math_functions.h"
return erfcf(a);
# 673 "/usr/local/cuda/bin/../include/math_functions.h"
}
# 675 "/usr/local/cuda/bin/../include/math_functions.h"
static inline float lgamma(float a)
# 676 "/usr/local/cuda/bin/../include/math_functions.h"
{
# 677 "/usr/local/cuda/bin/../include/math_functions.h"
return lgammaf(a);
# 678 "/usr/local/cuda/bin/../include/math_functions.h"
}
# 680 "/usr/local/cuda/bin/../include/math_functions.h"
static inline float tgamma(float a)
# 681 "/usr/local/cuda/bin/../include/math_functions.h"
{
# 682 "/usr/local/cuda/bin/../include/math_functions.h"
return tgammaf(a);
# 683 "/usr/local/cuda/bin/../include/math_functions.h"
}
# 685 "/usr/local/cuda/bin/../include/math_functions.h"
static inline float copysign(float a, float b)
# 686 "/usr/local/cuda/bin/../include/math_functions.h"
{
# 687 "/usr/local/cuda/bin/../include/math_functions.h"
return copysignf(a, b);
# 688 "/usr/local/cuda/bin/../include/math_functions.h"
}
# 690 "/usr/local/cuda/bin/../include/math_functions.h"
static inline double copysign(double a, float b)
# 691 "/usr/local/cuda/bin/../include/math_functions.h"
{
# 692 "/usr/local/cuda/bin/../include/math_functions.h"
return copysign(a, (double)b);
# 693 "/usr/local/cuda/bin/../include/math_functions.h"
}
# 695 "/usr/local/cuda/bin/../include/math_functions.h"
static inline float copysign(float a, double b)
# 696 "/usr/local/cuda/bin/../include/math_functions.h"
{
# 697 "/usr/local/cuda/bin/../include/math_functions.h"
return copysignf(a, (float)b);
# 698 "/usr/local/cuda/bin/../include/math_functions.h"
}
# 700 "/usr/local/cuda/bin/../include/math_functions.h"
static inline float nextafter(float a, float b)
# 701 "/usr/local/cuda/bin/../include/math_functions.h"
{
# 702 "/usr/local/cuda/bin/../include/math_functions.h"
return nextafterf(a, b);
# 703 "/usr/local/cuda/bin/../include/math_functions.h"
}
# 705 "/usr/local/cuda/bin/../include/math_functions.h"
static inline float remainder(float a, float b)
# 706 "/usr/local/cuda/bin/../include/math_functions.h"
{
# 707 "/usr/local/cuda/bin/../include/math_functions.h"
return remainderf(a, b);
# 708 "/usr/local/cuda/bin/../include/math_functions.h"
}
# 710 "/usr/local/cuda/bin/../include/math_functions.h"
static inline float remquo(float a, float b, int *quo)
# 711 "/usr/local/cuda/bin/../include/math_functions.h"
{
# 712 "/usr/local/cuda/bin/../include/math_functions.h"
return remquof(a, b, quo);
# 713 "/usr/local/cuda/bin/../include/math_functions.h"
}
# 715 "/usr/local/cuda/bin/../include/math_functions.h"
static inline float round(float a)
# 716 "/usr/local/cuda/bin/../include/math_functions.h"
{
# 717 "/usr/local/cuda/bin/../include/math_functions.h"
return roundf(a);
# 718 "/usr/local/cuda/bin/../include/math_functions.h"
}
# 720 "/usr/local/cuda/bin/../include/math_functions.h"
static inline long lround(float a)
# 721 "/usr/local/cuda/bin/../include/math_functions.h"
{
# 722 "/usr/local/cuda/bin/../include/math_functions.h"
return lroundf(a);
# 723 "/usr/local/cuda/bin/../include/math_functions.h"
}
# 725 "/usr/local/cuda/bin/../include/math_functions.h"
static inline long long llround(float a)
# 726 "/usr/local/cuda/bin/../include/math_functions.h"
{
# 727 "/usr/local/cuda/bin/../include/math_functions.h"
return llroundf(a);
# 728 "/usr/local/cuda/bin/../include/math_functions.h"
}
# 730 "/usr/local/cuda/bin/../include/math_functions.h"
static inline float trunc(float a)
# 731 "/usr/local/cuda/bin/../include/math_functions.h"
{
# 732 "/usr/local/cuda/bin/../include/math_functions.h"
return truncf(a);
# 733 "/usr/local/cuda/bin/../include/math_functions.h"
}
# 735 "/usr/local/cuda/bin/../include/math_functions.h"
static inline float rint(float a)
# 736 "/usr/local/cuda/bin/../include/math_functions.h"
{
# 737 "/usr/local/cuda/bin/../include/math_functions.h"
return rintf(a);
# 738 "/usr/local/cuda/bin/../include/math_functions.h"
}
# 740 "/usr/local/cuda/bin/../include/math_functions.h"
static inline long lrint(float a)
# 741 "/usr/local/cuda/bin/../include/math_functions.h"
{
# 742 "/usr/local/cuda/bin/../include/math_functions.h"
return lrintf(a);
# 743 "/usr/local/cuda/bin/../include/math_functions.h"
}
# 745 "/usr/local/cuda/bin/../include/math_functions.h"
static inline long long llrint(float a)
# 746 "/usr/local/cuda/bin/../include/math_functions.h"
{
# 747 "/usr/local/cuda/bin/../include/math_functions.h"
return llrintf(a);
# 748 "/usr/local/cuda/bin/../include/math_functions.h"
}
# 750 "/usr/local/cuda/bin/../include/math_functions.h"
static inline float nearbyint(float a)
# 751 "/usr/local/cuda/bin/../include/math_functions.h"
{
# 752 "/usr/local/cuda/bin/../include/math_functions.h"
return nearbyintf(a);
# 753 "/usr/local/cuda/bin/../include/math_functions.h"
}
# 755 "/usr/local/cuda/bin/../include/math_functions.h"
static inline float fdim(float a, float b)
# 756 "/usr/local/cuda/bin/../include/math_functions.h"
{
# 757 "/usr/local/cuda/bin/../include/math_functions.h"
return fdimf(a, b);
# 758 "/usr/local/cuda/bin/../include/math_functions.h"
}
# 760 "/usr/local/cuda/bin/../include/math_functions.h"
static inline float fma(float a, float b, float c)
# 761 "/usr/local/cuda/bin/../include/math_functions.h"
{
# 762 "/usr/local/cuda/bin/../include/math_functions.h"
return fmaf(a, b, c);
# 763 "/usr/local/cuda/bin/../include/math_functions.h"
}
# 765 "/usr/local/cuda/bin/../include/math_functions.h"
static inline unsigned min(unsigned a, unsigned b)
# 766 "/usr/local/cuda/bin/../include/math_functions.h"
{
# 767 "/usr/local/cuda/bin/../include/math_functions.h"
return umin(a, b);
# 768 "/usr/local/cuda/bin/../include/math_functions.h"
}
# 770 "/usr/local/cuda/bin/../include/math_functions.h"
static inline unsigned min(int a, unsigned b)
# 771 "/usr/local/cuda/bin/../include/math_functions.h"
{
# 772 "/usr/local/cuda/bin/../include/math_functions.h"
return umin((unsigned)a, b);
# 773 "/usr/local/cuda/bin/../include/math_functions.h"
}
# 775 "/usr/local/cuda/bin/../include/math_functions.h"
static inline unsigned min(unsigned a, int b)
# 776 "/usr/local/cuda/bin/../include/math_functions.h"
{
# 777 "/usr/local/cuda/bin/../include/math_functions.h"
return umin(a, (unsigned)b);
# 778 "/usr/local/cuda/bin/../include/math_functions.h"
}
# 780 "/usr/local/cuda/bin/../include/math_functions.h"
static inline float min(float a, float b)
# 781 "/usr/local/cuda/bin/../include/math_functions.h"
{
# 782 "/usr/local/cuda/bin/../include/math_functions.h"
return fminf(a, b);
# 783 "/usr/local/cuda/bin/../include/math_functions.h"
}
# 785 "/usr/local/cuda/bin/../include/math_functions.h"
static inline double min(double a, double b)
# 786 "/usr/local/cuda/bin/../include/math_functions.h"
{
# 787 "/usr/local/cuda/bin/../include/math_functions.h"
return fmin(a, b);
# 788 "/usr/local/cuda/bin/../include/math_functions.h"
}
# 790 "/usr/local/cuda/bin/../include/math_functions.h"
static inline double min(float a, double b)
# 791 "/usr/local/cuda/bin/../include/math_functions.h"
{
# 792 "/usr/local/cuda/bin/../include/math_functions.h"
return fmin((double)a, b);
# 793 "/usr/local/cuda/bin/../include/math_functions.h"
}
# 795 "/usr/local/cuda/bin/../include/math_functions.h"
static inline double min(double a, float b)
# 796 "/usr/local/cuda/bin/../include/math_functions.h"
{
# 797 "/usr/local/cuda/bin/../include/math_functions.h"
return fmin(a, (double)b);
# 798 "/usr/local/cuda/bin/../include/math_functions.h"
}
# 800 "/usr/local/cuda/bin/../include/math_functions.h"
static inline unsigned max(unsigned a, unsigned b)
# 801 "/usr/local/cuda/bin/../include/math_functions.h"
{
# 802 "/usr/local/cuda/bin/../include/math_functions.h"
return umax(a, b);
# 803 "/usr/local/cuda/bin/../include/math_functions.h"
}
# 805 "/usr/local/cuda/bin/../include/math_functions.h"
static inline unsigned max(int a, unsigned b)
# 806 "/usr/local/cuda/bin/../include/math_functions.h"
{
# 807 "/usr/local/cuda/bin/../include/math_functions.h"
return umax((unsigned)a, b);
# 808 "/usr/local/cuda/bin/../include/math_functions.h"
}
# 810 "/usr/local/cuda/bin/../include/math_functions.h"
static inline unsigned max(unsigned a, int b)
# 811 "/usr/local/cuda/bin/../include/math_functions.h"
{
# 812 "/usr/local/cuda/bin/../include/math_functions.h"
return umax(a, (unsigned)b);
# 813 "/usr/local/cuda/bin/../include/math_functions.h"
}
# 815 "/usr/local/cuda/bin/../include/math_functions.h"
static inline float max(float a, float b)
# 816 "/usr/local/cuda/bin/../include/math_functions.h"
{
# 817 "/usr/local/cuda/bin/../include/math_functions.h"
return fmaxf(a, b);
# 818 "/usr/local/cuda/bin/../include/math_functions.h"
}
# 820 "/usr/local/cuda/bin/../include/math_functions.h"
static inline double max(double a, double b)
# 821 "/usr/local/cuda/bin/../include/math_functions.h"
{
# 822 "/usr/local/cuda/bin/../include/math_functions.h"
return fmax(a, b);
# 823 "/usr/local/cuda/bin/../include/math_functions.h"
}
# 825 "/usr/local/cuda/bin/../include/math_functions.h"
static inline double max(float a, double b)
# 826 "/usr/local/cuda/bin/../include/math_functions.h"
{
# 827 "/usr/local/cuda/bin/../include/math_functions.h"
return fmax((double)a, b);
# 828 "/usr/local/cuda/bin/../include/math_functions.h"
}
# 830 "/usr/local/cuda/bin/../include/math_functions.h"
static inline double max(double a, float b)
# 831 "/usr/local/cuda/bin/../include/math_functions.h"
{
# 832 "/usr/local/cuda/bin/../include/math_functions.h"
return fmax(a, (double)b);
# 833 "/usr/local/cuda/bin/../include/math_functions.h"
}
# 59 "/usr/local/cuda/bin/../include/cuda_texture_types.h"
template<class T, int dim = 1, cudaTextureReadMode = cudaReadModeElementType>
# 60 "/usr/local/cuda/bin/../include/cuda_texture_types.h"
struct texture : public textureReference {
# 62 "/usr/local/cuda/bin/../include/cuda_texture_types.h"
texture(int norm = 0, cudaTextureFilterMode
# 63 "/usr/local/cuda/bin/../include/cuda_texture_types.h"
fMode = cudaFilterModePoint, cudaTextureAddressMode
# 64 "/usr/local/cuda/bin/../include/cuda_texture_types.h"
aMode = cudaAddressModeClamp)
# 65 "/usr/local/cuda/bin/../include/cuda_texture_types.h"
{
# 66 "/usr/local/cuda/bin/../include/cuda_texture_types.h"
(this->normalized) = norm;
# 67 "/usr/local/cuda/bin/../include/cuda_texture_types.h"
(this->filterMode) = fMode;
# 68 "/usr/local/cuda/bin/../include/cuda_texture_types.h"
((this->addressMode)[0]) = aMode;
# 69 "/usr/local/cuda/bin/../include/cuda_texture_types.h"
((this->addressMode)[1]) = aMode;
# 70 "/usr/local/cuda/bin/../include/cuda_texture_types.h"
((this->addressMode)[2]) = aMode;
# 71 "/usr/local/cuda/bin/../include/cuda_texture_types.h"
(this->channelDesc) = cudaCreateChannelDesc< T> ();
# 72 "/usr/local/cuda/bin/../include/cuda_texture_types.h"
}
# 74 "/usr/local/cuda/bin/../include/cuda_texture_types.h"
texture(int norm, cudaTextureFilterMode
# 75 "/usr/local/cuda/bin/../include/cuda_texture_types.h"
fMode, cudaTextureAddressMode
# 76 "/usr/local/cuda/bin/../include/cuda_texture_types.h"
aMode, cudaChannelFormatDesc
# 77 "/usr/local/cuda/bin/../include/cuda_texture_types.h"
desc)
# 78 "/usr/local/cuda/bin/../include/cuda_texture_types.h"
{
# 79 "/usr/local/cuda/bin/../include/cuda_texture_types.h"
(this->normalized) = norm;
# 80 "/usr/local/cuda/bin/../include/cuda_texture_types.h"
(this->filterMode) = fMode;
# 81 "/usr/local/cuda/bin/../include/cuda_texture_types.h"
((this->addressMode)[0]) = aMode;
# 82 "/usr/local/cuda/bin/../include/cuda_texture_types.h"
((this->addressMode)[1]) = aMode;
# 83 "/usr/local/cuda/bin/../include/cuda_texture_types.h"
((this->addressMode)[2]) = aMode;
# 84 "/usr/local/cuda/bin/../include/cuda_texture_types.h"
(this->channelDesc) = desc;
# 85 "/usr/local/cuda/bin/../include/cuda_texture_types.h"
}
# 86 "/usr/local/cuda/bin/../include/cuda_texture_types.h"
};
# 77 "/usr/local/cuda/bin/../include/cuda_runtime.h"
template<class T> inline cudaError_t
# 78 "/usr/local/cuda/bin/../include/cuda_runtime.h"
cudaSetupArgument(T
# 79 "/usr/local/cuda/bin/../include/cuda_runtime.h"
arg, size_t
# 80 "/usr/local/cuda/bin/../include/cuda_runtime.h"
offset)
# 82 "/usr/local/cuda/bin/../include/cuda_runtime.h"
{
# 83 "/usr/local/cuda/bin/../include/cuda_runtime.h"
return cudaSetupArgument((const void *)(&arg), sizeof(T), offset);
# 84 "/usr/local/cuda/bin/../include/cuda_runtime.h"
}
# 94 "/usr/local/cuda/bin/../include/cuda_runtime.h"
static inline cudaError_t cudaMemcpyToSymbol(char *
# 95 "/usr/local/cuda/bin/../include/cuda_runtime.h"
symbol, const void *
# 96 "/usr/local/cuda/bin/../include/cuda_runtime.h"
src, size_t
# 97 "/usr/local/cuda/bin/../include/cuda_runtime.h"
count, size_t
# 98 "/usr/local/cuda/bin/../include/cuda_runtime.h"
offset = (0), cudaMemcpyKind
# 99 "/usr/local/cuda/bin/../include/cuda_runtime.h"
kind = cudaMemcpyHostToDevice)
# 101 "/usr/local/cuda/bin/../include/cuda_runtime.h"
{
# 102 "/usr/local/cuda/bin/../include/cuda_runtime.h"
return cudaMemcpyToSymbol((const char *)symbol, src, count, offset, kind);
# 103 "/usr/local/cuda/bin/../include/cuda_runtime.h"
}
# 105 "/usr/local/cuda/bin/../include/cuda_runtime.h"
template<class T> inline cudaError_t
# 106 "/usr/local/cuda/bin/../include/cuda_runtime.h"
cudaMemcpyToSymbol(const T &
# 107 "/usr/local/cuda/bin/../include/cuda_runtime.h"
symbol, const void *
# 108 "/usr/local/cuda/bin/../include/cuda_runtime.h"
src, size_t
# 109 "/usr/local/cuda/bin/../include/cuda_runtime.h"
count, size_t
# 110 "/usr/local/cuda/bin/../include/cuda_runtime.h"
offset = (0), cudaMemcpyKind
# 111 "/usr/local/cuda/bin/../include/cuda_runtime.h"
kind = cudaMemcpyHostToDevice)
# 113 "/usr/local/cuda/bin/../include/cuda_runtime.h"
{
# 114 "/usr/local/cuda/bin/../include/cuda_runtime.h"
return cudaMemcpyToSymbol((const char *)(&symbol), src, count, offset, kind);
# 115 "/usr/local/cuda/bin/../include/cuda_runtime.h"
}
# 117 "/usr/local/cuda/bin/../include/cuda_runtime.h"
static inline cudaError_t cudaMemcpyToSymbolAsync(char *
# 118 "/usr/local/cuda/bin/../include/cuda_runtime.h"
symbol, const void *
# 119 "/usr/local/cuda/bin/../include/cuda_runtime.h"
src, size_t
# 120 "/usr/local/cuda/bin/../include/cuda_runtime.h"
count, size_t
# 121 "/usr/local/cuda/bin/../include/cuda_runtime.h"
offset, cudaMemcpyKind
# 122 "/usr/local/cuda/bin/../include/cuda_runtime.h"
kind, cudaStream_t
# 123 "/usr/local/cuda/bin/../include/cuda_runtime.h"
stream)
# 125 "/usr/local/cuda/bin/../include/cuda_runtime.h"
{
# 126 "/usr/local/cuda/bin/../include/cuda_runtime.h"
return cudaMemcpyToSymbolAsync((const char *)symbol, src, count, offset, kind, stream);
# 127 "/usr/local/cuda/bin/../include/cuda_runtime.h"
}
# 129 "/usr/local/cuda/bin/../include/cuda_runtime.h"
template<class T> inline cudaError_t
# 130 "/usr/local/cuda/bin/../include/cuda_runtime.h"
cudaMemcpyToSymbolAsync(const T &
# 131 "/usr/local/cuda/bin/../include/cuda_runtime.h"
symbol, const void *
# 132 "/usr/local/cuda/bin/../include/cuda_runtime.h"
src, size_t
# 133 "/usr/local/cuda/bin/../include/cuda_runtime.h"
count, size_t
# 134 "/usr/local/cuda/bin/../include/cuda_runtime.h"
offset, cudaMemcpyKind
# 135 "/usr/local/cuda/bin/../include/cuda_runtime.h"
kind, cudaStream_t
# 136 "/usr/local/cuda/bin/../include/cuda_runtime.h"
stream)
# 138 "/usr/local/cuda/bin/../include/cuda_runtime.h"
{
# 139 "/usr/local/cuda/bin/../include/cuda_runtime.h"
return cudaMemcpyToSymbolAsync((const char *)(&symbol), src, count, offset, kind, stream);
# 140 "/usr/local/cuda/bin/../include/cuda_runtime.h"
}
# 148 "/usr/local/cuda/bin/../include/cuda_runtime.h"
static inline cudaError_t cudaMemcpyFromSymbol(void *
# 149 "/usr/local/cuda/bin/../include/cuda_runtime.h"
dst, char *
# 150 "/usr/local/cuda/bin/../include/cuda_runtime.h"
symbol, size_t
# 151 "/usr/local/cuda/bin/../include/cuda_runtime.h"
count, size_t
# 152 "/usr/local/cuda/bin/../include/cuda_runtime.h"
offset = (0), cudaMemcpyKind
# 153 "/usr/local/cuda/bin/../include/cuda_runtime.h"
kind = cudaMemcpyDeviceToHost)
# 155 "/usr/local/cuda/bin/../include/cuda_runtime.h"
{
# 156 "/usr/local/cuda/bin/../include/cuda_runtime.h"
return cudaMemcpyFromSymbol(dst, (const char *)symbol, count, offset, kind);
# 157 "/usr/local/cuda/bin/../include/cuda_runtime.h"
}
# 159 "/usr/local/cuda/bin/../include/cuda_runtime.h"
template<class T> inline cudaError_t
# 160 "/usr/local/cuda/bin/../include/cuda_runtime.h"
cudaMemcpyFromSymbol(void *
# 161 "/usr/local/cuda/bin/../include/cuda_runtime.h"
dst, const T &
# 162 "/usr/local/cuda/bin/../include/cuda_runtime.h"
symbol, size_t
# 163 "/usr/local/cuda/bin/../include/cuda_runtime.h"
count, size_t
# 164 "/usr/local/cuda/bin/../include/cuda_runtime.h"
offset = (0), cudaMemcpyKind
# 165 "/usr/local/cuda/bin/../include/cuda_runtime.h"
kind = cudaMemcpyDeviceToHost)
# 167 "/usr/local/cuda/bin/../include/cuda_runtime.h"
{
# 168 "/usr/local/cuda/bin/../include/cuda_runtime.h"
return cudaMemcpyFromSymbol(dst, (const char *)(&symbol), count, offset, kind);
# 169 "/usr/local/cuda/bin/../include/cuda_runtime.h"
}
# 171 "/usr/local/cuda/bin/../include/cuda_runtime.h"
static inline cudaError_t cudaMemcpyFromSymbolAsync(void *
# 172 "/usr/local/cuda/bin/../include/cuda_runtime.h"
dst, char *
# 173 "/usr/local/cuda/bin/../include/cuda_runtime.h"
symbol, size_t
# 174 "/usr/local/cuda/bin/../include/cuda_runtime.h"
count, size_t
# 175 "/usr/local/cuda/bin/../include/cuda_runtime.h"
offset, cudaMemcpyKind
# 176 "/usr/local/cuda/bin/../include/cuda_runtime.h"
kind, cudaStream_t
# 177 "/usr/local/cuda/bin/../include/cuda_runtime.h"
stream)
# 179 "/usr/local/cuda/bin/../include/cuda_runtime.h"
{
# 180 "/usr/local/cuda/bin/../include/cuda_runtime.h"
return cudaMemcpyFromSymbolAsync(dst, (const char *)symbol, count, offset, kind, stream);
# 181 "/usr/local/cuda/bin/../include/cuda_runtime.h"
}
# 183 "/usr/local/cuda/bin/../include/cuda_runtime.h"
template<class T> inline cudaError_t
# 184 "/usr/local/cuda/bin/../include/cuda_runtime.h"
cudaMemcpyFromSymbolAsync(void *
# 185 "/usr/local/cuda/bin/../include/cuda_runtime.h"
dst, const T &
# 186 "/usr/local/cuda/bin/../include/cuda_runtime.h"
symbol, size_t
# 187 "/usr/local/cuda/bin/../include/cuda_runtime.h"
count, size_t
# 188 "/usr/local/cuda/bin/../include/cuda_runtime.h"
offset, cudaMemcpyKind
# 189 "/usr/local/cuda/bin/../include/cuda_runtime.h"
kind, cudaStream_t
# 190 "/usr/local/cuda/bin/../include/cuda_runtime.h"
stream)
# 192 "/usr/local/cuda/bin/../include/cuda_runtime.h"
{
# 193 "/usr/local/cuda/bin/../include/cuda_runtime.h"
return cudaMemcpyFromSymbolAsync(dst, (const char *)(&symbol), count, offset, kind, stream);
# 194 "/usr/local/cuda/bin/../include/cuda_runtime.h"
}
# 196 "/usr/local/cuda/bin/../include/cuda_runtime.h"
static inline cudaError_t cudaGetSymbolAddress(void **
# 197 "/usr/local/cuda/bin/../include/cuda_runtime.h"
devPtr, char *
# 198 "/usr/local/cuda/bin/../include/cuda_runtime.h"
symbol)
# 200 "/usr/local/cuda/bin/../include/cuda_runtime.h"
{
# 201 "/usr/local/cuda/bin/../include/cuda_runtime.h"
return cudaGetSymbolAddress(devPtr, (const char *)symbol);
# 202 "/usr/local/cuda/bin/../include/cuda_runtime.h"
}
# 204 "/usr/local/cuda/bin/../include/cuda_runtime.h"
template<class T> inline cudaError_t
# 205 "/usr/local/cuda/bin/../include/cuda_runtime.h"
cudaGetSymbolAddress(void **
# 206 "/usr/local/cuda/bin/../include/cuda_runtime.h"
devPtr, const T &
# 207 "/usr/local/cuda/bin/../include/cuda_runtime.h"
symbol)
# 209 "/usr/local/cuda/bin/../include/cuda_runtime.h"
{
# 210 "/usr/local/cuda/bin/../include/cuda_runtime.h"
return cudaGetSymbolAddress(devPtr, (const char *)(&symbol));
# 211 "/usr/local/cuda/bin/../include/cuda_runtime.h"
}
# 219 "/usr/local/cuda/bin/../include/cuda_runtime.h"
static inline cudaError_t cudaGetSymbolSize(size_t *
# 220 "/usr/local/cuda/bin/../include/cuda_runtime.h"
size, char *
# 221 "/usr/local/cuda/bin/../include/cuda_runtime.h"
symbol)
# 223 "/usr/local/cuda/bin/../include/cuda_runtime.h"
{
# 224 "/usr/local/cuda/bin/../include/cuda_runtime.h"
return cudaGetSymbolSize(size, (const char *)symbol);
# 225 "/usr/local/cuda/bin/../include/cuda_runtime.h"
}
# 227 "/usr/local/cuda/bin/../include/cuda_runtime.h"
template<class T> inline cudaError_t
# 228 "/usr/local/cuda/bin/../include/cuda_runtime.h"
cudaGetSymbolSize(size_t *
# 229 "/usr/local/cuda/bin/../include/cuda_runtime.h"
size, const T &
# 230 "/usr/local/cuda/bin/../include/cuda_runtime.h"
symbol)
# 232 "/usr/local/cuda/bin/../include/cuda_runtime.h"
{
# 233 "/usr/local/cuda/bin/../include/cuda_runtime.h"
return cudaGetSymbolSize(size, (const char *)(&symbol));
# 234 "/usr/local/cuda/bin/../include/cuda_runtime.h"
}
# 242 "/usr/local/cuda/bin/../include/cuda_runtime.h"
template<class T, int dim, cudaTextureReadMode readMode> inline cudaError_t
# 243 "/usr/local/cuda/bin/../include/cuda_runtime.h"
cudaBindTexture(size_t *
# 244 "/usr/local/cuda/bin/../include/cuda_runtime.h"
offset, const texture< T, dim, readMode> &
# 245 "/usr/local/cuda/bin/../include/cuda_runtime.h"
tex, const void *
# 246 "/usr/local/cuda/bin/../include/cuda_runtime.h"
devPtr, const cudaChannelFormatDesc &
# 247 "/usr/local/cuda/bin/../include/cuda_runtime.h"
desc, size_t
# 248 "/usr/local/cuda/bin/../include/cuda_runtime.h"
size = (((2147483647) * 2U) + 1U))
# 250 "/usr/local/cuda/bin/../include/cuda_runtime.h"
{
# 251 "/usr/local/cuda/bin/../include/cuda_runtime.h"
return cudaBindTexture(offset, &tex, devPtr, (&desc), size);
# 252 "/usr/local/cuda/bin/../include/cuda_runtime.h"
}
# 254 "/usr/local/cuda/bin/../include/cuda_runtime.h"
template<class T, int dim, cudaTextureReadMode readMode> inline cudaError_t
# 255 "/usr/local/cuda/bin/../include/cuda_runtime.h"
cudaBindTexture(size_t *
# 256 "/usr/local/cuda/bin/../include/cuda_runtime.h"
offset, const texture< T, dim, readMode> &
# 257 "/usr/local/cuda/bin/../include/cuda_runtime.h"
tex, const void *
# 258 "/usr/local/cuda/bin/../include/cuda_runtime.h"
devPtr, size_t
# 259 "/usr/local/cuda/bin/../include/cuda_runtime.h"
size = (((2147483647) * 2U) + 1U))
# 261 "/usr/local/cuda/bin/../include/cuda_runtime.h"
{
# 262 "/usr/local/cuda/bin/../include/cuda_runtime.h"
return cudaBindTexture(offset, tex, devPtr, (tex.channelDesc), size);
# 263 "/usr/local/cuda/bin/../include/cuda_runtime.h"
}
# 265 "/usr/local/cuda/bin/../include/cuda_runtime.h"
template<class T, int dim, cudaTextureReadMode readMode> inline cudaError_t
# 266 "/usr/local/cuda/bin/../include/cuda_runtime.h"
cudaBindTextureToArray(const texture< T, dim, readMode> &
# 267 "/usr/local/cuda/bin/../include/cuda_runtime.h"
tex, const cudaArray *
# 268 "/usr/local/cuda/bin/../include/cuda_runtime.h"
array, const cudaChannelFormatDesc &
# 269 "/usr/local/cuda/bin/../include/cuda_runtime.h"
desc)
# 271 "/usr/local/cuda/bin/../include/cuda_runtime.h"
{
# 272 "/usr/local/cuda/bin/../include/cuda_runtime.h"
return cudaBindTextureToArray(&tex, array, (&desc));
# 273 "/usr/local/cuda/bin/../include/cuda_runtime.h"
}
# 275 "/usr/local/cuda/bin/../include/cuda_runtime.h"
template<class T, int dim, cudaTextureReadMode readMode> inline cudaError_t
# 276 "/usr/local/cuda/bin/../include/cuda_runtime.h"
cudaBindTextureToArray(const texture< T, dim, readMode> &
# 277 "/usr/local/cuda/bin/../include/cuda_runtime.h"
tex, const cudaArray *
# 278 "/usr/local/cuda/bin/../include/cuda_runtime.h"
array)
# 280 "/usr/local/cuda/bin/../include/cuda_runtime.h"
{
# 281 "/usr/local/cuda/bin/../include/cuda_runtime.h"
auto cudaChannelFormatDesc desc;
# 282 "/usr/local/cuda/bin/../include/cuda_runtime.h"
auto cudaError_t err = cudaGetChannelDesc(&desc, array);
# 284 "/usr/local/cuda/bin/../include/cuda_runtime.h"
return (err == (cudaSuccess)) ? (cudaBindTextureToArray(tex, array, desc)) : err;
# 285 "/usr/local/cuda/bin/../include/cuda_runtime.h"
}
# 293 "/usr/local/cuda/bin/../include/cuda_runtime.h"
template<class T, int dim, cudaTextureReadMode readMode> inline cudaError_t
# 294 "/usr/local/cuda/bin/../include/cuda_runtime.h"
cudaUnbindTexture(const texture< T, dim, readMode> &
# 295 "/usr/local/cuda/bin/../include/cuda_runtime.h"
tex)
# 297 "/usr/local/cuda/bin/../include/cuda_runtime.h"
{
# 298 "/usr/local/cuda/bin/../include/cuda_runtime.h"
return cudaUnbindTexture(&tex);
# 299 "/usr/local/cuda/bin/../include/cuda_runtime.h"
}
# 307 "/usr/local/cuda/bin/../include/cuda_runtime.h"
template<class T, int dim, cudaTextureReadMode readMode> inline cudaError_t
# 308 "/usr/local/cuda/bin/../include/cuda_runtime.h"
cudaGetTextureAlignmentOffset(size_t *
# 309 "/usr/local/cuda/bin/../include/cuda_runtime.h"
offset, const texture< T, dim, readMode> &
# 310 "/usr/local/cuda/bin/../include/cuda_runtime.h"
tex)
# 312 "/usr/local/cuda/bin/../include/cuda_runtime.h"
{
# 313 "/usr/local/cuda/bin/../include/cuda_runtime.h"
return cudaGetTextureAlignmentOffset(offset, &tex);
# 314 "/usr/local/cuda/bin/../include/cuda_runtime.h"
}
# 322 "/usr/local/cuda/bin/../include/cuda_runtime.h"
template<class T> inline cudaError_t
# 323 "/usr/local/cuda/bin/../include/cuda_runtime.h"
cudaLaunch(T *
# 324 "/usr/local/cuda/bin/../include/cuda_runtime.h"
symbol)
# 326 "/usr/local/cuda/bin/../include/cuda_runtime.h"
{
# 327 "/usr/local/cuda/bin/../include/cuda_runtime.h"
return cudaLaunch((const char *)symbol);
# 328 "/usr/local/cuda/bin/../include/cuda_runtime.h"
}
# 45 "/usr/include/stdio.h" 3
struct _IO_FILE;
# 49 "/usr/include/stdio.h" 3
extern "C" { typedef _IO_FILE FILE; }
# 65 "/usr/include/stdio.h" 3
extern "C" { typedef _IO_FILE __FILE; }
# 90 "/usr/include/wchar.h" 3
extern "C" { typedef
# 79 "/usr/include/wchar.h" 3
struct __mbstate_t {
# 80 "/usr/include/wchar.h" 3
int __count;
# 82 "/usr/include/wchar.h" 3
union {
# 84 "/usr/include/wchar.h" 3
unsigned __wch;
# 88 "/usr/include/wchar.h" 3
char __wchb[4];
# 89 "/usr/include/wchar.h" 3
} __value;
# 90 "/usr/include/wchar.h" 3
} __mbstate_t; }
# 26 "/usr/include/_G_config.h" 3
extern "C" { typedef
# 23 "/usr/include/_G_config.h" 3
struct _G_fpos_t {
# 24 "/usr/include/_G_config.h" 3
__off_t __pos;
# 25 "/usr/include/_G_config.h" 3
__mbstate_t __state;
# 26 "/usr/include/_G_config.h" 3
} _G_fpos_t; }
# 31 "/usr/include/_G_config.h" 3
extern "C" { typedef
# 28 "/usr/include/_G_config.h" 3
struct _G_fpos64_t {
# 29 "/usr/include/_G_config.h" 3
__off64_t __pos;
# 30 "/usr/include/_G_config.h" 3
__mbstate_t __state;
# 31 "/usr/include/_G_config.h" 3
} _G_fpos64_t; }
# 53 "/usr/include/_G_config.h" 3
extern "C" { typedef short _G_int16_t; }
# 54 "/usr/include/_G_config.h" 3
extern "C" { typedef int _G_int32_t; }
# 55 "/usr/include/_G_config.h" 3
extern "C" { typedef unsigned short _G_uint16_t; }
# 56 "/usr/include/_G_config.h" 3
extern "C" { typedef unsigned _G_uint32_t; }
# 43 "/usr/lib/gcc/x86_64-redhat-linux/4.3.2/include/stdarg.h" 3
extern "C" { typedef __builtin_va_list __gnuc_va_list; }
# 170 "/usr/include/libio.h" 3
struct _IO_jump_t; struct _IO_FILE;
# 180 "/usr/include/libio.h" 3
extern "C" { typedef void _IO_lock_t; }
# 186 "/usr/include/libio.h" 3
extern "C" { struct _IO_marker {
# 187 "/usr/include/libio.h" 3
_IO_marker *_next;
# 188 "/usr/include/libio.h" 3
_IO_FILE *_sbuf;
# 192 "/usr/include/libio.h" 3
int _pos;
# 203 "/usr/include/libio.h" 3
}; }
# 206 "/usr/include/libio.h" 3
enum __codecvt_result {
# 208 "/usr/include/libio.h" 3
__codecvt_ok,
# 209 "/usr/include/libio.h" 3
__codecvt_partial,
# 210 "/usr/include/libio.h" 3
__codecvt_error,
# 211 "/usr/include/libio.h" 3
__codecvt_noconv
# 212 "/usr/include/libio.h" 3
};
# 271 "/usr/include/libio.h" 3
extern "C" { struct _IO_FILE {
# 272 "/usr/include/libio.h" 3
int _flags;
# 277 "/usr/include/libio.h" 3
char *_IO_read_ptr;
# 278 "/usr/include/libio.h" 3
char *_IO_read_end;
# 279 "/usr/include/libio.h" 3
char *_IO_read_base;
# 280 "/usr/include/libio.h" 3
char *_IO_write_base;
# 281 "/usr/include/libio.h" 3
char *_IO_write_ptr;
# 282 "/usr/include/libio.h" 3
char *_IO_write_end;
# 283 "/usr/include/libio.h" 3
char *_IO_buf_base;
# 284 "/usr/include/libio.h" 3
char *_IO_buf_end;
# 286 "/usr/include/libio.h" 3
char *_IO_save_base;
# 287 "/usr/include/libio.h" 3
char *_IO_backup_base;
# 288 "/usr/include/libio.h" 3
char *_IO_save_end;
# 290 "/usr/include/libio.h" 3
_IO_marker *_markers;
# 292 "/usr/include/libio.h" 3
_IO_FILE *_chain;
# 294 "/usr/include/libio.h" 3
int _fileno;
# 298 "/usr/include/libio.h" 3
int _flags2;
# 300 "/usr/include/libio.h" 3
__off_t _old_offset;
# 304 "/usr/include/libio.h" 3
unsigned short _cur_column;
# 305 "/usr/include/libio.h" 3
signed char _vtable_offset;
# 306 "/usr/include/libio.h" 3
char _shortbuf[1];
# 310 "/usr/include/libio.h" 3
_IO_lock_t *_lock;
# 319 "/usr/include/libio.h" 3
__off64_t _offset;
# 328 "/usr/include/libio.h" 3
void *__pad1;
# 329 "/usr/include/libio.h" 3
void *__pad2;
# 330 "/usr/include/libio.h" 3
void *__pad3;
# 331 "/usr/include/libio.h" 3
void *__pad4;
# 332 "/usr/include/libio.h" 3
size_t __pad5;
# 334 "/usr/include/libio.h" 3
int _mode;
# 336 "/usr/include/libio.h" 3
char _unused2[((((15) * sizeof(int)) - ((4) * sizeof(void *))) - sizeof(size_t))];
# 338 "/usr/include/libio.h" 3
}; }
# 344 "/usr/include/libio.h" 3
struct _IO_FILE_plus;
# 346 "/usr/include/libio.h" 3
extern "C" { extern _IO_FILE_plus _IO_2_1_stdin_; }
# 347 "/usr/include/libio.h" 3
extern "C" { extern _IO_FILE_plus _IO_2_1_stdout_; }
# 348 "/usr/include/libio.h" 3
extern "C" { extern _IO_FILE_plus _IO_2_1_stderr_; }
# 364 "/usr/include/libio.h" 3
extern "C" { typedef __ssize_t __io_read_fn(void *, char *, size_t); }
# 372 "/usr/include/libio.h" 3
extern "C" { typedef __ssize_t __io_write_fn(void *, const char *, size_t); }
# 381 "/usr/include/libio.h" 3
extern "C" { typedef int __io_seek_fn(void *, __off64_t *, int); }
# 384 "/usr/include/libio.h" 3
extern "C" { typedef int __io_close_fn(void *); }
# 389 "/usr/include/libio.h" 3
extern "C" { typedef __io_read_fn cookie_read_function_t; }
# 390 "/usr/include/libio.h" 3
extern "C" { typedef __io_write_fn cookie_write_function_t; }
# 391 "/usr/include/libio.h" 3
extern "C" { typedef __io_seek_fn cookie_seek_function_t; }
# 392 "/usr/include/libio.h" 3
extern "C" { typedef __io_close_fn cookie_close_function_t; }
# 401 "/usr/include/libio.h" 3
extern "C" { typedef
# 396 "/usr/include/libio.h" 3
struct _IO_cookie_io_functions_t {
# 397 "/usr/include/libio.h" 3
__io_read_fn *read;
# 398 "/usr/include/libio.h" 3
__io_write_fn *write;
# 399 "/usr/include/libio.h" 3
__io_seek_fn *seek;
# 400 "/usr/include/libio.h" 3
__io_close_fn *close;
# 401 "/usr/include/libio.h" 3
} _IO_cookie_io_functions_t; }
# 402 "/usr/include/libio.h" 3
extern "C" { typedef _IO_cookie_io_functions_t cookie_io_functions_t; }
# 404 "/usr/include/libio.h" 3
struct _IO_cookie_file;
# 407 "/usr/include/libio.h" 3
extern "C" void _IO_cookie_init(_IO_cookie_file *, int, void *, _IO_cookie_io_functions_t);
# 416 "/usr/include/libio.h" 3
extern "C" int __underflow(_IO_FILE *);
# 417 "/usr/include/libio.h" 3
extern "C" int __uflow(_IO_FILE *);
# 418 "/usr/include/libio.h" 3
extern "C" int __overflow(_IO_FILE *, int);
# 458 "/usr/include/libio.h" 3
extern "C" int _IO_getc(_IO_FILE *);
# 459 "/usr/include/libio.h" 3
extern "C" int _IO_putc(int, _IO_FILE *);
# 460 "/usr/include/libio.h" 3
extern "C" int _IO_feof(_IO_FILE *) throw();
# 461 "/usr/include/libio.h" 3
extern "C" int _IO_ferror(_IO_FILE *) throw();
# 463 "/usr/include/libio.h" 3
extern "C" int _IO_peekc_locked(_IO_FILE *);
# 469 "/usr/include/libio.h" 3
extern "C" void _IO_flockfile(_IO_FILE *) throw();
# 470 "/usr/include/libio.h" 3
extern "C" void _IO_funlockfile(_IO_FILE *) throw();
# 471 "/usr/include/libio.h" 3
extern "C" int _IO_ftrylockfile(_IO_FILE *) throw();
# 488 "/usr/include/libio.h" 3
extern "C" int _IO_vfscanf(_IO_FILE *__restrict__, const char *__restrict__, __gnuc_va_list, int *__restrict__);
# 490 "/usr/include/libio.h" 3
extern "C" int _IO_vfprintf(_IO_FILE *__restrict__, const char *__restrict__, __gnuc_va_list);
# 492 "/usr/include/libio.h" 3
extern "C" __ssize_t _IO_padn(_IO_FILE *, int, __ssize_t);
# 493 "/usr/include/libio.h" 3
extern "C" size_t _IO_sgetn(_IO_FILE *, void *, size_t);
# 495 "/usr/include/libio.h" 3
extern "C" __off64_t _IO_seekoff(_IO_FILE *, __off64_t, int, int);
# 496 "/usr/include/libio.h" 3
extern "C" __off64_t _IO_seekpos(_IO_FILE *, __off64_t, int);
# 498 "/usr/include/libio.h" 3
extern "C" void _IO_free_backup_area(_IO_FILE *) throw();
# 80 "/usr/include/stdio.h" 3
extern "C" { typedef __gnuc_va_list va_list; }
# 91 "/usr/include/stdio.h" 3
extern "C" { typedef _G_fpos_t fpos_t; }
# 97 "/usr/include/stdio.h" 3
extern "C" { typedef _G_fpos64_t fpos64_t; }
# 145 "/usr/include/stdio.h" 3
extern "C" { extern _IO_FILE *stdin; }
# 146 "/usr/include/stdio.h" 3
extern "C" { extern _IO_FILE *stdout; }
# 147 "/usr/include/stdio.h" 3
extern "C" { extern _IO_FILE *stderr; }
# 157 "/usr/include/stdio.h" 3
extern "C" int remove(const char *) throw();
# 159 "/usr/include/stdio.h" 3
extern "C" int rename(const char *, const char *) throw();
# 164 "/usr/include/stdio.h" 3
extern "C" int renameat(int, const char *, int, const char *) throw();
# 174 "/usr/include/stdio.h" 3
extern "C" FILE *tmpfile();
# 184 "/usr/include/stdio.h" 3
extern "C" FILE *tmpfile64();
# 188 "/usr/include/stdio.h" 3
extern "C" char *tmpnam(char *) throw();
# 194 "/usr/include/stdio.h" 3
extern "C" char *tmpnam_r(char *) throw();
# 206 "/usr/include/stdio.h" 3
extern "C" char *tempnam(const char *, const char *) throw() __attribute__((__malloc__));
# 216 "/usr/include/stdio.h" 3
extern "C" int fclose(FILE *);
# 221 "/usr/include/stdio.h" 3
extern "C" int fflush(FILE *);
# 231 "/usr/include/stdio.h" 3
extern "C" int fflush_unlocked(FILE *);
# 241 "/usr/include/stdio.h" 3
extern "C" int fcloseall();
# 251 "/usr/include/stdio.h" 3
extern "C" FILE *fopen(const char *__restrict__, const char *__restrict__);
# 257 "/usr/include/stdio.h" 3
extern "C" FILE *freopen(const char *__restrict__, const char *__restrict__, FILE *__restrict__);
# 276 "/usr/include/stdio.h" 3
extern "C" FILE *fopen64(const char *__restrict__, const char *__restrict__);
# 278 "/usr/include/stdio.h" 3
extern "C" FILE *freopen64(const char *__restrict__, const char *__restrict__, FILE *__restrict__);
# 285 "/usr/include/stdio.h" 3
extern "C" FILE *fdopen(int, const char *) throw();
# 291 "/usr/include/stdio.h" 3
extern "C" FILE *fopencookie(void *__restrict__, const char *__restrict__, _IO_cookie_io_functions_t) throw();
# 296 "/usr/include/stdio.h" 3
extern "C" FILE *fmemopen(void *, size_t, const char *) throw();
# 302 "/usr/include/stdio.h" 3
extern "C" FILE *open_memstream(char **, size_t *) throw();
# 309 "/usr/include/stdio.h" 3
extern "C" void setbuf(FILE *__restrict__, char *__restrict__) throw();
# 313 "/usr/include/stdio.h" 3
extern "C" int setvbuf(FILE *__restrict__, char *__restrict__, int, size_t) throw();
# 320 "/usr/include/stdio.h" 3
extern "C" void setbuffer(FILE *__restrict__, char *__restrict__, size_t) throw();
# 324 "/usr/include/stdio.h" 3
extern "C" void setlinebuf(FILE *) throw();
# 333 "/usr/include/stdio.h" 3
extern "C" int fprintf(FILE *__restrict__, const char *__restrict__, ...);
# 339 "/usr/include/stdio.h" 3
extern "C" int printf(const char *__restrict__, ...);
# 341 "/usr/include/stdio.h" 3
extern "C" int sprintf(char *__restrict__, const char *__restrict__, ...) throw();
# 348 "/usr/include/stdio.h" 3
extern "C" int vfprintf(FILE *__restrict__, const char *__restrict__, __gnuc_va_list);
# 354 "/usr/include/stdio.h" 3
extern "C" int vprintf(const char *__restrict__, __gnuc_va_list);
# 356 "/usr/include/stdio.h" 3
extern "C" int vsprintf(char *__restrict__, const char *__restrict__, __gnuc_va_list) throw();
# 363 "/usr/include/stdio.h" 3
extern "C" int snprintf(char *__restrict__, size_t, const char *__restrict__, ...) throw();
# 367 "/usr/include/stdio.h" 3
extern "C" int vsnprintf(char *__restrict__, size_t, const char *__restrict__, __gnuc_va_list) throw();
# 376 "/usr/include/stdio.h" 3
extern "C" int vasprintf(char **__restrict__, const char *__restrict__, __gnuc_va_list) throw();
# 379 "/usr/include/stdio.h" 3
extern "C" int __asprintf(char **__restrict__, const char *__restrict__, ...) throw();
# 382 "/usr/include/stdio.h" 3
extern "C" int asprintf(char **__restrict__, const char *__restrict__, ...) throw();
# 392 "/usr/include/stdio.h" 3
extern "C" int vdprintf(int, const char *__restrict__, __gnuc_va_list);
# 395 "/usr/include/stdio.h" 3
extern "C" int dprintf(int, const char *__restrict__, ...);
# 405 "/usr/include/stdio.h" 3
extern "C" int fscanf(FILE *__restrict__, const char *__restrict__, ...);
# 411 "/usr/include/stdio.h" 3
extern "C" int scanf(const char *__restrict__, ...);
# 413 "/usr/include/stdio.h" 3
extern "C" int sscanf(const char *__restrict__, const char *__restrict__, ...) throw();
# 451 "/usr/include/stdio.h" 3
extern "C" int vfscanf(FILE *__restrict__, const char *__restrict__, __gnuc_va_list);
# 459 "/usr/include/stdio.h" 3
extern "C" int vscanf(const char *__restrict__, __gnuc_va_list);
# 463 "/usr/include/stdio.h" 3
extern "C" int vsscanf(const char *__restrict__, const char *__restrict__, __gnuc_va_list) throw();
# 511 "/usr/include/stdio.h" 3
extern "C" int fgetc(FILE *);
# 512 "/usr/include/stdio.h" 3
extern "C" int getc(FILE *);
# 518 "/usr/include/stdio.h" 3
extern "C" int getchar();
# 530 "/usr/include/stdio.h" 3
extern "C" int getc_unlocked(FILE *);
# 531 "/usr/include/stdio.h" 3
extern "C" int getchar_unlocked();
# 541 "/usr/include/stdio.h" 3
extern "C" int fgetc_unlocked(FILE *);
# 553 "/usr/include/stdio.h" 3
extern "C" int fputc(int, FILE *);
# 554 "/usr/include/stdio.h" 3
extern "C" int putc(int, FILE *);
# 560 "/usr/include/stdio.h" 3
extern "C" int putchar(int);
# 574 "/usr/include/stdio.h" 3
extern "C" int fputc_unlocked(int, FILE *);
# 582 "/usr/include/stdio.h" 3
extern "C" int putc_unlocked(int, FILE *);
# 583 "/usr/include/stdio.h" 3
extern "C" int putchar_unlocked(int);
# 590 "/usr/include/stdio.h" 3
extern "C" int getw(FILE *);
# 593 "/usr/include/stdio.h" 3
extern "C" int putw(int, FILE *);
# 602 "/usr/include/stdio.h" 3
extern "C" char *fgets(char *__restrict__, int, FILE *__restrict__);
# 610 "/usr/include/stdio.h" 3
extern "C" char *gets(char *);
# 620 "/usr/include/stdio.h" 3
extern "C" char *fgets_unlocked(char *__restrict__, int, FILE *__restrict__);
# 636 "/usr/include/stdio.h" 3
extern "C" __ssize_t __getdelim(char **__restrict__, size_t *__restrict__, int, FILE *__restrict__);
# 639 "/usr/include/stdio.h" 3
extern "C" __ssize_t getdelim(char **__restrict__, size_t *__restrict__, int, FILE *__restrict__);
# 649 "/usr/include/stdio.h" 3
extern "C" __ssize_t getline(char **__restrict__, size_t *__restrict__, FILE *__restrict__);
# 660 "/usr/include/stdio.h" 3
extern "C" int fputs(const char *__restrict__, FILE *__restrict__);
# 666 "/usr/include/stdio.h" 3
extern "C" int puts(const char *);
# 673 "/usr/include/stdio.h" 3
extern "C" int ungetc(int, FILE *);
# 680 "/usr/include/stdio.h" 3
extern "C" size_t fread(void *__restrict__, size_t, size_t, FILE *__restrict__);
# 686 "/usr/include/stdio.h" 3
extern "C" size_t fwrite(const void *__restrict__, size_t, size_t, FILE *__restrict__);
# 697 "/usr/include/stdio.h" 3
extern "C" int fputs_unlocked(const char *__restrict__, FILE *__restrict__);
# 708 "/usr/include/stdio.h" 3
extern "C" size_t fread_unlocked(void *__restrict__, size_t, size_t, FILE *__restrict__);
# 710 "/usr/include/stdio.h" 3
extern "C" size_t fwrite_unlocked(const void *__restrict__, size_t, size_t, FILE *__restrict__);
# 720 "/usr/include/stdio.h" 3
extern "C" int fseek(FILE *, long, int);
# 725 "/usr/include/stdio.h" 3
extern "C" long ftell(FILE *);
# 730 "/usr/include/stdio.h" 3
extern "C" void rewind(FILE *);
# 744 "/usr/include/stdio.h" 3
extern "C" int fseeko(FILE *, __off_t, int);
# 749 "/usr/include/stdio.h" 3
extern "C" __off_t ftello(FILE *);
# 769 "/usr/include/stdio.h" 3
extern "C" int fgetpos(FILE *__restrict__, fpos_t *__restrict__);
# 774 "/usr/include/stdio.h" 3
extern "C" int fsetpos(FILE *, const fpos_t *);
# 789 "/usr/include/stdio.h" 3
extern "C" int fseeko64(FILE *, __off64_t, int);
# 790 "/usr/include/stdio.h" 3
extern "C" __off64_t ftello64(FILE *);
# 791 "/usr/include/stdio.h" 3
extern "C" int fgetpos64(FILE *__restrict__, fpos64_t *__restrict__);
# 792 "/usr/include/stdio.h" 3
extern "C" int fsetpos64(FILE *, const fpos64_t *);
# 797 "/usr/include/stdio.h" 3
extern "C" void clearerr(FILE *) throw();
# 799 "/usr/include/stdio.h" 3
extern "C" int feof(FILE *) throw();
# 801 "/usr/include/stdio.h" 3
extern "C" int ferror(FILE *) throw();
# 806 "/usr/include/stdio.h" 3
extern "C" void clearerr_unlocked(FILE *) throw();
# 807 "/usr/include/stdio.h" 3
extern "C" int feof_unlocked(FILE *) throw();
# 808 "/usr/include/stdio.h" 3
extern "C" int ferror_unlocked(FILE *) throw();
# 817 "/usr/include/stdio.h" 3
extern "C" void perror(const char *);
# 27 "/usr/include/bits/sys_errlist.h" 3
extern "C" { extern int sys_nerr; }
# 28 "/usr/include/bits/sys_errlist.h" 3
extern "C" { extern const char *const sys_errlist[]; }
# 31 "/usr/include/bits/sys_errlist.h" 3
extern "C" { extern int _sys_nerr; }
# 32 "/usr/include/bits/sys_errlist.h" 3
extern "C" { extern const char *const _sys_errlist[]; }
# 829 "/usr/include/stdio.h" 3
extern "C" int fileno(FILE *) throw();
# 834 "/usr/include/stdio.h" 3
extern "C" int fileno_unlocked(FILE *) throw();
# 844 "/usr/include/stdio.h" 3
extern "C" FILE *popen(const char *, const char *);
# 850 "/usr/include/stdio.h" 3
extern "C" int pclose(FILE *);
# 856 "/usr/include/stdio.h" 3
extern "C" char *ctermid(char *) throw();
# 862 "/usr/include/stdio.h" 3
extern "C" char *cuserid(char *);
# 867 "/usr/include/stdio.h" 3
struct obstack;
# 870 "/usr/include/stdio.h" 3
extern "C" int obstack_printf(obstack *__restrict__, const char *__restrict__, ...) throw();
# 873 "/usr/include/stdio.h" 3
extern "C" int obstack_vprintf(obstack *__restrict__, const char *__restrict__, __gnuc_va_list) throw();
# 884 "/usr/include/stdio.h" 3
extern "C" void flockfile(FILE *) throw();
# 888 "/usr/include/stdio.h" 3
extern "C" int ftrylockfile(FILE *) throw();
# 891 "/usr/include/stdio.h" 3
extern "C" void funlockfile(FILE *) throw();
# 65 "/usr/local/cuda/NVIDIA_CUDA_SDK/common/inc/cutil.h"
enum CUTBoolean {
# 67 "/usr/local/cuda/NVIDIA_CUDA_SDK/common/inc/cutil.h"
CUTFalse,
# 68 "/usr/local/cuda/NVIDIA_CUDA_SDK/common/inc/cutil.h"
CUTTrue
# 69 "/usr/local/cuda/NVIDIA_CUDA_SDK/common/inc/cutil.h"
};
# 77 "/usr/local/cuda/NVIDIA_CUDA_SDK/common/inc/cutil.h"
extern "C" void cutFree(void *);
# 95 "/usr/local/cuda/NVIDIA_CUDA_SDK/common/inc/cutil.h"
extern "C" void cutCheckBankAccess(unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, const char *, const int, const char *, const int);
# 108 "/usr/local/cuda/NVIDIA_CUDA_SDK/common/inc/cutil.h"
extern "C" char *cutFindFilePath(const char *, const char *);
# 123 "/usr/local/cuda/NVIDIA_CUDA_SDK/common/inc/cutil.h"
extern "C" CUTBoolean cutReadFilef(const char *, float **, unsigned *, bool = false);
# 139 "/usr/local/cuda/NVIDIA_CUDA_SDK/common/inc/cutil.h"
extern "C" CUTBoolean cutReadFiled(const char *, double **, unsigned *, bool = false);
# 155 "/usr/local/cuda/NVIDIA_CUDA_SDK/common/inc/cutil.h"
extern "C" CUTBoolean cutReadFilei(const char *, int **, unsigned *, bool = false);
# 170 "/usr/local/cuda/NVIDIA_CUDA_SDK/common/inc/cutil.h"
extern "C" CUTBoolean cutReadFileui(const char *, unsigned **, unsigned *, bool = false);
# 186 "/usr/local/cuda/NVIDIA_CUDA_SDK/common/inc/cutil.h"
extern "C" CUTBoolean cutReadFileb(const char *, char **, unsigned *, bool = false);
# 202 "/usr/local/cuda/NVIDIA_CUDA_SDK/common/inc/cutil.h"
extern "C" CUTBoolean cutReadFileub(const char *, unsigned char **, unsigned *, bool = false);
# 216 "/usr/local/cuda/NVIDIA_CUDA_SDK/common/inc/cutil.h"
extern "C" CUTBoolean cutWriteFilef(const char *, const float *, unsigned, const float, bool = false);
# 230 "/usr/local/cuda/NVIDIA_CUDA_SDK/common/inc/cutil.h"
extern "C" CUTBoolean cutWriteFiled(const char *, const float *, unsigned, const double, bool = false);
# 242 "/usr/local/cuda/NVIDIA_CUDA_SDK/common/inc/cutil.h"
extern "C" CUTBoolean cutWriteFilei(const char *, const int *, unsigned, bool = false);
# 254 "/usr/local/cuda/NVIDIA_CUDA_SDK/common/inc/cutil.h"
extern "C" CUTBoolean cutWriteFileui(const char *, const unsigned *, unsigned, bool = false);
# 266 "/usr/local/cuda/NVIDIA_CUDA_SDK/common/inc/cutil.h"
extern "C" CUTBoolean cutWriteFileb(const char *, const char *, unsigned, bool = false);
# 278 "/usr/local/cuda/NVIDIA_CUDA_SDK/common/inc/cutil.h"
extern "C" CUTBoolean cutWriteFileub(const char *, const unsigned char *, unsigned, bool = false);
# 294 "/usr/local/cuda/NVIDIA_CUDA_SDK/common/inc/cutil.h"
extern "C" CUTBoolean cutLoadPGMub(const char *, unsigned char **, unsigned *, unsigned *);
# 307 "/usr/local/cuda/NVIDIA_CUDA_SDK/common/inc/cutil.h"
extern "C" CUTBoolean cutLoadPPMub(const char *, unsigned char **, unsigned *, unsigned *);
# 321 "/usr/local/cuda/NVIDIA_CUDA_SDK/common/inc/cutil.h"
extern "C" CUTBoolean cutLoadPPM4ub(const char *, unsigned char **, unsigned *, unsigned *);
# 337 "/usr/local/cuda/NVIDIA_CUDA_SDK/common/inc/cutil.h"
extern "C" CUTBoolean cutLoadPGMi(const char *, unsigned **, unsigned *, unsigned *);
# 353 "/usr/local/cuda/NVIDIA_CUDA_SDK/common/inc/cutil.h"
extern "C" CUTBoolean cutLoadPGMs(const char *, unsigned short **, unsigned *, unsigned *);
# 368 "/usr/local/cuda/NVIDIA_CUDA_SDK/common/inc/cutil.h"
extern "C" CUTBoolean cutLoadPGMf(const char *, float **, unsigned *, unsigned *);
# 380 "/usr/local/cuda/NVIDIA_CUDA_SDK/common/inc/cutil.h"
extern "C" CUTBoolean cutSavePGMub(const char *, unsigned char *, unsigned, unsigned);
# 392 "/usr/local/cuda/NVIDIA_CUDA_SDK/common/inc/cutil.h"
extern "C" CUTBoolean cutSavePPMub(const char *, unsigned char *, unsigned, unsigned);
# 405 "/usr/local/cuda/NVIDIA_CUDA_SDK/common/inc/cutil.h"
extern "C" CUTBoolean cutSavePPM4ub(const char *, unsigned char *, unsigned, unsigned);
# 417 "/usr/local/cuda/NVIDIA_CUDA_SDK/common/inc/cutil.h"
extern "C" CUTBoolean cutSavePGMi(const char *, unsigned *, unsigned, unsigned);
# 429 "/usr/local/cuda/NVIDIA_CUDA_SDK/common/inc/cutil.h"
extern "C" CUTBoolean cutSavePGMs(const char *, unsigned short *, unsigned, unsigned);
# 441 "/usr/local/cuda/NVIDIA_CUDA_SDK/common/inc/cutil.h"
extern "C" CUTBoolean cutSavePGMf(const char *, float *, unsigned, unsigned);
# 462 "/usr/local/cuda/NVIDIA_CUDA_SDK/common/inc/cutil.h"
extern "C" CUTBoolean cutCheckCmdLineFlag(const int, const char **, const char *);
# 476 "/usr/local/cuda/NVIDIA_CUDA_SDK/common/inc/cutil.h"
extern "C" CUTBoolean cutGetCmdLineArgumenti(const int, const char **, const char *, int *);
# 490 "/usr/local/cuda/NVIDIA_CUDA_SDK/common/inc/cutil.h"
extern "C" CUTBoolean cutGetCmdLineArgumentf(const int, const char **, const char *, float *);
# 504 "/usr/local/cuda/NVIDIA_CUDA_SDK/common/inc/cutil.h"
extern "C" CUTBoolean cutGetCmdLineArgumentstr(const int, const char **, const char *, char **);
# 519 "/usr/local/cuda/NVIDIA_CUDA_SDK/common/inc/cutil.h"
extern "C" CUTBoolean cutGetCmdLineArgumentListstr(const int, const char **, const char *, char **, unsigned *);
# 533 "/usr/local/cuda/NVIDIA_CUDA_SDK/common/inc/cutil.h"
extern "C" CUTBoolean cutCheckCondition(int, const char *, const int);
# 545 "/usr/local/cuda/NVIDIA_CUDA_SDK/common/inc/cutil.h"
extern "C" CUTBoolean cutComparef(const float *, const float *, const unsigned);
# 558 "/usr/local/cuda/NVIDIA_CUDA_SDK/common/inc/cutil.h"
extern "C" CUTBoolean cutComparei(const int *, const int *, const unsigned);
# 571 "/usr/local/cuda/NVIDIA_CUDA_SDK/common/inc/cutil.h"
extern "C" CUTBoolean cutCompareub(const unsigned char *, const unsigned char *, const unsigned);
# 585 "/usr/local/cuda/NVIDIA_CUDA_SDK/common/inc/cutil.h"
extern "C" CUTBoolean cutCompareube(const unsigned char *, const unsigned char *, const unsigned, const float);
# 599 "/usr/local/cuda/NVIDIA_CUDA_SDK/common/inc/cutil.h"
extern "C" CUTBoolean cutComparefe(const float *, const float *, const unsigned, const float);
# 614 "/usr/local/cuda/NVIDIA_CUDA_SDK/common/inc/cutil.h"
extern "C" CUTBoolean cutCompareL2fe(const float *, const float *, const unsigned, const float);
# 627 "/usr/local/cuda/NVIDIA_CUDA_SDK/common/inc/cutil.h"
extern "C" CUTBoolean cutCreateTimer(unsigned *);
# 636 "/usr/local/cuda/NVIDIA_CUDA_SDK/common/inc/cutil.h"
extern "C" CUTBoolean cutDeleteTimer(unsigned);
# 644 "/usr/local/cuda/NVIDIA_CUDA_SDK/common/inc/cutil.h"
extern "C" CUTBoolean cutStartTimer(const unsigned);
# 652 "/usr/local/cuda/NVIDIA_CUDA_SDK/common/inc/cutil.h"
extern "C" CUTBoolean cutStopTimer(const unsigned);
# 660 "/usr/local/cuda/NVIDIA_CUDA_SDK/common/inc/cutil.h"
extern "C" CUTBoolean cutResetTimer(const unsigned);
# 669 "/usr/local/cuda/NVIDIA_CUDA_SDK/common/inc/cutil.h"
extern "C" float cutGetTimerValue(const unsigned);
# 680 "/usr/local/cuda/NVIDIA_CUDA_SDK/common/inc/cutil.h"
extern "C" float cutGetAverageTimerValue(const unsigned);
# 36 "/usr/include/rpc/types.h" 3
typedef int bool_t;
# 37 "/usr/include/rpc/types.h" 3
typedef int enum_t;
# 39 "/usr/include/rpc/types.h" 3
typedef unsigned long rpcprog_t;
# 40 "/usr/include/rpc/types.h" 3
typedef unsigned long rpcvers_t;
# 41 "/usr/include/rpc/types.h" 3
typedef unsigned long rpcproc_t;
# 42 "/usr/include/rpc/types.h" 3
typedef unsigned long rpcprot_t;
# 43 "/usr/include/rpc/types.h" 3
typedef unsigned long rpcport_t;
# 57 "/usr/include/sys/time.h" 3
extern "C" { struct timezone {
# 59 "/usr/include/sys/time.h" 3
int tz_minuteswest;
# 60 "/usr/include/sys/time.h" 3
int tz_dsttime;
# 61 "/usr/include/sys/time.h" 3
}; }
# 63 "/usr/include/sys/time.h" 3
extern "C" { typedef struct timezone *__restrict__ __timezone_ptr_t; }
# 73 "/usr/include/sys/time.h" 3
extern "C" int gettimeofday(timeval *__restrict__, __timezone_ptr_t) throw();
# 79 "/usr/include/sys/time.h" 3
extern "C" int settimeofday(const timeval *, const struct timezone *) throw();
# 87 "/usr/include/sys/time.h" 3
extern "C" int adjtime(const timeval *, timeval *) throw();
# 93 "/usr/include/sys/time.h" 3
enum __itimer_which {
# 96 "/usr/include/sys/time.h" 3
ITIMER_REAL,
# 99 "/usr/include/sys/time.h" 3
ITIMER_VIRTUAL,
# 103 "/usr/include/sys/time.h" 3
ITIMER_PROF
# 105 "/usr/include/sys/time.h" 3
};
# 109 "/usr/include/sys/time.h" 3
extern "C" { struct itimerval {
# 112 "/usr/include/sys/time.h" 3
timeval it_interval;
# 114 "/usr/include/sys/time.h" 3
timeval it_value;
# 115 "/usr/include/sys/time.h" 3
}; }
# 122 "/usr/include/sys/time.h" 3
extern "C" { typedef int __itimer_which_t; }
# 127 "/usr/include/sys/time.h" 3
extern "C" int getitimer(__itimer_which_t, itimerval *) throw();
# 133 "/usr/include/sys/time.h" 3
extern "C" int setitimer(__itimer_which_t, const itimerval *__restrict__, itimerval *__restrict__) throw();
# 140 "/usr/include/sys/time.h" 3
extern "C" int utimes(const char *, const timeval [2]) throw();
# 145 "/usr/include/sys/time.h" 3
extern "C" int lutimes(const char *, const timeval [2]) throw();
# 149 "/usr/include/sys/time.h" 3
extern "C" int futimes(int, const timeval [2]) throw();
# 156 "/usr/include/sys/time.h" 3
extern "C" int futimesat(int, const char *, const timeval [2]) throw();
# 49 "/usr/include/stdint.h" 3
typedef unsigned char uint8_t;
# 50 "/usr/include/stdint.h" 3
typedef unsigned short uint16_t;
# 52 "/usr/include/stdint.h" 3
typedef unsigned uint32_t;
# 56 "/usr/include/stdint.h" 3
typedef unsigned long uint64_t;
# 66 "/usr/include/stdint.h" 3
typedef signed char int_least8_t;
# 67 "/usr/include/stdint.h" 3
typedef short int_least16_t;
# 68 "/usr/include/stdint.h" 3
typedef int int_least32_t;
# 70 "/usr/include/stdint.h" 3
typedef long int_least64_t;
# 77 "/usr/include/stdint.h" 3
typedef unsigned char uint_least8_t;
# 78 "/usr/include/stdint.h" 3
typedef unsigned short uint_least16_t;
# 79 "/usr/include/stdint.h" 3
typedef unsigned uint_least32_t;
# 81 "/usr/include/stdint.h" 3
typedef unsigned long uint_least64_t;
# 91 "/usr/include/stdint.h" 3
typedef signed char int_fast8_t;
# 93 "/usr/include/stdint.h" 3
typedef long int_fast16_t;
# 94 "/usr/include/stdint.h" 3
typedef long int_fast32_t;
# 95 "/usr/include/stdint.h" 3
typedef long int_fast64_t;
# 104 "/usr/include/stdint.h" 3
typedef unsigned char uint_fast8_t;
# 106 "/usr/include/stdint.h" 3
typedef unsigned long uint_fast16_t;
# 107 "/usr/include/stdint.h" 3
typedef unsigned long uint_fast32_t;
# 108 "/usr/include/stdint.h" 3
typedef unsigned long uint_fast64_t;
# 120 "/usr/include/stdint.h" 3
typedef long intptr_t;
# 123 "/usr/include/stdint.h" 3
typedef unsigned long uintptr_t;
# 135 "/usr/include/stdint.h" 3
typedef long intmax_t;
# 136 "/usr/include/stdint.h" 3
typedef unsigned long uintmax_t;
# 44 "/usr/include/bits/uio.h" 3
extern "C" { struct iovec {
# 46 "/usr/include/bits/uio.h" 3
void *iov_base;
# 47 "/usr/include/bits/uio.h" 3
size_t iov_len;
# 48 "/usr/include/bits/uio.h" 3
}; }
# 40 "/usr/include/sys/uio.h" 3
extern "C" ssize_t readv(int, const iovec *, int);
# 50 "/usr/include/sys/uio.h" 3
extern "C" ssize_t writev(int, const iovec *, int);
# 35 "/usr/include/bits/socket.h" 3
extern "C" { typedef __socklen_t socklen_t; }
# 40 "/usr/include/bits/socket.h" 3
enum __socket_type {
# 42 "/usr/include/bits/socket.h" 3
SOCK_STREAM = 1,
# 45 "/usr/include/bits/socket.h" 3
SOCK_DGRAM,
# 48 "/usr/include/bits/socket.h" 3
SOCK_RAW,
# 50 "/usr/include/bits/socket.h" 3
SOCK_RDM,
# 52 "/usr/include/bits/socket.h" 3
SOCK_SEQPACKET,
# 55 "/usr/include/bits/socket.h" 3
SOCK_DCCP,
# 57 "/usr/include/bits/socket.h" 3
SOCK_PACKET = 10,
# 65 "/usr/include/bits/socket.h" 3
SOCK_CLOEXEC = 524288,
# 68 "/usr/include/bits/socket.h" 3
SOCK_NONBLOCK = 2048
# 71 "/usr/include/bits/socket.h" 3
};
# 29 "/usr/include/bits/sockaddr.h" 3
extern "C" { typedef unsigned short sa_family_t; }
# 162 "/usr/include/bits/socket.h" 3
extern "C" { struct sockaddr {
# 164 "/usr/include/bits/socket.h" 3
sa_family_t sa_family;
# 165 "/usr/include/bits/socket.h" 3
char sa_data[14];
# 166 "/usr/include/bits/socket.h" 3
}; }
# 175 "/usr/include/bits/socket.h" 3
extern "C" { struct sockaddr_storage {
# 177 "/usr/include/bits/socket.h" 3
sa_family_t ss_family;
# 178 "/usr/include/bits/socket.h" 3
unsigned long __ss_align;
# 179 "/usr/include/bits/socket.h" 3
char __ss_padding[((128) - ((2) * sizeof(unsigned long)))];
# 180 "/usr/include/bits/socket.h" 3
}; }
# 185 "/usr/include/bits/socket.h" 3
enum __cuda_MSG_OOB {
# 186 "/usr/include/bits/socket.h" 3
MSG_OOB = 1,
# 188 "/usr/include/bits/socket.h" 3
MSG_PEEK,
# 190 "/usr/include/bits/socket.h" 3
MSG_DONTROUTE = 4,
# 194 "/usr/include/bits/socket.h" 3
MSG_TRYHARD = 4,
# 197 "/usr/include/bits/socket.h" 3
MSG_CTRUNC = 8,
# 199 "/usr/include/bits/socket.h" 3
MSG_PROXY = 16,
# 201 "/usr/include/bits/socket.h" 3
MSG_TRUNC = 32,
# 203 "/usr/include/bits/socket.h" 3
MSG_DONTWAIT = 64,
# 205 "/usr/include/bits/socket.h" 3
MSG_EOR = 128,
# 207 "/usr/include/bits/socket.h" 3
MSG_WAITALL = 256,
# 209 "/usr/include/bits/socket.h" 3
MSG_FIN = 512,
# 211 "/usr/include/bits/socket.h" 3
MSG_SYN = 1024,
# 213 "/usr/include/bits/socket.h" 3
MSG_CONFIRM = 2048,
# 215 "/usr/include/bits/socket.h" 3
MSG_RST = 4096,
# 217 "/usr/include/bits/socket.h" 3
MSG_ERRQUEUE = 8192,
# 219 "/usr/include/bits/socket.h" 3
MSG_NOSIGNAL = 16384,
# 221 "/usr/include/bits/socket.h" 3
MSG_MORE = 32768,
# 224 "/usr/include/bits/socket.h" 3
MSG_CMSG_CLOEXEC = 1073741824
# 228 "/usr/include/bits/socket.h" 3
};
# 233 "/usr/include/bits/socket.h" 3
extern "C" { struct msghdr {
# 235 "/usr/include/bits/socket.h" 3
void *msg_name;
# 236 "/usr/include/bits/socket.h" 3
socklen_t msg_namelen;
# 238 "/usr/include/bits/socket.h" 3
iovec *msg_iov;
# 239 "/usr/include/bits/socket.h" 3
size_t msg_iovlen;
# 241 "/usr/include/bits/socket.h" 3
void *msg_control;
# 242 "/usr/include/bits/socket.h" 3
size_t msg_controllen;
# 247 "/usr/include/bits/socket.h" 3
int msg_flags;
# 248 "/usr/include/bits/socket.h" 3
}; }
# 251 "/usr/include/bits/socket.h" 3
extern "C" { struct cmsghdr {
# 253 "/usr/include/bits/socket.h" 3
size_t cmsg_len;
# 258 "/usr/include/bits/socket.h" 3
int cmsg_level;
# 259 "/usr/include/bits/socket.h" 3
int cmsg_type;
# 261 "/usr/include/bits/socket.h" 3
unsigned char __cmsg_data[0];
# 263 "/usr/include/bits/socket.h" 3
}; }
# 281 "/usr/include/bits/socket.h" 3
extern "C" cmsghdr *__cmsg_nxthdr(msghdr *, cmsghdr *) throw();
# 309 "/usr/include/bits/socket.h" 3
enum __cuda_SCM_RIGHTS {
# 310 "/usr/include/bits/socket.h" 3
SCM_RIGHTS = 1,
# 313 "/usr/include/bits/socket.h" 3
SCM_CREDENTIALS
# 316 "/usr/include/bits/socket.h" 3
};
# 320 "/usr/include/bits/socket.h" 3
extern "C" { struct ucred {
# 322 "/usr/include/bits/socket.h" 3
pid_t pid;
# 323 "/usr/include/bits/socket.h" 3
uid_t uid;
# 324 "/usr/include/bits/socket.h" 3
gid_t gid;
# 325 "/usr/include/bits/socket.h" 3
}; }
# 388 "/usr/include/bits/socket.h" 3
extern "C" { struct linger {
# 390 "/usr/include/bits/socket.h" 3
int l_onoff;
# 391 "/usr/include/bits/socket.h" 3
int l_linger;
# 392 "/usr/include/bits/socket.h" 3
}; }
# 45 "/usr/include/sys/socket.h" 3
extern "C" { struct osockaddr {
# 47 "/usr/include/sys/socket.h" 3
unsigned short sa_family;
# 48 "/usr/include/sys/socket.h" 3
unsigned char sa_data[14];
# 49 "/usr/include/sys/socket.h" 3
}; }
# 55 "/usr/include/sys/socket.h" 3
enum __cuda_SHUT_RD {
# 56 "/usr/include/sys/socket.h" 3
SHUT_RD,
# 58 "/usr/include/sys/socket.h" 3
SHUT_WR,
# 60 "/usr/include/sys/socket.h" 3
SHUT_RDWR
# 62 "/usr/include/sys/socket.h" 3
};
# 105 "/usr/include/sys/socket.h" 3
extern "C" int socket(int, int, int) throw();
# 111 "/usr/include/sys/socket.h" 3
extern "C" int socketpair(int, int, int, int [2]) throw();
# 115 "/usr/include/sys/socket.h" 3
extern "C" int bind(int, const sockaddr *, socklen_t) throw();
# 119 "/usr/include/sys/socket.h" 3
extern "C" int getsockname(int, sockaddr *__restrict__, socklen_t *__restrict__) throw();
# 129 "/usr/include/sys/socket.h" 3
extern "C" int connect(int, const sockaddr *, socklen_t);
# 133 "/usr/include/sys/socket.h" 3
extern "C" int getpeername(int, sockaddr *__restrict__, socklen_t *__restrict__) throw();
# 141 "/usr/include/sys/socket.h" 3
extern "C" ssize_t send(int, const void *, size_t, int);
# 148 "/usr/include/sys/socket.h" 3
extern "C" ssize_t recv(int, void *, size_t, int);
# 155 "/usr/include/sys/socket.h" 3
extern "C" ssize_t sendto(int, const void *, size_t, int, const sockaddr *, socklen_t);
# 166 "/usr/include/sys/socket.h" 3
extern "C" ssize_t recvfrom(int, void *__restrict__, size_t, int, sockaddr *__restrict__, socklen_t *__restrict__);
# 176 "/usr/include/sys/socket.h" 3
extern "C" ssize_t sendmsg(int, const msghdr *, int);
# 184 "/usr/include/sys/socket.h" 3
extern "C" ssize_t recvmsg(int, msghdr *, int);
# 190 "/usr/include/sys/socket.h" 3
extern "C" int getsockopt(int, int, int, void *__restrict__, socklen_t *__restrict__) throw();
# 197 "/usr/include/sys/socket.h" 3
extern "C" int setsockopt(int, int, int, const void *, socklen_t) throw();
# 204 "/usr/include/sys/socket.h" 3
extern "C" int listen(int, int) throw();
# 214 "/usr/include/sys/socket.h" 3
extern "C" int accept(int, sockaddr *__restrict__, socklen_t *__restrict__);
# 223 "/usr/include/sys/socket.h" 3
extern "C" int shutdown(int, int) throw();
# 228 "/usr/include/sys/socket.h" 3
extern "C" int sockatmark(int) throw();
# 236 "/usr/include/sys/socket.h" 3
extern "C" int isfdtype(int, int) throw();
# 33 "/usr/include/netinet/in.h" 3
enum __cuda_IPPROTO_IP {
# 34 "/usr/include/netinet/in.h" 3
IPPROTO_IP,
# 36 "/usr/include/netinet/in.h" 3
IPPROTO_HOPOPTS = 0,
# 38 "/usr/include/netinet/in.h" 3
IPPROTO_ICMP,
# 40 "/usr/include/netinet/in.h" 3
IPPROTO_IGMP,
# 42 "/usr/include/netinet/in.h" 3
IPPROTO_IPIP = 4,
# 44 "/usr/include/netinet/in.h" 3
IPPROTO_TCP = 6,
# 46 "/usr/include/netinet/in.h" 3
IPPROTO_EGP = 8,
# 48 "/usr/include/netinet/in.h" 3
IPPROTO_PUP = 12,
# 50 "/usr/include/netinet/in.h" 3
IPPROTO_UDP = 17,
# 52 "/usr/include/netinet/in.h" 3
IPPROTO_IDP = 22,
# 54 "/usr/include/netinet/in.h" 3
IPPROTO_TP = 29,
# 56 "/usr/include/netinet/in.h" 3
IPPROTO_DCCP = 33,
# 58 "/usr/include/netinet/in.h" 3
IPPROTO_IPV6 = 41,
# 60 "/usr/include/netinet/in.h" 3
IPPROTO_ROUTING = 43,
# 62 "/usr/include/netinet/in.h" 3
IPPROTO_FRAGMENT,
# 64 "/usr/include/netinet/in.h" 3
IPPROTO_RSVP = 46,
# 66 "/usr/include/netinet/in.h" 3
IPPROTO_GRE,
# 68 "/usr/include/netinet/in.h" 3
IPPROTO_ESP = 50,
# 70 "/usr/include/netinet/in.h" 3
IPPROTO_AH,
# 72 "/usr/include/netinet/in.h" 3
IPPROTO_ICMPV6 = 58,
# 74 "/usr/include/netinet/in.h" 3
IPPROTO_NONE,
# 76 "/usr/include/netinet/in.h" 3
IPPROTO_DSTOPTS,
# 78 "/usr/include/netinet/in.h" 3
IPPROTO_MTP = 92,
# 80 "/usr/include/netinet/in.h" 3
IPPROTO_ENCAP = 98,
# 82 "/usr/include/netinet/in.h" 3
IPPROTO_PIM = 103,
# 84 "/usr/include/netinet/in.h" 3
IPPROTO_COMP = 108,
# 86 "/usr/include/netinet/in.h" 3
IPPROTO_SCTP = 132,
# 88 "/usr/include/netinet/in.h" 3
IPPROTO_UDPLITE = 136,
# 90 "/usr/include/netinet/in.h" 3
IPPROTO_RAW = 255,
# 92 "/usr/include/netinet/in.h" 3
IPPROTO_MAX
# 93 "/usr/include/netinet/in.h" 3
};
# 97 "/usr/include/netinet/in.h" 3
extern "C" { typedef uint16_t in_port_t; }
# 101 "/usr/include/netinet/in.h" 3
enum __cuda_IPPORT_ECHO {
# 102 "/usr/include/netinet/in.h" 3
IPPORT_ECHO = 7,
# 103 "/usr/include/netinet/in.h" 3
IPPORT_DISCARD = 9,
# 104 "/usr/include/netinet/in.h" 3
IPPORT_SYSTAT = 11,
# 105 "/usr/include/netinet/in.h" 3
IPPORT_DAYTIME = 13,
# 106 "/usr/include/netinet/in.h" 3
IPPORT_NETSTAT = 15,
# 107 "/usr/include/netinet/in.h" 3
IPPORT_FTP = 21,
# 108 "/usr/include/netinet/in.h" 3
IPPORT_TELNET = 23,
# 109 "/usr/include/netinet/in.h" 3
IPPORT_SMTP = 25,
# 110 "/usr/include/netinet/in.h" 3
IPPORT_TIMESERVER = 37,
# 111 "/usr/include/netinet/in.h" 3
IPPORT_NAMESERVER = 42,
# 112 "/usr/include/netinet/in.h" 3
IPPORT_WHOIS,
# 113 "/usr/include/netinet/in.h" 3
IPPORT_MTP = 57,
# 115 "/usr/include/netinet/in.h" 3
IPPORT_TFTP = 69,
# 116 "/usr/include/netinet/in.h" 3
IPPORT_RJE = 77,
# 117 "/usr/include/netinet/in.h" 3
IPPORT_FINGER = 79,
# 118 "/usr/include/netinet/in.h" 3
IPPORT_TTYLINK = 87,
# 119 "/usr/include/netinet/in.h" 3
IPPORT_SUPDUP = 95,
# 122 "/usr/include/netinet/in.h" 3
IPPORT_EXECSERVER = 512,
# 123 "/usr/include/netinet/in.h" 3
IPPORT_LOGINSERVER,
# 124 "/usr/include/netinet/in.h" 3
IPPORT_CMDSERVER,
# 125 "/usr/include/netinet/in.h" 3
IPPORT_EFSSERVER = 520,
# 128 "/usr/include/netinet/in.h" 3
IPPORT_BIFFUDP = 512,
# 129 "/usr/include/netinet/in.h" 3
IPPORT_WHOSERVER,
# 130 "/usr/include/netinet/in.h" 3
IPPORT_ROUTESERVER = 520,
# 133 "/usr/include/netinet/in.h" 3
IPPORT_RESERVED = 1024,
# 136 "/usr/include/netinet/in.h" 3
IPPORT_USERRESERVED = 5000
# 137 "/usr/include/netinet/in.h" 3
};
# 141 "/usr/include/netinet/in.h" 3
extern "C" { typedef uint32_t in_addr_t; }
# 142 "/usr/include/netinet/in.h" 3
extern "C" { struct in_addr {
# 144 "/usr/include/netinet/in.h" 3
in_addr_t s_addr;
# 145 "/usr/include/netinet/in.h" 3
}; }
# 198 "/usr/include/netinet/in.h" 3
extern "C" { struct in6_addr {
# 201 "/usr/include/netinet/in.h" 3
union {
# 202 "/usr/include/netinet/in.h" 3
uint8_t __u6_addr8[16];
# 204 "/usr/include/netinet/in.h" 3
uint16_t __u6_addr16[8];
# 205 "/usr/include/netinet/in.h" 3
uint32_t __u6_addr32[4];
# 207 "/usr/include/netinet/in.h" 3
} __in6_u;
# 213 "/usr/include/netinet/in.h" 3
}; }
# 215 "/usr/include/netinet/in.h" 3
extern "C" { extern const in6_addr in6addr_any; }
# 216 "/usr/include/netinet/in.h" 3
extern "C" { extern const in6_addr in6addr_loopback; }
# 225 "/usr/include/netinet/in.h" 3
extern "C" { struct sockaddr_in {
# 227 "/usr/include/netinet/in.h" 3
sa_family_t sin_family;
# 228 "/usr/include/netinet/in.h" 3
in_port_t sin_port;
# 229 "/usr/include/netinet/in.h" 3
in_addr sin_addr;
# 232 "/usr/include/netinet/in.h" 3
unsigned char sin_zero[(((sizeof(sockaddr) - sizeof(unsigned short)) - sizeof(in_port_t)) - sizeof(in_addr))];
# 236 "/usr/include/netinet/in.h" 3
}; }
# 239 "/usr/include/netinet/in.h" 3
extern "C" { struct sockaddr_in6 {
# 241 "/usr/include/netinet/in.h" 3
sa_family_t sin6_family;
# 242 "/usr/include/netinet/in.h" 3
in_port_t sin6_port;
# 243 "/usr/include/netinet/in.h" 3
uint32_t sin6_flowinfo;
# 244 "/usr/include/netinet/in.h" 3
in6_addr sin6_addr;
# 245 "/usr/include/netinet/in.h" 3
uint32_t sin6_scope_id;
# 246 "/usr/include/netinet/in.h" 3
}; }
# 251 "/usr/include/netinet/in.h" 3
extern "C" { struct ip_mreq {
# 254 "/usr/include/netinet/in.h" 3
in_addr imr_multiaddr;
# 257 "/usr/include/netinet/in.h" 3
in_addr imr_interface;
# 258 "/usr/include/netinet/in.h" 3
}; }
# 260 "/usr/include/netinet/in.h" 3
extern "C" { struct ip_mreq_source {
# 263 "/usr/include/netinet/in.h" 3
in_addr imr_multiaddr;
# 266 "/usr/include/netinet/in.h" 3
in_addr imr_interface;
# 269 "/usr/include/netinet/in.h" 3
in_addr imr_sourceaddr;
# 270 "/usr/include/netinet/in.h" 3
}; }
# 275 "/usr/include/netinet/in.h" 3
extern "C" { struct ipv6_mreq {
# 278 "/usr/include/netinet/in.h" 3
in6_addr ipv6mr_multiaddr;
# 281 "/usr/include/netinet/in.h" 3
unsigned ipv6mr_interface;
# 282 "/usr/include/netinet/in.h" 3
}; }
# 287 "/usr/include/netinet/in.h" 3
extern "C" { struct group_req {
# 290 "/usr/include/netinet/in.h" 3
uint32_t gr_interface;
# 293 "/usr/include/netinet/in.h" 3
sockaddr_storage gr_group;
# 294 "/usr/include/netinet/in.h" 3
}; }
# 296 "/usr/include/netinet/in.h" 3
extern "C" { struct group_source_req {
# 299 "/usr/include/netinet/in.h" 3
uint32_t gsr_interface;
# 302 "/usr/include/netinet/in.h" 3
sockaddr_storage gsr_group;
# 305 "/usr/include/netinet/in.h" 3
sockaddr_storage gsr_source;
# 306 "/usr/include/netinet/in.h" 3
}; }
# 310 "/usr/include/netinet/in.h" 3
extern "C" { struct ip_msfilter {
# 313 "/usr/include/netinet/in.h" 3
in_addr imsf_multiaddr;
# 316 "/usr/include/netinet/in.h" 3
in_addr imsf_interface;
# 319 "/usr/include/netinet/in.h" 3
uint32_t imsf_fmode;
# 322 "/usr/include/netinet/in.h" 3
uint32_t imsf_numsrc;
# 324 "/usr/include/netinet/in.h" 3
in_addr imsf_slist[1];
# 325 "/usr/include/netinet/in.h" 3
}; }
# 331 "/usr/include/netinet/in.h" 3
extern "C" { struct group_filter {
# 334 "/usr/include/netinet/in.h" 3
uint32_t gf_interface;
# 337 "/usr/include/netinet/in.h" 3
sockaddr_storage gf_group;
# 340 "/usr/include/netinet/in.h" 3
uint32_t gf_fmode;
# 343 "/usr/include/netinet/in.h" 3
uint32_t gf_numsrc;
# 345 "/usr/include/netinet/in.h" 3
sockaddr_storage gf_slist[1];
# 346 "/usr/include/netinet/in.h" 3
}; }
# 86 "/usr/include/bits/in.h" 3
extern "C" { struct ip_opts {
# 88 "/usr/include/bits/in.h" 3
in_addr ip_dst;
# 89 "/usr/include/bits/in.h" 3
char ip_opts[40];
# 90 "/usr/include/bits/in.h" 3
}; }
# 93 "/usr/include/bits/in.h" 3
extern "C" { struct ip_mreqn {
# 95 "/usr/include/bits/in.h" 3
in_addr imr_multiaddr;
# 96 "/usr/include/bits/in.h" 3
in_addr imr_address;
# 97 "/usr/include/bits/in.h" 3
int imr_ifindex;
# 98 "/usr/include/bits/in.h" 3
}; }
# 101 "/usr/include/bits/in.h" 3
extern "C" { struct in_pktinfo {
# 103 "/usr/include/bits/in.h" 3
int ipi_ifindex;
# 104 "/usr/include/bits/in.h" 3
in_addr ipi_spec_dst;
# 105 "/usr/include/bits/in.h" 3
in_addr ipi_addr;
# 106 "/usr/include/bits/in.h" 3
}; }
# 365 "/usr/include/netinet/in.h" 3
extern "C" uint32_t ntohl(uint32_t) throw() __attribute__((__const__));
# 366 "/usr/include/netinet/in.h" 3
extern "C" uint16_t ntohs(uint16_t) throw() __attribute__((__const__));
# 368 "/usr/include/netinet/in.h" 3
extern "C" uint32_t htonl(uint32_t) throw() __attribute__((__const__));
# 370 "/usr/include/netinet/in.h" 3
extern "C" uint16_t htons(uint16_t) throw() __attribute__((__const__));
# 440 "/usr/include/netinet/in.h" 3
extern "C" int bindresvport(int, sockaddr_in *) throw();
# 443 "/usr/include/netinet/in.h" 3
extern "C" int bindresvport6(int, sockaddr_in6 *) throw();
# 471 "/usr/include/netinet/in.h" 3
extern "C" { struct in6_pktinfo {
# 473 "/usr/include/netinet/in.h" 3
in6_addr ipi6_addr;
# 474 "/usr/include/netinet/in.h" 3
unsigned ipi6_ifindex;
# 475 "/usr/include/netinet/in.h" 3
}; }
# 478 "/usr/include/netinet/in.h" 3
extern "C" { struct ip6_mtuinfo {
# 480 "/usr/include/netinet/in.h" 3
sockaddr_in6 ip6m_addr;
# 481 "/usr/include/netinet/in.h" 3
uint32_t ip6m_mtu;
# 482 "/usr/include/netinet/in.h" 3
}; }
# 486 "/usr/include/netinet/in.h" 3
extern "C" int inet6_option_space(int) throw() __attribute__((__deprecated__));
# 488 "/usr/include/netinet/in.h" 3
extern "C" int inet6_option_init(void *, cmsghdr **, int) throw() __attribute__((__deprecated__));
# 490 "/usr/include/netinet/in.h" 3
extern "C" int inet6_option_append(cmsghdr *, const uint8_t *, int, int) throw() __attribute__((__deprecated__));
# 493 "/usr/include/netinet/in.h" 3
extern "C" uint8_t *inet6_option_alloc(cmsghdr *, int, int, int) throw() __attribute__((__deprecated__));
# 496 "/usr/include/netinet/in.h" 3
extern "C" int inet6_option_next(const cmsghdr *, uint8_t **) throw() __attribute__((__deprecated__));
# 499 "/usr/include/netinet/in.h" 3
extern "C" int inet6_option_find(const cmsghdr *, uint8_t **, int) throw() __attribute__((__deprecated__));
# 505 "/usr/include/netinet/in.h" 3
extern "C" int inet6_opt_init(void *, socklen_t) throw();
# 506 "/usr/include/netinet/in.h" 3
extern "C" int inet6_opt_append(void *, socklen_t, int, uint8_t, socklen_t, uint8_t, void **) throw();
# 509 "/usr/include/netinet/in.h" 3
extern "C" int inet6_opt_finish(void *, socklen_t, int) throw();
# 511 "/usr/include/netinet/in.h" 3
extern "C" int inet6_opt_set_val(void *, int, void *, socklen_t) throw();
# 513 "/usr/include/netinet/in.h" 3
extern "C" int inet6_opt_next(void *, socklen_t, int, uint8_t *, socklen_t *, void **) throw();
# 516 "/usr/include/netinet/in.h" 3
extern "C" int inet6_opt_find(void *, socklen_t, int, uint8_t, socklen_t *, void **) throw();
# 519 "/usr/include/netinet/in.h" 3
extern "C" int inet6_opt_get_val(void *, int, void *, socklen_t) throw();
# 524 "/usr/include/netinet/in.h" 3
extern "C" socklen_t inet6_rth_space(int, int) throw();
# 525 "/usr/include/netinet/in.h" 3
extern "C" void *inet6_rth_init(void *, socklen_t, int, int) throw();
# 527 "/usr/include/netinet/in.h" 3
extern "C" int inet6_rth_add(void *, const in6_addr *) throw();
# 528 "/usr/include/netinet/in.h" 3
extern "C" int inet6_rth_reverse(const void *, void *) throw();
# 529 "/usr/include/netinet/in.h" 3
extern "C" int inet6_rth_segments(const void *) throw();
# 530 "/usr/include/netinet/in.h" 3
extern "C" in6_addr *inet6_rth_getaddr(const void *, int) throw();
# 537 "/usr/include/netinet/in.h" 3
extern "C" int getipv4sourcefilter(int, in_addr, in_addr, uint32_t *, uint32_t *, in_addr *) throw();
# 543 "/usr/include/netinet/in.h" 3
extern "C" int setipv4sourcefilter(int, in_addr, in_addr, uint32_t, uint32_t, const in_addr *) throw();
# 551 "/usr/include/netinet/in.h" 3
extern "C" int getsourcefilter(int, uint32_t, const sockaddr *, socklen_t, uint32_t *, uint32_t *, sockaddr_storage *) throw();
# 558 "/usr/include/netinet/in.h" 3
extern "C" int setsourcefilter(int, uint32_t, const sockaddr *, socklen_t, uint32_t, uint32_t, const sockaddr_storage *) throw();
# 83 "/usr/include/rpc/xdr.h" 3
enum xdr_op {
# 84 "/usr/include/rpc/xdr.h" 3
XDR_ENCODE,
# 85 "/usr/include/rpc/xdr.h" 3
XDR_DECODE,
# 86 "/usr/include/rpc/xdr.h" 3
XDR_FREE
# 87 "/usr/include/rpc/xdr.h" 3
};
# 111 "/usr/include/rpc/xdr.h" 3
extern "C" { typedef struct XDR XDR; }
# 112 "/usr/include/rpc/xdr.h" 3
extern "C" { struct XDR {
# 114 "/usr/include/rpc/xdr.h" 3
xdr_op x_op;
# 115 "/usr/include/rpc/xdr.h" 3
struct xdr_ops {
# 117 "/usr/include/rpc/xdr.h" 3
bool_t (*x_getlong)(struct XDR *, long *);
# 119 "/usr/include/rpc/xdr.h" 3
bool_t (*x_putlong)(struct XDR *, const long *);
# 121 "/usr/include/rpc/xdr.h" 3
bool_t (*x_getbytes)(struct XDR *, caddr_t, u_int);
# 123 "/usr/include/rpc/xdr.h" 3
bool_t (*x_putbytes)(struct XDR *, const char *, u_int);
# 125 "/usr/include/rpc/xdr.h" 3
u_int (*x_getpostn)(const struct XDR *);
# 127 "/usr/include/rpc/xdr.h" 3
bool_t (*x_setpostn)(struct XDR *, u_int);
# 129 "/usr/include/rpc/xdr.h" 3
int32_t *(*x_inline)(struct XDR *, u_int);
# 131 "/usr/include/rpc/xdr.h" 3
void (*x_destroy)(struct XDR *);
# 133 "/usr/include/rpc/xdr.h" 3
bool_t (*x_getint32)(struct XDR *, int32_t *);
# 135 "/usr/include/rpc/xdr.h" 3
bool_t (*x_putint32)(struct XDR *, const int32_t *);
# 138 "/usr/include/rpc/xdr.h" 3
} *x_ops;
# 139 "/usr/include/rpc/xdr.h" 3
caddr_t x_public;
# 140 "/usr/include/rpc/xdr.h" 3
caddr_t x_private;
# 141 "/usr/include/rpc/xdr.h" 3
caddr_t x_base;
# 142 "/usr/include/rpc/xdr.h" 3
u_int x_handy;
# 143 "/usr/include/rpc/xdr.h" 3
}; }
# 154 "/usr/include/rpc/xdr.h" 3
extern "C" { typedef bool_t (*xdrproc_t)(XDR *, void *, ...); }
# 234 "/usr/include/rpc/xdr.h" 3
extern "C" { struct xdr_discrim {
# 236 "/usr/include/rpc/xdr.h" 3
int value;
# 237 "/usr/include/rpc/xdr.h" 3
xdrproc_t proc;
# 238 "/usr/include/rpc/xdr.h" 3
}; }
# 287 "/usr/include/rpc/xdr.h" 3
extern "C" bool_t xdr_void() throw();
# 288 "/usr/include/rpc/xdr.h" 3
extern "C" bool_t xdr_short(XDR *, short *) throw();
# 289 "/usr/include/rpc/xdr.h" 3
extern "C" bool_t xdr_u_short(XDR *, u_short *) throw();
# 290 "/usr/include/rpc/xdr.h" 3
extern "C" bool_t xdr_int(XDR *, int *) throw();
# 291 "/usr/include/rpc/xdr.h" 3
extern "C" bool_t xdr_u_int(XDR *, u_int *) throw();
# 292 "/usr/include/rpc/xdr.h" 3
extern "C" bool_t xdr_long(XDR *, long *) throw();
# 293 "/usr/include/rpc/xdr.h" 3
extern "C" bool_t xdr_u_long(XDR *, u_long *) throw();
# 294 "/usr/include/rpc/xdr.h" 3
extern "C" bool_t xdr_hyper(XDR *, quad_t *) throw();
# 295 "/usr/include/rpc/xdr.h" 3
extern "C" bool_t xdr_u_hyper(XDR *, u_quad_t *) throw();
# 296 "/usr/include/rpc/xdr.h" 3
extern "C" bool_t xdr_longlong_t(XDR *, quad_t *) throw();
# 297 "/usr/include/rpc/xdr.h" 3
extern "C" bool_t xdr_u_longlong_t(XDR *, u_quad_t *) throw();
# 298 "/usr/include/rpc/xdr.h" 3
extern "C" bool_t xdr_int8_t(XDR *, int8_t *) throw();
# 299 "/usr/include/rpc/xdr.h" 3
extern "C" bool_t xdr_uint8_t(XDR *, uint8_t *) throw();
# 300 "/usr/include/rpc/xdr.h" 3
extern "C" bool_t xdr_int16_t(XDR *, int16_t *) throw();
# 301 "/usr/include/rpc/xdr.h" 3
extern "C" bool_t xdr_uint16_t(XDR *, uint16_t *) throw();
# 302 "/usr/include/rpc/xdr.h" 3
extern "C" bool_t xdr_int32_t(XDR *, int32_t *) throw();
# 303 "/usr/include/rpc/xdr.h" 3
extern "C" bool_t xdr_uint32_t(XDR *, uint32_t *) throw();
# 304 "/usr/include/rpc/xdr.h" 3
extern "C" bool_t xdr_int64_t(XDR *, int64_t *) throw();
# 305 "/usr/include/rpc/xdr.h" 3
extern "C" bool_t xdr_uint64_t(XDR *, uint64_t *) throw();
# 306 "/usr/include/rpc/xdr.h" 3
extern "C" bool_t xdr_quad_t(XDR *, quad_t *) throw();
# 307 "/usr/include/rpc/xdr.h" 3
extern "C" bool_t xdr_u_quad_t(XDR *, u_quad_t *) throw();
# 308 "/usr/include/rpc/xdr.h" 3
extern "C" bool_t xdr_bool(XDR *, bool_t *) throw();
# 309 "/usr/include/rpc/xdr.h" 3
extern "C" bool_t xdr_enum(XDR *, enum_t *) throw();
# 310 "/usr/include/rpc/xdr.h" 3
extern "C" bool_t xdr_array(XDR *, caddr_t *, u_int *, u_int, u_int, xdrproc_t) throw();
# 313 "/usr/include/rpc/xdr.h" 3
extern "C" bool_t xdr_bytes(XDR *, char **, u_int *, u_int) throw();
# 315 "/usr/include/rpc/xdr.h" 3
extern "C" bool_t xdr_opaque(XDR *, caddr_t, u_int) throw();
# 316 "/usr/include/rpc/xdr.h" 3
extern "C" bool_t xdr_string(XDR *, char **, u_int) throw();
# 317 "/usr/include/rpc/xdr.h" 3
extern "C" bool_t xdr_union(XDR *, enum_t *, char *, const xdr_discrim *, xdrproc_t) throw();
# 320 "/usr/include/rpc/xdr.h" 3
extern "C" bool_t xdr_char(XDR *, char *) throw();
# 321 "/usr/include/rpc/xdr.h" 3
extern "C" bool_t xdr_u_char(XDR *, u_char *) throw();
# 322 "/usr/include/rpc/xdr.h" 3
extern "C" bool_t xdr_vector(XDR *, char *, u_int, u_int, xdrproc_t) throw();
# 324 "/usr/include/rpc/xdr.h" 3
extern "C" bool_t xdr_float(XDR *, float *) throw();
# 325 "/usr/include/rpc/xdr.h" 3
extern "C" bool_t xdr_double(XDR *, double *) throw();
# 326 "/usr/include/rpc/xdr.h" 3
extern "C" bool_t xdr_reference(XDR *, caddr_t *, u_int, xdrproc_t) throw();
# 328 "/usr/include/rpc/xdr.h" 3
extern "C" bool_t xdr_pointer(XDR *, char **, u_int, xdrproc_t) throw();
# 330 "/usr/include/rpc/xdr.h" 3
extern "C" bool_t xdr_wrapstring(XDR *, char **) throw();
# 331 "/usr/include/rpc/xdr.h" 3
extern "C" u_long xdr_sizeof(xdrproc_t, void *) throw();
# 338 "/usr/include/rpc/xdr.h" 3
extern "C" { struct netobj {
# 340 "/usr/include/rpc/xdr.h" 3
u_int n_len;
# 341 "/usr/include/rpc/xdr.h" 3
char *n_bytes;
# 342 "/usr/include/rpc/xdr.h" 3
}; }
# 343 "/usr/include/rpc/xdr.h" 3
extern "C" { typedef netobj netobj; }
# 344 "/usr/include/rpc/xdr.h" 3
extern "C" bool_t xdr_netobj(XDR *, netobj *) throw();
# 352 "/usr/include/rpc/xdr.h" 3
extern "C" void xdrmem_create(XDR *, const caddr_t, u_int, xdr_op) throw();
# 356 "/usr/include/rpc/xdr.h" 3
extern "C" void xdrstdio_create(XDR *, FILE *, xdr_op) throw();
# 360 "/usr/include/rpc/xdr.h" 3
extern "C" void xdrrec_create(XDR *, u_int, u_int, caddr_t, int (*)(char *, char *, int), int (*)(char *, char *, int)) throw();
# 366 "/usr/include/rpc/xdr.h" 3
extern "C" bool_t xdrrec_endofrecord(XDR *, bool_t) throw();
# 369 "/usr/include/rpc/xdr.h" 3
extern "C" bool_t xdrrec_skiprecord(XDR *) throw();
# 372 "/usr/include/rpc/xdr.h" 3
extern "C" bool_t xdrrec_eof(XDR *) throw();
# 375 "/usr/include/rpc/xdr.h" 3
extern "C" void xdr_free(xdrproc_t, char *) throw();
# 55 "/usr/include/rpc/auth.h" 3
enum auth_stat {
# 56 "/usr/include/rpc/auth.h" 3
AUTH_OK,
# 60 "/usr/include/rpc/auth.h" 3
AUTH_BADCRED,
# 61 "/usr/include/rpc/auth.h" 3
AUTH_REJECTEDCRED,
# 62 "/usr/include/rpc/auth.h" 3
AUTH_BADVERF,
# 63 "/usr/include/rpc/auth.h" 3
AUTH_REJECTEDVERF,
# 64 "/usr/include/rpc/auth.h" 3
AUTH_TOOWEAK,
# 68 "/usr/include/rpc/auth.h" 3
AUTH_INVALIDRESP,
# 69 "/usr/include/rpc/auth.h" 3
AUTH_FAILED
# 70 "/usr/include/rpc/auth.h" 3
};
# 72 "/usr/include/rpc/auth.h" 3
extern "C" { union des_block {
# 73 "/usr/include/rpc/auth.h" 3
struct {
# 74 "/usr/include/rpc/auth.h" 3
u_int32_t high;
# 75 "/usr/include/rpc/auth.h" 3
u_int32_t low;
# 76 "/usr/include/rpc/auth.h" 3
} key;
# 77 "/usr/include/rpc/auth.h" 3
char c[8];
# 78 "/usr/include/rpc/auth.h" 3
}; }
# 79 "/usr/include/rpc/auth.h" 3
extern "C" { typedef des_block des_block; }
# 80 "/usr/include/rpc/auth.h" 3
extern "C" bool_t xdr_des_block(XDR *, des_block *) throw();
# 85 "/usr/include/rpc/auth.h" 3
extern "C" { struct opaque_auth {
# 86 "/usr/include/rpc/auth.h" 3
enum_t oa_flavor;
# 87 "/usr/include/rpc/auth.h" 3
caddr_t oa_base;
# 88 "/usr/include/rpc/auth.h" 3
u_int oa_length;
# 89 "/usr/include/rpc/auth.h" 3
}; }
# 94 "/usr/include/rpc/auth.h" 3
extern "C" { typedef struct AUTH AUTH; }
# 95 "/usr/include/rpc/auth.h" 3
extern "C" { struct AUTH {
# 96 "/usr/include/rpc/auth.h" 3
opaque_auth ah_cred;
# 97 "/usr/include/rpc/auth.h" 3
opaque_auth ah_verf;
# 98 "/usr/include/rpc/auth.h" 3
des_block ah_key;
# 99 "/usr/include/rpc/auth.h" 3
struct auth_ops {
# 100 "/usr/include/rpc/auth.h" 3
void (*ah_nextverf)(struct AUTH *);
# 101 "/usr/include/rpc/auth.h" 3
int (*ah_marshal)(struct AUTH *, XDR *);
# 102 "/usr/include/rpc/auth.h" 3
int (*ah_validate)(struct AUTH *, opaque_auth *);
# 104 "/usr/include/rpc/auth.h" 3
int (*ah_refresh)(struct AUTH *);
# 105 "/usr/include/rpc/auth.h" 3
void (*ah_destroy)(struct AUTH *);
# 106 "/usr/include/rpc/auth.h" 3
} *ah_ops;
# 107 "/usr/include/rpc/auth.h" 3
caddr_t ah_private;
# 108 "/usr/include/rpc/auth.h" 3
}; }
# 145 "/usr/include/rpc/auth.h" 3
extern "C" { extern opaque_auth _null_auth; }
# 161 "/usr/include/rpc/auth.h" 3
extern "C" AUTH *authunix_create(char *, __uid_t, __gid_t, int, __gid_t *);
# 163 "/usr/include/rpc/auth.h" 3
extern "C" AUTH *authunix_create_default();
# 164 "/usr/include/rpc/auth.h" 3
extern "C" AUTH *authnone_create() throw();
# 165 "/usr/include/rpc/auth.h" 3
extern "C" AUTH *authdes_create(const char *, u_int, sockaddr *, des_block *) throw();
# 168 "/usr/include/rpc/auth.h" 3
extern "C" AUTH *authdes_pk_create(const char *, netobj *, u_int, sockaddr *, des_block *) throw();
# 185 "/usr/include/rpc/auth.h" 3
extern "C" int getnetname(char *) throw();
# 186 "/usr/include/rpc/auth.h" 3
extern "C" int host2netname(char *, const char *, const char *) throw();
# 187 "/usr/include/rpc/auth.h" 3
extern "C" int user2netname(char *, const uid_t, const char *) throw();
# 188 "/usr/include/rpc/auth.h" 3
extern "C" int netname2user(const char *, uid_t *, gid_t *, int *, gid_t *) throw();
# 190 "/usr/include/rpc/auth.h" 3
extern "C" int netname2host(const char *, char *, const int) throw();
# 197 "/usr/include/rpc/auth.h" 3
extern "C" int key_decryptsession(char *, des_block *);
# 198 "/usr/include/rpc/auth.h" 3
extern "C" int key_decryptsession_pk(char *, netobj *, des_block *);
# 199 "/usr/include/rpc/auth.h" 3
extern "C" int key_encryptsession(char *, des_block *);
# 200 "/usr/include/rpc/auth.h" 3
extern "C" int key_encryptsession_pk(char *, netobj *, des_block *);
# 201 "/usr/include/rpc/auth.h" 3
extern "C" int key_gendes(des_block *);
# 202 "/usr/include/rpc/auth.h" 3
extern "C" int key_setsecret(char *);
# 203 "/usr/include/rpc/auth.h" 3
extern "C" int key_secretkey_is_set();
# 204 "/usr/include/rpc/auth.h" 3
extern "C" int key_get_conv(char *, des_block *);
# 209 "/usr/include/rpc/auth.h" 3
extern "C" bool_t xdr_opaque_auth(XDR *, opaque_auth *) throw();
# 30 "/usr/include/sys/un.h" 3
extern "C" { struct sockaddr_un {
# 32 "/usr/include/sys/un.h" 3
sa_family_t sun_family;
# 33 "/usr/include/sys/un.h" 3
char sun_path[108];
# 34 "/usr/include/sys/un.h" 3
}; }
# 53 "/usr/include/rpc/clnt.h" 3
enum clnt_stat {
# 54 "/usr/include/rpc/clnt.h" 3
RPC_SUCCESS,
# 58 "/usr/include/rpc/clnt.h" 3
RPC_CANTENCODEARGS,
# 59 "/usr/include/rpc/clnt.h" 3
RPC_CANTDECODERES,
# 60 "/usr/include/rpc/clnt.h" 3
RPC_CANTSEND,
# 61 "/usr/include/rpc/clnt.h" 3
RPC_CANTRECV,
# 62 "/usr/include/rpc/clnt.h" 3
RPC_TIMEDOUT,
# 66 "/usr/include/rpc/clnt.h" 3
RPC_VERSMISMATCH,
# 67 "/usr/include/rpc/clnt.h" 3
RPC_AUTHERROR,
# 68 "/usr/include/rpc/clnt.h" 3
RPC_PROGUNAVAIL,
# 69 "/usr/include/rpc/clnt.h" 3
RPC_PROGVERSMISMATCH,
# 70 "/usr/include/rpc/clnt.h" 3
RPC_PROCUNAVAIL,
# 71 "/usr/include/rpc/clnt.h" 3
RPC_CANTDECODEARGS,
# 72 "/usr/include/rpc/clnt.h" 3
RPC_SYSTEMERROR,
# 73 "/usr/include/rpc/clnt.h" 3
RPC_NOBROADCAST = 21,
# 77 "/usr/include/rpc/clnt.h" 3
RPC_UNKNOWNHOST = 13,
# 78 "/usr/include/rpc/clnt.h" 3
RPC_UNKNOWNPROTO = 17,
# 79 "/usr/include/rpc/clnt.h" 3
RPC_UNKNOWNADDR = 19,
# 84 "/usr/include/rpc/clnt.h" 3
RPC_RPCBFAILURE = 14,
# 86 "/usr/include/rpc/clnt.h" 3
RPC_PROGNOTREGISTERED,
# 87 "/usr/include/rpc/clnt.h" 3
RPC_N2AXLATEFAILURE = 22,
# 91 "/usr/include/rpc/clnt.h" 3
RPC_FAILED = 16,
# 92 "/usr/include/rpc/clnt.h" 3
RPC_INTR = 18,
# 93 "/usr/include/rpc/clnt.h" 3
RPC_TLIERROR = 20,
# 94 "/usr/include/rpc/clnt.h" 3
RPC_UDERROR = 23,
# 98 "/usr/include/rpc/clnt.h" 3
RPC_INPROGRESS,
# 99 "/usr/include/rpc/clnt.h" 3
RPC_STALERACHANDLE
# 100 "/usr/include/rpc/clnt.h" 3
};
# 106 "/usr/include/rpc/clnt.h" 3
extern "C" { struct rpc_err {
# 107 "/usr/include/rpc/clnt.h" 3
clnt_stat re_status;
# 108 "/usr/include/rpc/clnt.h" 3
union {
# 109 "/usr/include/rpc/clnt.h" 3
int RE_errno;
# 110 "/usr/include/rpc/clnt.h" 3
auth_stat RE_why;
# 111 "/usr/include/rpc/clnt.h" 3
struct {
# 112 "/usr/include/rpc/clnt.h" 3
u_long low;
# 113 "/usr/include/rpc/clnt.h" 3
u_long high;
# 114 "/usr/include/rpc/clnt.h" 3
} RE_vers;
# 115 "/usr/include/rpc/clnt.h" 3
struct {
# 116 "/usr/include/rpc/clnt.h" 3
long s1;
# 117 "/usr/include/rpc/clnt.h" 3
long s2;
# 118 "/usr/include/rpc/clnt.h" 3
} RE_lb;
# 119 "/usr/include/rpc/clnt.h" 3
} ru;
# 124 "/usr/include/rpc/clnt.h" 3
}; }
# 132 "/usr/include/rpc/clnt.h" 3
extern "C" { typedef struct CLIENT CLIENT; }
# 133 "/usr/include/rpc/clnt.h" 3
extern "C" { struct CLIENT {
# 134 "/usr/include/rpc/clnt.h" 3
AUTH *cl_auth;
# 135 "/usr/include/rpc/clnt.h" 3
struct clnt_ops {
# 136 "/usr/include/rpc/clnt.h" 3
clnt_stat (*cl_call)(struct CLIENT *, u_long, xdrproc_t, caddr_t, xdrproc_t, caddr_t, timeval);
# 139 "/usr/include/rpc/clnt.h" 3
void (*cl_abort)(void);
# 140 "/usr/include/rpc/clnt.h" 3
void (*cl_geterr)(struct CLIENT *, rpc_err *);
# 142 "/usr/include/rpc/clnt.h" 3
bool_t (*cl_freeres)(struct CLIENT *, xdrproc_t, caddr_t);
# 144 "/usr/include/rpc/clnt.h" 3
void (*cl_destroy)(struct CLIENT *);
# 145 "/usr/include/rpc/clnt.h" 3
bool_t (*cl_control)(struct CLIENT *, int, char *);
# 147 "/usr/include/rpc/clnt.h" 3
} *cl_ops;
# 148 "/usr/include/rpc/clnt.h" 3
caddr_t cl_private;
# 149 "/usr/include/rpc/clnt.h" 3
}; }
# 280 "/usr/include/rpc/clnt.h" 3
extern "C" CLIENT *clntraw_create(const u_long, const u_long) throw();
# 294 "/usr/include/rpc/clnt.h" 3
extern "C" CLIENT *clnt_create(const char *, const u_long, const u_long, const char *) throw();
# 310 "/usr/include/rpc/clnt.h" 3
extern "C" CLIENT *clnttcp_create(sockaddr_in *, u_long, u_long, int *, u_int, u_int) throw();
# 335 "/usr/include/rpc/clnt.h" 3
extern "C" CLIENT *clntudp_create(sockaddr_in *, u_long, u_long, timeval, int *) throw();
# 338 "/usr/include/rpc/clnt.h" 3
extern "C" CLIENT *clntudp_bufcreate(sockaddr_in *, u_long, u_long, timeval, int *, u_int, u_int) throw();
# 357 "/usr/include/rpc/clnt.h" 3
extern "C" CLIENT *clntunix_create(sockaddr_un *, u_long, u_long, int *, u_int, u_int) throw();
# 362 "/usr/include/rpc/clnt.h" 3
extern "C" int callrpc(const char *, const u_long, const u_long, const u_long, const xdrproc_t, const char *, const xdrproc_t, char *) throw();
# 366 "/usr/include/rpc/clnt.h" 3
extern "C" int _rpc_dtablesize() throw();
# 371 "/usr/include/rpc/clnt.h" 3
extern "C" void clnt_pcreateerror(const char *);
# 372 "/usr/include/rpc/clnt.h" 3
extern "C" char *clnt_spcreateerror(const char *) throw();
# 377 "/usr/include/rpc/clnt.h" 3
extern "C" void clnt_perrno(clnt_stat);
# 382 "/usr/include/rpc/clnt.h" 3
extern "C" void clnt_perror(CLIENT *, const char *);
# 384 "/usr/include/rpc/clnt.h" 3
extern "C" char *clnt_sperror(CLIENT *, const char *) throw();
# 390 "/usr/include/rpc/clnt.h" 3
extern "C" { struct rpc_createerr {
# 391 "/usr/include/rpc/clnt.h" 3
clnt_stat cf_stat;
# 392 "/usr/include/rpc/clnt.h" 3
rpc_err cf_error;
# 393 "/usr/include/rpc/clnt.h" 3
}; }
# 395 "/usr/include/rpc/clnt.h" 3
extern "C" { extern struct rpc_createerr rpc_createerr; }
# 402 "/usr/include/rpc/clnt.h" 3
extern "C" char *clnt_sperrno(clnt_stat) throw();
# 407 "/usr/include/rpc/clnt.h" 3
extern "C" int getrpcport(const char *, u_long, u_long, u_int) throw();
# 414 "/usr/include/rpc/clnt.h" 3
extern "C" void get_myaddress(sockaddr_in *) throw();
# 58 "/usr/include/rpc/rpc_msg.h" 3
enum msg_type {
# 59 "/usr/include/rpc/rpc_msg.h" 3
CALL,
# 60 "/usr/include/rpc/rpc_msg.h" 3
REPLY
# 61 "/usr/include/rpc/rpc_msg.h" 3
};
# 63 "/usr/include/rpc/rpc_msg.h" 3
enum reply_stat {
# 64 "/usr/include/rpc/rpc_msg.h" 3
MSG_ACCEPTED,
# 65 "/usr/include/rpc/rpc_msg.h" 3
MSG_DENIED
# 66 "/usr/include/rpc/rpc_msg.h" 3
};
# 68 "/usr/include/rpc/rpc_msg.h" 3
enum accept_stat {
# 69 "/usr/include/rpc/rpc_msg.h" 3
SUCCESS,
# 70 "/usr/include/rpc/rpc_msg.h" 3
PROG_UNAVAIL,
# 71 "/usr/include/rpc/rpc_msg.h" 3
PROG_MISMATCH,
# 72 "/usr/include/rpc/rpc_msg.h" 3
PROC_UNAVAIL,
# 73 "/usr/include/rpc/rpc_msg.h" 3
GARBAGE_ARGS,
# 74 "/usr/include/rpc/rpc_msg.h" 3
SYSTEM_ERR
# 75 "/usr/include/rpc/rpc_msg.h" 3
};
# 77 "/usr/include/rpc/rpc_msg.h" 3
enum reject_stat {
# 78 "/usr/include/rpc/rpc_msg.h" 3
RPC_MISMATCH,
# 79 "/usr/include/rpc/rpc_msg.h" 3
AUTH_ERROR
# 80 "/usr/include/rpc/rpc_msg.h" 3
};
# 91 "/usr/include/rpc/rpc_msg.h" 3
extern "C" { struct accepted_reply {
# 92 "/usr/include/rpc/rpc_msg.h" 3
opaque_auth ar_verf;
# 93 "/usr/include/rpc/rpc_msg.h" 3
accept_stat ar_stat;
# 94 "/usr/include/rpc/rpc_msg.h" 3
union {
# 95 "/usr/include/rpc/rpc_msg.h" 3
struct {
# 96 "/usr/include/rpc/rpc_msg.h" 3
u_long low;
# 97 "/usr/include/rpc/rpc_msg.h" 3
u_long high;
# 98 "/usr/include/rpc/rpc_msg.h" 3
} AR_versions;
# 99 "/usr/include/rpc/rpc_msg.h" 3
struct {
# 100 "/usr/include/rpc/rpc_msg.h" 3
caddr_t where;
# 101 "/usr/include/rpc/rpc_msg.h" 3
xdrproc_t proc;
# 102 "/usr/include/rpc/rpc_msg.h" 3
} AR_results;
# 104 "/usr/include/rpc/rpc_msg.h" 3
} ru;
# 107 "/usr/include/rpc/rpc_msg.h" 3
}; }
# 112 "/usr/include/rpc/rpc_msg.h" 3
extern "C" { struct rejected_reply {
# 113 "/usr/include/rpc/rpc_msg.h" 3
reject_stat rj_stat;
# 114 "/usr/include/rpc/rpc_msg.h" 3
union {
# 115 "/usr/include/rpc/rpc_msg.h" 3
struct {
# 116 "/usr/include/rpc/rpc_msg.h" 3
u_long low;
# 117 "/usr/include/rpc/rpc_msg.h" 3
u_long high;
# 118 "/usr/include/rpc/rpc_msg.h" 3
} RJ_versions;
# 119 "/usr/include/rpc/rpc_msg.h" 3
auth_stat RJ_why;
# 120 "/usr/include/rpc/rpc_msg.h" 3
} ru;
# 123 "/usr/include/rpc/rpc_msg.h" 3
}; }
# 128 "/usr/include/rpc/rpc_msg.h" 3
extern "C" { struct reply_body {
# 129 "/usr/include/rpc/rpc_msg.h" 3
reply_stat rp_stat;
# 130 "/usr/include/rpc/rpc_msg.h" 3
union {
# 131 "/usr/include/rpc/rpc_msg.h" 3
accepted_reply RP_ar;
# 132 "/usr/include/rpc/rpc_msg.h" 3
rejected_reply RP_dr;
# 133 "/usr/include/rpc/rpc_msg.h" 3
} ru;
# 136 "/usr/include/rpc/rpc_msg.h" 3
}; }
# 141 "/usr/include/rpc/rpc_msg.h" 3
extern "C" { struct call_body {
# 142 "/usr/include/rpc/rpc_msg.h" 3
u_long cb_rpcvers;
# 143 "/usr/include/rpc/rpc_msg.h" 3
u_long cb_prog;
# 144 "/usr/include/rpc/rpc_msg.h" 3
u_long cb_vers;
# 145 "/usr/include/rpc/rpc_msg.h" 3
u_long cb_proc;
# 146 "/usr/include/rpc/rpc_msg.h" 3
opaque_auth cb_cred;
# 147 "/usr/include/rpc/rpc_msg.h" 3
opaque_auth cb_verf;
# 148 "/usr/include/rpc/rpc_msg.h" 3
}; }
# 153 "/usr/include/rpc/rpc_msg.h" 3
extern "C" { struct rpc_msg {
# 154 "/usr/include/rpc/rpc_msg.h" 3
u_long rm_xid;
# 155 "/usr/include/rpc/rpc_msg.h" 3
msg_type rm_direction;
# 156 "/usr/include/rpc/rpc_msg.h" 3
union {
# 157 "/usr/include/rpc/rpc_msg.h" 3
call_body RM_cmb;
# 158 "/usr/include/rpc/rpc_msg.h" 3
reply_body RM_rmb;
# 159 "/usr/include/rpc/rpc_msg.h" 3
} ru;
# 162 "/usr/include/rpc/rpc_msg.h" 3
}; }
# 173 "/usr/include/rpc/rpc_msg.h" 3
extern "C" bool_t xdr_callmsg(XDR *, rpc_msg *) throw();
# 181 "/usr/include/rpc/rpc_msg.h" 3
extern "C" bool_t xdr_callhdr(XDR *, rpc_msg *) throw();
# 189 "/usr/include/rpc/rpc_msg.h" 3
extern "C" bool_t xdr_replymsg(XDR *, rpc_msg *) throw();
# 197 "/usr/include/rpc/rpc_msg.h" 3
extern "C" void _seterr_reply(rpc_msg *, rpc_err *) throw();
# 65 "/usr/include/rpc/auth_unix.h" 3
extern "C" { struct authunix_parms {
# 67 "/usr/include/rpc/auth_unix.h" 3
u_long aup_time;
# 68 "/usr/include/rpc/auth_unix.h" 3
char *aup_machname;
# 69 "/usr/include/rpc/auth_unix.h" 3
__uid_t aup_uid;
# 70 "/usr/include/rpc/auth_unix.h" 3
__gid_t aup_gid;
# 71 "/usr/include/rpc/auth_unix.h" 3
u_int aup_len;
# 72 "/usr/include/rpc/auth_unix.h" 3
__gid_t *aup_gids;
# 73 "/usr/include/rpc/auth_unix.h" 3
}; }
# 75 "/usr/include/rpc/auth_unix.h" 3
extern "C" bool_t xdr_authunix_parms(XDR *, authunix_parms *) throw();
# 83 "/usr/include/rpc/auth_unix.h" 3
extern "C" { struct short_hand_verf {
# 85 "/usr/include/rpc/auth_unix.h" 3
opaque_auth new_cred;
# 86 "/usr/include/rpc/auth_unix.h" 3
}; }
# 28 "/usr/include/rpc/auth_des.h" 3
enum authdes_namekind {
# 30 "/usr/include/rpc/auth_des.h" 3
ADN_FULLNAME,
# 31 "/usr/include/rpc/auth_des.h" 3
ADN_NICKNAME
# 32 "/usr/include/rpc/auth_des.h" 3
};
# 36 "/usr/include/rpc/auth_des.h" 3
extern "C" { struct authdes_fullname {
# 38 "/usr/include/rpc/auth_des.h" 3
char *name;
# 39 "/usr/include/rpc/auth_des.h" 3
des_block key;
# 40 "/usr/include/rpc/auth_des.h" 3
uint32_t window;
# 41 "/usr/include/rpc/auth_des.h" 3
}; }
# 44 "/usr/include/rpc/auth_des.h" 3
extern "C" { struct authdes_cred {
# 46 "/usr/include/rpc/auth_des.h" 3
authdes_namekind adc_namekind;
# 47 "/usr/include/rpc/auth_des.h" 3
authdes_fullname adc_fullname;
# 48 "/usr/include/rpc/auth_des.h" 3
uint32_t adc_nickname;
# 49 "/usr/include/rpc/auth_des.h" 3
}; }
# 52 "/usr/include/rpc/auth_des.h" 3
extern "C" { struct rpc_timeval {
# 54 "/usr/include/rpc/auth_des.h" 3
uint32_t tv_sec;
# 55 "/usr/include/rpc/auth_des.h" 3
uint32_t tv_usec;
# 56 "/usr/include/rpc/auth_des.h" 3
}; }
# 59 "/usr/include/rpc/auth_des.h" 3
extern "C" { struct authdes_verf {
# 62 "/usr/include/rpc/auth_des.h" 3
union {
# 63 "/usr/include/rpc/auth_des.h" 3
rpc_timeval adv_ctime;
# 64 "/usr/include/rpc/auth_des.h" 3
des_block adv_xtime;
# 66 "/usr/include/rpc/auth_des.h" 3
} adv_time_u;
# 67 "/usr/include/rpc/auth_des.h" 3
uint32_t adv_int_u;
# 68 "/usr/include/rpc/auth_des.h" 3
}; }
# 89 "/usr/include/rpc/auth_des.h" 3
extern "C" int authdes_getucred(const authdes_cred *, uid_t *, gid_t *, short *, gid_t *) throw();
# 96 "/usr/include/rpc/auth_des.h" 3
extern "C" int getpublickey(const char *, char *) throw();
# 103 "/usr/include/rpc/auth_des.h" 3
extern "C" int getsecretkey(const char *, char *, const char *) throw();
# 106 "/usr/include/rpc/auth_des.h" 3
extern "C" int rtime(sockaddr_in *, rpc_timeval *, rpc_timeval *) throw();
# 66 "/usr/include/rpc/svc.h" 3
enum xprt_stat {
# 67 "/usr/include/rpc/svc.h" 3
XPRT_DIED,
# 68 "/usr/include/rpc/svc.h" 3
XPRT_MOREREQS,
# 69 "/usr/include/rpc/svc.h" 3
XPRT_IDLE
# 70 "/usr/include/rpc/svc.h" 3
};
# 75 "/usr/include/rpc/svc.h" 3
extern "C" { typedef struct SVCXPRT SVCXPRT; }
# 76 "/usr/include/rpc/svc.h" 3
extern "C" { struct SVCXPRT {
# 77 "/usr/include/rpc/svc.h" 3
int xp_sock;
# 78 "/usr/include/rpc/svc.h" 3
u_short xp_port;
# 93 "/usr/include/rpc/svc.h" 3
const
# 79 "/usr/include/rpc/svc.h" 3
struct xp_ops {
# 80 "/usr/include/rpc/svc.h" 3
bool_t (*xp_recv)(struct SVCXPRT *, rpc_msg *);
# 82 "/usr/include/rpc/svc.h" 3
xprt_stat (*xp_stat)(struct SVCXPRT *);
# 84 "/usr/include/rpc/svc.h" 3
bool_t (*xp_getargs)(struct SVCXPRT *, xdrproc_t, caddr_t);
# 86 "/usr/include/rpc/svc.h" 3
bool_t (*xp_reply)(struct SVCXPRT *, rpc_msg *);
# 88 "/usr/include/rpc/svc.h" 3
bool_t (*xp_freeargs)(struct SVCXPRT *, xdrproc_t, caddr_t);
# 91 "/usr/include/rpc/svc.h" 3
void (*xp_destroy)(struct SVCXPRT *);
# 93 "/usr/include/rpc/svc.h" 3
} *xp_ops;
# 94 "/usr/include/rpc/svc.h" 3
int xp_addrlen;
# 95 "/usr/include/rpc/svc.h" 3
sockaddr_in xp_raddr;
# 96 "/usr/include/rpc/svc.h" 3
opaque_auth xp_verf;
# 97 "/usr/include/rpc/svc.h" 3
caddr_t xp_p1;
# 98 "/usr/include/rpc/svc.h" 3
caddr_t xp_p2;
# 99 "/usr/include/rpc/svc.h" 3
char xp_pad[256];
# 100 "/usr/include/rpc/svc.h" 3
}; }
# 149 "/usr/include/rpc/svc.h" 3
extern "C" { struct svc_req {
# 150 "/usr/include/rpc/svc.h" 3
rpcprog_t rq_prog;
# 151 "/usr/include/rpc/svc.h" 3
rpcvers_t rq_vers;
# 152 "/usr/include/rpc/svc.h" 3
rpcproc_t rq_proc;
# 153 "/usr/include/rpc/svc.h" 3
opaque_auth rq_cred;
# 154 "/usr/include/rpc/svc.h" 3
caddr_t rq_clntcred;
# 155 "/usr/include/rpc/svc.h" 3
SVCXPRT *rq_xprt;
# 156 "/usr/include/rpc/svc.h" 3
}; }
# 160 "/usr/include/rpc/svc.h" 3
extern "C" { typedef void (*__dispatch_fn_t)(svc_req *, SVCXPRT *); }
# 173 "/usr/include/rpc/svc.h" 3
extern "C" bool_t svc_register(SVCXPRT *, rpcprog_t, rpcvers_t, __dispatch_fn_t, rpcprot_t) throw();
# 184 "/usr/include/rpc/svc.h" 3
extern "C" void svc_unregister(rpcprog_t, rpcvers_t) throw();
# 192 "/usr/include/rpc/svc.h" 3
extern "C" void xprt_register(SVCXPRT *) throw();
# 200 "/usr/include/rpc/svc.h" 3
extern "C" void xprt_unregister(SVCXPRT *) throw();
# 229 "/usr/include/rpc/svc.h" 3
extern "C" bool_t svc_sendreply(SVCXPRT *, xdrproc_t, caddr_t) throw();
# 232 "/usr/include/rpc/svc.h" 3
extern "C" void svcerr_decode(SVCXPRT *) throw();
# 234 "/usr/include/rpc/svc.h" 3
extern "C" void svcerr_weakauth(SVCXPRT *) throw();
# 236 "/usr/include/rpc/svc.h" 3
extern "C" void svcerr_noproc(SVCXPRT *) throw();
# 238 "/usr/include/rpc/svc.h" 3
extern "C" void svcerr_progvers(SVCXPRT *, rpcvers_t, rpcvers_t) throw();
# 241 "/usr/include/rpc/svc.h" 3
extern "C" void svcerr_auth(SVCXPRT *, auth_stat) throw();
# 243 "/usr/include/rpc/svc.h" 3
extern "C" void svcerr_noprog(SVCXPRT *) throw();
# 245 "/usr/include/rpc/svc.h" 3
extern "C" void svcerr_systemerr(SVCXPRT *) throw();
# 263 "/usr/include/rpc/svc.h" 3
extern "C" { extern struct pollfd *svc_pollfd; }
# 264 "/usr/include/rpc/svc.h" 3
extern "C" { extern int svc_max_pollfd; }
# 265 "/usr/include/rpc/svc.h" 3
extern "C" { extern fd_set svc_fdset; }
# 272 "/usr/include/rpc/svc.h" 3
extern "C" void svc_getreq(int) throw();
# 273 "/usr/include/rpc/svc.h" 3
extern "C" void svc_getreq_common(const int) throw();
# 274 "/usr/include/rpc/svc.h" 3
extern "C" void svc_getreqset(fd_set *) throw();
# 275 "/usr/include/rpc/svc.h" 3
extern "C" void svc_getreq_poll(pollfd *, const int) throw();
# 276 "/usr/include/rpc/svc.h" 3
extern "C" void svc_exit() throw();
# 277 "/usr/include/rpc/svc.h" 3
extern "C" void svc_run() throw();
# 291 "/usr/include/rpc/svc.h" 3
extern "C" SVCXPRT *svcraw_create() throw();
# 296 "/usr/include/rpc/svc.h" 3
extern "C" SVCXPRT *svcudp_create(int) throw();
# 297 "/usr/include/rpc/svc.h" 3
extern "C" SVCXPRT *svcudp_bufcreate(int, u_int, u_int) throw();
# 303 "/usr/include/rpc/svc.h" 3
extern "C" SVCXPRT *svctcp_create(int, u_int, u_int) throw();
# 309 "/usr/include/rpc/svc.h" 3
extern "C" SVCXPRT *svcfd_create(int, u_int, u_int) throw();
# 315 "/usr/include/rpc/svc.h" 3
extern "C" SVCXPRT *svcunix_create(int, u_int, u_int, char *) throw();
# 49 "/usr/include/rpc/svc_auth.h" 3
extern "C" auth_stat _authenticate(svc_req *, rpc_msg *) throw();
# 46 "/usr/include/rpc/netdb.h" 3
extern "C" { struct rpcent {
# 48 "/usr/include/rpc/netdb.h" 3
char *r_name;
# 49 "/usr/include/rpc/netdb.h" 3
char **r_aliases;
# 50 "/usr/include/rpc/netdb.h" 3
int r_number;
# 51 "/usr/include/rpc/netdb.h" 3
}; }
# 53 "/usr/include/rpc/netdb.h" 3
extern "C" void setrpcent(int) throw();
# 54 "/usr/include/rpc/netdb.h" 3
extern "C" void endrpcent() throw();
# 55 "/usr/include/rpc/netdb.h" 3
extern "C" rpcent *getrpcbyname(const char *) throw();
# 56 "/usr/include/rpc/netdb.h" 3
extern "C" rpcent *getrpcbynumber(int) throw();
# 57 "/usr/include/rpc/netdb.h" 3
extern "C" rpcent *getrpcent() throw();
# 60 "/usr/include/rpc/netdb.h" 3
extern "C" int getrpcbyname_r(const char *, rpcent *, char *, size_t, rpcent **) throw();
# 64 "/usr/include/rpc/netdb.h" 3
extern "C" int getrpcbynumber_r(int, rpcent *, char *, size_t, rpcent **) throw();
# 68 "/usr/include/rpc/netdb.h" 3
extern "C" int getrpcent_r(rpcent *, char *, size_t, rpcent **) throw();
# 73 "/usr/include/rpc/rpc.h" 3
extern "C" fd_set *__rpc_thread_svc_fdset() __attribute__((__const__));
# 76 "/usr/include/rpc/rpc.h" 3
extern "C" struct rpc_createerr *__rpc_thread_createerr() __attribute__((__const__));
# 88 "/usr/include/rpc/rpc.h" 3
extern "C" pollfd **__rpc_thread_svc_pollfd() __attribute__((__const__));
# 92 "/usr/include/rpc/rpc.h" 3
extern "C" int *__rpc_thread_svc_max_pollfd() __attribute__((__const__));
# 17 "../../bin/../include/rcudarpc.h"
extern "C" { typedef u_long RCadr; }
# 19 "../../bin/../include/rcudarpc.h"
extern "C" { typedef u_int RCsize; }
# 21 "../../bin/../include/rcudarpc.h"
extern "C" { typedef u_int RCerror; }
# 26 "../../bin/../include/rcudarpc.h"
extern "C" { typedef
# 23 "../../bin/../include/rcudarpc.h"
struct RCbuf {
# 24 "../../bin/../include/rcudarpc.h"
u_int RCbuf_len;
# 25 "../../bin/../include/rcudarpc.h"
char *RCbuf_val;
# 26 "../../bin/../include/rcudarpc.h"
} RCbuf; }
# 28 "../../bin/../include/rcudarpc.h"
enum RCargType {
# 29 "../../bin/../include/rcudarpc.h"
rcudaArgTypeV,
# 30 "../../bin/../include/rcudarpc.h"
rcudaArgTypeI,
# 31 "../../bin/../include/rcudarpc.h"
rcudaArgTypeF
# 32 "../../bin/../include/rcudarpc.h"
};
# 33 "../../bin/../include/rcudarpc.h"
extern "C" { typedef RCargType RCargType; }
# 35 "../../bin/../include/rcudarpc.h"
extern "C" { struct RCargVal {
# 36 "../../bin/../include/rcudarpc.h"
RCargType type;
# 37 "../../bin/../include/rcudarpc.h"
union {
# 38 "../../bin/../include/rcudarpc.h"
RCadr address;
# 39 "../../bin/../include/rcudarpc.h"
u_int valuei;
# 40 "../../bin/../include/rcudarpc.h"
float valuef;
# 41 "../../bin/../include/rcudarpc.h"
} RCargVal_u;
# 42 "../../bin/../include/rcudarpc.h"
}; }
# 43 "../../bin/../include/rcudarpc.h"
extern "C" { typedef RCargVal RCargVal; }
# 45 "../../bin/../include/rcudarpc.h"
extern "C" { struct RCarg {
# 46 "../../bin/../include/rcudarpc.h"
RCargVal val;
# 47 "../../bin/../include/rcudarpc.h"
u_int offset;
# 48 "../../bin/../include/rcudarpc.h"
u_int size;
# 49 "../../bin/../include/rcudarpc.h"
}; }
# 50 "../../bin/../include/rcudarpc.h"
extern "C" { typedef RCarg RCarg; }
# 55 "../../bin/../include/rcudarpc.h"
extern "C" { typedef
# 52 "../../bin/../include/rcudarpc.h"
struct RCargs {
# 53 "../../bin/../include/rcudarpc.h"
u_int RCargs_len;
# 54 "../../bin/../include/rcudarpc.h"
RCarg *RCargs_val;
# 55 "../../bin/../include/rcudarpc.h"
} RCargs; }
# 57 "../../bin/../include/rcudarpc.h"
extern "C" { struct rcudaMallocResult {
# 58 "../../bin/../include/rcudarpc.h"
RCerror err;
# 59 "../../bin/../include/rcudarpc.h"
RCadr devAdr;
# 60 "../../bin/../include/rcudarpc.h"
}; }
# 61 "../../bin/../include/rcudarpc.h"
extern "C" { typedef rcudaMallocResult rcudaMallocResult; }
# 63 "../../bin/../include/rcudarpc.h"
extern "C" { struct rcudaMemcpyH2HResult {
# 64 "../../bin/../include/rcudarpc.h"
RCerror err;
# 65 "../../bin/../include/rcudarpc.h"
}; }
# 66 "../../bin/../include/rcudarpc.h"
extern "C" { typedef rcudaMemcpyH2HResult rcudaMemcpyH2HResult; }
# 68 "../../bin/../include/rcudarpc.h"
extern "C" { struct rcudaMemcpyH2DResult {
# 69 "../../bin/../include/rcudarpc.h"
RCerror err;
# 70 "../../bin/../include/rcudarpc.h"
}; }
# 71 "../../bin/../include/rcudarpc.h"
extern "C" { typedef rcudaMemcpyH2DResult rcudaMemcpyH2DResult; }
# 73 "../../bin/../include/rcudarpc.h"
extern "C" { struct rcudaMemcpyD2HResult {
# 74 "../../bin/../include/rcudarpc.h"
RCerror err;
# 75 "../../bin/../include/rcudarpc.h"
RCbuf buf;
# 76 "../../bin/../include/rcudarpc.h"
}; }
# 77 "../../bin/../include/rcudarpc.h"
extern "C" { typedef rcudaMemcpyD2HResult rcudaMemcpyD2HResult; }
# 79 "../../bin/../include/rcudarpc.h"
extern "C" { struct rcudaMemcpyD2DResult {
# 80 "../../bin/../include/rcudarpc.h"
RCerror err;
# 81 "../../bin/../include/rcudarpc.h"
}; }
# 82 "../../bin/../include/rcudarpc.h"
extern "C" { typedef rcudaMemcpyD2DResult rcudaMemcpyD2DResult; }
# 84 "../../bin/../include/rcudarpc.h"
extern "C" { struct rcudaFreeResult {
# 85 "../../bin/../include/rcudarpc.h"
RCerror err;
# 86 "../../bin/../include/rcudarpc.h"
}; }
# 87 "../../bin/../include/rcudarpc.h"
extern "C" { typedef rcudaFreeResult rcudaFreeResult; }
# 89 "../../bin/../include/rcudarpc.h"
extern "C" { struct RCdim3 {
# 90 "../../bin/../include/rcudarpc.h"
u_int x;
# 91 "../../bin/../include/rcudarpc.h"
u_int y;
# 92 "../../bin/../include/rcudarpc.h"
u_int z;
# 93 "../../bin/../include/rcudarpc.h"
}; }
# 94 "../../bin/../include/rcudarpc.h"
extern "C" { typedef RCdim3 RCdim3; }
# 96 "../../bin/../include/rcudarpc.h"
extern "C" { struct rcudamemcpyh2hid_1_argument {
# 97 "../../bin/../include/rcudarpc.h"
RCadr dst;
# 98 "../../bin/../include/rcudarpc.h"
RCadr src;
# 99 "../../bin/../include/rcudarpc.h"
RCsize count;
# 100 "../../bin/../include/rcudarpc.h"
}; }
# 101 "../../bin/../include/rcudarpc.h"
extern "C" { typedef rcudamemcpyh2hid_1_argument rcudamemcpyh2hid_1_argument; }
# 103 "../../bin/../include/rcudarpc.h"
extern "C" { struct rcudamemcpyh2did_1_argument {
# 104 "../../bin/../include/rcudarpc.h"
RCadr dst;
# 105 "../../bin/../include/rcudarpc.h"
RCbuf src;
# 106 "../../bin/../include/rcudarpc.h"
RCsize count;
# 107 "../../bin/../include/rcudarpc.h"
}; }
# 108 "../../bin/../include/rcudarpc.h"
extern "C" { typedef rcudamemcpyh2did_1_argument rcudamemcpyh2did_1_argument; }
# 110 "../../bin/../include/rcudarpc.h"
extern "C" { struct rcudamemcpyd2hid_1_argument {
# 111 "../../bin/../include/rcudarpc.h"
RCadr src;
# 112 "../../bin/../include/rcudarpc.h"
RCsize count;
# 113 "../../bin/../include/rcudarpc.h"
}; }
# 114 "../../bin/../include/rcudarpc.h"
extern "C" { typedef rcudamemcpyd2hid_1_argument rcudamemcpyd2hid_1_argument; }
# 116 "../../bin/../include/rcudarpc.h"
extern "C" { struct rcudamemcpyd2did_1_argument {
# 117 "../../bin/../include/rcudarpc.h"
RCadr dst;
# 118 "../../bin/../include/rcudarpc.h"
RCadr src;
# 119 "../../bin/../include/rcudarpc.h"
RCsize count;
# 120 "../../bin/../include/rcudarpc.h"
}; }
# 121 "../../bin/../include/rcudarpc.h"
extern "C" { typedef rcudamemcpyd2did_1_argument rcudamemcpyd2did_1_argument; }
# 123 "../../bin/../include/rcudarpc.h"
extern "C" { struct rcudalaunchkernel_1_argument {
# 124 "../../bin/../include/rcudarpc.h"
u_int kid;
# 125 "../../bin/../include/rcudarpc.h"
char *kname;
# 126 "../../bin/../include/rcudarpc.h"
char *kimage;
# 127 "../../bin/../include/rcudarpc.h"
RCdim3 gdim;
# 128 "../../bin/../include/rcudarpc.h"
RCdim3 bdim;
# 129 "../../bin/../include/rcudarpc.h"
RCargs args;
# 130 "../../bin/../include/rcudarpc.h"
}; }
# 131 "../../bin/../include/rcudarpc.h"
extern "C" { typedef rcudalaunchkernel_1_argument rcudalaunchkernel_1_argument; }
# 138 "../../bin/../include/rcudarpc.h"
extern "C" rcudaMallocResult *rcudamallocid_1(RCsize, CLIENT *);
# 139 "../../bin/../include/rcudarpc.h"
extern "C" rcudaMallocResult *rcudamallocid_1_svc(RCsize, svc_req *);
# 141 "../../bin/../include/rcudarpc.h"
extern "C" rcudaMemcpyH2HResult *rcudamemcpyh2hid_1(RCadr, RCadr, RCsize, CLIENT *);
# 142 "../../bin/../include/rcudarpc.h"
extern "C" rcudaMemcpyH2HResult *rcudamemcpyh2hid_1_svc(RCadr, RCadr, RCsize, svc_req *);
# 144 "../../bin/../include/rcudarpc.h"
extern "C" rcudaMemcpyH2DResult *rcudamemcpyh2did_1(RCadr, RCbuf, RCsize, CLIENT *);
# 145 "../../bin/../include/rcudarpc.h"
extern "C" rcudaMemcpyH2DResult *rcudamemcpyh2did_1_svc(RCadr, RCbuf, RCsize, svc_req *);
# 147 "../../bin/../include/rcudarpc.h"
extern "C" rcudaMemcpyD2HResult *rcudamemcpyd2hid_1(RCadr, RCsize, CLIENT *);
# 148 "../../bin/../include/rcudarpc.h"
extern "C" rcudaMemcpyD2HResult *rcudamemcpyd2hid_1_svc(RCadr, RCsize, svc_req *);
# 150 "../../bin/../include/rcudarpc.h"
extern "C" rcudaMemcpyD2DResult *rcudamemcpyd2did_1(RCadr, RCadr, RCsize, CLIENT *);
# 151 "../../bin/../include/rcudarpc.h"
extern "C" rcudaMemcpyD2DResult *rcudamemcpyd2did_1_svc(RCadr, RCadr, RCsize, svc_req *);
# 153 "../../bin/../include/rcudarpc.h"
extern "C" void *rcudalaunchkernel_1(u_int, char *, char *, RCdim3, RCdim3, RCargs, CLIENT *);
# 154 "../../bin/../include/rcudarpc.h"
extern "C" void *rcudalaunchkernel_1_svc(u_int, char *, char *, RCdim3, RCdim3, RCargs, svc_req *);
# 156 "../../bin/../include/rcudarpc.h"
extern "C" rcudaFreeResult *rcudafreeid_1(RCadr, CLIENT *);
# 157 "../../bin/../include/rcudarpc.h"
extern "C" rcudaFreeResult *rcudafreeid_1_svc(RCadr, svc_req *);
# 158 "../../bin/../include/rcudarpc.h"
extern "C" int rcuda_prog_1_freeresult(SVCXPRT *, xdrproc_t, caddr_t);
# 188 "../../bin/../include/rcudarpc.h"
extern "C" bool_t xdr_RCadr(XDR *, RCadr *);
# 189 "../../bin/../include/rcudarpc.h"
extern "C" bool_t xdr_RCsize(XDR *, RCsize *);
# 190 "../../bin/../include/rcudarpc.h"
extern "C" bool_t xdr_RCerror(XDR *, RCerror *);
# 191 "../../bin/../include/rcudarpc.h"
extern "C" bool_t xdr_RCbuf(XDR *, RCbuf *);
# 192 "../../bin/../include/rcudarpc.h"
extern "C" bool_t xdr_RCargType(XDR *, RCargType *);
# 193 "../../bin/../include/rcudarpc.h"
extern "C" bool_t xdr_RCargVal(XDR *, RCargVal *);
# 194 "../../bin/../include/rcudarpc.h"
extern "C" bool_t xdr_RCarg(XDR *, RCarg *);
# 195 "../../bin/../include/rcudarpc.h"
extern "C" bool_t xdr_RCargs(XDR *, RCargs *);
# 196 "../../bin/../include/rcudarpc.h"
extern "C" bool_t xdr_rcudaMallocResult(XDR *, rcudaMallocResult *);
# 197 "../../bin/../include/rcudarpc.h"
extern "C" bool_t xdr_rcudaMemcpyH2HResult(XDR *, rcudaMemcpyH2HResult *);
# 198 "../../bin/../include/rcudarpc.h"
extern "C" bool_t xdr_rcudaMemcpyH2DResult(XDR *, rcudaMemcpyH2DResult *);
# 199 "../../bin/../include/rcudarpc.h"
extern "C" bool_t xdr_rcudaMemcpyD2HResult(XDR *, rcudaMemcpyD2HResult *);
# 200 "../../bin/../include/rcudarpc.h"
extern "C" bool_t xdr_rcudaMemcpyD2DResult(XDR *, rcudaMemcpyD2DResult *);
# 201 "../../bin/../include/rcudarpc.h"
extern "C" bool_t xdr_rcudaFreeResult(XDR *, rcudaFreeResult *);
# 202 "../../bin/../include/rcudarpc.h"
extern "C" bool_t xdr_RCdim3(XDR *, RCdim3 *);
# 203 "../../bin/../include/rcudarpc.h"
extern "C" bool_t xdr_rcudamemcpyh2hid_1_argument(XDR *, rcudamemcpyh2hid_1_argument *);
# 204 "../../bin/../include/rcudarpc.h"
extern "C" bool_t xdr_rcudamemcpyh2did_1_argument(XDR *, rcudamemcpyh2did_1_argument *);
# 205 "../../bin/../include/rcudarpc.h"
extern "C" bool_t xdr_rcudamemcpyd2hid_1_argument(XDR *, rcudamemcpyd2hid_1_argument *);
# 206 "../../bin/../include/rcudarpc.h"
extern "C" bool_t xdr_rcudamemcpyd2did_1_argument(XDR *, rcudamemcpyd2did_1_argument *);
# 207 "../../bin/../include/rcudarpc.h"
extern "C" bool_t xdr_rcudalaunchkernel_1_argument(XDR *, rcudalaunchkernel_1_argument *);
# 11 "../../bin/../include/rcuda.h"
extern CLIENT *rcudaClient();
# 12 "../../bin/../include/rcuda.h"
extern char *rcudaLoadImage(char *);
# 13 "../../bin/../include/rcuda.h"
extern cudaError_t rcudaMalloc(void **, size_t);
# 14 "../../bin/../include/rcuda.h"
extern cudaError_t rcudaFree(void *);
# 15 "../../bin/../include/rcuda.h"
extern cudaError_t rcudaMemcpy(void *, const void *, size_t, cudaMemcpyKind);
# 16 "../../bin/../include/rcuda.h"
extern char *rcudaMemcpyKindName(cudaMemcpyKind);
# 6 "./rcudatmp/mr3.rcu.cu"
extern void rcudanacl_kernel(dim3, dim3, float *, int, int *, int, float *, float *, float *, float *, float *, float *, int, float, int, float *);
# 16 "./rcudatmp/mr3.rcu.cu"
extern "C" void MR3init()
# 17 "./rcudatmp/mr3.rcu.cu"
{
# 18 "./rcudatmp/mr3.rcu.cu"
}
# 21 "./rcudatmp/mr3.rcu.cu"
extern "C" void MR3free()
# 22 "./rcudatmp/mr3.rcu.cu"
{
# 23 "./rcudatmp/mr3.rcu.cu"
}

# 1 "/tmp/tmpxft_00001cd2_00000000-1_mr3.rcu.cudafe1.stub.h" 1

extern "C" {


extern void __device_stub_nacl_kernel(float *, int, int *, int, float *, float *, float *, float *, float *, float *, int, float, int, float *);


}
# 26 "./rcudatmp/mr3.rcu.cu" 2
# 91 "./rcudatmp/mr3.rcu.cu"
extern "C" void MR3calcnacl(double x[], int n, int atype[], int nat, double
# 92 "./rcudatmp/mr3.rcu.cu"
pol[], double sigm[], double ipotro[], double
# 93 "./rcudatmp/mr3.rcu.cu"
pc[], double pd[], double zz[], int
# 94 "./rcudatmp/mr3.rcu.cu"
tblno, double xmax, int periodicflag, double
# 95 "./rcudatmp/mr3.rcu.cu"
force[])
# 96 "./rcudatmp/mr3.rcu.cu"
{
# 97 "./rcudatmp/mr3.rcu.cu"
auto int i;
# 98 "./rcudatmp/mr3.rcu.cu"
auto float xmaxf = (xmax);
# 99 "./rcudatmp/mr3.rcu.cu"
static int *d_atype;
# 100 "./rcudatmp/mr3.rcu.cu"
static float *d_x; static float *d_pol; static float *d_sigm; static float *d_ipotro; static float *d_pc; static float *d_pd; static float *d_zz; static float *d_force;
# 101 "./rcudatmp/mr3.rcu.cu"
static int firstcall = 1;
# 104 "./rcudatmp/mr3.rcu.cu"
if (((sizeof(double) * n) * (3)) < ((sizeof(float) * nat) * nat)) {
# 105 "./rcudatmp/mr3.rcu.cu"
fprintf(stderr, "** error : n*3<nat*nat **\n");
# 106 "./rcudatmp/mr3.rcu.cu"
exit(1);
# 107 "./rcudatmp/mr3.rcu.cu"
}
# 112 "./rcudatmp/mr3.rcu.cu"
if (firstcall) {
# 113 "./rcudatmp/mr3.rcu.cu"
firstcall = 0;
# 114 "./rcudatmp/mr3.rcu.cu"
rcudaMalloc((void **)(&d_x), (sizeof(float) * n) * (3));
# 115 "./rcudatmp/mr3.rcu.cu"
rcudaMalloc((void **)(&d_pol), (sizeof(float) * nat) * nat);
# 116 "./rcudatmp/mr3.rcu.cu"
rcudaMalloc((void **)(&d_sigm), (sizeof(float) * nat) * nat);
# 117 "./rcudatmp/mr3.rcu.cu"
rcudaMalloc((void **)(&d_ipotro), (sizeof(float) * nat) * nat);
# 118 "./rcudatmp/mr3.rcu.cu"
rcudaMalloc((void **)(&d_pc), (sizeof(float) * nat) * nat);
# 119 "./rcudatmp/mr3.rcu.cu"
rcudaMalloc((void **)(&d_pd), (sizeof(float) * nat) * nat);
# 120 "./rcudatmp/mr3.rcu.cu"
rcudaMalloc((void **)(&d_zz), (sizeof(float) * nat) * nat);
# 121 "./rcudatmp/mr3.rcu.cu"
rcudaMalloc((void **)(&d_atype), sizeof(int) * n);
# 122 "./rcudatmp/mr3.rcu.cu"
rcudaMalloc((void **)(&d_force), (sizeof(float) * n) * (3));
# 123 "./rcudatmp/mr3.rcu.cu"
}
# 126 "./rcudatmp/mr3.rcu.cu"
for (int i = 0; i < (n * 3); i++) { (((float *)force)[i]) = x[i]; } rcudaMemcpy(d_x, force, sizeof(float) * (n * 3), cudaMemcpyHostToDevice); ;
# 127 "./rcudatmp/mr3.rcu.cu"
for (int i = 0; i < (nat * nat); i++) { (((float *)force)[i]) = pol[i]; } rcudaMemcpy(d_pol, force, sizeof(float) * (nat * nat), cudaMemcpyHostToDevice); ;
# 128 "./rcudatmp/mr3.rcu.cu"
for (int i = 0; i < (nat * nat); i++) { (((float *)force)[i]) = sigm[i]; } rcudaMemcpy(d_sigm, force, sizeof(float) * (nat * nat), cudaMemcpyHostToDevice); ;
# 129 "./rcudatmp/mr3.rcu.cu"
for (int i = 0; i < (nat * nat); i++) { (((float *)force)[i]) = ipotro[i]; } rcudaMemcpy(d_ipotro, force, sizeof(float) * (nat * nat), cudaMemcpyHostToDevice); ;
# 130 "./rcudatmp/mr3.rcu.cu"
for (int i = 0; i < (nat * nat); i++) { (((float *)force)[i]) = pc[i]; } rcudaMemcpy(d_pc, force, sizeof(float) * (nat * nat), cudaMemcpyHostToDevice); ;
# 131 "./rcudatmp/mr3.rcu.cu"
for (int i = 0; i < (nat * nat); i++) { (((float *)force)[i]) = pd[i]; } rcudaMemcpy(d_pd, force, sizeof(float) * (nat * nat), cudaMemcpyHostToDevice); ;
# 132 "./rcudatmp/mr3.rcu.cu"
for (int i = 0; i < (nat * nat); i++) { (((float *)force)[i]) = zz[i]; } rcudaMemcpy(d_zz, force, sizeof(float) * (nat * nat), cudaMemcpyHostToDevice); ;
# 133 "./rcudatmp/mr3.rcu.cu"
rcudaMemcpy(d_atype, atype, sizeof(int) * n, cudaMemcpyHostToDevice);
# 136 "./rcudatmp/mr3.rcu.cu"
auto dim3 threads(64, 1, 1);
# 137 "./rcudatmp/mr3.rcu.cu"
auto dim3 grid((n + 63) / 64, 1, 1);
# 138 "./rcudatmp/mr3.rcu.cu"
rcudanacl_kernel(grid, threads, d_x, n, d_atype, nat, d_pol, d_sigm, d_ipotro, d_pc, d_pd, d_zz, tblno, xmaxf, periodicflag, d_force);
# 140 "./rcudatmp/mr3.rcu.cu"
;
# 143 "./rcudatmp/mr3.rcu.cu"
rcudaMemcpy(force, d_force, (sizeof(float) * n) * (3), cudaMemcpyDeviceToHost);
# 144 "./rcudatmp/mr3.rcu.cu"
for (i = n * 3 - 1; i >= 0; i--) { (force[i]) = ((float *)force)[i]; }
# 193 "./rcudatmp/mr3.rcu.cu"
}
# 199 "./rcudatmp/mr3.rcu.cu"
void rcudanacl_kernel(dim3 gdim, dim3 bdim, float *x, int n, int *atype, int nat, float *pol, float *sigm, float *ipotro, float *
# 200 "./rcudatmp/mr3.rcu.cu"
pc, float *pd, float *zz, int tblno, float xmax, int periodicflag, float *
# 201 "./rcudatmp/mr3.rcu.cu"
force)
# 202 "./rcudatmp/mr3.rcu.cu"
{
# 203 "./rcudatmp/mr3.rcu.cu"
static int firstcall = 1;
# 204 "./rcudatmp/mr3.rcu.cu"
auto char *kimage = (__null);
# 205 "./rcudatmp/mr3.rcu.cu"
auto RCargs args;
# 206 "./rcudatmp/mr3.rcu.cu"
auto RCarg arg[14];
# 207 "./rcudatmp/mr3.rcu.cu"
auto RCarg *argp;
# 208 "./rcudatmp/mr3.rcu.cu"
auto RCdim3 gdimrc; auto RCdim3 bdimrc;
# 209 "./rcudatmp/mr3.rcu.cu"
auto int off = 0;
# 210 "./rcudatmp/mr3.rcu.cu"
auto int argc = 0;
# 211 "./rcudatmp/mr3.rcu.cu"
auto void *p;
# 212 "./rcudatmp/mr3.rcu.cu"
(args.RCargs_val) = arg;
# 213 "./rcudatmp/mr3.rcu.cu"
(args.RCargs_len) = (14);
# 214 "./rcudatmp/mr3.rcu.cu"
if (firstcall) {
# 215 "./rcudatmp/mr3.rcu.cu"
firstcall = 0;
# 216 "./rcudatmp/mr3.rcu.cu"
kimage = rcudaLoadImage((char *)("mr3.ptx"));
# 217 "./rcudatmp/mr3.rcu.cu"
}
# 221 "./rcudatmp/mr3.rcu.cu"
argp = args.RCargs_val + argc++;
# 222 "./rcudatmp/mr3.rcu.cu"
p = (void *)((size_t)x);
# 223 "./rcudatmp/mr3.rcu.cu"
off = ((off + 8UL) - (1)) & (~(8UL - (1)));
# 224 "./rcudatmp/mr3.rcu.cu"
((argp->val).type) = rcudaArgTypeV;
# 225 "./rcudatmp/mr3.rcu.cu"
(argp->offset) = off;
# 226 "./rcudatmp/mr3.rcu.cu"
(((argp->val).RCargVal_u).address) = (RCadr)p;
# 227 "./rcudatmp/mr3.rcu.cu"
(argp->size) = (sizeof(p));
# 228 "./rcudatmp/mr3.rcu.cu"
off += sizeof(p);
# 231 "./rcudatmp/mr3.rcu.cu"
argp = args.RCargs_val + argc++;
# 232 "./rcudatmp/mr3.rcu.cu"
off = ((off + 4UL) - (1)) & (~(4UL - (1)));
# 233 "./rcudatmp/mr3.rcu.cu"
((argp->val).type) = rcudaArgTypeI;
# 234 "./rcudatmp/mr3.rcu.cu"
(argp->offset) = off;
# 235 "./rcudatmp/mr3.rcu.cu"
(((argp->val).RCargVal_u).valuei) = n;
# 236 "./rcudatmp/mr3.rcu.cu"
(argp->size) = (sizeof(int));
# 237 "./rcudatmp/mr3.rcu.cu"
off += sizeof(int);
# 240 "./rcudatmp/mr3.rcu.cu"
argp = args.RCargs_val + argc++;
# 241 "./rcudatmp/mr3.rcu.cu"
p = (void *)((size_t)atype);
# 242 "./rcudatmp/mr3.rcu.cu"
off = ((off + 8UL) - (1)) & (~(8UL - (1)));
# 243 "./rcudatmp/mr3.rcu.cu"
((argp->val).type) = rcudaArgTypeV;
# 244 "./rcudatmp/mr3.rcu.cu"
(argp->offset) = off;
# 245 "./rcudatmp/mr3.rcu.cu"
(((argp->val).RCargVal_u).address) = (RCadr)p;
# 246 "./rcudatmp/mr3.rcu.cu"
(argp->size) = (sizeof(p));
# 247 "./rcudatmp/mr3.rcu.cu"
off += sizeof(p);
# 250 "./rcudatmp/mr3.rcu.cu"
argp = args.RCargs_val + argc++;
# 251 "./rcudatmp/mr3.rcu.cu"
off = ((off + 4UL) - (1)) & (~(4UL - (1)));
# 252 "./rcudatmp/mr3.rcu.cu"
((argp->val).type) = rcudaArgTypeI;
# 253 "./rcudatmp/mr3.rcu.cu"
(argp->offset) = off;
# 254 "./rcudatmp/mr3.rcu.cu"
(((argp->val).RCargVal_u).valuei) = nat;
# 255 "./rcudatmp/mr3.rcu.cu"
(argp->size) = (sizeof(int));
# 256 "./rcudatmp/mr3.rcu.cu"
off += sizeof(int);
# 259 "./rcudatmp/mr3.rcu.cu"
argp = args.RCargs_val + argc++;
# 260 "./rcudatmp/mr3.rcu.cu"
p = (void *)((size_t)pol);
# 261 "./rcudatmp/mr3.rcu.cu"
off = ((off + 8UL) - (1)) & (~(8UL - (1)));
# 262 "./rcudatmp/mr3.rcu.cu"
((argp->val).type) = rcudaArgTypeV;
# 263 "./rcudatmp/mr3.rcu.cu"
(argp->offset) = off;
# 264 "./rcudatmp/mr3.rcu.cu"
(((argp->val).RCargVal_u).address) = (RCadr)p;
# 265 "./rcudatmp/mr3.rcu.cu"
(argp->size) = (sizeof(p));
# 266 "./rcudatmp/mr3.rcu.cu"
off += sizeof(p);
# 269 "./rcudatmp/mr3.rcu.cu"
argp = args.RCargs_val + argc++;
# 270 "./rcudatmp/mr3.rcu.cu"
p = (void *)((size_t)sigm);
# 271 "./rcudatmp/mr3.rcu.cu"
off = ((off + 8UL) - (1)) & (~(8UL - (1)));
# 272 "./rcudatmp/mr3.rcu.cu"
((argp->val).type) = rcudaArgTypeV;
# 273 "./rcudatmp/mr3.rcu.cu"
(argp->offset) = off;
# 274 "./rcudatmp/mr3.rcu.cu"
(((argp->val).RCargVal_u).address) = (RCadr)p;
# 275 "./rcudatmp/mr3.rcu.cu"
(argp->size) = (sizeof(p));
# 276 "./rcudatmp/mr3.rcu.cu"
off += sizeof(p);
# 279 "./rcudatmp/mr3.rcu.cu"
argp = args.RCargs_val + argc++;
# 280 "./rcudatmp/mr3.rcu.cu"
p = (void *)((size_t)ipotro);
# 281 "./rcudatmp/mr3.rcu.cu"
off = ((off + 8UL) - (1)) & (~(8UL - (1)));
# 282 "./rcudatmp/mr3.rcu.cu"
((argp->val).type) = rcudaArgTypeV;
# 283 "./rcudatmp/mr3.rcu.cu"
(argp->offset) = off;
# 284 "./rcudatmp/mr3.rcu.cu"
(((argp->val).RCargVal_u).address) = (RCadr)p;
# 285 "./rcudatmp/mr3.rcu.cu"
(argp->size) = (sizeof(p));
# 286 "./rcudatmp/mr3.rcu.cu"
off += sizeof(p);
# 289 "./rcudatmp/mr3.rcu.cu"
argp = args.RCargs_val + argc++;
# 290 "./rcudatmp/mr3.rcu.cu"
p = (void *)((size_t)pc);
# 291 "./rcudatmp/mr3.rcu.cu"
off = ((off + 8UL) - (1)) & (~(8UL - (1)));
# 292 "./rcudatmp/mr3.rcu.cu"
((argp->val).type) = rcudaArgTypeV;
# 293 "./rcudatmp/mr3.rcu.cu"
(argp->offset) = off;
# 294 "./rcudatmp/mr3.rcu.cu"
(((argp->val).RCargVal_u).address) = (RCadr)p;
# 295 "./rcudatmp/mr3.rcu.cu"
(argp->size) = (sizeof(p));
# 296 "./rcudatmp/mr3.rcu.cu"
off += sizeof(p);
# 299 "./rcudatmp/mr3.rcu.cu"
argp = args.RCargs_val + argc++;
# 300 "./rcudatmp/mr3.rcu.cu"
p = (void *)((size_t)pd);
# 301 "./rcudatmp/mr3.rcu.cu"
off = ((off + 8UL) - (1)) & (~(8UL - (1)));
# 302 "./rcudatmp/mr3.rcu.cu"
((argp->val).type) = rcudaArgTypeV;
# 303 "./rcudatmp/mr3.rcu.cu"
(argp->offset) = off;
# 304 "./rcudatmp/mr3.rcu.cu"
(((argp->val).RCargVal_u).address) = (RCadr)p;
# 305 "./rcudatmp/mr3.rcu.cu"
(argp->size) = (sizeof(p));
# 306 "./rcudatmp/mr3.rcu.cu"
off += sizeof(p);
# 309 "./rcudatmp/mr3.rcu.cu"
argp = args.RCargs_val + argc++;
# 310 "./rcudatmp/mr3.rcu.cu"
p = (void *)((size_t)zz);
# 311 "./rcudatmp/mr3.rcu.cu"
off = ((off + 8UL) - (1)) & (~(8UL - (1)));
# 312 "./rcudatmp/mr3.rcu.cu"
((argp->val).type) = rcudaArgTypeV;
# 313 "./rcudatmp/mr3.rcu.cu"
(argp->offset) = off;
# 314 "./rcudatmp/mr3.rcu.cu"
(((argp->val).RCargVal_u).address) = (RCadr)p;
# 315 "./rcudatmp/mr3.rcu.cu"
(argp->size) = (sizeof(p));
# 316 "./rcudatmp/mr3.rcu.cu"
off += sizeof(p);
# 319 "./rcudatmp/mr3.rcu.cu"
argp = args.RCargs_val + argc++;
# 320 "./rcudatmp/mr3.rcu.cu"
off = ((off + 4UL) - (1)) & (~(4UL - (1)));
# 321 "./rcudatmp/mr3.rcu.cu"
((argp->val).type) = rcudaArgTypeI;
# 322 "./rcudatmp/mr3.rcu.cu"
(argp->offset) = off;
# 323 "./rcudatmp/mr3.rcu.cu"
(((argp->val).RCargVal_u).valuei) = tblno;
# 324 "./rcudatmp/mr3.rcu.cu"
(argp->size) = (sizeof(int));
# 325 "./rcudatmp/mr3.rcu.cu"
off += sizeof(int);
# 328 "./rcudatmp/mr3.rcu.cu"
argp = args.RCargs_val + argc++;
# 329 "./rcudatmp/mr3.rcu.cu"
off = ((off + 4UL) - (1)) & (~(4UL - (1)));
# 330 "./rcudatmp/mr3.rcu.cu"
((argp->val).type) = rcudaArgTypeF;
# 331 "./rcudatmp/mr3.rcu.cu"
(argp->offset) = off;
# 332 "./rcudatmp/mr3.rcu.cu"
(((argp->val).RCargVal_u).valuef) = xmax;
# 333 "./rcudatmp/mr3.rcu.cu"
(argp->size) = (sizeof(float));
# 334 "./rcudatmp/mr3.rcu.cu"
off += sizeof(float);
# 337 "./rcudatmp/mr3.rcu.cu"
argp = args.RCargs_val + argc++;
# 338 "./rcudatmp/mr3.rcu.cu"
off = ((off + 4UL) - (1)) & (~(4UL - (1)));
# 339 "./rcudatmp/mr3.rcu.cu"
((argp->val).type) = rcudaArgTypeI;
# 340 "./rcudatmp/mr3.rcu.cu"
(argp->offset) = off;
# 341 "./rcudatmp/mr3.rcu.cu"
(((argp->val).RCargVal_u).valuei) = periodicflag;
# 342 "./rcudatmp/mr3.rcu.cu"
(argp->size) = (sizeof(int));
# 343 "./rcudatmp/mr3.rcu.cu"
off += sizeof(int);
# 346 "./rcudatmp/mr3.rcu.cu"
argp = args.RCargs_val + argc++;
# 347 "./rcudatmp/mr3.rcu.cu"
p = (void *)((size_t)force);
# 348 "./rcudatmp/mr3.rcu.cu"
off = ((off + 8UL) - (1)) & (~(8UL - (1)));
# 349 "./rcudatmp/mr3.rcu.cu"
((argp->val).type) = rcudaArgTypeV;
# 350 "./rcudatmp/mr3.rcu.cu"
(argp->offset) = off;
# 351 "./rcudatmp/mr3.rcu.cu"
(((argp->val).RCargVal_u).address) = (RCadr)p;
# 352 "./rcudatmp/mr3.rcu.cu"
(argp->size) = (sizeof(p));
# 353 "./rcudatmp/mr3.rcu.cu"
off += sizeof(p);
# 355 "./rcudatmp/mr3.rcu.cu"
(gdimrc.x) = gdim.x; (gdimrc.y) = gdim.y; (gdimrc.z) = gdim.z;
# 356 "./rcudatmp/mr3.rcu.cu"
(bdimrc.x) = bdim.x; (bdimrc.y) = bdim.y; (bdimrc.z) = bdim.z;
# 357 "./rcudatmp/mr3.rcu.cu"
rcudalaunchkernel_1(0, (char *)("nacl_kernel"), kimage, gdimrc, bdimrc, args, rcudaClient());
# 358 "./rcudatmp/mr3.rcu.cu"
}

# 1 "/tmp/tmpxft_00001cd2_00000000-1_mr3.rcu.cudafe1.stub.c" 1

extern "C" {

# 1 "/tmp/tmpxft_00001cd2_00000000-3_mr3.rcu.fatbin.c" 1
# 1 "/usr/local/cuda/bin/../include/__cudaFatFormat.h" 1
# 83 "/usr/local/cuda/bin/../include/__cudaFatFormat.h"
extern "C" {
# 97 "/usr/local/cuda/bin/../include/__cudaFatFormat.h"
typedef struct {
    char* gpuProfileName;
    char* cubin;
} __cudaFatCubinEntry;
# 113 "/usr/local/cuda/bin/../include/__cudaFatFormat.h"
typedef struct {
    char* gpuProfileName;
    char* ptx;
} __cudaFatPtxEntry;
# 125 "/usr/local/cuda/bin/../include/__cudaFatFormat.h"
typedef struct {
    char* gpuProfileName;
    char* debug;
} __cudaFatDebugEntry;


typedef enum {
      __cudaFatDontSearchFlag = (1 << 0),
      __cudaFatDontCacheFlag = (1 << 1),
      __cudaFatSassDebugFlag = (1 << 2)
} __cudaFatCudaBinaryFlag;
# 144 "/usr/local/cuda/bin/../include/__cudaFatFormat.h"
typedef struct {
    char* name;
} __cudaFatSymbol;
# 157 "/usr/local/cuda/bin/../include/__cudaFatFormat.h"
typedef struct __cudaFatCudaBinaryRec {
    unsigned long magic;
    unsigned long version;
    unsigned long gpuInfoVersion;
    char* key;
    char* ident;
    char* usageMode;
    __cudaFatPtxEntry *ptx;
    __cudaFatCubinEntry *cubin;
    __cudaFatDebugEntry *debug;
    void* debugInfo;
    unsigned int flags;
    __cudaFatSymbol *exported;
    __cudaFatSymbol *imported;
    struct __cudaFatCudaBinaryRec *dependends;
} __cudaFatCudaBinary;
# 191 "/usr/local/cuda/bin/../include/__cudaFatFormat.h"
    typedef enum {
        __cudaFatAvoidPTX,
        __cudaFatPreferBestCode
    } __cudaFatCompilationPolicy;
# 214 "/usr/local/cuda/bin/../include/__cudaFatFormat.h"
void fatGetCubinForGpuWithPolicy( __cudaFatCudaBinary *binary, __cudaFatCompilationPolicy policy, char* gpuName, char* *cubin, char* *dbgInfoFile );
# 225 "/usr/local/cuda/bin/../include/__cudaFatFormat.h"
void fatFreeCubin( char* cubin, char* dbgInfoFile );


}
# 2 "/tmp/tmpxft_00001cd2_00000000-3_mr3.rcu.fatbin.c" 2




extern "C" {


static const unsigned long long __deviceText_$sm_10$[] = {
0x6365746968637261ull,0x6d737b2065727574ull,0x6962610a7d30315full,0x206e6f6973726576ull,
0x6f6d0a7d317b2020ull,0x202020656d616e64ull,0x696275637b202020ull,0x2065646f630a7d6eull,
0x20656d616e090a7bull,0x6b5f6c63616e203dull,0x6c090a6c656e7265ull,0x0a30203d206d656dull,
0x203d206d656d7309ull,0x676572090a303231ull,0x090a3432203d2020ull,0x30203d2020726162ull,
0x2074736e6f63090aull,0x6765730909090a7bull,0x63203d20656d616eull,0x0909090a74736e6full,
0x20206d756e676573ull,0x6f0909090a31203dull,0x3d20207465736666ull,0x79620909090a3020ull,
0x203d202020736574ull,0x206d656d09090a38ull,0x3078300909090a7bull,0x2031303030303030ull,
0x3030303030307830ull,0x0a7d09090a203034ull,0x636e6962090a7d09ull,0x09090a7b2065646full,
0x3566303830647830ull,0x3634307830206466ull,0x7830203863373030ull,0x3530326630303031ull,
0x6333323430783020ull,0x3009090a20303837ull,0x3032663030303178ull,0x3332343078302035ull,
0x6278302030303563ull,0x2035303266303030ull,0x3534303230307830ull,0x783009090a203030ull,
0x3930303030303061ull,0x3030303430783020ull,0x3136783020303837ull,0x3020393063323030ull,
0x3030303030303078ull,0x33783009090a2037ull,0x2064666463323030ull,0x3763303263367830ull,
0x3030337830203863ull,0x7830203330303030ull,0x3038323030303030ull,0x3033783009090a20ull,
0x3020646664636337ull,0x6337633032633678ull,0x3030303178302038ull,0x3078302031323866ull,
0x2030383763333034ull,0x303031783009090aull,0x7830206430386630ull,0x3038376333303430ull,
0x6630303031783020ull,0x3430783020313138ull,0x0a20303837633330ull,0x3630303178300909ull,
0x3078302033303036ull,0x2030383230303030ull,0x3430313030337830ull,0x3134637830203130ull,
0x090a203038373030ull,0x3030303032783009ull,0x3430783020313030ull,0x3020303837383030ull,
0x3230303230303378ull,0x3031346378302039ull,0x09090a2030383730ull,0x3863303030317830ull,
0x3234307830203931ull,0x7830203038376333ull,0x6431386630303031ull,0x6333303430783020ull,
0x3009090a20303837ull,0x3238633030303278ull,0x3232343078302035ull,0x6478302030383738ull,
0x2035333231653030ull,0x3730306330387830ull,0x783009090a203038ull,0x3130636331303033ull,
0x3030333463783020ull,0x3039783020303837ull,0x3020313332303030ull,0x3837303030303078ull,
0x31783009090a2030ull,0x2035313866303030ull,0x3763333034307830ull,0x3030327830203038ull,
0x7830206432323934ull,0x3330303030303030ull,0x3064783009090a20ull,0x3020393336316530ull,
0x3837303063303878ull,0x3030303278302030ull,0x3078302064326363ull,0x2030383730303234ull,
0x303032783009090aull,0x7830203130323938ull,0x3330303030303030ull,0x3065303064783020ull,
0x3038783020643330ull,0x0a20303837303063ull,0x6530306478300909ull,0x3878302031346330ull,
0x2030383730306330ull,0x6338343030327830ull,0x3030307830203130ull,0x090a203330303030ull,
0x3065303064783009ull,0x3038783020313030ull,0x3020303837303063ull,0x3263383830303278ull,
0x3030303078302035ull,0x09090a2033303030ull,0x3231653030647830ull,0x6330387830203532ull,
0x7830203038373030ull,0x3034613130353062ull,0x3130343062783020ull,0x3009090a20633463ull,
0x3438313031306378ull,0x3331306378302034ull,0x6278302038343831ull,0x2031306531303030ull,
0x3734323038307830ull,0x783009090a203038ull,0x3532323230303061ull,0x3430306363783020ull,
0x3061783020303837ull,0x3020313534323030ull,0x3837343030636378ull,0x63783009090a2030ull,
0x2039343831303030ull,0x3730303030307830ull,0x3030657830203038ull,0x7830203534323039ull,
0x3038373034303430ull,0x3065783009090a20ull,0x3020313432303431ull,0x3837633430343078ull,
0x3030306178302030ull,0x6378302035323432ull,0x2030383734303063ull,0x313063783009090aull,
0x7830206434323231ull,0x3038373030303030ull,0x3039303065783020ull,0x3430783020393432ull,
0x0a20303837303030ull,0x3031306578300909ull,0x3078302031303032ull,0x2030383763343030ull,
0x3432323130657830ull,0x3030307830206434ull,0x090a203038373030ull,0x3230303061783009ull,
0x3463783020313036ull,0x3020303837343030ull,0x6631306337306278ull,0x3330303678302064ull,
0x09090a2038633734ull,0x3066353030617830ull,0x3030307830203330ull,0x7830203030303030ull,
0x3330306635303031ull,0x3030303030783020ull,0x3009090a20303031ull,0x3234303230303378ull,
0x3031346378302035ull,0x3378302030383730ull,0x2031306130323030ull,0x3730303134637830ull,
0x783009090a203038ull,0x3532306430303032ull,0x3432323430783020ull,0x3064783020303837ull,
0x3020353232316530ull,0x3837303063303878ull,0x32783009090a2030ull,0x2031303064303030ull,
0x3730303234307830ull,0x3030647830203038ull,0x7830203135303065ull,0x3038373030633038ull,
0x3031783009090a20ull,0x3020313034643030ull,0x3837633332343078ull,0x3130303478302030ull,
0x3078302035353432ull,0x2030383730303030ull,0x303036783009090aull,0x7830203535363230ull,
0x3038373435303030ull,0x3230313033783020ull,0x3463783020353561ull,0x0a20303837303031ull,
0x3030303678300909ull,0x3078302031303432ull,0x2030383734353030ull,0x3030303030327830ull,
0x3034307830203130ull,0x090a203038373035ull,0x3032303033783009ull,0x3463783020313530ull,
0x3020303837303031ull,0x3236323030303978ull,0x3030303478302035ull,0x09090a2030383730ull,
0x6364303030327830ull,0x3234307830203130ull,0x7830203038373035ull,0x3130303065303064ull,
0x3030633038783020ull,0x3009090a20303837ull,0x3532313030303978ull,0x3030303078302039ull,
0x3278302030383730ull,0x2064343065303030ull,0x3730353234307830ull,0x783009090a203038ull,
0x3535363265303064ull,0x3030633038783020ull,0x3062783020303837ull,0x3020303030303635ull,
0x3061323030306378ull,0x63783009090a2030ull,0x2031303030623330ull,0x6138626633307830ull,
0x3030327830203361ull,0x7830206434386430ull,0x3038373035323430ull,0x3064783009090a20ull,
0x3020393536326530ull,0x3837303063303878ull,0x3030306278302030ull,0x6378302064353030ull,
0x2030383734303030ull,0x303032783009090aull,0x7830203130346530ull,0x3038373035323430ull,
0x3065303064783020ull,0x3038783020643430ull,0x0a20303837303063ull,0x3030303278300909ull,
0x3078302031303865ull,0x2030383730353234ull,0x3030653030647830ull,0x6330387830203130ull,
0x090a203038373030ull,0x6530303032783009ull,0x3430783020313563ull,0x3020303837303532ull,
0x3538326530306478ull,0x3063303878302031ull,0x09090a2030383730ull,0x6532303030397830ull,
0x3030637830206435ull,0x7830203038373030ull,0x3935633265313063ull,0x3037633330783020ull,
0x3009090a20623133ull,0x3565323631306378ull,0x3631306378302038ull,0x6378302063356132ull,
0x2034353231393030ull,0x6132353130637830ull,0x783009090a203835ull,0x3935633236313063ull,
0x3030303030783020ull,0x3063783020303837ull,0x3020643436323030ull,0x3030306330343078ull,
0x63783009090a2033ull,0x2064346332333130ull,0x3730303030307830ull,0x3130657830203038ull,
0x7830206434323137ull,0x3038376334303830ull,0x3063783009090a20ull,0x3020313030303030ull,
0x3030303031343078ull,0x3030306378302033ull,0x6378302038356332ull,0x2030303231353130ull,
0x313065783009090aull,0x7830203532613236ull,0x3038376334303430ull,0x3230303065783020ull,
0x3030783020313038ull,0x0a20303837343230ull,0x3030306578300909ull,0x3078302031323232ull,
0x2030383730323030ull,0x3030303130657830ull,0x3030307830206430ull,0x090a203038376330ull,
0x3032313065783009ull,0x3030783020313130ull,0x3020303837303130ull,0x3030303030306678ull,
0x3030306578302031ull,0x09090a2032303030ull,0x6538333030327830ull,0x3030307830206431ull,
0x7830203330303030ull,0x6466663062303033ull,0x3431306336783020ull,0x3009090a20386337ull,
0x3163386330303278ull,0x3030303078302039ull,0x3278302033303030ull,0x2035316138313030ull,
0x3030303030307830ull,0x783009090a203330ull,0x3330306331303031ull,0x3030303030783020ull,
0x3031783020303832ull,0x3020333030393630ull,0x3837303030303078ull,0x33783009090a2030ull,
0x2031303430313030ull,0x3730303134637830ull,0x3030327830203038ull,0x7830203130303030ull,
0x3038373830303430ull,0x3033783009090a20ull,0x3020393230303230ull,0x3837303031346378ull,
0x3030303278302030ull,0x3078302031303866ull,0x2030383738323234ull,0x303064783009090aull,
0x7830203132303065ull,0x3038373030633061ull,0x3834303032783020ull,0x3030783020353030ull,
0x0a20333030303030ull,0x6530306478300909ull,0x6178302064303230ull,0x2030383730306330ull,
0x3038383030327830ull,0x3030307830203130ull,0x090a203330303030ull,0x3065303064783009ull,
0x3061783020313130ull,0x0a20313837303063ull,0x0000000a7d0a7d09ull
};


}



extern "C" {


static const unsigned long long __deviceText_$compute_10$[] = {
0x6f69737265762e09ull,0x2e090a332e31206eull,0x7320746567726174ull,0x616d202c30315f6dull,
0x6f745f3436665f70ull,0x2f2f090a3233665full,0x656c69706d6f6320ull,0x2f20687469772064ull,
0x61636f6c2f727375ull,0x6f2f616475632f6cull,0x696c2f34366e6570ull,0x2f090a65622f2f62ull,
0x6e65706f766e202full,0x746c697562206363ull,0x38303032206e6f20ull,0x0a0a33302d32312dull,
0x752e206765722e09ull,0x313c617225203233ull,0x65722e090a3b3e37ull,0x25203436752e2067ull,
0x3b3e37313c616472ull,0x2e206765722e090aull,0x3c61662520323366ull,0x722e090a3b3e3731ull,
0x203436662e206765ull,0x3e37313c61646625ull,0x206765722e090a3bull,0x767225203233752eull,
0x722e090a3b3e353cull,0x203436752e206765ull,0x3b3e353c76647225ull,0x2e206765722e090aull,
0x3c76662520323366ull,0x65722e090a3b3e35ull,0x25203436662e2067ull,0x0a3b3e353c766466ull,
0x2d2d2d2f2f090a0aull,0x2d2d2d2d2d2d2d2dull,0x2d2d2d2d2d2d2d2dull,0x2d2d2d2d2d2d2d2dull,
0x2d2d2d2d2d2d2d2dull,0x2d2d2d2d2d2d2d2dull,0x2d2d2d2d2d2d2d2dull,0x2d2d2d2d2d2d2d2dull,
0x6d6f43202f2f090aull,0x2f20676e696c6970ull,0x78706d742f706d74ull,0x31303030305f7466ull,
0x303030305f326463ull,0x6d5f372d30303030ull,0x632e7563722e3372ull,0x2f2820692e337070ull,
0x494263632f706d74ull,0x776f6b5353552e23ull,0x2d2d2d2f2f090a29ull,0x2d2d2d2d2d2d2d2dull,
0x2d2d2d2d2d2d2d2dull,0x2d2d2d2d2d2d2d2dull,0x2d2d2d2d2d2d2d2dull,0x2d2d2d2d2d2d2d2dull,
0x2d2d2d2d2d2d2d2dull,0x2d2d2d2d2d2d2d2dull,0x2d2d2d2f2f090a0aull,0x2d2d2d2d2d2d2d2dull,
0x2d2d2d2d2d2d2d2dull,0x2d2d2d2d2d2d2d2dull,0x2d2d2d2d2d2d2d2dull,0x2d2d2d2d2d2d2d2dull,
0x2d2d2d2d2d2d2d2dull,0x2d2d2d2d2d2d2d2dull,0x74704f202f2f090aull,0x2f090a3a736e6f69ull,
0x2d2d2d2d2d2d2d2full,0x2d2d2d2d2d2d2d2dull,0x2d2d2d2d2d2d2d2dull,0x2d2d2d2d2d2d2d2dull,
0x2d2d2d2d2d2d2d2dull,0x2d2d2d2d2d2d2d2dull,0x2d2d2d2d2d2d2d2dull,0x2f2f090a2d2d2d2dull,
0x7465677261542020ull,0x5349202c7874703aull,0x2c30315f6d733a41ull,0x3a6e6169646e4520ull,
0x202c656c7474696cull,0x207265746e696f50ull,0x0a34363a657a6953ull,0x334f2d20202f2f09ull,
0x696d6974704f2809ull,0x6c206e6f6974617aull,0x2f090a296c657665ull,0x280930672d20202full,
0x656c206775626544ull,0x2f2f090a296c6576ull,0x522809326d2d2020ull,0x64612074726f7065ull,
0x736569726f736976ull,0x2d2d2d2f2f090a29ull,0x2d2d2d2d2d2d2d2dull,0x2d2d2d2d2d2d2d2dull,
0x2d2d2d2d2d2d2d2dull,0x2d2d2d2d2d2d2d2dull,0x2d2d2d2d2d2d2d2dull,0x2d2d2d2d2d2d2d2dull,
0x2d2d2d2d2d2d2d2dull,0x656c69662e090a0aull,0x6d6f633c22093109ull,0x6e696c2d646e616dull,
0x69662e090a223e65ull,0x742f22093209656cull,0x6678706d742f706dull,0x6331303030305f74ull,
0x30303030305f3264ull,0x726d5f362d303030ull,0x75632e7563722e33ull,0x70672e3265666164ull,
0x6c69662e090a2275ull,0x73752f2209330965ull,0x63672f62696c2f72ull,0x34365f3638782f63ull,
0x2d7461686465722dull,0x2e342f78756e696cull,0x6c636e692f322e33ull,0x646474732f656475ull,
0x2e090a22682e6665ull,0x22093409656c6966ull,0x636f6c2f7273752full,0x2f616475632f6c61ull,
0x692f2e2e2f6e6962ull,0x632f6564756c636eull,0x63697665642f7472ull,0x6d69746e75725f65ull,
0x662e090a22682e65ull,0x2f22093509656c69ull,0x61636f6c2f727375ull,0x622f616475632f6cull,
0x6e692f2e2e2f6e69ull,0x6f682f6564756c63ull,0x6e696665645f7473ull,0x2e090a22682e7365ull,
0x22093609656c6966ull,0x636f6c2f7273752full,0x2f616475632f6c61ull,0x692f2e2e2f6e6962ull,
0x622f6564756c636eull,0x745f6e69746c6975ull,0x0a22682e73657079ull,0x3709656c69662e09ull,
0x6c2f7273752f2209ull,0x6475632f6c61636full,0x2e2e2f6e69622f61ull,0x6564756c636e692full,
0x5f6563697665642full,0x22682e7365707974ull,0x09656c69662e090aull,0x2f7273752f220938ull,
0x75632f6c61636f6cull,0x2e2f6e69622f6164ull,0x64756c636e692f2eull,0x7265766972642f65ull,
0x682e73657079745full,0x656c69662e090a22ull,0x7273752f22093909ull,0x632f6c61636f6c2full,
0x2f6e69622f616475ull,0x756c636e692f2e2eull,0x75747865742f6564ull,0x73657079745f6572ull,
0x69662e090a22682eull,0x2f2209303109656cull,0x61636f6c2f727375ull,0x622f616475632f6cull,
0x6e692f2e2e2f6e69ull,0x65762f6564756c63ull,0x7079745f726f7463ull,0x2e090a22682e7365ull,
0x09313109656c6966ull,0x6f6c2f7273752f22ull,0x616475632f6c6163ull,0x2f2e2e2f6e69622full,
0x2f6564756c636e69ull,0x6c5f656369766564ull,0x61705f68636e7561ull,0x73726574656d6172ull,
0x69662e090a22682eull,0x2f2209323109656cull,0x61636f6c2f727375ull,0x622f616475632f6cull,
0x6e692f2e2e2f6e69ull,0x72632f6564756c63ull,0x6761726f74732f74ull,0x2e7373616c635f65ull,
0x6c69662e090a2268ull,0x752f220933310965ull,0x756c636e692f7273ull,0x2f737469622f6564ull,
0x22682e7365707974ull,0x09656c69662e090aull,0x7273752f22093431ull,0x6564756c636e692full,
0x22682e656d69742full,0x09656c69662e090aull,0x2e33726d22093531ull,0x0a2275632e756372ull,
0x3109656c69662e09ull,0x2f7273752f220936ull,0x75632f6c61636f6cull,0x2e2f6e69622f6164ull,
0x64756c636e692f2eull,0x6e6f6d6d6f632f65ull,0x6f6974636e75665full,0x2e090a22682e736eull,
0x09373109656c6966ull,0x6f6c2f7273752f22ull,0x616475632f6c6163ull,0x2f2e2e2f6e69622full,
0x2f6564756c636e69ull,0x636e75662f747263ull,0x682e6f7263616d5full,0x656c69662e090a22ull,
0x73752f2209383109ull,0x2f6c61636f6c2f72ull,0x6e69622f61647563ull,0x6c636e692f2e2e2full,
0x6874616d2f656475ull,0x6f6974636e75665full,0x2e090a22682e736eull,0x09393109656c6966ull,
0x6f6c2f7273752f22ull,0x616475632f6c6163ull,0x2f2e2e2f6e69622full,0x2f6564756c636e69ull,
0x665f656369766564ull,0x736e6f6974636e75ull,0x69662e090a22682eull,0x2f2209303209656cull,
0x61636f6c2f727375ull,0x622f616475632f6cull,0x6e692f2e2e2f6e69ull,0x616d2f6564756c63ull,
0x74736e6f635f6874ull,0x0a22682e73746e61ull,0x3209656c69662e09ull,0x2f7273752f220931ull,
0x75632f6c61636f6cull,0x2e2f6e69622f6164ull,0x64756c636e692f2eull,0x5f31315f6d732f65ull,
0x665f63696d6f7461ull,0x736e6f6974636e75ull,0x69662e090a22682eull,0x2f2209323209656cull,
0x61636f6c2f727375ull,0x622f616475632f6cull,0x6e692f2e2e2f6e69ull,0x6d732f6564756c63ull,
0x6d6f74615f32315full,0x74636e75665f6369ull,0x0a22682e736e6f69ull,0x3209656c69662e09ull,
0x2f7273752f220933ull,0x75632f6c61636f6cull,0x2e2f6e69622f6164ull,0x64756c636e692f2eull,
0x5f33315f6d732f65ull,0x665f656c62756f64ull,0x736e6f6974636e75ull,0x69662e090a22682eull,
0x2f2209343209656cull,0x61636f6c2f727375ull,0x622f616475632f6cull,0x6e692f2e2e2f6e69ull,
0x65742f6564756c63ull,0x65665f6572757478ull,0x636e75665f686374ull,0x22682e736e6f6974ull,
0x09656c69662e090aull,0x7273752f22093532ull,0x632f6c61636f6c2full,0x2f6e69622f616475ull,
0x756c636e692f2e2eull,0x5f6874616d2f6564ull,0x6e6f6974636e7566ull,0x74705f6c62645f73ull,
0x0a0a0a22682e3178ull,0x207972746e652e09ull,0x72656b5f6c63616eull,0x090a7b090a6c656eull,
0x31752e206765722eull,0x3e333c6872252036ull,0x206765722e090a3bull,0x3c7225203233752eull,
0x722e090a3b3e3232ull,0x203436752e206765ull,0x3b3e31333c647225ull,0x2e206765722e090aull,
0x373c662520323366ull,0x65722e090a3b3e38ull,0x25203436662e2067ull,0x090a3b3e343c6466ull,
0x72702e206765722eull,0x3e373c7025206465ull,0x617261702e090a3bull,0x5f203436752e206dull,
0x726170616475635full,0x6b5f6c63616e5f6dull,0x3b785f6c656e7265ull,0x6d617261702e090aull,
0x5f5f203233732e20ull,0x6d72617061647563ull,0x656b5f6c63616e5full,0x0a3b6e5f6c656e72ull,
0x206d617261702e09ull,0x635f5f203436752eull,0x5f6d726170616475ull,0x72656b5f6c63616eull,
0x707974615f6c656eull,0x7261702e090a3b65ull,0x203233732e206d61ull,0x6170616475635f5full,
0x5f6c63616e5f6d72ull,0x6e5f6c656e72656bull,0x61702e090a3b7461ull,0x3436752e206d6172ull,
0x70616475635f5f20ull,0x6c63616e5f6d7261ull,0x5f6c656e72656b5full,0x702e090a3b6c6f70ull,
0x36752e206d617261ull,0x616475635f5f2034ull,0x63616e5f6d726170ull,0x6c656e72656b5f6cull,
0x090a3b6d6769735full,0x2e206d617261702eull,0x75635f5f20343675ull,0x6e5f6d7261706164ull,
0x6e72656b5f6c6361ull,0x72746f70695f6c65ull,0x7261702e090a3b6full,0x203436752e206d61ull,
0x6170616475635f5full,0x5f6c63616e5f6d72ull,0x705f6c656e72656bull,0x7261702e090a3b63ull,
0x203436752e206d61ull,0x6170616475635f5full,0x5f6c63616e5f6d72ull,0x705f6c656e72656bull,
0x7261702e090a3b64ull,0x203436752e206d61ull,0x6170616475635f5full,0x5f6c63616e5f6d72ull,
0x7a5f6c656e72656bull,0x7261702e090a3b7aull,0x203233732e206d61ull,0x6170616475635f5full,
0x5f6c63616e5f6d72ull,0x745f6c656e72656bull,0x2e090a3b6f6e6c62ull,0x662e206d61726170ull,
0x6475635f5f203233ull,0x616e5f6d72617061ull,0x656e72656b5f6c63ull,0x5f6c61765f5f5f6cull,
0x616d786d61726170ull,0x7261702e090a3b78ull,0x203233732e206d61ull,0x6170616475635f5full,
0x5f6c63616e5f6d72ull,0x705f6c656e72656bull,0x666369646f697265ull,0x702e090a3b67616cull,
0x36752e206d617261ull,0x616475635f5f2034ull,0x63616e5f6d726170ull,0x6c656e72656b5f6cull,
0x0a3b6563726f665full,0x3d207264202f2f09ull,0x202f2f090a383220ull,0x0a3631203d206966ull,
0x353109636f6c2e09ull,0x4c240a3009393209ull,0x6c63616e5f314242ull,0x3a6c656e72656b5full,
0x7261702e646c090aull,0x09203233662e6d61ull,0x5f5f5b202c316625ull,0x6d72617061647563ull,
0x656b5f6c63616e5full,0x765f5f5f6c656e72ull,0x6d617261705f6c61ull,0x2f093b5d78616d78ull,
0x3635313a6469202full,0x70616475635f5f20ull,0x6c63616e5f6d7261ull,0x5f6c656e72656b5full,
0x61705f6c61765f5full,0x2b78616d786d6172ull,0x2e646c090a307830ull,0x33732e6d61726170ull,
0x202c317225092032ull,0x70616475635f5f5bull,0x6c63616e5f6d7261ull,0x5f6c656e72656b5full,
0x6369646f69726570ull,0x2f093b5d67616c66ull,0x3735313a6469202full,0x70616475635f5f20ull,
0x6c63616e5f6d7261ull,0x5f6c656e72656b5full,0x6369646f69726570ull,0x3078302b67616c66ull,
0x33622e646e61090aull,0x202c327225092032ull,0x203b31202c317225ull,0x2020202020202020ull,
0x6f6d090a202f2f09ull,0x2509203233752e76ull,0x20203b30202c3372ull,0x2020202020202020ull,
0x202f2f0920202020ull,0x6e2e70746573090aull,0x2509203233732e65ull,0x2c327225202c3170ull,
0x2020203b33722520ull,0x2540090a202f2f09ull,0x0920617262203170ull,0x30335f305f744c24ull,
0x202020202020203bull,0x202f2f0920202020ull,0x3109636f6c2e090aull,0x090a300937330935ull,
0x203233662e646461ull,0x6625202c31662509ull,0x203b316625202c31ull,0x2f09202020202020ull,
0x305f744c240a202full,0x6f6d090a3a30335full,0x2509203631752e76ull,0x746325202c316872ull,
0x20203b782e646961ull,0x202f2f0920202020ull,0x69772e6c756d090aull,0x09203631752e6564ull,
0x687225202c347225ull,0x20203b3436202c31ull,0x7663090a202f2f09ull,0x31752e3233752e74ull,
0x202c357225092036ull,0x203b782e64697425ull,0x202f2f0920202020ull,0x33752e646461090aull,
0x202c367225092032ull,0x347225202c357225ull,0x202020202020203bull,0x646c090a202f2f09ull,
0x732e6d617261702eull,0x2c37722509203233ull,0x616475635f5f5b20ull,0x63616e5f6d726170ull,
0x6c656e72656b5f6cull,0x202f2f093b5d6e5full,0x5f203535313a6469ull,0x726170616475635full,
0x6b5f6c63616e5f6dull,0x2b6e5f6c656e7265ull,0x746573090a307830ull,0x3233732e656c2e70ull,
0x25202c3270250920ull,0x3b367225202c3772ull,0x0a202f2f09202020ull,0x7262203270254009ull,
0x305f744c24092061ull,0x202020203b32335full,0x0920202020202020ull,0x6f6c2e090a202f2full,
0x0931340935310963ull,0x662e766f6d090a30ull,0x2c32662509203233ull,0x3030303030663020ull,
0x202020203b303030ull,0x090a30202f2f0920ull,0x203233662e766f6dull,0x6625202c33662509ull,
0x2020202020203b32ull,0x2f09202020202020ull,0x2e766f6d090a202full,0x3466250920323366ull,
0x303030306630202cull,0x2020203b30303030ull,0x0a30202f2f092020ull,0x3233662e766f6d09ull,
0x25202c3566250920ull,0x20202020203b3466ull,0x0920202020202020ull,0x766f6d090a202f2full,
0x662509203233662eull,0x3030306630202c36ull,0x20203b3030303030ull,0x30202f2f09202020ull,
0x33662e766f6d090aull,0x202c376625092032ull,0x202020203b366625ull,0x2020202020202020ull,
0x6f6d090a202f2f09ull,0x2509203233752e76ull,0x20203b30202c3872ull,0x2020202020202020ull,
0x202f2f0920202020ull,0x6c2e70746573090aull,0x2509203233732e65ull,0x2c377225202c3370ull,
0x2020203b38722520ull,0x2540090a202f2f09ull,0x0920617262203370ull,0x30345f305f744c24ull,
0x202020202020203bull,0x202f2f0920202020ull,0x33732e766f6d090aull,0x202c397225092032ull,
0x202020203b377225ull,0x2020202020202020ull,0x6f6d090a202f2f09ull,0x2509203233732e76ull,
0x203b30202c303172ull,0x2020202020202020ull,0x202f2f0920202020ull,0x6f6c2e6c756d090aull,
0x722509203233732eull,0x2c377225202c3131ull,0x20202020203b3320ull,0x6372090a202f2f09ull,
0x2509203233662e70ull,0x3b316625202c3866ull,0x2020202020202020ull,0x202f2f0920202020ull,
0x6f6c2e6c756d090aull,0x722509203233732eull,0x2c367225202c3231ull,0x20202020203b3320ull,
0x7663090a202f2f09ull,0x33732e3436732e74ull,0x2c31647225092032ull,0x20203b3231722520ull,
0x202f2f0920202020ull,0x7261702e646c090aull,0x09203436752e6d61ull,0x5f5b202c32647225ull,
0x726170616475635full,0x6b5f6c63616e5f6dull,0x5d785f6c656e7265ull,0x3a6469202f2f093bull,
0x75635f5f20333531ull,0x6e5f6d7261706164ull,0x6e72656b5f6c6361ull,0x3078302b785f6c65ull,
0x36732e766f6d090aull,0x2c33647225092034ull,0x20203b3264722520ull,0x2020202020202020ull,
0x756d090a202f2f09ull,0x3436752e6f6c2e6cull,0x202c346472250920ull,0x3b34202c31647225ull,
0x202f2f0920202020ull,0x36752e646461090aull,0x2c35647225092034ull,0x25202c3464722520ull,
0x202020203b326472ull,0x646c090a202f2f09ull,0x2e6c61626f6c672eull,0x3966250920323366ull,
0x2b356472255b202cull,0x202f2f09203b5d30ull,0x090a3336313a6469ull,0x61626f6c672e646cull,
0x2509203233662e6cull,0x72255b202c303166ull,0x2f093b5d342b3564ull,0x3436313a6469202full,
0x6f6c672e646c090aull,0x203233662e6c6162ull,0x5b202c3131662509ull,0x3b5d382b35647225ull,
0x313a6469202f2f09ull,0x2e766f6d090a3536ull,0x3172250920323373ull,0x2020203b30202c33ull,
0x2020202020202020ull,0x090a202f2f092020ull,0x203233732e766f6dull,0x25202c3431722509ull,
0x20202020203b3972ull,0x2f09202020202020ull,0x305f744c240a202full,0x2f2f200a3a36335full,
0x4c203e706f6f6c3cull,0x79646f6220706f6full,0x313420656e696c20ull,0x6e697473656e202cull,
0x3a68747065642067ull,0x69747365202c3120ull,0x746920646574616dull,0x736e6f6974617265ull,
0x776f6e6b6e75203aull,0x09636f6c2e090a6eull,0x0a30093634093531ull,0x626f6c672e646c09ull,
0x09203233662e6c61ull,0x255b202c32316625ull,0x093b5d302b336472ull,0x36313a6469202f2full,
0x662e627573090a36ull,0x3331662509203233ull,0x25202c396625202cull,0x202020203b323166ull,
0x6d090a202f2f0920ull,0x09203233662e6c75ull,0x6625202c34316625ull,0x3b33316625202c38ull,
0x2f2f092020202020ull,0x722e747663090a20ull,0x662e3233662e696eull,0x3531662509203233ull,
0x203b34316625202cull,0x6d090a202f2f0920ull,0x09203233662e6c75ull,0x6625202c36316625ull,
0x3b35316625202c31ull,0x2f2f092020202020ull,0x662e627573090a20ull,0x3731662509203233ull,
0x202c33316625202cull,0x2020203b36316625ull,0x6d090a202f2f0920ull,0x09203233662e766full,
0x6625202c38316625ull,0x20202020203b3731ull,0x2f2f092020202020ull,0x09636f6c2e090a20ull,
0x0a30093734093531ull,0x3233662e6c756d09ull,0x202c393166250920ull,0x6625202c37316625ull,
0x09202020203b3731ull,0x6f6c2e090a202f2full,0x0935340935310963ull,0x6c672e646c090a30ull,
0x3233662e6c61626full,0x202c303266250920ull,0x5d342b336472255bull,0x3a6469202f2f093bull,
0x627573090a373631ull,0x662509203233662eull,0x30316625202c3132ull,0x203b30326625202cull,
0x0a202f2f09202020ull,0x3233662e766f6d09ull,0x202c323266250920ull,0x2020203b31326625ull,
0x0920202020202020ull,0x6f6c2e090a202f2full,0x0936340935310963ull,0x662e6c756d090a30ull,
0x3332662509203233ull,0x25202c386625202cull,0x202020203b313266ull,0x63090a202f2f0920ull,
0x662e696e722e7476ull,0x09203233662e3233ull,0x6625202c34326625ull,0x2f2f0920203b3332ull,
0x662e6c756d090a20ull,0x3532662509203233ull,0x25202c316625202cull,0x202020203b343266ull,
0x73090a202f2f0920ull,0x09203233662e6275ull,0x6625202c36326625ull,0x35326625202c3132ull,
0x2f2f09202020203bull,0x662e766f6d090a20ull,0x3232662509203233ull,0x203b36326625202cull,
0x2020202020202020ull,0x2e090a202f2f0920ull,0x3409353109636f6cull,0x64616d090a300937ull,
0x662509203233662eull,0x36326625202c3931ull,0x202c36326625202cull,0x2f2f093b39316625ull,
0x09636f6c2e090a20ull,0x0a30093534093531ull,0x626f6c672e646c09ull,0x09203233662e6c61ull,
0x255b202c37326625ull,0x093b5d382b336472ull,0x36313a6469202f2full,0x662e627573090a38ull,
0x3832662509203233ull,0x202c31316625202cull,0x2020203b37326625ull,0x6d090a202f2f0920ull,
0x09203233662e766full,0x6625202c39326625ull,0x20202020203b3832ull,0x2f2f092020202020ull,
0x09636f6c2e090a20ull,0x0a30093634093531ull,0x3233662e6c756d09ull,0x202c303366250920ull,
0x326625202c386625ull,0x0920202020203b38ull,0x747663090a202f2full,0x3233662e696e722eull,
0x662509203233662eull,0x30336625202c3133ull,0x0a202f2f0920203bull,0x3233662e6c756d09ull,
0x202c323366250920ull,0x336625202c316625ull,0x0920202020203b31ull,0x627573090a202f2full,
0x662509203233662eull,0x38326625202c3333ull,0x203b32336625202cull,0x0a202f2f09202020ull,
0x3233662e766f6d09ull,0x202c393266250920ull,0x2020203b33336625ull,0x0920202020202020ull,
0x6f6c2e090a202f2full,0x0937340935310963ull,0x662e64616d090a30ull,0x3931662509203233ull,
0x202c33336625202cull,0x6625202c33336625ull,0x0a202f2f093b3931ull,0x3436662e74766309ull,
0x662509203233662eull,0x39316625202c3164ull,0x092020202020203bull,0x766f6d090a202f2full,
0x662509203436662eull,0x30306430202c3264ull,0x3030303030303030ull,0x093b303030303030ull,
0x6573090a30202f2full,0x662e75656e2e7074ull,0x2c34702509203436ull,0x25202c3164662520ull,
0x202f2f093b326466ull,0x203470252140090aull,0x744c240920617262ull,0x20203b37335f305full,
0x2020202020202020ull,0x2f2f200a202f2f09ull,0x50203e706f6f6c3cull,0x6c20666f20747261ull,
0x79646f6220706f6full,0x313420656e696c20ull,0x6c2064616568202cull,0x242064656c656261ull,
0x0a36335f305f744cull,0x353109636f6c2e09ull,0x6c090a3009353509ull,0x2e6d617261702e64ull,
0x6472250920343675ull,0x75635f5f5b202c36ull,0x6e5f6d7261706164ull,0x6e72656b5f6c6361ull,
0x65707974615f6c65ull,0x6469202f2f093b5dull,0x635f5f203834313aull,0x5f6d726170616475ull,
0x72656b5f6c63616eull,0x707974615f6c656eull,0x63090a3078302b65ull,0x732e3436752e7476ull,
0x3764722509203233ull,0x203b33317225202cull,0x2f2f092020202020ull,0x6c2e6c756d090a20ull,
0x2509203436752e6full,0x647225202c386472ull,0x2020203b34202c37ull,0x61090a202f2f0920ull,
0x09203436752e6464ull,0x7225202c39647225ull,0x38647225202c3664ull,0x2f2f09202020203bull,
0x6c672e646c090a20ull,0x3233732e6c61626full,0x202c353172250920ull,0x5d302b396472255bull,
0x3a6469202f2f093bull,0x747663090a393631ull,0x3233732e3436752eull,0x2c30316472250920ull,
0x2020203b36722520ull,0x0a202f2f09202020ull,0x2e6f6c2e6c756d09ull,0x6472250920343675ull,
0x31647225202c3131ull,0x0920203b34202c30ull,0x646461090a202f2full,0x722509203436752eull,
0x647225202c323164ull,0x3131647225202c36ull,0x0a202f2f0920203bull,0x626f6c672e646c09ull,
0x09203233732e6c61ull,0x255b202c36317225ull,0x3b5d302b32316472ull,0x313a6469202f2f09ull,
0x702e646c090a3037ull,0x3233732e6d617261ull,0x202c373172250920ull,0x70616475635f5f5bull,
0x6c63616e5f6d7261ull,0x5f6c656e72656b5full,0x2f2f093b5d74616eull,0x203137313a646920ull,
0x6170616475635f5full,0x5f6c63616e5f6d72ull,0x6e5f6c656e72656bull,0x090a3078302b7461ull,
0x732e6f6c2e6c756dull,0x3831722509203233ull,0x202c36317225202cull,0x2f09203b37317225ull,
0x2e646461090a202full,0x3172250920323373ull,0x2c35317225202c39ull,0x20203b3831722520ull,
0x090a202f2f092020ull,0x09353109636f6c2eull,0x7663090a30093635ull,0x33732e3436752e74ull,
0x3331647225092032ull,0x203b39317225202cull,0x202f2f0920202020ull,0x6f6c2e6c756d090aull,
0x722509203436752eull,0x647225202c343164ull,0x20203b34202c3331ull,0x646c090a202f2f09ull,
0x752e6d617261702eull,0x3164722509203436ull,0x75635f5f5b202c35ull,0x6e5f6d7261706164ull,
0x6e72656b5f6c6361ull,0x72746f70695f6c65ull,0x69202f2f093b5d6full,0x5f5f203237313a64ull,
0x6d72617061647563ull,0x656b5f6c63616e5full,0x6f70695f6c656e72ull,0x0a3078302b6f7274ull,
0x3436752e64646109ull,0x2c36316472250920ull,0x202c353164722520ull,0x09203b3431647225ull,
0x2e646c090a202f2full,0x662e6c61626f6c67ull,0x3433662509203233ull,0x36316472255b202cull,
0x202f2f093b5d302bull,0x090a3337313a6469ull,0x6d617261702e646cull,0x722509203436752eull,
0x5f5f5b202c373164ull,0x6d72617061647563ull,0x656b5f6c63616e5full,0x6769735f6c656e72ull,
0x69202f2f093b5d6dull,0x5f5f203437313a64ull,0x6d72617061647563ull,0x656b5f6c63616e5full,
0x6769735f6c656e72ull,0x61090a3078302b6dull,0x09203436752e6464ull,0x25202c3831647225ull,
0x7225202c37316472ull,0x2f2f09203b343164ull,0x6c672e646c090a20ull,0x3233662e6c61626full,
0x202c353366250920ull,0x302b38316472255bull,0x6469202f2f093b5dull,0x7173090a3537313aull,
0x09203233662e7472ull,0x6625202c36336625ull,0x20202020203b3931ull,0x202f2f0920202020ull,
0x33662e627573090aull,0x2c37336625092032ull,0x25202c3533662520ull,0x202020203b363366ull,
0x756d090a202f2f09ull,0x2509203233662e6cull,0x336625202c383366ull,0x3b37336625202c34ull,
0x202f2f0920202020ull,0x3109636f6c2e090aull,0x090a300937350935ull,0x33662e7472717372ull,
0x2c39336625092032ull,0x20203b3931662520ull,0x2f09202020202020ull,0x2e6c756d090a202full,
0x3466250920323366ull,0x2c39336625202c30ull,0x20203b3933662520ull,0x090a202f2f092020ull,
0x203233662e6c756dull,0x25202c3134662509ull,0x346625202c303466ull,0x2f09202020203b30ull,
0x2e6c756d090a202full,0x3466250920323366ull,0x2c31346625202c32ull,0x20203b3134662520ull,
0x090a202f2f092020ull,0x6d617261702e646cull,0x722509203436752eull,0x5f5f5b202c393164ull,
0x6d72617061647563ull,0x656b5f6c63616e5full,0x5d63705f6c656e72ull,0x3a6469202f2f093bull,
0x75635f5f20303831ull,0x6e5f6d7261706164ull,0x6e72656b5f6c6361ull,0x78302b63705f6c65ull,
0x752e646461090a30ull,0x3264722509203436ull,0x3931647225202c30ull,0x3b3431647225202cull,
0x6c090a202f2f0920ull,0x6c61626f6c672e64ull,0x662509203233662eull,0x6472255b202c3334ull,
0x2f093b5d302b3032ull,0x3138313a6469202full,0x33662e766f6d090aull,0x2c34346625092032ull,
0x3030633034663020ull,0x202020203b303030ull,0x6d090a36202f2f09ull,0x09203233662e6c75ull,
0x6625202c35346625ull,0x34346625202c3334ull,0x2f2f09202020203bull,0x662e6c756d090a20ull,
0x3634662509203233ull,0x202c32346625202cull,0x2020203b35346625ull,0x6d090a202f2f0920ull,
0x09203233662e766full,0x6630202c37346625ull,0x6233616138626633ull,0x2f2f09202020203bull,
0x0a373234342e3120ull,0x3233662e6c756d09ull,0x202c383466250920ull,0x6625202c38336625ull,
0x09202020203b3734ull,0x327865090a202f2full,0x662509203233662eull,0x38346625202c3934ull,
0x202020202020203bull,0x0a202f2f09202020ull,0x617261702e646c09ull,0x2509203436752e6dull,
0x5f5b202c31326472ull,0x726170616475635full,0x6b5f6c63616e5f6dull,0x6f705f6c656e7265ull,
0x69202f2f093b5d6cull,0x5f5f203837313a64ull,0x6d72617061647563ull,0x656b5f6c63616e5full,
0x6c6f705f6c656e72ull,0x6461090a3078302bull,0x2509203436752e64ull,0x7225202c32326472ull,
0x647225202c313264ull,0x202f2f09203b3431ull,0x6f6c672e646c090aull,0x203233662e6c6162ull,
0x5b202c3035662509ull,0x5d302b3232647225ull,0x3a6469202f2f093bull,0x766f6d090a393731ull,
0x662509203233662eull,0x63336630202c3135ull,0x203b653931333037ull,0x30202f2f09202020ull,
0x333036363431302eull,0x33662e6c756d090aull,0x2c32356625092032ull,0x25202c3035662520ull,
0x202020203b313566ull,0x756d090a202f2f09ull,0x2509203233662e6cull,0x346625202c333566ull,
0x3b32356625202c39ull,0x202f2f0920202020ull,0x33662e6c756d090aull,0x2c34356625092032ull,
0x25202c3433662520ull,0x202020203b333566ull,0x756d090a202f2f09ull,0x2509203233662e6cull,
0x356625202c353566ull,0x3b39336625202c34ull,0x202f2f0920202020ull,0x33662e627573090aull,
0x2c36356625092032ull,0x25202c3535662520ull,0x202020203b363466ull,0x646c090a202f2f09ull,
0x752e6d617261702eull,0x3264722509203436ull,0x75635f5f5b202c33ull,0x6e5f6d7261706164ull,
0x6e72656b5f6c6361ull,0x093b5d64705f6c65ull,0x38313a6469202f2full,0x616475635f5f2032ull,
0x63616e5f6d726170ull,0x6c656e72656b5f6cull,0x0a3078302b64705full,0x3436752e64646109ull,
0x2c34326472250920ull,0x202c333264722520ull,0x09203b3431647225ull,0x2e646c090a202f2full,
0x662e6c61626f6c67ull,0x3735662509203233ull,0x34326472255b202cull,0x202f2f093b5d302bull,
0x090a3338313a6469ull,0x203233662e766f6dull,0x30202c3835662509ull,0x3030303030313466ull,
0x2f09202020203b30ull,0x6c756d090a38202full,0x662509203233662eull,0x37356625202c3935ull,
0x203b38356625202cull,0x0a202f2f09202020ull,0x3233662e6c756d09ull,0x202c303666250920ull,
0x6625202c32346625ull,0x09202020203b3935ull,0x6c756d090a202f2full,0x662509203233662eull,
0x30366625202c3136ull,0x203b30346625202cull,0x0a202f2f09202020ull,0x3233662e62757309ull,
0x202c323666250920ull,0x6625202c36356625ull,0x09202020203b3136ull,0x2e646c090a202f2full,
0x36752e6d61726170ull,0x3532647225092034ull,0x6475635f5f5b202cull,0x616e5f6d72617061ull,
0x656e72656b5f6c63ull,0x2f093b5d7a7a5f6cull,0x3637313a6469202full,0x70616475635f5f20ull,
0x6c63616e5f6d7261ull,0x5f6c656e72656b5full,0x090a3078302b7a7aull,0x203436752e646461ull,
0x202c363264722509ull,0x25202c3532647225ull,0x2f09203b34316472ull,0x672e646c090a202full,
0x33662e6c61626f6cull,0x2c33366625092032ull,0x2b36326472255b20ull,0x69202f2f093b5d30ull,
0x6d090a3737313a64ull,0x09203233662e6c75ull,0x6625202c34366625ull,0x30346625202c3933ull,
0x2f2f09202020203bull,0x662e64616d090a20ull,0x3536662509203233ull,0x202c33366625202cull,
0x6625202c34366625ull,0x0a202f2f093b3236ull,0x353109636f6c2e09ull,0x6d090a3009313609ull,
0x09203233662e766full,0x6625202c36366625ull,0x2020202020203b33ull,0x2f2f092020202020ull,
0x662e766f6d090a20ull,0x3736662509203233ull,0x203b38316625202cull,0x2020202020202020ull,
0x6d090a202f2f0920ull,0x09203233662e6461ull,0x6625202c38366625ull,0x35366625202c3736ull,
0x093b36366625202cull,0x766f6d090a202f2full,0x662509203233662eull,0x3b38366625202c33ull,
0x2020202020202020ull,0x0a202f2f09202020ull,0x3233662e766f6d09ull,0x202c393666250920ull,
0x202020203b356625ull,0x0920202020202020ull,0x766f6d090a202f2full,0x662509203233662eull,
0x32326625202c3037ull,0x202020202020203bull,0x0a202f2f09202020ull,0x3233662e64616d09ull,
0x202c313766250920ull,0x6625202c35366625ull,0x39366625202c3037ull,0x6d090a202f2f093bull,
0x09203233662e766full,0x376625202c356625ull,0x2020202020203b31ull,0x2f2f092020202020ull,
0x662e766f6d090a20ull,0x3237662509203233ull,0x20203b376625202cull,0x2020202020202020ull,
0x6d090a202f2f0920ull,0x09203233662e6461ull,0x6625202c33376625ull,0x33336625202c3536ull,
0x093b32376625202cull,0x766f6d090a202f2full,0x662509203233662eull,0x3b33376625202c37ull,
0x2020202020202020ull,0x0a202f2f09202020ull,0x37335f305f744c24ull,0x6f6c3c2f2f200a3aull,
0x74726150203e706full,0x706f6f6c20666f20ull,0x696c2079646f6220ull,0x68202c313420656eull,
0x6562616c20646165ull,0x5f744c242064656cull,0x6461090a36335f30ull,0x2509203233732e64ull,
0x317225202c333172ull,0x2020203b31202c33ull,0x202f2f0920202020ull,0x33732e646461090aull,
0x2c30317225092032ull,0x33202c3031722520ull,0x202020202020203bull,0x6461090a202f2f09ull,
0x2509203436752e64ull,0x647225202c336472ull,0x20203b3231202c33ull,0x202f2f0920202020ull,
0x6e2e70746573090aull,0x2509203233732e65ull,0x30317225202c3570ull,0x203b31317225202cull,
0x2540090a202f2f09ull,0x0920617262203570ull,0x36335f305f744c24ull,0x202020202020203bull,
0x202f2f0920202020ull,0x6e752e617262090aull,0x305f744c24092069ull,0x202020203b34335full,
0x2020202020202020ull,0x744c240a202f2f09ull,0x090a3a30345f305full,0x732e6f6c2e6c756dull,
0x3032722509203233ull,0x33202c367225202cull,0x2f0920202020203bull,0x2e747663090a202full,
0x203233732e343673ull,0x202c373264722509ull,0x2020203b30327225ull,0x090a202f2f092020ull,
0x752e6f6c2e6c756dull,0x3464722509203436ull,0x2c3732647225202cull,0x2f092020203b3420ull,
0x305f744c240a202full,0x6c2e090a3a34335full,0x343609353109636full,0x702e646c090a3009ull,
0x3436752e6d617261ull,0x2c38326472250920ull,0x616475635f5f5b20ull,0x63616e5f6d726170ull,
0x6c656e72656b5f6cull,0x3b5d6563726f665full,0x313a6469202f2f09ull,0x6475635f5f203438ull,
0x616e5f6d72617061ull,0x656e72656b5f6c63ull,0x2b6563726f665f6cull,0x646461090a307830ull,
0x722509203436752eull,0x647225202c393264ull,0x34647225202c3832ull,0x0a202f2f0920203bull,
0x3233662e766f6d09ull,0x202c343766250920ull,0x202020203b336625ull,0x0920202020202020ull,
0x2e7473090a202f2full,0x662e6c61626f6c67ull,0x6472255b09203233ull,0x25202c5d302b3932ull,
0x202f2f093b343766ull,0x090a3538313a6469ull,0x203233662e766f6dull,0x25202c3537662509ull,
0x20202020203b3566ull,0x2f09202020202020ull,0x672e7473090a202full,0x33662e6c61626f6cull,
0x326472255b092032ull,0x6625202c5d342b39ull,0x69202f2f093b3537ull,0x6d090a3638313a64ull,
0x09203233662e766full,0x6625202c36376625ull,0x2020202020203b37ull,0x2f2f092020202020ull,
0x6c672e7473090a20ull,0x3233662e6c61626full,0x39326472255b0920ull,0x376625202c5d382bull,
0x6469202f2f093b36ull,0x744c240a3738313aull,0x090a3a32335f305full,0x09353109636f6c2eull,
0x7865090a30093838ull,0x20202020203b7469ull,0x2020202020202020ull,0x2020202020202020ull,
0x202f2f0920202020ull,0x646e6557444c240aull,0x656b5f6c63616e5full,0x7d090a3a6c656e72ull,
0x6c63616e202f2f20ull,0x0a6c656e72656b5full,0x000000000000000aull
};


}


static __cudaFatPtxEntry __ptxEntries [] = {{(char*)"compute_10",(char*)__deviceText_$compute_10$},{0,0}};
static __cudaFatCubinEntry __cubinEntries[] = {{(char*)"sm_10",(char*)__deviceText_$sm_10$},{0,0}};
static __cudaFatDebugEntry __debugEntries[] = {{0,0}};



static __cudaFatCudaBinary __fatDeviceText __attribute__ ((section (".nvFatBinSegment")))= {0x1ee55a01,0x00000003,0x8ecc680c,(char*)"15232d739ea4fcc8",(char*)"./rcudatmp/mr3.rcu.cu",(char*)" ",__ptxEntries,__cubinEntries,__debugEntries,0,0,0,0,0};
# 5 "/tmp/tmpxft_00001cd2_00000000-1_mr3.rcu.cudafe1.stub.c" 2
# 1 "/usr/local/cuda/bin/../include/crt/host_runtime.h" 1
# 74 "/usr/local/cuda/bin/../include/crt/host_runtime.h"
# 1 "/usr/local/cuda/bin/../include/host_defines.h" 1
# 75 "/usr/local/cuda/bin/../include/crt/host_runtime.h" 2
# 100 "/usr/local/cuda/bin/../include/crt/host_runtime.h"
extern "C" {


extern void** __cudaRegisterFatBinary(
  void *fatCubin
);

extern void __cudaUnregisterFatBinary(
  void **fatCubinHandle
);

extern void __cudaRegisterVar(
        void **fatCubinHandle,
        char *hostVar,
        char *deviceAddress,
  const char *deviceName,
        int ext,
        int size,
        int constant,
        int global
);

extern void __cudaRegisterTexture(
        void **fatCubinHandle,
  const struct textureReference *hostVar,
  const void **deviceAddress,
  const char *deviceName,
        int dim,
        int norm,
        int ext
);

extern void __cudaRegisterShared(
  void **fatCubinHandle,
  void **devicePtr
);

extern void __cudaRegisterSharedVar(
  void **fatCubinHandle,
  void **devicePtr,
  size_t size,
  size_t alignment,
  int storage
);

extern void __cudaRegisterFunction(
        void **fatCubinHandle,
  const char *hostFun,
        char *deviceFun,
  const char *deviceName,
        int thread_limit,
        uint3 *tid,
        uint3 *bid,
        dim3 *bDim,
        dim3 *gDim,
        int *wSize
);


}




extern int atexit(void(*)(void)) throw();







static void **__cudaFatCubinHandle;

static void __cudaUnregisterBinaryUtil(void)
{
  __cudaUnregisterFatBinary(__cudaFatCubinHandle);
}
# 215 "/usr/local/cuda/bin/../include/crt/host_runtime.h"
# 1 "/usr/local/cuda/bin/../include/common_functions.h" 1
# 68 "/usr/local/cuda/bin/../include/common_functions.h"
# 1 "/usr/local/cuda/bin/../include/crt/func_macro.h" 1 3
# 69 "/usr/local/cuda/bin/../include/common_functions.h" 2

static clock_t __cuda_clock(void)
{
  return clock();
}

static void *__cuda_memset(void *s, int c, size_t n)
{
  return memset(s, c, n);
}

static void *__cuda_memcpy(void *d, const void *s, size_t n)
{
  return memcpy(d, s, n);
}
# 93 "/usr/local/cuda/bin/../include/common_functions.h"
# 1 "/usr/local/cuda/bin/../include/math_functions.h" 1 3
# 844 "/usr/local/cuda/bin/../include/math_functions.h" 3
extern __attribute__((__weak__)) int __cuda_error_not_implememted(void); int __cuda_error_not_implememted(void);
# 900 "/usr/local/cuda/bin/../include/math_functions.h" 3
static int __cuda_abs(int a)
{
  return abs(a);
}

static float __cuda_fabsf(float a)
{
  return fabsf(a);
}

static long long int __cuda_llabs(long long int a)
{

  return ::llabs(a);



}

static float __cuda_exp2f(float a)
{
  return exp2f(a);
}

# 1 "/usr/local/cuda/bin/../include/device_functions.h" 1 3
# 347 "/usr/local/cuda/bin/../include/device_functions.h" 3
# 1 "/usr/local/cuda/bin/../include/math_constants.h" 1 3
# 348 "/usr/local/cuda/bin/../include/device_functions.h" 2 3



static int __cuda___isnan(double a);
static int __cuda___isnanf(float a);
static int __double2int_rz(double);
static unsigned int __double2uint_rz(double);
static long long int __double2ll_rz(double);
static unsigned long long int __double2ull_rz(double);
# 370 "/usr/local/cuda/bin/../include/device_functions.h" 3
static int __mulhi(int a, int b)
{
  long long int c = (long long int)a * (long long int)b;

  return (int)(c >> 32);
}

static unsigned int __umulhi(unsigned int a, unsigned int b)
{
  unsigned long long int c = (unsigned long long int)a * (unsigned long long int)b;

  return (unsigned int)(c >> 32);
}

static unsigned long long int __umul64hi(unsigned long long int a, unsigned long long int b)
{
  unsigned int a_lo = (unsigned int)a;
  unsigned long long int a_hi = a >> 32;
  unsigned int b_lo = (unsigned int)b;
  unsigned long long int b_hi = b >> 32;
  unsigned long long int m1 = a_lo * b_hi;
  unsigned long long int m2 = a_hi * b_lo;
  unsigned int carry;

  carry = (0ULL + __umulhi(a_lo, b_lo) + (unsigned int)m1 + (unsigned int)m2) >> 32;

  return a_hi * b_hi + (m1 >> 32) + (m2 >> 32) + carry;
}

static long long int __mul64hi(long long int a, long long int b)
{
  long long int res;
  res = __umul64hi(a, b);
  if (a < 0LL) res = res - b;
  if (b < 0LL) res = res - a;
  return res;
}

static float __saturatef(float a)
{
  return a >= 1.0f ? 1.0f : a <= 0.0f ? 0.0f : a;
}

static unsigned int __sad(int a, int b, unsigned int c)
{
  long long int diff = (long long int)a - (long long int)b;

  return (unsigned int)(__cuda_llabs(diff) + (long long int)c);
}

static unsigned int __usad(unsigned int a, unsigned int b, unsigned int c)
{
  long long int diff = (long long int)a - (long long int)b;

  return (unsigned int)(__cuda_llabs(diff) + (long long int)c);
}

static int __mul24(int a, int b)
{

  a &= 0xffffff;
  a = (a & 0x800000) != 0 ? a | ~0xffffff : a;
  b &= 0xffffff;
  b = (b & 0x800000) != 0 ? b | ~0xffffff : b;


  return a * b;
}

static unsigned int __umul24(unsigned int a, unsigned int b)
{

  a &= 0xffffff;
  b &= 0xffffff;


  return a * b;
}

static float __int_as_float(int a)
{
  volatile union {int a; float b;} u;

  u.a = a;

  return u.b;
}

static int __float_as_int(float a)
{
  volatile union {float a; int b;} u;

  u.a = a;

  return u.b;
}

static long long int __internal_float2ll_kernel(float a, long long int max, long long int min, long long int nan, enum cudaRoundMode rndMode)
{
  unsigned long long int res, t = 0ULL;
  int shift;
  unsigned int ia;

  if (sizeof(a) == sizeof(double) && __cuda___isnan((double)a)) return nan; if (sizeof(a) == sizeof(float) && __cuda___isnanf((float)a)) return nan; if (a >= max) return max; if (a <= min) return min;
  ia = __float_as_int(a);
  shift = 189 - ((ia >> 23) & 0xff);
  res = (unsigned long long int)(((ia << 8) | 0x80000000) >> 1) << 32;
  if (shift >= 64) {
    t = res;
    res = 0;
  } else if (shift) {
    t = res << (64 - shift);
    res = res >> shift;
  }
  if (rndMode == cudaRoundNearest && (long long int)t < 0LL) {
    res += t == 0x8000000000000000ULL ? res & 1ULL : 1ULL;
  }
  else if (rndMode == cudaRoundMinInf && t != 0ULL && ia > 0x80000000) {
    res++;
  }
  else if (rndMode == cudaRoundPosInf && t != 0ULL && (int)ia > 0) {
    res++;
  }
  if ((int)ia < 0) res = (unsigned long long int)-(long long int)res;
  return (long long int)res;
}

static int __internal_float2int(float a, enum cudaRoundMode rndMode)
{
  return (int)__internal_float2ll_kernel(a, 2147483647LL, -2147483648LL, 0LL, rndMode);
}

static int __float2int_rz(float a)
{



  return __internal_float2int(a, cudaRoundZero);

}

static int __float2int_ru(float a)
{
  return __internal_float2int(a, cudaRoundPosInf);
}

static int __float2int_rd(float a)
{
  return __internal_float2int(a, cudaRoundMinInf);
}

static int __float2int_rn(float a)
{
  return __internal_float2int(a, cudaRoundNearest);
}

static long long int __internal_float2ll(float a, enum cudaRoundMode rndMode)
{
  return __internal_float2ll_kernel(a, 9223372036854775807LL, -9223372036854775807LL -1LL, -9223372036854775807LL -1LL, rndMode);
}

static long long int __float2ll_rz(float a)
{



  return __internal_float2ll(a, cudaRoundZero);

}

static long long int __float2ll_ru(float a)
{
  return __internal_float2ll(a, cudaRoundPosInf);
}

static long long int __float2ll_rd(float a)
{
  return __internal_float2ll(a, cudaRoundMinInf);
}

static long long int __float2ll_rn(float a)
{
  return __internal_float2ll(a, cudaRoundNearest);
}

static unsigned long long int __internal_float2ull_kernel(float a, unsigned long long int max, unsigned long long int nan, enum cudaRoundMode rndMode)
{
  unsigned long long int res, t = 0ULL;
  int shift;
  unsigned int ia;

  if (sizeof(a) == sizeof(double) && __cuda___isnan((double)a)) return nan; if (sizeof(a) == sizeof(float) && __cuda___isnanf((float)a)) return nan; if (a >= max) return max; if (a <= 0LL) return 0LL;
  ia = __float_as_int(a);
  shift = 190 - ((ia >> 23) & 0xff);
  res = (unsigned long long int)((ia << 8) | 0x80000000) << 32;
  if (shift >= 64) {
    t = res >> (int)(shift > 64);
    res = 0;
  } else if (shift) {
    t = res << (64 - shift);
    res = res >> shift;
  }
  if (rndMode == cudaRoundNearest && (long long int)t < 0LL) {
    res += t == 0x8000000000000000ULL ? res & 1ULL : 1ULL;
  }
  else if (rndMode == cudaRoundPosInf && t != 0ULL) {
    res++;
  }
  return res;
}

static unsigned int __internal_float2uint(float a, enum cudaRoundMode rndMode)
{
  return (unsigned int)__internal_float2ull_kernel(a, 4294967295U, 0U, rndMode);
}

static unsigned int __float2uint_rz(float a)
{



  return __internal_float2uint(a, cudaRoundZero);

}

static unsigned int __float2uint_ru(float a)
{
  return __internal_float2uint(a, cudaRoundPosInf);
}

static unsigned int __float2uint_rd(float a)
{
  return __internal_float2uint(a, cudaRoundMinInf);
}

static unsigned int __float2uint_rn(float a)
{
  return __internal_float2uint(a, cudaRoundNearest);
}

static unsigned long long int __internal_float2ull(float a, enum cudaRoundMode rndMode)
{
  return __internal_float2ull_kernel(a, 18446744073709551615ULL, 9223372036854775808ULL, rndMode);
}

static unsigned long long int __float2ull_rz(float a)
{



  return __internal_float2ull(a, cudaRoundZero);

}

static unsigned long long int __float2ull_ru(float a)
{
  return __internal_float2ull(a, cudaRoundPosInf);
}

static unsigned long long int __float2ull_rd(float a)
{
  return __internal_float2ull(a, cudaRoundMinInf);
}

static unsigned long long int __float2ull_rn(float a)
{
  return __internal_float2ull(a, cudaRoundNearest);
}

static int __internal_normalize64(unsigned long long int *a)
{
  int lz = 0;

  if ((*a & 0xffffffff00000000ULL) == 0ULL) {
    *a <<= 32;
    lz += 32;
  }
  if ((*a & 0xffff000000000000ULL) == 0ULL) {
    *a <<= 16;
    lz += 16;
  }
  if ((*a & 0xff00000000000000ULL) == 0ULL) {
    *a <<= 8;
    lz += 8;
  }
  if ((*a & 0xf000000000000000ULL) == 0ULL) {
    *a <<= 4;
    lz += 4;
  }
  if ((*a & 0xC000000000000000ULL) == 0ULL) {
    *a <<= 2;
    lz += 2;
  }
  if ((*a & 0x8000000000000000ULL) == 0ULL) {
    *a <<= 1;
    lz += 1;
  }
  return lz;
}

static int __internal_normalize(unsigned int *a)
{
  unsigned long long int t = (unsigned long long int)*a;
  int lz = __internal_normalize64(&t);

  *a = (unsigned int)(t >> 32);

  return lz - 32;
}

static float __internal_int2float_kernel(int a, enum cudaRoundMode rndMode)
{
  volatile union {
    float f;
    unsigned int i;
  } res;
  int shift;
  unsigned int t;
  res.i = a;
  if (a == 0) return res.f;
  if (a < 0) res.i = (unsigned int)-a;
  shift = __internal_normalize((unsigned int*)&res.i);
  t = res.i << 24;
  res.i = (res.i >> 8);
  res.i += (127 + 30 - shift) << 23;
  if (a < 0) res.i |= 0x80000000;
  if ((rndMode == cudaRoundNearest) && (t >= 0x80000000)) {
    res.i += (t == 0x80000000) ? (res.i & 1) : (t >> 31);
  }
  else if ((rndMode == cudaRoundMinInf) && t && (a < 0)) {
    res.i++;
  }
  else if ((rndMode == cudaRoundPosInf) && t && (a > 0)) {
    res.i++;
  }
  return res.f;
}

static float __int2float_rz(int a)
{
  return __internal_int2float_kernel(a, cudaRoundZero);
}

static float __int2float_ru(int a)
{
  return __internal_int2float_kernel(a, cudaRoundPosInf);
}

static float __int2float_rd(int a)
{
  return __internal_int2float_kernel(a, cudaRoundMinInf);
}

static float __int2float_rn(int a)
{



  return __internal_int2float_kernel(a, cudaRoundNearest);

}

static float __internal_uint2float_kernel(unsigned int a, enum cudaRoundMode rndMode)
{
  volatile union {
    float f;
    unsigned int i;
  } res;
  int shift;
  unsigned int t;
  res.i = a;
  if (a == 0) return res.f;
  shift = __internal_normalize((unsigned int*)&res.i);
  t = res.i << 24;
  res.i = (res.i >> 8);
  res.i += (127 + 30 - shift) << 23;
  if ((rndMode == cudaRoundNearest) && (t >= 0x80000000)) {
    res.i += (t == 0x80000000) ? (res.i & 1) : (t >> 31);
  }
  else if ((rndMode == cudaRoundPosInf) && t) {
    res.i++;
  }
  return res.f;
}

static float __uint2float_rz(unsigned int a)
{
  return __internal_uint2float_kernel(a, cudaRoundZero);
}

static float __uint2float_ru(unsigned int a)
{
  return __internal_uint2float_kernel(a, cudaRoundPosInf);
}

static float __uint2float_rd(unsigned int a)
{
  return __internal_uint2float_kernel(a, cudaRoundMinInf);
}

static float __uint2float_rn(unsigned int a)
{



  return __internal_uint2float_kernel(a, cudaRoundNearest);

}

static float __ll2float_rn(long long int a)
{
  return (float)a;
}

static float __ull2float_rn(unsigned long long int a)
{



  unsigned long long int temp;
  unsigned int res, t;
  int shift;
  if (a == 0ULL) return 0.0f;
  temp = a;
  shift = __internal_normalize64(&temp);
  temp = (temp >> 8) | ((temp & 0xffULL) ? 1ULL : 0ULL);
  res = (unsigned int)(temp >> 32);
  t = (unsigned int)temp;
  res += (127 + 62 - shift) << 23;
  res += t == 0x80000000 ? res & 1 : t >> 31;
  return __int_as_float(res);

}

static float __internal_fmul_kernel(float a, float b, int rndNearest)
{
  unsigned long long product;
  volatile union {
    float f;
    unsigned int i;
  } xx, yy;
  unsigned expo_x, expo_y;

  xx.f = a;
  yy.f = b;

  expo_y = 0xFF;
  expo_x = expo_y & (xx.i >> 23);
  expo_x = expo_x - 1;
  expo_y = expo_y & (yy.i >> 23);
  expo_y = expo_y - 1;

  if ((expo_x <= 0xFD) &&
      (expo_y <= 0xFD)) {
multiply:
    expo_x = expo_x + expo_y;
    expo_y = xx.i ^ yy.i;
    xx.i = xx.i & 0x00ffffff;
    yy.i = yy.i << 8;
    xx.i = xx.i | 0x00800000;
    yy.i = yy.i | 0x80000000;

    product = ((unsigned long long)xx.i) * yy.i;
    expo_x = expo_x - 127 + 2;
    expo_y = expo_y & 0x80000000;
    xx.i = (unsigned int)(product >> 32);
    yy.i = (unsigned int)(product & 0xffffffff);

    if (xx.i < 0x00800000) {
      xx.i = (xx.i << 1) | (yy.i >> 31);
      yy.i = (yy.i << 1);
      expo_x--;
    }
    if (expo_x <= 0xFD) {
      xx.i = xx.i | expo_y;
      xx.i = xx.i + (expo_x << 23);

      if (yy.i < 0x80000000) return xx.f;
      xx.i += (((yy.i == 0x80000000) ? (xx.i & 1) : (yy.i >> 31))
               && rndNearest);
      return xx.f;
    } else if ((int)expo_x >= 254) {

      xx.i = (expo_y | 0x7F800000) - (!rndNearest);
      return xx.f;
    } else {

      expo_x = ((unsigned int)-((int)expo_x));
      if (expo_x > 25) {

        xx.i = expo_y;
        return xx.f;
      } else {
        yy.i = (xx.i << (32 - expo_x)) | ((yy.i) ? 1 : 0);
        xx.i = expo_y + (xx.i >> expo_x);
        xx.i += (((yy.i == 0x80000000) ? (xx.i & 1) : (yy.i >> 31))
                 && rndNearest);
        return xx.f;
      }
    }
  } else {
    product = xx.i ^ yy.i;
    product = product & 0x80000000;
    if (!(xx.i & 0x7fffffff)) {
      if (expo_y != 254) {
        xx.i = (unsigned int)product;
        return xx.f;
      }
      expo_y = yy.i << 1;
      if (expo_y == 0xFF000000) {
        xx.i = expo_y | 0x00C00000;
      } else {
        xx.i = yy.i | 0x00400000;
      }
      return xx.f;
    }
    if (!(yy.i & 0x7fffffff)) {
      if (expo_x != 254) {
        xx.i = (unsigned int)product;
        return xx.f;
      }
      expo_x = xx.i << 1;
      if (expo_x == 0xFF000000) {
        xx.i = expo_x | 0x00C00000;
      } else {
        xx.i = xx.i | 0x00400000;
      }
      return xx.f;
    }
    if ((expo_y != 254) && (expo_x != 254)) {
      expo_y++;
      expo_x++;
      if (expo_x == 0) {
        expo_y |= xx.i & 0x80000000;




        xx.i = xx.i << 8;
        while (!(xx.i & 0x80000000)) {
          xx.i <<= 1;
          expo_x--;
        }
        xx.i = (xx.i >> 8) | (expo_y & 0x80000000);
        expo_y &= ~0x80000000;
        expo_y--;
        goto multiply;
      }
      if (expo_y == 0) {
        expo_x |= yy.i & 0x80000000;
        yy.i = yy.i << 8;
        while (!(yy.i & 0x80000000)) {
          yy.i <<= 1;
          expo_y--;
        }
        yy.i = (yy.i >> 8) | (expo_x & 0x80000000);
        expo_x &= ~0x80000000;
        expo_x--;
        goto multiply;
      }
    }
    expo_x = xx.i << 1;
    expo_y = yy.i << 1;

    if (expo_x > 0xFF000000) {

      xx.i = xx.i | 0x00400000;
      return xx.f;
    }

    if (expo_y > 0xFF000000) {

      xx.i = yy.i | 0x00400000;
      return xx.f;
    }
    xx.i = (unsigned int)product | 0x7f800000;
    return xx.f;
  }
}

static float __internal_fadd_kernel(float a, float b, int rndNearest)
{
  volatile union {
    float f;
    unsigned int i;
  } xx, yy;
  unsigned int expo_x;
  unsigned int expo_y;
  unsigned int temp;

  xx.f = a;
  yy.f = b;


  expo_y = yy.i << 1;
  if (expo_y > (xx.i << 1)) {
    expo_y = xx.i;
    xx.i = yy.i;
    yy.i = expo_y;
  }

  temp = 0xff;
  expo_x = temp & (xx.i >> 23);
  expo_x = expo_x - 1;
  expo_y = temp & (yy.i >> 23);
  expo_y = expo_y - 1;

  if ((expo_x <= 0xFD) &&
      (expo_y <= 0xFD)) {

add:
    expo_y = expo_x - expo_y;
    if (expo_y > 25) {
      expo_y = 31;
    }
    temp = xx.i ^ yy.i;
    xx.i = xx.i & ~0x7f000000;
    xx.i = xx.i | 0x00800000;
    yy.i = yy.i & ~0xff000000;
    yy.i = yy.i | 0x00800000;

    if ((int)temp < 0) {

      temp = 32 - expo_y;
      temp = (expo_y) ? (yy.i << temp) : 0;
      temp = (unsigned int)(-((int)temp));
      xx.i = xx.i - (yy.i >> expo_y) - (temp ? 1 : 0);
      if (xx.i & 0x00800000) {
        if (expo_x <= 0xFD) {
          xx.i = xx.i & ~0x00800000;
          xx.i = (xx.i + (expo_x << 23)) + 0x00800000;
          if (temp < 0x80000000) return xx.f;
          xx.i += (((temp == 0x80000000) ? (xx.i & 1) : (temp >> 31))
                   && rndNearest);
          return xx.f;
        }
      } else {
        if ((temp | (xx.i << 1)) == 0) {

          xx.i = 0;
          return xx.f;
        }

        yy.i = xx.i & 0x80000000;
        do {
          xx.i = (xx.i << 1) | (temp >> 31);
          temp <<= 1;
          expo_x--;
        } while (!(xx.i & 0x00800000));
        xx.i = xx.i | yy.i;
      }
    } else {

      temp = 32 - expo_y;
      temp = (expo_y) ? (yy.i << temp) : 0;
      xx.i = xx.i + (yy.i >> expo_y);
      if (!(xx.i & 0x01000000)) {
        if (expo_x <= 0xFD) {
          expo_y = xx.i & 1;
          xx.i = xx.i + (expo_x << 23);
          if (temp < 0x80000000) return xx.f;
          xx.i += (((temp == 0x80000000) ? expo_y : (temp >> 31))
                   && rndNearest);
          return xx.f;
        }
      } else {

        temp = (xx.i << 31) | (temp >> 1);

        xx.i = ((xx.i & 0x80000000) | (xx.i >> 1)) & ~0x40000000;
        expo_x++;
      }
    }
    if (expo_x <= 0xFD) {
      expo_y = xx.i & 1;
      xx.i += (((temp == 0x80000000) ? expo_y : (temp >> 31))
               && rndNearest);
      xx.i = xx.i + (expo_x << 23);
      return xx.f;
    }
    if ((int)expo_x >= 254) {

        xx.i = ((xx.i & 0x80000000) | 0x7f800000) - (!rndNearest);
        return xx.f;
    }

    expo_y = expo_x + 32;
    yy.i = xx.i & 0x80000000;
    xx.i = xx.i & ~0xff000000;

    expo_x = (unsigned int)(-((int)expo_x));
    temp = xx.i << expo_y | ((temp) ? 1 : 0);
    xx.i = yy.i | (xx.i >> expo_x);
    xx.i += (((temp == 0x80000000) ? (xx.i & 1) : (temp >> 31))
             && rndNearest);
    return xx.f;
  } else {

    if (!(yy.i << 1)) {
      if (xx.i == 0x80000000) {
          xx.i = yy.i;
      }
      if ((xx.i << 1) > 0xff000000) {
          xx.i |= 0x00400000;
      }
      return xx.f;
    }
    if ((expo_y != 254) && (expo_x != 254)) {

      if (expo_x == (unsigned int) -1) {
        temp = xx.i & 0x80000000;
        xx.i = xx.i << 8;
        while (!(xx.i & 0x80000000)) {
          xx.i <<= 1;
          expo_x--;
        }
        expo_x++;
        xx.i = (xx.i >> 8) | temp;
      }
      if (expo_y == (unsigned int) -1) {
        temp = yy.i & 0x80000000;
        yy.i = yy.i << 8;
        while (!(yy.i & 0x80000000)) {
          yy.i <<= 1;
          expo_y--;
        }
        expo_y++;
        yy.i = (yy.i >> 8) | temp;
      }
      goto add;
    }
    expo_x = xx.i << 1;
    expo_y = yy.i << 1;

    if (expo_x > 0xff000000) {

      xx.i = xx.i | 0x00400000;
      return xx.f;
    }

    if (expo_y > 0xff000000) {

      xx.i = yy.i | 0x00400000;
      return xx.f;
    }
    if ((expo_x == 0xff000000) && (expo_y == 0xff000000)) {




      expo_x = xx.i ^ yy.i;
      xx.i = xx.i | ((expo_x) ? 0xffc00000 : 0);
      return xx.f;
    }

    if (expo_y == 0xff000000) {
      xx.i = yy.i;
    }
    return xx.f;
  }
}

static float __fadd_rz(float a, float b)
{
  return __internal_fadd_kernel(a, b, 0);
}

static float __fmul_rz(float a, float b)
{
  return __internal_fmul_kernel(a, b, 0);
}

static float __fadd_rn(float a, float b)
{
  return __internal_fadd_kernel(a, b, 1);
}

static float __fmul_rn(float a, float b)
{
  return __internal_fmul_kernel(a, b, 1);
}

static void __brkpt(int c)
{

}
# 1164 "/usr/local/cuda/bin/../include/device_functions.h" 3
extern int __cudaSynchronizeThreads(void**, void*);



static inline __attribute__((always_inline)) void __syncthreads(void)
{
  volatile int _ = 0;
  L: if (__cudaSynchronizeThreads((void**)&&L, (void*)&_)) goto L;
}
# 1185 "/usr/local/cuda/bin/../include/device_functions.h" 3
static void __trap(void)
{
  __builtin_trap();
}
# 1207 "/usr/local/cuda/bin/../include/device_functions.h" 3
static float __fdividef(float a, float b)
{






  if (__cuda_fabsf(b) > 8.507059173e37f) {
    if (__cuda_fabsf(a) <= 3.402823466e38f) {
      return ((a / b) / 3.402823466e38f) / 3.402823466e38f;
    } else {
      return __int_as_float(0x7fffffff);
    }
  } else {
    return a / b;
  }

}

static float __sinf(float a)
{
  return sinf(a);
}

static float __cosf(float a)
{
  return cosf(a);
}

static float __log2f(float a)
{
  return log2f(a);
}







static float __internal_accurate_fdividef(float a, float b)
{
  if (__cuda_fabsf(b) > 8.507059173e37f) {
    a *= .25f;
    b *= .25f;
  }
  return __fdividef(a, b);
}

static float __tanf(float a)
{



  return __sinf(a) / __cosf(a);

}

static void __sincosf(float a, float *sptr, float *cptr)
{



  *sptr = __sinf(a);
  *cptr = __cosf(a);

}

static float __expf(float a)
{



  return __cuda_exp2f(a * 1.442695041f);

}

static float __exp10f(float a)
{



  return __cuda_exp2f(a * 3.321928094f);

}

static float __log10f(float a)
{



  return 0.301029996f * __log2f(a);

}

static float __logf(float a)
{



  return 0.693147181f * __log2f(a);

}

static float __powf(float a, float b)
{



  return __cuda_exp2f(b * __log2f(a));

}

static float fdividef(float a, float b)
{





  return __internal_accurate_fdividef(a, b);

}

static int __clz(int a)
{
  return (a)?(158-(__float_as_int(__uint2float_rz((unsigned int)a))>>23)):32;
}

static int __ffs(int a)
{
  return 32 - __clz (a & -a);
}

static int __popc(unsigned int a)
{
  a = a - ((a >> 1) & 0x55555555);
  a = (a & 0x33333333) + ((a >> 2) & 0x33333333);
  a = (a + (a >> 4)) & 0x0f0f0f0f;
  a = ((__umul24(a, 0x808080) << 1) + a) >> 24;
  return a;
}

static int __clzll(long long int a)
{
  int ahi = ((int)((unsigned long long)a >> 32));
  int alo = ((int)((unsigned long long)a & 0xffffffffULL));
  int res;
  if (ahi) {
      res = 0;
  } else {
      res = 32;
      ahi = alo;
  }
  res = res + __clz(ahi);
  return res;
}

static int __ffsll(long long int a)
{
  return 64 - __clzll (a & -a);
}

static int __popcll(unsigned long long int a)
{
  unsigned int ahi = ((unsigned int)(a >> 32));
  unsigned int alo = ((unsigned int)(a & 0xffffffffULL));
  alo = alo - ((alo >> 1) & 0x55555555);
  alo = (alo & 0x33333333) + ((alo >> 2) & 0x33333333);
  ahi = ahi - ((ahi >> 1) & 0x55555555);
  ahi = (ahi & 0x33333333) + ((ahi >> 2) & 0x33333333);
  alo = alo + ahi;
  alo = (alo & 0x0f0f0f0f) + ((alo >> 4) & 0x0f0f0f0f);
  alo = ((__umul24(alo, 0x808080) << 1) + alo) >> 24;
  return alo;
}
# 1393 "/usr/local/cuda/bin/../include/device_functions.h" 3
static double fdivide(double a, double b)
{
  return (double)fdividef((float)a, (float)b);
}



static int __double2int_rz(double a)
{
  return __float2int_rz((float)a);
}

static unsigned int __double2uint_rz(double a)
{
  return __float2uint_rz((float)a);
}

static long long int __double2ll_rz(double a)
{
  return __float2ll_rz((float)a);
}

static unsigned long long int __double2ull_rz(double a)
{
  return __float2ull_rz((float)a);
}
# 1470 "/usr/local/cuda/bin/../include/device_functions.h" 3
# 1 "/usr/local/cuda/bin/../include/sm_11_atomic_functions.h" 1 3
# 257 "/usr/local/cuda/bin/../include/sm_11_atomic_functions.h" 3
static int __iAtomicAdd(int *address, int val)
{
  int old = *address;

  *address = old + val;

  return old;
}

static unsigned int __uAtomicAdd(unsigned int *address, unsigned int val)
{
  unsigned int old = *address;

  *address = old + val;

  return old;
}

static int __iAtomicExch(int *address, int val)
{
  int old = *address;

  *address = val;

  return old;
}

static unsigned int __uAtomicExch(unsigned int *address, unsigned int val)
{
  unsigned int old = *address;

  *address = val;

  return old;
}

static float __fAtomicExch(float *address, float val)
{
  float old = *address;

  *address = val;

  return old;
}

static int __iAtomicMin(int *address, int val)
{
  int old = *address;

  *address = old < val ? old : val;

  return old;
}

static unsigned int __uAtomicMin(unsigned int *address, unsigned int val)
{
  unsigned int old = *address;

  *address = old < val ? old : val;

  return old;
}

static int __iAtomicMax(int *address, int val)
{
  int old = *address;

  *address = old > val ? old : val;

  return old;
}

static unsigned int __uAtomicMax(unsigned int *address, unsigned int val)
{
  unsigned int old = *address;

  *address = old > val ? old : val;

  return old;
}

static unsigned int __uAtomicInc(unsigned int *address, unsigned int val)
{
  unsigned int old = *address;

  *address = (old >= val) ? 0 : old + 1;

  return old;
}

static unsigned int __uAtomicDec(unsigned int *address, unsigned int val)
{
  unsigned int old = *address;

  *address = ((old == 0) | (old > val)) ? val : (old - 1);

  return old;
}

static int __iAtomicAnd(int *address, int val)
{
  int old = *address;

  *address = old & val;

  return old;
}

static unsigned int __uAtomicAnd(unsigned int *address, unsigned int val)
{
  unsigned int old = *address;

  *address = old & val;

  return old;
}

static int __iAtomicOr(int *address, int val)
{
  int old = *address;

  *address = old | val;

  return old;
}

static unsigned int __uAtomicOr(unsigned int *address, unsigned int val)
{
  unsigned int old = *address;

  *address = old | val;

  return old;
}

static int __iAtomicXor(int *address, int val)
{
  int old = *address;

  *address = old ^ val;

  return old;
}

static unsigned int __uAtomicXor(unsigned int *address, unsigned int val)
{
  unsigned int old = *address;

  *address = old ^ val;

  return old;
}

static int __iAtomicCAS(int *address, int compare, int val)
{
  int old = *address;

  *address = old == compare ? val : old;

  return old;
}

static unsigned int __uAtomicCAS(unsigned int *address, unsigned int compare, unsigned int val)
{
  unsigned int old = *address;

  *address = old == compare ? val : old;

  return old;
}
# 1471 "/usr/local/cuda/bin/../include/device_functions.h" 2 3
# 1 "/usr/local/cuda/bin/../include/sm_12_atomic_functions.h" 1 3
# 118 "/usr/local/cuda/bin/../include/sm_12_atomic_functions.h" 3
static unsigned long long int __ullAtomicAdd(unsigned long long int *address, unsigned long long int val)
{
  unsigned long long int old = *address;

  *address = old + val;

  return old;
}

static unsigned long long int __ullAtomicExch(unsigned long long int *address, unsigned long long int val)
{
  unsigned long long int old = *address;

  *address = val;

  return old;
}

static unsigned long long int __ullAtomicCAS(unsigned long long int *address, unsigned long long int compare, unsigned long long int val)
{
  unsigned long long int old = *address;

  *address = old == compare ? val : old;

  return old;
}



static int __any(int cond)
{
  return cond;
}

static int __all(int cond)
{
  return cond;
}
# 1472 "/usr/local/cuda/bin/../include/device_functions.h" 2 3
# 1 "/usr/local/cuda/bin/../include/sm_13_double_functions.h" 1 3
# 266 "/usr/local/cuda/bin/../include/sm_13_double_functions.h" 3
static double __longlong_as_double(long long int a)
{
  volatile union {long long int a; double b;} u;

  u.a = a;

  return u.b;
}

static long long int __double_as_longlong(double a)
{
  volatile union {double a; long long int b;} u;

  u.a = a;

  return u.b;
}

static float __internal_double2float_kernel(double a)
{
  volatile union {
    double d;
    unsigned long long int i;
  } xx;
  volatile union {
    float f;
    unsigned int i;
  } res;
  int shift;
  xx.d = a;
  if (xx.i == 0) return 0.0f;
  res.i = (((unsigned int) (xx.i >> 32)) & 0x80000000);
  if ((xx.i & 0x7ff0000000000000ULL) == 0x7ff0000000000000ULL) {
    if ((xx.i & 0x7fffffffffffffffULL) > 0x7ff0000000000000ULL) {

      res.i = 0x7f8fffff;
    } else {

      res.i |= 0x7f800000;
    }
    return res.f;
  }
  shift = ((int) ((xx.i >> 52) & 0x7ff)) - 1023;

  xx.i = (xx.i & 0x000fffffffffffffULL);
  if (shift >= 128) {
    res.i |= 0x7f7fffff;
    return res.f;
  }
  if (shift <= -127) {
    if (shift < -180) {

      xx.i = 0;
    } else {
      xx.i |= 0x0010000000000000ULL;
      xx.i >>= 127 + shift;
    }
  } else {
    res.i |= (unsigned int) (127 + shift) << 23;
  }
  res.i |= ((unsigned int) (xx.i >> 29)) & 0x007fffff;
  xx.i &= 0x1fffffff;
  return res.f;
}

static double __internal_ll2double_kernel(long long int a, enum cudaRoundMode rndMode)
{
  volatile union {
    double d;
    unsigned long long int i;
  } res;
  int shift;
  unsigned int t;
  res.i = a;
  if (a == 0) return res.d;
  if (a < 0) res.i = (unsigned long long int)-a;
  shift = __internal_normalize64((unsigned long long int*)&res.i);
  t = ((unsigned int) res.i) << 21;
  res.i >>= 11;
  res.i += ((unsigned long long int)(1023 + 62 - shift)) << 52;
  if (a < 0) res.i |= 0x8000000000000000ULL;
  if ((rndMode == cudaRoundNearest) && (t >= 0x80000000)) {
    res.i += (t == 0x80000000) ? (res.i & 1) : 1;
  }
  else if ((rndMode == cudaRoundMinInf) && t && (a < 0)) {
    res.i++;
  }
  else if ((rndMode == cudaRoundPosInf) && t && (a > 0)) {
    res.i++;
  }
  return res.d;
}

static double __internal_ull2double_kernel(unsigned long long int a, enum cudaRoundMode rndMode)
{
  volatile union {
    double d;
    unsigned long long int i;
  } res;
  int shift;
  unsigned int t;
  res.i = a;
  if (a == 0) return res.d;
  shift = __internal_normalize64((unsigned long long int *)&res.i);
  t = ((unsigned int) res.i) << 21;
  res.i >>= 11;
  res.i += ((unsigned long long int)(1023 + 62 - shift)) << 52;
  if ((rndMode == cudaRoundNearest) && (t >= 0x80000000)) {
    res.i += (t == 0x80000000) ? (res.i & 1) : 1;
  }
  else if ((rndMode == cudaRoundPosInf) && t) {
    res.i++;
  }
  return res.d;
}

static long long int __internal_double2ll_kernel(double a, long long int max, long long int min, long long int nan, enum cudaRoundMode rndMode)
{
  volatile union {
    double d;
    unsigned long long int i;
  } xx, res;
  unsigned long long int t = 0;
  int shift;

  xx.d = a;
  if (sizeof(a) == sizeof(double) && __cuda___isnan((double)a)) return nan; if (sizeof(a) == sizeof(float) && __cuda___isnanf((float)a)) return nan; if (a >= max) return max; if (a <= min) return min;
  shift = (int) (1023 + 62 - ((xx.i >> 52) & 0x7ff));
  res.i = ((xx.i << 11) | 0x8000000000000000ULL) >> 1;
  if (shift >= 64) {
    t = res.i;
    res.i = 0;
  } else if (shift) {
    t = res.i << (64 - shift);
    res.i = res.i >> shift;
  }
  if ((rndMode == cudaRoundNearest) && (t >= 0x8000000000000000ULL)) {
    res.i += (t == 0x8000000000000000ULL) ? (res.i & 1) : 1;
  }
  else if ((rndMode == cudaRoundMinInf) && t &&
          (xx.i > 0x8000000000000000ULL)) {
    res.i++;
  }
  else if ((rndMode == cudaRoundPosInf) && t && ((long long int)xx.i > 0)) {
    res.i++;
  }
  if ((long long int)xx.i < 0) {
    res.i = (unsigned long long int)(-(long long int)res.i);
  }
  return res.i;
}

static unsigned long long int __internal_double2ull_kernel(double a, unsigned long long int max, unsigned long long int nan, enum cudaRoundMode rndMode)
{
  volatile union {
    double d;
    unsigned long long int i;
  } xx, res;
  unsigned long long int t = 0;
  int shift;
  xx.d = a;
  if (sizeof(a) == sizeof(double) && __cuda___isnan((double)a)) return nan; if (sizeof(a) == sizeof(float) && __cuda___isnanf((float)a)) return nan; if (a >= max) return max; if (a <= 0LL) return 0LL;

  if (a == 0.0) return 0LL;
  shift = (int) (1023 + 63 - ((xx.i >> 52) & 0x7ff));
  res.i = ((xx.i << 11) | 0x8000000000000000ULL);
  if (shift >= 64) {
    t = res.i >> (int)(shift > 64);
    res.i = 0;
  } else if (shift) {
    t = res.i << (64 - shift);
    res.i = res.i >> shift;
  }
  if ((rndMode == cudaRoundNearest) && (t >= 0x8000000000000000ULL)) {
    res.i += (t == 0x8000000000000000ULL) ? (res.i & 1) : 1;
  }
  else if ((rndMode == cudaRoundPosInf) && t) {
    res.i++;
  }
  return res.i;
}

static int __double2hiint(double a)
{
  volatile union {
    double d;
    signed int i[2];
  } cvt;

  cvt.d = a;

  return cvt.i[1];
}

static int __double2loint(double a)
{
  volatile union {
    double d;
    signed int i[2];
  } cvt;

  cvt.d = a;

  return cvt.i[0];
}

static double __hiloint2double(int a, int b)
{
  volatile union {
    double d;
    signed int i[2];
  } cvt;

  cvt.i[0] = b;
  cvt.i[1] = a;

  return cvt.d;
}

static float __double2float_rn(double a)
{
  return (float)a;
}

static float __double2float_rz(double a)
{
  return __internal_double2float_kernel(a);
}

static int __internal_double2int(double a, enum cudaRoundMode rndMode)
{
  return (int)__internal_double2ll_kernel(a, 2147483647LL, -2147483648LL, -2147483648LL, rndMode);
}

static int __double2int_rn(double a)
{
  return __internal_double2int(a, cudaRoundNearest);
}

static int __double2int_ru(double a)
{
  return __internal_double2int(a, cudaRoundPosInf);
}

static int __double2int_rd(double a)
{
  return __internal_double2int(a, cudaRoundMinInf);
}

static unsigned int __internal_double2uint(double a, enum cudaRoundMode rndMode)
{
  return (unsigned int)__internal_double2ull_kernel(a, 4294967295ULL, 2147483648ULL, rndMode);
}

static unsigned int __double2uint_rn(double a)
{
  return __internal_double2uint(a, cudaRoundNearest);
}

static unsigned int __double2uint_ru(double a)
{
  return __internal_double2uint(a, cudaRoundPosInf);
}

static unsigned int __double2uint_rd(double a)
{
  return __internal_double2uint(a, cudaRoundMinInf);
}

static long long int __internal_double2ll(double a, enum cudaRoundMode rndMode)
{
  return __internal_double2ll_kernel(a, 9223372036854775807LL, -9223372036854775807LL -1LL, -9223372036854775807LL -1LL, rndMode);
}

static long long int __double2ll_rn(double a)
{
  return __internal_double2ll(a, cudaRoundNearest);
}

static long long int __double2ll_ru(double a)
{
  return __internal_double2ll(a, cudaRoundPosInf);
}

static long long int __double2ll_rd(double a)
{
  return __internal_double2ll(a, cudaRoundMinInf);
}

static unsigned long long int __internal_double2ull(double a, enum cudaRoundMode rndMode)
{
  return __internal_double2ull_kernel(a, 18446744073709551615ULL, 9223372036854775808ULL, rndMode);
}

static unsigned long long int __double2ull_rn(double a)
{
  return __internal_double2ull(a, cudaRoundNearest);
}

static unsigned long long int __double2ull_ru(double a)
{
  return __internal_double2ull(a, cudaRoundPosInf);
}

static unsigned long long int __double2ull_rd(double a)
{
  return __internal_double2ull(a, cudaRoundMinInf);
}

static double __int2double_rn(int a)
{
  return (double)a;
}

static double __uint2double_rn(unsigned int a)
{
  return (double)a;
}

static double __ll2double_rn(long long int a)
{
  return (double)a;
}

static double __ll2double_rz(long long int a)
{
  return __internal_ll2double_kernel(a, cudaRoundZero);
}

static double __ll2double_rd(long long int a)
{
  return __internal_ll2double_kernel(a, cudaRoundMinInf);
}

static double __ll2double_ru(long long int a)
{
  return __internal_ll2double_kernel(a, cudaRoundPosInf);
}

static double __ull2double_rn(unsigned long long int a)
{
  return __internal_ull2double_kernel(a, cudaRoundNearest);
}

static double __ull2double_rz(unsigned long long int a)
{
  return __internal_ull2double_kernel(a, cudaRoundZero);
}

static double __ull2double_rd(unsigned long long int a)
{
  return __internal_ull2double_kernel(a, cudaRoundMinInf);
}

static double __ull2double_ru(unsigned long long int a)
{
  return __internal_ull2double_kernel(a, cudaRoundPosInf);
}





static double __internal_fma_kernel(double x, double y, double z, enum cudaRoundMode rndMode)
{



  struct {
    unsigned int lo;
    unsigned int hi;
  } xx, yy, zz, ww;
  unsigned int s, t, u, prod0, prod1, prod2, prod3, expo_x, expo_y, expo_z;

  xx.hi = __double2hiint(x);
  xx.lo = __double2loint(x);
  yy.hi = __double2hiint(y);
  yy.lo = __double2loint(y);
  zz.hi = __double2hiint(z);
  zz.lo = __double2loint(z);

  expo_z = 0x7FF;
  t = xx.hi >> 20;
  expo_x = expo_z & t;
  expo_x = expo_x - 1;
  t = yy.hi >> 20;
  expo_y = expo_z & t;
  expo_y = expo_y - 1;
  t = zz.hi >> 20;
  expo_z = expo_z & t;
  expo_z = expo_z - 1;

  if (!((expo_x <= 0x7FD) &&
        (expo_y <= 0x7FD) &&
        (expo_z <= 0x7FD))) {





    if (((yy.hi << 1) | (yy.lo != 0)) > 0xffe00000) {
      yy.hi |= 0x00080000;
      return __hiloint2double(yy.hi, yy.lo);
    }
    if (((zz.hi << 1) | (zz.lo != 0)) > 0xffe00000) {
      zz.hi |= 0x00080000;
      return __hiloint2double(zz.hi, zz.lo);
    }
    if (((xx.hi << 1) | (xx.lo != 0)) > 0xffe00000) {
      xx.hi |= 0x00080000;
      return __hiloint2double(xx.hi, xx.lo);
    }
# 690 "/usr/local/cuda/bin/../include/sm_13_double_functions.h" 3
    if (((((xx.hi << 1) | xx.lo) == 0) &&
         (((yy.hi << 1) | (yy.lo != 0)) == 0xffe00000)) ||
        ((((yy.hi << 1) | yy.lo) == 0) &&
         (((xx.hi << 1) | (xx.lo != 0)) == 0xffe00000))) {
      xx.hi = 0xfff80000;
      xx.lo = 0x00000000;
      return __hiloint2double(xx.hi, xx.lo);
    }
    if (((zz.hi << 1) | (zz.lo != 0)) == 0xffe00000) {
      if ((((yy.hi << 1) | (yy.lo != 0)) == 0xffe00000) ||
          (((xx.hi << 1) | (xx.lo != 0)) == 0xffe00000)) {
        if ((int)(xx.hi ^ yy.hi ^ zz.hi) < 0) {
          xx.hi = 0xfff80000;
          xx.lo = 0x00000000;
          return __hiloint2double(xx.hi, xx.lo);
        }
      }
    }




    if (((xx.hi << 1) | (xx.lo != 0)) == 0xffe00000) {
      xx.hi = xx.hi ^ (yy.hi & 0x80000000);
      return __hiloint2double(xx.hi, xx.lo);
    }
    if (((yy.hi << 1) | (yy.lo != 0)) == 0xffe00000) {
      yy.hi = yy.hi ^ (xx.hi & 0x80000000);
      return __hiloint2double(yy.hi, yy.lo);
    }
    if (((zz.hi << 1) | (zz.lo != 0)) == 0xffe00000) {
      return __hiloint2double(zz.hi, zz.lo);
    }





    if ((zz.hi == 0x80000000) && (zz.lo == 0)) {
      if ((((xx.hi << 1) | xx.lo) == 0) ||
          (((yy.hi << 1) | yy.lo) == 0)) {
        if ((int)(xx.hi ^ yy.hi) < 0) {
          return __hiloint2double(zz.hi, zz.lo);
        }
      }
    }



    if ((((zz.hi << 1) | zz.lo) == 0) &&
        ((((xx.hi << 1) | xx.lo) == 0) ||
         (((yy.hi << 1) | yy.lo) == 0))) {
      if (rndMode == cudaRoundMinInf) {
        return __hiloint2double((xx.hi ^ yy.hi ^ zz.hi) & 0x80000000, zz.lo);
      } else {
        zz.hi &= 0x7fffffff;
        return __hiloint2double(zz.hi, zz.lo);
      }
    }




    if ((((xx.hi << 1) | xx.lo) == 0) ||
        (((yy.hi << 1) | yy.lo) == 0)) {
      return __hiloint2double(zz.hi, zz.lo);
    }

    if (expo_x == 0xffffffff) {
      expo_x++;
      t = xx.hi & 0x80000000;
      s = xx.lo >> 21;
      xx.lo = xx.lo << 11;
      xx.hi = xx.hi << 11;
      xx.hi = xx.hi | s;
      if (!xx.hi) {
        xx.hi = xx.lo;
        xx.lo = 0;
        expo_x -= 32;
      }
      while ((int)xx.hi > 0) {
        s = xx.lo >> 31;
        xx.lo = xx.lo + xx.lo;
        xx.hi = xx.hi + xx.hi;
        xx.hi = xx.hi | s;
        expo_x--;
      }
      xx.lo = (xx.lo >> 11);
      xx.lo |= (xx.hi << 21);
      xx.hi = (xx.hi >> 11) | t;
    }
    if (expo_y == 0xffffffff) {
      expo_y++;
      t = yy.hi & 0x80000000;
      s = yy.lo >> 21;
      yy.lo = yy.lo << 11;
      yy.hi = yy.hi << 11;
      yy.hi = yy.hi | s;
      if (!yy.hi) {
        yy.hi = yy.lo;
        yy.lo = 0;
        expo_y -= 32;
      }
      while ((int)yy.hi > 0) {
        s = yy.lo >> 31;
        yy.lo = yy.lo + yy.lo;
        yy.hi = yy.hi + yy.hi;
        yy.hi = yy.hi | s;
        expo_y--;
      }
      yy.lo = (yy.lo >> 11);
      yy.lo |= (yy.hi << 21);
      yy.hi = (yy.hi >> 11) | t;
    }
    if (expo_z == 0xffffffff) {
      expo_z++;
      t = zz.hi & 0x80000000;
      s = zz.lo >> 21;
      zz.lo = zz.lo << 11;
      zz.hi = zz.hi << 11;
      zz.hi = zz.hi | s;
      if (!zz.hi) {
        zz.hi = zz.lo;
        zz.lo = 0;
        expo_z -= 32;
      }
      while ((int)zz.hi > 0) {
        s = zz.lo >> 31;
        zz.lo = zz.lo + zz.lo;
        zz.hi = zz.hi + zz.hi;
        zz.hi = zz.hi | s;
        expo_z--;
      }
      zz.lo = (zz.lo >> 11);
      zz.lo |= (zz.hi << 21);
      zz.hi = (zz.hi >> 11) | t;
    }
  }

  expo_x = expo_x + expo_y;
  expo_y = xx.hi ^ yy.hi;
  t = xx.lo >> 21;
  xx.lo = xx.lo << 11;
  xx.hi = xx.hi << 11;
  xx.hi = xx.hi | t;
  yy.hi = yy.hi & 0x000fffff;
  xx.hi = xx.hi | 0x80000000;
  yy.hi = yy.hi | 0x00100000;

  prod0 = xx.lo * yy.lo;
  prod1 = __umulhi (xx.lo, yy.lo);
  prod2 = xx.hi * yy.lo;
  prod3 = xx.lo * yy.hi;
  prod1 += prod2;
  t = prod1 < prod2;
  prod1 += prod3;
  t += prod1 < prod3;
  prod2 = __umulhi (xx.hi, yy.lo);
  prod3 = __umulhi (xx.lo, yy.hi);
  prod2 += prod3;
  s = prod2 < prod3;
  prod3 = xx.hi * yy.hi;
  prod2 += prod3;
  s += prod2 < prod3;
  prod2 += t;
  s += prod2 < t;
  prod3 = __umulhi (xx.hi, yy.hi) + s;

  yy.lo = prod0;
  yy.hi = prod1;
  xx.lo = prod2;
  xx.hi = prod3;
  expo_x = expo_x - (1023 - 2);
  expo_y = expo_y & 0x80000000;

  if (xx.hi < 0x00100000) {
    s = xx.lo >> 31;
    s = (xx.hi << 1) + s;
    xx.hi = s;
    s = yy.hi >> 31;
    s = (xx.lo << 1) + s;
    xx.lo = s;
    s = yy.lo >> 31;
    s = (yy.hi << 1) + s;
    yy.hi = s;
    s = yy.lo << 1;
    yy.lo = s;
    expo_x--;
  }

  t = 0;
  if (((zz.hi << 1) | zz.lo) != 0) {

    s = zz.hi & 0x80000000;

    zz.hi &= 0x000fffff;
    zz.hi |= 0x00100000;
    ww.hi = 0;
    ww.lo = 0;


    if ((int)expo_z > (int)expo_x) {
      t = expo_z;
      expo_z = expo_x;
      expo_x = t;
      t = zz.hi;
      zz.hi = xx.hi;
      xx.hi = t;
      t = zz.lo;
      zz.lo = xx.lo;
      xx.lo = t;
      t = ww.hi;
      ww.hi = yy.hi;
      yy.hi = t;
      t = ww.lo;
      ww.lo = yy.lo;
      yy.lo = t;
      t = expo_y;
      expo_y = s;
      s = t;
    }



    expo_z = expo_x - expo_z;
    u = expo_y ^ s;
    if (expo_z <= 107) {

      t = 0;
      while (expo_z >= 32) {
        t = ww.lo | (t != 0);
        ww.lo = ww.hi;
        ww.hi = zz.lo;
        zz.lo = zz.hi;
        zz.hi = 0;
        expo_z -= 32;
      }
      if (expo_z) {
        t = (t >> expo_z) | (ww.lo << (32 - expo_z)) |
                ((t << (32 - expo_z)) != 0);
        ww.lo = (ww.lo >> expo_z) | (ww.hi << (32 - expo_z));
        ww.hi = (ww.hi >> expo_z) | (zz.lo << (32 - expo_z));
        zz.lo = (zz.lo >> expo_z) | (zz.hi << (32 - expo_z));
        zz.hi = (zz.hi >> expo_z);
      }
    } else {
      t = 1;
      ww.lo = 0;
      ww.hi = 0;
      zz.lo = 0;
      zz.hi = 0;
    }
    if ((int)u < 0) {

      t = (unsigned)(-(int)t);
      s = (t != 0);
      u = yy.lo - s;
      s = u > yy.lo;
      yy.lo = u - ww.lo;
      s += yy.lo > u;
      u = yy.hi - s;
      s = u > yy.hi;
      yy.hi = u - ww.hi;
      s += yy.hi > u;
      u = xx.lo - s;
      s = u > xx.lo;
      xx.lo = u - zz.lo;
      s += xx.lo > u;
      xx.hi = (xx.hi - zz.hi) - s;
      if (!(xx.hi | xx.lo | yy.hi | yy.lo | t)) {

        if (rndMode == cudaRoundMinInf) {
          return __hiloint2double(0x80000000, xx.lo);
        } else {
          return __hiloint2double(xx.hi, xx.lo);
        }
      }
      if ((int)xx.hi < 0) {



        t = ~t;
        yy.lo = ~yy.lo;
        yy.hi = ~yy.hi;
        xx.lo = ~xx.lo;
        xx.hi = ~xx.hi;
        if (++t == 0) {
          if (++yy.lo == 0) {
            if (++yy.hi == 0) {
              if (++xx.lo == 0) {
              ++xx.hi;
              }
            }
          }
        }
        expo_y ^= 0x80000000;
      }


      while (!(xx.hi & 0x00100000)) {
        xx.hi = (xx.hi << 1) | (xx.lo >> 31);
        xx.lo = (xx.lo << 1) | (yy.hi >> 31);
        yy.hi = (yy.hi << 1) | (yy.lo >> 31);
        yy.lo = (yy.lo << 1);
        expo_x--;
      }
    } else {

      yy.lo = yy.lo + ww.lo;
      s = yy.lo < ww.lo;
      yy.hi = yy.hi + s;
      u = yy.hi < s;
      yy.hi = yy.hi + ww.hi;
      u += yy.hi < ww.hi;
      xx.lo = xx.lo + u;
      s = xx.lo < u;
      xx.lo = xx.lo + zz.lo;
      s += xx.lo < zz.lo;
      xx.hi = xx.hi + zz.hi + s;
      if (xx.hi & 0x00200000) {
        t = t | (yy.lo << 31);
        yy.lo = (yy.lo >> 1) | (yy.hi << 31);
        yy.hi = (yy.hi >> 1) | (xx.lo << 31);
        xx.lo = (xx.lo >> 1) | (xx.hi << 31);
        xx.hi = ((xx.hi & 0x80000000) | (xx.hi >> 1)) & ~0x40000000;
        expo_x++;
      }
    }
  }
  t = yy.lo | (t != 0);
  t = yy.hi | (t != 0);

  xx.hi |= expo_y;
  if (expo_x <= 0x7FD) {

    xx.hi = xx.hi & ~0x00100000;
    s = xx.lo & 1;
    u = xx.lo;
    if (rndMode == cudaRoundNearest) {
      xx.lo += (t == 0x80000000) ? s : (t >> 31);
    } else if (((rndMode == cudaRoundPosInf) && t && (!expo_y)) ||
               ((rndMode == cudaRoundMinInf) && t && expo_y)) {
      xx.lo += 1;
    }
    xx.hi += (u > xx.lo);
    xx.hi += ((expo_x + 1) << 20);
    return __hiloint2double(xx.hi, xx.lo);
  } else if ((int)expo_x >= 2046) {

    if ((rndMode == cudaRoundNearest) ||
        ((rndMode == cudaRoundPosInf) && (!expo_y)) ||
        ((rndMode == cudaRoundMinInf) && expo_y)) {
      xx.hi = (xx.hi & 0x80000000) | 0x7ff00000;
      xx.lo = 0;
    } else {
      xx.hi = (xx.hi & 0x80000000) | 0x7fefffff;
      xx.lo = 0xffffffff;
    }
    return __hiloint2double(xx.hi, xx.lo);
  }

  expo_x = (unsigned)(-(int)expo_x);
  if (expo_x > 54) {

    if (((rndMode == cudaRoundPosInf) && (!expo_y)) ||
        ((rndMode == cudaRoundMinInf) && expo_y)) {
      return __hiloint2double(xx.hi & 0x80000000, 1);
    } else {
      return __hiloint2double(xx.hi & 0x80000000, 0);
    }
  }
  yy.hi = xx.hi & 0x80000000;
  xx.hi = xx.hi & ~0xffe00000;
  if (expo_x >= 32) {
    t = xx.lo | (t != 0);
    xx.lo = xx.hi;
    xx.hi = 0;
    expo_x -= 32;
  }
  if (expo_x) {
    t = (t >> expo_x) | (xx.lo << (32 - expo_x)) | (t != 0);
    xx.lo = (xx.lo >> expo_x) | (xx.hi << (32 - expo_x));
    xx.hi = (xx.hi >> expo_x);
  }
  expo_x = xx.lo & 1;
  u = xx.lo;
  if (rndMode == cudaRoundNearest) {
    xx.lo += (t == 0x80000000) ? expo_x : (t >> 31);
  } else if (((rndMode == cudaRoundPosInf) && t && (!expo_y)) ||
             ((rndMode == cudaRoundMinInf) && t && expo_y)) {
    xx.lo += 1;
  }
  xx.hi += (u > xx.lo);
  xx.hi |= yy.hi;
  return __hiloint2double(xx.hi, xx.lo);
}

static double __fma_rn(double x, double y, double z)
{
  return __internal_fma_kernel(x, y, z, cudaRoundNearest);
}

static double __fma_rd(double x, double y, double z)
{
  return __internal_fma_kernel(x, y, z, cudaRoundMinInf);
}

static double __fma_ru(double x, double y, double z)
{
  return __internal_fma_kernel(x, y, z, cudaRoundPosInf);
}

static double __fma_rz(double x, double y, double z)
{
  return __internal_fma_kernel(x, y, z, cudaRoundZero);
}

static double __dadd_rz(double a, double b)
{
  return __fma_rz(a, 1.0, b);
}

static double __dadd_ru(double a, double b)
{
  return __fma_ru(a, 1.0, b);
}

static double __dadd_rd(double a, double b)
{
  return __fma_rd(a, 1.0, b);
}

static double __dmul_rz(double a, double b)
{
  return __fma_rz(a, b, __hiloint2double(0x80000000, 0x00000000));
}

static double __dmul_ru(double a, double b)
{
  return __fma_ru(a, b, __hiloint2double(0x80000000, 0x00000000));
}

static double __dmul_rd(double a, double b)
{
  return __fma_rd(a, b, 0.0);
}

static double __dadd_rn(double a, double b)
{
  return __fma_rn(a, 1.0, b);
}

static double __dmul_rn(double a, double b)
{
  return __fma_rn(a, b, __hiloint2double(0x80000000, 0x00000000));
}
# 1473 "/usr/local/cuda/bin/../include/device_functions.h" 2 3
# 1 "/usr/local/cuda/bin/../include/texture_fetch_functions.h" 1 3
# 1910 "/usr/local/cuda/bin/../include/texture_fetch_functions.h" 3
extern void __cudaTextureFetch(const void *tex, void *index, int integer, void *val);

static int4 __itexfetchi(const void *tex, int4 index)
{
  int4 val;

  __cudaTextureFetch(tex, (void*)&index, 1, (void*)&val);

  return val;
}

static uint4 __utexfetchi(const void *tex, int4 index)
{
  uint4 val;

  __cudaTextureFetch(tex, (void*)&index, 1, (void*)&val);

  return val;
}

static float4 __ftexfetchi(const void *tex, int4 index)
{
  float4 val;

  __cudaTextureFetch(tex, (void*)&index, 1, (void*)&val);

  return val;
}

static int4 __itexfetch(const void *tex, float4 index, int dim)
{
  int4 val;

  __cudaTextureFetch(tex, (void*)&index, 0, (void*)&val);

  return val;
}

static uint4 __utexfetch(const void *tex, float4 index, int dim)
{
  uint4 val;

  __cudaTextureFetch(tex, (void*)&index, 0, (void*)&val);

  return val;
}

static float4 __ftexfetch(const void *tex, float4 index, int dim)
{
  float4 val;

  __cudaTextureFetch(tex, (void*)&index, 0, (void*)&val);

  return val;
}
# 1474 "/usr/local/cuda/bin/../include/device_functions.h" 2 3
# 925 "/usr/local/cuda/bin/../include/math_functions.h" 2 3


static int __cuda___signbitf(float a)
{
  return (int)((unsigned int)__float_as_int(a) >> 31);
}




static float __cuda_copysignf(float a, float b)
{
  return __int_as_float((__float_as_int(b) & 0x80000000) |
                        (__float_as_int(a) & ~0x80000000));
}
# 949 "/usr/local/cuda/bin/../include/math_functions.h" 3
extern __attribute__((__weak__)) int min(int a, int b); int min(int a, int b)
{
  return a < b ? a : b;
}

extern __attribute__((__weak__)) unsigned int umin(unsigned int a, unsigned int b); unsigned int umin(unsigned int a, unsigned int b)
{
  return a < b ? a : b;
}

extern __attribute__((__weak__)) int max(int a, int b); int max(int a, int b)
{
  return a > b ? a : b;
}

extern __attribute__((__weak__)) unsigned int umax(unsigned int a, unsigned int b); unsigned int umax(unsigned int a, unsigned int b)
{
  return a > b ? a : b;
}
# 1035 "/usr/local/cuda/bin/../include/math_functions.h" 3
static float __internal_nearbyintf(float a)
{
  float fa = fabsf(a);

  if (fa >= 8388608.0f) {
    return a;
  } else {
    volatile float u = 8388608.0f + fa;

    u = u - 8388608.0f;
    return copysignf(u, a);
  }
}

static float __internal_fminf(float a, float b)
{
  volatile union {
    float f;
    unsigned int i;
  } cvta, cvtb;

  cvta.f = a;
  cvtb.f = b;
  if ((cvta.i << 1) > 0xff000000) return b;
  if ((cvtb.i << 1) > 0xff000000) return a;
  if ((cvta.i | cvtb.i) == 0x80000000) {
    return __int_as_float(0x80000000);
  }
  return a < b ? a : b;
}

static float __internal_fmaxf(float a, float b)
{
  volatile union {
    float f;
    unsigned int i;
  } cvta, cvtb;

  cvta.f = a;
  cvtb.f = b;
  if ((cvta.i << 1) > 0xff000000) return b;
  if ((cvtb.i << 1) > 0xff000000) return a;
  if ((cvta.f == 0.0f) && (cvtb.f == 0.0f)) {
    cvta.i &= cvtb.i;
    return cvta.f;
  }
  return a > b ? a : b;
}
# 1123 "/usr/local/cuda/bin/../include/math_functions.h" 3
static long int __cuda_labs(long int a)
{
  return labs(a);
}

static float __cuda_ceilf(float a)
{
  return ceilf(a);
}

static float __cuda_floorf(float a)
{
  return floorf(a);
}

static float __cuda_sqrtf(float a)
{
   return sqrtf(a);
}

static float __cuda_rsqrtf(float a)
{
   return 1.0f / sqrtf(a);
}

static float __cuda_truncf(float a)
{
  return truncf(a);
}

static int __cuda_max(int a, int b)
{
  return max(a, b);
}

static int __cuda_min(int a, int b)
{
  return min(a, b);
}

static unsigned int __cuda_umax(unsigned int a, unsigned int b)
{
  return umax(a, b);
}

static unsigned int __cuda_umin(unsigned int a, unsigned int b)
{
  return umin(a, b);
}

static long long int __cuda_llrintf(float a)
{



  return __float2ll_rn(a);

}

static long int __cuda_lrintf(float a)
{




  return (long int)__cuda_llrintf(a);




}

static float __cuda_nearbyintf(float a)
{





  return __internal_nearbyintf(a);

}

static float __cuda_fmaxf(float a, float b)
{





  return __internal_fmaxf(a, b);

}

static float __cuda_fminf(float a, float b)
{





  return __internal_fminf(a, b);

}
# 1244 "/usr/local/cuda/bin/../include/math_functions.h" 3
static int __cuda___finitef(float a)
{
  return __cuda_fabsf(a) < __int_as_float(0x7f800000);
}
# 1258 "/usr/local/cuda/bin/../include/math_functions.h" 3
static int __cuda___isinff(float a)
{
  return __cuda_fabsf(a) == __int_as_float(0x7f800000);
}

static int __cuda___isnanf(float a)
{
  return !(__cuda_fabsf(a) <= __int_as_float(0x7f800000));
}

static float __cuda_nextafterf(float a, float b)
{
  unsigned int ia;
  unsigned int ib;
  ia = __float_as_int(a);
  ib = __float_as_int(b);




  if (__cuda___isnanf(a) || __cuda___isnanf(b)) return a + b;
  if (__int_as_float (ia | ib) == 0.0f) return __int_as_float(ib);





  if (__int_as_float(ia) == 0.0f) {
    return __cuda_copysignf(__int_as_float(0x00000001), b);
  }

  if ((a < b) && (a < 0.0f)) ia--;
  if ((a < b) && (a > 0.0f)) ia++;
  if ((a > b) && (a < 0.0f)) ia++;
  if ((a > b) && (a > 0.0f)) ia--;
  a = __int_as_float(ia);





  return a;
}

static float __cuda_nanf(const char *tagp)
{

  return __int_as_float(0x7fffffff);
}


static float __internal_atanhf_kernel(float a_1, float a_2)
{
  float a, a2, t;

  a = a_1 + a_2;
  a2 = a * a;
  t = 1.566305595598990E-001f/64.0f;
  t = t * a2 + 1.995081856004762E-001f/16.0f;
  t = t * a2 + 3.333382699617026E-001f/4.0f;
  t = t * a2;
  t = t * a + a_2;
  t = t + a_1;
  return t;
}




static float __internal_atanf_kernel(float a)
{
  float t4, t0, t1;

  t4 = a * a;
  t0 = - 5.674867153f;
  t0 = t4 * - 0.823362947f + t0;
  t0 = t0 * t4 - 6.565555096f;
  t0 = t0 * t4;
  t0 = t0 * a;
  t1 = t4 + 11.33538818f;
  t1 = t1 * t4 + 28.84246826f;
  t1 = t1 * t4 + 19.69667053f;
  t1 = 1.0f / t1;
  a = t0 * t1 + a;
  return a;
}


static float __internal_tan_kernel(float a)
{
  float a2, s, t;

  a2 = a * a;
  t = 4.114678393115178E-003f * a2 - 8.231194034909670E-001f;
  s = a2 - 2.469348886157666E+000f;
  s = 1.0f / s;
  t = t * s;
  t = t * a2;
  t = t * a + a;
  return t;
}

static float __internal_accurate_logf(float a)
{
  float t;
  float z;
  float m;
  int ia, e;
  ia = __float_as_int(a);

  if ((ia < 0x00800000) || (ia > 0x7f7fffff)) {
    return __logf(a);
  }

  m = __int_as_float((ia & 0x807fffff) | 0x3f800000);
  e = ((unsigned)ia >> 23) - 127;
  if (m > 1.414213562f) {
    m = m * 0.5f;
    e = e + 1;
  }
  t = m - 1.0f;
  z = m + 1.0f;
  z = t / z;
  z = -t * z;
  z = __internal_atanhf_kernel(t, z);
  z = (float)e * 0.693147181f + z;
  return z;
}

static float2 __internal_log_ep(float a)
{
  float2 res;
  int expo;
  float m;
  float log_hi, log_lo;
  float t_hi, t_lo;
  float f, g, u, v, q;



  float r, s, e;

  expo = (__float_as_int(a) >> 23) & 0xff;


  if (expo == 0) {
    a *= 16777216.0f;
    expo = (__float_as_int(a) >> 23) & 0xff;
    expo -= 24;
  }

  expo -= 127;
  m = __int_as_float((__float_as_int(a) & 0x807fffff) | 0x3f800000);
  if (m > 1.414213562f) {
    m = m * 0.5f;
    expo = expo + 1;
  }






  f = m - 1.0f;
  g = m + 1.0f;
  g = 1.0f / g;
  u = 2.0f * f * g;
  v = u * u;
  q = 1.49356810919559350E-001f/64.0f;
  q = q * v + 1.99887797540072460E-001f/16.0f;
  q = q * v + 3.33333880955515580E-001f/4.0f;
  q = q * v;
  q = q * u;
  log_hi = __int_as_float(__float_as_int(u) & 0xfffff000);
  v = __int_as_float(__float_as_int(f) & 0xfffff000);
  u = 2.0f * (f - log_hi);
  f = f - v;
  u = u - log_hi * v;
  u = u - log_hi * f;
  u = g * u;



  r = log_hi + u;
  s = u - (r - log_hi);
  log_hi = r;
  log_lo = s;

  r = log_hi + q;
  s = ((log_hi - r) + q) + log_lo;
  log_hi = e = r + s;
  log_lo = (r - e) + s;


  t_hi = expo * 0.6931457519f;
  t_lo = expo * 1.4286067653e-6f;


  r = t_hi + log_hi;
  s = (((t_hi - r) + log_hi) + log_lo) + t_lo;
  res.y = e = r + s;
  res.x = (r - e) + s;
  return res;
}

static float __internal_accurate_log2f(float a)
{
  return 1.442695041f * __internal_accurate_logf(a);
}




static float2 __internal_dsmul (float2 x, float2 y)
{
    float2 z;

    volatile float up, vp, u1, u2, v1, v2, mh, ml;



    up = x.y * 4097.0f;
    u1 = (x.y - up) + up;
    u2 = x.y - u1;
    vp = y.y * 4097.0f;
    v1 = (y.y - vp) + vp;
    v2 = y.y - v1;
    mh = __fmul_rn(x.y,y.y);
    ml = (((u1 * v1 - mh) + u1 * v2) + u2 * v1) + u2 * v2;
    ml = (__fmul_rn(x.y,y.x) + __fmul_rn(x.x,y.y)) + ml;
    z.y = up = mh + ml;
    z.x = (mh - up) + ml;
    return z;
}


static unsigned int __cudart_i2opi_f [] = {
  0x3c439041,
  0xdb629599,
  0xf534ddc0,
  0xfc2757d1,
  0x4e441529,
  0xa2f9836e,
};


static float __internal_trig_reduction_kernel(float a, int *quadrant)
{
  float j;
  int q;
  if (__cuda_fabsf(a) > 48039.0f) {

    unsigned int ia = __float_as_int(a);
    unsigned int s = ia & 0x80000000;
    unsigned int result[7];
    unsigned int phi, plo;
    unsigned int hi, lo;
    unsigned int e;
    int idx;
    e = ((ia >> 23) & 0xff) - 128;
    ia = (ia << 8) | 0x80000000;

    idx = 4 - (e >> 5);
    hi = 0;



    for (q = 0; q < 6; q++) {
      plo = __cudart_i2opi_f[q] * ia;
      phi = __umulhi (__cudart_i2opi_f[q], ia);
      lo = hi + plo;
      hi = phi + (lo < plo);
      result[q] = lo;
    }
    result[q] = hi;
    e = e & 31;



    hi = result[idx+2];
    lo = result[idx+1];
    if (e) {
      q = 32 - e;
      hi = (hi << e) | (lo >> q);
      lo = (lo << e) | (result[idx] >> q);
    }
    q = hi >> 30;

    hi = (hi << 2) | (lo >> 30);
    lo = (lo << 2);
    e = (hi + (lo > 0)) > 0x80000000;
    q += e;
    if (s) q = -q;
    if (e) {
      unsigned int t;
      hi = ~hi;
      lo = -(int)lo;
      t = (lo == 0);
      hi += t;
      s = s ^ 0x80000000;
    }
    *quadrant = q;

    e = 0;
    while ((int)hi > 0) {
      hi = (hi << 1) | (lo >> 31);
      lo = (lo << 1);
      e--;
    }
    lo = hi * 0xc90fdaa2;
    hi = __umulhi(hi, 0xc90fdaa2);
    if ((int)hi > 0) {
      hi = (hi << 1) | (lo >> 31);
      lo = (lo << 1);
      e--;
    }
    hi = hi + (lo > 0);
    ia = s | (((e + 126) << 23) + (hi >> 8) + ((hi << 24) >= 0x80000000));
    return __int_as_float(ia);
  }
  q = __float2int_rn(a * 0.636619772f);
  j = (float)q;
  a = a - j * 1.5703125000000000e+000f;
  a = a - j * 4.8351287841796875e-004f;
  a = a - j * 3.1385570764541626e-007f;
  a = a - j * 6.0771005065061922e-011f;
  *quadrant = q;
  return a;
}
# 1597 "/usr/local/cuda/bin/../include/math_functions.h" 3
static float __internal_expf_kernel(float a, float scale)
{
  float j, z;

  j = __cuda_truncf(a * 1.442695041f);
  z = a - j * 0.6931457519f;
  z = z - j * 1.4286067653e-6f;
  z = z * 1.442695041f;
  z = __cuda_exp2f(z) * __cuda_exp2f(j + scale);
  return z;
}

static float __internal_accurate_expf(float a)
{
  float z;
  z = __internal_expf_kernel(a, 0.0f);
  if (a < -105.0f) z = 0.0f;
  if (a > 105.0f) z = __int_as_float(0x7f800000);
  return z;
}

static float __internal_accurate_exp10f(float a)
{
  float j, z;
  j = __cuda_truncf(a * 3.321928094f);
  z = a - j * 3.0102920532226563e-001f;
  z = z - j * 7.9034171557301747e-007f;
  z = z * 3.321928094f;
  z = __cuda_exp2f(z) * __cuda_exp2f(j);
  if (a < -46.0f) z = 0.0f;
  if (a > 46.0f) z = __int_as_float(0x7f800000);
  return z;
}

static float __internal_lgammaf_pos(float a)
{
  float sum;
  float s, t;

  if (a == __int_as_float(0x7f800000)) {
    return a;
  }
  if (a >= 3.0f) {
    if (a >= 7.8f) {



      s = 1.0f / a;
      t = s * s;
      sum = 0.77783067e-3f;
      sum = sum * t - 0.2777655457e-2f;
      sum = sum * t + 0.83333273853e-1f;
      sum = sum * s + 0.918938533204672f;
      s = 0.5f * __internal_accurate_logf(a);
      t = a - 0.5f;
      s = s * t;
      t = s - a;
      s = __fadd_rn(s, sum);
      t = t + s;
      return t;
    } else {
      a = a - 3.0f;
      s = - 7.488903254816711E+002f;
      s = s * a - 1.234974215949363E+004f;
      s = s * a - 4.106137688064877E+004f;
      s = s * a - 4.831066242492429E+004f;
      s = s * a - 1.430333998207429E+005f;
      t = a - 2.592509840117874E+002f;
      t = t * a - 1.077717972228532E+004f;
      t = t * a - 9.268505031444956E+004f;
      t = t * a - 2.063535768623558E+005f;
      t = s / t;
      t = t + a;
      return t;
    }
  } else if (a >= 1.5f) {
    a = a - 2.0f;
    t = + 4.959849168282574E-005f;
    t = t * a - 2.208948403848352E-004f;
    t = t * a + 5.413142447864599E-004f;
    t = t * a - 1.204516976842832E-003f;
    t = t * a + 2.884251838546602E-003f;
    t = t * a - 7.382757963931180E-003f;
    t = t * a + 2.058131963026755E-002f;
    t = t * a - 6.735248600734503E-002f;
    t = t * a + 3.224670187176319E-001f;
    t = t * a + 4.227843368636472E-001f;
    t = t * a;
    return t;
  } else if (a >= 0.7f) {
    a = 1.0f - a;
    t = + 4.588266515364258E-002f;
    t = t * a + 1.037396712740616E-001f;
    t = t * a + 1.228036339653591E-001f;
    t = t * a + 1.275242157462838E-001f;
    t = t * a + 1.432166835245778E-001f;
    t = t * a + 1.693435824224152E-001f;
    t = t * a + 2.074079329483975E-001f;
    t = t * a + 2.705875136435339E-001f;
    t = t * a + 4.006854436743395E-001f;
    t = t * a + 8.224669796332661E-001f;
    t = t * a + 5.772156651487230E-001f;
    t = t * a;
    return t;
  } else {
    t = + 3.587515669447039E-003f;
    t = t * a - 5.471285428060787E-003f;
    t = t * a - 4.462712795343244E-002f;
    t = t * a + 1.673177015593242E-001f;
    t = t * a - 4.213597883575600E-002f;
    t = t * a - 6.558672843439567E-001f;
    t = t * a + 5.772153712885004E-001f;
    t = t * a;
    t = t * a + a;
    return -__internal_accurate_logf(t);
  }
}


static float __internal_sin_kernel(float x)
{
  float x2, z;

  x2 = x * x;
  z = - 1.95152959e-4f;
  z = z * x2 + 8.33216087e-3f;
  z = z * x2 - 1.66666546e-1f;
  z = z * x2;
  z = z * x + x;

  return z;
}


static float __internal_cos_kernel(float x)
{
  float x2, z;

  x2 = x * x;
  z = 2.44331571e-5f;
  z = z * x2 - 1.38873163e-3f;
  z = z * x2 + 4.16666457e-2f;
  z = z * x2 - 5.00000000e-1f;
  z = z * x2 + 1.00000000e+0f;
  return z;
}

static float __internal_accurate_sinf(float a)
{
  float z;
  int i;

  if ((__cuda___isinff(a)) || (a == 0.0f)) {
    return __fmul_rn (a, 0.0f);
  }
  z = __internal_trig_reduction_kernel(a, &i);

  if (i & 1) {
    z = __internal_cos_kernel(z);
  } else {
    z = __internal_sin_kernel(z);
  }
  if (i & 2) {
    z = -z;
  }
  return z;
}







static float __cuda_rintf(float a)
{



  return __cuda_nearbyintf(a);

}

static float __cuda_sinf(float a)
{





  return __internal_accurate_sinf(a);

}

static float __cuda_cosf(float a)
{





  float z;
  int i;

  if (__cuda___isinff(a)) {
    return __int_as_float(0x7fffffff);
  }
  z = __internal_trig_reduction_kernel(a, &i);

  i++;
  if (i & 1) {
    z = __internal_cos_kernel(z);
  } else {
    z = __internal_sin_kernel(z);
  }
  if (i & 2) {
    z = -z;
  }
  return z;

}

static float __cuda_tanf(float a)
{





  float z;
  int i;

  if (__cuda___isinff(a)) {
    return __int_as_float(0x7fffffff);
  }
  z = __internal_trig_reduction_kernel(a, &i);

  z = __internal_tan_kernel(z);
  if (i & 1) {
    z = -1.0f / z;
  }
  return z;

}

static float __cuda_log2f(float a)
{





  return __internal_accurate_log2f(a);

}

static float __cuda_expf(float a)
{





  return __internal_accurate_expf(a);

}

static float __cuda_exp10f(float a)
{





  return __internal_accurate_exp10f(a);

}

static float __cuda_coshf(float a)
{
  float z;

  a = __cuda_fabsf(a);
  z = __internal_expf_kernel(a, -2.0f);
  z = 2.0f * z + 0.125f / z;
  if (a >= 90.0f) {
    z = __int_as_float(0x7f800000);
  }
  return z;
}

static float __cuda_sinhf(float a)
{
  float s, z;

  s = a;
  a = __cuda_fabsf(a);
  if (a < 1.0f) {
    float a2 = a * a;

    z = 2.816951222e-6f;
    z = z * a2 + 1.983615978e-4f;
    z = z * a2 + 8.333350058e-3f;
    z = z * a2 + 1.666666650e-1f;
    z = z * a2;
    z = z * a + a;
  } else {
    z = __internal_expf_kernel(a, -2.0f);
    z = 2.0f * z - 0.125f / z;
    if (a >= 90.0f) {
      z = __int_as_float(0x7f800000);
    }
  }
  return __cuda_copysignf(z, s);
}

static float __cuda_tanhf(float a)
{
  float s, t;

  t = __cuda_fabsf(a);
  if (t < 0.55f) {
    float z, z2;
    z = t;
    z2 = z * z;
    t = 1.643758066599993e-2f;
    t = t * z2 - 5.267181327760551e-2f;
    t = t * z2 + 1.332072505223051e-1f;
    t = t * z2 - 3.333294663641083e-1f;
    t = t * z2;
    s = t * z + z;
  } else {
    s = 1.0f - 2.0f / (__internal_expf_kernel(2.0f * t, 0.0f) + 1.0f);
    if (t >= 88.0f) {
      s = 1.0f;
    }
  }
  return __cuda_copysignf(s, a);
}

static float __cuda_atan2f(float a, float b)
{



  float t0, t1, t3, fa, fb;



  fb = __cuda_fabsf(b);
  fa = __cuda_fabsf(a);

  if (fa == 0.0f && fb == 0.0f) {
    t3 = __cuda___signbitf(b) ? 3.141592654f : 0;
  } else if ((fa == __int_as_float(0x7f800000)) && (fb == __int_as_float(0x7f800000))) {
    t3 = __cuda___signbitf(b) ? 2.356194490f : 0.785398163f;
  } else {

    if (fb < fa) {
      t0 = fa;
      t1 = fb;
    } else {
      t0 = fb;
      t1 = fa;
    }
    t3 = __internal_accurate_fdividef(t1, t0);
    t3 = __internal_atanf_kernel(t3);

    if (fa > fb) t3 = 1.570796327f - t3;
    if (b < 0.0f) t3 = 3.141592654f - t3;
  }
  t3 = __cuda_copysignf(t3, a);

  return t3;

}

static float __cuda_atanf(float a)
{
  float t0, t1;


  t0 = __cuda_fabsf(a);
  t1 = t0;
  if (t0 > 1.0f) {
    t1 = 1.0f / t1;
  }

  t1 = __internal_atanf_kernel(t1);

  if (t0 > 1.0f) {
    t1 = 1.570796327f - t1;
  }
  return __cuda_copysignf(t1, a);
}


static float __internal_asinf_kernel(float a)
{
  float t2, t3, t4;

  t2 = a * a;
  t3 = - 0.501162291f;
  t3 = t3 * t2 + 0.915201485f;
  t3 = t3 * t2;
  t3 = t3 * a;
  t4 = t2 - 5.478654385f;
  t4 = t4 * t2 + 5.491230488f;
  t4 = 1.0f / t4;
  a = t3 * t4 + a;
  return a;
}

static float __cuda_asinf(float a)
{
  float t0, t1, t2;

  t0 = __cuda_fabsf(a);
  t2 = 1.0f - t0;
  t2 = 0.5f * t2;
  t2 = __cuda_sqrtf(t2);
  t1 = t0 > 0.575f ? t2 : t0;
  t1 = __internal_asinf_kernel(t1);
  t2 = -2.0f * t1 + 1.570796327f;
  if (t0 > 0.575f) {
    t1 = t2;
  }
  return __cuda_copysignf(t1, a);
}

static float __cuda_acosf(float a)
{
  float t0, t1, t2;

  t0 = __cuda_fabsf(a);
  t2 = 1.0f - t0;
  t2 = 0.5f * t2;
  t2 = __cuda_sqrtf(t2);
  t1 = t0 > 0.575f ? t2 : t0;
  t1 = __internal_asinf_kernel(t1);
  t1 = t0 > 0.575f ? 2.0f * t1 : 1.570796327f - t1;
  if (__cuda___signbitf(a)) {
    t1 = 3.141592654f - t1;
  }
  return t1;
}

static float __cuda_logf(float a)
{





  return __internal_accurate_logf(a);

}

static float __cuda_log10f(float a)
{





  return 0.434294482f * __internal_accurate_logf(a);

}

static float __cuda_log1pf(float a)
{



  float t;




  if (a >= -0.394f && a <= 0.65f) {

    t = a + 2.0f;
    t = a / t;
    t = -a * t;
    t = __internal_atanhf_kernel (a, t);
  } else {
    t = __internal_accurate_logf (1.0f + a);
  }
  return t;

}

static float __cuda_acoshf(float a)
{



  float t;

  t = a - 1.0f;
  if (__cuda_fabsf(t) > 8388608.0f) {

    return 0.693147181f + __internal_accurate_logf(a);
  } else {
    t = t + __cuda_sqrtf(a * t + t);
    return __cuda_log1pf(t);
  }

}

static float __cuda_asinhf(float a)
{



  float fa, oofa, t;

  fa = __cuda_fabsf(a);
  if (fa > 8.507059173e37f) {
    t = 0.693147181f + __logf(fa);
  } else {
    oofa = 1.0f / fa;
    t = fa + fa / (oofa + __cuda_sqrtf(1.0f + oofa * oofa));
    t = __cuda_log1pf(t);
  }
  return __cuda_copysignf(t, a);

}

static float __cuda_atanhf(float a)
{



  float fa, t;

  fa = __cuda_fabsf(a);
  t = (2.0f * fa) / (1.0f - fa);
  t = 0.5f * __cuda_log1pf(t);
  return __cuda_copysignf(t, a);

}

static float __cuda_expm1f(float a)
{
  float t, z, j, u;

  t = __cuda_rintf (a * 1.442695041f);
  z = a - t * 0.6931457519f;
  z = z - t * 1.4286067653e-6f;

  if (__cuda_fabsf(a) < 0.41f) {
    z = a;
    t = 0.0f;
  }

  j = t;
  if (t == 128.0f) j = j - 1.0f;

  u = 1.38795078474044430E-003f;
  u = u * z + 8.38241261853264930E-003f;
  u = u * z + 4.16678317762833940E-002f;
  u = u * z + 1.66663978874356580E-001f;
  u = u * z + 4.99999940395997040E-001f;
  u = u * z;
  u = u * z + z;
  if (a == 0.0f) u = a;

  z = __cuda_exp2f (j);
  a = z - 1.0f;
  if (a != 0.0f) u = u * z + a;
  if (t == 128.0f) u = u + u;

  if (j > 128.0f) u = __int_as_float(0x7f800000);
  if (j < -25.0f) u = -1.0f;
  return u;
}

static float __cuda_hypotf(float a, float b)
{



  float v, w, t;

  a = __cuda_fabsf(a);
  b = __cuda_fabsf(b);

  if (a > b) {
    v = a;
    w = b;
  } else {
    v = b;
    w = a;
  }
  t = __internal_accurate_fdividef(w, v);
  t = 1.0f + t * t;
  t = v * __cuda_sqrtf(t);
  if (v == 0.0f) {
    t = v + w;
  }
  if ((v == __int_as_float(0x7f800000)) || (w == __int_as_float(0x7f800000))) {
    t = __int_as_float(0x7f800000);
  }
  return t;

}

static float __cuda_cbrtf(float a)
{



  float s, t;

  s = __cuda_fabsf(a);
  if ((a == 0.0f) || (s == __int_as_float(0x7f800000))) {
    return a;
  }
  t = __cuda_exp2f(0.333333333f * __log2f(s));
  t = t - (t - (s / (t * t))) * 0.333333333f;
  if (__cuda___signbitf(a)) {
     t = -t;
  }
  return t;

}

static float __cuda_erff(float a)
{
  float t, r, q;

  t = __cuda_fabsf(a);
  if (t < 1.0f) {
    t = t * t;
    r = -5.58510127926029810E-004f;
    r = r * t + 4.90688891415893070E-003f;
    r = r * t - 2.67027980930150640E-002f;
    r = r * t + 1.12799056505903940E-001f;
    r = r * t - 3.76122956138427440E-001f;
    r = r * t + 1.12837911712623450E+000f;
    a = a * r;
  } else if (t <= __int_as_float(0x7f800000)) {



    q = 0.3275911f * t + 1.0f;
    q = 1.0f / q;
    r = 1.061405429f;
    r = r * q - 1.453152027f;
    r = r * q + 1.421413741f;
    r = r * q - 0.284496736f;
    r = r * q + 0.254829592f;
    r = r * q;
    q = __internal_expf_kernel(-a * a, 0.0f);
    r = 1.0f - q * r;
    if (t >= 5.5f) {
      r = 1.0f;
    }
    a = __int_as_float (__float_as_int(r) | (__float_as_int(a) & 0x80000000));
  }
  return a;
}

static float __cuda_erfcf(float a)
{
  if (a <= 0.55f) {
    return 1.0f - __cuda_erff(a);
  } else if (a > 10.0f) {
    return 0.0f;
  } else {
    float p;
    float q;
    float h;
    float l;




    p = + 4.014893410762552E-006f;
    p = p * a + 5.640401259462436E-001f;
    p = p * a + 2.626649872281140E+000f;
    p = p * a + 5.486372652389673E+000f;
    p = p * a + 5.250714831459401E+000f;
    q = a + 4.651376250488319E+000f;
    q = q * a + 1.026302828878470E+001f;
    q = q * a + 1.140762166021288E+001f;
    q = q * a + 5.251211619089947E+000f;

    h = 1.0f / q;
    q = 2.0f * h - q * h * h;
    p = p * q;

    h = __int_as_float(__float_as_int(a) & 0xfffff000);
    l = __fadd_rn (a, -h);
    q = __fmul_rn (-h, h);
    q = __internal_expf_kernel(q, 0.0f);
    a = a + h;
    l = l * a;
    h = __internal_expf_kernel(-l, 0.0f);
    q = q * h;
    p = p * q;
    return p;
  }
}

static float __cuda_lgammaf(float a)
{
  float t;
  float i;
  int quot;
  t = __internal_lgammaf_pos(__cuda_fabsf(a));
  if (a >= 0.0f) return t;
  a = __cuda_fabsf(a);
  i = __cuda_floorf(a);
  if (a == i) return __int_as_float(0x7f800000);
  if (a < 1e-19f) return -__internal_accurate_logf(a);
  i = __cuda_rintf (2.0f * a);
  quot = (int)i;
  i = a - 0.5f * i;
  i = i * 3.141592654f;
  if (quot & 1) {
    i = __internal_cos_kernel(i);
  } else {
    i = __internal_sin_kernel(i);
  }
  i = __cuda_fabsf(i);
  t = 1.144729886f - __internal_accurate_logf(i * a) - t;
  return t;
}

static float __cuda_ldexpf(float a, int b)
{



  float fa = __cuda_fabsf(a);

  if ((fa == 0.0f) || (fa == __int_as_float(0x7f800000)) || (b == 0)) {
    return a;
  }
  else if (__cuda_abs(b) < 126) {
    return a * __cuda_exp2f((float)b);
  }
  else if (__cuda_abs(b) < 252) {
    int bhalf = b / 2;
    return a * __cuda_exp2f((float)bhalf) * __cuda_exp2f((float)(b - bhalf));
  }
  else {
    int bquarter = b / 4;
    float t = __cuda_exp2f((float)bquarter);
    return a * t * t * t * __cuda_exp2f((float)(b - 3 * bquarter));
  }

}

static float __cuda_scalbnf(float a, int b)
{




  return __cuda_ldexpf(a, b);

}

static float __cuda_scalblnf(float a, long int b)
{



  int t;
  if (b > 2147483647L) {
    t = 2147483647;
  } else if (b < (-2147483647 - 1)) {
    t = (-2147483647 - 1);
  } else {
    t = (int)b;
  }
  return __cuda_scalbnf(a, t);

}

static float __cuda_frexpf(float a, int *b)
{
  float fa = __cuda_fabsf(a);
  unsigned int expo;
  unsigned int denorm;

  if (fa < 1.175494351e-38f) {
    a *= 16777216.0f;
    denorm = 24;
  } else {
    denorm = 0;
  }
  expo = ((__float_as_int(a) >> 23) & 0xff);
  if ((fa == 0.0f) || (expo == 0xff)) {
    expo = 0;
    a = a + a;
  } else {
    expo = expo - denorm - 126;
    a = __int_as_float(((__float_as_int(a) & 0x807fffff) | 0x3f000000));
  }
  *b = expo;
  return a;
}

static float __cuda_modff(float a, float *b)
{



  float t;
  if (__cuda___finitef(a)) {
    t = __cuda_truncf(a);
    *b = t;
    t = a - t;
    return __cuda_copysignf(t, a);
  } else if (__cuda___isinff(a)) {
    t = 0.0f;
    *b = a;
    return __cuda_copysignf(t, a);
  } else {
    *b = a;
    return a;
  }

}

static float __cuda_fmodf(float a, float b)
{



  float orig_a = a;
  float orig_b = b;
  a = __cuda_fabsf(a);
  b = __cuda_fabsf(b);
  if (!((a <= __int_as_float(0x7f800000)) && (b <= __int_as_float(0x7f800000)))) {
    return orig_a + orig_b;
  }
  if ((a == __int_as_float(0x7f800000)) || (b == 0.0f)) {
    return __int_as_float(0x7fffffff);
  } else if (a >= b) {


    int expoa = (a < 1.175494351e-38f) ?
        ((int)__log2f(a)) : (((__float_as_int(a) >> 23) & 0xff) - 127);
    int expob = (b < 1.175494351e-38f) ?
        ((int)__log2f(b)) : (((__float_as_int(b) >> 23) & 0xff) - 127);
    int scale = expoa - expob;
    float scaled_b = __cuda_ldexpf(b, scale);
    if (scaled_b <= 0.5f * a) {
      scaled_b *= 2.0f;
    }







    while (scaled_b >= b) {
      if (a >= scaled_b) {
        a -= scaled_b;
      }
      scaled_b *= 0.5f;
    }
    return __cuda_copysignf(a, orig_a);
  } else {
    return orig_a;
  }

}

static float __cuda_remainderf(float a, float b)
{

  float twoa = 0.0f;
  unsigned int quot0 = 0;
  float orig_a = a;
  float orig_b = b;

  a = __cuda_fabsf(a);
  b = __cuda_fabsf(b);
  if (!((a <= __int_as_float(0x7f800000)) && (b <= __int_as_float(0x7f800000)))) {
    return orig_a + orig_b;
  }
  if ((a == __int_as_float(0x7f800000)) || (b == 0.0f)) {
    return __int_as_float(0x7fffffff);
  } else if (a >= b) {

    int expoa = (a < 1.175494351e-38f) ?
        ((int)__log2f(a)) : (((__float_as_int(a) >> 23) & 0xff) - 127);
    int expob = (b < 1.175494351e-38f) ?
        ((int)__log2f(b)) : (((__float_as_int(b) >> 23) & 0xff) - 127);
    int scale = expoa - expob;
    float scaled_b = __cuda_ldexpf(b, scale);
    if (scaled_b <= 0.5f * a) {
      scaled_b *= 2.0f;
    }
# 2509 "/usr/local/cuda/bin/../include/math_functions.h" 3
    while (scaled_b >= b) {
      quot0 = 0;
      if (a >= scaled_b) {
        twoa = (2.0f * a - scaled_b) - scaled_b;
        a -= scaled_b;
        quot0 = 1;
      }
      scaled_b *= 0.5f;
    }
  }


  twoa = 2.0f * a;
  if ((twoa > b) || ((twoa == b) && quot0)) {
    a -= b;
    a = __cuda_copysignf (a, -1.0f);
  }
# 2541 "/usr/local/cuda/bin/../include/math_functions.h" 3
  a = __int_as_float((__float_as_int(orig_a) & 0x80000000)^
                     __float_as_int(a));
  return a;
}

static float __cuda_remquof(float a, float b, int* quo)
{
  float twoa = 0.0f;
  unsigned int quot = 0;
  unsigned int sign;
  float orig_a = a;
  float orig_b = b;


  sign = 0 - (__cuda___signbitf(a) != __cuda___signbitf(b));
  a = __cuda_fabsf(a);
  b = __cuda_fabsf(b);
  if (!((a <= __int_as_float(0x7f800000)) && (b <= __int_as_float(0x7f800000)))) {
    *quo = quot;
    return orig_a + orig_b;
  }
  if ((a == __int_as_float(0x7f800000)) || (b == 0.0f)) {
    *quo = quot;
    return __int_as_float(0x7fffffff);
  } else if (a >= b) {


    int expoa = (a < 1.175494351e-38f) ?
        ((int)__log2f(a)) : (((__float_as_int(a) >> 23) & 0xff) - 127);
    int expob = (b < 1.175494351e-38f) ?
        ((int)__log2f(b)) : (((__float_as_int(b) >> 23) & 0xff) - 127);
    int scale = expoa - expob;
    float scaled_b = __cuda_ldexpf(b, scale);
    if (scaled_b <= 0.5f * a) {
      scaled_b *= 2.0f;
    }
# 2594 "/usr/local/cuda/bin/../include/math_functions.h" 3
    while (scaled_b >= b) {
      quot <<= 1;
      if (a >= scaled_b) {
        twoa = (2.0f * a - scaled_b) - scaled_b;
        a -= scaled_b;
        quot += 1;
      }
      scaled_b *= 0.5f;
    }
  }


  twoa = 2.0f * a;
  if ((twoa > b) || ((twoa == b) && (quot & 1))) {
    quot++;
    a -= b;
    a = __cuda_copysignf (a, -1.0f);
  }
# 2629 "/usr/local/cuda/bin/../include/math_functions.h" 3
  a = __int_as_float((__float_as_int(orig_a) & 0x80000000)^
                     __float_as_int(a));
  quot = quot & (~((~0)<<3));
  quot = quot ^ sign;
  quot = quot - sign;
  *quo = quot;
  return a;
}

static float __cuda_fmaf(float a, float b, float c)
{
  unsigned int xx, yy, zz, ww;
  unsigned int temp, s, u;
  unsigned int expo_x, expo_y, expo_z;

  xx = __float_as_int(a);
  yy = __float_as_int(b);
  zz = __float_as_int(c);
# 2655 "/usr/local/cuda/bin/../include/math_functions.h" 3
  temp = 0xff;
  expo_x = temp & (xx >> 23);
  expo_x = expo_x - 1;
  expo_y = temp & (yy >> 23);
  expo_y = expo_y - 1;
  expo_z = temp & (zz >> 23);
  expo_z = expo_z - 1;

  if (!((expo_x <= 0xFD) &&
        (expo_y <= 0xFD) &&
        (expo_z <= 0xFD))) {




    if ((yy << 1) > 0xff000000) {
      return __int_as_float(0x7fffffff);
    }
    if ((zz << 1) > 0xff000000) {
      return __int_as_float(0x7fffffff);
    }
    if ((xx << 1) > 0xff000000) {
      return __int_as_float(0x7fffffff);
    }
# 2690 "/usr/local/cuda/bin/../include/math_functions.h" 3
    if ((((xx << 1) == 0) && ((yy << 1) == 0xff000000)) ||
        (((yy << 1) == 0) && ((xx << 1) == 0xff000000))) {
      return __int_as_float(0x7fffffff);
    }
    if ((zz << 1) == 0xff000000) {
      if (((yy << 1) == 0xff000000) || ((xx << 1) == 0xff000000)) {
        if ((int)(xx ^ yy ^ zz) < 0) {
          return __int_as_float(0x7fffffff);
        }
      }
    }




    if ((xx << 1) == 0xff000000) {
      xx = xx ^ (yy & 0x80000000);
      return __int_as_float(xx);
    }
    if ((yy << 1) == 0xff000000) {
      yy = yy ^ (xx & 0x80000000);
      return __int_as_float(yy);
    }
    if ((zz << 1) == 0xff000000) {
      return __int_as_float(zz);
    }





    if (zz == 0x80000000) {
      if (((xx << 1) == 0) || ((yy << 1) == 0)) {
        if ((int)(xx ^ yy) < 0) {
          return __int_as_float(zz);
        }
      }
    }



    if (((zz << 1) == 0) &&
        (((xx << 1) == 0) || ((yy << 1) == 0))) {
      zz &= 0x7fffffff;
      return __int_as_float(zz);
    }



    if (((xx << 1) == 0) || ((yy << 1) == 0)) {
      return __int_as_float(zz);
    }

    if (expo_x == (unsigned int)-1) {
      temp = xx & 0x80000000;
      xx = xx << 8;
      while (!(xx & 0x80000000)) {
        xx <<= 1;
        expo_x--;
      }
      expo_x++;
      xx = (xx >> 8) | temp;
    }

    if (expo_y == (unsigned int)-1) {
      temp = yy & 0x80000000;
      yy = yy << 8;
      while (!(yy & 0x80000000)) {
        yy <<= 1;
        expo_y--;
      }
      expo_y++;
      yy = (yy >> 8) | temp;
    }

    if ((expo_z == (unsigned int)-1) && ((zz << 1) != 0)) {
      temp = zz & 0x80000000;
      zz = zz << 8;
      while (!(zz & 0x80000000)) {
        zz <<= 1;
        expo_z--;
      }
      expo_z++;
      zz = (zz >> 8) | temp;
    }
  }

  expo_x = expo_x + expo_y;
  expo_y = xx ^ yy;
  xx = xx & 0x00ffffff;
  yy = yy << 8;
  xx = xx | 0x00800000;
  yy = yy | 0x80000000;

  s = __umulhi(xx, yy);
  yy = xx * yy;
  xx = s;
  expo_x = expo_x - 127 + 2;
  expo_y = expo_y & 0x80000000;


  if (xx < 0x00800000) {
      xx = (xx << 1) | (yy >> 31);
      yy = (yy << 1);
      expo_x--;
  }
  temp = 0;
  if ((zz << 1) != 0) {
    s = zz & 0x80000000;
    zz &= 0x00ffffff;
    zz |= 0x00800000;
    ww = 0;

    if ((int)expo_z > (int)expo_x) {
      temp = expo_z;
      expo_z = expo_x;
      expo_x = temp;
      temp = zz;
      zz = xx;
      xx = temp;
      temp = ww;
      ww = yy;
      yy = temp;
      temp = expo_y;
      expo_y = s;
      s = temp;
    }


    expo_z = expo_x - expo_z;
    u = expo_y ^ s;
    if (expo_z <= 49) {

      temp = 0;
      while (expo_z >= 32) {
        temp = ww | (temp != 0);
        ww = zz;
        zz = 0;
        expo_z -= 32;
      }
      if (expo_z) {
        temp = ((temp >> expo_z) | (ww << (32 - expo_z)) |
                ((temp << (32 - expo_z)) != 0));
        ww = (ww >> expo_z) | (zz << (32 - expo_z));
        zz = (zz >> expo_z);
      }
    } else {
      temp = 1;
      ww = 0;
      zz = 0;
    }
    if ((int)u < 0) {

      temp = (unsigned)(-(int)temp);
      s = (temp != 0);
      u = yy - s;
      s = u > yy;
      yy = u - ww;
      s += yy > u;
      xx = (xx - zz) - s;
      if (!(xx | yy | temp)) {

        return __int_as_float(xx);
      }
      if ((int)xx < 0) {



        temp = ~temp;
        yy = ~yy;
        xx = ~xx;
        if (++temp == 0) {
          if (++yy == 0) {
            ++xx;
          }
        }
        expo_y ^= 0x80000000;
      }

      while (!(xx & 0x00800000)) {
        xx = (xx << 1) | (yy >> 31);
        yy = (yy << 1);
        expo_x--;
      }
    } else {

      yy = yy + ww;
      s = yy < ww;
      xx = xx + zz + s;
      if (xx & 0x01000000) {
        temp = temp | (yy << 31);
        yy = (yy >> 1) | (xx << 31);
        xx = ((xx & 0x80000000) | (xx >> 1)) & ~0x40000000;
        expo_x++;
      }
    }
  }
  temp = yy | (temp != 0);
  if (expo_x <= 0xFD) {

    xx |= expo_y;
    s = xx & 1;
    xx += (temp == 0x80000000) ? s : (temp >> 31);
    xx = xx + (expo_x << 23);
    return __int_as_float(xx);
  } else if ((int)expo_x >= 126) {

    xx = expo_y | 0x7f800000;
    return __int_as_float(xx);
  }

  expo_x = (unsigned int)(-(int)expo_x);
  if (expo_x > 25) {

    return __int_as_float(expo_y);
  }
  yy = (xx << (32 - expo_x)) | ((yy) ? 1 : 0);
  xx = expo_y + (xx >> expo_x);
  xx = xx + ((yy==0x80000000) ? (xx & 1) : (yy >> 31));
  xx |= expo_y;




  return __int_as_float(xx);
}

static float __internal_accurate_powf(float a, float b)
{
  float2 loga, prod;



  float t;



  loga = __internal_log_ep(a);


  if (__cuda_fabsf(b) > 1.0e34f) b *= 1.220703125e-4f;
  prod.y = b;
  prod.x = 0.0f;
  prod = __internal_dsmul (prod, loga);


  if (__float_as_int(prod.y) == 0x42b17218) {
    prod.y = __int_as_float(__float_as_int(prod.y) - 1);
    prod.x = prod.x + __int_as_float(0x37000000);
  }


  t = __cuda_expf(prod.y);

  if (t != __int_as_float(0x7f800000)) {



    t = t * prod.x + t;
  }
  return t;
}

static float __cuda_powif(float a, int b)
{
  unsigned int e = __cuda_abs(b);
  float r = 1.0f;

  while (1) {
    if ((e & 1) != 0) {
      r = r * a;
    }
    e = e >> 1;
    if (e == 0) {
      return b < 0 ? 1.0f/r : r;
    }
    a = a * a;
  }
}

static double __cuda_powi(double a, int b)
{
  unsigned int e = __cuda_abs(b);
  double r = 1.0;

  while (1) {
    if ((e & 1) != 0) {
      r = r * a;
    }
    e = e >> 1;
    if (e == 0) {
      return b < 0 ? 1.0/r : r;
    }
    a = a * a;
  }
}

static float __cuda_powf(float a, float b)
{





  int bIsOddInteger;
  float t;
  if (a == 1.0f || b == 0.0f) {
    return 1.0f;
  }
  if (__cuda___isnanf(a) || __cuda___isnanf(b)) {
    return a + b;
  }
  if (a == __int_as_float(0x7f800000)) {
    return __cuda___signbitf(b) ? 0.0f : __int_as_float(0x7f800000);
  }
  if (__cuda___isinff(b)) {
    if (a == -1.0f) {
      return 1.0f;
    }
    t = (__cuda_fabsf(a) > 1.0f) ? __int_as_float(0x7f800000) : 0.0f;
    if (b < 0.0f) {
      t = 1.0f / t;
    }
    return t;
  }
  bIsOddInteger = (b - (2.0f * floorf(0.5f * b))) == 1.0f;
  if (a == 0.0f) {
    t = bIsOddInteger ? a : 0.0f;
    if (b < 0.0f) {
      t = 1.0f / t;
    }
    return t;
  }
  if (a == -__int_as_float(0x7f800000)) {
    t = (b < 0.0f) ? -1.0f/a : -a;
    if (bIsOddInteger) {
      t = __int_as_float(__float_as_int(t) ^ 0x80000000);
    }
    return t;
  }
  if ((a < 0.0f) && (b != __cuda_truncf(b))) {
    return __int_as_float(0x7fffffff);
  }
  t = __cuda_fabsf(a);
  t = __internal_accurate_powf(t, b);
  if ((a < 0.0f) && bIsOddInteger) {
    t = __int_as_float(__float_as_int(t) ^ 0x80000000);
  }
  return t;

}


static float __internal_tgammaf_kernel(float a)
{
  float t;
  t = - 1.05767296987211380E-003f;
  t = t * a + 7.09279059435508670E-003f;
  t = t * a - 9.65347121958557050E-003f;
  t = t * a - 4.21736613253687960E-002f;
  t = t * a + 1.66542401247154280E-001f;
  t = t * a - 4.20043267827838460E-002f;
  t = t * a - 6.55878234051332940E-001f;
  t = t * a + 5.77215696929794240E-001f;
  t = t * a + 1.00000000000000000E+000f;
  return t;
}





static float __cuda_tgammaf(float a)
{
  float s, xx, x=a;
  if (x >= 0.0f) {
    if (x > 36.0f) x = 36.0f;
    s = 1.0f;
    xx = x;
    if (x > 34.03f) {
      xx -= 1.0f;
    }
    while (xx > 1.5f) {
      xx = xx - 1.0f;
      s = s * xx;
    }
    if (x >= 0.5f) {
      xx = xx - 1.0f;
    }
    xx = __internal_tgammaf_kernel(xx);
    if (x < 0.5f) {
      xx = xx * x;
    }
    s = s / xx;
    if (x > 34.03f) {

      xx = x - 1.0f;
      s = s * xx;
    }
    return s;
  } else {
    if (x == __cuda_floorf(x)) {
      x = __int_as_float(0x7fffffff);

      return x;

    }
    if (x < -41.1f) x = -41.1f;
    xx = x;
    if (x < -34.03f) {
      xx += 6.0f;
    }
    s = xx;
    while (xx < -0.5f) {
      xx = xx + 1.0f;
      s = s * xx;
    }
    xx = __internal_tgammaf_kernel(xx);
    s = s * xx;
    s = 1.0f / s;
    if (x < -34.03f) {
      xx = x;
      xx *= (x + 1.0f);
      xx *= (x + 2.0f);
      xx *= (x + 3.0f);
      xx *= (x + 4.0f);
      xx *= (x + 5.0f);
      xx = 1.0f / xx;
      s = s * xx;
      if ((a < -42.0f) && !(((int)a)&1)) {
        s = __int_as_float(0x80000000);
      }
    }
    return s;
  }
}

static float __cuda_roundf(float a)
{



  float fa = __cuda_fabsf(a);
  if (fa > 8388608.0f) {
    return a;
  } else {
    float u = __cuda_floorf(fa + 0.5f);
    if (fa < 0.5f) u = 0.0f;
    return __cuda_copysignf(u, a);
  }

}

static long long int __internal_llroundf_kernel(float a)
{
  unsigned long long int res, t = 0LL;
  int shift;
  unsigned int ia = __float_as_int(a);
  if ((ia << 1) > 0xff000000) return 0LL;
  if ((int)ia >= 0x5f000000) return 0x7fffffffffffffffLL;
  if (ia >= 0xdf000000) return 0x8000000000000000LL;
  shift = 189 - ((ia >> 23) & 0xff);
  res = ((long long int)(((ia << 8) | 0x80000000) >> 1)) << 32;
  if (shift >= 64) {
    t = res;
    res = 0;
  } else if (shift) {
    t = res << (64 - shift);
    res = res >> shift;
  }
  if (t >= 0x8000000000000000LL) {
      res++;
  }
  if ((int)ia < 0) res = (unsigned long long int)(-(long long int)res);
  return (long long int)res;
}

static long long int __cuda_llroundf(float a)
{



  return __internal_llroundf_kernel(a);

}

static long int __cuda_lroundf(float a)
{




  return (long int)__cuda_llroundf(a);
# 3192 "/usr/local/cuda/bin/../include/math_functions.h" 3
}

static float __cuda_fdimf(float a, float b)
{
  float t;
  t = a - b;
  if (a <= b) {
    t = 0.0f;
  }
  return t;
}

static int __cuda_ilogbf(float a)
{
  unsigned int i;
  int expo;
  a = __cuda_fabsf(a);
  if (a <= 1.175494351e-38f) {

    if (a == 0.0f) {
      expo = -((int)((unsigned int)-1 >> 1))-1;
    } else {
      expo = -126;
      i = __float_as_int(a);
      i = i << 8;
      while ((int)i >= 0) {
        expo--;
        i = i + i;
      }
    }
  } else {
    i = __float_as_int(a);
    expo = ((int)((i >> 23) & 0xff)) - 127;
    if ((i == 0x7f800000)) {
      expo = ((int)((unsigned int)-1 >> 1));
    }
    if ((i > 0x7f800000)) {
      expo = -((int)((unsigned int)-1 >> 1))-1;
    }
  }
  return expo;
}

static float __cuda_logbf(float a)
{



  unsigned int i;
  int expo;
  float res;

  if (__cuda___isnanf(a)) return a + a;

  a = __cuda_fabsf(a);
  if (a <= 1.175494351e-38f) {

    if (a == 0.0f) {
      res = -__int_as_float(0x7f800000);
    } else {
      expo = -126;
      i = __float_as_int(a);
      i = i << 8;
      while ((int)i >= 0) {
        expo--;
        i = i + i;
      }
      res = (float)expo;
    }
  } else {
    i = __float_as_int(a);
    expo = ((int)((i >> 23) & 0xff)) - 127;
    res = (float)expo;
    if ((i >= 0x7f800000)) {

      res = a + a;
    }
  }
  return res;

}

static void __cuda_sincosf(float a, float *sptr, float *cptr)
{





  float t, u, s, c;
  int quadrant;
  if (__cuda___isinff(a)) {
    *sptr = __int_as_float(0x7fffffff);
    *cptr = __int_as_float(0x7fffffff);
    return;
  }
  if (a == 0.0f) {
    *sptr = a;
    *cptr = 1.0f;
    return;
  }
  t = __internal_trig_reduction_kernel(a, &quadrant);
  u = __internal_cos_kernel(t);
  t = __internal_sin_kernel(t);
  if (quadrant & 1) {
    s = u;
    c = t;
  } else {
    s = t;
    c = u;
  }
  if (quadrant & 2) {
    s = -s;
  }
  quadrant++;
  if (quadrant & 2) {
    c = -c;
  }
  *sptr = s;
  *cptr = c;

}
# 3323 "/usr/local/cuda/bin/../include/math_functions.h" 3
extern __attribute__((__weak__)) double rsqrt(double a); double rsqrt(double a)
{
  return 1.0 / sqrt(a);
}

extern __attribute__((__weak__)) float rsqrtf(float a); float rsqrtf(float a)
{
  return (float)rsqrt((double)a);
}
# 3855 "/usr/local/cuda/bin/../include/math_functions.h" 3
# 1 "/usr/local/cuda/bin/../include/math_functions_dbl_ptx1.h" 1 3
# 45 "/usr/local/cuda/bin/../include/math_functions_dbl_ptx1.h" 3
static double __cuda_fabs(double a)
{
  return (float)__cuda_fabsf((float)a);
}

static double __cuda_fmax(double a, double b)
{
  return (float)__cuda_fmaxf((float)a, (float)b);
}

static double __cuda_fmin(double a, double b)
{
  return (float)__cuda_fminf((float)a, (float)b);
}

static int __cuda___finite(double a)
{
  return __cuda___finitef((float)a);
}

static int __cuda___isinf(double a)
{
  return __cuda___isinff((float)a);
}

static int __cuda___isnan(double a)
{
  return __cuda___isnanf((float)a);
}

static int __cuda___signbit(double a)
{
  return __cuda___signbitf((float)a);
}

static double __cuda_sqrt(double a)
{
  return (double)__cuda_sqrtf((float)a);
}

static double __cuda_rsqrt(double a)
{
  return (double)__cuda_rsqrtf((float)a);
}

static double __cuda_ceil(double a)
{
  return (double)__cuda_ceilf((float)a);
}

static double __cuda_trunc(double a)
{
  return (double)__cuda_truncf((float)a);
}

static double __cuda_floor(double a)
{
  return (double)__cuda_floorf((float)a);
}

static double __cuda_copysign(double a, double b)
{
  return (double)__cuda_copysignf((float)a, (float)b);
}

static double __cuda_sin(double a)
{
  return (double)__cuda_sinf((float)a);
}

static double __cuda_cos(double a)
{
  return (double)__cuda_cosf((float)a);
}

static void __cuda_sincos(double a, double *sptr, double *cptr)
{
  float fs, fc;

  __cuda_sincosf((float)a, &fs, &fc);

  *sptr = (double)fs;
  *cptr = (double)fc;
}

static double __cuda_tan(double a)
{
  return (double)__cuda_tanf((float)a);
}

static double __cuda_exp(double a)
{
  return (double)__cuda_expf((float)a);
}

static double __cuda_exp2(double a)
{
  return (double)__cuda_exp2f((float)a);
}

static double __cuda_exp10(double a)
{
  return (double)__cuda_exp10f((float)a);
}

static double __cuda_expm1(double a)
{
  return (double)__cuda_expm1f((float)a);
}

static double __cuda_cosh(double a)
{
  return (double)__cuda_coshf((float)a);
}

static double __cuda_sinh(double a)
{
  return (double)__cuda_sinhf((float)a);
}

static double __cuda_tanh(double a)
{
  return (double)__cuda_tanhf((float)a);
}

static double __cuda_asin(double a)
{
  return (double)__cuda_asinf((float)a);
}

static double __cuda_acos(double a)
{
  return (double)__cuda_acosf((float)a);
}

static double __cuda_atan(double a)
{
  return (double)__cuda_atanf((float)a);
}

static double __cuda_atan2(double a, double b)
{
  return (double)__cuda_atan2f((float)a, (float)b);
}

static double __cuda_log(double a)
{
  return (double)__cuda_logf((float)a);
}

static double __cuda_log2(double a)
{
  return (double)__cuda_log2f((float)a);
}

static double __cuda_log10(double a)
{
  return (double)__cuda_log10f((float)a);
}

static double __cuda_log1p(double a)
{
  return (double)__cuda_log1pf((float)a);
}

static double __cuda_acosh(double a)
{
  return (double)__cuda_acoshf((float)a);
}

static double __cuda_asinh(double a)
{
  return (double)__cuda_asinhf((float)a);
}

static double __cuda_atanh(double a)
{
  return (double)__cuda_atanhf((float)a);
}

static double __cuda_hypot(double a, double b)
{
  return (double)__cuda_hypotf((float)a, (float)b);
}

static double __cuda_cbrt(double a)
{
  return (double)__cuda_cbrtf((float)a);
}

static double __cuda_erf(double a)
{
  return (double)__cuda_erff((float)a);
}

static double __cuda_erfc(double a)
{
  return (double)__cuda_erfcf((float)a);
}

static double __cuda_lgamma(double a)
{
  return (double)__cuda_lgammaf((float)a);
}

static double __cuda_tgamma(double a)
{
  return (double)__cuda_tgammaf((float)a);
}

static double __cuda_ldexp(double a, int b)
{
  return (double)__cuda_ldexpf((float)a, b);
}

static double __cuda_scalbn(double a, int b)
{
  return (double)__cuda_scalbnf((float)a, b);
}

static double __cuda_scalbln(double a, long b)
{
  return (double)__cuda_scalblnf((float)a, b);
}

static double __cuda_frexp(double a, int *b)
{
  return (double)__cuda_frexpf((float)a, b);
}

static double __cuda_modf(double a, double *b)
{
  float fb;
  float fa = __cuda_modff((float)a, &fb);

  *b = (double)fb;

  return (double)fa;
}

static double __cuda_fmod(double a, double b)
{
  return (double)__cuda_fmodf((float)a, (float)b);
}

static double __cuda_remainder(double a, double b)
{
  return (double)__cuda_remainderf((float)a, (float)b);
}

static double __cuda_remquo(double a, double b, int *c)
{
  return (double)__cuda_remquof((float)a, (float)b, c);
}

static double __cuda_nextafter(double a, double b)
{
  return (double)__cuda_nextafterf((float)a, (float)b);
}

static double __cuda_nan(const char *tagp)
{
  return (double)__cuda_nanf(tagp);
}

static double __cuda_pow(double a, double b)
{
  return (double)__cuda_powf((float)a, (float)b);
}

static double __cuda_round(double a)
{
  return (double)__cuda_roundf((float)a);
}

static long __cuda_lround(double a)
{
  return __cuda_lroundf((float)a);
}

static long long __cuda_llround(double a)
{
  return __cuda_llroundf((float)a);
}

static double __cuda_rint(double a)
{
  return (double)__cuda_rintf((float)a);
}

static long __cuda_lrint(double a)
{
  return __cuda_lrintf((float)a);
}

static long long __cuda_llrint(double a)
{
  return __cuda_llrintf((float)a);
}

static double __cuda_nearbyint(double a)
{
  return (double)__cuda_nearbyintf((float)a);
}

static double __cuda_fdim(double a, double b)
{
  return (double)__cuda_fdimf((float)a, (float)b);
}

static int __cuda_ilogb(double a)
{
  return __cuda_ilogbf((float)a);
}

static double __cuda_logb(double a)
{
  return (double)__cuda_logbf((float)a);
}

static double __cuda_fma(double a, double b, double c)
{
  return (double)__cuda_fmaf((float)a, (float)b, (float)c);
}
# 3856 "/usr/local/cuda/bin/../include/math_functions.h" 2 3
# 94 "/usr/local/cuda/bin/../include/common_functions.h" 2
# 216 "/usr/local/cuda/bin/../include/crt/host_runtime.h" 2
# 6 "/tmp/tmpxft_00001cd2_00000000-1_mr3.rcu.cudafe1.stub.c" 2
struct __T20;
struct __T20 {float *__par0;int __par1;int *__par2;int __par3;float *__par4;float *__par5;float *__par6;float *__par7;float *__par8;float *__par9;int __par10;float __par11;int __par12;float *__par13;int __dummy_field;volatile char __dummy[4];};



static void __sti____cudaRegisterAll_42_tmpxft_00001cd2_00000000_4_mr3_rcu_cpp1_ii_MR3init(void) __attribute__((__constructor__));
void __device_stub_nacl_kernel(float *__par0, int __par1, int *__par2, int __par3, float *__par4, float *__par5, float *__par6, float *__par7, float *__par8, float *__par9, int __par10, float __par11, int __par12, float *__par13){auto struct __T20 *__T21;
char __[256]; *(char**)&__T21 = __;if (cudaSetupArgument((void*)(char*)&__par0, sizeof(__par0), (size_t)&__T21->__par0 - (size_t)__T21) != cudaSuccess) return;if (cudaSetupArgument((void*)(char*)&__par1, sizeof(__par1), (size_t)&__T21->__par1 - (size_t)__T21) != cudaSuccess) return;if (cudaSetupArgument((void*)(char*)&__par2, sizeof(__par2), (size_t)&__T21->__par2 - (size_t)__T21) != cudaSuccess) return;if (cudaSetupArgument((void*)(char*)&__par3, sizeof(__par3), (size_t)&__T21->__par3 - (size_t)__T21) != cudaSuccess) return;if (cudaSetupArgument((void*)(char*)&__par4, sizeof(__par4), (size_t)&__T21->__par4 - (size_t)__T21) != cudaSuccess) return;if (cudaSetupArgument((void*)(char*)&__par5, sizeof(__par5), (size_t)&__T21->__par5 - (size_t)__T21) != cudaSuccess) return;if (cudaSetupArgument((void*)(char*)&__par6, sizeof(__par6), (size_t)&__T21->__par6 - (size_t)__T21) != cudaSuccess) return;if (cudaSetupArgument((void*)(char*)&__par7, sizeof(__par7), (size_t)&__T21->__par7 - (size_t)__T21) != cudaSuccess) return;if (cudaSetupArgument((void*)(char*)&__par8, sizeof(__par8), (size_t)&__T21->__par8 - (size_t)__T21) != cudaSuccess) return;if (cudaSetupArgument((void*)(char*)&__par9, sizeof(__par9), (size_t)&__T21->__par9 - (size_t)__T21) != cudaSuccess) return;if (cudaSetupArgument((void*)(char*)&__par10, sizeof(__par10), (size_t)&__T21->__par10 - (size_t)__T21) != cudaSuccess) return;if (cudaSetupArgument((void*)(char*)&__par11, sizeof(__par11), (size_t)&__T21->__par11 - (size_t)__T21) != cudaSuccess) return;if (cudaSetupArgument((void*)(char*)&__par12, sizeof(__par12), (size_t)&__T21->__par12 - (size_t)__T21) != cudaSuccess) return;if (cudaSetupArgument((void*)(char*)&__par13, sizeof(__par13), (size_t)&__T21->__par13 - (size_t)__T21) != cudaSuccess) return;{ volatile static char *__f; __f = ((char *)__device_stub_nacl_kernel); (void)cudaLaunch(((char *)__device_stub_nacl_kernel)); };}



static void __sti____cudaRegisterAll_42_tmpxft_00001cd2_00000000_4_mr3_rcu_cpp1_ii_MR3init(void){__cudaFatCubinHandle = __cudaRegisterFatBinary((void*)(&__fatDeviceText)); atexit(__cudaUnregisterBinaryUtil);__cudaRegisterFunction(__cudaFatCubinHandle, (const char*)__device_stub_nacl_kernel, (char*)"nacl_kernel", "nacl_kernel", (-1), (uint3*)0, (uint3*)0, (dim3*)0, (dim3*)0, (int*)0);}

}
# 360 "./rcudatmp/mr3.rcu.cu" 2
