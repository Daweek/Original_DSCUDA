static char *Ptxdata = 
    "	.version 1.4\n"
    "	.target sm_10, map_f64_to_f32\n"
    "	// compiled with /opt/cuda/4.1/open64/lib//be\n"
    "	// nvopencc 4.1 built on 2012-01-12\n"
    "\n"
    "	//-----------------------------------------------------------\n"
    "	// Compiling /tmp/tmpxft_00001429_00000000-9_userapp.cpp3.i (/tmp/ccBI#.Mu4GYr)\n"
    "	//-----------------------------------------------------------\n"
    "\n"
    "	//-----------------------------------------------------------\n"
    "	// Options:\n"
    "	//-----------------------------------------------------------\n"
    "	//  Target:ptx, ISA:sm_10, Endian:little, Pointer Size:64\n"
    "	//  -O3	(Optimization level)\n"
    "	//  -g0	(Debug level)\n"
    "	//  -m2	(Report advisories)\n"
    "	//-----------------------------------------------------------\n"
    "\n"
    "	.file	1	\"<command-line>\"\n"
    "	.file	2	\"/tmp/tmpxft_00001429_00000000-8_userapp.cudafe2.gpu\"\n"
    "	.file	3	\"/usr/lib64/gcc/x86_64-suse-linux/4.3/include/stddef.h\"\n"
    "	.file	4	\"/opt/cuda/4.1/bin/../include/crt/device_runtime.h\"\n"
    "	.file	5	\"/opt/cuda/4.1/bin/../include/host_defines.h\"\n"
    "	.file	6	\"/opt/cuda/4.1/bin/../include/builtin_types.h\"\n"
    "	.file	7	\"/opt/cuda/4.1/bin/../include/device_types.h\"\n"
    "	.file	8	\"/opt/cuda/4.1/bin/../include/driver_types.h\"\n"
    "	.file	9	\"/opt/cuda/4.1/bin/../include/surface_types.h\"\n"
    "	.file	10	\"/opt/cuda/4.1/bin/../include/texture_types.h\"\n"
    "	.file	11	\"/opt/cuda/4.1/bin/../include/vector_types.h\"\n"
    "	.file	12	\"/opt/cuda/4.1/bin/../include/device_launch_parameters.h\"\n"
    "	.file	13	\"/opt/cuda/4.1/bin/../include/crt/storage_class.h\"\n"
    "	.file	14	\"userapp.cuh\"\n"
    "	.file	15	\"/opt/cuda/4.1/bin/../include/common_functions.h\"\n"
    "	.file	16	\"/opt/cuda/4.1/bin/../include/math_functions.h\"\n"
    "	.file	17	\"/opt/cuda/4.1/bin/../include/math_constants.h\"\n"
    "	.file	18	\"/opt/cuda/4.1/bin/../include/device_functions.h\"\n"
    "	.file	19	\"/opt/cuda/4.1/bin/../include/sm_11_atomic_functions.h\"\n"
    "	.file	20	\"/opt/cuda/4.1/bin/../include/sm_12_atomic_functions.h\"\n"
    "	.file	21	\"/opt/cuda/4.1/bin/../include/sm_13_double_functions.h\"\n"
    "	.file	22	\"/opt/cuda/4.1/bin/../include/sm_20_atomic_functions.h\"\n"
    "	.file	23	\"/opt/cuda/4.1/bin/../include/sm_20_intrinsics.h\"\n"
    "	.file	24	\"/opt/cuda/4.1/bin/../include/surface_functions.h\"\n"
    "	.file	25	\"/opt/cuda/4.1/bin/../include/texture_fetch_functions.h\"\n"
    "	.file	26	\"/opt/cuda/4.1/bin/../include/math_functions_dbl_ptx1.h\"\n"
    "\n"
    "\n"
    "	.entry _Z6vecAddPfS_S_ (\n"
    "		.param .u64 __cudaparm__Z6vecAddPfS_S__a,\n"
    "		.param .u64 __cudaparm__Z6vecAddPfS_S__b,\n"
    "		.param .u64 __cudaparm__Z6vecAddPfS_S__c)\n"
    "	{\n"
    "	.reg .u16 %rh<4>;\n"
    "	.reg .u32 %r<5>;\n"
    "	.reg .u64 %rd<10>;\n"
    "	.reg .f32 %f<5>;\n"
    "	.loc	14	2	0\n"
    "$LDWbegin__Z6vecAddPfS_S_:\n"
    "	.loc	14	5	0\n"
    "	cvt.u32.u16 	%r1, %tid.x;\n"
    "	mov.u16 	%rh1, %ctaid.x;\n"
    "	mov.u16 	%rh2, %ntid.x;\n"
    "	mul.wide.u16 	%r2, %rh1, %rh2;\n"
    "	add.u32 	%r3, %r1, %r2;\n"
    "	cvt.s64.s32 	%rd1, %r3;\n"
    "	mul.wide.s32 	%rd2, %r3, 4;\n"
    "	ld.param.u64 	%rd3, [__cudaparm__Z6vecAddPfS_S__a];\n"
    "	add.u64 	%rd4, %rd3, %rd2;\n"
    "	ld.global.f32 	%f1, [%rd4+0];\n"
    "	ld.param.u64 	%rd5, [__cudaparm__Z6vecAddPfS_S__b];\n"
    "	add.u64 	%rd6, %rd5, %rd2;\n"
    "	ld.global.f32 	%f2, [%rd6+0];\n"
    "	add.f32 	%f3, %f1, %f2;\n"
    "	ld.param.u64 	%rd7, [__cudaparm__Z6vecAddPfS_S__c];\n"
    "	add.u64 	%rd8, %rd7, %rd2;\n"
    "	st.global.f32 	[%rd8+0], %f3;\n"
    "	.loc	14	6	0\n"
    "	exit;\n"
    "$LDWend__Z6vecAddPfS_S_:\n"
    "	} // _Z6vecAddPfS_S_\n"
    "\n"
    "	.entry _Z6vecMulPfS_fS_iPi (\n"
    "		.param .u64 __cudaparm__Z6vecMulPfS_fS_iPi_a,\n"
    "		.param .u64 __cudaparm__Z6vecMulPfS_fS_iPi_b,\n"
    "		.param .f32 __cudaparm__Z6vecMulPfS_fS_iPi_c,\n"
    "		.param .u64 __cudaparm__Z6vecMulPfS_fS_iPi_d,\n"
    "		.param .s32 __cudaparm__Z6vecMulPfS_fS_iPi_e,\n"
    "		.param .u64 __cudaparm__Z6vecMulPfS_fS_iPi_f)\n"
    "	{\n"
    "	.reg .u32 %r<5>;\n"
    "	.reg .u64 %rd<12>;\n"
    "	.reg .f32 %f<10>;\n"
    "	.loc	14	9	0\n"
    "$LDWbegin__Z6vecMulPfS_fS_iPi:\n"
    "	.loc	14	12	0\n"
    "	cvt.s32.u16 	%r1, %tid.x;\n"
    "	cvt.s64.s32 	%rd1, %r1;\n"
    "	mul.wide.s32 	%rd2, %r1, 4;\n"
    "	ld.param.u64 	%rd3, [__cudaparm__Z6vecMulPfS_fS_iPi_f];\n"
    "	add.u64 	%rd4, %rd3, %rd2;\n"
    "	ld.global.s32 	%r2, [%rd4+0];\n"
    "	cvt.rn.f32.s32 	%f1, %r2;\n"
    "	ld.param.s32 	%r3, [__cudaparm__Z6vecMulPfS_fS_iPi_e];\n"
    "	cvt.rn.f32.s32 	%f2, %r3;\n"
    "	ld.param.f32 	%f3, [__cudaparm__Z6vecMulPfS_fS_iPi_c];\n"
    "	ld.param.u64 	%rd5, [__cudaparm__Z6vecMulPfS_fS_iPi_a];\n"
    "	add.u64 	%rd6, %rd5, %rd2;\n"
    "	ld.global.f32 	%f4, [%rd6+0];\n"
    "	ld.param.u64 	%rd7, [__cudaparm__Z6vecMulPfS_fS_iPi_b];\n"
    "	add.u64 	%rd8, %rd7, %rd2;\n"
    "	ld.global.f32 	%f5, [%rd8+0];\n"
    "	mad.f32 	%f6, %f4, %f5, %f3;\n"
    "	add.f32 	%f7, %f2, %f6;\n"
    "	add.f32 	%f8, %f1, %f7;\n"
    "	ld.param.u64 	%rd9, [__cudaparm__Z6vecMulPfS_fS_iPi_d];\n"
    "	add.u64 	%rd10, %rd9, %rd2;\n"
    "	st.global.f32 	[%rd10+0], %f8;\n"
    "	.loc	14	13	0\n"
    "	exit;\n"
    "$LDWend__Z6vecMulPfS_fS_iPi:\n"
    "	} // _Z6vecMulPfS_fS_iPi\n"
    "\n";
#pragma dscuda endofptx
#pragma begin dscuda.h
#ifndef _DSCUDA_H
#define _DSCUDA_H

#include <cuda_runtime_api.h>
#include <cutil.h>
#include <builtin_types.h>
#include <driver_types.h>
#include <cuda_texture_types.h>
#include <texture_types.h>
#pragma begin dscudarpc.h


#ifndef _DSCUDARPC_H_RPCGEN
#define _DSCUDARPC_H_RPCGEN

#include <rpc/rpc.h>


#ifdef __cplusplus
extern "C" {
#endif


typedef u_quad_t RCadr;

typedef u_quad_t RCstream;

typedef u_quad_t RCevent;

typedef u_quad_t RCipaddr;

typedef u_int RCsize;

typedef u_int RCerror;

typedef struct {
	u_int RCbuf_len;
	char *RCbuf_val;
} RCbuf;

typedef u_int RCchannelformat;

typedef u_long RCpid;

struct RCchanneldesc_t {
	RCchannelformat f;
	int w;
	int x;
	int y;
	int z;
};
typedef struct RCchanneldesc_t RCchanneldesc_t;

typedef RCchanneldesc_t RCchanneldesc;

struct RCtexture_t {
	int normalized;
	int filterMode;
	int addressMode[3];
	RCchannelformat f;
	int w;
	int x;
	int y;
	int z;
};
typedef struct RCtexture_t RCtexture_t;

typedef RCtexture_t RCtexture;

struct RCfuncattr_t {
	int binaryVersion;
	RCsize constSizeBytes;
	RCsize localSizeBytes;
	int maxThreadsPerBlock;
	int numRegs;
	int ptxVersion;
	RCsize sharedSizeBytes;
};
typedef struct RCfuncattr_t RCfuncattr_t;

typedef RCfuncattr_t RCfuncattr;

enum RCargType {
	dscudaArgTypeP = 0,
	dscudaArgTypeI = 1,
	dscudaArgTypeF = 2,
	dscudaArgTypeV = 3,
};
typedef enum RCargType RCargType;

struct RCargVal {
	RCargType type;
	union {
		RCadr address;
		u_int valuei;
		float valuef;
		char valuev[64];
	} RCargVal_u;
};
typedef struct RCargVal RCargVal;

struct RCarg {
	RCargVal val;
	u_int offset;
	u_int size;
};
typedef struct RCarg RCarg;

typedef struct {
	u_int RCargs_len;
	RCarg *RCargs_val;
} RCargs;

struct dscudaResult {
	RCerror err;
};
typedef struct dscudaResult dscudaResult;

struct dscudaThreadGetLimitResult {
	RCerror err;
	RCsize value;
};
typedef struct dscudaThreadGetLimitResult dscudaThreadGetLimitResult;

struct dscudaThreadGetCacheConfigResult {
	RCerror err;
	int cacheConfig;
};
typedef struct dscudaThreadGetCacheConfigResult dscudaThreadGetCacheConfigResult;

struct dscudaMallocResult {
	RCerror err;
	RCadr devAdr;
};
typedef struct dscudaMallocResult dscudaMallocResult;

struct dscudaHostAllocResult {
	RCerror err;
	RCadr pHost;
};
typedef struct dscudaHostAllocResult dscudaHostAllocResult;

struct dscudaMallocHostResult {
	RCerror err;
	RCadr ptr;
};
typedef struct dscudaMallocHostResult dscudaMallocHostResult;

struct dscudaMallocArrayResult {
	RCerror err;
	RCadr array;
};
typedef struct dscudaMallocArrayResult dscudaMallocArrayResult;

struct dscudaMallocPitchResult {
	RCerror err;
	RCadr devPtr;
	RCsize pitch;
};
typedef struct dscudaMallocPitchResult dscudaMallocPitchResult;

struct dscudaMemcpyD2HResult {
	RCerror err;
	RCbuf buf;
};
typedef struct dscudaMemcpyD2HResult dscudaMemcpyD2HResult;

struct dscudaMemcpyH2HResult {
	RCerror err;
	RCbuf buf;
};
typedef struct dscudaMemcpyH2HResult dscudaMemcpyH2HResult;

struct dscudaMemcpyToArrayD2HResult {
	RCerror err;
	RCbuf buf;
};
typedef struct dscudaMemcpyToArrayD2HResult dscudaMemcpyToArrayD2HResult;

struct dscudaMemcpyToArrayH2HResult {
	RCerror err;
	RCbuf buf;
};
typedef struct dscudaMemcpyToArrayH2HResult dscudaMemcpyToArrayH2HResult;

struct dscudaMemcpy2DToArrayD2HResult {
	RCerror err;
	RCbuf buf;
};
typedef struct dscudaMemcpy2DToArrayD2HResult dscudaMemcpy2DToArrayD2HResult;

struct dscudaMemcpy2DToArrayH2HResult {
	RCerror err;
	RCbuf buf;
};
typedef struct dscudaMemcpy2DToArrayH2HResult dscudaMemcpy2DToArrayH2HResult;

struct dscudaMemcpy2DD2HResult {
	RCerror err;
	RCbuf buf;
};
typedef struct dscudaMemcpy2DD2HResult dscudaMemcpy2DD2HResult;

struct dscudaMemcpy2DH2HResult {
	RCerror err;
	RCbuf buf;
};
typedef struct dscudaMemcpy2DH2HResult dscudaMemcpy2DH2HResult;

struct dscudaGetDeviceResult {
	RCerror err;
	int device;
};
typedef struct dscudaGetDeviceResult dscudaGetDeviceResult;

struct dscudaGetDeviceCountResult {
	RCerror err;
	int count;
};
typedef struct dscudaGetDeviceCountResult dscudaGetDeviceCountResult;

struct dscudaGetDevicePropertiesResult {
	RCerror err;
	RCbuf prop;
};
typedef struct dscudaGetDevicePropertiesResult dscudaGetDevicePropertiesResult;

struct dscudaDriverGetVersionResult {
	RCerror err;
	int ver;
};
typedef struct dscudaDriverGetVersionResult dscudaDriverGetVersionResult;

struct dscudaRuntimeGetVersionResult {
	RCerror err;
	int ver;
};
typedef struct dscudaRuntimeGetVersionResult dscudaRuntimeGetVersionResult;

struct dscudaGetErrorStringResult {
	char *errmsg;
};
typedef struct dscudaGetErrorStringResult dscudaGetErrorStringResult;

struct dscudaCreateChannelDescResult {
	int x;
	int y;
	int z;
	int w;
	RCchannelformat f;
};
typedef struct dscudaCreateChannelDescResult dscudaCreateChannelDescResult;

struct dscudaGetChannelDescResult {
	RCerror err;
	int x;
	int y;
	int z;
	int w;
	RCchannelformat f;
};
typedef struct dscudaGetChannelDescResult dscudaGetChannelDescResult;

struct dscudaChooseDeviceResult {
	RCerror err;
	int device;
};
typedef struct dscudaChooseDeviceResult dscudaChooseDeviceResult;

struct dscudaMemcpyAsyncD2HResult {
	RCerror err;
	RCbuf buf;
};
typedef struct dscudaMemcpyAsyncD2HResult dscudaMemcpyAsyncD2HResult;

struct dscudaMemcpyAsyncH2HResult {
	RCerror err;
	RCbuf buf;
};
typedef struct dscudaMemcpyAsyncH2HResult dscudaMemcpyAsyncH2HResult;

struct dscudaMemcpyFromSymbolD2HResult {
	RCerror err;
	RCbuf buf;
};
typedef struct dscudaMemcpyFromSymbolD2HResult dscudaMemcpyFromSymbolD2HResult;

struct dscudaMemcpyFromSymbolAsyncD2HResult {
	RCerror err;
	RCbuf buf;
};
typedef struct dscudaMemcpyFromSymbolAsyncD2HResult dscudaMemcpyFromSymbolAsyncD2HResult;

struct dscudaStreamCreateResult {
	RCerror err;
	RCadr stream;
};
typedef struct dscudaStreamCreateResult dscudaStreamCreateResult;

struct dscudaEventCreateResult {
	RCerror err;
	RCadr event;
};
typedef struct dscudaEventCreateResult dscudaEventCreateResult;

struct dscudaEventElapsedTimeResult {
	RCerror err;
	float ms;
};
typedef struct dscudaEventElapsedTimeResult dscudaEventElapsedTimeResult;

struct dscudaHostGetDevicePointerResult {
	RCerror err;
	RCadr pDevice;
};
typedef struct dscudaHostGetDevicePointerResult dscudaHostGetDevicePointerResult;

struct dscudaHostGetFlagsResult {
	RCerror err;
	u_int flags;
};
typedef struct dscudaHostGetFlagsResult dscudaHostGetFlagsResult;

struct dscudaLoadModuleResult {
	u_int id;
};
typedef struct dscudaLoadModuleResult dscudaLoadModuleResult;

struct dscudaFuncGetAttributesResult {
	RCerror err;
	RCfuncattr attr;
};
typedef struct dscudaFuncGetAttributesResult dscudaFuncGetAttributesResult;

struct dscudaBindTextureResult {
	RCerror err;
	RCsize offset;
};
typedef struct dscudaBindTextureResult dscudaBindTextureResult;

struct dscudaBindTexture2DResult {
	RCerror err;
	RCsize offset;
};
typedef struct dscudaBindTexture2DResult dscudaBindTexture2DResult;

struct dscufftResult {
	RCerror err;
};
typedef struct dscufftResult dscufftResult;

struct dscufftPlanResult {
	RCerror err;
	u_int plan;
};
typedef struct dscufftPlanResult dscufftPlanResult;

struct dscublasResult {
	RCerror err;
	u_int stat;
};
typedef struct dscublasResult dscublasResult;

struct dscublasCreateResult {
	RCerror err;
	u_int stat;
	RCadr handle;
};
typedef struct dscublasCreateResult dscublasCreateResult;

struct dscublasGetVectorResult {
	RCerror err;
	u_int stat;
	RCbuf y;
};
typedef struct dscublasGetVectorResult dscublasGetVectorResult;

struct RCdim3 {
	u_int x;
	u_int y;
	u_int z;
};
typedef struct RCdim3 RCdim3;

struct dscudathreadsetlimitid_1_argument {
	int limit;
	RCsize value;
};
typedef struct dscudathreadsetlimitid_1_argument dscudathreadsetlimitid_1_argument;

struct dscudastreamwaiteventid_1_argument {
	RCstream stream;
	RCevent event;
	u_int flags;
};
typedef struct dscudastreamwaiteventid_1_argument dscudastreamwaiteventid_1_argument;

struct dscudaeventelapsedtimeid_1_argument {
	RCevent start;
	RCevent end;
};
typedef struct dscudaeventelapsedtimeid_1_argument dscudaeventelapsedtimeid_1_argument;

struct dscudaeventrecordid_1_argument {
	RCevent event;
	RCstream stream;
};
typedef struct dscudaeventrecordid_1_argument dscudaeventrecordid_1_argument;

struct dscudalaunchkernelid_1_argument {
	int moduleid;
	int kid;
	char *kname;
	RCdim3 gdim;
	RCdim3 bdim;
	RCsize smemsize;
	RCstream stream;
	RCargs args;
};
typedef struct dscudalaunchkernelid_1_argument dscudalaunchkernelid_1_argument;

struct dscudaloadmoduleid_1_argument {
	RCipaddr ipaddr;
	RCpid pid;
	char *mname;
	char *image;
};
typedef struct dscudaloadmoduleid_1_argument dscudaloadmoduleid_1_argument;

struct dscudafuncgetattributesid_1_argument {
	int moduleid;
	char *kname;
};
typedef struct dscudafuncgetattributesid_1_argument dscudafuncgetattributesid_1_argument;

struct dscudamemcpyh2hid_1_argument {
	RCadr dst;
	RCbuf src;
	RCsize count;
};
typedef struct dscudamemcpyh2hid_1_argument dscudamemcpyh2hid_1_argument;

struct dscudamemcpyh2did_1_argument {
	RCadr dst;
	RCbuf src;
	RCsize count;
};
typedef struct dscudamemcpyh2did_1_argument dscudamemcpyh2did_1_argument;

struct dscudamemcpyd2hid_1_argument {
	RCadr src;
	RCsize count;
};
typedef struct dscudamemcpyd2hid_1_argument dscudamemcpyd2hid_1_argument;

struct dscudamemcpyd2did_1_argument {
	RCadr dst;
	RCadr src;
	RCsize count;
};
typedef struct dscudamemcpyd2did_1_argument dscudamemcpyd2did_1_argument;

struct dscudamemcpyasynch2hid_1_argument {
	RCadr dst;
	RCbuf src;
	RCsize count;
	RCstream stream;
};
typedef struct dscudamemcpyasynch2hid_1_argument dscudamemcpyasynch2hid_1_argument;

struct dscudamemcpyasynch2did_1_argument {
	RCadr dst;
	RCbuf src;
	RCsize count;
	RCstream stream;
};
typedef struct dscudamemcpyasynch2did_1_argument dscudamemcpyasynch2did_1_argument;

struct dscudamemcpyasyncd2hid_1_argument {
	RCadr src;
	RCsize count;
	RCstream stream;
};
typedef struct dscudamemcpyasyncd2hid_1_argument dscudamemcpyasyncd2hid_1_argument;

struct dscudamemcpyasyncd2did_1_argument {
	RCadr dst;
	RCadr src;
	RCsize count;
	RCstream stream;
};
typedef struct dscudamemcpyasyncd2did_1_argument dscudamemcpyasyncd2did_1_argument;

struct dscudamemcpytosymbolh2did_1_argument {
	int moduleid;
	char *symbol;
	RCbuf src;
	RCsize count;
	RCsize offset;
};
typedef struct dscudamemcpytosymbolh2did_1_argument dscudamemcpytosymbolh2did_1_argument;

struct dscudamemcpytosymbold2did_1_argument {
	int moduleid;
	char *symbol;
	RCadr src;
	RCsize count;
	RCsize offset;
};
typedef struct dscudamemcpytosymbold2did_1_argument dscudamemcpytosymbold2did_1_argument;

struct dscudamemcpyfromsymbold2hid_1_argument {
	int moduleid;
	char *symbol;
	RCsize count;
	RCsize offset;
};
typedef struct dscudamemcpyfromsymbold2hid_1_argument dscudamemcpyfromsymbold2hid_1_argument;

struct dscudamemcpyfromsymbold2did_1_argument {
	int moduleid;
	RCadr dst;
	char *symbol;
	RCsize count;
	RCsize offset;
};
typedef struct dscudamemcpyfromsymbold2did_1_argument dscudamemcpyfromsymbold2did_1_argument;

struct dscudamemsetid_1_argument {
	RCadr dst;
	int value;
	RCsize count;
};
typedef struct dscudamemsetid_1_argument dscudamemsetid_1_argument;

struct dscudahostallocid_1_argument {
	RCsize size;
	u_int flags;
};
typedef struct dscudahostallocid_1_argument dscudahostallocid_1_argument;

struct dscudahostgetdevicepointerid_1_argument {
	RCadr pHost;
	u_int flags;
};
typedef struct dscudahostgetdevicepointerid_1_argument dscudahostgetdevicepointerid_1_argument;

struct dscudamallocarrayid_1_argument {
	RCchanneldesc desc;
	RCsize width;
	RCsize height;
	u_int flags;
};
typedef struct dscudamallocarrayid_1_argument dscudamallocarrayid_1_argument;

struct dscudamemcpytoarrayh2hid_1_argument {
	RCadr dst;
	RCsize wOffset;
	RCsize hOffset;
	RCbuf src;
	RCsize count;
};
typedef struct dscudamemcpytoarrayh2hid_1_argument dscudamemcpytoarrayh2hid_1_argument;

struct dscudamemcpytoarrayh2did_1_argument {
	RCadr dst;
	RCsize wOffset;
	RCsize hOffset;
	RCbuf src;
	RCsize count;
};
typedef struct dscudamemcpytoarrayh2did_1_argument dscudamemcpytoarrayh2did_1_argument;

struct dscudamemcpytoarrayd2hid_1_argument {
	RCsize wOffset;
	RCsize hOffset;
	RCadr src;
	RCsize count;
};
typedef struct dscudamemcpytoarrayd2hid_1_argument dscudamemcpytoarrayd2hid_1_argument;

struct dscudamemcpytoarrayd2did_1_argument {
	RCadr dst;
	RCsize wOffset;
	RCsize hOffset;
	RCadr src;
	RCsize count;
};
typedef struct dscudamemcpytoarrayd2did_1_argument dscudamemcpytoarrayd2did_1_argument;

struct dscudamallocpitchid_1_argument {
	RCsize width;
	RCsize height;
};
typedef struct dscudamallocpitchid_1_argument dscudamallocpitchid_1_argument;

struct dscudamemcpy2dtoarrayh2hid_1_argument {
	RCadr dst;
	RCsize wOffset;
	RCsize hOffset;
	RCbuf src;
	RCsize spitch;
	RCsize width;
	RCsize height;
};
typedef struct dscudamemcpy2dtoarrayh2hid_1_argument dscudamemcpy2dtoarrayh2hid_1_argument;

struct dscudamemcpy2dtoarrayh2did_1_argument {
	RCadr dst;
	RCsize wOffset;
	RCsize hOffset;
	RCbuf srcbuf;
	RCsize spitch;
	RCsize width;
	RCsize height;
};
typedef struct dscudamemcpy2dtoarrayh2did_1_argument dscudamemcpy2dtoarrayh2did_1_argument;

struct dscudamemcpy2dtoarrayd2hid_1_argument {
	RCsize wOffset;
	RCsize hOffset;
	RCadr src;
	RCsize spitch;
	RCsize width;
	RCsize height;
};
typedef struct dscudamemcpy2dtoarrayd2hid_1_argument dscudamemcpy2dtoarrayd2hid_1_argument;

struct dscudamemcpy2dtoarrayd2did_1_argument {
	RCadr dst;
	RCsize wOffset;
	RCsize hOffset;
	RCadr src;
	RCsize spitch;
	RCsize width;
	RCsize height;
};
typedef struct dscudamemcpy2dtoarrayd2did_1_argument dscudamemcpy2dtoarrayd2did_1_argument;

struct dscudamemcpy2dh2hid_1_argument {
	RCadr dst;
	RCsize dpitch;
	RCbuf src;
	RCsize spitch;
	RCsize width;
	RCsize height;
};
typedef struct dscudamemcpy2dh2hid_1_argument dscudamemcpy2dh2hid_1_argument;

struct dscudamemcpy2dh2did_1_argument {
	RCadr dst;
	RCsize dpitch;
	RCbuf src;
	RCsize spitch;
	RCsize width;
	RCsize height;
};
typedef struct dscudamemcpy2dh2did_1_argument dscudamemcpy2dh2did_1_argument;

struct dscudamemcpy2dd2hid_1_argument {
	RCsize dpitch;
	RCadr src;
	RCsize spitch;
	RCsize width;
	RCsize height;
};
typedef struct dscudamemcpy2dd2hid_1_argument dscudamemcpy2dd2hid_1_argument;

struct dscudamemcpy2dd2did_1_argument {
	RCadr dst;
	RCsize dpitch;
	RCadr src;
	RCsize spitch;
	RCsize width;
	RCsize height;
};
typedef struct dscudamemcpy2dd2did_1_argument dscudamemcpy2dd2did_1_argument;

struct dscudamemset2did_1_argument {
	RCadr dst;
	RCsize pitch;
	int value;
	RCsize width;
	RCsize height;
};
typedef struct dscudamemset2did_1_argument dscudamemset2did_1_argument;

struct dscudamemcpytosymbolasynch2did_1_argument {
	int moduleid;
	char *symbol;
	RCbuf src;
	RCsize count;
	RCsize offset;
	RCstream stream;
};
typedef struct dscudamemcpytosymbolasynch2did_1_argument dscudamemcpytosymbolasynch2did_1_argument;

struct dscudamemcpytosymbolasyncd2did_1_argument {
	int moduleid;
	char *symbol;
	RCadr src;
	RCsize count;
	RCsize offset;
	RCstream stream;
};
typedef struct dscudamemcpytosymbolasyncd2did_1_argument dscudamemcpytosymbolasyncd2did_1_argument;

struct dscudamemcpyfromsymbolasyncd2hid_1_argument {
	int moduleid;
	char *symbol;
	RCsize count;
	RCsize offset;
	RCstream stream;
};
typedef struct dscudamemcpyfromsymbolasyncd2hid_1_argument dscudamemcpyfromsymbolasyncd2hid_1_argument;

struct dscudamemcpyfromsymbolasyncd2did_1_argument {
	int moduleid;
	RCadr dst;
	char *symbol;
	RCsize count;
	RCsize offset;
	RCstream stream;
};
typedef struct dscudamemcpyfromsymbolasyncd2did_1_argument dscudamemcpyfromsymbolasyncd2did_1_argument;

struct dscudacreatechanneldescid_1_argument {
	int x;
	int y;
	int z;
	int w;
	RCchannelformat f;
};
typedef struct dscudacreatechanneldescid_1_argument dscudacreatechanneldescid_1_argument;

struct dscudabindtextureid_1_argument {
	int moduleid;
	char *texname;
	RCadr devPtr;
	RCsize size;
	RCtexture texbuf;
};
typedef struct dscudabindtextureid_1_argument dscudabindtextureid_1_argument;

struct dscudabindtexture2did_1_argument {
	int moduleid;
	char *texname;
	RCadr devPtr;
	RCsize width;
	RCsize height;
	RCsize pitch;
	RCtexture texbuf;
};
typedef struct dscudabindtexture2did_1_argument dscudabindtexture2did_1_argument;

struct dscudabindtexturetoarrayid_1_argument {
	int moduleid;
	char *texname;
	RCadr array;
	RCtexture texbuf;
};
typedef struct dscudabindtexturetoarrayid_1_argument dscudabindtexturetoarrayid_1_argument;

struct dscufftplan3did_1_argument {
	int nx;
	int ny;
	int nz;
	u_int type;
};
typedef struct dscufftplan3did_1_argument dscufftplan3did_1_argument;

struct dscufftexecc2cid_1_argument {
	u_int plan;
	RCadr idata;
	RCadr odata;
	int direction;
};
typedef struct dscufftexecc2cid_1_argument dscufftexecc2cid_1_argument;

#define DSCUDA_PROG 60000
#define DSCUDA_VER 1

#if defined(__STDC__) || defined(__cplusplus)
#define dscudaThreadExitId 100
extern  dscudaResult * dscudathreadexitid_1(CLIENT *);
extern  dscudaResult * dscudathreadexitid_1_svc(struct svc_req *);
#define dscudaThreadSynchronizeId 101
extern  dscudaResult * dscudathreadsynchronizeid_1(CLIENT *);
extern  dscudaResult * dscudathreadsynchronizeid_1_svc(struct svc_req *);
#define dscudaThreadSetLimitId 102
extern  dscudaResult * dscudathreadsetlimitid_1(int , RCsize , CLIENT *);
extern  dscudaResult * dscudathreadsetlimitid_1_svc(int , RCsize , struct svc_req *);
#define dscudaThreadGetLimitId 103
extern  dscudaThreadGetLimitResult * dscudathreadgetlimitid_1(int , CLIENT *);
extern  dscudaThreadGetLimitResult * dscudathreadgetlimitid_1_svc(int , struct svc_req *);
#define dscudaThreadSetCacheConfigId 104
extern  dscudaResult * dscudathreadsetcacheconfigid_1(int , CLIENT *);
extern  dscudaResult * dscudathreadsetcacheconfigid_1_svc(int , struct svc_req *);
#define dscudaThreadGetCacheConfigId 105
extern  dscudaThreadGetCacheConfigResult * dscudathreadgetcacheconfigid_1(CLIENT *);
extern  dscudaThreadGetCacheConfigResult * dscudathreadgetcacheconfigid_1_svc(struct svc_req *);
#define dscudaGetLastErrorId 200
extern  dscudaResult * dscudagetlasterrorid_1(CLIENT *);
extern  dscudaResult * dscudagetlasterrorid_1_svc(struct svc_req *);
#define dscudaPeekAtLastErrorId 201
extern  dscudaResult * dscudapeekatlasterrorid_1(CLIENT *);
extern  dscudaResult * dscudapeekatlasterrorid_1_svc(struct svc_req *);
#define dscudaGetErrorStringId 202
extern  dscudaGetErrorStringResult * dscudageterrorstringid_1(int , CLIENT *);
extern  dscudaGetErrorStringResult * dscudageterrorstringid_1_svc(int , struct svc_req *);
#define dscudaGetDeviceId 300
extern  dscudaGetDeviceResult * dscudagetdeviceid_1(CLIENT *);
extern  dscudaGetDeviceResult * dscudagetdeviceid_1_svc(struct svc_req *);
#define dscudaGetDeviceCountId 301
extern  dscudaGetDeviceCountResult * dscudagetdevicecountid_1(CLIENT *);
extern  dscudaGetDeviceCountResult * dscudagetdevicecountid_1_svc(struct svc_req *);
#define dscudaGetDevicePropertiesId 302
extern  dscudaGetDevicePropertiesResult * dscudagetdevicepropertiesid_1(int , CLIENT *);
extern  dscudaGetDevicePropertiesResult * dscudagetdevicepropertiesid_1_svc(int , struct svc_req *);
#define dscudaDriverGetVersionId 303
extern  dscudaDriverGetVersionResult * dscudadrivergetversionid_1(CLIENT *);
extern  dscudaDriverGetVersionResult * dscudadrivergetversionid_1_svc(struct svc_req *);
#define dscudaRuntimeGetVersionId 304
extern  dscudaRuntimeGetVersionResult * dscudaruntimegetversionid_1(CLIENT *);
extern  dscudaRuntimeGetVersionResult * dscudaruntimegetversionid_1_svc(struct svc_req *);
#define dscudaSetDeviceId 305
extern  dscudaResult * dscudasetdeviceid_1(int , CLIENT *);
extern  dscudaResult * dscudasetdeviceid_1_svc(int , struct svc_req *);
#define dscudaSetDeviceFlagsId 306
extern  dscudaResult * dscudasetdeviceflagsid_1(u_int , CLIENT *);
extern  dscudaResult * dscudasetdeviceflagsid_1_svc(u_int , struct svc_req *);
#define dscudaChooseDeviceId 307
extern  dscudaChooseDeviceResult * dscudachoosedeviceid_1(RCbuf , CLIENT *);
extern  dscudaChooseDeviceResult * dscudachoosedeviceid_1_svc(RCbuf , struct svc_req *);
#define dscudaDeviceSynchronize 308
extern  dscudaResult * dscudadevicesynchronize_1(CLIENT *);
extern  dscudaResult * dscudadevicesynchronize_1_svc(struct svc_req *);
#define dscudaDeviceReset 309
extern  dscudaResult * dscudadevicereset_1(CLIENT *);
extern  dscudaResult * dscudadevicereset_1_svc(struct svc_req *);
#define dscudaStreamCreateId 400
extern  dscudaStreamCreateResult * dscudastreamcreateid_1(CLIENT *);
extern  dscudaStreamCreateResult * dscudastreamcreateid_1_svc(struct svc_req *);
#define dscudaStreamDestroyId 401
extern  dscudaResult * dscudastreamdestroyid_1(RCstream , CLIENT *);
extern  dscudaResult * dscudastreamdestroyid_1_svc(RCstream , struct svc_req *);
#define dscudaStreamSynchronizeId 402
extern  dscudaResult * dscudastreamsynchronizeid_1(RCstream , CLIENT *);
extern  dscudaResult * dscudastreamsynchronizeid_1_svc(RCstream , struct svc_req *);
#define dscudaStreamQueryId 403
extern  dscudaResult * dscudastreamqueryid_1(RCstream , CLIENT *);
extern  dscudaResult * dscudastreamqueryid_1_svc(RCstream , struct svc_req *);
#define dscudaStreamWaitEventId 404
extern  dscudaResult * dscudastreamwaiteventid_1(RCstream , RCevent , u_int , CLIENT *);
extern  dscudaResult * dscudastreamwaiteventid_1_svc(RCstream , RCevent , u_int , struct svc_req *);
#define dscudaEventCreateId 500
extern  dscudaEventCreateResult * dscudaeventcreateid_1(CLIENT *);
extern  dscudaEventCreateResult * dscudaeventcreateid_1_svc(struct svc_req *);
#define dscudaEventCreateWithFlagsId 501
extern  dscudaEventCreateResult * dscudaeventcreatewithflagsid_1(u_int , CLIENT *);
extern  dscudaEventCreateResult * dscudaeventcreatewithflagsid_1_svc(u_int , struct svc_req *);
#define dscudaEventDestroyId 502
extern  dscudaResult * dscudaeventdestroyid_1(RCevent , CLIENT *);
extern  dscudaResult * dscudaeventdestroyid_1_svc(RCevent , struct svc_req *);
#define dscudaEventElapsedTimeId 503
extern  dscudaEventElapsedTimeResult * dscudaeventelapsedtimeid_1(RCevent , RCevent , CLIENT *);
extern  dscudaEventElapsedTimeResult * dscudaeventelapsedtimeid_1_svc(RCevent , RCevent , struct svc_req *);
#define dscudaEventRecordId 504
extern  dscudaResult * dscudaeventrecordid_1(RCevent , RCstream , CLIENT *);
extern  dscudaResult * dscudaeventrecordid_1_svc(RCevent , RCstream , struct svc_req *);
#define dscudaEventSynchronizeId 505
extern  dscudaResult * dscudaeventsynchronizeid_1(RCevent , CLIENT *);
extern  dscudaResult * dscudaeventsynchronizeid_1_svc(RCevent , struct svc_req *);
#define dscudaEventQueryId 506
extern  dscudaResult * dscudaeventqueryid_1(RCevent , CLIENT *);
extern  dscudaResult * dscudaeventqueryid_1_svc(RCevent , struct svc_req *);
#define dscudaLaunchKernelId 600
extern  void * dscudalaunchkernelid_1(int , int , char *, RCdim3 , RCdim3 , RCsize , RCstream , RCargs , CLIENT *);
extern  void * dscudalaunchkernelid_1_svc(int , int , char *, RCdim3 , RCdim3 , RCsize , RCstream , RCargs , struct svc_req *);
#define dscudaLoadModuleId 601
extern  dscudaLoadModuleResult * dscudaloadmoduleid_1(RCipaddr , RCpid , char *, char *, CLIENT *);
extern  dscudaLoadModuleResult * dscudaloadmoduleid_1_svc(RCipaddr , RCpid , char *, char *, struct svc_req *);
#define dscudaFuncGetAttributesId 602
extern  dscudaFuncGetAttributesResult * dscudafuncgetattributesid_1(int , char *, CLIENT *);
extern  dscudaFuncGetAttributesResult * dscudafuncgetattributesid_1_svc(int , char *, struct svc_req *);
#define dscudaMallocId 700
extern  dscudaMallocResult * dscudamallocid_1(RCsize , CLIENT *);
extern  dscudaMallocResult * dscudamallocid_1_svc(RCsize , struct svc_req *);
#define dscudaFreeId 701
extern  dscudaResult * dscudafreeid_1(RCadr , CLIENT *);
extern  dscudaResult * dscudafreeid_1_svc(RCadr , struct svc_req *);
#define dscudaMemcpyH2HId 702
extern  dscudaMemcpyH2HResult * dscudamemcpyh2hid_1(RCadr , RCbuf , RCsize , CLIENT *);
extern  dscudaMemcpyH2HResult * dscudamemcpyh2hid_1_svc(RCadr , RCbuf , RCsize , struct svc_req *);
#define dscudaMemcpyH2DId 703
extern  dscudaResult * dscudamemcpyh2did_1(RCadr , RCbuf , RCsize , CLIENT *);
extern  dscudaResult * dscudamemcpyh2did_1_svc(RCadr , RCbuf , RCsize , struct svc_req *);
#define dscudaMemcpyD2HId 704
extern  dscudaMemcpyD2HResult * dscudamemcpyd2hid_1(RCadr , RCsize , CLIENT *);
extern  dscudaMemcpyD2HResult * dscudamemcpyd2hid_1_svc(RCadr , RCsize , struct svc_req *);
#define dscudaMemcpyD2DId 705
extern  dscudaResult * dscudamemcpyd2did_1(RCadr , RCadr , RCsize , CLIENT *);
extern  dscudaResult * dscudamemcpyd2did_1_svc(RCadr , RCadr , RCsize , struct svc_req *);
#define dscudaMemcpyAsyncH2HId 706
extern  dscudaMemcpyAsyncH2HResult * dscudamemcpyasynch2hid_1(RCadr , RCbuf , RCsize , RCstream , CLIENT *);
extern  dscudaMemcpyAsyncH2HResult * dscudamemcpyasynch2hid_1_svc(RCadr , RCbuf , RCsize , RCstream , struct svc_req *);
#define dscudaMemcpyAsyncH2DId 707
extern  dscudaResult * dscudamemcpyasynch2did_1(RCadr , RCbuf , RCsize , RCstream , CLIENT *);
extern  dscudaResult * dscudamemcpyasynch2did_1_svc(RCadr , RCbuf , RCsize , RCstream , struct svc_req *);
#define dscudaMemcpyAsyncD2HId 708
extern  dscudaMemcpyAsyncD2HResult * dscudamemcpyasyncd2hid_1(RCadr , RCsize , RCstream , CLIENT *);
extern  dscudaMemcpyAsyncD2HResult * dscudamemcpyasyncd2hid_1_svc(RCadr , RCsize , RCstream , struct svc_req *);
#define dscudaMemcpyAsyncD2DId 709
extern  dscudaResult * dscudamemcpyasyncd2did_1(RCadr , RCadr , RCsize , RCstream , CLIENT *);
extern  dscudaResult * dscudamemcpyasyncd2did_1_svc(RCadr , RCadr , RCsize , RCstream , struct svc_req *);
#define dscudaMemcpyToSymbolH2DId 710
extern  dscudaResult * dscudamemcpytosymbolh2did_1(int , char *, RCbuf , RCsize , RCsize , CLIENT *);
extern  dscudaResult * dscudamemcpytosymbolh2did_1_svc(int , char *, RCbuf , RCsize , RCsize , struct svc_req *);
#define dscudaMemcpyToSymbolD2DId 711
extern  dscudaResult * dscudamemcpytosymbold2did_1(int , char *, RCadr , RCsize , RCsize , CLIENT *);
extern  dscudaResult * dscudamemcpytosymbold2did_1_svc(int , char *, RCadr , RCsize , RCsize , struct svc_req *);
#define dscudaMemcpyFromSymbolD2HId 712
extern  dscudaMemcpyFromSymbolD2HResult * dscudamemcpyfromsymbold2hid_1(int , char *, RCsize , RCsize , CLIENT *);
extern  dscudaMemcpyFromSymbolD2HResult * dscudamemcpyfromsymbold2hid_1_svc(int , char *, RCsize , RCsize , struct svc_req *);
#define dscudaMemcpyFromSymbolD2DId 713
extern  dscudaResult * dscudamemcpyfromsymbold2did_1(int , RCadr , char *, RCsize , RCsize , CLIENT *);
extern  dscudaResult * dscudamemcpyfromsymbold2did_1_svc(int , RCadr , char *, RCsize , RCsize , struct svc_req *);
#define dscudaMemsetId 714
extern  dscudaResult * dscudamemsetid_1(RCadr , int , RCsize , CLIENT *);
extern  dscudaResult * dscudamemsetid_1_svc(RCadr , int , RCsize , struct svc_req *);
#define dscudaHostAllocId 715
extern  dscudaHostAllocResult * dscudahostallocid_1(RCsize , u_int , CLIENT *);
extern  dscudaHostAllocResult * dscudahostallocid_1_svc(RCsize , u_int , struct svc_req *);
#define dscudaMallocHostId 716
extern  dscudaMallocHostResult * dscudamallochostid_1(RCsize , CLIENT *);
extern  dscudaMallocHostResult * dscudamallochostid_1_svc(RCsize , struct svc_req *);
#define dscudaFreeHostId 717
extern  dscudaResult * dscudafreehostid_1(RCadr , CLIENT *);
extern  dscudaResult * dscudafreehostid_1_svc(RCadr , struct svc_req *);
#define dscudaHostGetDevicePointerId 718
extern  dscudaHostGetDevicePointerResult * dscudahostgetdevicepointerid_1(RCadr , u_int , CLIENT *);
extern  dscudaHostGetDevicePointerResult * dscudahostgetdevicepointerid_1_svc(RCadr , u_int , struct svc_req *);
#define dscudaHostGetFlagsID 719
extern  dscudaHostGetFlagsResult * dscudahostgetflagsid_1(RCadr , CLIENT *);
extern  dscudaHostGetFlagsResult * dscudahostgetflagsid_1_svc(RCadr , struct svc_req *);
#define dscudaMallocArrayId 720
extern  dscudaMallocArrayResult * dscudamallocarrayid_1(RCchanneldesc , RCsize , RCsize , u_int , CLIENT *);
extern  dscudaMallocArrayResult * dscudamallocarrayid_1_svc(RCchanneldesc , RCsize , RCsize , u_int , struct svc_req *);
#define dscudaFreeArrayId 721
extern  dscudaResult * dscudafreearrayid_1(RCadr , CLIENT *);
extern  dscudaResult * dscudafreearrayid_1_svc(RCadr , struct svc_req *);
#define dscudaMemcpyToArrayH2HId 722
extern  dscudaMemcpyToArrayH2HResult * dscudamemcpytoarrayh2hid_1(RCadr , RCsize , RCsize , RCbuf , RCsize , CLIENT *);
extern  dscudaMemcpyToArrayH2HResult * dscudamemcpytoarrayh2hid_1_svc(RCadr , RCsize , RCsize , RCbuf , RCsize , struct svc_req *);
#define dscudaMemcpyToArrayH2DId 723
extern  dscudaResult * dscudamemcpytoarrayh2did_1(RCadr , RCsize , RCsize , RCbuf , RCsize , CLIENT *);
extern  dscudaResult * dscudamemcpytoarrayh2did_1_svc(RCadr , RCsize , RCsize , RCbuf , RCsize , struct svc_req *);
#define dscudaMemcpyToArrayD2HId 724
extern  dscudaMemcpyToArrayD2HResult * dscudamemcpytoarrayd2hid_1(RCsize , RCsize , RCadr , RCsize , CLIENT *);
extern  dscudaMemcpyToArrayD2HResult * dscudamemcpytoarrayd2hid_1_svc(RCsize , RCsize , RCadr , RCsize , struct svc_req *);
#define dscudaMemcpyToArrayD2DId 725
extern  dscudaResult * dscudamemcpytoarrayd2did_1(RCadr , RCsize , RCsize , RCadr , RCsize , CLIENT *);
extern  dscudaResult * dscudamemcpytoarrayd2did_1_svc(RCadr , RCsize , RCsize , RCadr , RCsize , struct svc_req *);
#define dscudaMallocPitchId 726
extern  dscudaMallocPitchResult * dscudamallocpitchid_1(RCsize , RCsize , CLIENT *);
extern  dscudaMallocPitchResult * dscudamallocpitchid_1_svc(RCsize , RCsize , struct svc_req *);
#define dscudaMemcpy2DToArrayH2HId 727
extern  dscudaMemcpy2DToArrayH2HResult * dscudamemcpy2dtoarrayh2hid_1(RCadr , RCsize , RCsize , RCbuf , RCsize , RCsize , RCsize , CLIENT *);
extern  dscudaMemcpy2DToArrayH2HResult * dscudamemcpy2dtoarrayh2hid_1_svc(RCadr , RCsize , RCsize , RCbuf , RCsize , RCsize , RCsize , struct svc_req *);
#define dscudaMemcpy2DToArrayH2DId 728
extern  dscudaResult * dscudamemcpy2dtoarrayh2did_1(RCadr , RCsize , RCsize , RCbuf , RCsize , RCsize , RCsize , CLIENT *);
extern  dscudaResult * dscudamemcpy2dtoarrayh2did_1_svc(RCadr , RCsize , RCsize , RCbuf , RCsize , RCsize , RCsize , struct svc_req *);
#define dscudaMemcpy2DToArrayD2HId 729
extern  dscudaMemcpy2DToArrayD2HResult * dscudamemcpy2dtoarrayd2hid_1(RCsize , RCsize , RCadr , RCsize , RCsize , RCsize , CLIENT *);
extern  dscudaMemcpy2DToArrayD2HResult * dscudamemcpy2dtoarrayd2hid_1_svc(RCsize , RCsize , RCadr , RCsize , RCsize , RCsize , struct svc_req *);
#define dscudaMemcpy2DToArrayD2DId 730
extern  dscudaResult * dscudamemcpy2dtoarrayd2did_1(RCadr , RCsize , RCsize , RCadr , RCsize , RCsize , RCsize , CLIENT *);
extern  dscudaResult * dscudamemcpy2dtoarrayd2did_1_svc(RCadr , RCsize , RCsize , RCadr , RCsize , RCsize , RCsize , struct svc_req *);
#define dscudaMemcpy2DH2HId 731
extern  dscudaMemcpy2DH2HResult * dscudamemcpy2dh2hid_1(RCadr , RCsize , RCbuf , RCsize , RCsize , RCsize , CLIENT *);
extern  dscudaMemcpy2DH2HResult * dscudamemcpy2dh2hid_1_svc(RCadr , RCsize , RCbuf , RCsize , RCsize , RCsize , struct svc_req *);
#define dscudaMemcpy2DH2DId 732
extern  dscudaResult * dscudamemcpy2dh2did_1(RCadr , RCsize , RCbuf , RCsize , RCsize , RCsize , CLIENT *);
extern  dscudaResult * dscudamemcpy2dh2did_1_svc(RCadr , RCsize , RCbuf , RCsize , RCsize , RCsize , struct svc_req *);
#define dscudaMemcpy2DD2HId 733
extern  dscudaMemcpy2DD2HResult * dscudamemcpy2dd2hid_1(RCsize , RCadr , RCsize , RCsize , RCsize , CLIENT *);
extern  dscudaMemcpy2DD2HResult * dscudamemcpy2dd2hid_1_svc(RCsize , RCadr , RCsize , RCsize , RCsize , struct svc_req *);
#define dscudaMemcpy2DD2DId 734
extern  dscudaResult * dscudamemcpy2dd2did_1(RCadr , RCsize , RCadr , RCsize , RCsize , RCsize , CLIENT *);
extern  dscudaResult * dscudamemcpy2dd2did_1_svc(RCadr , RCsize , RCadr , RCsize , RCsize , RCsize , struct svc_req *);
#define dscudaMemset2DId 735
extern  dscudaResult * dscudamemset2did_1(RCadr , RCsize , int , RCsize , RCsize , CLIENT *);
extern  dscudaResult * dscudamemset2did_1_svc(RCadr , RCsize , int , RCsize , RCsize , struct svc_req *);
#define dscudaMemcpyToSymbolAsyncH2DId 736
extern  dscudaResult * dscudamemcpytosymbolasynch2did_1(int , char *, RCbuf , RCsize , RCsize , RCstream , CLIENT *);
extern  dscudaResult * dscudamemcpytosymbolasynch2did_1_svc(int , char *, RCbuf , RCsize , RCsize , RCstream , struct svc_req *);
#define dscudaMemcpyToSymbolAsyncD2DId 737
extern  dscudaResult * dscudamemcpytosymbolasyncd2did_1(int , char *, RCadr , RCsize , RCsize , RCstream , CLIENT *);
extern  dscudaResult * dscudamemcpytosymbolasyncd2did_1_svc(int , char *, RCadr , RCsize , RCsize , RCstream , struct svc_req *);
#define dscudaMemcpyFromSymbolAsyncD2HId 738
extern  dscudaMemcpyFromSymbolAsyncD2HResult * dscudamemcpyfromsymbolasyncd2hid_1(int , char *, RCsize , RCsize , RCstream , CLIENT *);
extern  dscudaMemcpyFromSymbolAsyncD2HResult * dscudamemcpyfromsymbolasyncd2hid_1_svc(int , char *, RCsize , RCsize , RCstream , struct svc_req *);
#define dscudaMemcpyFromSymbolAsyncD2DId 739
extern  dscudaResult * dscudamemcpyfromsymbolasyncd2did_1(int , RCadr , char *, RCsize , RCsize , RCstream , CLIENT *);
extern  dscudaResult * dscudamemcpyfromsymbolasyncd2did_1_svc(int , RCadr , char *, RCsize , RCsize , RCstream , struct svc_req *);
#define dscudaCreateChannelDescId 1400
extern  dscudaCreateChannelDescResult * dscudacreatechanneldescid_1(int , int , int , int , RCchannelformat , CLIENT *);
extern  dscudaCreateChannelDescResult * dscudacreatechanneldescid_1_svc(int , int , int , int , RCchannelformat , struct svc_req *);
#define dscudaGetChannelDescId 1401
extern  dscudaGetChannelDescResult * dscudagetchanneldescid_1(RCadr , CLIENT *);
extern  dscudaGetChannelDescResult * dscudagetchanneldescid_1_svc(RCadr , struct svc_req *);
#define dscudaBindTextureId 1402
extern  dscudaBindTextureResult * dscudabindtextureid_1(int , char *, RCadr , RCsize , RCtexture , CLIENT *);
extern  dscudaBindTextureResult * dscudabindtextureid_1_svc(int , char *, RCadr , RCsize , RCtexture , struct svc_req *);
#define dscudaBindTexture2DId 1403
extern  dscudaBindTexture2DResult * dscudabindtexture2did_1(int , char *, RCadr , RCsize , RCsize , RCsize , RCtexture , CLIENT *);
extern  dscudaBindTexture2DResult * dscudabindtexture2did_1_svc(int , char *, RCadr , RCsize , RCsize , RCsize , RCtexture , struct svc_req *);
#define dscudaBindTextureToArrayId 1404
extern  dscudaResult * dscudabindtexturetoarrayid_1(int , char *, RCadr , RCtexture , CLIENT *);
extern  dscudaResult * dscudabindtexturetoarrayid_1_svc(int , char *, RCadr , RCtexture , struct svc_req *);
#define dscudaUnbindTextureId 1405
extern  dscudaResult * dscudaunbindtextureid_1(RCtexture , CLIENT *);
extern  dscudaResult * dscudaunbindtextureid_1_svc(RCtexture , struct svc_req *);
#define dscufftPlan3dId 2002
extern  dscufftPlanResult * dscufftplan3did_1(int , int , int , u_int , CLIENT *);
extern  dscufftPlanResult * dscufftplan3did_1_svc(int , int , int , u_int , struct svc_req *);
#define dscufftDestroyId 2004
extern  dscufftResult * dscufftdestroyid_1(u_int , CLIENT *);
extern  dscufftResult * dscufftdestroyid_1_svc(u_int , struct svc_req *);
#define dscufftExecC2CId 2005
extern  dscufftResult * dscufftexecc2cid_1(u_int , RCadr , RCadr , int , CLIENT *);
extern  dscufftResult * dscufftexecc2cid_1_svc(u_int , RCadr , RCadr , int , struct svc_req *);
extern int dscuda_prog_1_freeresult (SVCXPRT *, xdrproc_t, caddr_t);

#else 
#define dscudaThreadExitId 100
extern  dscudaResult * dscudathreadexitid_1();
extern  dscudaResult * dscudathreadexitid_1_svc();
#define dscudaThreadSynchronizeId 101
extern  dscudaResult * dscudathreadsynchronizeid_1();
extern  dscudaResult * dscudathreadsynchronizeid_1_svc();
#define dscudaThreadSetLimitId 102
extern  dscudaResult * dscudathreadsetlimitid_1();
extern  dscudaResult * dscudathreadsetlimitid_1_svc();
#define dscudaThreadGetLimitId 103
extern  dscudaThreadGetLimitResult * dscudathreadgetlimitid_1();
extern  dscudaThreadGetLimitResult * dscudathreadgetlimitid_1_svc();
#define dscudaThreadSetCacheConfigId 104
extern  dscudaResult * dscudathreadsetcacheconfigid_1();
extern  dscudaResult * dscudathreadsetcacheconfigid_1_svc();
#define dscudaThreadGetCacheConfigId 105
extern  dscudaThreadGetCacheConfigResult * dscudathreadgetcacheconfigid_1();
extern  dscudaThreadGetCacheConfigResult * dscudathreadgetcacheconfigid_1_svc();
#define dscudaGetLastErrorId 200
extern  dscudaResult * dscudagetlasterrorid_1();
extern  dscudaResult * dscudagetlasterrorid_1_svc();
#define dscudaPeekAtLastErrorId 201
extern  dscudaResult * dscudapeekatlasterrorid_1();
extern  dscudaResult * dscudapeekatlasterrorid_1_svc();
#define dscudaGetErrorStringId 202
extern  dscudaGetErrorStringResult * dscudageterrorstringid_1();
extern  dscudaGetErrorStringResult * dscudageterrorstringid_1_svc();
#define dscudaGetDeviceId 300
extern  dscudaGetDeviceResult * dscudagetdeviceid_1();
extern  dscudaGetDeviceResult * dscudagetdeviceid_1_svc();
#define dscudaGetDeviceCountId 301
extern  dscudaGetDeviceCountResult * dscudagetdevicecountid_1();
extern  dscudaGetDeviceCountResult * dscudagetdevicecountid_1_svc();
#define dscudaGetDevicePropertiesId 302
extern  dscudaGetDevicePropertiesResult * dscudagetdevicepropertiesid_1();
extern  dscudaGetDevicePropertiesResult * dscudagetdevicepropertiesid_1_svc();
#define dscudaDriverGetVersionId 303
extern  dscudaDriverGetVersionResult * dscudadrivergetversionid_1();
extern  dscudaDriverGetVersionResult * dscudadrivergetversionid_1_svc();
#define dscudaRuntimeGetVersionId 304
extern  dscudaRuntimeGetVersionResult * dscudaruntimegetversionid_1();
extern  dscudaRuntimeGetVersionResult * dscudaruntimegetversionid_1_svc();
#define dscudaSetDeviceId 305
extern  dscudaResult * dscudasetdeviceid_1();
extern  dscudaResult * dscudasetdeviceid_1_svc();
#define dscudaSetDeviceFlagsId 306
extern  dscudaResult * dscudasetdeviceflagsid_1();
extern  dscudaResult * dscudasetdeviceflagsid_1_svc();
#define dscudaChooseDeviceId 307
extern  dscudaChooseDeviceResult * dscudachoosedeviceid_1();
extern  dscudaChooseDeviceResult * dscudachoosedeviceid_1_svc();
#define dscudaDeviceSynchronize 308
extern  dscudaResult * dscudadevicesynchronize_1();
extern  dscudaResult * dscudadevicesynchronize_1_svc();
#define dscudaDeviceReset 309
extern  dscudaResult * dscudadevicereset_1();
extern  dscudaResult * dscudadevicereset_1_svc();
#define dscudaStreamCreateId 400
extern  dscudaStreamCreateResult * dscudastreamcreateid_1();
extern  dscudaStreamCreateResult * dscudastreamcreateid_1_svc();
#define dscudaStreamDestroyId 401
extern  dscudaResult * dscudastreamdestroyid_1();
extern  dscudaResult * dscudastreamdestroyid_1_svc();
#define dscudaStreamSynchronizeId 402
extern  dscudaResult * dscudastreamsynchronizeid_1();
extern  dscudaResult * dscudastreamsynchronizeid_1_svc();
#define dscudaStreamQueryId 403
extern  dscudaResult * dscudastreamqueryid_1();
extern  dscudaResult * dscudastreamqueryid_1_svc();
#define dscudaStreamWaitEventId 404
extern  dscudaResult * dscudastreamwaiteventid_1();
extern  dscudaResult * dscudastreamwaiteventid_1_svc();
#define dscudaEventCreateId 500
extern  dscudaEventCreateResult * dscudaeventcreateid_1();
extern  dscudaEventCreateResult * dscudaeventcreateid_1_svc();
#define dscudaEventCreateWithFlagsId 501
extern  dscudaEventCreateResult * dscudaeventcreatewithflagsid_1();
extern  dscudaEventCreateResult * dscudaeventcreatewithflagsid_1_svc();
#define dscudaEventDestroyId 502
extern  dscudaResult * dscudaeventdestroyid_1();
extern  dscudaResult * dscudaeventdestroyid_1_svc();
#define dscudaEventElapsedTimeId 503
extern  dscudaEventElapsedTimeResult * dscudaeventelapsedtimeid_1();
extern  dscudaEventElapsedTimeResult * dscudaeventelapsedtimeid_1_svc();
#define dscudaEventRecordId 504
extern  dscudaResult * dscudaeventrecordid_1();
extern  dscudaResult * dscudaeventrecordid_1_svc();
#define dscudaEventSynchronizeId 505
extern  dscudaResult * dscudaeventsynchronizeid_1();
extern  dscudaResult * dscudaeventsynchronizeid_1_svc();
#define dscudaEventQueryId 506
extern  dscudaResult * dscudaeventqueryid_1();
extern  dscudaResult * dscudaeventqueryid_1_svc();
#define dscudaLaunchKernelId 600
extern  void * dscudalaunchkernelid_1();
extern  void * dscudalaunchkernelid_1_svc();
#define dscudaLoadModuleId 601
extern  dscudaLoadModuleResult * dscudaloadmoduleid_1();
extern  dscudaLoadModuleResult * dscudaloadmoduleid_1_svc();
#define dscudaFuncGetAttributesId 602
extern  dscudaFuncGetAttributesResult * dscudafuncgetattributesid_1();
extern  dscudaFuncGetAttributesResult * dscudafuncgetattributesid_1_svc();
#define dscudaMallocId 700
extern  dscudaMallocResult * dscudamallocid_1();
extern  dscudaMallocResult * dscudamallocid_1_svc();
#define dscudaFreeId 701
extern  dscudaResult * dscudafreeid_1();
extern  dscudaResult * dscudafreeid_1_svc();
#define dscudaMemcpyH2HId 702
extern  dscudaMemcpyH2HResult * dscudamemcpyh2hid_1();
extern  dscudaMemcpyH2HResult * dscudamemcpyh2hid_1_svc();
#define dscudaMemcpyH2DId 703
extern  dscudaResult * dscudamemcpyh2did_1();
extern  dscudaResult * dscudamemcpyh2did_1_svc();
#define dscudaMemcpyD2HId 704
extern  dscudaMemcpyD2HResult * dscudamemcpyd2hid_1();
extern  dscudaMemcpyD2HResult * dscudamemcpyd2hid_1_svc();
#define dscudaMemcpyD2DId 705
extern  dscudaResult * dscudamemcpyd2did_1();
extern  dscudaResult * dscudamemcpyd2did_1_svc();
#define dscudaMemcpyAsyncH2HId 706
extern  dscudaMemcpyAsyncH2HResult * dscudamemcpyasynch2hid_1();
extern  dscudaMemcpyAsyncH2HResult * dscudamemcpyasynch2hid_1_svc();
#define dscudaMemcpyAsyncH2DId 707
extern  dscudaResult * dscudamemcpyasynch2did_1();
extern  dscudaResult * dscudamemcpyasynch2did_1_svc();
#define dscudaMemcpyAsyncD2HId 708
extern  dscudaMemcpyAsyncD2HResult * dscudamemcpyasyncd2hid_1();
extern  dscudaMemcpyAsyncD2HResult * dscudamemcpyasyncd2hid_1_svc();
#define dscudaMemcpyAsyncD2DId 709
extern  dscudaResult * dscudamemcpyasyncd2did_1();
extern  dscudaResult * dscudamemcpyasyncd2did_1_svc();
#define dscudaMemcpyToSymbolH2DId 710
extern  dscudaResult * dscudamemcpytosymbolh2did_1();
extern  dscudaResult * dscudamemcpytosymbolh2did_1_svc();
#define dscudaMemcpyToSymbolD2DId 711
extern  dscudaResult * dscudamemcpytosymbold2did_1();
extern  dscudaResult * dscudamemcpytosymbold2did_1_svc();
#define dscudaMemcpyFromSymbolD2HId 712
extern  dscudaMemcpyFromSymbolD2HResult * dscudamemcpyfromsymbold2hid_1();
extern  dscudaMemcpyFromSymbolD2HResult * dscudamemcpyfromsymbold2hid_1_svc();
#define dscudaMemcpyFromSymbolD2DId 713
extern  dscudaResult * dscudamemcpyfromsymbold2did_1();
extern  dscudaResult * dscudamemcpyfromsymbold2did_1_svc();
#define dscudaMemsetId 714
extern  dscudaResult * dscudamemsetid_1();
extern  dscudaResult * dscudamemsetid_1_svc();
#define dscudaHostAllocId 715
extern  dscudaHostAllocResult * dscudahostallocid_1();
extern  dscudaHostAllocResult * dscudahostallocid_1_svc();
#define dscudaMallocHostId 716
extern  dscudaMallocHostResult * dscudamallochostid_1();
extern  dscudaMallocHostResult * dscudamallochostid_1_svc();
#define dscudaFreeHostId 717
extern  dscudaResult * dscudafreehostid_1();
extern  dscudaResult * dscudafreehostid_1_svc();
#define dscudaHostGetDevicePointerId 718
extern  dscudaHostGetDevicePointerResult * dscudahostgetdevicepointerid_1();
extern  dscudaHostGetDevicePointerResult * dscudahostgetdevicepointerid_1_svc();
#define dscudaHostGetFlagsID 719
extern  dscudaHostGetFlagsResult * dscudahostgetflagsid_1();
extern  dscudaHostGetFlagsResult * dscudahostgetflagsid_1_svc();
#define dscudaMallocArrayId 720
extern  dscudaMallocArrayResult * dscudamallocarrayid_1();
extern  dscudaMallocArrayResult * dscudamallocarrayid_1_svc();
#define dscudaFreeArrayId 721
extern  dscudaResult * dscudafreearrayid_1();
extern  dscudaResult * dscudafreearrayid_1_svc();
#define dscudaMemcpyToArrayH2HId 722
extern  dscudaMemcpyToArrayH2HResult * dscudamemcpytoarrayh2hid_1();
extern  dscudaMemcpyToArrayH2HResult * dscudamemcpytoarrayh2hid_1_svc();
#define dscudaMemcpyToArrayH2DId 723
extern  dscudaResult * dscudamemcpytoarrayh2did_1();
extern  dscudaResult * dscudamemcpytoarrayh2did_1_svc();
#define dscudaMemcpyToArrayD2HId 724
extern  dscudaMemcpyToArrayD2HResult * dscudamemcpytoarrayd2hid_1();
extern  dscudaMemcpyToArrayD2HResult * dscudamemcpytoarrayd2hid_1_svc();
#define dscudaMemcpyToArrayD2DId 725
extern  dscudaResult * dscudamemcpytoarrayd2did_1();
extern  dscudaResult * dscudamemcpytoarrayd2did_1_svc();
#define dscudaMallocPitchId 726
extern  dscudaMallocPitchResult * dscudamallocpitchid_1();
extern  dscudaMallocPitchResult * dscudamallocpitchid_1_svc();
#define dscudaMemcpy2DToArrayH2HId 727
extern  dscudaMemcpy2DToArrayH2HResult * dscudamemcpy2dtoarrayh2hid_1();
extern  dscudaMemcpy2DToArrayH2HResult * dscudamemcpy2dtoarrayh2hid_1_svc();
#define dscudaMemcpy2DToArrayH2DId 728
extern  dscudaResult * dscudamemcpy2dtoarrayh2did_1();
extern  dscudaResult * dscudamemcpy2dtoarrayh2did_1_svc();
#define dscudaMemcpy2DToArrayD2HId 729
extern  dscudaMemcpy2DToArrayD2HResult * dscudamemcpy2dtoarrayd2hid_1();
extern  dscudaMemcpy2DToArrayD2HResult * dscudamemcpy2dtoarrayd2hid_1_svc();
#define dscudaMemcpy2DToArrayD2DId 730
extern  dscudaResult * dscudamemcpy2dtoarrayd2did_1();
extern  dscudaResult * dscudamemcpy2dtoarrayd2did_1_svc();
#define dscudaMemcpy2DH2HId 731
extern  dscudaMemcpy2DH2HResult * dscudamemcpy2dh2hid_1();
extern  dscudaMemcpy2DH2HResult * dscudamemcpy2dh2hid_1_svc();
#define dscudaMemcpy2DH2DId 732
extern  dscudaResult * dscudamemcpy2dh2did_1();
extern  dscudaResult * dscudamemcpy2dh2did_1_svc();
#define dscudaMemcpy2DD2HId 733
extern  dscudaMemcpy2DD2HResult * dscudamemcpy2dd2hid_1();
extern  dscudaMemcpy2DD2HResult * dscudamemcpy2dd2hid_1_svc();
#define dscudaMemcpy2DD2DId 734
extern  dscudaResult * dscudamemcpy2dd2did_1();
extern  dscudaResult * dscudamemcpy2dd2did_1_svc();
#define dscudaMemset2DId 735
extern  dscudaResult * dscudamemset2did_1();
extern  dscudaResult * dscudamemset2did_1_svc();
#define dscudaMemcpyToSymbolAsyncH2DId 736
extern  dscudaResult * dscudamemcpytosymbolasynch2did_1();
extern  dscudaResult * dscudamemcpytosymbolasynch2did_1_svc();
#define dscudaMemcpyToSymbolAsyncD2DId 737
extern  dscudaResult * dscudamemcpytosymbolasyncd2did_1();
extern  dscudaResult * dscudamemcpytosymbolasyncd2did_1_svc();
#define dscudaMemcpyFromSymbolAsyncD2HId 738
extern  dscudaMemcpyFromSymbolAsyncD2HResult * dscudamemcpyfromsymbolasyncd2hid_1();
extern  dscudaMemcpyFromSymbolAsyncD2HResult * dscudamemcpyfromsymbolasyncd2hid_1_svc();
#define dscudaMemcpyFromSymbolAsyncD2DId 739
extern  dscudaResult * dscudamemcpyfromsymbolasyncd2did_1();
extern  dscudaResult * dscudamemcpyfromsymbolasyncd2did_1_svc();
#define dscudaCreateChannelDescId 1400
extern  dscudaCreateChannelDescResult * dscudacreatechanneldescid_1();
extern  dscudaCreateChannelDescResult * dscudacreatechanneldescid_1_svc();
#define dscudaGetChannelDescId 1401
extern  dscudaGetChannelDescResult * dscudagetchanneldescid_1();
extern  dscudaGetChannelDescResult * dscudagetchanneldescid_1_svc();
#define dscudaBindTextureId 1402
extern  dscudaBindTextureResult * dscudabindtextureid_1();
extern  dscudaBindTextureResult * dscudabindtextureid_1_svc();
#define dscudaBindTexture2DId 1403
extern  dscudaBindTexture2DResult * dscudabindtexture2did_1();
extern  dscudaBindTexture2DResult * dscudabindtexture2did_1_svc();
#define dscudaBindTextureToArrayId 1404
extern  dscudaResult * dscudabindtexturetoarrayid_1();
extern  dscudaResult * dscudabindtexturetoarrayid_1_svc();
#define dscudaUnbindTextureId 1405
extern  dscudaResult * dscudaunbindtextureid_1();
extern  dscudaResult * dscudaunbindtextureid_1_svc();
#define dscufftPlan3dId 2002
extern  dscufftPlanResult * dscufftplan3did_1();
extern  dscufftPlanResult * dscufftplan3did_1_svc();
#define dscufftDestroyId 2004
extern  dscufftResult * dscufftdestroyid_1();
extern  dscufftResult * dscufftdestroyid_1_svc();
#define dscufftExecC2CId 2005
extern  dscufftResult * dscufftexecc2cid_1();
extern  dscufftResult * dscufftexecc2cid_1_svc();
extern int dscuda_prog_1_freeresult ();
#endif 



#if defined(__STDC__) || defined(__cplusplus)
extern  bool_t xdr_RCadr (XDR *, RCadr*);
extern  bool_t xdr_RCstream (XDR *, RCstream*);
extern  bool_t xdr_RCevent (XDR *, RCevent*);
extern  bool_t xdr_RCipaddr (XDR *, RCipaddr*);
extern  bool_t xdr_RCsize (XDR *, RCsize*);
extern  bool_t xdr_RCerror (XDR *, RCerror*);
extern  bool_t xdr_RCbuf (XDR *, RCbuf*);
extern  bool_t xdr_RCchannelformat (XDR *, RCchannelformat*);
extern  bool_t xdr_RCpid (XDR *, RCpid*);
extern  bool_t xdr_RCchanneldesc_t (XDR *, RCchanneldesc_t*);
extern  bool_t xdr_RCchanneldesc (XDR *, RCchanneldesc*);
extern  bool_t xdr_RCtexture_t (XDR *, RCtexture_t*);
extern  bool_t xdr_RCtexture (XDR *, RCtexture*);
extern  bool_t xdr_RCfuncattr_t (XDR *, RCfuncattr_t*);
extern  bool_t xdr_RCfuncattr (XDR *, RCfuncattr*);
extern  bool_t xdr_RCargType (XDR *, RCargType*);
extern  bool_t xdr_RCargVal (XDR *, RCargVal*);
extern  bool_t xdr_RCarg (XDR *, RCarg*);
extern  bool_t xdr_RCargs (XDR *, RCargs*);
extern  bool_t xdr_dscudaResult (XDR *, dscudaResult*);
extern  bool_t xdr_dscudaThreadGetLimitResult (XDR *, dscudaThreadGetLimitResult*);
extern  bool_t xdr_dscudaThreadGetCacheConfigResult (XDR *, dscudaThreadGetCacheConfigResult*);
extern  bool_t xdr_dscudaMallocResult (XDR *, dscudaMallocResult*);
extern  bool_t xdr_dscudaHostAllocResult (XDR *, dscudaHostAllocResult*);
extern  bool_t xdr_dscudaMallocHostResult (XDR *, dscudaMallocHostResult*);
extern  bool_t xdr_dscudaMallocArrayResult (XDR *, dscudaMallocArrayResult*);
extern  bool_t xdr_dscudaMallocPitchResult (XDR *, dscudaMallocPitchResult*);
extern  bool_t xdr_dscudaMemcpyD2HResult (XDR *, dscudaMemcpyD2HResult*);
extern  bool_t xdr_dscudaMemcpyH2HResult (XDR *, dscudaMemcpyH2HResult*);
extern  bool_t xdr_dscudaMemcpyToArrayD2HResult (XDR *, dscudaMemcpyToArrayD2HResult*);
extern  bool_t xdr_dscudaMemcpyToArrayH2HResult (XDR *, dscudaMemcpyToArrayH2HResult*);
extern  bool_t xdr_dscudaMemcpy2DToArrayD2HResult (XDR *, dscudaMemcpy2DToArrayD2HResult*);
extern  bool_t xdr_dscudaMemcpy2DToArrayH2HResult (XDR *, dscudaMemcpy2DToArrayH2HResult*);
extern  bool_t xdr_dscudaMemcpy2DD2HResult (XDR *, dscudaMemcpy2DD2HResult*);
extern  bool_t xdr_dscudaMemcpy2DH2HResult (XDR *, dscudaMemcpy2DH2HResult*);
extern  bool_t xdr_dscudaGetDeviceResult (XDR *, dscudaGetDeviceResult*);
extern  bool_t xdr_dscudaGetDeviceCountResult (XDR *, dscudaGetDeviceCountResult*);
extern  bool_t xdr_dscudaGetDevicePropertiesResult (XDR *, dscudaGetDevicePropertiesResult*);
extern  bool_t xdr_dscudaDriverGetVersionResult (XDR *, dscudaDriverGetVersionResult*);
extern  bool_t xdr_dscudaRuntimeGetVersionResult (XDR *, dscudaRuntimeGetVersionResult*);
extern  bool_t xdr_dscudaGetErrorStringResult (XDR *, dscudaGetErrorStringResult*);
extern  bool_t xdr_dscudaCreateChannelDescResult (XDR *, dscudaCreateChannelDescResult*);
extern  bool_t xdr_dscudaGetChannelDescResult (XDR *, dscudaGetChannelDescResult*);
extern  bool_t xdr_dscudaChooseDeviceResult (XDR *, dscudaChooseDeviceResult*);
extern  bool_t xdr_dscudaMemcpyAsyncD2HResult (XDR *, dscudaMemcpyAsyncD2HResult*);
extern  bool_t xdr_dscudaMemcpyAsyncH2HResult (XDR *, dscudaMemcpyAsyncH2HResult*);
extern  bool_t xdr_dscudaMemcpyFromSymbolD2HResult (XDR *, dscudaMemcpyFromSymbolD2HResult*);
extern  bool_t xdr_dscudaMemcpyFromSymbolAsyncD2HResult (XDR *, dscudaMemcpyFromSymbolAsyncD2HResult*);
extern  bool_t xdr_dscudaStreamCreateResult (XDR *, dscudaStreamCreateResult*);
extern  bool_t xdr_dscudaEventCreateResult (XDR *, dscudaEventCreateResult*);
extern  bool_t xdr_dscudaEventElapsedTimeResult (XDR *, dscudaEventElapsedTimeResult*);
extern  bool_t xdr_dscudaHostGetDevicePointerResult (XDR *, dscudaHostGetDevicePointerResult*);
extern  bool_t xdr_dscudaHostGetFlagsResult (XDR *, dscudaHostGetFlagsResult*);
extern  bool_t xdr_dscudaLoadModuleResult (XDR *, dscudaLoadModuleResult*);
extern  bool_t xdr_dscudaFuncGetAttributesResult (XDR *, dscudaFuncGetAttributesResult*);
extern  bool_t xdr_dscudaBindTextureResult (XDR *, dscudaBindTextureResult*);
extern  bool_t xdr_dscudaBindTexture2DResult (XDR *, dscudaBindTexture2DResult*);
extern  bool_t xdr_dscufftResult (XDR *, dscufftResult*);
extern  bool_t xdr_dscufftPlanResult (XDR *, dscufftPlanResult*);
extern  bool_t xdr_dscublasResult (XDR *, dscublasResult*);
extern  bool_t xdr_dscublasCreateResult (XDR *, dscublasCreateResult*);
extern  bool_t xdr_dscublasGetVectorResult (XDR *, dscublasGetVectorResult*);
extern  bool_t xdr_RCdim3 (XDR *, RCdim3*);
extern  bool_t xdr_dscudathreadsetlimitid_1_argument (XDR *, dscudathreadsetlimitid_1_argument*);
extern  bool_t xdr_dscudastreamwaiteventid_1_argument (XDR *, dscudastreamwaiteventid_1_argument*);
extern  bool_t xdr_dscudaeventelapsedtimeid_1_argument (XDR *, dscudaeventelapsedtimeid_1_argument*);
extern  bool_t xdr_dscudaeventrecordid_1_argument (XDR *, dscudaeventrecordid_1_argument*);
extern  bool_t xdr_dscudalaunchkernelid_1_argument (XDR *, dscudalaunchkernelid_1_argument*);
extern  bool_t xdr_dscudaloadmoduleid_1_argument (XDR *, dscudaloadmoduleid_1_argument*);
extern  bool_t xdr_dscudafuncgetattributesid_1_argument (XDR *, dscudafuncgetattributesid_1_argument*);
extern  bool_t xdr_dscudamemcpyh2hid_1_argument (XDR *, dscudamemcpyh2hid_1_argument*);
extern  bool_t xdr_dscudamemcpyh2did_1_argument (XDR *, dscudamemcpyh2did_1_argument*);
extern  bool_t xdr_dscudamemcpyd2hid_1_argument (XDR *, dscudamemcpyd2hid_1_argument*);
extern  bool_t xdr_dscudamemcpyd2did_1_argument (XDR *, dscudamemcpyd2did_1_argument*);
extern  bool_t xdr_dscudamemcpyasynch2hid_1_argument (XDR *, dscudamemcpyasynch2hid_1_argument*);
extern  bool_t xdr_dscudamemcpyasynch2did_1_argument (XDR *, dscudamemcpyasynch2did_1_argument*);
extern  bool_t xdr_dscudamemcpyasyncd2hid_1_argument (XDR *, dscudamemcpyasyncd2hid_1_argument*);
extern  bool_t xdr_dscudamemcpyasyncd2did_1_argument (XDR *, dscudamemcpyasyncd2did_1_argument*);
extern  bool_t xdr_dscudamemcpytosymbolh2did_1_argument (XDR *, dscudamemcpytosymbolh2did_1_argument*);
extern  bool_t xdr_dscudamemcpytosymbold2did_1_argument (XDR *, dscudamemcpytosymbold2did_1_argument*);
extern  bool_t xdr_dscudamemcpyfromsymbold2hid_1_argument (XDR *, dscudamemcpyfromsymbold2hid_1_argument*);
extern  bool_t xdr_dscudamemcpyfromsymbold2did_1_argument (XDR *, dscudamemcpyfromsymbold2did_1_argument*);
extern  bool_t xdr_dscudamemsetid_1_argument (XDR *, dscudamemsetid_1_argument*);
extern  bool_t xdr_dscudahostallocid_1_argument (XDR *, dscudahostallocid_1_argument*);
extern  bool_t xdr_dscudahostgetdevicepointerid_1_argument (XDR *, dscudahostgetdevicepointerid_1_argument*);
extern  bool_t xdr_dscudamallocarrayid_1_argument (XDR *, dscudamallocarrayid_1_argument*);
extern  bool_t xdr_dscudamemcpytoarrayh2hid_1_argument (XDR *, dscudamemcpytoarrayh2hid_1_argument*);
extern  bool_t xdr_dscudamemcpytoarrayh2did_1_argument (XDR *, dscudamemcpytoarrayh2did_1_argument*);
extern  bool_t xdr_dscudamemcpytoarrayd2hid_1_argument (XDR *, dscudamemcpytoarrayd2hid_1_argument*);
extern  bool_t xdr_dscudamemcpytoarrayd2did_1_argument (XDR *, dscudamemcpytoarrayd2did_1_argument*);
extern  bool_t xdr_dscudamallocpitchid_1_argument (XDR *, dscudamallocpitchid_1_argument*);
extern  bool_t xdr_dscudamemcpy2dtoarrayh2hid_1_argument (XDR *, dscudamemcpy2dtoarrayh2hid_1_argument*);
extern  bool_t xdr_dscudamemcpy2dtoarrayh2did_1_argument (XDR *, dscudamemcpy2dtoarrayh2did_1_argument*);
extern  bool_t xdr_dscudamemcpy2dtoarrayd2hid_1_argument (XDR *, dscudamemcpy2dtoarrayd2hid_1_argument*);
extern  bool_t xdr_dscudamemcpy2dtoarrayd2did_1_argument (XDR *, dscudamemcpy2dtoarrayd2did_1_argument*);
extern  bool_t xdr_dscudamemcpy2dh2hid_1_argument (XDR *, dscudamemcpy2dh2hid_1_argument*);
extern  bool_t xdr_dscudamemcpy2dh2did_1_argument (XDR *, dscudamemcpy2dh2did_1_argument*);
extern  bool_t xdr_dscudamemcpy2dd2hid_1_argument (XDR *, dscudamemcpy2dd2hid_1_argument*);
extern  bool_t xdr_dscudamemcpy2dd2did_1_argument (XDR *, dscudamemcpy2dd2did_1_argument*);
extern  bool_t xdr_dscudamemset2did_1_argument (XDR *, dscudamemset2did_1_argument*);
extern  bool_t xdr_dscudamemcpytosymbolasynch2did_1_argument (XDR *, dscudamemcpytosymbolasynch2did_1_argument*);
extern  bool_t xdr_dscudamemcpytosymbolasyncd2did_1_argument (XDR *, dscudamemcpytosymbolasyncd2did_1_argument*);
extern  bool_t xdr_dscudamemcpyfromsymbolasyncd2hid_1_argument (XDR *, dscudamemcpyfromsymbolasyncd2hid_1_argument*);
extern  bool_t xdr_dscudamemcpyfromsymbolasyncd2did_1_argument (XDR *, dscudamemcpyfromsymbolasyncd2did_1_argument*);
extern  bool_t xdr_dscudacreatechanneldescid_1_argument (XDR *, dscudacreatechanneldescid_1_argument*);
extern  bool_t xdr_dscudabindtextureid_1_argument (XDR *, dscudabindtextureid_1_argument*);
extern  bool_t xdr_dscudabindtexture2did_1_argument (XDR *, dscudabindtexture2did_1_argument*);
extern  bool_t xdr_dscudabindtexturetoarrayid_1_argument (XDR *, dscudabindtexturetoarrayid_1_argument*);
extern  bool_t xdr_dscufftplan3did_1_argument (XDR *, dscufftplan3did_1_argument*);
extern  bool_t xdr_dscufftexecc2cid_1_argument (XDR *, dscufftexecc2cid_1_argument*);

#else 
extern bool_t xdr_RCadr ();
extern bool_t xdr_RCstream ();
extern bool_t xdr_RCevent ();
extern bool_t xdr_RCipaddr ();
extern bool_t xdr_RCsize ();
extern bool_t xdr_RCerror ();
extern bool_t xdr_RCbuf ();
extern bool_t xdr_RCchannelformat ();
extern bool_t xdr_RCpid ();
extern bool_t xdr_RCchanneldesc_t ();
extern bool_t xdr_RCchanneldesc ();
extern bool_t xdr_RCtexture_t ();
extern bool_t xdr_RCtexture ();
extern bool_t xdr_RCfuncattr_t ();
extern bool_t xdr_RCfuncattr ();
extern bool_t xdr_RCargType ();
extern bool_t xdr_RCargVal ();
extern bool_t xdr_RCarg ();
extern bool_t xdr_RCargs ();
extern bool_t xdr_dscudaResult ();
extern bool_t xdr_dscudaThreadGetLimitResult ();
extern bool_t xdr_dscudaThreadGetCacheConfigResult ();
extern bool_t xdr_dscudaMallocResult ();
extern bool_t xdr_dscudaHostAllocResult ();
extern bool_t xdr_dscudaMallocHostResult ();
extern bool_t xdr_dscudaMallocArrayResult ();
extern bool_t xdr_dscudaMallocPitchResult ();
extern bool_t xdr_dscudaMemcpyD2HResult ();
extern bool_t xdr_dscudaMemcpyH2HResult ();
extern bool_t xdr_dscudaMemcpyToArrayD2HResult ();
extern bool_t xdr_dscudaMemcpyToArrayH2HResult ();
extern bool_t xdr_dscudaMemcpy2DToArrayD2HResult ();
extern bool_t xdr_dscudaMemcpy2DToArrayH2HResult ();
extern bool_t xdr_dscudaMemcpy2DD2HResult ();
extern bool_t xdr_dscudaMemcpy2DH2HResult ();
extern bool_t xdr_dscudaGetDeviceResult ();
extern bool_t xdr_dscudaGetDeviceCountResult ();
extern bool_t xdr_dscudaGetDevicePropertiesResult ();
extern bool_t xdr_dscudaDriverGetVersionResult ();
extern bool_t xdr_dscudaRuntimeGetVersionResult ();
extern bool_t xdr_dscudaGetErrorStringResult ();
extern bool_t xdr_dscudaCreateChannelDescResult ();
extern bool_t xdr_dscudaGetChannelDescResult ();
extern bool_t xdr_dscudaChooseDeviceResult ();
extern bool_t xdr_dscudaMemcpyAsyncD2HResult ();
extern bool_t xdr_dscudaMemcpyAsyncH2HResult ();
extern bool_t xdr_dscudaMemcpyFromSymbolD2HResult ();
extern bool_t xdr_dscudaMemcpyFromSymbolAsyncD2HResult ();
extern bool_t xdr_dscudaStreamCreateResult ();
extern bool_t xdr_dscudaEventCreateResult ();
extern bool_t xdr_dscudaEventElapsedTimeResult ();
extern bool_t xdr_dscudaHostGetDevicePointerResult ();
extern bool_t xdr_dscudaHostGetFlagsResult ();
extern bool_t xdr_dscudaLoadModuleResult ();
extern bool_t xdr_dscudaFuncGetAttributesResult ();
extern bool_t xdr_dscudaBindTextureResult ();
extern bool_t xdr_dscudaBindTexture2DResult ();
extern bool_t xdr_dscufftResult ();
extern bool_t xdr_dscufftPlanResult ();
extern bool_t xdr_dscublasResult ();
extern bool_t xdr_dscublasCreateResult ();
extern bool_t xdr_dscublasGetVectorResult ();
extern bool_t xdr_RCdim3 ();
extern bool_t xdr_dscudathreadsetlimitid_1_argument ();
extern bool_t xdr_dscudastreamwaiteventid_1_argument ();
extern bool_t xdr_dscudaeventelapsedtimeid_1_argument ();
extern bool_t xdr_dscudaeventrecordid_1_argument ();
extern bool_t xdr_dscudalaunchkernelid_1_argument ();
extern bool_t xdr_dscudaloadmoduleid_1_argument ();
extern bool_t xdr_dscudafuncgetattributesid_1_argument ();
extern bool_t xdr_dscudamemcpyh2hid_1_argument ();
extern bool_t xdr_dscudamemcpyh2did_1_argument ();
extern bool_t xdr_dscudamemcpyd2hid_1_argument ();
extern bool_t xdr_dscudamemcpyd2did_1_argument ();
extern bool_t xdr_dscudamemcpyasynch2hid_1_argument ();
extern bool_t xdr_dscudamemcpyasynch2did_1_argument ();
extern bool_t xdr_dscudamemcpyasyncd2hid_1_argument ();
extern bool_t xdr_dscudamemcpyasyncd2did_1_argument ();
extern bool_t xdr_dscudamemcpytosymbolh2did_1_argument ();
extern bool_t xdr_dscudamemcpytosymbold2did_1_argument ();
extern bool_t xdr_dscudamemcpyfromsymbold2hid_1_argument ();
extern bool_t xdr_dscudamemcpyfromsymbold2did_1_argument ();
extern bool_t xdr_dscudamemsetid_1_argument ();
extern bool_t xdr_dscudahostallocid_1_argument ();
extern bool_t xdr_dscudahostgetdevicepointerid_1_argument ();
extern bool_t xdr_dscudamallocarrayid_1_argument ();
extern bool_t xdr_dscudamemcpytoarrayh2hid_1_argument ();
extern bool_t xdr_dscudamemcpytoarrayh2did_1_argument ();
extern bool_t xdr_dscudamemcpytoarrayd2hid_1_argument ();
extern bool_t xdr_dscudamemcpytoarrayd2did_1_argument ();
extern bool_t xdr_dscudamallocpitchid_1_argument ();
extern bool_t xdr_dscudamemcpy2dtoarrayh2hid_1_argument ();
extern bool_t xdr_dscudamemcpy2dtoarrayh2did_1_argument ();
extern bool_t xdr_dscudamemcpy2dtoarrayd2hid_1_argument ();
extern bool_t xdr_dscudamemcpy2dtoarrayd2did_1_argument ();
extern bool_t xdr_dscudamemcpy2dh2hid_1_argument ();
extern bool_t xdr_dscudamemcpy2dh2did_1_argument ();
extern bool_t xdr_dscudamemcpy2dd2hid_1_argument ();
extern bool_t xdr_dscudamemcpy2dd2did_1_argument ();
extern bool_t xdr_dscudamemset2did_1_argument ();
extern bool_t xdr_dscudamemcpytosymbolasynch2did_1_argument ();
extern bool_t xdr_dscudamemcpytosymbolasyncd2did_1_argument ();
extern bool_t xdr_dscudamemcpyfromsymbolasyncd2hid_1_argument ();
extern bool_t xdr_dscudamemcpyfromsymbolasyncd2did_1_argument ();
extern bool_t xdr_dscudacreatechanneldescid_1_argument ();
extern bool_t xdr_dscudabindtextureid_1_argument ();
extern bool_t xdr_dscudabindtexture2did_1_argument ();
extern bool_t xdr_dscudabindtexturetoarrayid_1_argument ();
extern bool_t xdr_dscufftplan3did_1_argument ();
extern bool_t xdr_dscufftexecc2cid_1_argument ();

#endif 

#ifdef __cplusplus
}
#endif

#endif 
#pragma end dscudarpc.h
#pragma begin dscudadefs.h
#ifndef _DSCUDADEFS_H
#define _DSCUDADEFS_H

#define RC_NSERVERMAX 4    
#define RC_NDEVICEMAX 4    
#define RC_NREDUNDANCYMAX 2 
#define RC_NVDEVMAX 1024      
#define RC_NPTHREADMAX 64   

#define RC_BUFSIZE (1024*1024) 
#define RC_NKMODULEMAX 128  
#define RC_NKFUNCMAX   128  
#define RC_KARGMAX     64   
#define RC_KMODULENAMELEN 64   
#define RC_KNAMELEN       64   
#define RC_KMODULEIMAGELEN (1024*1024*2)   
#define RC_SNAMELEN       64   

#define RC_CACHE_MODULE (1) 
#define RC_CLIENT_CACHE_LIFETIME (30) 
#define RC_SERVER_CACHE_LIFETIME (RC_CLIENT_CACHE_LIFETIME+30) 

#define RC_SUPPORT_PAGELOCK (0)  
#define RC_SUPPORT_STREAM (0)
#define RC_SUPPORT_CONCURRENT_EXEC (0)

#define RC_DAEMON_IP_PORT  (65432)
#define RC_SERVER_IP_PORT  (RC_DAEMON_IP_PORT+1)

#endif 
#pragma end dscudadefs.h
#pragma begin dscudamacros.h
#ifndef DSCUDA_MACROS_H
#define DSCUDA_MACROS_H

#define WARN(lv, fmt, args...) if (lv <= dscudaWarnLevel()) fprintf(stderr, fmt, ## args);
#define WARNONCE(lv, fmt, args...) if (lv <= dscudaWarnLevel()) { \
        static int firstcall = 1;                                 \
        if (firstcall) {                                          \
            firstcall = 0;                                        \
            fprintf(stderr, fmt, ## args);                        \
        }                                                         \
    }

#define ALIGN_UP(off, align) (off) = ((off) + (align) - 1) & ~((align) - 1)
int dscudaWarnLevel(void);
void dscudaSetWarnLevel(int level);

#endif 
#pragma end dscudamacros.h
#pragma begin ibv_rdma.h
#ifndef RDMA_COMMON_H
#define RDMA_COMMON_H

#ifdef RPC_ONLY

typedef struct {
    int type;
    union {
        uint64_t pointerval;
        unsigned int intval;
        float floatval;
        char customval[RC_KARGMAX];
    } val;
    unsigned int offset;
    unsigned int size;
} IbvArg;

#else

#include <netdb.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <rdma/rdma_cma.h>
#include <cuda_runtime_api.h>
#pragma begin dscudadefs.h
#ifndef _DSCUDADEFS_H
#define _DSCUDADEFS_H

#define RC_NSERVERMAX 4    
#define RC_NDEVICEMAX 4    
#define RC_NREDUNDANCYMAX 2 
#define RC_NVDEVMAX 1024      
#define RC_NPTHREADMAX 64   

#define RC_BUFSIZE (1024*1024) 
#define RC_NKMODULEMAX 128  
#define RC_NKFUNCMAX   128  
#define RC_KARGMAX     64   
#define RC_KMODULENAMELEN 64   
#define RC_KNAMELEN       64   
#define RC_KMODULEIMAGELEN (1024*1024*2)   
#define RC_SNAMELEN       64   

#define RC_CACHE_MODULE (1) 
#define RC_CLIENT_CACHE_LIFETIME (30) 
#define RC_SERVER_CACHE_LIFETIME (RC_CLIENT_CACHE_LIFETIME+30) 

#define RC_SUPPORT_PAGELOCK (0)  
#define RC_SUPPORT_STREAM (0)
#define RC_SUPPORT_CONCURRENT_EXEC (0)

#define RC_DAEMON_IP_PORT  (65432)
#define RC_SERVER_IP_PORT  (RC_DAEMON_IP_PORT+1)

#endif 
#pragma end dscudadefs.h
#pragma begin dscudarpc.h


#ifndef _DSCUDARPC_H_RPCGEN
#define _DSCUDARPC_H_RPCGEN

#include <rpc/rpc.h>


#ifdef __cplusplus
extern "C" {
#endif


typedef u_quad_t RCadr;

typedef u_quad_t RCstream;

typedef u_quad_t RCevent;

typedef u_quad_t RCipaddr;

typedef u_int RCsize;

typedef u_int RCerror;

typedef struct {
	u_int RCbuf_len;
	char *RCbuf_val;
} RCbuf;

typedef u_int RCchannelformat;

typedef u_long RCpid;

struct RCchanneldesc_t {
	RCchannelformat f;
	int w;
	int x;
	int y;
	int z;
};
typedef struct RCchanneldesc_t RCchanneldesc_t;

typedef RCchanneldesc_t RCchanneldesc;

struct RCtexture_t {
	int normalized;
	int filterMode;
	int addressMode[3];
	RCchannelformat f;
	int w;
	int x;
	int y;
	int z;
};
typedef struct RCtexture_t RCtexture_t;

typedef RCtexture_t RCtexture;

struct RCfuncattr_t {
	int binaryVersion;
	RCsize constSizeBytes;
	RCsize localSizeBytes;
	int maxThreadsPerBlock;
	int numRegs;
	int ptxVersion;
	RCsize sharedSizeBytes;
};
typedef struct RCfuncattr_t RCfuncattr_t;

typedef RCfuncattr_t RCfuncattr;

enum RCargType {
	dscudaArgTypeP = 0,
	dscudaArgTypeI = 1,
	dscudaArgTypeF = 2,
	dscudaArgTypeV = 3,
};
typedef enum RCargType RCargType;

struct RCargVal {
	RCargType type;
	union {
		RCadr address;
		u_int valuei;
		float valuef;
		char valuev[64];
	} RCargVal_u;
};
typedef struct RCargVal RCargVal;

struct RCarg {
	RCargVal val;
	u_int offset;
	u_int size;
};
typedef struct RCarg RCarg;

typedef struct {
	u_int RCargs_len;
	RCarg *RCargs_val;
} RCargs;

struct dscudaResult {
	RCerror err;
};
typedef struct dscudaResult dscudaResult;

struct dscudaThreadGetLimitResult {
	RCerror err;
	RCsize value;
};
typedef struct dscudaThreadGetLimitResult dscudaThreadGetLimitResult;

struct dscudaThreadGetCacheConfigResult {
	RCerror err;
	int cacheConfig;
};
typedef struct dscudaThreadGetCacheConfigResult dscudaThreadGetCacheConfigResult;

struct dscudaMallocResult {
	RCerror err;
	RCadr devAdr;
};
typedef struct dscudaMallocResult dscudaMallocResult;

struct dscudaHostAllocResult {
	RCerror err;
	RCadr pHost;
};
typedef struct dscudaHostAllocResult dscudaHostAllocResult;

struct dscudaMallocHostResult {
	RCerror err;
	RCadr ptr;
};
typedef struct dscudaMallocHostResult dscudaMallocHostResult;

struct dscudaMallocArrayResult {
	RCerror err;
	RCadr array;
};
typedef struct dscudaMallocArrayResult dscudaMallocArrayResult;

struct dscudaMallocPitchResult {
	RCerror err;
	RCadr devPtr;
	RCsize pitch;
};
typedef struct dscudaMallocPitchResult dscudaMallocPitchResult;

struct dscudaMemcpyD2HResult {
	RCerror err;
	RCbuf buf;
};
typedef struct dscudaMemcpyD2HResult dscudaMemcpyD2HResult;

struct dscudaMemcpyH2HResult {
	RCerror err;
	RCbuf buf;
};
typedef struct dscudaMemcpyH2HResult dscudaMemcpyH2HResult;

struct dscudaMemcpyToArrayD2HResult {
	RCerror err;
	RCbuf buf;
};
typedef struct dscudaMemcpyToArrayD2HResult dscudaMemcpyToArrayD2HResult;

struct dscudaMemcpyToArrayH2HResult {
	RCerror err;
	RCbuf buf;
};
typedef struct dscudaMemcpyToArrayH2HResult dscudaMemcpyToArrayH2HResult;

struct dscudaMemcpy2DToArrayD2HResult {
	RCerror err;
	RCbuf buf;
};
typedef struct dscudaMemcpy2DToArrayD2HResult dscudaMemcpy2DToArrayD2HResult;

struct dscudaMemcpy2DToArrayH2HResult {
	RCerror err;
	RCbuf buf;
};
typedef struct dscudaMemcpy2DToArrayH2HResult dscudaMemcpy2DToArrayH2HResult;

struct dscudaMemcpy2DD2HResult {
	RCerror err;
	RCbuf buf;
};
typedef struct dscudaMemcpy2DD2HResult dscudaMemcpy2DD2HResult;

struct dscudaMemcpy2DH2HResult {
	RCerror err;
	RCbuf buf;
};
typedef struct dscudaMemcpy2DH2HResult dscudaMemcpy2DH2HResult;

struct dscudaGetDeviceResult {
	RCerror err;
	int device;
};
typedef struct dscudaGetDeviceResult dscudaGetDeviceResult;

struct dscudaGetDeviceCountResult {
	RCerror err;
	int count;
};
typedef struct dscudaGetDeviceCountResult dscudaGetDeviceCountResult;

struct dscudaGetDevicePropertiesResult {
	RCerror err;
	RCbuf prop;
};
typedef struct dscudaGetDevicePropertiesResult dscudaGetDevicePropertiesResult;

struct dscudaDriverGetVersionResult {
	RCerror err;
	int ver;
};
typedef struct dscudaDriverGetVersionResult dscudaDriverGetVersionResult;

struct dscudaRuntimeGetVersionResult {
	RCerror err;
	int ver;
};
typedef struct dscudaRuntimeGetVersionResult dscudaRuntimeGetVersionResult;

struct dscudaGetErrorStringResult {
	char *errmsg;
};
typedef struct dscudaGetErrorStringResult dscudaGetErrorStringResult;

struct dscudaCreateChannelDescResult {
	int x;
	int y;
	int z;
	int w;
	RCchannelformat f;
};
typedef struct dscudaCreateChannelDescResult dscudaCreateChannelDescResult;

struct dscudaGetChannelDescResult {
	RCerror err;
	int x;
	int y;
	int z;
	int w;
	RCchannelformat f;
};
typedef struct dscudaGetChannelDescResult dscudaGetChannelDescResult;

struct dscudaChooseDeviceResult {
	RCerror err;
	int device;
};
typedef struct dscudaChooseDeviceResult dscudaChooseDeviceResult;

struct dscudaMemcpyAsyncD2HResult {
	RCerror err;
	RCbuf buf;
};
typedef struct dscudaMemcpyAsyncD2HResult dscudaMemcpyAsyncD2HResult;

struct dscudaMemcpyAsyncH2HResult {
	RCerror err;
	RCbuf buf;
};
typedef struct dscudaMemcpyAsyncH2HResult dscudaMemcpyAsyncH2HResult;

struct dscudaMemcpyFromSymbolD2HResult {
	RCerror err;
	RCbuf buf;
};
typedef struct dscudaMemcpyFromSymbolD2HResult dscudaMemcpyFromSymbolD2HResult;

struct dscudaMemcpyFromSymbolAsyncD2HResult {
	RCerror err;
	RCbuf buf;
};
typedef struct dscudaMemcpyFromSymbolAsyncD2HResult dscudaMemcpyFromSymbolAsyncD2HResult;

struct dscudaStreamCreateResult {
	RCerror err;
	RCadr stream;
};
typedef struct dscudaStreamCreateResult dscudaStreamCreateResult;

struct dscudaEventCreateResult {
	RCerror err;
	RCadr event;
};
typedef struct dscudaEventCreateResult dscudaEventCreateResult;

struct dscudaEventElapsedTimeResult {
	RCerror err;
	float ms;
};
typedef struct dscudaEventElapsedTimeResult dscudaEventElapsedTimeResult;

struct dscudaHostGetDevicePointerResult {
	RCerror err;
	RCadr pDevice;
};
typedef struct dscudaHostGetDevicePointerResult dscudaHostGetDevicePointerResult;

struct dscudaHostGetFlagsResult {
	RCerror err;
	u_int flags;
};
typedef struct dscudaHostGetFlagsResult dscudaHostGetFlagsResult;

struct dscudaLoadModuleResult {
	u_int id;
};
typedef struct dscudaLoadModuleResult dscudaLoadModuleResult;

struct dscudaFuncGetAttributesResult {
	RCerror err;
	RCfuncattr attr;
};
typedef struct dscudaFuncGetAttributesResult dscudaFuncGetAttributesResult;

struct dscudaBindTextureResult {
	RCerror err;
	RCsize offset;
};
typedef struct dscudaBindTextureResult dscudaBindTextureResult;

struct dscudaBindTexture2DResult {
	RCerror err;
	RCsize offset;
};
typedef struct dscudaBindTexture2DResult dscudaBindTexture2DResult;

struct dscufftResult {
	RCerror err;
};
typedef struct dscufftResult dscufftResult;

struct dscufftPlanResult {
	RCerror err;
	u_int plan;
};
typedef struct dscufftPlanResult dscufftPlanResult;

struct dscublasResult {
	RCerror err;
	u_int stat;
};
typedef struct dscublasResult dscublasResult;

struct dscublasCreateResult {
	RCerror err;
	u_int stat;
	RCadr handle;
};
typedef struct dscublasCreateResult dscublasCreateResult;

struct dscublasGetVectorResult {
	RCerror err;
	u_int stat;
	RCbuf y;
};
typedef struct dscublasGetVectorResult dscublasGetVectorResult;

struct RCdim3 {
	u_int x;
	u_int y;
	u_int z;
};
typedef struct RCdim3 RCdim3;

struct dscudathreadsetlimitid_1_argument {
	int limit;
	RCsize value;
};
typedef struct dscudathreadsetlimitid_1_argument dscudathreadsetlimitid_1_argument;

struct dscudastreamwaiteventid_1_argument {
	RCstream stream;
	RCevent event;
	u_int flags;
};
typedef struct dscudastreamwaiteventid_1_argument dscudastreamwaiteventid_1_argument;

struct dscudaeventelapsedtimeid_1_argument {
	RCevent start;
	RCevent end;
};
typedef struct dscudaeventelapsedtimeid_1_argument dscudaeventelapsedtimeid_1_argument;

struct dscudaeventrecordid_1_argument {
	RCevent event;
	RCstream stream;
};
typedef struct dscudaeventrecordid_1_argument dscudaeventrecordid_1_argument;

struct dscudalaunchkernelid_1_argument {
	int moduleid;
	int kid;
	char *kname;
	RCdim3 gdim;
	RCdim3 bdim;
	RCsize smemsize;
	RCstream stream;
	RCargs args;
};
typedef struct dscudalaunchkernelid_1_argument dscudalaunchkernelid_1_argument;

struct dscudaloadmoduleid_1_argument {
	RCipaddr ipaddr;
	RCpid pid;
	char *mname;
	char *image;
};
typedef struct dscudaloadmoduleid_1_argument dscudaloadmoduleid_1_argument;

struct dscudafuncgetattributesid_1_argument {
	int moduleid;
	char *kname;
};
typedef struct dscudafuncgetattributesid_1_argument dscudafuncgetattributesid_1_argument;

struct dscudamemcpyh2hid_1_argument {
	RCadr dst;
	RCbuf src;
	RCsize count;
};
typedef struct dscudamemcpyh2hid_1_argument dscudamemcpyh2hid_1_argument;

struct dscudamemcpyh2did_1_argument {
	RCadr dst;
	RCbuf src;
	RCsize count;
};
typedef struct dscudamemcpyh2did_1_argument dscudamemcpyh2did_1_argument;

struct dscudamemcpyd2hid_1_argument {
	RCadr src;
	RCsize count;
};
typedef struct dscudamemcpyd2hid_1_argument dscudamemcpyd2hid_1_argument;

struct dscudamemcpyd2did_1_argument {
	RCadr dst;
	RCadr src;
	RCsize count;
};
typedef struct dscudamemcpyd2did_1_argument dscudamemcpyd2did_1_argument;

struct dscudamemcpyasynch2hid_1_argument {
	RCadr dst;
	RCbuf src;
	RCsize count;
	RCstream stream;
};
typedef struct dscudamemcpyasynch2hid_1_argument dscudamemcpyasynch2hid_1_argument;

struct dscudamemcpyasynch2did_1_argument {
	RCadr dst;
	RCbuf src;
	RCsize count;
	RCstream stream;
};
typedef struct dscudamemcpyasynch2did_1_argument dscudamemcpyasynch2did_1_argument;

struct dscudamemcpyasyncd2hid_1_argument {
	RCadr src;
	RCsize count;
	RCstream stream;
};
typedef struct dscudamemcpyasyncd2hid_1_argument dscudamemcpyasyncd2hid_1_argument;

struct dscudamemcpyasyncd2did_1_argument {
	RCadr dst;
	RCadr src;
	RCsize count;
	RCstream stream;
};
typedef struct dscudamemcpyasyncd2did_1_argument dscudamemcpyasyncd2did_1_argument;

struct dscudamemcpytosymbolh2did_1_argument {
	int moduleid;
	char *symbol;
	RCbuf src;
	RCsize count;
	RCsize offset;
};
typedef struct dscudamemcpytosymbolh2did_1_argument dscudamemcpytosymbolh2did_1_argument;

struct dscudamemcpytosymbold2did_1_argument {
	int moduleid;
	char *symbol;
	RCadr src;
	RCsize count;
	RCsize offset;
};
typedef struct dscudamemcpytosymbold2did_1_argument dscudamemcpytosymbold2did_1_argument;

struct dscudamemcpyfromsymbold2hid_1_argument {
	int moduleid;
	char *symbol;
	RCsize count;
	RCsize offset;
};
typedef struct dscudamemcpyfromsymbold2hid_1_argument dscudamemcpyfromsymbold2hid_1_argument;

struct dscudamemcpyfromsymbold2did_1_argument {
	int moduleid;
	RCadr dst;
	char *symbol;
	RCsize count;
	RCsize offset;
};
typedef struct dscudamemcpyfromsymbold2did_1_argument dscudamemcpyfromsymbold2did_1_argument;

struct dscudamemsetid_1_argument {
	RCadr dst;
	int value;
	RCsize count;
};
typedef struct dscudamemsetid_1_argument dscudamemsetid_1_argument;

struct dscudahostallocid_1_argument {
	RCsize size;
	u_int flags;
};
typedef struct dscudahostallocid_1_argument dscudahostallocid_1_argument;

struct dscudahostgetdevicepointerid_1_argument {
	RCadr pHost;
	u_int flags;
};
typedef struct dscudahostgetdevicepointerid_1_argument dscudahostgetdevicepointerid_1_argument;

struct dscudamallocarrayid_1_argument {
	RCchanneldesc desc;
	RCsize width;
	RCsize height;
	u_int flags;
};
typedef struct dscudamallocarrayid_1_argument dscudamallocarrayid_1_argument;

struct dscudamemcpytoarrayh2hid_1_argument {
	RCadr dst;
	RCsize wOffset;
	RCsize hOffset;
	RCbuf src;
	RCsize count;
};
typedef struct dscudamemcpytoarrayh2hid_1_argument dscudamemcpytoarrayh2hid_1_argument;

struct dscudamemcpytoarrayh2did_1_argument {
	RCadr dst;
	RCsize wOffset;
	RCsize hOffset;
	RCbuf src;
	RCsize count;
};
typedef struct dscudamemcpytoarrayh2did_1_argument dscudamemcpytoarrayh2did_1_argument;

struct dscudamemcpytoarrayd2hid_1_argument {
	RCsize wOffset;
	RCsize hOffset;
	RCadr src;
	RCsize count;
};
typedef struct dscudamemcpytoarrayd2hid_1_argument dscudamemcpytoarrayd2hid_1_argument;

struct dscudamemcpytoarrayd2did_1_argument {
	RCadr dst;
	RCsize wOffset;
	RCsize hOffset;
	RCadr src;
	RCsize count;
};
typedef struct dscudamemcpytoarrayd2did_1_argument dscudamemcpytoarrayd2did_1_argument;

struct dscudamallocpitchid_1_argument {
	RCsize width;
	RCsize height;
};
typedef struct dscudamallocpitchid_1_argument dscudamallocpitchid_1_argument;

struct dscudamemcpy2dtoarrayh2hid_1_argument {
	RCadr dst;
	RCsize wOffset;
	RCsize hOffset;
	RCbuf src;
	RCsize spitch;
	RCsize width;
	RCsize height;
};
typedef struct dscudamemcpy2dtoarrayh2hid_1_argument dscudamemcpy2dtoarrayh2hid_1_argument;

struct dscudamemcpy2dtoarrayh2did_1_argument {
	RCadr dst;
	RCsize wOffset;
	RCsize hOffset;
	RCbuf srcbuf;
	RCsize spitch;
	RCsize width;
	RCsize height;
};
typedef struct dscudamemcpy2dtoarrayh2did_1_argument dscudamemcpy2dtoarrayh2did_1_argument;

struct dscudamemcpy2dtoarrayd2hid_1_argument {
	RCsize wOffset;
	RCsize hOffset;
	RCadr src;
	RCsize spitch;
	RCsize width;
	RCsize height;
};
typedef struct dscudamemcpy2dtoarrayd2hid_1_argument dscudamemcpy2dtoarrayd2hid_1_argument;

struct dscudamemcpy2dtoarrayd2did_1_argument {
	RCadr dst;
	RCsize wOffset;
	RCsize hOffset;
	RCadr src;
	RCsize spitch;
	RCsize width;
	RCsize height;
};
typedef struct dscudamemcpy2dtoarrayd2did_1_argument dscudamemcpy2dtoarrayd2did_1_argument;

struct dscudamemcpy2dh2hid_1_argument {
	RCadr dst;
	RCsize dpitch;
	RCbuf src;
	RCsize spitch;
	RCsize width;
	RCsize height;
};
typedef struct dscudamemcpy2dh2hid_1_argument dscudamemcpy2dh2hid_1_argument;

struct dscudamemcpy2dh2did_1_argument {
	RCadr dst;
	RCsize dpitch;
	RCbuf src;
	RCsize spitch;
	RCsize width;
	RCsize height;
};
typedef struct dscudamemcpy2dh2did_1_argument dscudamemcpy2dh2did_1_argument;

struct dscudamemcpy2dd2hid_1_argument {
	RCsize dpitch;
	RCadr src;
	RCsize spitch;
	RCsize width;
	RCsize height;
};
typedef struct dscudamemcpy2dd2hid_1_argument dscudamemcpy2dd2hid_1_argument;

struct dscudamemcpy2dd2did_1_argument {
	RCadr dst;
	RCsize dpitch;
	RCadr src;
	RCsize spitch;
	RCsize width;
	RCsize height;
};
typedef struct dscudamemcpy2dd2did_1_argument dscudamemcpy2dd2did_1_argument;

struct dscudamemset2did_1_argument {
	RCadr dst;
	RCsize pitch;
	int value;
	RCsize width;
	RCsize height;
};
typedef struct dscudamemset2did_1_argument dscudamemset2did_1_argument;

struct dscudamemcpytosymbolasynch2did_1_argument {
	int moduleid;
	char *symbol;
	RCbuf src;
	RCsize count;
	RCsize offset;
	RCstream stream;
};
typedef struct dscudamemcpytosymbolasynch2did_1_argument dscudamemcpytosymbolasynch2did_1_argument;

struct dscudamemcpytosymbolasyncd2did_1_argument {
	int moduleid;
	char *symbol;
	RCadr src;
	RCsize count;
	RCsize offset;
	RCstream stream;
};
typedef struct dscudamemcpytosymbolasyncd2did_1_argument dscudamemcpytosymbolasyncd2did_1_argument;

struct dscudamemcpyfromsymbolasyncd2hid_1_argument {
	int moduleid;
	char *symbol;
	RCsize count;
	RCsize offset;
	RCstream stream;
};
typedef struct dscudamemcpyfromsymbolasyncd2hid_1_argument dscudamemcpyfromsymbolasyncd2hid_1_argument;

struct dscudamemcpyfromsymbolasyncd2did_1_argument {
	int moduleid;
	RCadr dst;
	char *symbol;
	RCsize count;
	RCsize offset;
	RCstream stream;
};
typedef struct dscudamemcpyfromsymbolasyncd2did_1_argument dscudamemcpyfromsymbolasyncd2did_1_argument;

struct dscudacreatechanneldescid_1_argument {
	int x;
	int y;
	int z;
	int w;
	RCchannelformat f;
};
typedef struct dscudacreatechanneldescid_1_argument dscudacreatechanneldescid_1_argument;

struct dscudabindtextureid_1_argument {
	int moduleid;
	char *texname;
	RCadr devPtr;
	RCsize size;
	RCtexture texbuf;
};
typedef struct dscudabindtextureid_1_argument dscudabindtextureid_1_argument;

struct dscudabindtexture2did_1_argument {
	int moduleid;
	char *texname;
	RCadr devPtr;
	RCsize width;
	RCsize height;
	RCsize pitch;
	RCtexture texbuf;
};
typedef struct dscudabindtexture2did_1_argument dscudabindtexture2did_1_argument;

struct dscudabindtexturetoarrayid_1_argument {
	int moduleid;
	char *texname;
	RCadr array;
	RCtexture texbuf;
};
typedef struct dscudabindtexturetoarrayid_1_argument dscudabindtexturetoarrayid_1_argument;

struct dscufftplan3did_1_argument {
	int nx;
	int ny;
	int nz;
	u_int type;
};
typedef struct dscufftplan3did_1_argument dscufftplan3did_1_argument;

struct dscufftexecc2cid_1_argument {
	u_int plan;
	RCadr idata;
	RCadr odata;
	int direction;
};
typedef struct dscufftexecc2cid_1_argument dscufftexecc2cid_1_argument;

#define DSCUDA_PROG 60000
#define DSCUDA_VER 1

#if defined(__STDC__) || defined(__cplusplus)
#define dscudaThreadExitId 100
extern  dscudaResult * dscudathreadexitid_1(CLIENT *);
extern  dscudaResult * dscudathreadexitid_1_svc(struct svc_req *);
#define dscudaThreadSynchronizeId 101
extern  dscudaResult * dscudathreadsynchronizeid_1(CLIENT *);
extern  dscudaResult * dscudathreadsynchronizeid_1_svc(struct svc_req *);
#define dscudaThreadSetLimitId 102
extern  dscudaResult * dscudathreadsetlimitid_1(int , RCsize , CLIENT *);
extern  dscudaResult * dscudathreadsetlimitid_1_svc(int , RCsize , struct svc_req *);
#define dscudaThreadGetLimitId 103
extern  dscudaThreadGetLimitResult * dscudathreadgetlimitid_1(int , CLIENT *);
extern  dscudaThreadGetLimitResult * dscudathreadgetlimitid_1_svc(int , struct svc_req *);
#define dscudaThreadSetCacheConfigId 104
extern  dscudaResult * dscudathreadsetcacheconfigid_1(int , CLIENT *);
extern  dscudaResult * dscudathreadsetcacheconfigid_1_svc(int , struct svc_req *);
#define dscudaThreadGetCacheConfigId 105
extern  dscudaThreadGetCacheConfigResult * dscudathreadgetcacheconfigid_1(CLIENT *);
extern  dscudaThreadGetCacheConfigResult * dscudathreadgetcacheconfigid_1_svc(struct svc_req *);
#define dscudaGetLastErrorId 200
extern  dscudaResult * dscudagetlasterrorid_1(CLIENT *);
extern  dscudaResult * dscudagetlasterrorid_1_svc(struct svc_req *);
#define dscudaPeekAtLastErrorId 201
extern  dscudaResult * dscudapeekatlasterrorid_1(CLIENT *);
extern  dscudaResult * dscudapeekatlasterrorid_1_svc(struct svc_req *);
#define dscudaGetErrorStringId 202
extern  dscudaGetErrorStringResult * dscudageterrorstringid_1(int , CLIENT *);
extern  dscudaGetErrorStringResult * dscudageterrorstringid_1_svc(int , struct svc_req *);
#define dscudaGetDeviceId 300
extern  dscudaGetDeviceResult * dscudagetdeviceid_1(CLIENT *);
extern  dscudaGetDeviceResult * dscudagetdeviceid_1_svc(struct svc_req *);
#define dscudaGetDeviceCountId 301
extern  dscudaGetDeviceCountResult * dscudagetdevicecountid_1(CLIENT *);
extern  dscudaGetDeviceCountResult * dscudagetdevicecountid_1_svc(struct svc_req *);
#define dscudaGetDevicePropertiesId 302
extern  dscudaGetDevicePropertiesResult * dscudagetdevicepropertiesid_1(int , CLIENT *);
extern  dscudaGetDevicePropertiesResult * dscudagetdevicepropertiesid_1_svc(int , struct svc_req *);
#define dscudaDriverGetVersionId 303
extern  dscudaDriverGetVersionResult * dscudadrivergetversionid_1(CLIENT *);
extern  dscudaDriverGetVersionResult * dscudadrivergetversionid_1_svc(struct svc_req *);
#define dscudaRuntimeGetVersionId 304
extern  dscudaRuntimeGetVersionResult * dscudaruntimegetversionid_1(CLIENT *);
extern  dscudaRuntimeGetVersionResult * dscudaruntimegetversionid_1_svc(struct svc_req *);
#define dscudaSetDeviceId 305
extern  dscudaResult * dscudasetdeviceid_1(int , CLIENT *);
extern  dscudaResult * dscudasetdeviceid_1_svc(int , struct svc_req *);
#define dscudaSetDeviceFlagsId 306
extern  dscudaResult * dscudasetdeviceflagsid_1(u_int , CLIENT *);
extern  dscudaResult * dscudasetdeviceflagsid_1_svc(u_int , struct svc_req *);
#define dscudaChooseDeviceId 307
extern  dscudaChooseDeviceResult * dscudachoosedeviceid_1(RCbuf , CLIENT *);
extern  dscudaChooseDeviceResult * dscudachoosedeviceid_1_svc(RCbuf , struct svc_req *);
#define dscudaDeviceSynchronize 308
extern  dscudaResult * dscudadevicesynchronize_1(CLIENT *);
extern  dscudaResult * dscudadevicesynchronize_1_svc(struct svc_req *);
#define dscudaDeviceReset 309
extern  dscudaResult * dscudadevicereset_1(CLIENT *);
extern  dscudaResult * dscudadevicereset_1_svc(struct svc_req *);
#define dscudaStreamCreateId 400
extern  dscudaStreamCreateResult * dscudastreamcreateid_1(CLIENT *);
extern  dscudaStreamCreateResult * dscudastreamcreateid_1_svc(struct svc_req *);
#define dscudaStreamDestroyId 401
extern  dscudaResult * dscudastreamdestroyid_1(RCstream , CLIENT *);
extern  dscudaResult * dscudastreamdestroyid_1_svc(RCstream , struct svc_req *);
#define dscudaStreamSynchronizeId 402
extern  dscudaResult * dscudastreamsynchronizeid_1(RCstream , CLIENT *);
extern  dscudaResult * dscudastreamsynchronizeid_1_svc(RCstream , struct svc_req *);
#define dscudaStreamQueryId 403
extern  dscudaResult * dscudastreamqueryid_1(RCstream , CLIENT *);
extern  dscudaResult * dscudastreamqueryid_1_svc(RCstream , struct svc_req *);
#define dscudaStreamWaitEventId 404
extern  dscudaResult * dscudastreamwaiteventid_1(RCstream , RCevent , u_int , CLIENT *);
extern  dscudaResult * dscudastreamwaiteventid_1_svc(RCstream , RCevent , u_int , struct svc_req *);
#define dscudaEventCreateId 500
extern  dscudaEventCreateResult * dscudaeventcreateid_1(CLIENT *);
extern  dscudaEventCreateResult * dscudaeventcreateid_1_svc(struct svc_req *);
#define dscudaEventCreateWithFlagsId 501
extern  dscudaEventCreateResult * dscudaeventcreatewithflagsid_1(u_int , CLIENT *);
extern  dscudaEventCreateResult * dscudaeventcreatewithflagsid_1_svc(u_int , struct svc_req *);
#define dscudaEventDestroyId 502
extern  dscudaResult * dscudaeventdestroyid_1(RCevent , CLIENT *);
extern  dscudaResult * dscudaeventdestroyid_1_svc(RCevent , struct svc_req *);
#define dscudaEventElapsedTimeId 503
extern  dscudaEventElapsedTimeResult * dscudaeventelapsedtimeid_1(RCevent , RCevent , CLIENT *);
extern  dscudaEventElapsedTimeResult * dscudaeventelapsedtimeid_1_svc(RCevent , RCevent , struct svc_req *);
#define dscudaEventRecordId 504
extern  dscudaResult * dscudaeventrecordid_1(RCevent , RCstream , CLIENT *);
extern  dscudaResult * dscudaeventrecordid_1_svc(RCevent , RCstream , struct svc_req *);
#define dscudaEventSynchronizeId 505
extern  dscudaResult * dscudaeventsynchronizeid_1(RCevent , CLIENT *);
extern  dscudaResult * dscudaeventsynchronizeid_1_svc(RCevent , struct svc_req *);
#define dscudaEventQueryId 506
extern  dscudaResult * dscudaeventqueryid_1(RCevent , CLIENT *);
extern  dscudaResult * dscudaeventqueryid_1_svc(RCevent , struct svc_req *);
#define dscudaLaunchKernelId 600
extern  void * dscudalaunchkernelid_1(int , int , char *, RCdim3 , RCdim3 , RCsize , RCstream , RCargs , CLIENT *);
extern  void * dscudalaunchkernelid_1_svc(int , int , char *, RCdim3 , RCdim3 , RCsize , RCstream , RCargs , struct svc_req *);
#define dscudaLoadModuleId 601
extern  dscudaLoadModuleResult * dscudaloadmoduleid_1(RCipaddr , RCpid , char *, char *, CLIENT *);
extern  dscudaLoadModuleResult * dscudaloadmoduleid_1_svc(RCipaddr , RCpid , char *, char *, struct svc_req *);
#define dscudaFuncGetAttributesId 602
extern  dscudaFuncGetAttributesResult * dscudafuncgetattributesid_1(int , char *, CLIENT *);
extern  dscudaFuncGetAttributesResult * dscudafuncgetattributesid_1_svc(int , char *, struct svc_req *);
#define dscudaMallocId 700
extern  dscudaMallocResult * dscudamallocid_1(RCsize , CLIENT *);
extern  dscudaMallocResult * dscudamallocid_1_svc(RCsize , struct svc_req *);
#define dscudaFreeId 701
extern  dscudaResult * dscudafreeid_1(RCadr , CLIENT *);
extern  dscudaResult * dscudafreeid_1_svc(RCadr , struct svc_req *);
#define dscudaMemcpyH2HId 702
extern  dscudaMemcpyH2HResult * dscudamemcpyh2hid_1(RCadr , RCbuf , RCsize , CLIENT *);
extern  dscudaMemcpyH2HResult * dscudamemcpyh2hid_1_svc(RCadr , RCbuf , RCsize , struct svc_req *);
#define dscudaMemcpyH2DId 703
extern  dscudaResult * dscudamemcpyh2did_1(RCadr , RCbuf , RCsize , CLIENT *);
extern  dscudaResult * dscudamemcpyh2did_1_svc(RCadr , RCbuf , RCsize , struct svc_req *);
#define dscudaMemcpyD2HId 704
extern  dscudaMemcpyD2HResult * dscudamemcpyd2hid_1(RCadr , RCsize , CLIENT *);
extern  dscudaMemcpyD2HResult * dscudamemcpyd2hid_1_svc(RCadr , RCsize , struct svc_req *);
#define dscudaMemcpyD2DId 705
extern  dscudaResult * dscudamemcpyd2did_1(RCadr , RCadr , RCsize , CLIENT *);
extern  dscudaResult * dscudamemcpyd2did_1_svc(RCadr , RCadr , RCsize , struct svc_req *);
#define dscudaMemcpyAsyncH2HId 706
extern  dscudaMemcpyAsyncH2HResult * dscudamemcpyasynch2hid_1(RCadr , RCbuf , RCsize , RCstream , CLIENT *);
extern  dscudaMemcpyAsyncH2HResult * dscudamemcpyasynch2hid_1_svc(RCadr , RCbuf , RCsize , RCstream , struct svc_req *);
#define dscudaMemcpyAsyncH2DId 707
extern  dscudaResult * dscudamemcpyasynch2did_1(RCadr , RCbuf , RCsize , RCstream , CLIENT *);
extern  dscudaResult * dscudamemcpyasynch2did_1_svc(RCadr , RCbuf , RCsize , RCstream , struct svc_req *);
#define dscudaMemcpyAsyncD2HId 708
extern  dscudaMemcpyAsyncD2HResult * dscudamemcpyasyncd2hid_1(RCadr , RCsize , RCstream , CLIENT *);
extern  dscudaMemcpyAsyncD2HResult * dscudamemcpyasyncd2hid_1_svc(RCadr , RCsize , RCstream , struct svc_req *);
#define dscudaMemcpyAsyncD2DId 709
extern  dscudaResult * dscudamemcpyasyncd2did_1(RCadr , RCadr , RCsize , RCstream , CLIENT *);
extern  dscudaResult * dscudamemcpyasyncd2did_1_svc(RCadr , RCadr , RCsize , RCstream , struct svc_req *);
#define dscudaMemcpyToSymbolH2DId 710
extern  dscudaResult * dscudamemcpytosymbolh2did_1(int , char *, RCbuf , RCsize , RCsize , CLIENT *);
extern  dscudaResult * dscudamemcpytosymbolh2did_1_svc(int , char *, RCbuf , RCsize , RCsize , struct svc_req *);
#define dscudaMemcpyToSymbolD2DId 711
extern  dscudaResult * dscudamemcpytosymbold2did_1(int , char *, RCadr , RCsize , RCsize , CLIENT *);
extern  dscudaResult * dscudamemcpytosymbold2did_1_svc(int , char *, RCadr , RCsize , RCsize , struct svc_req *);
#define dscudaMemcpyFromSymbolD2HId 712
extern  dscudaMemcpyFromSymbolD2HResult * dscudamemcpyfromsymbold2hid_1(int , char *, RCsize , RCsize , CLIENT *);
extern  dscudaMemcpyFromSymbolD2HResult * dscudamemcpyfromsymbold2hid_1_svc(int , char *, RCsize , RCsize , struct svc_req *);
#define dscudaMemcpyFromSymbolD2DId 713
extern  dscudaResult * dscudamemcpyfromsymbold2did_1(int , RCadr , char *, RCsize , RCsize , CLIENT *);
extern  dscudaResult * dscudamemcpyfromsymbold2did_1_svc(int , RCadr , char *, RCsize , RCsize , struct svc_req *);
#define dscudaMemsetId 714
extern  dscudaResult * dscudamemsetid_1(RCadr , int , RCsize , CLIENT *);
extern  dscudaResult * dscudamemsetid_1_svc(RCadr , int , RCsize , struct svc_req *);
#define dscudaHostAllocId 715
extern  dscudaHostAllocResult * dscudahostallocid_1(RCsize , u_int , CLIENT *);
extern  dscudaHostAllocResult * dscudahostallocid_1_svc(RCsize , u_int , struct svc_req *);
#define dscudaMallocHostId 716
extern  dscudaMallocHostResult * dscudamallochostid_1(RCsize , CLIENT *);
extern  dscudaMallocHostResult * dscudamallochostid_1_svc(RCsize , struct svc_req *);
#define dscudaFreeHostId 717
extern  dscudaResult * dscudafreehostid_1(RCadr , CLIENT *);
extern  dscudaResult * dscudafreehostid_1_svc(RCadr , struct svc_req *);
#define dscudaHostGetDevicePointerId 718
extern  dscudaHostGetDevicePointerResult * dscudahostgetdevicepointerid_1(RCadr , u_int , CLIENT *);
extern  dscudaHostGetDevicePointerResult * dscudahostgetdevicepointerid_1_svc(RCadr , u_int , struct svc_req *);
#define dscudaHostGetFlagsID 719
extern  dscudaHostGetFlagsResult * dscudahostgetflagsid_1(RCadr , CLIENT *);
extern  dscudaHostGetFlagsResult * dscudahostgetflagsid_1_svc(RCadr , struct svc_req *);
#define dscudaMallocArrayId 720
extern  dscudaMallocArrayResult * dscudamallocarrayid_1(RCchanneldesc , RCsize , RCsize , u_int , CLIENT *);
extern  dscudaMallocArrayResult * dscudamallocarrayid_1_svc(RCchanneldesc , RCsize , RCsize , u_int , struct svc_req *);
#define dscudaFreeArrayId 721
extern  dscudaResult * dscudafreearrayid_1(RCadr , CLIENT *);
extern  dscudaResult * dscudafreearrayid_1_svc(RCadr , struct svc_req *);
#define dscudaMemcpyToArrayH2HId 722
extern  dscudaMemcpyToArrayH2HResult * dscudamemcpytoarrayh2hid_1(RCadr , RCsize , RCsize , RCbuf , RCsize , CLIENT *);
extern  dscudaMemcpyToArrayH2HResult * dscudamemcpytoarrayh2hid_1_svc(RCadr , RCsize , RCsize , RCbuf , RCsize , struct svc_req *);
#define dscudaMemcpyToArrayH2DId 723
extern  dscudaResult * dscudamemcpytoarrayh2did_1(RCadr , RCsize , RCsize , RCbuf , RCsize , CLIENT *);
extern  dscudaResult * dscudamemcpytoarrayh2did_1_svc(RCadr , RCsize , RCsize , RCbuf , RCsize , struct svc_req *);
#define dscudaMemcpyToArrayD2HId 724
extern  dscudaMemcpyToArrayD2HResult * dscudamemcpytoarrayd2hid_1(RCsize , RCsize , RCadr , RCsize , CLIENT *);
extern  dscudaMemcpyToArrayD2HResult * dscudamemcpytoarrayd2hid_1_svc(RCsize , RCsize , RCadr , RCsize , struct svc_req *);
#define dscudaMemcpyToArrayD2DId 725
extern  dscudaResult * dscudamemcpytoarrayd2did_1(RCadr , RCsize , RCsize , RCadr , RCsize , CLIENT *);
extern  dscudaResult * dscudamemcpytoarrayd2did_1_svc(RCadr , RCsize , RCsize , RCadr , RCsize , struct svc_req *);
#define dscudaMallocPitchId 726
extern  dscudaMallocPitchResult * dscudamallocpitchid_1(RCsize , RCsize , CLIENT *);
extern  dscudaMallocPitchResult * dscudamallocpitchid_1_svc(RCsize , RCsize , struct svc_req *);
#define dscudaMemcpy2DToArrayH2HId 727
extern  dscudaMemcpy2DToArrayH2HResult * dscudamemcpy2dtoarrayh2hid_1(RCadr , RCsize , RCsize , RCbuf , RCsize , RCsize , RCsize , CLIENT *);
extern  dscudaMemcpy2DToArrayH2HResult * dscudamemcpy2dtoarrayh2hid_1_svc(RCadr , RCsize , RCsize , RCbuf , RCsize , RCsize , RCsize , struct svc_req *);
#define dscudaMemcpy2DToArrayH2DId 728
extern  dscudaResult * dscudamemcpy2dtoarrayh2did_1(RCadr , RCsize , RCsize , RCbuf , RCsize , RCsize , RCsize , CLIENT *);
extern  dscudaResult * dscudamemcpy2dtoarrayh2did_1_svc(RCadr , RCsize , RCsize , RCbuf , RCsize , RCsize , RCsize , struct svc_req *);
#define dscudaMemcpy2DToArrayD2HId 729
extern  dscudaMemcpy2DToArrayD2HResult * dscudamemcpy2dtoarrayd2hid_1(RCsize , RCsize , RCadr , RCsize , RCsize , RCsize , CLIENT *);
extern  dscudaMemcpy2DToArrayD2HResult * dscudamemcpy2dtoarrayd2hid_1_svc(RCsize , RCsize , RCadr , RCsize , RCsize , RCsize , struct svc_req *);
#define dscudaMemcpy2DToArrayD2DId 730
extern  dscudaResult * dscudamemcpy2dtoarrayd2did_1(RCadr , RCsize , RCsize , RCadr , RCsize , RCsize , RCsize , CLIENT *);
extern  dscudaResult * dscudamemcpy2dtoarrayd2did_1_svc(RCadr , RCsize , RCsize , RCadr , RCsize , RCsize , RCsize , struct svc_req *);
#define dscudaMemcpy2DH2HId 731
extern  dscudaMemcpy2DH2HResult * dscudamemcpy2dh2hid_1(RCadr , RCsize , RCbuf , RCsize , RCsize , RCsize , CLIENT *);
extern  dscudaMemcpy2DH2HResult * dscudamemcpy2dh2hid_1_svc(RCadr , RCsize , RCbuf , RCsize , RCsize , RCsize , struct svc_req *);
#define dscudaMemcpy2DH2DId 732
extern  dscudaResult * dscudamemcpy2dh2did_1(RCadr , RCsize , RCbuf , RCsize , RCsize , RCsize , CLIENT *);
extern  dscudaResult * dscudamemcpy2dh2did_1_svc(RCadr , RCsize , RCbuf , RCsize , RCsize , RCsize , struct svc_req *);
#define dscudaMemcpy2DD2HId 733
extern  dscudaMemcpy2DD2HResult * dscudamemcpy2dd2hid_1(RCsize , RCadr , RCsize , RCsize , RCsize , CLIENT *);
extern  dscudaMemcpy2DD2HResult * dscudamemcpy2dd2hid_1_svc(RCsize , RCadr , RCsize , RCsize , RCsize , struct svc_req *);
#define dscudaMemcpy2DD2DId 734
extern  dscudaResult * dscudamemcpy2dd2did_1(RCadr , RCsize , RCadr , RCsize , RCsize , RCsize , CLIENT *);
extern  dscudaResult * dscudamemcpy2dd2did_1_svc(RCadr , RCsize , RCadr , RCsize , RCsize , RCsize , struct svc_req *);
#define dscudaMemset2DId 735
extern  dscudaResult * dscudamemset2did_1(RCadr , RCsize , int , RCsize , RCsize , CLIENT *);
extern  dscudaResult * dscudamemset2did_1_svc(RCadr , RCsize , int , RCsize , RCsize , struct svc_req *);
#define dscudaMemcpyToSymbolAsyncH2DId 736
extern  dscudaResult * dscudamemcpytosymbolasynch2did_1(int , char *, RCbuf , RCsize , RCsize , RCstream , CLIENT *);
extern  dscudaResult * dscudamemcpytosymbolasynch2did_1_svc(int , char *, RCbuf , RCsize , RCsize , RCstream , struct svc_req *);
#define dscudaMemcpyToSymbolAsyncD2DId 737
extern  dscudaResult * dscudamemcpytosymbolasyncd2did_1(int , char *, RCadr , RCsize , RCsize , RCstream , CLIENT *);
extern  dscudaResult * dscudamemcpytosymbolasyncd2did_1_svc(int , char *, RCadr , RCsize , RCsize , RCstream , struct svc_req *);
#define dscudaMemcpyFromSymbolAsyncD2HId 738
extern  dscudaMemcpyFromSymbolAsyncD2HResult * dscudamemcpyfromsymbolasyncd2hid_1(int , char *, RCsize , RCsize , RCstream , CLIENT *);
extern  dscudaMemcpyFromSymbolAsyncD2HResult * dscudamemcpyfromsymbolasyncd2hid_1_svc(int , char *, RCsize , RCsize , RCstream , struct svc_req *);
#define dscudaMemcpyFromSymbolAsyncD2DId 739
extern  dscudaResult * dscudamemcpyfromsymbolasyncd2did_1(int , RCadr , char *, RCsize , RCsize , RCstream , CLIENT *);
extern  dscudaResult * dscudamemcpyfromsymbolasyncd2did_1_svc(int , RCadr , char *, RCsize , RCsize , RCstream , struct svc_req *);
#define dscudaCreateChannelDescId 1400
extern  dscudaCreateChannelDescResult * dscudacreatechanneldescid_1(int , int , int , int , RCchannelformat , CLIENT *);
extern  dscudaCreateChannelDescResult * dscudacreatechanneldescid_1_svc(int , int , int , int , RCchannelformat , struct svc_req *);
#define dscudaGetChannelDescId 1401
extern  dscudaGetChannelDescResult * dscudagetchanneldescid_1(RCadr , CLIENT *);
extern  dscudaGetChannelDescResult * dscudagetchanneldescid_1_svc(RCadr , struct svc_req *);
#define dscudaBindTextureId 1402
extern  dscudaBindTextureResult * dscudabindtextureid_1(int , char *, RCadr , RCsize , RCtexture , CLIENT *);
extern  dscudaBindTextureResult * dscudabindtextureid_1_svc(int , char *, RCadr , RCsize , RCtexture , struct svc_req *);
#define dscudaBindTexture2DId 1403
extern  dscudaBindTexture2DResult * dscudabindtexture2did_1(int , char *, RCadr , RCsize , RCsize , RCsize , RCtexture , CLIENT *);
extern  dscudaBindTexture2DResult * dscudabindtexture2did_1_svc(int , char *, RCadr , RCsize , RCsize , RCsize , RCtexture , struct svc_req *);
#define dscudaBindTextureToArrayId 1404
extern  dscudaResult * dscudabindtexturetoarrayid_1(int , char *, RCadr , RCtexture , CLIENT *);
extern  dscudaResult * dscudabindtexturetoarrayid_1_svc(int , char *, RCadr , RCtexture , struct svc_req *);
#define dscudaUnbindTextureId 1405
extern  dscudaResult * dscudaunbindtextureid_1(RCtexture , CLIENT *);
extern  dscudaResult * dscudaunbindtextureid_1_svc(RCtexture , struct svc_req *);
#define dscufftPlan3dId 2002
extern  dscufftPlanResult * dscufftplan3did_1(int , int , int , u_int , CLIENT *);
extern  dscufftPlanResult * dscufftplan3did_1_svc(int , int , int , u_int , struct svc_req *);
#define dscufftDestroyId 2004
extern  dscufftResult * dscufftdestroyid_1(u_int , CLIENT *);
extern  dscufftResult * dscufftdestroyid_1_svc(u_int , struct svc_req *);
#define dscufftExecC2CId 2005
extern  dscufftResult * dscufftexecc2cid_1(u_int , RCadr , RCadr , int , CLIENT *);
extern  dscufftResult * dscufftexecc2cid_1_svc(u_int , RCadr , RCadr , int , struct svc_req *);
extern int dscuda_prog_1_freeresult (SVCXPRT *, xdrproc_t, caddr_t);

#else 
#define dscudaThreadExitId 100
extern  dscudaResult * dscudathreadexitid_1();
extern  dscudaResult * dscudathreadexitid_1_svc();
#define dscudaThreadSynchronizeId 101
extern  dscudaResult * dscudathreadsynchronizeid_1();
extern  dscudaResult * dscudathreadsynchronizeid_1_svc();
#define dscudaThreadSetLimitId 102
extern  dscudaResult * dscudathreadsetlimitid_1();
extern  dscudaResult * dscudathreadsetlimitid_1_svc();
#define dscudaThreadGetLimitId 103
extern  dscudaThreadGetLimitResult * dscudathreadgetlimitid_1();
extern  dscudaThreadGetLimitResult * dscudathreadgetlimitid_1_svc();
#define dscudaThreadSetCacheConfigId 104
extern  dscudaResult * dscudathreadsetcacheconfigid_1();
extern  dscudaResult * dscudathreadsetcacheconfigid_1_svc();
#define dscudaThreadGetCacheConfigId 105
extern  dscudaThreadGetCacheConfigResult * dscudathreadgetcacheconfigid_1();
extern  dscudaThreadGetCacheConfigResult * dscudathreadgetcacheconfigid_1_svc();
#define dscudaGetLastErrorId 200
extern  dscudaResult * dscudagetlasterrorid_1();
extern  dscudaResult * dscudagetlasterrorid_1_svc();
#define dscudaPeekAtLastErrorId 201
extern  dscudaResult * dscudapeekatlasterrorid_1();
extern  dscudaResult * dscudapeekatlasterrorid_1_svc();
#define dscudaGetErrorStringId 202
extern  dscudaGetErrorStringResult * dscudageterrorstringid_1();
extern  dscudaGetErrorStringResult * dscudageterrorstringid_1_svc();
#define dscudaGetDeviceId 300
extern  dscudaGetDeviceResult * dscudagetdeviceid_1();
extern  dscudaGetDeviceResult * dscudagetdeviceid_1_svc();
#define dscudaGetDeviceCountId 301
extern  dscudaGetDeviceCountResult * dscudagetdevicecountid_1();
extern  dscudaGetDeviceCountResult * dscudagetdevicecountid_1_svc();
#define dscudaGetDevicePropertiesId 302
extern  dscudaGetDevicePropertiesResult * dscudagetdevicepropertiesid_1();
extern  dscudaGetDevicePropertiesResult * dscudagetdevicepropertiesid_1_svc();
#define dscudaDriverGetVersionId 303
extern  dscudaDriverGetVersionResult * dscudadrivergetversionid_1();
extern  dscudaDriverGetVersionResult * dscudadrivergetversionid_1_svc();
#define dscudaRuntimeGetVersionId 304
extern  dscudaRuntimeGetVersionResult * dscudaruntimegetversionid_1();
extern  dscudaRuntimeGetVersionResult * dscudaruntimegetversionid_1_svc();
#define dscudaSetDeviceId 305
extern  dscudaResult * dscudasetdeviceid_1();
extern  dscudaResult * dscudasetdeviceid_1_svc();
#define dscudaSetDeviceFlagsId 306
extern  dscudaResult * dscudasetdeviceflagsid_1();
extern  dscudaResult * dscudasetdeviceflagsid_1_svc();
#define dscudaChooseDeviceId 307
extern  dscudaChooseDeviceResult * dscudachoosedeviceid_1();
extern  dscudaChooseDeviceResult * dscudachoosedeviceid_1_svc();
#define dscudaDeviceSynchronize 308
extern  dscudaResult * dscudadevicesynchronize_1();
extern  dscudaResult * dscudadevicesynchronize_1_svc();
#define dscudaDeviceReset 309
extern  dscudaResult * dscudadevicereset_1();
extern  dscudaResult * dscudadevicereset_1_svc();
#define dscudaStreamCreateId 400
extern  dscudaStreamCreateResult * dscudastreamcreateid_1();
extern  dscudaStreamCreateResult * dscudastreamcreateid_1_svc();
#define dscudaStreamDestroyId 401
extern  dscudaResult * dscudastreamdestroyid_1();
extern  dscudaResult * dscudastreamdestroyid_1_svc();
#define dscudaStreamSynchronizeId 402
extern  dscudaResult * dscudastreamsynchronizeid_1();
extern  dscudaResult * dscudastreamsynchronizeid_1_svc();
#define dscudaStreamQueryId 403
extern  dscudaResult * dscudastreamqueryid_1();
extern  dscudaResult * dscudastreamqueryid_1_svc();
#define dscudaStreamWaitEventId 404
extern  dscudaResult * dscudastreamwaiteventid_1();
extern  dscudaResult * dscudastreamwaiteventid_1_svc();
#define dscudaEventCreateId 500
extern  dscudaEventCreateResult * dscudaeventcreateid_1();
extern  dscudaEventCreateResult * dscudaeventcreateid_1_svc();
#define dscudaEventCreateWithFlagsId 501
extern  dscudaEventCreateResult * dscudaeventcreatewithflagsid_1();
extern  dscudaEventCreateResult * dscudaeventcreatewithflagsid_1_svc();
#define dscudaEventDestroyId 502
extern  dscudaResult * dscudaeventdestroyid_1();
extern  dscudaResult * dscudaeventdestroyid_1_svc();
#define dscudaEventElapsedTimeId 503
extern  dscudaEventElapsedTimeResult * dscudaeventelapsedtimeid_1();
extern  dscudaEventElapsedTimeResult * dscudaeventelapsedtimeid_1_svc();
#define dscudaEventRecordId 504
extern  dscudaResult * dscudaeventrecordid_1();
extern  dscudaResult * dscudaeventrecordid_1_svc();
#define dscudaEventSynchronizeId 505
extern  dscudaResult * dscudaeventsynchronizeid_1();
extern  dscudaResult * dscudaeventsynchronizeid_1_svc();
#define dscudaEventQueryId 506
extern  dscudaResult * dscudaeventqueryid_1();
extern  dscudaResult * dscudaeventqueryid_1_svc();
#define dscudaLaunchKernelId 600
extern  void * dscudalaunchkernelid_1();
extern  void * dscudalaunchkernelid_1_svc();
#define dscudaLoadModuleId 601
extern  dscudaLoadModuleResult * dscudaloadmoduleid_1();
extern  dscudaLoadModuleResult * dscudaloadmoduleid_1_svc();
#define dscudaFuncGetAttributesId 602
extern  dscudaFuncGetAttributesResult * dscudafuncgetattributesid_1();
extern  dscudaFuncGetAttributesResult * dscudafuncgetattributesid_1_svc();
#define dscudaMallocId 700
extern  dscudaMallocResult * dscudamallocid_1();
extern  dscudaMallocResult * dscudamallocid_1_svc();
#define dscudaFreeId 701
extern  dscudaResult * dscudafreeid_1();
extern  dscudaResult * dscudafreeid_1_svc();
#define dscudaMemcpyH2HId 702
extern  dscudaMemcpyH2HResult * dscudamemcpyh2hid_1();
extern  dscudaMemcpyH2HResult * dscudamemcpyh2hid_1_svc();
#define dscudaMemcpyH2DId 703
extern  dscudaResult * dscudamemcpyh2did_1();
extern  dscudaResult * dscudamemcpyh2did_1_svc();
#define dscudaMemcpyD2HId 704
extern  dscudaMemcpyD2HResult * dscudamemcpyd2hid_1();
extern  dscudaMemcpyD2HResult * dscudamemcpyd2hid_1_svc();
#define dscudaMemcpyD2DId 705
extern  dscudaResult * dscudamemcpyd2did_1();
extern  dscudaResult * dscudamemcpyd2did_1_svc();
#define dscudaMemcpyAsyncH2HId 706
extern  dscudaMemcpyAsyncH2HResult * dscudamemcpyasynch2hid_1();
extern  dscudaMemcpyAsyncH2HResult * dscudamemcpyasynch2hid_1_svc();
#define dscudaMemcpyAsyncH2DId 707
extern  dscudaResult * dscudamemcpyasynch2did_1();
extern  dscudaResult * dscudamemcpyasynch2did_1_svc();
#define dscudaMemcpyAsyncD2HId 708
extern  dscudaMemcpyAsyncD2HResult * dscudamemcpyasyncd2hid_1();
extern  dscudaMemcpyAsyncD2HResult * dscudamemcpyasyncd2hid_1_svc();
#define dscudaMemcpyAsyncD2DId 709
extern  dscudaResult * dscudamemcpyasyncd2did_1();
extern  dscudaResult * dscudamemcpyasyncd2did_1_svc();
#define dscudaMemcpyToSymbolH2DId 710
extern  dscudaResult * dscudamemcpytosymbolh2did_1();
extern  dscudaResult * dscudamemcpytosymbolh2did_1_svc();
#define dscudaMemcpyToSymbolD2DId 711
extern  dscudaResult * dscudamemcpytosymbold2did_1();
extern  dscudaResult * dscudamemcpytosymbold2did_1_svc();
#define dscudaMemcpyFromSymbolD2HId 712
extern  dscudaMemcpyFromSymbolD2HResult * dscudamemcpyfromsymbold2hid_1();
extern  dscudaMemcpyFromSymbolD2HResult * dscudamemcpyfromsymbold2hid_1_svc();
#define dscudaMemcpyFromSymbolD2DId 713
extern  dscudaResult * dscudamemcpyfromsymbold2did_1();
extern  dscudaResult * dscudamemcpyfromsymbold2did_1_svc();
#define dscudaMemsetId 714
extern  dscudaResult * dscudamemsetid_1();
extern  dscudaResult * dscudamemsetid_1_svc();
#define dscudaHostAllocId 715
extern  dscudaHostAllocResult * dscudahostallocid_1();
extern  dscudaHostAllocResult * dscudahostallocid_1_svc();
#define dscudaMallocHostId 716
extern  dscudaMallocHostResult * dscudamallochostid_1();
extern  dscudaMallocHostResult * dscudamallochostid_1_svc();
#define dscudaFreeHostId 717
extern  dscudaResult * dscudafreehostid_1();
extern  dscudaResult * dscudafreehostid_1_svc();
#define dscudaHostGetDevicePointerId 718
extern  dscudaHostGetDevicePointerResult * dscudahostgetdevicepointerid_1();
extern  dscudaHostGetDevicePointerResult * dscudahostgetdevicepointerid_1_svc();
#define dscudaHostGetFlagsID 719
extern  dscudaHostGetFlagsResult * dscudahostgetflagsid_1();
extern  dscudaHostGetFlagsResult * dscudahostgetflagsid_1_svc();
#define dscudaMallocArrayId 720
extern  dscudaMallocArrayResult * dscudamallocarrayid_1();
extern  dscudaMallocArrayResult * dscudamallocarrayid_1_svc();
#define dscudaFreeArrayId 721
extern  dscudaResult * dscudafreearrayid_1();
extern  dscudaResult * dscudafreearrayid_1_svc();
#define dscudaMemcpyToArrayH2HId 722
extern  dscudaMemcpyToArrayH2HResult * dscudamemcpytoarrayh2hid_1();
extern  dscudaMemcpyToArrayH2HResult * dscudamemcpytoarrayh2hid_1_svc();
#define dscudaMemcpyToArrayH2DId 723
extern  dscudaResult * dscudamemcpytoarrayh2did_1();
extern  dscudaResult * dscudamemcpytoarrayh2did_1_svc();
#define dscudaMemcpyToArrayD2HId 724
extern  dscudaMemcpyToArrayD2HResult * dscudamemcpytoarrayd2hid_1();
extern  dscudaMemcpyToArrayD2HResult * dscudamemcpytoarrayd2hid_1_svc();
#define dscudaMemcpyToArrayD2DId 725
extern  dscudaResult * dscudamemcpytoarrayd2did_1();
extern  dscudaResult * dscudamemcpytoarrayd2did_1_svc();
#define dscudaMallocPitchId 726
extern  dscudaMallocPitchResult * dscudamallocpitchid_1();
extern  dscudaMallocPitchResult * dscudamallocpitchid_1_svc();
#define dscudaMemcpy2DToArrayH2HId 727
extern  dscudaMemcpy2DToArrayH2HResult * dscudamemcpy2dtoarrayh2hid_1();
extern  dscudaMemcpy2DToArrayH2HResult * dscudamemcpy2dtoarrayh2hid_1_svc();
#define dscudaMemcpy2DToArrayH2DId 728
extern  dscudaResult * dscudamemcpy2dtoarrayh2did_1();
extern  dscudaResult * dscudamemcpy2dtoarrayh2did_1_svc();
#define dscudaMemcpy2DToArrayD2HId 729
extern  dscudaMemcpy2DToArrayD2HResult * dscudamemcpy2dtoarrayd2hid_1();
extern  dscudaMemcpy2DToArrayD2HResult * dscudamemcpy2dtoarrayd2hid_1_svc();
#define dscudaMemcpy2DToArrayD2DId 730
extern  dscudaResult * dscudamemcpy2dtoarrayd2did_1();
extern  dscudaResult * dscudamemcpy2dtoarrayd2did_1_svc();
#define dscudaMemcpy2DH2HId 731
extern  dscudaMemcpy2DH2HResult * dscudamemcpy2dh2hid_1();
extern  dscudaMemcpy2DH2HResult * dscudamemcpy2dh2hid_1_svc();
#define dscudaMemcpy2DH2DId 732
extern  dscudaResult * dscudamemcpy2dh2did_1();
extern  dscudaResult * dscudamemcpy2dh2did_1_svc();
#define dscudaMemcpy2DD2HId 733
extern  dscudaMemcpy2DD2HResult * dscudamemcpy2dd2hid_1();
extern  dscudaMemcpy2DD2HResult * dscudamemcpy2dd2hid_1_svc();
#define dscudaMemcpy2DD2DId 734
extern  dscudaResult * dscudamemcpy2dd2did_1();
extern  dscudaResult * dscudamemcpy2dd2did_1_svc();
#define dscudaMemset2DId 735
extern  dscudaResult * dscudamemset2did_1();
extern  dscudaResult * dscudamemset2did_1_svc();
#define dscudaMemcpyToSymbolAsyncH2DId 736
extern  dscudaResult * dscudamemcpytosymbolasynch2did_1();
extern  dscudaResult * dscudamemcpytosymbolasynch2did_1_svc();
#define dscudaMemcpyToSymbolAsyncD2DId 737
extern  dscudaResult * dscudamemcpytosymbolasyncd2did_1();
extern  dscudaResult * dscudamemcpytosymbolasyncd2did_1_svc();
#define dscudaMemcpyFromSymbolAsyncD2HId 738
extern  dscudaMemcpyFromSymbolAsyncD2HResult * dscudamemcpyfromsymbolasyncd2hid_1();
extern  dscudaMemcpyFromSymbolAsyncD2HResult * dscudamemcpyfromsymbolasyncd2hid_1_svc();
#define dscudaMemcpyFromSymbolAsyncD2DId 739
extern  dscudaResult * dscudamemcpyfromsymbolasyncd2did_1();
extern  dscudaResult * dscudamemcpyfromsymbolasyncd2did_1_svc();
#define dscudaCreateChannelDescId 1400
extern  dscudaCreateChannelDescResult * dscudacreatechanneldescid_1();
extern  dscudaCreateChannelDescResult * dscudacreatechanneldescid_1_svc();
#define dscudaGetChannelDescId 1401
extern  dscudaGetChannelDescResult * dscudagetchanneldescid_1();
extern  dscudaGetChannelDescResult * dscudagetchanneldescid_1_svc();
#define dscudaBindTextureId 1402
extern  dscudaBindTextureResult * dscudabindtextureid_1();
extern  dscudaBindTextureResult * dscudabindtextureid_1_svc();
#define dscudaBindTexture2DId 1403
extern  dscudaBindTexture2DResult * dscudabindtexture2did_1();
extern  dscudaBindTexture2DResult * dscudabindtexture2did_1_svc();
#define dscudaBindTextureToArrayId 1404
extern  dscudaResult * dscudabindtexturetoarrayid_1();
extern  dscudaResult * dscudabindtexturetoarrayid_1_svc();
#define dscudaUnbindTextureId 1405
extern  dscudaResult * dscudaunbindtextureid_1();
extern  dscudaResult * dscudaunbindtextureid_1_svc();
#define dscufftPlan3dId 2002
extern  dscufftPlanResult * dscufftplan3did_1();
extern  dscufftPlanResult * dscufftplan3did_1_svc();
#define dscufftDestroyId 2004
extern  dscufftResult * dscufftdestroyid_1();
extern  dscufftResult * dscufftdestroyid_1_svc();
#define dscufftExecC2CId 2005
extern  dscufftResult * dscufftexecc2cid_1();
extern  dscufftResult * dscufftexecc2cid_1_svc();
extern int dscuda_prog_1_freeresult ();
#endif 



#if defined(__STDC__) || defined(__cplusplus)
extern  bool_t xdr_RCadr (XDR *, RCadr*);
extern  bool_t xdr_RCstream (XDR *, RCstream*);
extern  bool_t xdr_RCevent (XDR *, RCevent*);
extern  bool_t xdr_RCipaddr (XDR *, RCipaddr*);
extern  bool_t xdr_RCsize (XDR *, RCsize*);
extern  bool_t xdr_RCerror (XDR *, RCerror*);
extern  bool_t xdr_RCbuf (XDR *, RCbuf*);
extern  bool_t xdr_RCchannelformat (XDR *, RCchannelformat*);
extern  bool_t xdr_RCpid (XDR *, RCpid*);
extern  bool_t xdr_RCchanneldesc_t (XDR *, RCchanneldesc_t*);
extern  bool_t xdr_RCchanneldesc (XDR *, RCchanneldesc*);
extern  bool_t xdr_RCtexture_t (XDR *, RCtexture_t*);
extern  bool_t xdr_RCtexture (XDR *, RCtexture*);
extern  bool_t xdr_RCfuncattr_t (XDR *, RCfuncattr_t*);
extern  bool_t xdr_RCfuncattr (XDR *, RCfuncattr*);
extern  bool_t xdr_RCargType (XDR *, RCargType*);
extern  bool_t xdr_RCargVal (XDR *, RCargVal*);
extern  bool_t xdr_RCarg (XDR *, RCarg*);
extern  bool_t xdr_RCargs (XDR *, RCargs*);
extern  bool_t xdr_dscudaResult (XDR *, dscudaResult*);
extern  bool_t xdr_dscudaThreadGetLimitResult (XDR *, dscudaThreadGetLimitResult*);
extern  bool_t xdr_dscudaThreadGetCacheConfigResult (XDR *, dscudaThreadGetCacheConfigResult*);
extern  bool_t xdr_dscudaMallocResult (XDR *, dscudaMallocResult*);
extern  bool_t xdr_dscudaHostAllocResult (XDR *, dscudaHostAllocResult*);
extern  bool_t xdr_dscudaMallocHostResult (XDR *, dscudaMallocHostResult*);
extern  bool_t xdr_dscudaMallocArrayResult (XDR *, dscudaMallocArrayResult*);
extern  bool_t xdr_dscudaMallocPitchResult (XDR *, dscudaMallocPitchResult*);
extern  bool_t xdr_dscudaMemcpyD2HResult (XDR *, dscudaMemcpyD2HResult*);
extern  bool_t xdr_dscudaMemcpyH2HResult (XDR *, dscudaMemcpyH2HResult*);
extern  bool_t xdr_dscudaMemcpyToArrayD2HResult (XDR *, dscudaMemcpyToArrayD2HResult*);
extern  bool_t xdr_dscudaMemcpyToArrayH2HResult (XDR *, dscudaMemcpyToArrayH2HResult*);
extern  bool_t xdr_dscudaMemcpy2DToArrayD2HResult (XDR *, dscudaMemcpy2DToArrayD2HResult*);
extern  bool_t xdr_dscudaMemcpy2DToArrayH2HResult (XDR *, dscudaMemcpy2DToArrayH2HResult*);
extern  bool_t xdr_dscudaMemcpy2DD2HResult (XDR *, dscudaMemcpy2DD2HResult*);
extern  bool_t xdr_dscudaMemcpy2DH2HResult (XDR *, dscudaMemcpy2DH2HResult*);
extern  bool_t xdr_dscudaGetDeviceResult (XDR *, dscudaGetDeviceResult*);
extern  bool_t xdr_dscudaGetDeviceCountResult (XDR *, dscudaGetDeviceCountResult*);
extern  bool_t xdr_dscudaGetDevicePropertiesResult (XDR *, dscudaGetDevicePropertiesResult*);
extern  bool_t xdr_dscudaDriverGetVersionResult (XDR *, dscudaDriverGetVersionResult*);
extern  bool_t xdr_dscudaRuntimeGetVersionResult (XDR *, dscudaRuntimeGetVersionResult*);
extern  bool_t xdr_dscudaGetErrorStringResult (XDR *, dscudaGetErrorStringResult*);
extern  bool_t xdr_dscudaCreateChannelDescResult (XDR *, dscudaCreateChannelDescResult*);
extern  bool_t xdr_dscudaGetChannelDescResult (XDR *, dscudaGetChannelDescResult*);
extern  bool_t xdr_dscudaChooseDeviceResult (XDR *, dscudaChooseDeviceResult*);
extern  bool_t xdr_dscudaMemcpyAsyncD2HResult (XDR *, dscudaMemcpyAsyncD2HResult*);
extern  bool_t xdr_dscudaMemcpyAsyncH2HResult (XDR *, dscudaMemcpyAsyncH2HResult*);
extern  bool_t xdr_dscudaMemcpyFromSymbolD2HResult (XDR *, dscudaMemcpyFromSymbolD2HResult*);
extern  bool_t xdr_dscudaMemcpyFromSymbolAsyncD2HResult (XDR *, dscudaMemcpyFromSymbolAsyncD2HResult*);
extern  bool_t xdr_dscudaStreamCreateResult (XDR *, dscudaStreamCreateResult*);
extern  bool_t xdr_dscudaEventCreateResult (XDR *, dscudaEventCreateResult*);
extern  bool_t xdr_dscudaEventElapsedTimeResult (XDR *, dscudaEventElapsedTimeResult*);
extern  bool_t xdr_dscudaHostGetDevicePointerResult (XDR *, dscudaHostGetDevicePointerResult*);
extern  bool_t xdr_dscudaHostGetFlagsResult (XDR *, dscudaHostGetFlagsResult*);
extern  bool_t xdr_dscudaLoadModuleResult (XDR *, dscudaLoadModuleResult*);
extern  bool_t xdr_dscudaFuncGetAttributesResult (XDR *, dscudaFuncGetAttributesResult*);
extern  bool_t xdr_dscudaBindTextureResult (XDR *, dscudaBindTextureResult*);
extern  bool_t xdr_dscudaBindTexture2DResult (XDR *, dscudaBindTexture2DResult*);
extern  bool_t xdr_dscufftResult (XDR *, dscufftResult*);
extern  bool_t xdr_dscufftPlanResult (XDR *, dscufftPlanResult*);
extern  bool_t xdr_dscublasResult (XDR *, dscublasResult*);
extern  bool_t xdr_dscublasCreateResult (XDR *, dscublasCreateResult*);
extern  bool_t xdr_dscublasGetVectorResult (XDR *, dscublasGetVectorResult*);
extern  bool_t xdr_RCdim3 (XDR *, RCdim3*);
extern  bool_t xdr_dscudathreadsetlimitid_1_argument (XDR *, dscudathreadsetlimitid_1_argument*);
extern  bool_t xdr_dscudastreamwaiteventid_1_argument (XDR *, dscudastreamwaiteventid_1_argument*);
extern  bool_t xdr_dscudaeventelapsedtimeid_1_argument (XDR *, dscudaeventelapsedtimeid_1_argument*);
extern  bool_t xdr_dscudaeventrecordid_1_argument (XDR *, dscudaeventrecordid_1_argument*);
extern  bool_t xdr_dscudalaunchkernelid_1_argument (XDR *, dscudalaunchkernelid_1_argument*);
extern  bool_t xdr_dscudaloadmoduleid_1_argument (XDR *, dscudaloadmoduleid_1_argument*);
extern  bool_t xdr_dscudafuncgetattributesid_1_argument (XDR *, dscudafuncgetattributesid_1_argument*);
extern  bool_t xdr_dscudamemcpyh2hid_1_argument (XDR *, dscudamemcpyh2hid_1_argument*);
extern  bool_t xdr_dscudamemcpyh2did_1_argument (XDR *, dscudamemcpyh2did_1_argument*);
extern  bool_t xdr_dscudamemcpyd2hid_1_argument (XDR *, dscudamemcpyd2hid_1_argument*);
extern  bool_t xdr_dscudamemcpyd2did_1_argument (XDR *, dscudamemcpyd2did_1_argument*);
extern  bool_t xdr_dscudamemcpyasynch2hid_1_argument (XDR *, dscudamemcpyasynch2hid_1_argument*);
extern  bool_t xdr_dscudamemcpyasynch2did_1_argument (XDR *, dscudamemcpyasynch2did_1_argument*);
extern  bool_t xdr_dscudamemcpyasyncd2hid_1_argument (XDR *, dscudamemcpyasyncd2hid_1_argument*);
extern  bool_t xdr_dscudamemcpyasyncd2did_1_argument (XDR *, dscudamemcpyasyncd2did_1_argument*);
extern  bool_t xdr_dscudamemcpytosymbolh2did_1_argument (XDR *, dscudamemcpytosymbolh2did_1_argument*);
extern  bool_t xdr_dscudamemcpytosymbold2did_1_argument (XDR *, dscudamemcpytosymbold2did_1_argument*);
extern  bool_t xdr_dscudamemcpyfromsymbold2hid_1_argument (XDR *, dscudamemcpyfromsymbold2hid_1_argument*);
extern  bool_t xdr_dscudamemcpyfromsymbold2did_1_argument (XDR *, dscudamemcpyfromsymbold2did_1_argument*);
extern  bool_t xdr_dscudamemsetid_1_argument (XDR *, dscudamemsetid_1_argument*);
extern  bool_t xdr_dscudahostallocid_1_argument (XDR *, dscudahostallocid_1_argument*);
extern  bool_t xdr_dscudahostgetdevicepointerid_1_argument (XDR *, dscudahostgetdevicepointerid_1_argument*);
extern  bool_t xdr_dscudamallocarrayid_1_argument (XDR *, dscudamallocarrayid_1_argument*);
extern  bool_t xdr_dscudamemcpytoarrayh2hid_1_argument (XDR *, dscudamemcpytoarrayh2hid_1_argument*);
extern  bool_t xdr_dscudamemcpytoarrayh2did_1_argument (XDR *, dscudamemcpytoarrayh2did_1_argument*);
extern  bool_t xdr_dscudamemcpytoarrayd2hid_1_argument (XDR *, dscudamemcpytoarrayd2hid_1_argument*);
extern  bool_t xdr_dscudamemcpytoarrayd2did_1_argument (XDR *, dscudamemcpytoarrayd2did_1_argument*);
extern  bool_t xdr_dscudamallocpitchid_1_argument (XDR *, dscudamallocpitchid_1_argument*);
extern  bool_t xdr_dscudamemcpy2dtoarrayh2hid_1_argument (XDR *, dscudamemcpy2dtoarrayh2hid_1_argument*);
extern  bool_t xdr_dscudamemcpy2dtoarrayh2did_1_argument (XDR *, dscudamemcpy2dtoarrayh2did_1_argument*);
extern  bool_t xdr_dscudamemcpy2dtoarrayd2hid_1_argument (XDR *, dscudamemcpy2dtoarrayd2hid_1_argument*);
extern  bool_t xdr_dscudamemcpy2dtoarrayd2did_1_argument (XDR *, dscudamemcpy2dtoarrayd2did_1_argument*);
extern  bool_t xdr_dscudamemcpy2dh2hid_1_argument (XDR *, dscudamemcpy2dh2hid_1_argument*);
extern  bool_t xdr_dscudamemcpy2dh2did_1_argument (XDR *, dscudamemcpy2dh2did_1_argument*);
extern  bool_t xdr_dscudamemcpy2dd2hid_1_argument (XDR *, dscudamemcpy2dd2hid_1_argument*);
extern  bool_t xdr_dscudamemcpy2dd2did_1_argument (XDR *, dscudamemcpy2dd2did_1_argument*);
extern  bool_t xdr_dscudamemset2did_1_argument (XDR *, dscudamemset2did_1_argument*);
extern  bool_t xdr_dscudamemcpytosymbolasynch2did_1_argument (XDR *, dscudamemcpytosymbolasynch2did_1_argument*);
extern  bool_t xdr_dscudamemcpytosymbolasyncd2did_1_argument (XDR *, dscudamemcpytosymbolasyncd2did_1_argument*);
extern  bool_t xdr_dscudamemcpyfromsymbolasyncd2hid_1_argument (XDR *, dscudamemcpyfromsymbolasyncd2hid_1_argument*);
extern  bool_t xdr_dscudamemcpyfromsymbolasyncd2did_1_argument (XDR *, dscudamemcpyfromsymbolasyncd2did_1_argument*);
extern  bool_t xdr_dscudacreatechanneldescid_1_argument (XDR *, dscudacreatechanneldescid_1_argument*);
extern  bool_t xdr_dscudabindtextureid_1_argument (XDR *, dscudabindtextureid_1_argument*);
extern  bool_t xdr_dscudabindtexture2did_1_argument (XDR *, dscudabindtexture2did_1_argument*);
extern  bool_t xdr_dscudabindtexturetoarrayid_1_argument (XDR *, dscudabindtexturetoarrayid_1_argument*);
extern  bool_t xdr_dscufftplan3did_1_argument (XDR *, dscufftplan3did_1_argument*);
extern  bool_t xdr_dscufftexecc2cid_1_argument (XDR *, dscufftexecc2cid_1_argument*);

#else 
extern bool_t xdr_RCadr ();
extern bool_t xdr_RCstream ();
extern bool_t xdr_RCevent ();
extern bool_t xdr_RCipaddr ();
extern bool_t xdr_RCsize ();
extern bool_t xdr_RCerror ();
extern bool_t xdr_RCbuf ();
extern bool_t xdr_RCchannelformat ();
extern bool_t xdr_RCpid ();
extern bool_t xdr_RCchanneldesc_t ();
extern bool_t xdr_RCchanneldesc ();
extern bool_t xdr_RCtexture_t ();
extern bool_t xdr_RCtexture ();
extern bool_t xdr_RCfuncattr_t ();
extern bool_t xdr_RCfuncattr ();
extern bool_t xdr_RCargType ();
extern bool_t xdr_RCargVal ();
extern bool_t xdr_RCarg ();
extern bool_t xdr_RCargs ();
extern bool_t xdr_dscudaResult ();
extern bool_t xdr_dscudaThreadGetLimitResult ();
extern bool_t xdr_dscudaThreadGetCacheConfigResult ();
extern bool_t xdr_dscudaMallocResult ();
extern bool_t xdr_dscudaHostAllocResult ();
extern bool_t xdr_dscudaMallocHostResult ();
extern bool_t xdr_dscudaMallocArrayResult ();
extern bool_t xdr_dscudaMallocPitchResult ();
extern bool_t xdr_dscudaMemcpyD2HResult ();
extern bool_t xdr_dscudaMemcpyH2HResult ();
extern bool_t xdr_dscudaMemcpyToArrayD2HResult ();
extern bool_t xdr_dscudaMemcpyToArrayH2HResult ();
extern bool_t xdr_dscudaMemcpy2DToArrayD2HResult ();
extern bool_t xdr_dscudaMemcpy2DToArrayH2HResult ();
extern bool_t xdr_dscudaMemcpy2DD2HResult ();
extern bool_t xdr_dscudaMemcpy2DH2HResult ();
extern bool_t xdr_dscudaGetDeviceResult ();
extern bool_t xdr_dscudaGetDeviceCountResult ();
extern bool_t xdr_dscudaGetDevicePropertiesResult ();
extern bool_t xdr_dscudaDriverGetVersionResult ();
extern bool_t xdr_dscudaRuntimeGetVersionResult ();
extern bool_t xdr_dscudaGetErrorStringResult ();
extern bool_t xdr_dscudaCreateChannelDescResult ();
extern bool_t xdr_dscudaGetChannelDescResult ();
extern bool_t xdr_dscudaChooseDeviceResult ();
extern bool_t xdr_dscudaMemcpyAsyncD2HResult ();
extern bool_t xdr_dscudaMemcpyAsyncH2HResult ();
extern bool_t xdr_dscudaMemcpyFromSymbolD2HResult ();
extern bool_t xdr_dscudaMemcpyFromSymbolAsyncD2HResult ();
extern bool_t xdr_dscudaStreamCreateResult ();
extern bool_t xdr_dscudaEventCreateResult ();
extern bool_t xdr_dscudaEventElapsedTimeResult ();
extern bool_t xdr_dscudaHostGetDevicePointerResult ();
extern bool_t xdr_dscudaHostGetFlagsResult ();
extern bool_t xdr_dscudaLoadModuleResult ();
extern bool_t xdr_dscudaFuncGetAttributesResult ();
extern bool_t xdr_dscudaBindTextureResult ();
extern bool_t xdr_dscudaBindTexture2DResult ();
extern bool_t xdr_dscufftResult ();
extern bool_t xdr_dscufftPlanResult ();
extern bool_t xdr_dscublasResult ();
extern bool_t xdr_dscublasCreateResult ();
extern bool_t xdr_dscublasGetVectorResult ();
extern bool_t xdr_RCdim3 ();
extern bool_t xdr_dscudathreadsetlimitid_1_argument ();
extern bool_t xdr_dscudastreamwaiteventid_1_argument ();
extern bool_t xdr_dscudaeventelapsedtimeid_1_argument ();
extern bool_t xdr_dscudaeventrecordid_1_argument ();
extern bool_t xdr_dscudalaunchkernelid_1_argument ();
extern bool_t xdr_dscudaloadmoduleid_1_argument ();
extern bool_t xdr_dscudafuncgetattributesid_1_argument ();
extern bool_t xdr_dscudamemcpyh2hid_1_argument ();
extern bool_t xdr_dscudamemcpyh2did_1_argument ();
extern bool_t xdr_dscudamemcpyd2hid_1_argument ();
extern bool_t xdr_dscudamemcpyd2did_1_argument ();
extern bool_t xdr_dscudamemcpyasynch2hid_1_argument ();
extern bool_t xdr_dscudamemcpyasynch2did_1_argument ();
extern bool_t xdr_dscudamemcpyasyncd2hid_1_argument ();
extern bool_t xdr_dscudamemcpyasyncd2did_1_argument ();
extern bool_t xdr_dscudamemcpytosymbolh2did_1_argument ();
extern bool_t xdr_dscudamemcpytosymbold2did_1_argument ();
extern bool_t xdr_dscudamemcpyfromsymbold2hid_1_argument ();
extern bool_t xdr_dscudamemcpyfromsymbold2did_1_argument ();
extern bool_t xdr_dscudamemsetid_1_argument ();
extern bool_t xdr_dscudahostallocid_1_argument ();
extern bool_t xdr_dscudahostgetdevicepointerid_1_argument ();
extern bool_t xdr_dscudamallocarrayid_1_argument ();
extern bool_t xdr_dscudamemcpytoarrayh2hid_1_argument ();
extern bool_t xdr_dscudamemcpytoarrayh2did_1_argument ();
extern bool_t xdr_dscudamemcpytoarrayd2hid_1_argument ();
extern bool_t xdr_dscudamemcpytoarrayd2did_1_argument ();
extern bool_t xdr_dscudamallocpitchid_1_argument ();
extern bool_t xdr_dscudamemcpy2dtoarrayh2hid_1_argument ();
extern bool_t xdr_dscudamemcpy2dtoarrayh2did_1_argument ();
extern bool_t xdr_dscudamemcpy2dtoarrayd2hid_1_argument ();
extern bool_t xdr_dscudamemcpy2dtoarrayd2did_1_argument ();
extern bool_t xdr_dscudamemcpy2dh2hid_1_argument ();
extern bool_t xdr_dscudamemcpy2dh2did_1_argument ();
extern bool_t xdr_dscudamemcpy2dd2hid_1_argument ();
extern bool_t xdr_dscudamemcpy2dd2did_1_argument ();
extern bool_t xdr_dscudamemset2did_1_argument ();
extern bool_t xdr_dscudamemcpytosymbolasynch2did_1_argument ();
extern bool_t xdr_dscudamemcpytosymbolasyncd2did_1_argument ();
extern bool_t xdr_dscudamemcpyfromsymbolasyncd2hid_1_argument ();
extern bool_t xdr_dscudamemcpyfromsymbolasyncd2did_1_argument ();
extern bool_t xdr_dscudacreatechanneldescid_1_argument ();
extern bool_t xdr_dscudabindtextureid_1_argument ();
extern bool_t xdr_dscudabindtexture2did_1_argument ();
extern bool_t xdr_dscudabindtexturetoarrayid_1_argument ();
extern bool_t xdr_dscufftplan3did_1_argument ();
extern bool_t xdr_dscufftexecc2cid_1_argument ();

#endif 

#ifdef __cplusplus
}
#endif

#endif 
#pragma end dscudarpc.h
#pragma begin dscudamacros.h
#ifndef DSCUDA_MACROS_H
#define DSCUDA_MACROS_H

#define WARN(lv, fmt, args...) if (lv <= dscudaWarnLevel()) fprintf(stderr, fmt, ## args);
#define WARNONCE(lv, fmt, args...) if (lv <= dscudaWarnLevel()) { \
        static int firstcall = 1;                                 \
        if (firstcall) {                                          \
            firstcall = 0;                                        \
            fprintf(stderr, fmt, ## args);                        \
        }                                                         \
    }

#define ALIGN_UP(off, align) (off) = ((off) + (align) - 1) & ~((align) - 1)
int dscudaWarnLevel(void);
void dscudaSetWarnLevel(int level);

#endif 
#pragma end dscudamacros.h

#define TEST_NZ(x) do { if ( (x)) {WARN(0, #x " failed (returned non-zero).\n" ); exit(EXIT_FAILURE); } } while (0)
#define TEST_Z(x)  do { if (!(x)) {WARN(0, #x " failed (returned zero/null).\n"); exit(EXIT_FAILURE); } } while (0)


#define RC_NWR_PER_POST (16) 
#define RC_SGE_SIZE (1024 * 1024 * 2) 
#define RC_WR_MAX (RC_NWR_PER_POST * 16) 
#define RC_RDMA_BUF_SIZE (RC_NWR_PER_POST * RC_SGE_SIZE) 


#if RC_RDMA_BUF_SIZE  < RC_KMODULEIMAGELEN
#error "RC_RDMA_BUF_SIZE too small."

#endif

#define RC_SERVER_IBV_CQ_SIZE (RC_WR_MAX)
#define RC_CLIENT_IBV_CQ_SIZE (65536)

#define RC_IBV_IP_PORT_BASE  (65432)
#define RC_IBV_TIMEOUT (500)  

struct message {
    struct ibv_mr mr[RC_NWR_PER_POST];
};

enum rdma_state_t {
    STATE_INIT,
    STATE_READY,
    STATE_BUSY,
};

typedef struct {
    
    struct rdma_cm_id *id;
    struct ibv_qp *qp;
    struct ibv_context *ibvctx;
    struct ibv_pd *pd;
    struct ibv_cq *cq;
    struct ibv_comp_channel *comp_channel;

    
    struct message *recv_msg;
    struct message *send_msg;

    
    char *rdma_local_region;
    char *rdma_remote_region;

    
    struct ibv_mr *recv_mr;
    struct ibv_mr *send_mr;
    struct ibv_mr peer_mr[RC_NWR_PER_POST];

    
    struct ibv_mr *rdma_local_mr[RC_NWR_PER_POST];
    struct ibv_mr *rdma_remote_mr[RC_NWR_PER_POST];

    
    pthread_t cq_poller_thread;
    int connected;
    enum rdma_state_t rdma_state;
    int rdma_nreq_pending;
} IbvConnection;

typedef enum {
    RCMethodNone = 0,
    RCMethodMemcpyH2D,
    RCMethodMemcpyD2H,
    RCMethodMemcpyD2D,
    RCMethodMalloc,
    RCMethodFree,
    RCMethodGetErrorString,
    RCMethodGetDeviceProperties,
    RCMethodRuntimeGetVersion,
    RCMethodThreadSynchronize,
    RCMethodThreadExit,
    RCMethodDeviceSynchronize,
    RCMethodDscudaMemcpyToSymbolH2D,
    RCMethodDscudaMemcpyToSymbolD2D,
    RCMethodDscudaMemcpyFromSymbolD2H,
    RCMethodDscudaMemcpyFromSymbolD2D,
    RCMethodDscudaMemcpyToSymbolAsyncH2D,
    RCMethodDscudaMemcpyToSymbolAsyncD2D,
    RCMethodDscudaMemcpyFromSymbolAsyncD2H,
    RCMethodDscudaMemcpyFromSymbolAsyncD2D,
    RCMethodDscudaLoadModule,
    RCMethodDscudaLaunchKernel,

    

    RCMethodEnd
} RCMethod;


typedef struct {
    RCMethod method;
    int payload;
} IbvHdr;


typedef struct {
    RCMethod method;
    size_t count;
    RCadr dstadr;
    void *srcbuf;
} IbvMemcpyH2DInvokeHdr;

typedef struct {
    RCMethod method;
    cudaError_t err;
} IbvMemcpyH2DReturnHdr;


typedef struct {
    RCMethod method;
    size_t count;
    RCadr srcadr;
} IbvMemcpyD2HInvokeHdr;

typedef struct {
    RCMethod method;
    cudaError_t err;
    void *dstbuf;
} IbvMemcpyD2HReturnHdr;


typedef struct {
    RCMethod method;
    size_t count;
    RCadr dstadr;
    RCadr srcadr;
} IbvMemcpyD2DInvokeHdr;

typedef struct {
    RCMethod method;
    cudaError_t err;
} IbvMemcpyD2DReturnHdr;


typedef struct {
    RCMethod method;
    size_t size;
} IbvMallocInvokeHdr;

typedef struct {
    RCMethod method;
    cudaError_t err;
    RCadr devAdr;
} IbvMallocReturnHdr;


typedef struct {
    RCMethod method;
    RCadr devAdr;
} IbvFreeInvokeHdr;

typedef struct {
    RCMethod method;
    cudaError_t err;
} IbvFreeReturnHdr;


typedef struct {
    RCMethod method;
    int device;
    cudaError_t err;
} IbvGetErrorStringInvokeHdr;

typedef struct {
    RCMethod method;
    char *errmsg;
} IbvGetErrorStringReturnHdr;


typedef struct {
    RCMethod method;
    int device;
} IbvGetDevicePropertiesInvokeHdr;

typedef struct {
    RCMethod method;
    cudaError_t err;
    cudaDeviceProp prop;
} IbvGetDevicePropertiesReturnHdr;


typedef struct {
    RCMethod method;
    char dummy[8];
} IbvRuntimeGetVersionInvokeHdr;

typedef struct {
    RCMethod method;
    cudaError_t err;
    int version;
} IbvRuntimeGetVersionReturnHdr;


typedef struct {
    RCMethod method;
    char dummy[8];
} IbvThreadSynchronizeInvokeHdr;

typedef struct {
    RCMethod method;
    cudaError_t err;
} IbvThreadSynchronizeReturnHdr;


typedef struct {
    RCMethod method;
    char dummy[8];
} IbvThreadExitInvokeHdr;

typedef struct {
    RCMethod method;
    cudaError_t err;
} IbvThreadExitReturnHdr;


typedef struct {
    RCMethod method;
    char dummy[8];
} IbvDeviceSynchronizeInvokeHdr;

typedef struct {
    RCMethod method;
    cudaError_t err;
} IbvDeviceSynchronizeReturnHdr;


typedef struct {
    RCMethod method;
    int moduleid;
    char symbol[RC_SNAMELEN];
    size_t count;
    size_t offset;
    void *src;
} IbvDscudaMemcpyToSymbolH2DInvokeHdr;

typedef struct {
    RCMethod method;
    cudaError_t err;
} IbvDscudaMemcpyToSymbolH2DReturnHdr;


typedef struct {
    RCMethod method;
    int moduleid;
    char symbol[RC_SNAMELEN];
    size_t count;
    size_t offset;
    RCadr srcadr;
} IbvDscudaMemcpyToSymbolD2DInvokeHdr;

typedef struct {
    RCMethod method;
    cudaError_t err;
} IbvDscudaMemcpyToSymbolD2DReturnHdr;



typedef struct {
    RCMethod method;
    int moduleid;
    char symbol[RC_SNAMELEN];
    size_t count;
    size_t offset;
} IbvDscudaMemcpyFromSymbolD2HInvokeHdr;

typedef struct {
    RCMethod method;
    cudaError_t err;
    void *dst;
} IbvDscudaMemcpyFromSymbolD2HReturnHdr;


typedef struct {
    RCMethod method;
    int moduleid;
    char symbol[RC_SNAMELEN];
    size_t count;
    size_t offset;
    RCadr dstadr;
} IbvDscudaMemcpyFromSymbolD2DInvokeHdr;

typedef struct {
    RCMethod method;
    cudaError_t err;
} IbvDscudaMemcpyFromSymbolD2DReturnHdr;


typedef struct {
    RCMethod method;
    int moduleid;
    char symbol[RC_SNAMELEN];
    size_t count;
    size_t offset;
    RCstream stream;
    void *src;
} IbvDscudaMemcpyToSymbolAsyncH2DInvokeHdr;

typedef struct {
    RCMethod method;
    cudaError_t err;
} IbvDscudaMemcpyToSymbolAsyncH2DReturnHdr;


typedef struct {
    RCMethod method;
    int moduleid;
    char symbol[RC_SNAMELEN];
    size_t count;
    size_t offset;
    RCstream stream;
    RCadr srcadr;
} IbvDscudaMemcpyToSymbolAsyncD2DInvokeHdr;

typedef struct {
    RCMethod method;
    cudaError_t err;
} IbvDscudaMemcpyToSymbolAsyncD2DReturnHdr;



typedef struct {
    RCMethod method;
    int moduleid;
    char symbol[RC_SNAMELEN];
    size_t count;
    size_t offset;
    RCstream stream;
} IbvDscudaMemcpyFromSymbolAsyncD2HInvokeHdr;

typedef struct {
    RCMethod method;
    cudaError_t err;
    void *dst;
} IbvDscudaMemcpyFromSymbolAsyncD2HReturnHdr;


typedef struct {
    RCMethod method;
    int moduleid;
    char symbol[RC_SNAMELEN];
    size_t count;
    size_t offset;
    RCstream stream;
    RCadr dstadr;
} IbvDscudaMemcpyFromSymbolAsyncD2DInvokeHdr;

typedef struct {
    RCMethod method;
    cudaError_t err;
} IbvDscudaMemcpyFromSymbolAsyncD2DReturnHdr;



typedef struct {
    RCMethod method;
    uint64_t ipaddr;
    unsigned long int pid;
    char modulename[RC_KMODULENAMELEN];
    void *moduleimage;
} IbvDscudaLoadModuleInvokeHdr;

typedef struct {
    RCMethod method;
    cudaError_t err;
    int moduleid;
} IbvDscudaLoadModuleReturnHdr;


typedef struct {
    RCMethod method;
    int moduleid;
    int kernelid;
    char kernelname[RC_KNAMELEN];
    unsigned int gdim[3];
    unsigned int bdim[3];
    unsigned int smemsize;
    RCstream stream;
    int narg;
    void *args;
} IbvDscudaLaunchKernelInvokeHdr;

typedef struct {
    RCMethod method;
    cudaError_t err;
} IbvDscudaLaunchKernelReturnHdr;

typedef struct {
    int type;
    union {
        uint64_t pointerval;
        unsigned int intval;
        float floatval;
        char customval[RC_KARGMAX];
    } val;
    unsigned int offset;
    unsigned int size;
} IbvArg;

void rdmaBuildConnection(struct rdma_cm_id *id, bool is_server);
void rdmaBuildParams(struct rdma_conn_param *params);
void rdmaDestroyConnection(IbvConnection *conn);
void rdmaSetOnCompletionHandler(void (*handler)(struct ibv_wc *));
void rdmaOnCompletionClient(struct ibv_wc *);
void rdmaOnCompletionServer(struct ibv_wc *);
void rdmaWaitEvent(struct rdma_event_channel *ec, rdma_cm_event_type et, int (*handler)(struct rdma_cm_id *id));
void rdmaWaitReadyToKickoff(IbvConnection *conn);
void rdmaWaitReadyToDisconnect(IbvConnection *conn);
void rdmaKickoff(IbvConnection *conn, int length);
void rdmaPipelinedKickoff(IbvConnection *conn, int length, char *payload_buf, char *payload_src, int payload_size);
void rdmaSendMr(IbvConnection *conn);

#endif 

#endif 
#pragma end ibv_rdma.h

enum {
    RC_REMOTECALL_TYPE_RPC,
    RC_REMOTECALL_TYPE_IBV,
};


int dscudaWarnLevel(void);
void dscudaSetWarnLevel(int level);
char *dscudaMemcpyKindName(cudaMemcpyKind kind);
const char *dscudaGetIpaddrString(unsigned int addr);
double RCgetCputime(double *t0);


void *dscudaUvaOfAdr(void *adr, int devid);
int dscudaDevidOfUva(void *adr);
void *dscudaAdrOfUva(void *adr);
int dscudaNredundancy(void);
void dscudaSetAutoVerb(int verb);
int dscudaRemoteCallType(void);
void dscudaSetErrorHandler(void (*handler)(void *), void *handler_arg);
void dscudaGetMangledFunctionName(char *name, const char *funcif, const char *ptxdata);
int *dscudaLoadModule(char *srcname, char *strdata);
void rpcDscudaLaunchKernelWrapper(int *moduleid, int kid, char *kname,
                              RCdim3 gdim, RCdim3 bdim, RCsize smemsize, RCstream stream,
                              RCargs args);
void ibvDscudaLaunchKernelWrapper(int *moduleid, int kid, char *kname,
                                 int *gdim, int *bdim, RCsize smemsize, RCstream stream,
                                 int narg, IbvArg *arg);

cudaError_t dscudaFuncGetAttributesWrapper(int *moduleid, struct cudaFuncAttributes *attr, const char *func);

cudaError_t dscudaMemcpyToSymbolWrapper(int *moduleid, const char *symbol, const void *src,
                                       size_t count, size_t offset = 0,
                                       enum cudaMemcpyKind kind = cudaMemcpyHostToDevice);

cudaError_t dscudaMemcpyToSymbolAsyncWrapper(int *moduleid, const char *symbol, const void *src,
					    size_t count, size_t offset = 0,
					    enum cudaMemcpyKind kind = cudaMemcpyHostToDevice, cudaStream_t stream = 0);

cudaError_t dscudaMemcpyFromSymbolWrapper(int *moduleid, void *dst, const char *symbol,
					 size_t count, size_t offset = 0,
					 enum cudaMemcpyKind kind = cudaMemcpyDeviceToHost);

cudaError_t dscudaMemcpyFromSymbolAsyncWrapper(int *moduleid, void *dst, const char *symbol,
					      size_t count, size_t offset = 0,
					      enum cudaMemcpyKind kind = cudaMemcpyDeviceToHost, cudaStream_t stream = 0);

cudaError_t dscudaBindTextureWrapper(int *moduleid, char *texname,
                                    size_t *offset,
                                    const struct textureReference *tex,
                                    const void *devPtr,
                                    const struct cudaChannelFormatDesc *desc,
                                    size_t size = UINT_MAX);

template<class T, int dim, enum cudaTextureReadMode readMode>
cudaError_t dscudaBindTextureWrapper(int *moduleid, char *texname,
                                    size_t *offset,
                                    const struct texture<T, dim, readMode> &tex,
                                    const void *devPtr,
                                    const struct cudaChannelFormatDesc &desc,
                                    size_t size = UINT_MAX)
{
  return     dscudaBindTextureWrapper(dscudaLoadModule("./dscudatmp/userapp.cu.ptx", Ptxdata), "tex", offset, &tex, devPtr, &desc, size);
}

template<class T, int dim, enum cudaTextureReadMode readMode>
cudaError_t dscudaBindTextureWrapper(int *moduleid, char *texname,
                                    size_t *offset,
                                    const struct texture<T, dim, readMode> &tex,
                                    const void *devPtr,
                                    size_t size = UINT_MAX)
{
  return     dscudaBindTextureWrapper(dscudaLoadModule("./dscudatmp/userapp.cu.ptx", Ptxdata), "tex", offset, tex, devPtr, tex.channelDesc, size);
}


cudaError_t dscudaBindTexture2DWrapper(int *moduleid, char *texname,
                                      size_t *offset,
                                      const struct textureReference *tex,
                                      const void *devPtr,
                                      const struct cudaChannelFormatDesc *desc,
                                      size_t width, size_t height, size_t pitch);

template<class T, int dim, enum cudaTextureReadMode readMode>
cudaError_t dscudaBindTexture2DWrapper(int *moduleid, char *texname,
                                      size_t *offset,
                                      const struct texture<T, dim, readMode> &tex,
                                      const void *devPtr,
                                      const struct cudaChannelFormatDesc &desc,
                                      size_t width, size_t height, size_t pitch)
{
    return dscudaBindTexture2DWrapper(moduleid, texname,
                                     offset, &tex, devPtr, &desc, width, height, pitch);
}

template<class T, int dim, enum cudaTextureReadMode readMode>
cudaError_t dscudaBindTexture2DWrapper(int *moduleid, char *texname,
                                      size_t *offset,
                                      const struct texture<T, dim, readMode> &tex,
                                      const void *devPtr,
                                      size_t width, size_t height, size_t pitch)
{
    return dscudaBindTexture2DWrapper(moduleid, texname,
                                     offset, &tex, devPtr, &tex.channelDesc, width, height, pitch);
}

cudaError_t dscudaBindTextureToArrayWrapper(int *moduleid, char *texname,
                                           const struct textureReference *tex,
                                           const struct cudaArray * array,
                                           const struct cudaChannelFormatDesc *desc);

template<class T, int dim, enum cudaTextureReadMode readMode>
cudaError_t dscudaBindTextureToArrayWrapper(int *moduleid, char *texname,
                                           const struct texture<T, dim, readMode> &tex,
                                           const struct cudaArray * array,
                                           const struct cudaChannelFormatDesc & desc)
{
    return dscudaBindTextureToArrayWrapper(moduleid, texname, &tex, array, &desc);
}

template<class T, int dim, enum cudaTextureReadMode readMode>
cudaError_t dscudaBindTextureToArrayWrapper(int *moduleid, char *texname,
                                           const struct texture<T, dim, readMode> &tex,
                                           const struct cudaArray * array)
{
    struct cudaChannelFormatDesc desc;
    cudaError_t err = cudaGetChannelDesc(&desc, array);
    return err == cudaSuccess ? dscudaBindTextureToArrayWrapper(moduleid, texname, &tex, array, &desc) : err;
}

#endif 
#pragma end dscuda.h
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cutil.h>

#ifdef MIN
#undef MIN
#endif
#ifdef MAX
#undef MAX
#endif
#include <cutil_inline.h>

#pragma begin userapp.cuh

/*
 * stub for remote call to vecAdd.
 */
void
dscudavecAdd(dim3 _gdim, dim3 _bdim, size_t _smemsize, cudaStream_t _stream , float *a, float *b, float *c)
{
    int _narg = 3;
    int _ibvgdim[3], _ibvbdim[3];
    IbvArg _ibvarg[3], *_ibvargp;
    RCargs _rcargs;
    RCarg _rcarg[3], *_rcargp;
    RCdim3 _gdimrc, _bdimrc;
    int _off = 0;
    int _rcargc = 0;
    void *_devptr;
    _rcargs.RCargs_val = _rcarg;
    _rcargs.RCargs_len = _narg;
    static char mangledname_[512] = {0,};
    if (!mangledname_[0]) {
        if (1) {
          dscudaGetMangledFunctionName(mangledname_, __PRETTY_FUNCTION__, Ptxdata);
        }
        else {
          char buf_[256];
          sprintf(buf_, "%s", __FUNCTION__);
          strcpy(mangledname_, buf_ + strlen("dscuda")); // obtain original function name.
        }
        WARN(3, "mangled name : %s\n", mangledname_);
    }

    if (dscudaRemoteCallType() == RC_REMOTECALL_TYPE_IBV) {

        // a pointer to a device-address 'dscudaAdrOfUva(a)'.
        _ibvargp = _ibvarg + _rcargc;
        _rcargc++;
        _devptr = (void*)(size_t)dscudaAdrOfUva(a);
        ALIGN_UP(_off, __alignof(_devptr));
        _ibvargp->type = dscudaArgTypeP;
        _ibvargp->offset = _off;
        _ibvargp->val.pointerval = (RCadr)_devptr;
        _ibvargp->size = sizeof(_devptr);
        _off += _ibvargp->size;


        // a pointer to a device-address 'dscudaAdrOfUva(b)'.
        _ibvargp = _ibvarg + _rcargc;
        _rcargc++;
        _devptr = (void*)(size_t)dscudaAdrOfUva(b);
        ALIGN_UP(_off, __alignof(_devptr));
        _ibvargp->type = dscudaArgTypeP;
        _ibvargp->offset = _off;
        _ibvargp->val.pointerval = (RCadr)_devptr;
        _ibvargp->size = sizeof(_devptr);
        _off += _ibvargp->size;


        // a pointer to a device-address 'dscudaAdrOfUva(c)'.
        _ibvargp = _ibvarg + _rcargc;
        _rcargc++;
        _devptr = (void*)(size_t)dscudaAdrOfUva(c);
        ALIGN_UP(_off, __alignof(_devptr));
        _ibvargp->type = dscudaArgTypeP;
        _ibvargp->offset = _off;
        _ibvargp->val.pointerval = (RCadr)_devptr;
        _ibvargp->size = sizeof(_devptr);
        _off += _ibvargp->size;

        _ibvgdim[0] = _gdim.x; _ibvgdim[1] = _gdim.y; _ibvgdim[2] = _gdim.z;
        _ibvbdim[0] = _bdim.x; _ibvbdim[1] = _bdim.y; _ibvbdim[2] = _gdim.z;
#if !RPC_ONLY
        ibvDscudaLaunchKernelWrapper(dscudaLoadModule("./dscudatmp/userapp.cu.ptx", Ptxdata), 0, mangledname_,
                                 _ibvgdim, _ibvbdim, _smemsize, (RCstream)_stream,
                                 _narg, _ibvarg);
#endif
    }
    else {

        // a pointer to a device-address 'dscudaAdrOfUva(a)'.
        _rcargp = &(_rcargs.RCargs_val[_rcargc++]);
        _devptr = (void*)(size_t)dscudaAdrOfUva(a);
        ALIGN_UP(_off, __alignof(_devptr));
        _rcargp->val.type = dscudaArgTypeP;
        _rcargp->offset = _off;
        _rcargp->val.RCargVal_u.address = (RCadr)_devptr;
        _rcargp->size = sizeof(_devptr);
        _off += _rcargp->size;

        // a pointer to a device-address 'dscudaAdrOfUva(b)'.
        _rcargp = &(_rcargs.RCargs_val[_rcargc++]);
        _devptr = (void*)(size_t)dscudaAdrOfUva(b);
        ALIGN_UP(_off, __alignof(_devptr));
        _rcargp->val.type = dscudaArgTypeP;
        _rcargp->offset = _off;
        _rcargp->val.RCargVal_u.address = (RCadr)_devptr;
        _rcargp->size = sizeof(_devptr);
        _off += _rcargp->size;

        // a pointer to a device-address 'dscudaAdrOfUva(c)'.
        _rcargp = &(_rcargs.RCargs_val[_rcargc++]);
        _devptr = (void*)(size_t)dscudaAdrOfUva(c);
        ALIGN_UP(_off, __alignof(_devptr));
        _rcargp->val.type = dscudaArgTypeP;
        _rcargp->offset = _off;
        _rcargp->val.RCargVal_u.address = (RCadr)_devptr;
        _rcargp->size = sizeof(_devptr);
        _off += _rcargp->size;
        _gdimrc.x = _gdim.x; _gdimrc.y = _gdim.y; _gdimrc.z = _gdim.z;
        _bdimrc.x = _bdim.x; _bdimrc.y = _bdim.y; _bdimrc.z = _bdim.z;
        rpcDscudaLaunchKernelWrapper(dscudaLoadModule("./dscudatmp/userapp.cu.ptx", Ptxdata), 0, mangledname_,
                                 _gdimrc, _bdimrc, _smemsize, (RCstream)_stream,
                                 _rcargs);
    }
}
  void
vecAdd(float *a, float *b, float *c)
{
    /* nop */
}




/*
 * stub for remote call to vecMul.
 */
void
dscudavecMul(dim3 _gdim, dim3 _bdim, size_t _smemsize, cudaStream_t _stream , float *a, float *b, float c, float *d, int e, int * f)
{
    int _narg = 6;
    int _ibvgdim[3], _ibvbdim[3];
    IbvArg _ibvarg[6], *_ibvargp;
    RCargs _rcargs;
    RCarg _rcarg[6], *_rcargp;
    RCdim3 _gdimrc, _bdimrc;
    int _off = 0;
    int _rcargc = 0;
    void *_devptr;
    _rcargs.RCargs_val = _rcarg;
    _rcargs.RCargs_len = _narg;
    static char mangledname_[512] = {0,};
    if (!mangledname_[0]) {
        if (1) {
          dscudaGetMangledFunctionName(mangledname_, __PRETTY_FUNCTION__, Ptxdata);
        }
        else {
          char buf_[256];
          sprintf(buf_, "%s", __FUNCTION__);
          strcpy(mangledname_, buf_ + strlen("dscuda")); // obtain original function name.
        }
        WARN(3, "mangled name : %s\n", mangledname_);
    }

    if (dscudaRemoteCallType() == RC_REMOTECALL_TYPE_IBV) {

        // a pointer to a device-address 'dscudaAdrOfUva(a)'.
        _ibvargp = _ibvarg + _rcargc;
        _rcargc++;
        _devptr = (void*)(size_t)dscudaAdrOfUva(a);
        ALIGN_UP(_off, __alignof(_devptr));
        _ibvargp->type = dscudaArgTypeP;
        _ibvargp->offset = _off;
        _ibvargp->val.pointerval = (RCadr)_devptr;
        _ibvargp->size = sizeof(_devptr);
        _off += _ibvargp->size;


        // a pointer to a device-address 'dscudaAdrOfUva(b)'.
        _ibvargp = _ibvarg + _rcargc;
        _rcargc++;
        _devptr = (void*)(size_t)dscudaAdrOfUva(b);
        ALIGN_UP(_off, __alignof(_devptr));
        _ibvargp->type = dscudaArgTypeP;
        _ibvargp->offset = _off;
        _ibvargp->val.pointerval = (RCadr)_devptr;
        _ibvargp->size = sizeof(_devptr);
        _off += _ibvargp->size;


        // a float 'c'.
        _ibvargp = _ibvarg + _rcargc;
        _rcargc++;
        ALIGN_UP(_off, __alignof(float));
        _ibvargp->type = dscudaArgTypeF;
        _ibvargp->offset = _off;
        _ibvargp->val.floatval = c;
        _ibvargp->size = sizeof(float);
        _off += _ibvargp->size;


        // a pointer to a device-address 'dscudaAdrOfUva(d)'.
        _ibvargp = _ibvarg + _rcargc;
        _rcargc++;
        _devptr = (void*)(size_t)dscudaAdrOfUva(d);
        ALIGN_UP(_off, __alignof(_devptr));
        _ibvargp->type = dscudaArgTypeP;
        _ibvargp->offset = _off;
        _ibvargp->val.pointerval = (RCadr)_devptr;
        _ibvargp->size = sizeof(_devptr);
        _off += _ibvargp->size;


        // an integer 'e'.
        _ibvargp = _ibvarg + _rcargc;
        _rcargc++;
        ALIGN_UP(_off, __alignof(int));
        _ibvargp->type = dscudaArgTypeI;
        _ibvargp->offset = _off;
        _ibvargp->val.intval = e;
        _ibvargp->size = sizeof(int);
        _off += _ibvargp->size;


        // a pointer to a device-address 'dscudaAdrOfUva(f)'.
        _ibvargp = _ibvarg + _rcargc;
        _rcargc++;
        _devptr = (void*)(size_t)dscudaAdrOfUva(f);
        ALIGN_UP(_off, __alignof(_devptr));
        _ibvargp->type = dscudaArgTypeP;
        _ibvargp->offset = _off;
        _ibvargp->val.pointerval = (RCadr)_devptr;
        _ibvargp->size = sizeof(_devptr);
        _off += _ibvargp->size;

        _ibvgdim[0] = _gdim.x; _ibvgdim[1] = _gdim.y; _ibvgdim[2] = _gdim.z;
        _ibvbdim[0] = _bdim.x; _ibvbdim[1] = _bdim.y; _ibvbdim[2] = _gdim.z;
#if !RPC_ONLY
        ibvDscudaLaunchKernelWrapper(dscudaLoadModule("./dscudatmp/userapp.cu.ptx", Ptxdata), 1, mangledname_,
                                 _ibvgdim, _ibvbdim, _smemsize, (RCstream)_stream,
                                 _narg, _ibvarg);
#endif
    }
    else {

        // a pointer to a device-address 'dscudaAdrOfUva(a)'.
        _rcargp = &(_rcargs.RCargs_val[_rcargc++]);
        _devptr = (void*)(size_t)dscudaAdrOfUva(a);
        ALIGN_UP(_off, __alignof(_devptr));
        _rcargp->val.type = dscudaArgTypeP;
        _rcargp->offset = _off;
        _rcargp->val.RCargVal_u.address = (RCadr)_devptr;
        _rcargp->size = sizeof(_devptr);
        _off += _rcargp->size;

        // a pointer to a device-address 'dscudaAdrOfUva(b)'.
        _rcargp = &(_rcargs.RCargs_val[_rcargc++]);
        _devptr = (void*)(size_t)dscudaAdrOfUva(b);
        ALIGN_UP(_off, __alignof(_devptr));
        _rcargp->val.type = dscudaArgTypeP;
        _rcargp->offset = _off;
        _rcargp->val.RCargVal_u.address = (RCadr)_devptr;
        _rcargp->size = sizeof(_devptr);
        _off += _rcargp->size;

        // a float 'c'.
        _rcargp = &(_rcargs.RCargs_val[_rcargc++]);
        ALIGN_UP(_off, __alignof(float));
        _rcargp->val.type = dscudaArgTypeF;
        _rcargp->offset = _off;
        _rcargp->val.RCargVal_u.valuef = c;
        _rcargp->size = sizeof(float);
        _off += _rcargp->size;

        // a pointer to a device-address 'dscudaAdrOfUva(d)'.
        _rcargp = &(_rcargs.RCargs_val[_rcargc++]);
        _devptr = (void*)(size_t)dscudaAdrOfUva(d);
        ALIGN_UP(_off, __alignof(_devptr));
        _rcargp->val.type = dscudaArgTypeP;
        _rcargp->offset = _off;
        _rcargp->val.RCargVal_u.address = (RCadr)_devptr;
        _rcargp->size = sizeof(_devptr);
        _off += _rcargp->size;

        // an integer 'e'.
        _rcargp = &(_rcargs.RCargs_val[_rcargc++]);
        ALIGN_UP(_off, __alignof(int));
        _rcargp->val.type = dscudaArgTypeI;
        _rcargp->offset = _off;
        _rcargp->val.RCargVal_u.valuei = e;
        _rcargp->size = sizeof(int);
        _off += _rcargp->size;

        // a pointer to a device-address 'dscudaAdrOfUva(f)'.
        _rcargp = &(_rcargs.RCargs_val[_rcargc++]);
        _devptr = (void*)(size_t)dscudaAdrOfUva(f);
        ALIGN_UP(_off, __alignof(_devptr));
        _rcargp->val.type = dscudaArgTypeP;
        _rcargp->offset = _off;
        _rcargp->val.RCargVal_u.address = (RCadr)_devptr;
        _rcargp->size = sizeof(_devptr);
        _off += _rcargp->size;
        _gdimrc.x = _gdim.x; _gdimrc.y = _gdim.y; _gdimrc.z = _gdim.z;
        _bdimrc.x = _bdim.x; _bdimrc.y = _bdim.y; _bdimrc.z = _bdim.z;
        rpcDscudaLaunchKernelWrapper(dscudaLoadModule("./dscudatmp/userapp.cu.ptx", Ptxdata), 1, mangledname_,
                                 _gdimrc, _bdimrc, _smemsize, (RCstream)_stream,
                                 _rcargs);
    }
}
  void
vecMul(float *a, float *b, float c, float *d, int e, int * f)
{
    /* nop */
}



#pragma end userapp.cuh

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
        dscudavecAdd(grids, threads, 0, NULL, d_a, d_b, d_c);
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
