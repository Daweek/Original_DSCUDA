static char *Ptxdata = 
    "	.version 1.4\n"
    "	.target sm_10, map_f64_to_f32\n"
    "	// compiled with /usr/local/cuda4.1/cuda/open64/lib//be\n"
    "	// nvopencc 4.1 built on 2012-01-12\n"
    "\n"
    "	//-----------------------------------------------------------\n"
    "	// Compiling /tmp/tmpxft_0000756a_00000000-9_userapp.cpp3.i (/tmp/ccBI#.PXPWRK)\n"
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
    "	.file	2	\"/tmp/tmpxft_0000756a_00000000-8_userapp.cudafe2.gpu\"\n"
    "	.file	3	\"/usr/lib/gcc/x86_64-redhat-linux/4.5.1/include/stddef.h\"\n"
    "	.file	4	\"/usr/local/cuda4.1/cuda/bin/../include/crt/device_runtime.h\"\n"
    "	.file	5	\"/usr/local/cuda4.1/cuda/bin/../include/host_defines.h\"\n"
    "	.file	6	\"/usr/local/cuda4.1/cuda/bin/../include/builtin_types.h\"\n"
    "	.file	7	\"/usr/local/cuda4.1/cuda/bin/../include/device_types.h\"\n"
    "	.file	8	\"/usr/local/cuda4.1/cuda/bin/../include/driver_types.h\"\n"
    "	.file	9	\"/usr/local/cuda4.1/cuda/bin/../include/surface_types.h\"\n"
    "	.file	10	\"/usr/local/cuda4.1/cuda/bin/../include/texture_types.h\"\n"
    "	.file	11	\"/usr/local/cuda4.1/cuda/bin/../include/vector_types.h\"\n"
    "	.file	12	\"/usr/local/cuda4.1/cuda/bin/../include/device_launch_parameters.h\"\n"
    "	.file	13	\"/usr/local/cuda4.1/cuda/bin/../include/crt/storage_class.h\"\n"
    "	.file	14	\"userapp.cuh\"\n"
    "	.file	15	\"/usr/local/cuda4.1/cuda/bin/../include/common_functions.h\"\n"
    "	.file	16	\"/usr/local/cuda4.1/cuda/bin/../include/math_functions.h\"\n"
    "	.file	17	\"/usr/local/cuda4.1/cuda/bin/../include/math_constants.h\"\n"
    "	.file	18	\"/usr/local/cuda4.1/cuda/bin/../include/device_functions.h\"\n"
    "	.file	19	\"/usr/local/cuda4.1/cuda/bin/../include/sm_11_atomic_functions.h\"\n"
    "	.file	20	\"/usr/local/cuda4.1/cuda/bin/../include/sm_12_atomic_functions.h\"\n"
    "	.file	21	\"/usr/local/cuda4.1/cuda/bin/../include/sm_13_double_functions.h\"\n"
    "	.file	22	\"/usr/local/cuda4.1/cuda/bin/../include/sm_20_atomic_functions.h\"\n"
    "	.file	23	\"/usr/local/cuda4.1/cuda/bin/../include/sm_20_intrinsics.h\"\n"
    "	.file	24	\"/usr/local/cuda4.1/cuda/bin/../include/surface_functions.h\"\n"
    "	.file	25	\"/usr/local/cuda4.1/cuda/bin/../include/texture_fetch_functions.h\"\n"
    "	.file	26	\"/usr/local/cuda4.1/cuda/bin/../include/math_functions_dbl_ptx1.h\"\n"
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
#include "dscuda.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
// #include <cutil.h>
// #include <cutil_inline.h>

#define cutilSafeCall // nop

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
