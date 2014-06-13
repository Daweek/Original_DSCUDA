static char *Ptxdata = 
    "	.version 1.4\n"
    "	.target sm_10, map_f64_to_f32\n"
    "	// compiled with /usr/local/cuda4.1/cuda/open64/lib//be\n"
    "	// nvopencc 4.1 built on 2012-01-12\n"
    "\n"
    "	//-----------------------------------------------------------\n"
    "	// Compiling /tmp/tmpxft_000076cf_00000000-9_direct.cpp3.i (/tmp/ccBI#.9lbT3b)\n"
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
    "	.file	2	\"/tmp/tmpxft_000076cf_00000000-8_direct.cudafe2.gpu\"\n"
    "	.file	3	\"/usr/lib/gcc/x86_64-redhat-linux/4.5.1/include/stddef.h\"\n"
    "	.file	4	\"/usr/local/cuda4.1/cuda/include/crt/device_runtime.h\"\n"
    "	.file	5	\"/usr/local/cuda4.1/cuda/include/host_defines.h\"\n"
    "	.file	6	\"/usr/local/cuda4.1/cuda/include/builtin_types.h\"\n"
    "	.file	7	\"/usr/local/cuda4.1/cuda/include/device_types.h\"\n"
    "	.file	8	\"/usr/local/cuda4.1/cuda/include/driver_types.h\"\n"
    "	.file	9	\"/usr/local/cuda4.1/cuda/include/surface_types.h\"\n"
    "	.file	10	\"/usr/local/cuda4.1/cuda/include/texture_types.h\"\n"
    "	.file	11	\"/usr/local/cuda4.1/cuda/include/vector_types.h\"\n"
    "	.file	12	\"/usr/local/cuda4.1/cuda/include/device_launch_parameters.h\"\n"
    "	.file	13	\"/usr/local/cuda4.1/cuda/include/crt/storage_class.h\"\n"
    "	.file	14	\"direct.cu\"\n"
    "	.file	15	\"/usr/local/cuda4.1/cuda/include/common_functions.h\"\n"
    "	.file	16	\"/usr/local/cuda4.1/cuda/include/math_functions.h\"\n"
    "	.file	17	\"/usr/local/cuda4.1/cuda/include/math_constants.h\"\n"
    "	.file	18	\"/usr/local/cuda4.1/cuda/include/device_functions.h\"\n"
    "	.file	19	\"/usr/local/cuda4.1/cuda/include/sm_11_atomic_functions.h\"\n"
    "	.file	20	\"/usr/local/cuda4.1/cuda/include/sm_12_atomic_functions.h\"\n"
    "	.file	21	\"/usr/local/cuda4.1/cuda/include/sm_13_double_functions.h\"\n"
    "	.file	22	\"/usr/local/cuda4.1/cuda/include/sm_20_atomic_functions.h\"\n"
    "	.file	23	\"/usr/local/cuda4.1/cuda/include/sm_20_intrinsics.h\"\n"
    "	.file	24	\"/usr/local/cuda4.1/cuda/include/surface_functions.h\"\n"
    "	.file	25	\"/usr/local/cuda4.1/cuda/include/texture_fetch_functions.h\"\n"
    "	.file	26	\"/usr/local/cuda4.1/cuda/include/math_functions_dbl_ptx1.h\"\n"
    "\n"
    "\n"
    "	.entry _Z14gravity_kernelPfPA3_ffS1_S_i (\n"
    "		.param .u64 __cudaparm__Z14gravity_kernelPfPA3_ffS1_S_i_m,\n"
    "		.param .u64 __cudaparm__Z14gravity_kernelPfPA3_ffS1_S_i_x,\n"
    "		.param .f32 __cudaparm__Z14gravity_kernelPfPA3_ffS1_S_i_eps,\n"
    "		.param .u64 __cudaparm__Z14gravity_kernelPfPA3_ffS1_S_i_a,\n"
    "		.param .u64 __cudaparm__Z14gravity_kernelPfPA3_ffS1_S_i_p,\n"
    "		.param .s32 __cudaparm__Z14gravity_kernelPfPA3_ffS1_S_i_n)\n"
    "	{\n"
    "	.reg .u16 %rh<4>;\n"
    "	.reg .u32 %r<12>;\n"
    "	.reg .u64 %rd<13>;\n"
    "	.reg .f32 %f<37>;\n"
    "	.reg .pred %p<4>;\n"
    "	.loc	14	13	0\n"
    "$LDWbegin__Z14gravity_kernelPfPA3_ffS1_S_i:\n"
    "	.loc	14	52	0\n"
    "	mov.f32 	%f1, 0f00000000;     	// 0\n"
    "	mov.f32 	%f2, %f1;\n"
    "	mov.f32 	%f3, 0f00000000;     	// 0\n"
    "	mov.f32 	%f4, %f3;\n"
    "	mov.f32 	%f5, 0f00000000;     	// 0\n"
    "	mov.f32 	%f6, %f5;\n"
    "	cvt.u32.u16 	%r1, %tid.x;\n"
    "	mov.u16 	%rh1, %ntid.x;\n"
    "	mov.u16 	%rh2, %ctaid.x;\n"
    "	ld.param.s32 	%r2, [__cudaparm__Z14gravity_kernelPfPA3_ffS1_S_i_n];\n"
    "	mov.u32 	%r3, 0;\n"
    "	setp.le.s32 	%p1, %r2, %r3;\n"
    "	@%p1 bra 	$Lt_0_7426;\n"
    "	ld.param.s32 	%r2, [__cudaparm__Z14gravity_kernelPfPA3_ffS1_S_i_n];\n"
    "	mov.s32 	%r4, %r2;\n"
    "	mul.wide.u16 	%r5, %rh1, %rh2;\n"
    "	ld.param.f32 	%f7, [__cudaparm__Z14gravity_kernelPfPA3_ffS1_S_i_eps];\n"
    "	mul.f32 	%f8, %f7, %f7;\n"
    "	add.u32 	%r6, %r5, %r1;\n"
    "	cvt.s64.s32 	%rd1, %r6;\n"
    "	ld.param.u64 	%rd2, [__cudaparm__Z14gravity_kernelPfPA3_ffS1_S_i_x];\n"
    "	mov.s64 	%rd3, %rd2;\n"
    "	ld.param.u64 	%rd4, [__cudaparm__Z14gravity_kernelPfPA3_ffS1_S_i_m];\n"
    "	mul.wide.s32 	%rd5, %r6, 12;\n"
    "	add.u64 	%rd6, %rd5, %rd2;\n"
    "	ld.global.f32 	%f9, [%rd6+0];\n"
    "	ld.global.f32 	%f10, [%rd6+4];\n"
    "	ld.global.f32 	%f11, [%rd6+8];\n"
    "	mov.s32 	%r7, 0;\n"
    "	mov.f32 	%f12, 0f00000000;    	// 0\n"
    "	mov.s32 	%r8, %r4;\n"
    "$Lt_0_6914:\n"
    " //<loop> Loop body line 52, nesting depth: 1, estimated iterations: unknown\n"
    "	.loc	14	63	0\n"
    "	ld.global.f32 	%f13, [%rd3+0];\n"
    "	ld.global.f32 	%f14, [%rd3+4];\n"
    "	ld.global.f32 	%f15, [%rd3+8];\n"
    "	sub.f32 	%f16, %f13, %f9;\n"
    "	sub.f32 	%f17, %f14, %f10;\n"
    "	sub.f32 	%f18, %f15, %f11;\n"
    "	mad.f32 	%f19, %f16, %f16, %f8;\n"
    "	mad.f32 	%f20, %f17, %f17, %f19;\n"
    "	mad.f32 	%f21, %f18, %f18, %f20;\n"
    "	.loc	14	66	0\n"
    "	ld.global.f32 	%f22, [%rd4+0];\n"
    "	.loc	14	68	0\n"
    "	sqrt.approx.f32 	%f23, %f21;\n"
    "	mul.f32 	%f24, %f23, %f21;\n"
    "	div.full.f32 	%f25, %f22, %f24;\n"
    "	mov.f32 	%f26, %f2;\n"
    "	mad.f32 	%f27, %f16, %f25, %f26;\n"
    "	mov.f32 	%f2, %f27;\n"
    "	mov.f32 	%f28, %f4;\n"
    "	mad.f32 	%f29, %f17, %f25, %f28;\n"
    "	mov.f32 	%f4, %f29;\n"
    "	mov.f32 	%f30, %f6;\n"
    "	mad.f32 	%f31, %f18, %f25, %f30;\n"
    "	mov.f32 	%f6, %f31;\n"
    "	.loc	14	70	0\n"
    "	div.full.f32 	%f32, %f22, %f23;\n"
    "	sub.f32 	%f12, %f12, %f32;\n"
    "	add.s32 	%r7, %r7, 1;\n"
    "	add.u64 	%rd4, %rd4, 4;\n"
    "	add.u64 	%rd3, %rd3, 12;\n"
    "	.loc	14	52	0\n"
    "	ld.param.s32 	%r2, [__cudaparm__Z14gravity_kernelPfPA3_ffS1_S_i_n];\n"
    "	.loc	14	70	0\n"
    "	setp.ne.s32 	%p2, %r2, %r7;\n"
    "	@%p2 bra 	$Lt_0_6914;\n"
    "	bra.uni 	$Lt_0_6402;\n"
    "$Lt_0_7426:\n"
    "	mul.wide.u16 	%r9, %rh1, %rh2;\n"
    "	add.u32 	%r10, %r1, %r9;\n"
    "	cvt.s64.s32 	%rd1, %r10;\n"
    "	mul.wide.s32 	%rd5, %r10, 12;\n"
    "	mov.f32 	%f12, 0f00000000;    	// 0\n"
    "$Lt_0_6402:\n"
    "	.loc	14	73	0\n"
    "	ld.param.u64 	%rd7, [__cudaparm__Z14gravity_kernelPfPA3_ffS1_S_i_a];\n"
    "	add.u64 	%rd8, %rd7, %rd5;\n"
    "	mov.f32 	%f33, %f2;\n"
    "	st.global.f32 	[%rd8+0], %f33;\n"
    "	mov.f32 	%f34, %f4;\n"
    "	st.global.f32 	[%rd8+4], %f34;\n"
    "	mov.f32 	%f35, %f6;\n"
    "	st.global.f32 	[%rd8+8], %f35;\n"
    "	.loc	14	75	0\n"
    "	ld.param.u64 	%rd9, [__cudaparm__Z14gravity_kernelPfPA3_ffS1_S_i_p];\n"
    "	mul.lo.u64 	%rd10, %rd1, 4;\n"
    "	add.u64 	%rd11, %rd9, %rd10;\n"
    "	st.global.f32 	[%rd11+0], %f12;\n"
    "	.loc	14	78	0\n"
    "	exit;\n"
    "$LDWend__Z14gravity_kernelPfPA3_ffS1_S_i:\n"
    "	} // _Z14gravity_kernelPfPA3_ffS1_S_i\n"
    "\n";
#pragma dscuda endofptx
#include "dscuda.h"
#include <stdio.h>
#include <stdlib.h>
// #include <cutil.h>
// #include <cutil_inline.h>
#define cutilSafeCall // nop
#include "direct.h"

__global__ void gravity_kernel(float *m, float (*x)[3], float eps, float (*a)[3], float *p, int n);
static void calc_gravity_gpu(double *m, double (*x)[3], double eps, double (*a)[3], double *p, int n);
static void calc_gravity(double *m, double (*x)[3], double eps, double (*a)[3], double *p, int n);

__global__ void
gravity_kernel(float *m, float (*x)[3], float eps, float (*a)[3], float *p, int n)
{
#if 0 // naive implementation.

    float r, r2, mf, dx[3];
    int i, j, k;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    for (k = 0; k < 3; k++) {
        a[i][k] = 0.0;
    }
    p[i] = 0.0;

    for (j = 0; j < n; j++) {
        if (i == j) continue;
        for (k = 0; k < 3; k++) {
            dx[k] = x[j][k] - x[i][k];
        }
        r2 = eps * eps;
        for (k = 0; k < 3; k++) {
            r2 += dx[k] * dx[k];
        }            
        r = sqrtf(r2);
        mf = m[j] / (r * r2);
        for (k = 0; k < 3; k++) {
            a[i][k] += mf * dx[k];
        }            
        p[i] -= m[j] / r;
    }

#else // this gives better performance.

    float r, r2, mf, dx[3], atmp[3], ptmp;
    int i, j, k;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    for (k = 0; k < 3; k++) {
        atmp[k] = 0.0;
    }
    ptmp = 0.0;

    for (j = 0; j < n; j++) {
        //        if (i == j) continue; // !!!
        for (k = 0; k < 3; k++) {
            dx[k] = x[j][k] - x[i][k];
        }
        r2 = eps * eps;
        for (k = 0; k < 3; k++) {
            r2 += dx[k] * dx[k];
        }            
        r = sqrtf(r2);
        mf = m[j] / (r * r2);
        for (k = 0; k < 3; k++) {
            atmp[k] += mf * dx[k];
        }            
        ptmp -= m[j] / r;
    }
    for (k = 0; k < 3; k++) {
        a[i][k] = atmp[k];
    }
    p[i] = ptmp;

#endif
}

static void
calc_gravity_gpu(double *m, double (*x)[3], double eps, double (*a)[3], double *p, int n)
{
    static int firstcall = 1;
    static float *d_m, (*d_x)[3], (*d_a)[3], *d_p;
    static float floatbuf[NMAX*3];
    int i, k;
    int nth = 64;
    dim3 threads(nth, 1, 1);
    dim3 grids((n+nth-1)/nth, 1, 1);

    if (firstcall) {
        firstcall = 0;
        cutilSafeCall(cudaMalloc((void**)&d_m, sizeof(float) * n));
        cutilSafeCall(cudaMalloc((void**)&d_x, sizeof(float) * 3 * n));
        cutilSafeCall(cudaMalloc((void**)&d_a, sizeof(float) * 3 * n));
        cutilSafeCall(cudaMalloc((void**)&d_p, sizeof(float) * n));
    }
    for (i = 0 ; i < n; i++) {
        floatbuf[i] = (float)m[i];
    }
    cutilSafeCall(cudaMemcpy(d_m, floatbuf, sizeof(float) * n, cudaMemcpyHostToDevice));

    for (i = 0 ; i < n; i++) {
        for (k = 0; k < 3; k++) {
            floatbuf[3 * i + k] = (float)x[i][k];
        }
    }
    cutilSafeCall(cudaMemcpy(d_x, floatbuf, sizeof(float) * 3 * n, cudaMemcpyHostToDevice));

    gravity_kernel<<<grids, threads>>>(d_m, d_x, (float)eps, d_a, d_p, n);

    cutilSafeCall(cudaMemcpy(floatbuf, d_a, sizeof(float) * 3 * n, cudaMemcpyDeviceToHost));
    for (i = 0 ; i < n; i++) {
        for (k = 0; k < 3; k++) {
            a[i][k] = (double)floatbuf[3 * i + k];
        }
    }

    cutilSafeCall(cudaMemcpy(floatbuf, d_p, sizeof(float) * n, cudaMemcpyDeviceToHost));
    for (i = 0 ; i < n; i++) {
        p[i]= (double)floatbuf[i];
    }
}

static void
calc_gravity(double *m, double (*x)[3], double eps, double (*a)[3], double *p, int n)
{
    double r, r2, mf, dx[3];
    int i, j, k;

    for (i = 0; i < n; i++) {
        for (k = 0; k < 3; k++) {
            a[i][k] = 0.0;
        }
        p[i] = 0.0;
        for (j = 0; j < n; j++) {
            for (k = 0; k < 3; k++) {
                dx[k] = x[j][k] - x[i][k];
            }
            r2 = eps * eps;
            for (k = 0; k < 3; k++) {
                r2 += dx[k] * dx[k];
            }            
            r = sqrt(r2);
            mf = m[j] / (r * r2);
            for (k = 0; k < 3; k++) {
                a[i][k] += mf * dx[k];
            }            
            p[i] -= m[j] / r;
        }
    }

    if (eps != 0.0) {
        double epsinv;
        epsinv = 1.0 / eps;
        for (i = 0; i < n; i++) {
            p[i] += m[i] * epsinv;
        }
    }
}

#ifdef __DSCUDA__
static void
errhandler(void *arg)
{
    fprintf(stderr, "calculation error on some GPU at timestep: %d\n",
            *(int *)arg);
    exit(1);
}
#endif

int
main(int argc, char **argv)
{
    static double mj[NMAX], xj[NMAX][3], vj[NMAX][3];
    static double a[NMAX][3], p[NMAX];
    double time, dt, endt;;
    double eps;
    double e, e0, ke, pe;
    double lt=0.0, st=0.0, sustained;
    int n, nstep, interval;
    static int step;

#ifdef __DSCUDA__
    dscudaSetErrorHandler(errhandler, (void *)&step);
#endif

    eps = 0.02;
    dt = 0.01;
    endt = 1.1;
    //    endt = 10000.1;
    time = 0.0;
    nstep = endt/dt;

    if (argc < 3) {
        fprintf(stderr, "performs gravitational N-body simulation with naive direct summation algorithm.\n"
                "usage: %s <infile> <outfile>\n",  argv[0]);
        exit(1);
    }
  
    readnbody(&n, mj, xj, vj, argv[1]);
    interval = 500 * (10000.0/n) * (10000.0/n);    
    if (interval * 10 > nstep) {
	interval = nstep / 10;
    }
    interval = 1;
    fprintf(stderr, "interval: %d\n", interval);

    get_cputime(&lt,&st);
#if 1
    calc_gravity_gpu(mj, xj, eps, a, p, n);
#else
    calc_gravity(mj, xj, eps, a, p, n);
#endif
    energy(mj, vj, p, n, &ke, &pe);
    e0 = ke+pe;
    printf("ke: %f  pe: %f  e0: %f\n", ke, pe, e0);
    for (step = 1; step < nstep; step++) {
        push_velocity(vj, a, 0.5*dt, n);
        push_position(xj, vj, a, dt, n);
        time = time + dt;
#if 1
        calc_gravity_gpu(mj, xj, eps, a, p, n);
#else
        calc_gravity(mj, xj, eps, a, p, n);
#endif

        push_velocity(vj, a, 0.5*dt, n);

        if (step % interval == 0) {
            energy(mj, vj, p, n, &ke, &pe);
            e = ke+pe;
            sustained = 38.0*((double)n)*((double)n)*interval/lt/1e9;
            printf("speed: %g Gflops\n", sustained);
            printf("step: %d time: %e\n", step, time);
            printf("e: %e de: %e\n", e, e-e0);
            printf("ke: %e pe: %e\n", ke, pe);
            printf("ke/pe: %e\n\n", ke/pe);
            get_cputime(&lt,&st);
        }
    }
    writenbody(n, mj, xj, vj, argv[2]);
}
