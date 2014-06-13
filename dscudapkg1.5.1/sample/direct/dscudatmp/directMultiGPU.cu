static char *Ptxdata = 
    "	.version 1.4\n"
    "	.target sm_10, map_f64_to_f32\n"
    "	// compiled with /usr/local/cuda4.1/cuda/open64/lib//be\n"
    "	// nvopencc 4.1 built on 2012-01-12\n"
    "\n"
    "	//-----------------------------------------------------------\n"
    "	// Compiling /tmp/tmpxft_00007623_00000000-9_directMultiGPU.cpp3.i (/tmp/ccBI#.WF1789)\n"
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
    "	.file	2	\"/tmp/tmpxft_00007623_00000000-8_directMultiGPU.cudafe2.gpu\"\n"
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
    "	.file	14	\"directMultiGPU.cu\"\n"
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
    "	.entry _Z14gravity_kernelPfPA3_ffS1_S_ii (\n"
    "		.param .u64 __cudaparm__Z14gravity_kernelPfPA3_ffS1_S_ii_m,\n"
    "		.param .u64 __cudaparm__Z14gravity_kernelPfPA3_ffS1_S_ii_x,\n"
    "		.param .f32 __cudaparm__Z14gravity_kernelPfPA3_ffS1_S_ii_eps,\n"
    "		.param .u64 __cudaparm__Z14gravity_kernelPfPA3_ffS1_S_ii_a,\n"
    "		.param .u64 __cudaparm__Z14gravity_kernelPfPA3_ffS1_S_ii_p,\n"
    "		.param .s32 __cudaparm__Z14gravity_kernelPfPA3_ffS1_S_ii_nj,\n"
    "		.param .s32 __cudaparm__Z14gravity_kernelPfPA3_ffS1_S_ii_ioff)\n"
    "	{\n"
    "	.reg .u16 %rh<4>;\n"
    "	.reg .u32 %r<13>;\n"
    "	.reg .u64 %rd<15>;\n"
    "	.reg .f32 %f<37>;\n"
    "	.reg .pred %p<4>;\n"
    "	.loc	14	20	0\n"
    "$LDWbegin__Z14gravity_kernelPfPA3_ffS1_S_ii:\n"
    "	.loc	14	29	0\n"
    "	mov.f32 	%f1, 0f00000000;     	// 0\n"
    "	mov.f32 	%f2, %f1;\n"
    "	mov.f32 	%f3, 0f00000000;     	// 0\n"
    "	mov.f32 	%f4, %f3;\n"
    "	mov.f32 	%f5, 0f00000000;     	// 0\n"
    "	mov.f32 	%f6, %f5;\n"
    "	cvt.u32.u16 	%r1, %tid.x;\n"
    "	mov.u16 	%rh1, %ntid.x;\n"
    "	mov.u16 	%rh2, %ctaid.x;\n"
    "	ld.param.s32 	%r2, [__cudaparm__Z14gravity_kernelPfPA3_ffS1_S_ii_nj];\n"
    "	mov.u32 	%r3, 0;\n"
    "	setp.le.s32 	%p1, %r2, %r3;\n"
    "	@%p1 bra 	$Lt_0_7426;\n"
    "	ld.param.s32 	%r2, [__cudaparm__Z14gravity_kernelPfPA3_ffS1_S_ii_nj];\n"
    "	mov.s32 	%r4, %r2;\n"
    "	mul.wide.u16 	%r5, %rh1, %rh2;\n"
    "	ld.param.f32 	%f7, [__cudaparm__Z14gravity_kernelPfPA3_ffS1_S_ii_eps];\n"
    "	mul.f32 	%f8, %f7, %f7;\n"
    "	add.u32 	%r6, %r5, %r1;\n"
    "	ld.param.u64 	%rd1, [__cudaparm__Z14gravity_kernelPfPA3_ffS1_S_ii_x];\n"
    "	mov.s64 	%rd2, %rd1;\n"
    "	ld.param.u64 	%rd3, [__cudaparm__Z14gravity_kernelPfPA3_ffS1_S_ii_m];\n"
    "	ld.param.s32 	%r7, [__cudaparm__Z14gravity_kernelPfPA3_ffS1_S_ii_ioff];\n"
    "	add.s32 	%r8, %r7, %r6;\n"
    "	cvt.s64.s32 	%rd4, %r8;\n"
    "	mul.wide.s32 	%rd5, %r8, 12;\n"
    "	add.u64 	%rd6, %rd1, %rd5;\n"
    "	ld.global.f32 	%f9, [%rd6+0];\n"
    "	ld.global.f32 	%f10, [%rd6+4];\n"
    "	ld.global.f32 	%f11, [%rd6+8];\n"
    "	mov.s32 	%r9, 0;\n"
    "	mov.f32 	%f12, 0f00000000;    	// 0\n"
    "	mov.s32 	%r10, %r4;\n"
    "$Lt_0_6914:\n"
    " //<loop> Loop body line 29, nesting depth: 1, estimated iterations: unknown\n"
    "	.loc	14	40	0\n"
    "	ld.global.f32 	%f13, [%rd2+0];\n"
    "	ld.global.f32 	%f14, [%rd2+4];\n"
    "	ld.global.f32 	%f15, [%rd2+8];\n"
    "	sub.f32 	%f16, %f13, %f9;\n"
    "	sub.f32 	%f17, %f14, %f10;\n"
    "	sub.f32 	%f18, %f15, %f11;\n"
    "	mad.f32 	%f19, %f16, %f16, %f8;\n"
    "	mad.f32 	%f20, %f17, %f17, %f19;\n"
    "	mad.f32 	%f21, %f18, %f18, %f20;\n"
    "	.loc	14	43	0\n"
    "	ld.global.f32 	%f22, [%rd3+0];\n"
    "	.loc	14	45	0\n"
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
    "	.loc	14	47	0\n"
    "	div.full.f32 	%f32, %f22, %f23;\n"
    "	sub.f32 	%f12, %f12, %f32;\n"
    "	add.s32 	%r9, %r9, 1;\n"
    "	add.u64 	%rd3, %rd3, 4;\n"
    "	add.u64 	%rd2, %rd2, 12;\n"
    "	.loc	14	29	0\n"
    "	ld.param.s32 	%r2, [__cudaparm__Z14gravity_kernelPfPA3_ffS1_S_ii_nj];\n"
    "	.loc	14	47	0\n"
    "	setp.ne.s32 	%p2, %r2, %r9;\n"
    "	@%p2 bra 	$Lt_0_6914;\n"
    "	bra.uni 	$Lt_0_6402;\n"
    "$Lt_0_7426:\n"
    "	mul.wide.u16 	%r11, %rh1, %rh2;\n"
    "	add.u32 	%r6, %r1, %r11;\n"
    "	mov.f32 	%f12, 0f00000000;    	// 0\n"
    "$Lt_0_6402:\n"
    "	.loc	14	50	0\n"
    "	cvt.s64.s32 	%rd7, %r6;\n"
    "	ld.param.u64 	%rd8, [__cudaparm__Z14gravity_kernelPfPA3_ffS1_S_ii_a];\n"
    "	mul.wide.s32 	%rd9, %r6, 12;\n"
    "	add.u64 	%rd10, %rd8, %rd9;\n"
    "	mov.f32 	%f33, %f2;\n"
    "	st.global.f32 	[%rd10+0], %f33;\n"
    "	mov.f32 	%f34, %f4;\n"
    "	st.global.f32 	[%rd10+4], %f34;\n"
    "	mov.f32 	%f35, %f6;\n"
    "	st.global.f32 	[%rd10+8], %f35;\n"
    "	.loc	14	52	0\n"
    "	ld.param.u64 	%rd11, [__cudaparm__Z14gravity_kernelPfPA3_ffS1_S_ii_p];\n"
    "	mul.wide.s32 	%rd12, %r6, 4;\n"
    "	add.u64 	%rd13, %rd11, %rd12;\n"
    "	st.global.f32 	[%rd13+0], %f12;\n"
    "	.loc	14	53	0\n"
    "	exit;\n"
    "$LDWend__Z14gravity_kernelPfPA3_ffS1_S_ii:\n"
    "	} // _Z14gravity_kernelPfPA3_ffS1_S_ii\n"
    "\n";
#pragma dscuda endofptx
#include "dscuda.h"
#include <stdio.h>
#include <stdlib.h>
// #include <cutil.h>
// #include <cutil_inline.h>
#define cutilSafeCall // nop
#include "direct.h"

__global__ void gravity_kernel(float *m, float (*x)[3], float eps, float (*a)[3], float *p, int n, int ioff);
static void calc_gravity_gpus(double *m, double (*x)[3], double eps, double (*a)[3], double *p, int n, int ndev_req);
static void calc_gravity(double *m, double (*x)[3], double eps, double (*a)[3], double *p, int n);

cudaError_t cudaMemcpyToAlldev(int ndev, void **dst, const void *src, size_t count, enum cudaMemcpyKind kind);

#define TREEPERFTEST (0) // only for performance estimate for the treecode.
#if TREEPERFTEST
static int Listlen;
#endif

__global__ void
gravity_kernel(float *m, float (*x)[3], float eps, float (*a)[3], float *p, int nj, int ioff)
{
    float r, r2, mf, dx[3], atmp[3], ptmp;
    int locali, i, j, k;

    locali = blockIdx.x * blockDim.x + threadIdx.x;
    i = locali + ioff;

    for (k = 0; k < 3; k++) {
        atmp[k] = 0.0;
    }
    ptmp = 0.0;

    for (j = 0; j < nj; j++) {
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
        a[locali][k] = atmp[k];
    }
    p[locali] = ptmp;
}


static int
roundUp(int src, int by)
{
    return ((src - 1) / by + 1) * by;
}

typedef struct {
    float *m;
    float (*x)[3];
    float (*a)[3];
    float *p;
    float *buf;
} Device_t;

static void
calc_gravity_gpus(double *m, double (*x)[3], double eps, double (*a)[3], double *p, int n, int ndev_req)
{
    static int firstcall = 1;
    static Device_t device[NDEVMAX];
    static int ndev;
    static int ni, nj;
    int nth = 64;
    int i, k, idev, ioff;

#if TREEPERFTEST
    nj = Listlen;
#endif

    if (firstcall) {
        firstcall = 0;
	cutilSafeCall(cudaGetDeviceCount(&ndev));
	if (ndev < 1) {
            fprintf(stderr, "No CUDA device found.\n");
            exit(1);
	}
	else if (NDEVMAX < ndev) {
            fprintf(stderr, "Too many CUDA devices (%d) found.\n", ndev);
            exit(1);
	}
	else {
            fprintf(stderr, "%d CUDA devices found.\n", ndev);
	}

        if (ndev < ndev_req) {
            fprintf(stderr, "too many devices (=%d) requested.\n", ndev_req);
            exit(1);
        }
        ndev = ndev_req;
        fprintf(stderr, "use %d CUDA devices.\n", ndev);

	ni = (n - 1) / ndev + 1;
	nj = n;
	int njru = roundUp(nj, nth);
	int niru = roundUp(ni, nth);
#pragma omp parallel for
	for (idev = 0; idev < ndev; idev++) {
            Device_t *dev = device + idev;
            cudaSetDevice(idev);
#if 0 // malloc size for the last device probably be wroing. run with n larger than ~64k sometimes fails.
            cutilSafeCall(cudaMalloc((void**)&dev->m, sizeof(float) * njru));
            cutilSafeCall(cudaMalloc((void**)&dev->x, sizeof(float) * 3 * njru));
            cutilSafeCall(cudaMalloc((void**)&dev->a, sizeof(float) * 3 * niru));
            cutilSafeCall(cudaMalloc((void**)&dev->p, sizeof(float) * niru));
            dev->buf = (float *)malloc(sizeof(float) * 3 * n);
#else // alloc large enough buffer. this is too much. should be refined.
            cutilSafeCall(cudaMalloc((void**)&dev->m, 2 * sizeof(float) * njru));
            cutilSafeCall(cudaMalloc((void**)&dev->x, 2 * sizeof(float) * 3 * njru));
            cutilSafeCall(cudaMalloc((void**)&dev->a, 2 * sizeof(float) * 3 * niru));
            cutilSafeCall(cudaMalloc((void**)&dev->p, 2 * sizeof(float) * niru));
            dev->buf = (float *)malloc(2 * sizeof(float) * 3 * n);
#endif
	}
    }

    // send JPs
#pragma omp parallel for private(i, k)
    for (idev = 0; idev < ndev; idev++) {
        Device_t *dev = device + idev;

        cudaSetDevice(idev);

        for (i = 0 ; i < nj; i++) {
            dev->buf[i] = (float)m[i];
        }
        cutilSafeCall(cudaMemcpy(dev->m, dev->buf, sizeof(float) * nj, cudaMemcpyHostToDevice));

        for (i = 0 ; i < nj; i++) {
            for (k = 0; k < 3; k++) {
                dev->buf[3 * i + k] = (float)x[i][k];
            }
        }
        cutilSafeCall(cudaMemcpy(dev->x, dev->buf, sizeof(float) * 3 * nj, cudaMemcpyHostToDevice));
    }

    // i-parallelism kernel execution.
#pragma omp parallel for private(ioff)
    for (idev = 0; idev < ndev; idev++) {
        dim3 threads(nth, 1, 1);
        dim3 grids((ni + nth - 1) / nth, 1, 1);
        Device_t *dev = device + idev;
        ioff = ni * idev;
        cudaSetDevice(idev);
        gravity_kernel<<<grids, threads>>>(dev->m, dev->x, (float)eps, dev->a, dev->p, n, ioff);
    }

#pragma omp parallel for private(i, k, ioff)
    for (idev = 0; idev < ndev; idev++) {
        Device_t *dev = device + idev;
        ioff = ni * idev;
        cudaSetDevice(idev);
        cutilSafeCall(cudaMemcpy(dev->buf, dev->a, sizeof(float) * 3 * ni, cudaMemcpyDeviceToHost));
        for (i = 0 ; i < ni; i++) {
            for (k = 0; k < 3; k++) {
                a[ioff + i][k] = (double)dev->buf[3 * i + k];
            }
        }

        cutilSafeCall(cudaMemcpy(dev->buf, dev->p, sizeof(float) * ni, cudaMemcpyDeviceToHost));
        for (i = 0 ; i < ni; i++) {
            p[ioff + i]= (double)dev->buf[i];
        }
    }

    cudaThreadSynchronize();
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
    fprintf(stderr, "calculation error on some GPUs at timestep: %d\n",
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
    double lt=0.0, st=0.0, sustained_intps, sustained_flops;
    int n, nstep, interval;
    int ndev_req = 1;
    static int step;

#ifdef __DSCUDA__
    dscudaSetErrorHandler(errhandler, (void *)&step);
#endif

    eps = 0.02;
    dt = 0.01;
    endt = 0.10;
    time = 0.0;
    nstep = endt/dt;

    if (argc < 3) {
#if TREEPERFTEST
        fprintf(stderr, "usage: %s <infile> <outfile> [# of devices] [listlen]\n",  argv[0]);
#else
        fprintf(stderr, "performs gravitational N-body simulation with naive direct summation algorithm.\n"
                "usage: %s <infile> <outfile> [# of devices]\n",  argv[0]);
#endif
        exit(1);
    }
    if (3 < argc) {
        ndev_req = atoi(argv[3]);
    }
#if TREEPERFTEST
    if (4 < argc) {
        Listlen = atoi(argv[4]);
        fprintf(stderr, "Listlen:%d\n", Listlen);
    }
#endif
  
    readnbody(&n, mj, xj, vj, argv[1]);
    interval = 500 * (10000.0/n) * (10000.0/n);    
    if (interval * 10 > nstep) {
	interval = nstep / 10;
    }
    interval = 1;
    fprintf(stderr, "interval: %d  nstep:%d\n", interval, nstep);

    get_cputime(&lt,&st);
#if 1
    calc_gravity_gpus(mj, xj, eps, a, p, n, ndev_req);
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
        calc_gravity_gpus(mj, xj, eps, a, p, n, ndev_req);
#else
        calc_gravity(mj, xj, eps, a, p, n);
#endif

        push_velocity(vj, a, 0.5*dt, n);

        if (step % interval == 0) {
            energy(mj, vj, p, n, &ke, &pe);
            e = ke+pe;
            sustained_intps = ((double)n)*((double)n)*interval/lt/1e9;
            sustained_flops = 38.0 * sustained_intps;
            printf("time: %g s\n", lt / interval);
            printf("speed: %g Gint/s\n", sustained_intps);
            printf("speed: %g Gflops\n", sustained_flops);
            printf("step: %d time: %e\n", step, time);
            printf("e: %e de: %e\n", e, e-e0);
            printf("ke: %e pe: %e\n", ke, pe);
            printf("ke/pe: %e\n\n", ke/pe);
            get_cputime(&lt,&st);
        }
    }
    //    writenbody(n, mj, xj, vj, argv[2]);
}
