#include <stdio.h>
#include <stdlib.h>
#include <cutil.h>
#include <cutil_inline.h>
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
