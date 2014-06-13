#include <stdio.h>
#include <stdlib.h>
#include <cutil.h>
#include <cutil_inline.h>
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
