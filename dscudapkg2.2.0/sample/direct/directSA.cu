#include <stdio.h>
#include <stdlib.h>
#include "direct.h"

static float *DevM;
static float (*DevX)[3];
static float (*DevV)[3];
static float (*DevA)[3];
static float *DevP;

#define safeCall(err)             __safeCall   (err, __FILE__, __LINE__)

static inline void
__safeCall(cudaError err, const char *file, const int line)
{
    if (cudaSuccess != err) {
        fprintf(stderr, "%s(%i) : __unsafeCall() Runtime API error %d: %s.\n",
                file, line, (int)err, cudaGetErrorString(err));
        exit(1);
    }
}

__global__ void
position_kernel(float (*x)[3], float (*v)[3], float dt, int n)
{
    float xtmp[3], vtmp[3];
    int i, k;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    for (k = 0; k < 3; k++) {
        vtmp[k] = v[i][k];
    }
    for (k = 0; k < 3; k++) {
        xtmp[k] = dt * vtmp[k];
    }
    for (k = 0; k < 3; k++) {
        x[i][k] += xtmp[k];
    }
}

__global__ void
velocity_kernel(float (*v)[3], float (*a)[3], float dt, int n)
{
    float vtmp[3], atmp[3];
    int i, k;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    for (k = 0; k < 3; k++) {
        atmp[k] = a[i][k];
    }
    for (k = 0; k < 3; k++) {
        vtmp[k] = dt * atmp[k];
    }
    for (k = 0; k < 3; k++) {
        v[i][k] += vtmp[k];
    }
}

__global__ void
gravity_kernel(float *m, float (*x)[3], float eps, float (*a)[3], float *p, int n)
{
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
}

static void
alloc_bufs(int n)
{
    static int firstcall = 1;
    if (firstcall) {
        firstcall = 0;
        safeCall(cudaMalloc((void**)&DevM, sizeof(float) * n));
        safeCall(cudaMalloc((void**)&DevX, sizeof(float) * 3 * n));
        safeCall(cudaMalloc((void**)&DevV, sizeof(float) * 3 * n));
        safeCall(cudaMalloc((void**)&DevA, sizeof(float) * 3 * n));
        safeCall(cudaMalloc((void**)&DevP, sizeof(float) * n));
    }
}

static void
send_particle(double *m, double (*x)[3], double (*v)[3], int n)
{
    int i, k;
     static float floatbuf[NMAX*3];

    for (i = 0 ; i < n; i++) {
        floatbuf[i] = (float)m[i];
    }
    safeCall(cudaMemcpy(DevM, floatbuf, sizeof(float) * n, cudaMemcpyHostToDevice));

    for (i = 0 ; i < n; i++) {
        for (k = 0; k < 3; k++) {
            floatbuf[3 * i + k] = (float)x[i][k];
        }
    }
    safeCall(cudaMemcpy(DevX, floatbuf, sizeof(float) * 3 * n, cudaMemcpyHostToDevice));

    for (i = 0 ; i < n; i++) {
        for (k = 0; k < 3; k++) {
            floatbuf[3 * i + k] = (float)v[i][k];
        }
    }
    safeCall(cudaMemcpy(DevV, floatbuf, sizeof(float) * 3 * n, cudaMemcpyHostToDevice));
}

static void
recv_particle(double *m, double (*x)[3], double (*v)[3], double (*a)[3], double *p, int n)
{
    int i, k;
    static float floatbuf[NMAX*3];

    safeCall(cudaMemcpy(floatbuf, DevX, sizeof(float) * 3 * n, cudaMemcpyDeviceToHost));
    for (i = 0 ; i < n; i++) {
        for (k = 0; k < 3; k++) {
            x[i][k] = (double)floatbuf[3 * i + k];
        }
    }

    safeCall(cudaMemcpy(floatbuf, DevM, sizeof(float) * n, cudaMemcpyDeviceToHost));
    for (i = 0 ; i < n; i++) {
        m[i]= (double)floatbuf[i];
    }

    safeCall(cudaMemcpy(floatbuf, DevV, sizeof(float) * 3 * n, cudaMemcpyDeviceToHost));
    for (i = 0 ; i < n; i++) {
        for (k = 0; k < 3; k++) {
            v[i][k] = (double)floatbuf[3 * i + k];
        }
    }

    safeCall(cudaMemcpy(floatbuf, DevA, sizeof(float) * 3 * n, cudaMemcpyDeviceToHost));
    for (i = 0 ; i < n; i++) {
        for (k = 0; k < 3; k++) {
            a[i][k] = (double)floatbuf[3 * i + k];
        }
    }

    safeCall(cudaMemcpy(floatbuf, DevP, sizeof(float) * n, cudaMemcpyDeviceToHost));
    for (i = 0 ; i < n; i++) {
        p[i]= (double)floatbuf[i];
    }
}

static void
calc_gravity_gpu(double eps, int n)
{
    int i, k;
    int nth = 64;
    dim3 threads(nth, 1, 1);
    dim3 grids((n+nth-1)/nth, 1, 1);

    gravity_kernel<<<grids, threads>>>(DevM, DevX, (float)eps, DevA, DevP, n);
}

static void
push_position_gpu(double dt, int n)
{
    int i, k;
    int nth = 64;
    dim3 threads(nth, 1, 1);
    dim3 grids((n+nth-1)/nth, 1, 1);

    position_kernel<<<grids, threads>>>(DevX, DevV, (float)dt, n);
}

static void
push_velocity_gpu(double dt, int n)
{
    int i, k;
    int nth = 64;
    dim3 threads(nth, 1, 1);
    dim3 grids((n+nth-1)/nth, 1, 1);

    velocity_kernel<<<grids, threads>>>(DevV, DevA, (float)dt, n);
}


static void
errhandler(void *arg)
{
    fprintf(stderr, "calculation error on some GPU at timestep: %d\n",
            *(int *)arg);
    exit(1);
}

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
    int i;

#ifdef __DSCUDA_CLIENT__
    dscudaSetErrorHandler(errhandler, (void *)&step);
#endif

    eps = 0.02;
    dt = 0.01;
    endt = 1.1;
    //    endt = 10000.1;
    time = 0.0;
    nstep = endt/dt;

    if (argc < 3) {
        fprintf(stderr, "Performs gravitational N-body simulation with naive direct summation algorithm. "
                "In this version, the orbital integration is performed not on the host but on the GPU.\n"
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

    alloc_bufs(n);
    send_particle(mj, xj, vj, n);
    calc_gravity_gpu(eps, n);

    recv_particle(mj, xj, vj, a, p, n);
    energy(mj, vj, p, n, &ke, &pe);
    e0 = ke+pe;
    printf("ke: %f  pe: %f  e0: %f\n", ke, pe, e0);
    for (step = 1; step < nstep; step++) {
        push_velocity_gpu(0.5*dt, n);
        push_position_gpu(dt, n);
        time = time + dt;
        calc_gravity_gpu(eps, n);
        push_velocity_gpu(0.5*dt, n);

        if (step % interval == 0) {
            recv_particle(mj, xj, vj, a, p, n);
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
