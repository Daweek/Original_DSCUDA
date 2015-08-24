#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

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

#if __DSCUDA__
#include <dscuda.h>
#endif

#include "direct.h"



typedef struct {
    float *m;
    float (*x)[3];
    float (*v)[3];
    float (*a)[3];
    float *p;
    float *mj;
    float (*xj)[3];
    float *hostbuf;
} Device_t;

static int Ndev;
static int Nthread = 64;
static Device_t Device[NDEVMAX];

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
gravity_kernel(float *mj, float (*xj)[3], float (*xi)[3], float eps,
               float (*a)[3], float *p, int n, int do_clear)
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
            dx[k] = xj[j][k] - xi[i][k];
        }
        r2 = eps * eps;
        for (k = 0; k < 3; k++) {
            r2 += dx[k] * dx[k];
        }            
        r = sqrtf(r2);
        mf = mj[j] / (r * r2);
        for (k = 0; k < 3; k++) {
            atmp[k] += mf * dx[k];
        }            
        ptmp -= mj[j] / r;
    }

    if (do_clear) {
        for (k = 0; k < 3; k++) {
            a[i][k] = atmp[k];
        }
        p[i] = ptmp;
    }
    else {
        for (k = 0; k < 3; k++) {
            a[i][k] += atmp[k];
        }
        p[i] += ptmp;
    }
}

static int
roundUp(int src, int by)
{
    return ((src - 1) / by + 1) * by;
}

static void
alloc_bufs(int n, int ndev_req)
{
    static int firstcall = 1;
    int ndev, idev, ni, niru;

    if (!firstcall) return;

    firstcall = 0;

    safeCall(cudaGetDeviceCount(&ndev));
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
    Ndev = ndev = ndev_req;
    fprintf(stderr, "use %d CUDA devices.\n", ndev);

    ni = (n - 1) / ndev + 1;
    niru = roundUp(ni, Nthread);

#pragma omp parallel for
    for (idev = 0; idev < ndev; idev++) {
        Device_t *dev = Device + idev;
        cudaSetDevice(idev);
        safeCall(cudaMalloc((void**)&dev->m, sizeof(float) * niru));
        safeCall(cudaMalloc((void**)&dev->x, sizeof(float) * 3 * niru));
        safeCall(cudaMalloc((void**)&dev->v, sizeof(float) * 3 * niru));
        safeCall(cudaMalloc((void**)&dev->a, sizeof(float) * 3 * niru));
        safeCall(cudaMalloc((void**)&dev->p, sizeof(float) * niru));
        safeCall(cudaMalloc((void**)&dev->mj, sizeof(float) * niru));
        safeCall(cudaMalloc((void**)&dev->xj, sizeof(float) * 3 * niru));
        dev->hostbuf = (float *)malloc(sizeof(float) * 3 * niru);
    }
}

static void
send_iparticle(double *m, double (*x)[3], double (*v)[3], int n)
{
    int i, k, ioff, idev;
    int ni = (n - 1) / Ndev + 1;

#pragma omp parallel for private(i, k, ioff)
    for (idev = 0; idev < Ndev; idev++) {
        Device_t *dev = Device + idev;
        ioff = ni * idev;
        cudaSetDevice(idev);

        for (i = 0 ; i < ni; i++) {
            dev->hostbuf[i] = (float)m[ioff + i];
        }
        safeCall(cudaMemcpy(dev->m, dev->hostbuf, sizeof(float) * ni, cudaMemcpyHostToDevice));

        for (i = 0 ; i < ni; i++) {
            for (k = 0; k < 3; k++) {
                dev->hostbuf[3 * i + k] = (float)x[ioff + i][k];
            }
        }
        safeCall(cudaMemcpy(dev->x, dev->hostbuf, sizeof(float) * 3 * ni, cudaMemcpyHostToDevice));

        for (i = 0 ; i < ni; i++) {
            for (k = 0; k < 3; k++) {
                dev->hostbuf[3 * i + k] = (float)v[ioff + i][k];
            }
        }
        safeCall(cudaMemcpy(dev->v, dev->hostbuf, sizeof(float) * 3 * ni, cudaMemcpyHostToDevice));
    }
}

static void
recv_iparticle(double *m, double (*x)[3], double (*v)[3], double (*a)[3], double *p, int n)
{
    int i, k, ioff, idev;
    int ni = (n - 1) / Ndev + 1;

#pragma omp parallel for private(i, k, ioff)
    for (idev = 0; idev < Ndev; idev++) {
        Device_t *dev = Device + idev;
        ioff = ni * idev;
        cudaSetDevice(idev);

        safeCall(cudaMemcpy(dev->hostbuf, dev->m, sizeof(float) * ni, cudaMemcpyDeviceToHost));
        for (i = 0 ; i < ni; i++) {
            m[ioff + i]= (double)dev->hostbuf[i];
        }

        safeCall(cudaMemcpy(dev->hostbuf, dev->x, sizeof(float) * 3 * ni, cudaMemcpyDeviceToHost));
        for (i = 0 ; i < ni; i++) {
            for (k = 0; k < 3; k++) {
                x[ioff + i][k] = (double)dev->hostbuf[3 * i + k];
            }
        }

        safeCall(cudaMemcpy(dev->hostbuf, dev->v, sizeof(float) * 3 * ni, cudaMemcpyDeviceToHost));
        for (i = 0 ; i < ni; i++) {
            for (k = 0; k < 3; k++) {
                v[ioff + i][k] = (double)dev->hostbuf[3 * i + k];
            }
        }

        safeCall(cudaMemcpy(dev->hostbuf, dev->a, sizeof(float) * 3 * ni, cudaMemcpyDeviceToHost));
        for (i = 0 ; i < ni; i++) {
            for (k = 0; k < 3; k++) {
                a[ioff + i][k] = (double)dev->hostbuf[3 * i + k];
            }
        }

        safeCall(cudaMemcpy(dev->hostbuf, dev->p, sizeof(float) * ni, cudaMemcpyDeviceToHost));
        for (i = 0 ; i < ni; i++) {
            p[ioff + i]= (double)dev->hostbuf[i];
        }
    }

    cudaThreadSynchronize();
}

static void
broadcast_jparticle(int sdevid, int ni)
{
    int idev;
    Device_t *sdev = Device + sdevid;

#pragma omp parallel for
    for (idev = 0; idev < Ndev; idev++) {
        Device_t *ddev = Device + idev;
        cudaSetDevice(idev);
        cudaMemcpy(ddev->mj, sdev->m, sizeof(float) * ni, cudaMemcpyDefault);
        cudaMemcpy(ddev->xj, sdev->x, sizeof(float) * 3 * ni, cudaMemcpyDefault);
    }
}

static void
broadcast_jparticle2(int sdevid, int ni)
{
    int idev;
    Device_t *sdev = Device + sdevid;
    void *dbufs[NDEVMAX], *sbufs[NDEVMAX];
    int counts[NDEVMAX];

    for (idev = 0; idev < Ndev; idev++) {
        Device_t *ddev = Device + idev;
        dbufs[idev] = ddev->mj;
        sbufs[idev] = sdev->m;
        counts[idev] = sizeof(float) * ni;
    }
    dscudaMemcopies(dbufs, sbufs, counts, Ndev);

    for (idev = 0; idev < Ndev; idev++) {
        Device_t *ddev = Device + idev;
        dbufs[idev] = ddev->xj;
        sbufs[idev] = sdev->x;
        counts[idev] = sizeof(float) * 3 * ni;
    }
    dscudaMemcopies(dbufs, sbufs, counts, Ndev);
}

static void
broadcast_jparticle3(int sdevid, int ni)
{
    int idev;
    Device_t *sdev = Device + sdevid;
    void *dbufs[NDEVMAX];

    for (idev = 0; idev < Ndev; idev++) {
        Device_t *ddev = Device + idev;
        dbufs[idev] = ddev->mj;
    }
    dscudaBroadcast(dbufs, sdev->m, sizeof(float) * ni, Ndev);

    for (idev = 0; idev < Ndev; idev++) {
        Device_t *ddev = Device + idev;
        dbufs[idev] = ddev->xj;
    }
    dscudaBroadcast(dbufs, sdev->x, sizeof(float) * 3 * ni, Ndev);
}

static void
calc_gravity_gpu(double eps, int n)
{
    int idev, jdev;
    int ni = (n - 1) / Ndev + 1, nj;
    int nth = Nthread;
    dim3 threads(nth, 1, 1);
    dim3 grids((ni + nth - 1) / nth, 1, 1);
    int do_clear = 1;

    for (jdev = 0; jdev < Ndev; jdev++) {
        broadcast_jparticle3(jdev, ni);

        nj = n / Ndev;
        if (jdev < n % Ndev) {
            nj = nj + 1;
        }
#pragma omp parallel for
        for (idev = 0; idev < Ndev; idev++) {
            Device_t *dev = Device + idev;
            cudaSetDevice(idev);
            gravity_kernel<<<grids, threads>>>(dev->mj, dev->xj, dev->x, (float)eps,
                                               dev->a, dev->p, nj, do_clear);
        }
        do_clear = 0;
    }
}

static void
push_position_gpu(double dt, int n)
{
    int idev;
    int ni = (n - 1) / Ndev + 1;
    int nth = Nthread;
    dim3 threads(nth, 1, 1);
    dim3 grids((ni + nth - 1) / nth, 1, 1);

#pragma omp parallel for
    for (idev = 0; idev < Ndev; idev++) {
        Device_t *dev = Device + idev;
        cudaSetDevice(idev);
        position_kernel<<<grids, threads>>>(dev->x, dev->v, (float)dt, ni);
    }
}

static void
push_velocity_gpu(double dt, int n)
{
    int idev;
    int ni = (n - 1) / Ndev + 1;
    int nth = Nthread;
    dim3 threads(nth, 1, 1);
    dim3 grids((ni +  nth - 1) / nth, 1, 1);

#pragma omp parallel for
    for (idev = 0; idev < Ndev; idev++) {
        Device_t *dev = Device + idev;
        cudaSetDevice(idev);
        velocity_kernel<<<grids, threads>>>(dev->v, dev->a, (float)dt, ni);
    }
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
    static int step;
    double time, dt, endt;;
    double eps;
    double e, e0, ke, pe;
    double lt=0.0, st=0.0, sustained_intps, sustained_flops;
    int n, nstep, interval;
    int ndev_req = 1;

#ifdef __DSCUDA_CLIENT__
    dscudaSetErrorHandler(errhandler, (void *)&step);
#endif

    eps = 0.02;
    dt = 0.01;
    endt = 1.1;
    time = 0.0;
    nstep = endt/dt;

    if (argc < 3) {
        fprintf(stderr, "Performs gravitational N-body simulation with naive direct summation algorithm. "
                "In this version, the orbital integration is performed not on the host but on the GPU.\n"
                "usage: %s <infile> <outfile> [# of devices]\n",  argv[0]);
        exit(1);
    }
    if (3 < argc) {
        ndev_req = atoi(argv[3]);
    }
  
    readnbody(&n, mj, xj, vj, argv[1]);
    interval = 500 * (10000.0/n) * (10000.0/n);    
    if (interval * 10 > nstep) {
	interval = nstep / 10;
    }
    interval = 1;
    fprintf(stderr, "interval: %d\n", interval);

    get_cputime(&lt,&st);

    alloc_bufs(n, ndev_req);
    send_iparticle(mj, xj, vj, n);
    calc_gravity_gpu(eps, n);

    recv_iparticle(mj, xj, vj, a, p, n);
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
            recv_iparticle(mj, xj, vj, a, p, n);
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
    writenbody(n, mj, xj, vj, argv[2]);
    exit(0);
}
