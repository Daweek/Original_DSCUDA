#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cutil.h>

#define D2F_AND_COPY(n,host_mem,device_mem,float_mem) \
  for(int i=0;i<(n);i++) ((float *)(float_mem))[i]=(host_mem)[i];\
  CUDA_SAFE_CALL(cudaMalloc((void **)&(device_mem),sizeof(float)*(n)));\
  CUDA_SAFE_CALL(cudaMemcpy((device_mem),(float_mem),sizeof(float)*(n),cudaMemcpyHostToDevice));

#define NMAX      8192
#define NTHRE      256
#define ATYPE       32
#define ATYPE2    (ATYPE * ATYPE)
#define NDIVBIT      4         // 0 means no reduction, 2 means 4 reduction
#define NDIV      (1<<NDIVBIT) // number of j-parallelism
#define NTHRE2    (NTHRE/NDIV) // number of i-particles per block

typedef struct {
  float r[3];
  int atype;
} VG_XVEC;

typedef struct {
  float pol;
  float sigm;
  float ipotro;
  float pc;
  float pd;
  float zz;
} VG_MATRIX;

__constant__ VG_MATRIX c_matrix[ATYPE2];

static int Dev=-1;

extern "C"
void MR3init(void)
{
  if(Dev<0){
    char *s;
    s=getenv("VG_DEVICEID");
    if(s!=NULL){
      sscanf(s,"%d",&Dev);
      printf("VG_DEVICEID is set %d\n",Dev);
    }
    else{
      Dev=0;
    }
    cudaSetDevice(Dev);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, Dev);
    printf("Device %d: %s\n", Dev, deviceProp.name);
  }
}

extern "C"
void MR3free(void)
{
}


extern "C" __global__ 
void nacl_kernel(float *x, int n, int *atype, int nat, float *pol, float *sigm, float *ipotro,
		 float *pc, float *pd, float *zz, int tblno, float xmax, int periodicflag, 
		 float *force)
{
  int i,j,k,t;
  float xmax1,dn2,r,inr,inr2,inr4,inr8,d3,dr[3],fi[3];
  float pb=(float)(0.338e-19/(14.39*1.60219e-19)),dphir; 

  if((periodicflag & 1)==0) xmax *= 2.0f;
  xmax1 = 1.0f / xmax;
  i = blockIdx.x * 64 + threadIdx.x;
  if(i<n){
    for(k=0; k<3; k++) fi[k] = 0.0f;
    for(j=0; j<n; j++){
      dn2 = 0.0f;
      for(k=0; k<3; k++){
	dr[k] =  x[i*3+k] - x[j*3+k];
	dr[k] -= rintf(dr[k] * xmax1) * xmax;
	dn2   += dr[k] * dr[k];
      }
      if(dn2 != 0.0f){
	r     = sqrtf(dn2);
	inr   = 1.0f  / r;
	inr2  = inr  * inr;
	inr4  = inr2 * inr2;
	inr8  = inr4 * inr4;
	t     = atype[i] * nat + atype[j];
	d3    = pb * pol[t] * exp( (sigm[t] - r) * ipotro[t]);
	dphir = ( d3 * ipotro[t] * inr
		  - 6.0f * pc[t] * inr8
		  - 8.0f * pd[t] * inr8 * inr2
		  + inr2 * inr * zz[t] );
	for(k=0; k<3; k++) fi[k] += dphir * dr[k];
      }
    }
    for(k=0; k<3; k++) force[i*3+k] = fi[k];
  }
}

extern "C" __global__ 
void nacl_kernel_kadai8(float *x, int n, int *atype, int nat, float *pol, float *sigm, float *ipotro,
		 float *pc, float *pd, float *zz, int tblno, float xmax, int periodicflag, 
		 float *force)
{
  int i,j,k,t,js;
  float xmax1,dn2,r,inr,inr2,inr4,inr8,d3,dr[3],fi[3];
  float pb=(float)(0.338e-19/(14.39*1.60219e-19)),dphir; 

  if((periodicflag & 1)==0) xmax *= 2.0f;
  xmax1 = 1.0f / xmax;
  i = blockIdx.x * 64 + threadIdx.x;
  if(i<n){
    __shared__ float s_x[64*3];
    for(k=0; k<3; k++) fi[k] = 0.0f;
    for(j=0; j<n; j+=64){
     __syncthreads();
     for(k=0;k<3;k++) s_x[threadIdx.x*3+k]=x[(j+threadIdx.x)*3+k];
     __syncthreads();
     for(js=0;js<64;js++){
      dn2 = 0.0f;
      for(k=0; k<3; k++){
	dr[k] =  x[i*3+k] - s_x[js*3+k];
	dr[k] -= rintf(dr[k] * xmax1) * xmax;
	dn2   += dr[k] * dr[k];
      }
      if(dn2 != 0.0f){
	r     = sqrtf(dn2);
	inr   = 1.0f  / r;
	inr2  = inr  * inr;
	inr4  = inr2 * inr2;
	inr8  = inr4 * inr4;
	t     = atype[i] * nat + atype[j+js];
	d3    = pb * pol[t] * exp( (sigm[t] - r) * ipotro[t]);
	dphir = ( d3 * ipotro[t] * inr
		  - 6.0f * pc[t] * inr8
		  - 8.0f * pd[t] * inr8 * inr2
		  + inr2 * inr * zz[t] );
	for(k=0; k<3; k++) fi[k] += dphir * dr[k];
      }
     }
    }
    for(k=0; k<3; k++) force[i*3+k] = fi[k];
  }
}

extern "C"
void MR3calcnacl_kadai8(double x[], int n, int atype[], int nat,
		 double pol[], double sigm[], double ipotro[],
		 double pc[], double pd[], double zz[],
		 int tblno, double xmax, int periodicflag,
		 double force[])
{
  int i,*d_atype;
  float *d_x,*d_pol,*d_sigm,*d_ipotro,*d_pc,*d_pd,*d_zz,*d_force,xmaxf=xmax;

  // ensure force has enough size for temporary array
  if(sizeof(double)*n*3<sizeof(float)*nat*nat){
    fprintf(stderr,"** error : n*3<nat*nat **\n");
    exit(1);
  }

  // allocate global memory and copy from host to GPU
  D2F_AND_COPY(n*3,x,d_x,force);
  D2F_AND_COPY(nat*nat,pol,d_pol,force);
  D2F_AND_COPY(nat*nat,sigm,d_sigm,force);
  D2F_AND_COPY(nat*nat,ipotro,d_ipotro,force);
  D2F_AND_COPY(nat*nat,pc,d_pc,force);
  D2F_AND_COPY(nat*nat,pd,d_pd,force);
  D2F_AND_COPY(nat*nat,zz,d_zz,force);
  CUDA_SAFE_CALL(cudaMalloc((void**)&d_atype,sizeof(int)*n));
  CUDA_SAFE_CALL(cudaMemcpy(d_atype,atype,sizeof(int)*n,cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMalloc((void**)&d_force,sizeof(float)*n*3));

  // call GPU kernel
  dim3 threads(64);
  dim3 grid((n+63)/64);
  nacl_kernel_kadai8<<< grid, threads >>>(d_x,n,d_atype,nat,d_pol,d_sigm,d_ipotro,
				   d_pc,d_pd,d_zz,tblno,xmaxf,periodicflag,d_force);
  CUT_CHECK_ERROR("Kernel execution failed");

  // copy GPU result to host, and convert it to double
  CUDA_SAFE_CALL(cudaMemcpy(force,d_force,sizeof(float)*n*3,cudaMemcpyDeviceToHost));
  for(i=n*3-1;i>=0;i--) force[i]=((float *)force)[i];

  // free allocated global memory
  CUDA_SAFE_CALL(cudaFree(d_x));
  CUDA_SAFE_CALL(cudaFree(d_atype));
  CUDA_SAFE_CALL(cudaFree(d_pol));
  CUDA_SAFE_CALL(cudaFree(d_sigm));
  CUDA_SAFE_CALL(cudaFree(d_ipotro));
  CUDA_SAFE_CALL(cudaFree(d_pc));
  CUDA_SAFE_CALL(cudaFree(d_pd));
  CUDA_SAFE_CALL(cudaFree(d_zz));
  CUDA_SAFE_CALL(cudaFree(d_force));
}


__device__ __inline__ 
void inter_kadai91011(float xj[3], float xi[3], float fi[3], 
	   VG_MATRIX *d_matrix, int t, float xmax, float xmax1)
{
  int k;
  float dn2,r,inr,inr2,inr4,inr8,d3,dr[3];
  float pb=(float)(0.338e-19/(14.39*1.60219e-19)),dphir;

  dn2 = 0.0f;
  for(k=0; k<3; k++){
    dr[k]  = xi[k] - xj[k];
    dr[k] -= rintf(dr[k] * xmax1) * xmax;
    dn2   += dr[k] * dr[k];
  }
  if(dn2 != 0.0f){
    r     = sqrtf(dn2);
    inr   = 1.0f / r;
    inr2  = inr  * inr;
    inr4  = inr2 * inr2;
    inr8  = inr4 * inr4;
    d3    = pb * c_matrix[t].pol * expf( (c_matrix[t].sigm - r) * c_matrix[t].ipotro);
    dphir = ( d3 * c_matrix[t].ipotro * inr
	    - 6.0f * c_matrix[t].pc * inr8
	    - 8.0f * c_matrix[t].pd * inr8 * inr2
	    + inr2 * inr * c_matrix[t].zz );
    for(k=0; k<3; k++) fi[k] += dphir * dr[k];
  }
}

extern "C" __global__ 
void nacl_kernel_gpu_kadai9(VG_XVEC *x, int n, int nat, VG_MATRIX *d_matrix, float xmax, float *fvec)
{
  int tid = threadIdx.x;
  int i = blockIdx.x * NTHRE + tid;
  int j,k;
  float fi[3],xmax1=1.0f/xmax;
  int atypei;
  float xi[3];
  __shared__ VG_XVEC s_xj[NTHRE];

  for(k=0; k<3; k++) fi[k] = 0.0f;
  for(k=0; k<3; k++) xi[k] = x[i].r[k];
  atypei = x[i].atype * nat;
  for (j = 0; j < n; j++){
    inter_kadai91011(x[j].r, xi, fi, d_matrix, atypei + x[j].atype, xmax, xmax1);
  }
  if(i<n) for(k=0; k<3; k++) fvec[i*3+k] = fi[k];
}


extern "C"
void MR3calcnacl_kadai9(double x[], int n, int atype[], int nat,
		 double pol[], double sigm[], double ipotro[],
		 double pc[], double pd[], double zz[],
		 int tblno, double xmax, int periodicflag,
		 double force[])
{
  int i,j;
  static VG_XVEC *d_x=NULL;
  static float *d_force=NULL;
  static VG_MATRIX *d_matrix=NULL;
  float xmaxf;
  VG_MATRIX *matrix=(VG_MATRIX *)force;
  VG_XVEC   *vec=(VG_XVEC *)force;
  if((periodicflag & 1)==0) xmax*=2.0;
  xmaxf=xmax;
  static float *forcef=NULL;
  static int n_bak=0;

  // ensure force has enough size for temporary array
  if(sizeof(double)*n*3<sizeof(VG_MATRIX)*nat*nat){
    fprintf(stderr,"** error : n*3<nat*nat **\n");
    exit(1);
  }
  if(nat>ATYPE){
    fprintf(stderr,"** error : nat is too large **\n");
    exit(1);
  }

  if(n!=n_bak){
    // free and allocate global memory
    int nalloc;
    static int nalloc_bak=0;
    if(n>NMAX) nalloc=n;
    else       nalloc=NMAX;
    if(nalloc!=nalloc_bak){
      CUDA_SAFE_CALL(cudaFree(d_x));
      CUDA_SAFE_CALL(cudaFree(d_force));
      CUDA_SAFE_CALL(cudaFree(d_matrix));
      CUDA_SAFE_CALL(cudaMalloc((void**)&d_x,sizeof(VG_XVEC)*(nalloc+NTHRE)));
      CUDA_SAFE_CALL(cudaMalloc((void**)&d_force,sizeof(float)*nalloc*3));
      CUDA_SAFE_CALL(cudaMalloc((void**)&d_matrix,sizeof(VG_MATRIX)*nat*nat));
      
      free(forcef);
      if((forcef=(float *)malloc(sizeof(float)*nalloc*3))==NULL){
	fprintf(stderr,"** error : can't malloc forcef **\n");
	exit(1);
      }
      bzero(forcef,sizeof(float)*nalloc*3);
      nalloc_bak=nalloc;
    }

    // send matrix
    for(i=0;i<nat;i++){
      for(j=0;j<nat;j++){
	matrix[i*nat+j].pol=(float)(pol[i*nat+j]);
	matrix[i*nat+j].sigm=(float)(sigm[i*nat+j]);
	matrix[i*nat+j].ipotro=(float)(ipotro[i*nat+j]);
	matrix[i*nat+j].pc=(float)(pc[i*nat+j]);
	matrix[i*nat+j].pd=(float)(pd[i*nat+j]);
	matrix[i*nat+j].zz=(float)(zz[i*nat+j]);
      }
    }
//    CUDA_SAFE_CALL(cudaMemcpy(d_matrix,matrix,sizeof(VG_MATRIX)*nat*nat,cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_matrix,matrix,sizeof(VG_MATRIX)*nat*nat));

    n_bak=n;
  }

  for(i=0;i<n;i++){
    for(j=0;j<3;j++){
      vec[i].r[j]=x[i*3+j];
    }
    vec[i].atype=atype[i];
  }
  CUDA_SAFE_CALL(cudaMemcpy(d_x,vec,sizeof(VG_XVEC)*n,cudaMemcpyHostToDevice));

  // call GPU kernel
  dim3 threads(NTHRE);
  dim3 grid((n+NTHRE-1)/NTHRE);
  nacl_kernel_gpu_kadai9<<< grid, threads >>>(d_x,n,nat,d_matrix,xmaxf,d_force);
  CUT_CHECK_ERROR("Kernel execution failed");

  // copy GPU result to host, and convert it to double
  CUDA_SAFE_CALL(cudaMemcpy(forcef,d_force,sizeof(float)*n*3,cudaMemcpyDeviceToHost));
  for(i=0;i<n;i++) for(j=0;j<3;j++) force[i*3+j]=forcef[i*3+j];
}

extern "C" __global__ 
void nacl_kernel_gpu_kadai10(VG_XVEC *x, int n, int nat, VG_MATRIX *d_matrix, float xmax, float *fvec)
{
#if 1
  int tid = threadIdx.x;
  int i = blockIdx.x * NTHRE + tid;
  int j,k;
  float fi[3],xmax1=1.0f/xmax;
  int atypei;
  float xi[3];
  __shared__ VG_XVEC s_xj[NTHRE];

  for(k=0; k<3; k++) fi[k] = 0.0f;
  for(k=0; k<3; k++) xi[k] = x[i].r[k];
  atypei = x[i].atype * nat;
  __syncthreads();
  s_xj[tid]=x[tid];
  __syncthreads();
  for(j=0;j<n % NTHRE;j++){
    inter_kadai91011(s_xj[j].r, xi, fi, d_matrix, atypei + s_xj[j].atype, xmax, xmax1);
  }
  for (; j < n; j+=NTHRE){
   __syncthreads();
   s_xj[tid]=x[j+tid];
   __syncthreads();
   for(int js=0;js<NTHRE;js++){
    inter_kadai91011(s_xj[js].r, xi, fi, d_matrix, atypei + s_xj[js].atype, xmax, xmax1);
   }
  }
  if(i<n) for(k=0; k<3; k++) fvec[i*3+k] = fi[k];
#else
  int tid = threadIdx.x;
  int i = blockIdx.x * NTHRE + tid;
  int j,k;
  float fi[3],xmax1=1.0f/xmax;
  int atypei;
  float xi[3];
  __shared__ VG_XVEC s_xj[NTHRE];

  for(k=0; k<3; k++) fi[k] = 0.0f;
  for(k=0; k<3; k++) xi[k] = x[i].r[k];
  atypei = x[i].atype * nat;
  for (j = 0; j < n-NTHRE+1; j+=NTHRE){
   __syncthreads();
   s_xj[tid]=x[j+tid];
   __syncthreads();
   for(int js=0;js<NTHRE;js++){
    inter_kadai91011(s_xj[js].r, xi, fi, d_matrix, atypei + s_xj[js].atype, xmax, xmax1);
   }
  }
  __syncthreads();
  if(tid<n-j) s_xj[tid]=x[j+tid];
  __syncthreads();
  for(int js=0;js<n-j;js++){
    inter_kadai91011(s_xj[js].r, xi, fi, d_matrix, atypei + s_xj[js].atype, xmax, xmax1);
  }
  if(i<n) for(k=0; k<3; k++) fvec[i*3+k] = fi[k];
#endif
}

extern "C"
void MR3calcnacl_kadai10(double x[], int n, int atype[], int nat,
		 double pol[], double sigm[], double ipotro[],
		 double pc[], double pd[], double zz[],
		 int tblno, double xmax, int periodicflag,
		 double force[])
{
  int i,j;
  static VG_XVEC *d_x=NULL;
  static float *d_force=NULL;
  static VG_MATRIX *d_matrix=NULL;
  float xmaxf;
  VG_MATRIX *matrix=(VG_MATRIX *)force;
  VG_XVEC   *vec=(VG_XVEC *)force;
  if((periodicflag & 1)==0) xmax*=2.0;
  xmaxf=xmax;
  static float *forcef=NULL;
  static int n_bak=0;

  // ensure force has enough size for temporary array
  if(sizeof(double)*n*3<sizeof(VG_MATRIX)*nat*nat){
    fprintf(stderr,"** error : n*3<nat*nat **\n");
    exit(1);
  }
  if(nat>ATYPE){
    fprintf(stderr,"** error : nat is too large **\n");
    exit(1);
  }

  if(n!=n_bak){
    // free and allocate global memory
    int nalloc;
    static int nalloc_bak=0;
    if(n>NMAX) nalloc=n;
    else       nalloc=NMAX;
    if(nalloc!=nalloc_bak){
      CUDA_SAFE_CALL(cudaFree(d_x));
      CUDA_SAFE_CALL(cudaFree(d_force));
      CUDA_SAFE_CALL(cudaFree(d_matrix));
      CUDA_SAFE_CALL(cudaMalloc((void**)&d_x,sizeof(VG_XVEC)*(nalloc+NTHRE)));
      CUDA_SAFE_CALL(cudaMalloc((void**)&d_force,sizeof(float)*nalloc*3));
      CUDA_SAFE_CALL(cudaMalloc((void**)&d_matrix,sizeof(VG_MATRIX)*nat*nat));
      
      free(forcef);
      if((forcef=(float *)malloc(sizeof(float)*nalloc*3))==NULL){
	fprintf(stderr,"** error : can't malloc forcef **\n");
	exit(1);
      }
      bzero(forcef,sizeof(float)*nalloc*3);
      nalloc_bak=nalloc;
    }

    // send matrix
    for(i=0;i<nat;i++){
      for(j=0;j<nat;j++){
	matrix[i*nat+j].pol=(float)(pol[i*nat+j]);
	matrix[i*nat+j].sigm=(float)(sigm[i*nat+j]);
	matrix[i*nat+j].ipotro=(float)(ipotro[i*nat+j]);
	matrix[i*nat+j].pc=(float)(pc[i*nat+j]);
	matrix[i*nat+j].pd=(float)(pd[i*nat+j]);
	matrix[i*nat+j].zz=(float)(zz[i*nat+j]);
      }
    }
//    CUDA_SAFE_CALL(cudaMemcpy(d_matrix,matrix,sizeof(VG_MATRIX)*nat*nat,cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_matrix,matrix,sizeof(VG_MATRIX)*nat*nat));

    n_bak=n;
  }

  for(i=0;i<n;i++){
    for(j=0;j<3;j++){
      vec[i].r[j]=x[i*3+j];
    }
    vec[i].atype=atype[i];
  }
  CUDA_SAFE_CALL(cudaMemcpy(d_x,vec,sizeof(VG_XVEC)*n,cudaMemcpyHostToDevice));

  // call GPU kernel
  dim3 threads(NTHRE);
  dim3 grid((n+NTHRE-1)/NTHRE);
  nacl_kernel_gpu_kadai10<<< grid, threads >>>(d_x,n,nat,d_matrix,xmaxf,d_force);
  CUT_CHECK_ERROR("Kernel execution failed");

  // copy GPU result to host, and convert it to double
  CUDA_SAFE_CALL(cudaMemcpy(forcef,d_force,sizeof(float)*n*3,cudaMemcpyDeviceToHost));
  for(i=0;i<n;i++) for(j=0;j<3;j++) force[i*3+j]=forcef[i*3+j];
}

extern "C" __global__ 
void nacl_kernel_gpu_kadai11(VG_XVEC *x, int n, int nat, VG_MATRIX *d_matrix, float xmax, float *fvec)
{
  int tid = threadIdx.x;
  int i = blockIdx.x * NTHRE + tid;
  int j,k;
  float fi[3],xmax1=1.0f/xmax;
  int atypei;
  float xi[3];
  __shared__ VG_XVEC s_xj[NTHRE];

  for(k=0; k<3; k++) fi[k] = 0.0f;
  for(k=0; k<3; k++) xi[k] = x[i].r[k];
  atypei = x[i].atype * nat;
  for (j = 0; j < n-NTHRE+1; j+=NTHRE){
   __syncthreads();
   s_xj[tid]=x[j+tid];
   __syncthreads();
   for(int js=0;js<NTHRE;js+=8){
    inter_kadai91011(s_xj[js  ].r, xi, fi, d_matrix, atypei + s_xj[js  ].atype, xmax, xmax1);
    inter_kadai91011(s_xj[js+1].r, xi, fi, d_matrix, atypei + s_xj[js+1].atype, xmax, xmax1);
    inter_kadai91011(s_xj[js+2].r, xi, fi, d_matrix, atypei + s_xj[js+2].atype, xmax, xmax1);
    inter_kadai91011(s_xj[js+3].r, xi, fi, d_matrix, atypei + s_xj[js+3].atype, xmax, xmax1);
    inter_kadai91011(s_xj[js+4].r, xi, fi, d_matrix, atypei + s_xj[js+4].atype, xmax, xmax1);
    inter_kadai91011(s_xj[js+5].r, xi, fi, d_matrix, atypei + s_xj[js+5].atype, xmax, xmax1);
    inter_kadai91011(s_xj[js+6].r, xi, fi, d_matrix, atypei + s_xj[js+6].atype, xmax, xmax1);
    inter_kadai91011(s_xj[js+7].r, xi, fi, d_matrix, atypei + s_xj[js+7].atype, xmax, xmax1);
   }
  }
  __syncthreads();
  if(tid<n-j) s_xj[tid]=x[j+tid];
  __syncthreads();
  for(int js=0;js<n-j;js++){
    inter_kadai91011(s_xj[js].r, xi, fi, d_matrix, atypei + s_xj[js].atype, xmax, xmax1);
  }
  if(i<n) for(k=0; k<3; k++) fvec[i*3+k] = fi[k];
}


extern "C"
void MR3calcnacl_kadai11(double x[], int n, int atype[], int nat,
		 double pol[], double sigm[], double ipotro[],
		 double pc[], double pd[], double zz[],
		 int tblno, double xmax, int periodicflag,
		 double force[])
{
  int i,j;
  static VG_XVEC *d_x=NULL;
  static float *d_force=NULL;
  static VG_MATRIX *d_matrix=NULL;
  float xmaxf;
  VG_MATRIX *matrix=(VG_MATRIX *)force;
  VG_XVEC   *vec=(VG_XVEC *)force;
  if((periodicflag & 1)==0) xmax*=2.0;
  xmaxf=xmax;
  static float *forcef=NULL;
  static int n_bak=0;

  // ensure force has enough size for temporary array
  if(sizeof(double)*n*3<sizeof(VG_MATRIX)*nat*nat){
    fprintf(stderr,"** error : n*3<nat*nat **\n");
    exit(1);
  }
  if(nat>ATYPE){
    fprintf(stderr,"** error : nat is too large **\n");
    exit(1);
  }

  if(n!=n_bak){
    // free and allocate global memory
    int nalloc;
    static int nalloc_bak=0;
    if(n>NMAX) nalloc=n;
    else       nalloc=NMAX;
    if(nalloc!=nalloc_bak){
      CUDA_SAFE_CALL(cudaFree(d_x));
      CUDA_SAFE_CALL(cudaFree(d_force));
      CUDA_SAFE_CALL(cudaFree(d_matrix));
      CUDA_SAFE_CALL(cudaMalloc((void**)&d_x,sizeof(VG_XVEC)*(nalloc+NTHRE)));
      CUDA_SAFE_CALL(cudaMalloc((void**)&d_force,sizeof(float)*nalloc*3));
      CUDA_SAFE_CALL(cudaMalloc((void**)&d_matrix,sizeof(VG_MATRIX)*nat*nat));
      
      free(forcef);
      if((forcef=(float *)malloc(sizeof(float)*nalloc*3))==NULL){
	fprintf(stderr,"** error : can't malloc forcef **\n");
	exit(1);
      }
      bzero(forcef,sizeof(float)*nalloc*3);
      nalloc_bak=nalloc;
    }

    // send matrix
    for(i=0;i<nat;i++){
      for(j=0;j<nat;j++){
	matrix[i*nat+j].pol=(float)(pol[i*nat+j]);
	matrix[i*nat+j].sigm=(float)(sigm[i*nat+j]);
	matrix[i*nat+j].ipotro=(float)(ipotro[i*nat+j]);
	matrix[i*nat+j].pc=(float)(pc[i*nat+j]);
	matrix[i*nat+j].pd=(float)(pd[i*nat+j]);
	matrix[i*nat+j].zz=(float)(zz[i*nat+j]);
      }
    }
//    CUDA_SAFE_CALL(cudaMemcpy(d_matrix,matrix,sizeof(VG_MATRIX)*nat*nat,cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_matrix,matrix,sizeof(VG_MATRIX)*nat*nat));

    n_bak=n;
  }

  for(i=0;i<n;i++){
    for(j=0;j<3;j++){
      vec[i].r[j]=x[i*3+j];
    }
    vec[i].atype=atype[i];
  }
  CUDA_SAFE_CALL(cudaMemcpy(d_x,vec,sizeof(VG_XVEC)*n,cudaMemcpyHostToDevice));

  // call GPU kernel
  dim3 threads(NTHRE);
  dim3 grid((n+NTHRE-1)/NTHRE);
  nacl_kernel_gpu_kadai11<<< grid, threads >>>(d_x,n,nat,d_matrix,xmaxf,d_force);
  CUT_CHECK_ERROR("Kernel execution failed");

  // copy GPU result to host, and convert it to double
  CUDA_SAFE_CALL(cudaMemcpy(forcef,d_force,sizeof(float)*n*3,cudaMemcpyDeviceToHost));
  for(i=0;i<n;i++) for(j=0;j<3;j++) force[i*3+j]=forcef[i*3+j];
}

__device__ __inline__ 
void inter_kadai12(float xj[3], float xi[3], float fi[3], 
	   VG_MATRIX *d_matrix, int t, float xmax, float xmax1)
{
  int k;
  float dn2,r,inr,inr2,inr4,inr8,d3,dr[3];
  float pb=(float)(0.338e-19/(14.39*1.60219e-19)),dphir;

  dn2 = 0.0f;
  for(k=0; k<3; k++){
    dr[k]  = xi[k] - xj[k];
    dr[k] -= rintf(dr[k] * xmax1) * xmax;
    dn2   += dr[k] * dr[k];
  }
  r     = sqrtf(dn2);
#if 0
  if(dn2!=0.0f) inr = 1.0f / r;
  else          inr = 0.0f;
#else
  inr   = 1.0f / r;
#endif
  inr2  = inr  * inr;
  inr4  = inr2 * inr2;
  inr8  = inr4 * inr4;
  d3    = pb * c_matrix[t].pol * expf( (c_matrix[t].sigm - r) * c_matrix[t].ipotro);
  dphir = ( d3 * c_matrix[t].ipotro * inr
	    - 6.0f * c_matrix[t].pc * inr8
	    - 8.0f * c_matrix[t].pd * inr8 * inr2
	    + inr2 * inr * c_matrix[t].zz );
#if 1
  if(dn2==0.0f) dphir = 0.0f;
#endif
  for(k=0; k<3; k++) fi[k] += dphir * dr[k];
}

extern "C" __global__ 
void nacl_kernel_gpu_kadai12(VG_XVEC *x, int n, int nat, VG_MATRIX *d_matrix, float xmax, float *fvec)
{
  int tid = threadIdx.x;
  int i = blockIdx.x * NTHRE + tid;
  int j,k;
  float fi[3],xmax1=1.0f/xmax;
  int atypei;
  float xi[3];
  __shared__ VG_XVEC s_xj[NTHRE];

  for(k=0; k<3; k++) fi[k] = 0.0f;
  for(k=0; k<3; k++) xi[k] = x[i].r[k];
  atypei = x[i].atype * nat;
  for (j = 0; j < n-NTHRE+1; j+=NTHRE){
   __syncthreads();
   s_xj[tid]=x[j+tid];
   __syncthreads();
   for(int js=0;js<NTHRE;js+=8){
    inter_kadai12(s_xj[js  ].r, xi, fi, d_matrix, atypei + s_xj[js  ].atype, xmax, xmax1);
    inter_kadai12(s_xj[js+1].r, xi, fi, d_matrix, atypei + s_xj[js+1].atype, xmax, xmax1);
    inter_kadai12(s_xj[js+2].r, xi, fi, d_matrix, atypei + s_xj[js+2].atype, xmax, xmax1);
    inter_kadai12(s_xj[js+3].r, xi, fi, d_matrix, atypei + s_xj[js+3].atype, xmax, xmax1);
    inter_kadai12(s_xj[js+4].r, xi, fi, d_matrix, atypei + s_xj[js+4].atype, xmax, xmax1);
    inter_kadai12(s_xj[js+5].r, xi, fi, d_matrix, atypei + s_xj[js+5].atype, xmax, xmax1);
    inter_kadai12(s_xj[js+6].r, xi, fi, d_matrix, atypei + s_xj[js+6].atype, xmax, xmax1);
    inter_kadai12(s_xj[js+7].r, xi, fi, d_matrix, atypei + s_xj[js+7].atype, xmax, xmax1);
   }
  }
  __syncthreads();
  if(tid<n-j) s_xj[tid]=x[j+tid];
  __syncthreads();
  for(int js=0;js<n-j;js++){
    inter_kadai12(s_xj[js].r, xi, fi, d_matrix, atypei + s_xj[js].atype, xmax, xmax1);
  }
  if(i<n) for(k=0; k<3; k++) fvec[i*3+k] = fi[k];
}


extern "C"
void MR3calcnacl_kadai12(double x[], int n, int atype[], int nat,
		 double pol[], double sigm[], double ipotro[],
		 double pc[], double pd[], double zz[],
		 int tblno, double xmax, int periodicflag,
		 double force[])
{
  int i,j;
  static VG_XVEC *d_x=NULL;
  static float *d_force=NULL;
  static VG_MATRIX *d_matrix=NULL;
  float xmaxf;
  VG_MATRIX *matrix=(VG_MATRIX *)force;
  VG_XVEC   *vec=(VG_XVEC *)force;
  if((periodicflag & 1)==0) xmax*=2.0;
  xmaxf=xmax;
  static float *forcef=NULL;
  static int n_bak=0;

  // ensure force has enough size for temporary array
  if(sizeof(double)*n*3<sizeof(VG_MATRIX)*nat*nat){
    fprintf(stderr,"** error : n*3<nat*nat **\n");
    exit(1);
  }
  if(nat>ATYPE){
    fprintf(stderr,"** error : nat is too large **\n");
    exit(1);
  }

  if(n!=n_bak){
    // free and allocate global memory
    int nalloc;
    static int nalloc_bak=0;
    if(n>NMAX) nalloc=n;
    else       nalloc=NMAX;
    if(nalloc!=nalloc_bak){
      CUDA_SAFE_CALL(cudaFree(d_x));
      CUDA_SAFE_CALL(cudaFree(d_force));
      CUDA_SAFE_CALL(cudaFree(d_matrix));
      CUDA_SAFE_CALL(cudaMalloc((void**)&d_x,sizeof(VG_XVEC)*(nalloc+NTHRE)));
      CUDA_SAFE_CALL(cudaMalloc((void**)&d_force,sizeof(float)*nalloc*3));
      CUDA_SAFE_CALL(cudaMalloc((void**)&d_matrix,sizeof(VG_MATRIX)*nat*nat));
      
      free(forcef);
      if((forcef=(float *)malloc(sizeof(float)*nalloc*3))==NULL){
	fprintf(stderr,"** error : can't malloc forcef **\n");
	exit(1);
      }
      bzero(forcef,sizeof(float)*nalloc*3);
      nalloc_bak=nalloc;
    }

    // send matrix
    for(i=0;i<nat;i++){
      for(j=0;j<nat;j++){
	matrix[i*nat+j].pol=(float)(pol[i*nat+j]);
	matrix[i*nat+j].sigm=(float)(sigm[i*nat+j]);
	matrix[i*nat+j].ipotro=(float)(ipotro[i*nat+j]);
	matrix[i*nat+j].pc=(float)(pc[i*nat+j]);
	matrix[i*nat+j].pd=(float)(pd[i*nat+j]);
	matrix[i*nat+j].zz=(float)(zz[i*nat+j]);
      }
    }
//    CUDA_SAFE_CALL(cudaMemcpy(d_matrix,matrix,sizeof(VG_MATRIX)*nat*nat,cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_matrix,matrix,sizeof(VG_MATRIX)*nat*nat));

    n_bak=n;
  }

  for(i=0;i<n;i++){
    for(j=0;j<3;j++){
      vec[i].r[j]=x[i*3+j];
    }
    vec[i].atype=atype[i];
  }
  CUDA_SAFE_CALL(cudaMemcpy(d_x,vec,sizeof(VG_XVEC)*n,cudaMemcpyHostToDevice));

  // call GPU kernel
  dim3 threads(NTHRE);
  dim3 grid((n+NTHRE-1)/NTHRE);
  nacl_kernel_gpu_kadai12<<< grid, threads >>>(d_x,n,nat,d_matrix,xmaxf,d_force);
  CUT_CHECK_ERROR("Kernel execution failed");

  // copy GPU result to host, and convert it to double
  CUDA_SAFE_CALL(cudaMemcpy(forcef,d_force,sizeof(float)*n*3,cudaMemcpyDeviceToHost));
  for(i=0;i<n;i++) for(j=0;j<3;j++) force[i*3+j]=forcef[i*3+j];
}


extern "C"
void MR3calcnacl_kadai13(double x[], int n, int atype[], int nat,
		 double pol[], double sigm[], double ipotro[],
		 double pc[], double pd[], double zz[],
		 int tblno, double xmax, int periodicflag,
		 double force[])
{
  int i,j,*index,count[ATYPE],offset[ATYPE],*atype2;
  double *x2,*f2;
  
  if((index=(int *)malloc(sizeof(int)*n))==NULL){
    fprintf(stderr,"** error : can't malloc index **\n");
    exit(1);
  }
  if((atype2=(int *)malloc(sizeof(int)*n))==NULL){
    fprintf(stderr,"** error : can't malloc atype2 **\n");
    exit(1);
  }
  if((x2=(double *)malloc(sizeof(double)*n*3))==NULL){
    fprintf(stderr,"** error : can't malloc x2 **\n");
    exit(1);
  }
  if((f2=(double *)malloc(sizeof(double)*n*3))==NULL){
    fprintf(stderr,"** error : can't malloc f2 **\n");
    exit(1);
  }
  for(i=0;i<nat;i++) count[i]=0;
  for(i=0;i<n;i++) count[atype[i]]++;
  for(i=1;i<nat;i++) offset[i]=count[i-1];
  offset[0]=0;
  for(i=0;i<n;i++) index[i]=offset[atype[i]]++;
  for(i=0;i<n;i++){
    for(j=0;j<3;j++) x2[index[i]*3+j]=x[i*3+j];
    atype2[index[i]]=atype[i];
  }
  MR3calcnacl_kadai12(x2,n,atype2,nat,pol,sigm,ipotro,pc,pd,zz,
		      tblno,xmax,periodicflag,f2);
  for(i=0;i<n;i++){
    for(j=0;j<3;j++) force[i*3+j]=f2[index[i]*3+j];
  }
  
  free(index);
  free(atype2);
  free(x2);
  free(f2);
}


extern "C"
void MR3calcnacl_org(double x[], int n, int atype[], int nat,
		 double pol[], double sigm[], double ipotro[],
		 double pc[], double pd[], double zz[],
		 int tblno, double xmax, int periodicflag,
		 double force[])
{
  int i,*d_atype;
  float *d_x,*d_pol,*d_sigm,*d_ipotro,*d_pc,*d_pd,*d_zz,*d_force,xmaxf=xmax;

  // ensure force has enough size for temporary array
  if(sizeof(double)*n*3<sizeof(float)*nat*nat){
    fprintf(stderr,"** error : n*3<nat*nat **\n");
    exit(1);
  }

  // allocate global memory and copy from host to GPU
  D2F_AND_COPY(n*3,x,d_x,force);
  D2F_AND_COPY(nat*nat,pol,d_pol,force);
  D2F_AND_COPY(nat*nat,sigm,d_sigm,force);
  D2F_AND_COPY(nat*nat,ipotro,d_ipotro,force);
  D2F_AND_COPY(nat*nat,pc,d_pc,force);
  D2F_AND_COPY(nat*nat,pd,d_pd,force);
  D2F_AND_COPY(nat*nat,zz,d_zz,force);
  CUDA_SAFE_CALL(cudaMalloc((void**)&d_atype,sizeof(int)*n));
  CUDA_SAFE_CALL(cudaMemcpy(d_atype,atype,sizeof(int)*n,cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMalloc((void**)&d_force,sizeof(float)*n*3));

  // call GPU kernel
  dim3 threads(64);
  dim3 grid((n+63)/64);
  nacl_kernel<<< grid, threads >>>(d_x,n,d_atype,nat,d_pol,d_sigm,d_ipotro,
				   d_pc,d_pd,d_zz,tblno,xmaxf,periodicflag,d_force);
  CUT_CHECK_ERROR("Kernel execution failed");

  // copy GPU result to host, and convert it to double
  CUDA_SAFE_CALL(cudaMemcpy(force,d_force,sizeof(float)*n*3,cudaMemcpyDeviceToHost));
  for(i=n*3-1;i>=0;i--) force[i]=((float *)force)[i];

  // free allocated global memory
  CUDA_SAFE_CALL(cudaFree(d_x));
  CUDA_SAFE_CALL(cudaFree(d_atype));
  CUDA_SAFE_CALL(cudaFree(d_pol));
  CUDA_SAFE_CALL(cudaFree(d_sigm));
  CUDA_SAFE_CALL(cudaFree(d_ipotro));
  CUDA_SAFE_CALL(cudaFree(d_pc));
  CUDA_SAFE_CALL(cudaFree(d_pd));
  CUDA_SAFE_CALL(cudaFree(d_zz));
  CUDA_SAFE_CALL(cudaFree(d_force));
}


extern "C" __global__
void nacl_kernel_initial(float *x, int n, int *atype, int nat, 
                 float *pol, float *sigm, float *ipotro,
                 float *pc, float *pd, float *zz, 
                 int tblno, float xmax, int periodicflag,
                 float *force)
{
  int i,j,k,t;
  float xmax1,dn2,r,inr,inr2,inr4,inr8,d3,dr[3],fi[3];
  float pb=(float)(0.338e-19/(14.39*1.60219e-19)),dphir;

  if((periodicflag & 1)==0) xmax *= 2.0f;
  xmax1 = 1.0f / xmax;
  i = blockIdx.x * 64 + threadIdx.x;
  if(i<n){
    for(k=0; k<3; k++) fi[k] = 0.0f;
    for(j=0; j<n; j++){
      dn2 = 0.0f;
      for(k=0; k<3; k++){
        dr[k] =  x[i*3+k] - x[j*3+k];
        dr[k] -= rintf(dr[k] * xmax1) * xmax;
        dn2   += dr[k] * dr[k];
      }
      if(dn2 != 0.0f){
        r     = sqrtf(dn2);
        inr   = 1.0f  / r;
        inr2  = inr  * inr;
        inr4  = inr2 * inr2;
        inr8  = inr4 * inr4;
        t     = atype[i] * nat + atype[j];
        d3    = pb * pol[t] * expf( (sigm[t] - r) * ipotro[t]);
        dphir = ( d3 * ipotro[t] * inr
                  - 6.0f * pc[t] * inr8
                  - 8.0f * pd[t] * inr8 * inr2
                  + inr2 * inr * zz[t] );
        for(k=0; k<3; k++) fi[k] += dphir * dr[k];
      }
    }
    for(k=0; k<3; k++) force[i*3+k] = fi[k];
  }
}

extern "C"
void MR3calcnacl_initial(double x[], int n, int atype[], int nat,
		 double pol[], double sigm[], double ipotro[],
		 double pc[], double pd[], double zz[],
		 int tblno, double xmax, int periodicflag,
		 double force[])
{
  int i,*d_atype;
  float *d_x,*d_pol,*d_sigm,*d_ipotro,*d_pc,*d_pd,*d_zz,*d_force,xmaxf=xmax;

  // ensure force has enough size for temporary array
  if(sizeof(double)*n*3<sizeof(float)*nat*nat){
    fprintf(stderr,"** error : n*3<nat*nat **\n");
    exit(1);
  }

  // allocate global memory and copy from host to GPU
  D2F_AND_COPY(n*3,x,d_x,force);
  D2F_AND_COPY(nat*nat,pol,d_pol,force);
  D2F_AND_COPY(nat*nat,sigm,d_sigm,force);
  D2F_AND_COPY(nat*nat,ipotro,d_ipotro,force);
  D2F_AND_COPY(nat*nat,pc,d_pc,force);
  D2F_AND_COPY(nat*nat,pd,d_pd,force);
  D2F_AND_COPY(nat*nat,zz,d_zz,force);
  CUDA_SAFE_CALL(cudaMalloc((void**)&d_atype,sizeof(int)*n));
  CUDA_SAFE_CALL(cudaMemcpy(d_atype,atype,sizeof(int)*n,cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMalloc((void**)&d_force,sizeof(float)*n*3));

  // call GPU kernel
  dim3 threads(64);
  dim3 grid((n+63)/64);
  nacl_kernel_initial<<< grid, threads >>>(d_x,n,d_atype,nat,d_pol,d_sigm,d_ipotro,
				   d_pc,d_pd,d_zz,tblno,xmaxf,periodicflag,d_force);
  CUT_CHECK_ERROR("Kernel execution failed");

  // copy GPU result to host, and convert it to double
  CUDA_SAFE_CALL(cudaMemcpy(force,d_force,sizeof(float)*n*3,cudaMemcpyDeviceToHost));
  for(i=n*3-1;i>=0;i--) force[i]=((float *)force)[i];

  // free allocated global memory
  CUDA_SAFE_CALL(cudaFree(d_x));
  CUDA_SAFE_CALL(cudaFree(d_atype));
  CUDA_SAFE_CALL(cudaFree(d_pol));
  CUDA_SAFE_CALL(cudaFree(d_sigm));
  CUDA_SAFE_CALL(cudaFree(d_ipotro));
  CUDA_SAFE_CALL(cudaFree(d_pc));
  CUDA_SAFE_CALL(cudaFree(d_pd));
  CUDA_SAFE_CALL(cudaFree(d_zz));
  CUDA_SAFE_CALL(cudaFree(d_force));
}


__device__ __inline__
void inter_128bit(float xj[3], float xi[3], float fi[3],
           VG_MATRIX d_matrix[], int t, float xmax, float xmax1)
{
  int k;
  float dn2,r,inr,inr2,inr4,inr8,d3,dr[3];
  float pb=(float)(0.338e-19/(14.39*1.60219e-19)),dphir;

  dn2 = 0.0f;
  for(k=0; k<3; k++){
    dr[k]  = xi[k] - xj[k];
    dr[k] -= rintf(dr[k] * xmax1) * xmax;
    dn2   += dr[k] * dr[k];
  }
  if(dn2 != 0.0f){
    r     = sqrtf(dn2);
    inr   = 1.0f / r;
    inr2  = inr  * inr;
    inr4  = inr2 * inr2;
    inr8  = inr4 * inr4;
    d3    = pb * d_matrix[t].pol * expf( (d_matrix[t].sigm - r) 
					 * d_matrix[t].ipotro);
    dphir = ( d3 * d_matrix[t].ipotro * inr
            - 6.0f * d_matrix[t].pc * inr8
            - 8.0f * d_matrix[t].pd * inr8 * inr2
	      + inr2 * inr * d_matrix[t].zz );
    for(k=0; k<3; k++) fi[k] += dphir * dr[k];
  }
}

extern "C" __global__
void nacl_kernel_128bit(VG_XVEC *x, int n, int nat, VG_MATRIX *d_matrix, 
                 float xmax, float *fvec)
{
  int tid = threadIdx.x;
  int i = blockIdx.x * NTHRE + tid;
  int j,k;
  float fi[3],xmax1=1.0f/xmax;
  int atypei;
  float xi[3];

  for(k=0; k<3; k++) fi[k] = 0.0f;
  for(k=0; k<3; k++) xi[k] = x[i].r[k];
  atypei = x[i].atype * nat;
  for (j=0; j<n; j++){
    inter_128bit(x[j].r, xi, fi, d_matrix, atypei + x[j].atype, xmax, xmax1);
  }
  for(k=0; k<3; k++) fvec[i*3+k] = fi[k];
}

extern "C"
void MR3calcnacl_128bit(double *x, int n, int *atype, int nat,
		 double *pol, double *sigm, double *ipotro,
		 double *pc, double *pd, double *zz,
		 int tblno, double xmax, int periodicflag,
		 double *force)
{
  int i,j;
  static VG_XVEC *d_x=NULL;
  static float *d_force=NULL;
  static VG_MATRIX *d_matrix=NULL;
  float xmaxf;
  VG_MATRIX *matrix=(VG_MATRIX *)force;
  static VG_XVEC   *vec=NULL;
  if((periodicflag & 1)==0) xmax*=2.0;
  xmaxf=xmax;
  static float *forcef=NULL;
  static int n_bak=0;

  // ensure force has enough size for temporary array
  if(sizeof(double)*n*3<sizeof(VG_MATRIX)*nat*nat){
    fprintf(stderr,"** error : n*3<nat*nat **\n");
    exit(1);
  }
  if(nat>ATYPE){
    fprintf(stderr,"** error : nat is too large **\n");
    exit(1);
  }

  if(n!=n_bak){
    // free and allocate global memory
    int nalloc;
    static int nalloc_bak=0;
    if(n>NMAX) nalloc=n;
    else       nalloc=NMAX;
    if(nalloc!=nalloc_bak){
      CUDA_SAFE_CALL(cudaFree(d_x));
      CUDA_SAFE_CALL(cudaFree(d_force));
      CUDA_SAFE_CALL(cudaFree(d_matrix));
      CUDA_SAFE_CALL(cudaMalloc((void**)&d_x,sizeof(VG_XVEC)*(nalloc+NTHRE)));
      CUDA_SAFE_CALL(cudaMalloc((void**)&d_force,sizeof(float)*(nalloc+NTHRE)*3));
      CUDA_SAFE_CALL(cudaMalloc((void**)&d_matrix,sizeof(VG_MATRIX)*nat*nat));
      
      free(vec);
      if((vec=(VG_XVEC *)malloc(sizeof(VG_XVEC)*(nalloc+NTHRE)))==NULL){
	fprintf(stderr,"** error : can't malloc vec **\n");
	exit(1);
      }
      free(forcef);
      if((forcef=(float *)malloc(sizeof(float)*nalloc*3))==NULL){
	fprintf(stderr,"** error : can't malloc forcef **\n");
	exit(1);
      }
      bzero(forcef,sizeof(float)*nalloc*3);
      nalloc_bak=nalloc;
    }

    // send matrix
    for(i=0;i<nat;i++){
      for(j=0;j<nat;j++){
	matrix[i*nat+j].pol=(float)(pol[i*nat+j]);
	matrix[i*nat+j].sigm=(float)(sigm[i*nat+j]);
	matrix[i*nat+j].ipotro=(float)(ipotro[i*nat+j]);
	matrix[i*nat+j].pc=(float)(pc[i*nat+j]);
	matrix[i*nat+j].pd=(float)(pd[i*nat+j]);
	matrix[i*nat+j].zz=(float)(zz[i*nat+j]);
      }
    }
    CUDA_SAFE_CALL(cudaMemcpy(d_matrix,matrix,sizeof(VG_MATRIX)*nat*nat,cudaMemcpyHostToDevice));

    n_bak=n;
  }

  for(i=0;i<(n+NTHRE-1)/NTHRE*NTHRE;i++){
    if(i<n){
      for(j=0;j<3;j++){
	vec[i].r[j]=x[i*3+j];
      }
      vec[i].atype=atype[i];
    }
    else{
      for(j=0;j<3;j++){
	vec[i].r[j]=0.0f;
      }
      vec[i].atype=0;
    }
  }
  CUDA_SAFE_CALL(cudaMemcpy(d_x,vec,sizeof(VG_XVEC)*((n+NTHRE-1)/NTHRE*NTHRE),
  			    cudaMemcpyHostToDevice));

  // call GPU kernel
  dim3 threads(NTHRE);
  dim3 grid((n+NTHRE-1)/NTHRE);
  nacl_kernel_128bit<<< grid, threads >>>(d_x,n,nat,d_matrix,xmaxf,d_force);
  CUT_CHECK_ERROR("Kernel execution failed");

  // copy GPU result to host, and convert it to double
  CUDA_SAFE_CALL(cudaMemcpy(forcef,d_force,sizeof(float)*n*3,cudaMemcpyDeviceToHost));
  for(i=0;i<n;i++) for(j=0;j<3;j++) force[i*3+j]=forcef[i*3+j];
}


__device__ __inline__
void inter_shared(float xj[3], float xi[3], float fi[3],
           VG_MATRIX d_matrix[], int t, float xmax, float xmax1)
{
  int k;
  float dn2,r,inr,inr2,inr4,inr8,d3,dr[3];
  float pb=(float)(0.338e-19/(14.39*1.60219e-19)),dphir;

  dn2 = 0.0f;
  for(k=0; k<3; k++){
    dr[k]  = xi[k] - xj[k];
    dr[k] -= rintf(dr[k] * xmax1) * xmax;
    dn2   += dr[k] * dr[k];
  }
  if(dn2 != 0.0f){
    r     = sqrtf(dn2);
    inr   = 1.0f / r;
    inr2  = inr  * inr;
    inr4  = inr2 * inr2;
    inr8  = inr4 * inr4;
    d3    = pb * d_matrix[t].pol * expf( (d_matrix[t].sigm - r) 
					 * d_matrix[t].ipotro);
    dphir = ( d3 * d_matrix[t].ipotro * inr
            - 6.0f * d_matrix[t].pc * inr8
            - 8.0f * d_matrix[t].pd * inr8 * inr2
	      + inr2 * inr * d_matrix[t].zz );
    for(k=0; k<3; k++) fi[k] += dphir * dr[k];
  }
}

extern "C" __global__
void nacl_kernel_shared(VG_XVEC *x, int n, int nat, VG_MATRIX *d_matrix, 
                 float xmax, float *fvec)
{
  int tid = threadIdx.x;
  int i = blockIdx.x * NTHRE + tid;
  int j,k;
  float fi[3],xmax1=1.0f/xmax;
  int atypei;
  float xi[3];
  __shared__ VG_XVEC s_xj[NTHRE];

  for(k=0; k<3; k++) fi[k] = 0.0f;
  for(k=0; k<3; k++) xi[k] = x[i].r[k];
  atypei = x[i].atype * nat;
  int na;
  na = n / NTHRE;
  na = na * NTHRE;
  for(j=0; j<na; j+=NTHRE){
    __syncthreads();
    s_xj[tid] = x[j+tid];
    __syncthreads();
#pragma unroll 64
    for(int js=0; js<NTHRE; js++)
      inter_shared(s_xj[js].r, xi, fi, d_matrix, atypei + s_xj[js].atype, xmax, xmax1);
  }
  for(j=na; j<n; j++){
    inter_shared(x[j].r, xi, fi, d_matrix, atypei + x[j].atype, xmax, xmax1);
  }
  for(k=0; k<3; k++) fvec[i*3+k] = fi[k];
}

extern "C"
void MR3calcnacl_shared(double x[], int n, int atype[], int nat,
		 double pol[], double sigm[], double ipotro[],
		 double pc[], double pd[], double zz[],
		 int tblno, double xmax, int periodicflag,
		 double force[])
{
  int i,j;
  static VG_XVEC *d_x=NULL;
  static float *d_force=NULL;
  static VG_MATRIX *d_matrix=NULL;
  float xmaxf;
  VG_MATRIX *matrix=(VG_MATRIX *)force;
  static VG_XVEC   *vec=NULL;
  if((periodicflag & 1)==0) xmax*=2.0;
  xmaxf=xmax;
  static float *forcef=NULL;
  static int n_bak=0;

  // ensure force has enough size for temporary array
  if(sizeof(double)*n*3<sizeof(VG_MATRIX)*nat*nat){
    fprintf(stderr,"** error : n*3<nat*nat **\n");
    exit(1);
  }
  if(nat>ATYPE){
    fprintf(stderr,"** error : nat is too large **\n");
    exit(1);
  }

  if(n!=n_bak){
    // free and allocate global memory
    int nalloc;
    static int nalloc_bak=0;
    if(n>NMAX) nalloc=n;
    else       nalloc=NMAX;
    if(nalloc!=nalloc_bak){
      CUDA_SAFE_CALL(cudaFree(d_x));
      CUDA_SAFE_CALL(cudaFree(d_force));
      CUDA_SAFE_CALL(cudaFree(d_matrix));
      CUDA_SAFE_CALL(cudaMalloc((void**)&d_x,sizeof(VG_XVEC)*(nalloc+NTHRE)));
      CUDA_SAFE_CALL(cudaMalloc((void**)&d_force,sizeof(float)*(nalloc+NTHRE)*3));
      CUDA_SAFE_CALL(cudaMalloc((void**)&d_matrix,sizeof(VG_MATRIX)*nat*nat));
      
      free(vec);
      if((vec=(VG_XVEC *)malloc(sizeof(VG_XVEC)*(nalloc+NTHRE)))==NULL){
	fprintf(stderr,"** error : can't malloc vec **\n");
	exit(1);
      }
      free(forcef);
      if((forcef=(float *)malloc(sizeof(float)*nalloc*3))==NULL){
	fprintf(stderr,"** error : can't malloc forcef **\n");
	exit(1);
      }
      bzero(forcef,sizeof(float)*nalloc*3);
      nalloc_bak=nalloc;
    }

    // send matrix
    for(i=0;i<nat;i++){
      for(j=0;j<nat;j++){
	matrix[i*nat+j].pol=(float)(pol[i*nat+j]);
	matrix[i*nat+j].sigm=(float)(sigm[i*nat+j]);
	matrix[i*nat+j].ipotro=(float)(ipotro[i*nat+j]);
	matrix[i*nat+j].pc=(float)(pc[i*nat+j]);
	matrix[i*nat+j].pd=(float)(pd[i*nat+j]);
	matrix[i*nat+j].zz=(float)(zz[i*nat+j]);
      }
    }
    CUDA_SAFE_CALL(cudaMemcpy(d_matrix,matrix,sizeof(VG_MATRIX)*nat*nat,cudaMemcpyHostToDevice));

    n_bak=n;
  }

  for(i=0;i<(n+NTHRE-1)/NTHRE*NTHRE;i++){
    if(i<n){
      for(j=0;j<3;j++){
	vec[i].r[j]=x[i*3+j];
      }
      vec[i].atype=atype[i];
    }
    else{
      for(j=0;j<3;j++){
	vec[i].r[j]=0.0f;
      }
      vec[i].atype=0;
    }
  }
  CUDA_SAFE_CALL(cudaMemcpy(d_x,vec,sizeof(VG_XVEC)*((n+NTHRE-1)/NTHRE*NTHRE),
  			    cudaMemcpyHostToDevice));

  // call GPU kernel
  dim3 threads(NTHRE);
  dim3 grid((n+NTHRE-1)/NTHRE);
  nacl_kernel_shared<<< grid, threads >>>(d_x,n,nat,d_matrix,xmaxf,d_force);
  CUT_CHECK_ERROR("Kernel execution failed");

  // copy GPU result to host, and convert it to double
  CUDA_SAFE_CALL(cudaMemcpy(forcef,d_force,sizeof(float)*n*3,cudaMemcpyDeviceToHost));
  for(i=0;i<n;i++) for(j=0;j<3;j++) force[i*3+j]=forcef[i*3+j];
}

__device__ __inline__
void inter_constant(float xj[3], float xi[3], float fi[3],
           int t, float xmax, float xmax1)
{
  int k;
  float dn2,r,inr,inr2,inr4,inr8,d3,dr[3];
  float pb=(float)(0.338e-19/(14.39*1.60219e-19)),dphir;

  dn2 = 0.0f;
  for(k=0; k<3; k++){
    dr[k]  = xi[k] - xj[k];
    dr[k] -= rintf(dr[k] * xmax1) * xmax;
    dn2   += dr[k] * dr[k];
  }
  if(dn2 != 0.0f){
    r     = sqrtf(dn2);
    inr   = 1.0f / r;
    inr2  = inr  * inr;
    inr4  = inr2 * inr2;
    inr8  = inr4 * inr4;
    d3    = pb * c_matrix[t].pol * expf( (c_matrix[t].sigm - r) 
					 * c_matrix[t].ipotro);
    dphir = ( d3 * c_matrix[t].ipotro * inr
            - 6.0f * c_matrix[t].pc * inr8
            - 8.0f * c_matrix[t].pd * inr8 * inr2
	      + inr2 * inr * c_matrix[t].zz );
    for(k=0; k<3; k++) fi[k] += dphir * dr[k];
  }
}

extern "C" __global__
void nacl_kernel_constant(VG_XVEC *x, int n, int nat, 
                 float xmax, float *fvec)
{
  int tid = threadIdx.x;
  int i = blockIdx.x * NTHRE + tid;
  int j,k;
  float fi[3],xmax1=1.0f/xmax;
  int atypei;
  float xi[3];
  __shared__ VG_XVEC s_xj[NTHRE];

  for(k=0; k<3; k++) fi[k] = 0.0f;
  for(k=0; k<3; k++) xi[k] = x[i].r[k];
  atypei = x[i].atype * nat;
  int na;
  na = n / NTHRE;
  na = na * NTHRE;
  for(j=0; j<na; j+=NTHRE){
    __syncthreads();
    s_xj[tid] = x[j+tid];
    __syncthreads();
#pragma unroll 64
    for(int js=0; js<NTHRE; js++)
      inter_constant(s_xj[js].r, xi, fi, atypei + s_xj[js].atype, xmax, xmax1);
  }
  for(j=na; j<n; j++){
    inter_constant(x[j].r, xi, fi, atypei + x[j].atype, xmax, xmax1);
  }
  for(k=0; k<3; k++) fvec[i*3+k] = fi[k];
}

extern "C"
void MR3calcnacl_constant(double x[], int n, int atype[], int nat,
		 double pol[], double sigm[], double ipotro[],
		 double pc[], double pd[], double zz[],
		 int tblno, double xmax, int periodicflag,
		 double force[])
{
  int i,j;
  static VG_XVEC *d_x=NULL;
  static float *d_force=NULL;
  float xmaxf;
  VG_MATRIX *matrix=(VG_MATRIX *)force;
  static VG_XVEC   *vec=NULL;
  if((periodicflag & 1)==0) xmax*=2.0;
  xmaxf=xmax;
  static float *forcef=NULL;
  static int n_bak=0;

  // ensure force has enough size for temporary array
  if(sizeof(double)*n*3<sizeof(VG_MATRIX)*nat*nat){
    fprintf(stderr,"** error : n*3<nat*nat **\n");
    exit(1);
  }
  if(nat>ATYPE){
    fprintf(stderr,"** error : nat is too large **\n");
    exit(1);
  }

  if(n!=n_bak){
    // free and allocate global memory
    int nalloc;
    static int nalloc_bak=0;
    if(n>NMAX) nalloc=n;
    else       nalloc=NMAX;
    if(nalloc!=nalloc_bak){
      CUDA_SAFE_CALL(cudaFree(d_x));
      CUDA_SAFE_CALL(cudaFree(d_force));
      CUDA_SAFE_CALL(cudaMalloc((void**)&d_x,sizeof(VG_XVEC)*(nalloc+NTHRE)));
      CUDA_SAFE_CALL(cudaMalloc((void**)&d_force,sizeof(float)*(nalloc+NTHRE)*3));
      
      free(vec);
      if((vec=(VG_XVEC *)malloc(sizeof(VG_XVEC)*(nalloc+NTHRE)))==NULL){
	fprintf(stderr,"** error : can't malloc vec **\n");
	exit(1);
      }
      free(forcef);
      if((forcef=(float *)malloc(sizeof(float)*nalloc*3))==NULL){
	fprintf(stderr,"** error : can't malloc forcef **\n");
	exit(1);
      }
      bzero(forcef,sizeof(float)*nalloc*3);
      nalloc_bak=nalloc;
    }

    // send matrix
    for(i=0;i<nat;i++){
      for(j=0;j<nat;j++){
	matrix[i*nat+j].pol=(float)(pol[i*nat+j]);
	matrix[i*nat+j].sigm=(float)(sigm[i*nat+j]);
	matrix[i*nat+j].ipotro=(float)(ipotro[i*nat+j]);
	matrix[i*nat+j].pc=(float)(pc[i*nat+j]);
	matrix[i*nat+j].pd=(float)(pd[i*nat+j]);
	matrix[i*nat+j].zz=(float)(zz[i*nat+j]);
      }
    }
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_matrix,matrix,sizeof(VG_MATRIX)*nat*nat));

    n_bak=n;
  }

  for(i=0;i<(n+NTHRE-1)/NTHRE*NTHRE;i++){
    if(i<n){
      for(j=0;j<3;j++){
	vec[i].r[j]=x[i*3+j];
      }
      vec[i].atype=atype[i];
    }
    else{
      for(j=0;j<3;j++){
	vec[i].r[j]=0.0f;
      }
      vec[i].atype=0;
    }
  }
  CUDA_SAFE_CALL(cudaMemcpy(d_x,vec,sizeof(VG_XVEC)*((n+NTHRE-1)/NTHRE*NTHRE),
  			    cudaMemcpyHostToDevice));

  // call GPU kernel
  dim3 threads(NTHRE);
  dim3 grid((n+NTHRE-1)/NTHRE);
  nacl_kernel_constant<<< grid, threads >>>(d_x,n,nat,xmaxf,d_force);
  CUT_CHECK_ERROR("Kernel execution failed");

  // copy GPU result to host, and convert it to double
  CUDA_SAFE_CALL(cudaMemcpy(forcef,d_force,sizeof(float)*n*3,cudaMemcpyDeviceToHost));
  for(i=0;i<n;i++) for(j=0;j<3;j++) force[i*3+j]=forcef[i*3+j];
}

__device__ __inline__
void inter_if(float xj[3], float xi[3], float fi[3],
           int t, float xmax, float xmax1)
{
  int k;
  float dn2,r,inr,inr2,inr4,inr8,d3,dr[3];
  float pb=(float)(0.338e-19/(14.39*1.60219e-19)),dphir;

  dn2 = 0.0f;
  for(k=0; k<3; k++){
    dr[k]  = xi[k] - xj[k];
    dr[k] -= rintf(dr[k] * xmax1) * xmax;
    dn2   += dr[k] * dr[k];
  }
  r     = sqrtf(dn2);
#if 1
  inr   = 1.0f / r;
#elif 0
  if(dn2 != 0.0f) inr   = 1.0f / r;
  else            inr   = 0.0f;
#elif 0
  if(dn2 == 0.0f) inr   = 0.0f;
  else            inr   = 1.0f / r;
#else
  inr   = 1.0f / r;
  if(dn2 == 0.0f) inr   = 0.0f;
#endif
  inr2  = inr  * inr;
  inr4  = inr2 * inr2;
  inr8  = inr4 * inr4;
  d3    = pb * c_matrix[t].pol * expf( (c_matrix[t].sigm - r) 
                               * c_matrix[t].ipotro);
  dphir = ( d3 * c_matrix[t].ipotro * inr
          - 6.0f * c_matrix[t].pc * inr8
          - 8.0f * c_matrix[t].pd * inr8 * inr2
          + inr2 * inr * c_matrix[t].zz );
#if 1
  if(dn2 == 0.0f) dphir = 0.0f;
#endif
  for(k=0; k<3; k++) fi[k] += dphir * dr[k];
}

extern "C" __global__
void nacl_kernel_if(VG_XVEC *x, int n, int nat, 
                 float xmax, float *fvec)
{
  int tid = threadIdx.x;
  int i = blockIdx.x * NTHRE + tid;
  int j,k;
  float fi[3],xmax1=1.0f/xmax;
  int atypei;
  float xi[3];
  __shared__ VG_XVEC s_xj[NTHRE];

  for(k=0; k<3; k++) fi[k] = 0.0f;
  for(k=0; k<3; k++) xi[k] = x[i].r[k];
  atypei = x[i].atype * nat;
  int na;
  na = n / NTHRE;
  na = na * NTHRE;
  for(j=0; j<na; j+=NTHRE){
    __syncthreads();
    s_xj[tid] = x[j+tid];
    __syncthreads();
#pragma unroll 64
    for(int js=0; js<NTHRE; js++)
      inter_if(s_xj[js].r, xi, fi, atypei + s_xj[js].atype, xmax, xmax1);
  }
  for(j=na; j<n; j++){
    inter_if(x[j].r, xi, fi, atypei + x[j].atype, xmax, xmax1);
  }
  for(k=0; k<3; k++) fvec[i*3+k] = fi[k];
}

extern "C"
void MR3calcnacl_if(double x[], int n, int atype[], int nat,
		 double pol[], double sigm[], double ipotro[],
		 double pc[], double pd[], double zz[],
		 int tblno, double xmax, int periodicflag,
		 double force[])
{
  int i,j;
  static VG_XVEC *d_x=NULL;
  static float *d_force=NULL;
  float xmaxf;
  VG_MATRIX *matrix=(VG_MATRIX *)force;
  static VG_XVEC   *vec=NULL;
  if((periodicflag & 1)==0) xmax*=2.0;
  xmaxf=xmax;
  static float *forcef=NULL;
  static int n_bak=0;

#if 0
  vec=(VG_XVEC *)force;
#endif

  // ensure force has enough size for temporary array
  if(sizeof(double)*n*3<sizeof(VG_MATRIX)*nat*nat){
    fprintf(stderr,"** error : n*3<nat*nat **\n");
    exit(1);
  }
  if(nat>ATYPE){
    fprintf(stderr,"** error : nat is too large **\n");
    exit(1);
  }

  if(n!=n_bak){
    // free and allocate global memory
    int nalloc;
    static int nalloc_bak=0;
    if(n>NMAX) nalloc=n;
    else       nalloc=NMAX;
    if(nalloc!=nalloc_bak){
      CUDA_SAFE_CALL(cudaFree(d_x));
      CUDA_SAFE_CALL(cudaFree(d_force));
      CUDA_SAFE_CALL(cudaMalloc((void**)&d_x,sizeof(VG_XVEC)*(nalloc+NTHRE)));
      CUDA_SAFE_CALL(cudaMalloc((void**)&d_force,sizeof(float)*(nalloc+NTHRE)*3));

#if 1     
      free(vec);
      if((vec=(VG_XVEC *)malloc(sizeof(VG_XVEC)*(nalloc+NTHRE)))==NULL){
	fprintf(stderr,"** error : can't malloc vec **\n");
	exit(1);
      }
#endif
      free(forcef);
      if((forcef=(float *)malloc(sizeof(float)*nalloc*3))==NULL){
	fprintf(stderr,"** error : can't malloc forcef **\n");
	exit(1);
      }
      bzero(forcef,sizeof(float)*nalloc*3);
      nalloc_bak=nalloc;
    }

    // send matrix
    for(i=0;i<nat;i++){
      for(j=0;j<nat;j++){
	matrix[i*nat+j].pol=(float)(pol[i*nat+j]);
	matrix[i*nat+j].sigm=(float)(sigm[i*nat+j]);
	matrix[i*nat+j].ipotro=(float)(ipotro[i*nat+j]);
	matrix[i*nat+j].pc=(float)(pc[i*nat+j]);
	matrix[i*nat+j].pd=(float)(pd[i*nat+j]);
	matrix[i*nat+j].zz=(float)(zz[i*nat+j]);
      }
    }
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_matrix,matrix,sizeof(VG_MATRIX)*nat*nat));

    n_bak=n;
  }

#if 0
  for(i=0;i<n;i++){
#else
  for(i=0;i<(n+NTHRE-1)/NTHRE*NTHRE;i++){
#endif
    if(i<n){
      for(j=0;j<3;j++){
	vec[i].r[j]=x[i*3+j];
      }
      vec[i].atype=atype[i];
    }
    else{
      for(j=0;j<3;j++){
	vec[i].r[j]=0.0f;
      }
      vec[i].atype=0;
    }
  }
#if 0
  CUDA_SAFE_CALL(cudaMemcpy(d_x,vec,sizeof(VG_XVEC)*n,
  			    cudaMemcpyHostToDevice));
#else
  CUDA_SAFE_CALL(cudaMemcpy(d_x,vec,sizeof(VG_XVEC)*((n+NTHRE-1)/NTHRE*NTHRE),
  			    cudaMemcpyHostToDevice));
#endif

  // call GPU kernel
  dim3 threads(NTHRE);
  dim3 grid((n+NTHRE-1)/NTHRE);
  nacl_kernel_if<<< grid, threads >>>(d_x,n,nat,xmaxf,d_force);
//  nacl_kernel_gpu_kadai12<<< grid, threads >>>(d_x,n,nat,NULL,xmaxf,d_force);
  CUT_CHECK_ERROR("Kernel execution failed");

  // copy GPU result to host, and convert it to double
  CUDA_SAFE_CALL(cudaMemcpy(forcef,d_force,sizeof(float)*n*3,cudaMemcpyDeviceToHost));
  for(i=0;i<n;i++) for(j=0;j<3;j++) force[i*3+j]=forcef[i*3+j];
}


extern "C" __global__
void nacl_kernel_if2(VG_XVEC *x, int n, int nat, 
                 float xmax, float *fvec)
{
  int tid = threadIdx.x;
  int jdiv = tid/NTHRE2;
  int i = blockIdx.x * NTHRE2 + (tid & (NTHRE2-1));
  int j,k;
  float xmax1=1.0f/xmax;
  int atypei;
  float xi[3];
  __shared__ VG_XVEC s_xj[NTHRE];
  __shared__ float   s_fi[NTHRE][3];

  for(k=0; k<3; k++) s_fi[tid][k] = 0.0f;
  for(k=0; k<3; k++) xi[k] = x[i].r[k];
  atypei = x[i].atype * nat;
  int na;
  na = n / NTHRE;
  na = na * NTHRE;
  for(j=0; j<na; j+=NTHRE){
    __syncthreads();
    s_xj[tid] = x[j+tid];
    __syncthreads();
#pragma unroll 16
    for(int js=jdiv; js<NTHRE; js+=NDIV)
      inter_if(s_xj[js].r, xi, s_fi[tid], atypei + s_xj[js].atype, xmax, xmax1);
  }
  for(j=na+jdiv; j<n; j+=NDIV){
    inter_if(x[j].r, xi, s_fi[tid], atypei + x[j].atype, xmax, xmax1);
  }
#if NTHRE>=512 && NTHRE2<=256
  __syncthreads();
  if(tid<256) for(k=0;k<3;k++) s_fi[tid][k]+=s_fi[tid+256][k];
#endif
#if NTHRE>=256 && NTHRE2<=128
  __syncthreads();
  if(tid<128) for(k=0;k<3;k++) s_fi[tid][k]+=s_fi[tid+128][k];
#endif
#if NTHRE>=128 && NTHRE2<=64
  __syncthreads();
  if(tid<64) for(k=0;k<3;k++) s_fi[tid][k]+=s_fi[tid+64][k];
#endif
#if NTHRE>=64 && NTHRE2<=32
  __syncthreads();
  if(tid<32) for(k=0;k<3;k++) s_fi[tid][k]+=s_fi[tid+32][k];
#endif
#if NTHRE2<=16
  if(tid<16) for(k=0;k<3;k++) s_fi[tid][k]+=s_fi[tid+16][k];
#endif
#if NTHRE2<=8
  if(tid<8) for(k=0;k<3;k++) s_fi[tid][k]+=s_fi[tid+8][k];
#endif
#if NTHRE2<=4
  if(tid<4) for(k=0;k<3;k++) s_fi[tid][k]+=s_fi[tid+4][k];
#endif
#if NTHRE2<=2
  if(tid<2) for(k=0;k<3;k++) s_fi[tid][k]+=s_fi[tid+2][k];
#endif
#if NTHRE2<=1
  if(tid<1) for(k=0;k<3;k++) s_fi[tid][k]+=s_fi[tid+1][k];
#endif
  if(jdiv==0) for(k=0; k<3; k++) fvec[i*3+k] = s_fi[tid][k];
}

extern "C"
void MR3calcnacl_if2(double x[], int n, int atype[], int nat,
		 double pol[], double sigm[], double ipotro[],
		 double pc[], double pd[], double zz[],
		 int tblno, double xmax, int periodicflag,
		 double force[])
{
  int i,j;
  static VG_XVEC *d_x=NULL;
  static float *d_force=NULL;
  float xmaxf;
  VG_MATRIX *matrix=(VG_MATRIX *)force;
  static VG_XVEC   *vec=NULL;
  if((periodicflag & 1)==0) xmax*=2.0;
  xmaxf=xmax;
  static float *forcef=NULL;
  static int n_bak=0;

#if 0
  vec=(VG_XVEC *)force;
#endif

  // ensure force has enough size for temporary array
  if(sizeof(double)*n*3<sizeof(VG_MATRIX)*nat*nat){
    fprintf(stderr,"** error : n*3<nat*nat **\n");
    exit(1);
  }
  if(nat>ATYPE){
    fprintf(stderr,"** error : nat is too large **\n");
    exit(1);
  }

  if(n!=n_bak){
    // free and allocate global memory
    int nalloc;
    static int nalloc_bak=0;
    if(n>NMAX) nalloc=n;
    else       nalloc=NMAX;
    if(nalloc!=nalloc_bak){
      CUDA_SAFE_CALL(cudaFree(d_x));
      CUDA_SAFE_CALL(cudaFree(d_force));
      CUDA_SAFE_CALL(cudaMalloc((void**)&d_x,sizeof(VG_XVEC)*(nalloc+NTHRE2)));
      CUDA_SAFE_CALL(cudaMalloc((void**)&d_force,sizeof(float)*(nalloc+NTHRE2)*3));

#if 1     
      free(vec);
      if((vec=(VG_XVEC *)malloc(sizeof(VG_XVEC)*(nalloc+NTHRE2)))==NULL){
	fprintf(stderr,"** error : can't malloc vec **\n");
	exit(1);
      }
#endif
      free(forcef);
      if((forcef=(float *)malloc(sizeof(float)*nalloc*3))==NULL){
	fprintf(stderr,"** error : can't malloc forcef **\n");
	exit(1);
      }
      bzero(forcef,sizeof(float)*nalloc*3);
      nalloc_bak=nalloc;
    }

    // send matrix
    for(i=0;i<nat;i++){
      for(j=0;j<nat;j++){
	matrix[i*nat+j].pol=(float)(pol[i*nat+j]);
	matrix[i*nat+j].sigm=(float)(sigm[i*nat+j]);
	matrix[i*nat+j].ipotro=(float)(ipotro[i*nat+j]);
	matrix[i*nat+j].pc=(float)(pc[i*nat+j]);
	matrix[i*nat+j].pd=(float)(pd[i*nat+j]);
	matrix[i*nat+j].zz=(float)(zz[i*nat+j]);
      }
    }
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_matrix,matrix,sizeof(VG_MATRIX)*nat*nat));

    n_bak=n;
  }

  for(i=0;i<(n+NTHRE2-1)/NTHRE2*NTHRE2;i++){
    if(i<n){
      for(j=0;j<3;j++){
	vec[i].r[j]=x[i*3+j];
      }
      vec[i].atype=atype[i];
    }
    else{
      for(j=0;j<3;j++){
	vec[i].r[j]=0.0f;
      }
      vec[i].atype=0;
    }
  }
  CUDA_SAFE_CALL(cudaMemcpy(d_x,vec,sizeof(VG_XVEC)*((n+NTHRE2-1)/NTHRE2*NTHRE2),
  			    cudaMemcpyHostToDevice));

  // call GPU kernel
  dim3 threads(NTHRE);
  dim3 grid((n*NDIV+NTHRE-1)/NTHRE);
  nacl_kernel_if2<<< grid, threads >>>(d_x,n,nat,xmaxf,d_force);
//  nacl_kernel_gpu_kadai12<<< grid, threads >>>(d_x,n,nat,NULL,xmaxf,d_force);
  CUT_CHECK_ERROR("Kernel execution failed");

  // copy GPU result to host, and convert it to double
  CUDA_SAFE_CALL(cudaMemcpy(forcef,d_force,sizeof(float)*n*3,cudaMemcpyDeviceToHost));
  for(i=0;i<n;i++) for(j=0;j<3;j++) force[i*3+j]=forcef[i*3+j];
}


extern "C"
void MR3calcnacl_sort(double x[], int n, int atype[], int nat,
		 double pol[], double sigm[], double ipotro[],
		 double pc[], double pd[], double zz[],
		 int tblno, double xmax, int periodicflag,
		 double force[])
{
  int i,j,*index,count[ATYPE],offset[ATYPE],*atype2;
  double *x2,*f2;
  
  if((index=(int *)malloc(sizeof(int)*n))==NULL){
    fprintf(stderr,"** error : can't malloc index **\n");
    exit(1);
  }
  if((atype2=(int *)malloc(sizeof(int)*n))==NULL){
    fprintf(stderr,"** error : can't malloc atype2 **\n");
    exit(1);
  }
  if((x2=(double *)malloc(sizeof(double)*n*3))==NULL){
    fprintf(stderr,"** error : can't malloc x2 **\n");
    exit(1);
  }
  if((f2=(double *)malloc(sizeof(double)*n*3))==NULL){
    fprintf(stderr,"** error : can't malloc f2 **\n");
    exit(1);
  }
  for(i=0;i<nat;i++) count[i]=0;
  for(i=0;i<n;i++) count[atype[i]]++;
  for(i=1;i<nat;i++) offset[i]=count[i-1];
  offset[0]=0;
  for(i=0;i<n;i++) index[i]=offset[atype[i]]++;
  for(i=0;i<n;i++){
    for(j=0;j<3;j++) x2[index[i]*3+j]=x[i*3+j];
    atype2[index[i]]=atype[i];
  }
  MR3calcnacl_if(x2,n,atype2,nat,pol,sigm,ipotro,pc,pd,zz,
		      tblno,xmax,periodicflag,f2);
  for(i=0;i<n;i++){
    for(j=0;j<3;j++) force[i*3+j]=f2[index[i]*3+j];
  }
  
  free(index);
  free(atype2);
  free(x2);
  free(f2);
}


extern "C"
void MR3calcnacl(double x[], int n, int atype[], int nat,
		 double pol[], double sigm[], double ipotro[],
		 double pc[], double pd[], double zz[],
		 int tblno, double xmax, int periodicflag,
		 double force[])
{
  MR3calcnacl_if2(x,n,atype,nat,
		  pol,sigm,ipotro,
		  pc,pd,zz,
		  tblno,xmax,periodicflag,
		  force);
}


