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
void nacl_kernel_if2(float *x_org, int n, int nat, 
                 float xmax, float *fvec)
{
  VG_XVEC *x = (VG_XVEC *)x_org;
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
//  static VG_XVEC *d_x=NULL;
  static float *d_x=NULL;
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
		WARN(3, "matrix pointer: %p\n", &matrix);
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
  CUT_CHECK_ERROR("Kernel execution failed");

  // copy GPU result to host, and convert it to double
  CUDA_SAFE_CALL(cudaMemcpy(forcef,d_force,sizeof(float)*n*3,cudaMemcpyDeviceToHost));
  for(i=0;i<n;i++) for(j=0;j<3;j++) force[i*3+j]=forcef[i*3+j];
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


