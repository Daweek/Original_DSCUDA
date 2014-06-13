#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void MR3init(void)
{
}

void MR3free(void)
{
}

void MR3calcnacl(double x[], int n, int atype[], int nat,
		 double pol[], double sigm[], double ipotro[],
		 double pc[], double pd[], double zz[],
		 int tblno, double xmax, int periodicflag,
		 double force[])
{
  int i,j,k,t;
  double xmax1,dn2,r,inr,inr2,inr4,inr8,d3,dr[3],fi[3];
  double pb=0.338e-19/(14.39*1.60219e-19),dphir; 
  if((periodicflag & 1)==0) xmax *= 2;
  xmax1 = 1.0 / xmax;
#pragma omp parallel for private(k,j,dn2,dr,r,inr,inr2,inr4,inr8,t,d3,dphir,fi)
  for(i=0; i<n; i++){
    for(k=0; k<3; k++) fi[k] = 0.0;
    for(j=0; j<n; j++){
      dn2 = 0.0;
      for(k=0; k<3; k++){
	dr[k] =  x[i*3+k] - x[j*3+k];
	dr[k] -= rint(dr[k] * xmax1) * xmax;
	dn2   += dr[k] * dr[k];
      }
      if(dn2 != 0.0){
	r     = sqrt(dn2);
	inr   = 1.0  / r;
	inr2  = inr  * inr;
	inr4  = inr2 * inr2;
	inr8  = inr4 * inr4;
	t     = atype[i] * nat + atype[j];
	d3    = pb * pol[t] * exp( (sigm[t] - r) * ipotro[t]);
	dphir = ( d3 * ipotro[t] * inr
		  - 6.0 * pc[t] * inr8
		  - 8.0 * pd[t] * inr8 * inr2
		  + inr2 * inr * zz[t] );
	for(k=0; k<3; k++) fi[k] += dphir * dr[k];
      }
    }
    for(k=0; k<3; k++) force[i*3+k] = fi[k];
  }
}
    
