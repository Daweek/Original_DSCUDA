#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <sys/time.h>
#include <math.h>

void
create_cold_homosphere(int n, double *mj, double (*xj)[3], double (*vj)[3])
{
    int i, k;

    srand48(1234);

    /*
     * mass
     */
    double mass = 1.0/n;
    for (i = 0; i < n; i++) {
	mj[i] = mass;
    }

    /*
     * velocity.
     */
    for (i = 0; i < n; i++) {
	vj[i][0] = 0.0;
	vj[i][1] = 0.0;
	vj[i][2] = 0.0;
    }

    /*
     * position
     */
    i = 0;
    while (i < n) {
	double x, y, z;
	x = 2.0*drand48()-1.0;
	y = 2.0*drand48()-1.0;
	z = 2.0*drand48()-1.0;
	if (x*x+y*y+z*z<1.0) {
	    xj[i][0] = x;
	    xj[i][1] = y;
	    xj[i][2] = z;
	    i++;
	}
    }

    /* centrate the distributioin */
    double cm_pos[3] = {0.0, 0.0, 0.0};
    double cm_vel[3] = {0.0, 0.0, 0.0};
    for (i = 0; i < n; i++) {
        for (k = 0; k < 3; k++) {
            cm_pos[k] += xj[i][k];
        }
        for (k = 0; k < 3; k++) {
            cm_vel[k] += vj[i][k];
        }
    }
    for (k = 0; k < 3; k++) {
        cm_pos[k] /= n;
        cm_vel[k] /= n;
    }
    for (i = 0; i < n; i++) {
        for (k = 0; k < 3; k++) {
            xj[i][k] -= cm_pos[k];
        }
        for (k = 0; k < 3; k++) {
            vj[i][k] -= cm_vel[k];
        }
    }
}

static double
xrandom(double min, double max)
{
    double val = drand48();
    val = val * (max - min) + min;
    return val;
}

/*
 * Plummer distribution.
 * equal-mass, 
 * region of radius 1.0. total mass is normalized to 1.0.
 */
void
create_plummer(int n, double *mj, double (*xj)[3], double (*vj)[3])
{
    int i, k;

    srand48(1234);

    /*
     * mass
     */
    double mass = 1.0/n;
    for (i = 0; i < n; i++) {
	mj[i] = mass;
    }

    /*
     * position & velocity.
     */
    double scalefactor;
    double inv_scalefactor;
    double sqrt_scalefactor;
    double mrfrac;
    double mlow = 0.0;
    double mfrac = 0.999;
    double rfrac=22.8042468;

    scalefactor = 16.0 / (3.0 * M_PI);
    inv_scalefactor = 1.0 / scalefactor;
    sqrt_scalefactor = sqrt(scalefactor);

    rfrac *= scalefactor;          /* from VIRIAL to STRUCTURAL units */
    mrfrac = rfrac*rfrac*rfrac / pow(1.0 + rfrac*rfrac, 1.5);
    if (mrfrac < mfrac) {
	mfrac = mrfrac;            /* mfrac = min(mfrac, m(rfrac)) */
    }

    for (i = 0; i < n; i++) {
        double  radius;
        double  velocity;
        double  theta, phi;
        double  x, y;

        radius = 1.0 / sqrt( pow (xrandom(mlow, mfrac), -2.0/3.0) - 1.0);

	theta = acos(xrandom(-1.0, 1.0));
	phi = xrandom(0.0, 2.0 * M_PI);
	xj[i][0] = radius * sin( theta ) * cos( phi );
	xj[i][1] = radius * sin( theta ) * sin( phi );
        xj[i][2] = radius * cos( theta );

	x = 0.0;
	y = 0.1;

	while (y > x*x*pow( 1.0 - x*x, 3.5)) {
	    x = xrandom(0.0, 1.0);
	    y = xrandom(0.0, 0.1);
        }

	velocity = x * sqrt(2.0) * pow( 1.0 + radius*radius, -0.25);
	theta = acos(xrandom(-1.0, 1.0));
	phi = xrandom(0.0, 2.0 * M_PI);
	vj[i][0] = velocity * sin( theta ) * cos( phi );
	vj[i][1] = velocity * sin( theta ) * sin( phi );
	vj[i][2] = velocity * cos( theta );
    }

    /* rescale position & velocity. */
    for (i = 0; i < n; i++) {
        for (k = 0; k < 3; k++) {
            xj[i][k] *= inv_scalefactor;
        }
        for (k = 0; k < 3; k++) {
            vj[i][k] *= sqrt_scalefactor;
        }
    }

    /* centrate the distributioin */
    double cm_pos[3] = {0.0, 0.0, 0.0};
    double cm_vel[3] = {0.0, 0.0, 0.0};
    for (i = 0; i < n; i++) {
        for (k = 0; k < 3; k++) {
            cm_pos[k] += xj[i][k];
        }
        for (k = 0; k < 3; k++) {
            cm_vel[k] += vj[i][k];
        }
    }
    for (k = 0; k < 3; k++) {
        cm_pos[k] /= n;
        cm_vel[k] /= n;
    }
    for (i = 0; i < n; i++) {
        for (k = 0; k < 3; k++) {
            xj[i][k] -= cm_pos[k];
        }
        for (k = 0; k < 3; k++) {
            vj[i][k] -= cm_vel[k];
        }
    }
}


void
readnbody(int *nj, double *mj, double (*xj)[3], double (*vj)[3], char *fname)
{
    int i, dummy;
    double dummyd;
    FILE *fp;

    fp = fopen(fname, "r");
    if (fp == NULL) {
        char msg[256];
        sprintf(msg, "readnbody: fname:%s", fname);
        perror(msg);
        exit(1);
    }
    fscanf(fp, "%d\n", nj);
    fscanf(fp, "%d\n", &dummy);
    fscanf(fp, "%lf\n", &dummyd);
    fprintf(stderr, "nj: %d\n", *nj);
    for (i = 0; i < *nj; i++) {
        fscanf(fp, "%lf\n", mj+i);
    }
    for (i = 0; i < *nj; i++) {
        fscanf(fp, "%lf %lf %lf\n",
               xj[i]+0, xj[i]+1, xj[i]+2);
    }
    for (i = 0; i < *nj; i++) {
        fscanf(fp, "%lf %lf %lf\n",
               vj[i]+0, vj[i]+1, vj[i]+2);
    }
}

void
writenbody(int nj, double *mj, double (*xj)[3], double (*vj)[3], char *fname)
{
    int i, dummy;
    FILE *fp;

    fp = fopen(fname, "w");
    fprintf(fp, "%d\n", nj);
    fprintf(fp, "%d\n", 3);
    fprintf(fp, "%e\n", 0.0);
    for (i = 0; i < nj; i++) {
        fprintf(fp, " % 15.13E\n", mj[i]);
    }
    for (i = 0; i < nj; i++) {
        fprintf(fp, " % 15.13E % 15.13E % 15.13E\n",
                xj[i][0], xj[i][1], xj[i][2]);
    }
    for (i = 0; i < nj; i++) {
        fprintf(fp, " % 15.13E % 15.13E % 15.13E\n",
                vj[i][0], vj[i][1], vj[i][2]);
    }

    fclose(fp);
}

void
writefingerprint(int nj, double *mj, double (*xj)[3], double (*vj)[3], char *fname)
{
    unsigned long long int checksum = 0LL;
    int i, k;
    FILE *fp;

    fp = fopen(fname, "w");

    for (i = 0; i < nj; i++) {
        checksum += *(unsigned long long *)&(mj[i]);
    }

    for (i = 0; i < nj; i++) {
        for (k = 0; k < 3; k++) {
            checksum += *(unsigned long long *)&(xj[i][k]);
        }
    }

    for (i = 0; i < nj; i++) {
        for (k = 0; k < 3; k++) {
            checksum += *(unsigned long long *)&(vj[i][k]);
        }
    }

    fprintf(fp, "checksum: %016llx\n", checksum);
    fclose(fp);
}

void
push_velocity(double (*vj)[3], double (*a)[3], double dt, int nj)
{
    int j, k;

    for (j = 0; j < nj; j++) {
        for (k = 0; k < 3; k++) {
            vj[j][k] += dt * a[j][k];
        }
    }
}

void
push_position(double (*xj)[3], double (*vj)[3], double (*a)[3],
	      double dt, int nj)
{
    int j, k;

    for (j = 0; j < nj; j++) {
        for (k = 0; k < 3; k++) {
            xj[j][k] += dt * vj[j][k];
        }
    }
}

void
energy(double *mj, double (*vj)[3], double *p, int nj, double *ke, double *pe)
{
    int i, k;
     
    *pe = 0;
    *ke = 0;
    for (i = 0; i < nj; i++) {
        *pe += mj[i] * p[i];
        for (k = 0; k < 3; k++) {
            *ke += 0.5 * mj[i] * vj[i][k] * vj[i][k];
        }
    }
    *pe /= 2.0;
}

#if 0

/* CPU time */
void
get_cputime(double *lap, double *split)
{
    struct rusage x;
    double sec,microsec;

    getrusage(RUSAGE_SELF,&x);
    sec = x.ru_utime.tv_sec + x.ru_stime.tv_sec ;
    microsec = x.ru_utime.tv_usec + x.ru_stime.tv_usec ;

    *lap = sec + microsec / 1000000.0 - *split;
    *split = sec + microsec / 1000000.0;
}

#else

/* elapsed time in real world */
void
get_cputime(double *lap, double *split)
{
    struct timeval x;
    double sec,microsec;

    gettimeofday(&x, NULL);
    sec = x.tv_sec;
    microsec = x.tv_usec;

    *lap = sec + microsec / 1000000.0 - *split;
    *split = sec + microsec / 1000000.0;
}

#endif
