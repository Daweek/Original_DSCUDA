/*
 * create initial distribution of particles in NEMO stoa format.
 * usage: mkdist <dist_type>
 *   dist_type:
 *     0 -- homogeneous sphere
 *     1 -- Plummer
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "direct.h"

static double *Mass;
static double (*Pos)[3];
static double (*Vel)[3];

static void alloc_particle_buf(int n);

int
main(int argc, char **argv)
{
    if (argc < 4) {
        fprintf(stderr, "create initial distribution of particles in NEMO stoa format.\n");
        fprintf(stderr, "  usage: %s <num_particle> <dist_type> <file_name>\n", argv[0]);
        fprintf(stderr, "    dist_type 0 -- homogeneous sphere\n");
        fprintf(stderr, "              1 -- Plummer model\n");
        exit(1);
    }

    int n = atoi(argv[1]);
    if (n < 1) {
        fprintf(stderr, "too small n (%d). abort\n", n);
        exit(1);
    }

#if 0 // no size-dependent part resides in this code.
    if (n > NMAX) {
        fprintf(stderr, "too large n (%d). abort\n", n);
        exit(1);
    }
#endif

    char fname[1024];
    sprintf(fname, "%s", argv[3]);
    fprintf(stderr, "n: %d  output file: %s\n", n, fname);
    alloc_particle_buf(n);

    int dist_type = atoi(argv[2]);
    switch (dist_type) {
      case 0:
        fprintf(stderr, "distribution: homosphere\n");
        create_cold_homosphere(n, Mass, Pos, Vel);
        break;

      case 1:
        fprintf(stderr, "distribution: Plummer\n");
        create_plummer(n, Mass, Pos, Vel);
        break;

      default:
        fprintf(stderr, "dist_type %d not defined. abort.\n", dist_type);
        exit(1);
        break;
    }

    writenbody(n, Mass, Pos, Vel, fname);

    exit(0);
}


static void
alloc_particle_buf(int n)
{
    Mass = (double *)malloc(sizeof(double)*n);
    if (!Mass) {
	perror("alloc_particle_buf");
	exit(1);
    }
    Pos = (double (*)[3])malloc(sizeof(double)*3*n);
    Vel = (double (*)[3])malloc(sizeof(double)*3*n);
    if (!Pos || !Vel) {
	perror("alloc_particle_buf");
	exit(1);
    }
}
