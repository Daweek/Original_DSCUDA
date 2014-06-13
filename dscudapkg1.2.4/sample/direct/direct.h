#ifndef DIRECT_H
#define DIRECT_H

#define NMAX (1024*1024*8)
#define NDEVMAX 1024

void readnbody(int *nj, double *mj, double (*xj)[3], double (*vj)[3], char *fname);
void writenbody(int nj, double *mj, double (*xj)[3], double (*vj)[3], char *fname);
void push_velocity(double (*vj)[3], double (*a)[3], double dt, int nj);
void push_position(double (*xj)[3], double (*vj)[3], double (*a)[3], double dt, int nj);
void energy(double *mj, double (*vj)[3], double *p, int nj, double *ke, double *pe);
void get_cputime(double *lap, double *split);
void plot_star(double x[NMAX][3], int n, double time, double ratio, double m[NMAX], double initm);
void create_cold_homosphere(int n, double *mj, double (*xj)[3], double (*vj)[3]);
void create_plummer(int n, double *mj, double (*xj)[3], double (*vj)[3]);

#endif /* DIRECT_H */
