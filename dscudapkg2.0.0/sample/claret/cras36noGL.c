/*
        Visualized MD simulation claret for water-NaCl system

        s0 : NaCl nB (B*B*B*8 = number of NaCl)
        s1 : water fcc nB (B*B*B*4 = number of water)
        s2 : water ice nB (B*B*B*8 = number of water)
        s3 : water diamond nB (B*B*B*8 = number of water)
        s4 : water diamond + NaCl nnA nB (A = number of NaCl)
        ex. cras11 s4 n2 nn10   water44 NaCl20
        s5 : NaCl + water nwA nB (A = number of water)
        ex. cras11 s5 n2 nw5 water5 NaCl59
        s6 : NaCl + water nwA nB (A = number of water of layer)
        ex. cras11 s6 n3 nw1 water152 NaCl64
*/
#define VER 0.35

#define STEREO 0

//#define VTGRAPE // use Virtualized GRAPE library

/*#define SOCK_ON*/
#define GL_ON
#define LAP_TIME

#define C_MASS
/*#define TELOP*/
#define SUBWIN
#define CROSS

#define INFO
/*#define SWAP_ENDIAN*/

#if defined(MDGRAPE3) || defined(VTGRAPE)
#define MDM 2      /* 0:host 2:m2 */
#else
#define MDM 0      /* 0:host 2:m2 */
#endif
#define SPC 0
#define ST2 0
#define TIP5P 1
#define SYS 0 /* 0:NaCl 1:water(fcc) 2:water(ice) 3:water(ice2) 4:NaCl-water */

#define S_NUM_MAX 10*10*10*8
#define W_NUM_MAX 10*10*10*8

#define ZERO_P 1
#define V_SCALE 0
#define T_CONST 1
#define P_CONST 0
#define KNUM 5                    /* number of particle type */
#define VMAX 462 /*1535*/        /* max value of wave nubmer vector */
#define EFT 12000
#define my_min(x,y) ((x)<(y) ? (x):(y))
#define my_max(x,y) ((x)>(y) ? (x):(y))

#if defined(_WIN32) && !defined(__CYGWIN__)
#define M_PI 3.14159265
#endif
#define PI M_PI              /* pi */
#define PIT M_PI*2.0         /* 2 * pi */
#define PI2 M_PI*M_PI        /* pi*pi */
#define IPI M_1_PI           /* 1/pi */
#define ISPI M_2_SQRTPI*0.5  /* 1 / sqrt(pi) */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifdef GL_ON
#include <GL/glut.h>
#endif
#if MDM == 2
#ifdef MDGRAPE3
#include "mdgrape3.h"
#elif defined(VTGRAPE)
#else
#include <m2_unit.h>
#endif
#endif
#ifdef SOCK_ON
#include "sockhelp.h"
#include <unistd.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <ctype.h>
#endif
#ifdef LAP_TIME
#if defined(_WIN32) && !defined(__CYGWIN__)
#include <windows.h>
#include <mmsystem.h>
#else
#include <sys/time.h>
#endif
struct timeval time_v;
double md_time,md_time0;
double disp_time,disp_time0;
double sock_time,sock_time0;
#endif

#if MDM == 2
int grape_flg = 1;
#else
int grape_flg = 0;
#endif

int sc_flg = 0;    /* 0:non  1:server 2:client */
double m_matrix[16];
double i_matrix[16];

double trans[3] = {0.0, 0.0, 0.0};
double eye_len;

#ifdef INFO
double trans0[3];
double matrix0[16];
#endif

int auto_flg = 0;

#if defined(VTGRAPE)
int bond_flg = 0;
#else
int bond_flg = 1;
#endif

int temp_unit_type;
char temp_unit[2][5];

#ifdef GL_ON
#define CIRCLE 10
#define TEXTURE 0

int save_flg = 0;
int kabe_flg = 1;
int ini_flg = 1;

GLuint base;

//double eye_width = 0.4;
double eye_width = 0.8;
int eye_pos = 0;

int mouse_l = 0;
int mouse_m = 0;
int mouse_r = 0;

double angle[3] = {0.0, 0.0, 0.0};
/*
GLfloat red[]   = { 0.8, 0.25, 0.25, 1.0 };
GLfloat green[] = { 0.25, 0.8, 0.25, 1.0 };
*/
GLfloat red[]   = { 1.0, 0.0, 0.0, 1.0 };
GLfloat green[] = { 0.0, 1.0, 0.0, 1.0 };
GLfloat blue[]  = { 0.0, 0.0, 1.0, 1.0 };
GLfloat black[]  = { 1.0, 1.0, 1.0, 0.0 };
GLfloat white[]  = { 5.0, 5.0, 5.0, 1.0 };
GLfloat color_table[10][4];
GLfloat moji_c[2][4]  = { 0.8, 0.8, 0.8, 1.0,
                          0.0, 0.0, 0.0, 1.0 };

int mpos[2];

GLfloat bond_color[] = { 0.12, 0.12, 0.35, 1.0 };

double clear_color = 0.0;
double radius = 0.5;
int ditail = 15;

double circle_cd[CIRCLE][3];
GLfloat p_color[1][4];

double r_table[5];
int drow_flg[5] = {1,1,1,1,1};

int clip_flg = 0;
double clip[6][4];

#endif

/* for MD */

int sys_num = SYS;

int run_flg = 1;
int c_flg = 0;
int c_num = 0;
int velp_flg = 0;
double start_vl = -1;
double t_cd[3];
int w_add,s_add;
#define C_STEP 100

#ifdef LAP_TIME
int vflg = 3;
#else
int vflg = 1;
#endif
int kflg = 0;
int tflg = 0;

char k_file[50];
FILE *fp;

#if defined(MDGRAPE3) || defined(VTGRAPE)
int md_step = 10;
#else
int md_step = 1;
#endif
int md_stepf = 0;
int m_clock = 0;
int b_clock = 1;
int timemx = -1;

double avo  = 6.0221367e+23;    /* avogdro's number (mol^-1) */
double kb   = 8.617080363e-5;   /* Boltzmann's number (eV K^-1) */
double e    = 1.60217733e-19;   /* unit charge */

double delt = .5e-15;          /* dt sec */
//double delt = 0.125e-15;          /* dt sec */
double sigma = 1.0e-10;         /* unit of length (m) */
double mass  = 3.8175e-26;      /* unit of mass (Kg) */
double epsv  = 14.39;           /* unit of energy (eV) */
double epsj;

double a_massi[KNUM];
double a_mass[4] = {
  22.989768,   /* Atomic weight of Na */
  35.4527,     /* Atomic weight of Cl */
  15.9994,     /* Atomic weight of O */
  1.00794};    /* Atomic weight of H */

double bond[3] = {.9572, 0.15}; /* distance of O-C and O-M */
double hoh_deg = 104.52;

double m_cdx[4];
double m_cdy[4];
double m_cdz[4];
double moi[3];                  /* moment of inertia */

double temp  = 293;             /* temperature (K) */
double nden = -1;               /* density \AA^-3 */
double pres;
double ini_temp;

double  *cd;         /* position */
double  *vl;         /* velocity */
double  *fc;         /* force */

double  *fcc;

double *iphi;

double *ang;             /* angle */
double *agv;             /* angular velocity */
double *agvp;            /* angular velocity */
double *angh;            /* angle */
double *agvh;            /* angular velocity */
double *agvph;           /* angular velocity */
double *trq;             /* trque */

int *w_index;
int *w_rindex;
int *w_info;
int w_site;
int w_num,w_num3;
int s_num,s_num3;
int ws_num,ws_num3;

long *nig,*nli;
int *nig_data,*nig_num;

int *atype;          /* particle type */
                     /* 0:Na 1:Cl 2:O 3:H1 4:H2 5:M 6:L1 7:L2 8:C */
int atype_mat[20];
int atype_num[KNUM+4];  /* particle number of each type */

double tmrdp,jrdp;
double crdp,vclrdp;
double erdp;

double side0;
double side[3],sideh[3],iside[3];
double side_s[3],side_e[3];
double h,hsq,hsq2;
double tscale,sc;
double mtemp;
double rtemp;
double ekin,ekin1,ekin2;
double r,rd,rr,inr;
double vir;

double mpres,rpres;
double vol;
double lp=0;
double pist = 0.001;

double xs = 1.0;
double lq = .1;

double center_mass;

int np = 2;
int npx,npy,npz;
int n1;
int n2;
int n3;

int nn = 0;
int nw = 0;

double pb;
double pc[2][2],pd[2][2],ipotro[2][2];
double pol[2][2];
double sigm[2][2];

/* local */

double neighbor_radius = 3.1;
double min_angle = 15.0;
double max_angle = 75.0;

char keiname[256];
double z[KNUM+4],zz[KNUM+4][KNUM+4];
double wpa,wpc;
double as_s[KNUM][KNUM];
double as_e[KNUM][KNUM];
double as_a[KNUM][KNUM];
double as_c[KNUM][KNUM];
int vmax;
double oalpha = 6, alpha , alpha2, ial2si2;
float *erfct;
int *vecn[VMAX];
int knum=KNUM;
#if MDM != 0
  double gscale[(KNUM+4)*(KNUM+4)];
  double rscale[(KNUM+4)*(KNUM+4)];
  double gscale2[(KNUM+4)*(KNUM+4)];
  double rscale2[(KNUM+4)*(KNUM+4)];

  double charge[(KNUM+4)*(KNUM+4)];
  double roffset[(KNUM+4)*(KNUM+4)];

  double cellsize[3];
  double vecr;
#endif
#if MDM == 2
#ifndef VTGRAPE
  M2_UNIT *mu;
  M2_CELL cells[2];
#endif
  double side_min,side_max;
  char f_table_name[50];
  char p_table_name[50];
#endif
double phir_corr;
double phi[3],phir;
int pcun = 1;

#ifdef GL_ON
#define X_PIXEL 256
#define Y_PIXEL 256
static GLubyte teximage[X_PIXEL][Y_PIXEL][4];
static GLubyte teximage128[128][128][4];
#ifdef GL_VERSION_1_1
static GLuint sp_tex[KNUM+2];
static GLuint kabe_tex[2];
#endif

#define X_PIX_SIZE 1024
#define Y_PIX_SIZE 786

FILE *fps;
int file_num = 0;
GLubyte *pix;
struct BITMAPFILEHEADER {
    char                bfType[2];
    unsigned long       bfSize;
    unsigned short      bfReserved1;
    unsigned short      bfReserved2;
    unsigned long       bfOffBits;
} bmp_header;

struct BITMAPINFOHEADER {
    unsigned long       biSize;
    long                biWidth;
    long                biHeight;
    unsigned short      biPlanes;
    unsigned short      biBitCount;
    unsigned long       biCompression;
    unsigned long       biSizeImage;
    long                biXPixPerMeter;
    long                biYPixPerMeter;
    unsigned long       biClrUsed;
    unsigned long       biClrImporant;
} bmp_info;
#endif

#define TIMETABLE_MAX 10000
typedef struct{
  int mouse[3];
  double move[3];
  double rot[3];
  char command;
  double temp;
  double matrix[16];
} TIMETABLE;

TIMETABLE *tt;

#if defined(SUBWIN) && defined(GL_ON)
#define DATA_NUM 100
static int temp_data[DATA_NUM];
int temp_max = 0,temp_ymax = 10;
double sub_x,sub_y,sub_off;
int p_count = 0;
GLfloat line[4][4]   = {{ 1.0, 1.0, 0.0, 1.0 },
			{ 0.0, 1.0, 1.0, 1.0 },
			{ 1.0, 0.0, 1.0, 1.0 },
			{ 1.0, 1.0, 1.0, 1.0 }};
GLfloat waku[]   = { .7, .7, .7, 1.0 };
#endif

void make_time_table(char* file_name);
void keep_mem(int num, int num_w);
void init_MD(void);
void set_cd(int ini_m2);
void md_run(void);
void sock_md_run(void);
void potpar5(int xp,int xp2,int xm,int xm2, char keiname[]);
double nden_set(double tmp);
void velset6(double tref,double dh,double tscale,int knum,int num);
void ice_set(double *side);
void ice_set2(double* side);
void vecset();
void fccset2(int lnp,double lside,double cod[]);
double mass_den3(int xp, int xp2, int xm, int xm2, double comp, double temp);
void fccset_w(double* side);
int strsrc2(char str[],char key[], double *d);

#ifdef SOCK_ON

#ifdef SWAP_ENDIAN
#define SIZE_OF_INT 4
#define SIZE_OF_DOUBLE 8
static int *send_buf_int;
static double *send_buf_double;
#endif

static int listensock = -1; /* So that we can close sockets on ctrl-c */
static int connectsock = -1;
static int s_sock;

/* This waits for all children, so that they don't become zombies. */
void sig_chld(signal_type)
int signal_type;
{
  int pid;
  int status;

  while ( (pid = wait3(&status, WNOHANG, NULL)) > 0);
}
void sock_send_char(int sockfd, char* buf, int size)
{
  sock_write(sockfd,buf,sizeof(char)*size);
}
void sock_send_int(int sockfd, int* buf, int size)
{
#ifdef SWAP_ENDIAN
  int i,j;
  char c_buf[SIZE_OF_INT];

  for(i = 0; i < size; i++){
    for(j = 0; j < SIZE_OF_INT; j++){
      c_buf[SIZE_OF_INT-1-j] = ((char*)&buf[i])[j];
      send_buf_int[i] = *(int*)c_buf;
    }
  }
  sock_write(sockfd,send_buf_int,sizeof(int)*size);
#else
  sock_write(sockfd,buf,sizeof(int)*size);
#endif
}
void sock_send_double(int sockfd, double* buf, int size)
{
#ifdef SWAP_ENDIAN
  int i,j;
  char c_buf[SIZE_OF_DOUBLE];

  for(i = 0; i < size; i++){
    for(j = 0; j < SIZE_OF_DOUBLE; j++){
      c_buf[SIZE_OF_DOUBLE-1-j] = ((char*)&buf[i])[j];
      send_buf_double[i] = *(double*)c_buf;
    }
  }
  sock_write(sockfd,send_buf_double,sizeof(double)*size);
#else
  sock_write(sockfd,buf,sizeof(double)*size);
#endif
}
void sock_recv_char(int sockfd, char* buf, int size)
{
  sock_read(sockfd,buf,sizeof(char)*size);
}
void sock_recv_int(int sockfd, int* buf, int size)
{
#ifdef SWAP_ENDIAN
  int i,j;
  char c_buf[SIZE_OF_INT];

  sock_read(sockfd,send_buf_int,sizeof(int)*size);

  for(i = 0; i < size; i++){
    for(j = 0; j < SIZE_OF_INT; j++){
      c_buf[SIZE_OF_INT-1-j] = ((char*)&send_buf_int[i])[j];
      buf[i] = *(int*)c_buf;
    }
  }
#else
  sock_read(sockfd,buf,sizeof(int)*size);
#endif
}
void sock_recv_double(int sockfd, double* buf, int size)
{
#ifdef SWAP_ENDIAN
  int i,j;
  char c_buf[SIZE_OF_DOUBLE];

  sock_read(sockfd,send_buf_double,sizeof(double)*size);

  for(i = 0; i < size; i++){
    for(j = 0; j < SIZE_OF_DOUBLE; j++){
      c_buf[SIZE_OF_DOUBLE-1-j] = ((char*)&send_buf_double[i])[j];
      buf[i] = *(double*)c_buf;
    }
  }

#else
  sock_read(sockfd,buf,sizeof(double)*size);
#endif
}
#endif
#ifdef GL_ON
void mat_inv(double a[4][4])
{
  int i,j,k;
  double t, u, det;
  int n = 3;

  det = 1;
  for(k = 0; k < n; k++){
    t = a[k][k]; det *= t;
    for(i = 0; i < n; i++) a[k][i] /= t;
    a[k][k] = 1 / t;
    for(j = 0; j < n; j++)
      if(j != k){
        u = a[j][k];
        for(i = 0; i < n; i++)
          if(i != k) a[j][i] -= a[k][i] * u;
          else       a[j][i] = -u/t;
      }
  }
}
#if TEXTURE == 1
void TexCirSphere(double r, int num)
{
  int i;

  glEnable(GL_TEXTURE_2D);
  glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL);
#ifdef GL_VERSION_1_1
   glBindTexture(GL_TEXTURE_2D, sp_tex[num]);
#endif

  glBegin(GL_POLYGON);
  glNormal3d(1,0,0);
  for(i = 0; i < CIRCLE; i++){
    glTexCoord2f(circle_cd[i][1]*.5+.5, circle_cd[i][2]*.5+.5);
    glVertex3d(circle_cd[i][0],r*circle_cd[i][1],r*circle_cd[i][2]);
  }
  glEnd();
  glDisable(GL_TEXTURE_2D);
}
void TexSphere(double r, int num)
{

  glEnable(GL_TEXTURE_2D);
  glEnable(GL_ALPHA_TEST);
  /*  glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);*/
  glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
#ifdef GL_VERSION_1_1
   glBindTexture(GL_TEXTURE_2D, sp_tex[num]);
#endif

  glBegin(GL_QUADS);
  glNormal3f(1.0, 0.0, 0.0);
  glTexCoord2f(0.0, 0.0); glVertex3f(0.0, -r, -r);
  glTexCoord2f(0.0, 1.0); glVertex3f(0.0,  r, -r);
  glTexCoord2f(1.0, 1.0); glVertex3f(0.0,  r,  r);
  glTexCoord2f(1.0, 0.0); glVertex3f(0.0, -r,  r);

  glEnd();
  glDisable(GL_ALPHA_TEST);
  glDisable(GL_TEXTURE_2D);
}
#endif
void readtexture3(int ci)
{
  int i,j,k;
  char buf[256];
  FILE *fp;

  if((fp = fopen("SP_W.PPM","r")) == NULL){
    printf("texture file open error SP_W.PPM\n");
    exit(1);
  }
  fgets(buf,256,fp);
  fgets(buf,256,fp);
  fgets(buf,256,fp);

  for(i = 0; i < X_PIXEL; i++){
    for(j = 0; j < Y_PIXEL; j++){
      /*
      for(k = 0; k < 3; k++){
        teximage[j][Y_PIXEL-1-i][k==2 ? 0:k+1] = fgetc(fp);
      }
      */
      teximage[j][Y_PIXEL-1-i][0] = fgetc(fp);
      teximage[j][Y_PIXEL-1-i][1] = fgetc(fp);
      teximage[j][Y_PIXEL-1-i][2] = fgetc(fp);
      /*
      teximage[j][Y_PIXEL-1-i][0] = fgetc(fp)*color_table[ci][0];
      teximage[j][Y_PIXEL-1-i][1] = fgetc(fp)*color_table[ci][1];
      teximage[j][Y_PIXEL-1-i][2] = fgetc(fp)*color_table[ci][2];
      */
      if(teximage[j][Y_PIXEL-1-i][0] == 0 &&
         teximage[j][Y_PIXEL-1-i][1] == 0 &&
         teximage[j][Y_PIXEL-1-i][2] == 0)
        teximage[j][Y_PIXEL-1-i][3] = (GLubyte)0;
      else
        teximage[j][Y_PIXEL-1-i][3] = (GLubyte)255;
    }
  }
  fclose(fp);

}
void readtexture(char* file_name, int lx, int ly)
{
  int i,j,k;
  char buf[256];
  FILE *fp;

  if((fp = fopen(file_name,"rb")) == NULL){
    printf("texture file open error %s\n",file_name);
    exit(1);
  }
  fgets(buf,256,fp);
  fgets(buf,256,fp);
  fgets(buf,256,fp);
  fgets(buf,256,fp);

  for(i = 0; i < lx; i++){
    for(j = 0; j < ly; j++){
      teximage128[j][ly-1-i][0] = fgetc(fp);
      teximage128[j][ly-1-i][1] = fgetc(fp);
      teximage128[j][ly-1-i][2] = fgetc(fp);

      if(teximage128[j][ly-1-i][0] == 0 &&
         teximage128[j][ly-1-i][1] == 0 &&
         teximage128[j][ly-1-i][2] == 0)
        teximage128[j][ly-1-i][3] = (GLubyte)0;
      else
        teximage128[j][ly-1-i][3] = (GLubyte)255;
    }
  }
  fclose(fp);
}
void init(void)
{
  int i,j;
  int i0;
  int a_num = 1;

  GLfloat mat_specular[] = {0.2, 0.2, 0.2, 1.0};
  GLfloat mat_ambient[] = {0.1, 0.1, 0.1, 1.0};
  GLfloat mat_shininess[] = {64.0};
  GLfloat light_position[] = {1.0, 1.1, 1.2, 0.0};

  glShadeModel(GL_SMOOTH);
/*  glShadeModel(GL_FLAT);*/
  glLightfv(GL_LIGHT0, GL_SPECULAR, mat_specular);
  glLightfv(GL_LIGHT0, GL_SHININESS, mat_shininess);
  glLightfv(GL_LIGHT0, GL_AMBIENT, mat_ambient);
  glLightfv(GL_LIGHT0, GL_POSITION, light_position);

  glMatrixMode(GL_MODELVIEW);
  glGetDoublev(GL_MODELVIEW_MATRIX,m_matrix);
  glGetDoublev(GL_MODELVIEW_MATRIX,i_matrix);

  base = glGenLists(128);
  for(i = 0; i < 128; i++){
    glNewList(base+i, GL_COMPILE);
    glutBitmapCharacter(GLUT_BITMAP_TIMES_ROMAN_24, i);
    glEndList();
  }
  glListBase(base);

#if TEXTURE == 1
  glAlphaFunc(GL_GREATER,0.5);
#endif

  color_table[0][0] = 0.7;
  color_table[0][1] = 0.38;
  color_table[0][2] = 0.38;
  color_table[0][3] = 1;

  color_table[1][0] = 0.38;
  color_table[1][1] = 0.55;
  color_table[1][2] = 0.38;
  color_table[1][3] = 1;

  for(i = 0; i < 3; i++){
    color_table[0][i] /= 2.0;
    color_table[1][i] /= 2.0;
  }
  /*
  printf("%f %f %f\n",color_table[0][0],color_table[0][1],color_table[0][2]);
  printf("%f %f %f\n",color_table[1][0],color_table[1][1],color_table[1][2]);
  */
  color_table[2][0] = 1;
  color_table[2][1] = .4;
  color_table[2][2] = 1;
  color_table[2][3] = 1;

  color_table[3][0] = 0;
  color_table[3][1] = 0.8;
  color_table[3][2] = 1;
  color_table[3][3] = 1;

  color_table[4][0] = 1;
  color_table[4][1] = 1;
  color_table[4][2] = 1;
  color_table[4][3] = 1;

  r_table[0] = 2.443/2;
  r_table[1] = 3.487/2;
  r_table[2] = 3.156/2;
  r_table[3] = .7;
  r_table[4] = .7;

#if TEXTURE == 1

  for(i = 0; i < CIRCLE; i++){
    circle_cd[i][0] = 0;
    circle_cd[i][1] = cos(2.*PI/CIRCLE*i);
    circle_cd[i][2] = sin(2.*PI/CIRCLE*i);
  }

#ifdef GL_VERSION_1_1
  glGenTextures(knum+2, sp_tex);
#endif
  for(i = 0; i < knum+2; i++){
    readtexture3(i);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

#ifdef GL_VERSION_1_1
    glBindTexture(GL_TEXTURE_2D, sp_tex[i]);
#endif
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
#ifdef GL_VERSION_1_1
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, Y_PIXEL, X_PIXEL, 
                 0, GL_RGBA, GL_UNSIGNED_BYTE, teximage);
#else
    glTexImage2D(GL_TEXTURE_2D, 0, 4, Y_PIXEL, X_PIXEL, 
                 0, GL_RGBA, GL_UNSIGNED_BYTE, teximage);
#endif
  }

#ifdef GL_VERSION_1_1
  glGenTextures(2, kabe_tex);
#endif
  for(i = 0; i < 1; i++){
    readtexture("riken.ppm",128,128);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

#ifdef GL_VERSION_1_1
    glBindTexture(GL_TEXTURE_2D, kabe_tex[i]);
#endif
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
#ifdef GL_VERSION_1_1
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 128, 128, 
		 0, GL_RGBA, GL_UNSIGNED_BYTE, teximage128);
#else
    glTexImage2D(GL_TEXTURE_2D, 0, 4, 128, 128, 
		 0, GL_RGBA, GL_UNSIGNED_BYTE, teximage128);
#endif
  }

#endif
  if(kflg == 1){
    if( ( fp = fopen( k_file, "w" ) ) == NULL ){
      printf("DATA file open error\n");
      exit( 1 );
    }
  }
#ifdef INFO
  for(i = 0; i < 3; i++)
    trans0[i] = 0;
  for(i = 0; i < 4; i++){
    for(j = 0; j < 4; j++){
      if(i == j){
	matrix0[i*4+j] = 1;
      } else {
	matrix0[i*4+j] = 0;
      }
    }
  }
#endif
}
void hako(int flg)
{
  double d0;
  int i;
  static GLfloat kabe[]  = { 0.0, 0.0, 0.4, 1.0 };
  static GLfloat kabe2[]  = { 0.0, 0.0, 0.8, 1.0 };
  double side_s[3],side_e[3];

  for(i = 0; i < 3; i++){
    side_s[i] = -sideh[i];
    side_e[i] = side[i]-sideh[i];
  }

  if(flg == 0){
    for(i = 0; i < 3; i++){
      side_s[i] += -radius;
      side_e[i] +=  radius;
    }
    /*    glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, kabe);*/
    glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, kabe);
    glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, kabe2);

    glBegin(GL_POLYGON);
    glNormal3d(1,0,0);
    glVertex3d(side_s[0],side_s[1],side_s[2]);
    glVertex3d(side_s[0],side_e[1],side_s[2]);
    glVertex3d(side_s[0],side_e[1],side_e[2]);
    glVertex3d(side_s[0],side_s[1],side_e[2]);
    glEnd();
    glBegin(GL_POLYGON);
    glNormal3d(-1,0,0);
    glVertex3d(side_e[0],side_s[1],side_s[2]);
    glVertex3d(side_e[0],side_s[1],side_e[2]);
    glVertex3d(side_e[0],side_e[1],side_e[2]);
    glVertex3d(side_e[0],side_e[1],side_s[2]);
    glEnd();

    glBegin(GL_POLYGON);
    glNormal3d(0,-1,0);
    glVertex3d(side_s[0],side_e[1],side_s[2]);
    glVertex3d(side_e[0],side_e[1],side_s[2]);
    glVertex3d(side_e[0],side_e[1],side_e[2]);
    glVertex3d(side_s[0],side_e[1],side_e[2]);
    glEnd();
    glBegin(GL_POLYGON);
    glNormal3d(0,1,0);
    glVertex3d(side_s[0],side_s[1],side_s[2]);
    glVertex3d(side_s[0],side_s[1],side_e[2]);
    glVertex3d(side_e[0],side_s[1],side_e[2]);
    glVertex3d(side_e[0],side_s[1],side_s[2]);
    glEnd();

    glBegin(GL_POLYGON);
    glNormal3d(0,0,1);
    glVertex3d(side_s[0],side_s[1],side_s[2]);
    glVertex3d(side_e[0],side_s[1],side_s[2]);
    glVertex3d(side_e[0],side_e[1],side_s[2]);
    glVertex3d(side_s[0],side_e[1],side_s[2]);
    glEnd();

    glBegin(GL_POLYGON);
    glNormal3d(0,0,-1);
    glVertex3d(side_s[0],side_s[1],side_e[2]);
    glVertex3d(side_s[0],side_e[1],side_e[2]);
    glVertex3d(side_e[0],side_e[1],side_e[2]);
    glVertex3d(side_e[0],side_s[1],side_e[2]);
    glEnd();
  }
  if(flg == 1){
    glColor3d(1.0,1.0,1.0);
    glBegin(GL_LINE_LOOP);
    glVertex3d(side_s[0],side_s[1],side_s[2]);
    glVertex3d(side_s[0],side_e[1],side_s[2]);
    glVertex3d(side_s[0],side_e[1],side_e[2]);
    glVertex3d(side_s[0],side_s[1],side_e[2]);
    glEnd();
    glBegin(GL_LINE_LOOP);
    glVertex3d(side_e[0],side_s[1],side_s[2]);
    glVertex3d(side_e[0],side_e[1],side_s[2]);
    glVertex3d(side_e[0],side_e[1],side_e[2]);
    glVertex3d(side_e[0],side_s[1],side_e[2]);
    glEnd();
    glBegin(GL_LINES);
    glVertex3d(side_e[0],side_s[1],side_s[2]);
    glVertex3d(side_s[0],side_s[1],side_s[2]);
    glVertex3d(side_e[0],side_e[1],side_s[2]);
    glVertex3d(side_s[0],side_e[1],side_s[2]);
    glVertex3d(side_e[0],side_e[1],side_e[2]);
    glVertex3d(side_s[0],side_e[1],side_e[2]);
    glVertex3d(side_e[0],side_s[1],side_e[2]);
    glVertex3d(side_s[0],side_s[1],side_e[2]);
    glEnd();
  }
}
void bou2(double x0, double y0, double z0, double x1, double y1, double z1
   ,double wid,int dit)
{
  double d0,d2;
  GLUquadricObj *qobj;

  qobj = gluNewQuadric();
  gluQuadricDrawStyle(qobj,GLU_FILL);
  gluQuadricNormals(qobj,GLU_SMOOTH);

  d0 = sqrt((x0-x1)*(x0-x1) + (y0-y1)*(y0-y1) + (z0-z1)*(z0-z1));

  glPushMatrix();
  glTranslated(x0,y0,z0);
  d2 =-acos((z1-z0)/d0)/M_PI*180;
  /*
  printf("%f %f %f  %f %f %f  %f %f\n",x0,y0,z0,x1,y1,z1,d0,d2);
  */
  if(y0 == y1 && x0 == x1)
    glRotatef(d2,1,0,0);
  else
    glRotatef(d2,(y1-y0),-(x1-x0),0);
  gluCylinder(qobj,wid,wid,d0,dit,1);
  glPopMatrix();

  glPushMatrix();
  glTranslated(x1,y1,z1);
  glPopMatrix();

  gluDeleteQuadric(qobj);
}
void bond_drow(int s_num, int e_num, int c_num, double wid,int dit)
{
  glMaterialfv(GL_FRONT,GL_AMBIENT_AND_DIFFUSE,bond_color);
  bou2(cd[s_num*3]-sideh[0],cd[s_num*3+1]-sideh[1],cd[s_num*3+2]-sideh[2],
       cd[e_num*3]-sideh[0],cd[e_num*3+1]-sideh[1],cd[e_num*3+2]-sideh[2]
       ,wid,dit);
}
void line_drow(int s_num, int e_num)
{
  int i;
  GLfloat color[4];
  /*  
  glMaterialfv(GL_FRONT, GL_DIFFUSE,bond_color);
  for(i = 0; i < 3; i++)
    color[i] = bond_color[i]*5;
  glMaterialfv(GL_FRONT, GL_AMBIENT,color);
  */
  glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE,bond_color);
  glLineWidth(1.0);
  glBegin(GL_LINES);
  glVertex3d(cd[s_num*3]-sideh[0],cd[s_num*3+1]-sideh[1],cd[s_num*3+2]-sideh[2]);
  glVertex3d(cd[e_num*3]-sideh[0],cd[e_num*3+1]-sideh[1],cd[e_num*3+2]-sideh[2]);
  glEnd();

}
void small_font(double px, double py, double pz, char *moji)
{
  int i;
  int len;
  double wid,adj;

  wid = 0.1;
  len = strlen(moji);
  glColor4fv(moji_c[(int)(clear_color+.5)]);
  for(i = 0;i < len; i++){
    if(moji[i] == '1') adj = 0.55;
    else if(moji[i] >= '2' && moji[i] <= '9') adj = 0.7;
    else if(moji[i] == '0') adj = 0.7;
    else if(moji[i] == 'B') adj = 0.7;
    else adj = 1.0;
    glRasterPos3d(px,py,pz);
    px += wid*adj;
    glutBitmapCharacter(GLUT_BITMAP_HELVETICA_10, moji[i]);
  }
}
void medium_font(double px, double py, double pz, char *moji)
{
  int i;
  int len;
  double wid,adj;

  wid = 0.1;
  len = strlen(moji);
  glColor4fv(moji_c[(int)(clear_color+.5)]);
  for(i = 0;i < len; i++){
    if(moji[i] == '1') adj = 0.55;
    else if(moji[i] >= '2' && moji[i] <= '9') adj = 0.7;
    else if(moji[i] == '0') adj = 0.7;
    else if(moji[i] == 'B') adj = 0.7;
    else adj = 1.0;
    glRasterPos3d(px,py,pz);
    px += wid*adj;
    glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, moji[i]);
  }
}
void single_display(int which)
{
  double d0,d1,d2,d3,d4,d5;
  double mag;
  int i,j;
  int i0;
  char str_buf[256];
  char str_buf2[256];
  GLfloat particle_color[4];
  GLfloat color[4];

  glClearColor(clear_color, clear_color, clear_color, 0.0);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  /*
  glColor3d(1.0, 0.0, 1.0);
  glBegin(GL_LINE_LOOP);
  {
    glVertex3d(-0.9, -0.9,0.0);
    glVertex3d(0.9, -0.9,0.0);
    glVertex3d(0.9, 0.9,0.0);
    glVertex3d(-0.9, 0.9,0.0);
  }
  glEnd();
  glFlush();
  */

  glEnable(GL_DEPTH_TEST);
  glEnable(GL_CULL_FACE);
  glEnable(GL_LIGHTING);
  glEnable(GL_LIGHT0);
  glCullFace(GL_BACK);

  glLoadIdentity();

  glPushMatrix();

  d3 = atan((eye_width*which)/eye_len);
  d1 = sin(d3)*eye_len;
  d0 = cos(d3)*eye_len;

  gluLookAt(d0, d1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0);

  glTranslated(trans[0], trans[1], trans[2]);

  glPushMatrix();
  glLoadIdentity();
  glRotatef( angle[0],1.0,0.0,0.0);
  glRotatef( angle[1],0.0,1.0,0.0);
  glRotatef( angle[2],0.0,0.0,1.0);
  glMultMatrixd(m_matrix);
  glGetDoublev(GL_MODELVIEW_MATRIX, m_matrix);
  glPopMatrix();

  for(i = 0; i < 16; i++)
    i_matrix[i] = m_matrix[i];
  mat_inv((double(*)[4])i_matrix);

  glMultMatrixd(m_matrix);

#ifdef CROSS

  if(kabe_flg == 1){
    d2 = (i_matrix[0]*(1.0)+i_matrix[4]*(1.0)+i_matrix[8]*(1.0));
    d3 = (i_matrix[1]*(1.0)+i_matrix[5]*(1.0)+i_matrix[9]*(1.0));
    d4 = (i_matrix[2]*(1.0)+i_matrix[6]*(1.0)+i_matrix[10]*(1.0));

    d0 = side0/2;
    d1 = 0.3;
    color[0] = d1; color[1] = d1; color[2] = d1; color[3] = 1.0;
    glMaterialfv(GL_FRONT,GL_AMBIENT_AND_DIFFUSE,color);
    glLineWidth(2.0);
    glBegin(GL_LINES);
    glNormal3d(d2,d3,d4);
    glVertex3d(0,0,-d0);
    glVertex3d(0,0, d0);
    glEnd();
    glBegin(GL_LINES);
    glNormal3d(d2,d3,d4);
    glVertex3d(0,-d0,0);
    glVertex3d(0, d0,0);
    glEnd();
    glBegin(GL_LINES);
    glNormal3d(d2,d3,d4);
    glVertex3d(-d0,0,0);
    glVertex3d( d0,0,0);
    glEnd();
  }
#endif

  if(clip_flg == 0){
    glDisable(GL_CLIP_PLANE0);
  } else if(clip_flg == 1){
    glClipPlane(GL_CLIP_PLANE0, clip[0]);
    glClipPlane(GL_CLIP_PLANE1, clip[1]);
    glEnable(GL_CLIP_PLANE0);
    glEnable(GL_CLIP_PLANE1);
  } else if(clip_flg == 2){
    glClipPlane(GL_CLIP_PLANE0, clip[2]);
    glClipPlane(GL_CLIP_PLANE1, clip[3]);
    glEnable(GL_CLIP_PLANE0);
    glEnable(GL_CLIP_PLANE1);
  } else if(clip_flg == 3){
    glClipPlane(GL_CLIP_PLANE0, clip[4]);
    glClipPlane(GL_CLIP_PLANE1, clip[5]);
    glEnable(GL_CLIP_PLANE0);
    glEnable(GL_CLIP_PLANE1);
  } else if(clip_flg == 4){
    glClipPlane(GL_CLIP_PLANE0, clip[0]);
    glClipPlane(GL_CLIP_PLANE1, clip[1]);
    glClipPlane(GL_CLIP_PLANE2, clip[2]);
    glClipPlane(GL_CLIP_PLANE3, clip[3]);
    glClipPlane(GL_CLIP_PLANE4, clip[4]);
    glClipPlane(GL_CLIP_PLANE5, clip[5]);
    glEnable(GL_CLIP_PLANE0);
    glEnable(GL_CLIP_PLANE1);
    glEnable(GL_CLIP_PLANE2);
    glEnable(GL_CLIP_PLANE3);
    glEnable(GL_CLIP_PLANE4);
    glEnable(GL_CLIP_PLANE5);
  }

  angle[0] = 0;
  if(mouse_l == 1 || mouse_m == 1 || mouse_r == 1){
    angle[1] = 0;
    angle[2] = 0;
  }
  if(ini_flg == 1){
    mouse_l = 0;
    ini_flg = 0;
  }
  /*
  if(kabe_flg == 1)
    hako(ZERO_P ? 0:1);
  */
  if(kabe_flg == 1)
    hako(1);

#if 1
#if defined(MDGRAPE3) || defined(VTGRAPE)
  if(bond_flg == 1 && grape_flg==0){
#else
  if(bond_flg == 1){
#endif
    d2 = (i_matrix[0]*(1.0)+i_matrix[4]*(1.0)+i_matrix[8]*(1.0));
    d3 = (i_matrix[1]*(1.0)+i_matrix[5]*(1.0)+i_matrix[9]*(1.0));
    d4 = (i_matrix[2]*(1.0)+i_matrix[6]*(1.0)+i_matrix[10]*(1.0));
    glNormal3d(d2,d3,d4);
    for(i = 0; i < n1; i++){
      for(j = 0; j < nig_num[i]; j++){
	/*	  bond_drow(i,nig_data[i][j],0, 0.03, 5);*/
	line_drow(i,nig_data[i*6+j]);
      }
    }
  }
#endif
  /*
  for(i = 0; i < n3; i += 3){
    printf("%d % f % f % f %.10f\n",i/3,vl[i],vl[i+1],vl[i+2]
	   ,vl[i]*vl[i]+vl[i+1]*vl[i+1]+vl[i+2]*vl[i+2]);
  }
  exit(0);
  */

  for(i = 0; i < n3; i += 3){
    if(drow_flg[atype_mat[atype[i/3]]] == 1){
      glPushMatrix();
      glTranslated(cd[i]-sideh[0], cd[i+1]-sideh[1], cd[i+2]-sideh[2]);

#if TEXTURE == 1
      glMultMatrixd(i_matrix);
      /*      glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, black);*/
      d0 = (vl[i]*vl[i]+vl[i+1]*vl[i+1]+vl[i+2]*vl[i+2])*500;
      particle_color[0] = color_table[atype_mat[atype[i/3]]][0]+d0;
      particle_color[1] = color_table[atype_mat[atype[i/3]]][1]+d0/3;
      particle_color[2] = color_table[atype_mat[atype[i/3]]][2]+d0/3;
      particle_color[3] = color_table[atype_mat[atype[i/3]]][3];
      glMaterialfv(GL_FRONT, GL_AMBIENT,particle_color);

      particle_color[0] = color_table[atype_mat[atype[i/3]]][0]+d0/4;
      particle_color[1] = color_table[atype_mat[atype[i/3]]][1]+d0/12;
      particle_color[2] = color_table[atype_mat[atype[i/3]]][2]+d0/12;
      particle_color[3] = color_table[atype_mat[atype[i/3]]][3];
      glMaterialfv(GL_FRONT, GL_DIFFUSE,particle_color);

      /*      TexSphere(radius*r_table[atype_mat[atype[i/3]]],atype_mat[atype[i/3]]);*/
      TexSphere(radius*r_table[atype_mat[atype[i/3]]],0);
#else
      if(atype[i/3] == 8)
        glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE,red);
      else{
	/*
        glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE,
                   red);
	*/
	d0 = (vl[i]*vl[i]+vl[i+1]*vl[i+1]+vl[i+2]*vl[i+2])*500;
	/*
	if(atype[i/3] == 0){
	  particle_color[0] = color_table[atype_mat[atype[i/3]]][0]+d0;
	  particle_color[1] = color_table[atype_mat[atype[i/3]]][1];
	  particle_color[2] = color_table[atype_mat[atype[i/3]]][2];
	  particle_color[3] = color_table[atype_mat[atype[i/3]]][3];
	} else {
	  particle_color[0] = color_table[atype_mat[atype[i/3]]][0]+d0;
	  particle_color[1] = color_table[atype_mat[atype[i/3]]][1]-d0/2;
	  particle_color[2] = color_table[atype_mat[atype[i/3]]][2];
	  particle_color[3] = color_table[atype_mat[atype[i/3]]][3];
	}
	*/
	particle_color[0] = color_table[atype_mat[atype[i/3]]][0]+d0;
	particle_color[1] = color_table[atype_mat[atype[i/3]]][1]+d0/3;
	particle_color[2] = color_table[atype_mat[atype[i/3]]][2]+d0/3;
	particle_color[3] = color_table[atype_mat[atype[i/3]]][3];
	glMaterialfv(GL_FRONT, GL_AMBIENT,particle_color);
	
        glMaterialfv(GL_FRONT, GL_AMBIENT,particle_color);

	particle_color[0] = color_table[atype_mat[atype[i/3]]][0]+d0/4;
	particle_color[1] = color_table[atype_mat[atype[i/3]]][1]+d0/12;
	particle_color[2] = color_table[atype_mat[atype[i/3]]][2]+d0/12;
	particle_color[3] = color_table[atype_mat[atype[i/3]]][3];
        glMaterialfv(GL_FRONT, GL_DIFFUSE,particle_color);
      }
      glutSolidSphere(radius*r_table[atype_mat[atype[i/3]]], ditail, ditail/2);
#endif
      glPopMatrix();
    }
  }
  if(c_flg != 0){
    for(i = n3; i < n3+c_num*3; i += 3){
      glPushMatrix();
      glTranslated(cd[i]-sideh[0], cd[i+1]-sideh[1], cd[i+2]-sideh[2]);
      glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE,
                   color_table[atype_mat[atype[i/3]]]);
      glutWireSphere(radius*r_table[atype_mat[atype[i/3]]]*c_flg/C_STEP
                     , ditail, ditail/2);
      glPopMatrix();
    }
    if(c_flg+md_step <= C_STEP) c_flg += md_step; else c_flg = C_STEP;
  }

  glPopMatrix();

  glDisable(GL_DEPTH_TEST);
  if(clip_flg != 0){
    glDisable(GL_CLIP_PLANE0);
    glDisable(GL_CLIP_PLANE1);
    glDisable(GL_CLIP_PLANE2);
    glDisable(GL_CLIP_PLANE3);
    glDisable(GL_CLIP_PLANE4);
    glDisable(GL_CLIP_PLANE5);
  }

#ifdef LAP_TIME
#if defined(_WIN32) && !defined(__CYGWIN__)
  disp_time0 = disp_time;
  disp_time = (double)timeGetTime()/1000.;
#elif defined(MAC)
  disp_time0 = disp_time;
  disp_time = (double)clock()/60.;
#else
  gettimeofday(&time_v,NULL);
  disp_time0 = disp_time;
  disp_time = (time_v.tv_sec + time_v.tv_usec / 1000000.0);
#endif
#endif

#ifdef TELOP
  d0 = -3.0;
  d1 = -2.9;
  d2 = -10;
  glEnable(GL_ALPHA_TEST);
  glEnable(GL_TEXTURE_2D);
  glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
#ifdef GL_VERSION_1_1
  glBindTexture(GL_TEXTURE_2D, kabe_tex[0]);
#endif
  d3 = 1.35;
  glBegin(GL_POLYGON);
  glNormal3d(1,0,0);
  glTexCoord2f(0.0, 0.0); glVertex3d(d0,d1,d2);
  glTexCoord2f(0.0, 1.0); glVertex3d(d0+d3,d1,d2);
  glTexCoord2f(1.0, 1.0); glVertex3d(d0+d3,d1+d3,d2);
  glTexCoord2f(1.0, 0.0); glVertex3d(d0,d1+d3,d2);
  glEnd();
  glDisable(GL_ALPHA_TEST);
  glDisable(GL_TEXTURE_2D);
#endif

  glDisable(GL_LIGHTING);

  if(vflg >= 1){

    d0 = -2.4;
    d1 = 2.2;
    d2 = -10;

    /*    sprintf(str_buf,"T=%.0fK N=%d (W%d N%d)",temp,n1,w_num,s_num);*/
    if(temp_unit_type == 1)
      sprintf(str_buf,"T=%.0fC N=%d",temp-273,n1);
    else
      sprintf(str_buf,"T=%.0fK N=%d",temp,n1);

    if(vflg >= 3 && auto_flg == 0){
      if(grape_flg == 1)
#ifdef MDGRAPE3
	strcat(str_buf,"  MDGRAPE3:ON");
#elif defined(VTGRAPE)
	strcat(str_buf,"  GPU:ON");
#else
	strcat(str_buf,"  MDGRAPE2:ON");
#endif
      else
#ifdef MDGRAPE3
	strcat(str_buf,"  MDGRAPE3:OFF");
#elif defined(VTGRAPE)
	strcat(str_buf,"  GPU:OFF");
#else
	strcat(str_buf,"  MDGRAPE2:OFF");
#endif
    }

    glColor4fv(moji_c[(int)(clear_color+.5)]);
    glRasterPos3d(d0, d1, d2);
    glCallLists(strlen(str_buf), GL_BYTE, str_buf);

    d1 -= .3;
    if(temp_unit_type == 1)
      sprintf(str_buf,"temp:%4.0fC time:%.3es"
	      ,mtemp*epsv/kb-273,delt*m_clock);
    else
      sprintf(str_buf,"temp:%4.0fK time:%.3es"
	      ,mtemp*epsv/kb,delt*m_clock);

		
		static int t_step = 0;
		printf("%d\t%f\n", t_step, mtemp*epsv/kb);
		t_step++;
		if (t_step > 1000)
			exit(0);


    /*
    if(save_flg == 0){
      sprintf(str_buf2," %d",b_clock);
      strcat(str_buf,str_buf2);
    }
    */
    /*
    glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE
                 ,moji_c[(int)(clear_color+.5)]);
    */
    glRasterPos3d(d0, d1, d2);
    glCallLists(strlen(str_buf), GL_BYTE, str_buf);
    /*
    d1 -= .3;
    sprintf(str_buf,"pressure:%.4ePa"
            ,mpres*epsj/(sigma*sigma*sigma));
    glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE
                 ,moji_c[(int)(clear_color+.5)]);
    glRasterPos3d(d0, d1, d2);
    glCallLists(strlen(str_buf), GL_BYTE, str_buf);
    */

    if(velp_flg > 0){
      d1 -= .3;
      d4 = 0;
      d5 = 1. / 3. /((double)(s_num+s_add + (w_num+w_add)*2) - 1);
      for(i = 0; i < c_num; i++)
        d4 += pow(start_vl,2)*a_mass[atype_mat[atype[(n3+i*3)/3]]]*d5/hsq*epsv/kb;
      if(auto_flg == 1)
	sprintf(str_buf,"Vc = %.0f(m/s) = %.0f(km/h)"
		,start_vl/h*sigma/tmrdp
		,start_vl/h*sigma/tmrdp*3.6);
      else
	sprintf(str_buf,"Vc = %.0f(m/s) = %.0f(km/h) (%4.0fK)"
		,start_vl/h*sigma/tmrdp
		,start_vl/h*sigma/tmrdp*3.6,d4);
      glRasterPos3d(d0, d1, d2);
      glCallLists(strlen(str_buf), GL_BYTE, str_buf);
    }

    if(md_stepf > 0){
      d1 -= .3;
      sprintf(str_buf,"md_step=%d",md_step);
      md_stepf--;
      glRasterPos3d(d0, d1, d2);
      glCallLists(strlen(str_buf), GL_BYTE, str_buf);
    }
    if(c_flg == C_STEP  && start_vl <= 0){
      d1 -= .3;
      sprintf(str_buf,"select velocity [0]-[9] keys");
      glRasterPos3d(d0, d1, d2);
      glCallLists(strlen(str_buf), GL_BYTE, str_buf);
    }

#ifdef LAP_TIME
    if(vflg >= 2){
			static double flops_sum = 0.0;
			static int f_steps = 0;

      d1 = -2.2;
      sprintf(str_buf,"%.3fs/step %.1fGflops",md_time-md_time0
#if defined(MDGRAPE3) || defined(VTGRAPE)
	      ,(double)n1*(double)n1*78/(md_time-md_time0)*1e-9);
			flops_sum += (double)n1*(double)n1*78/(md_time-md_time0)*1e-9;
#else
	      ,(double)n1*(double)n1/2*40/(md_time-md_time0)*1e-9);
			flops_sum += (double)n1*(double)n1/2*40/(md_time-md_time0)*1e-9;
#endif
			f_steps++;
			//printf("%f\n", flops_sum / (double)f_steps);

      glRasterPos3d(d0, d1, d2);
      glCallLists(strlen(str_buf), GL_BYTE, str_buf);
      d1 -= .3;
      sprintf(str_buf,"%.3fs/frm %.1ffrm/s",disp_time-disp_time0
	      ,1./(disp_time-disp_time0));
      glRasterPos3d(d0, d1, d2);
      glCallLists(strlen(str_buf), GL_BYTE, str_buf);
    }
#endif
    glEnable(GL_LIGHTING);

    d0 = 1.8;
    d1 = -2.4;
    d2 = -10;
    glPushMatrix();
    glTranslated(d0,d1,d2);
    d3 = temp*0.00016;
    particle_color[0] = color_table[atype_mat[0]][0]+d3;
    particle_color[1] = color_table[atype_mat[0]][1]+d3/3;
    particle_color[2] = color_table[atype_mat[0]][2]+d3/3;
    particle_color[3] = color_table[atype_mat[0]][3];
    glMaterialfv(GL_FRONT, GL_AMBIENT,particle_color);
    particle_color[0] = color_table[atype_mat[0]][0]+d3/4;
    particle_color[1] = color_table[atype_mat[0]][1]+d3/12;
    particle_color[2] = color_table[atype_mat[0]][2]+d3/12;
    particle_color[3] = color_table[atype_mat[0]][3];
    glMaterialfv(GL_FRONT, GL_DIFFUSE,particle_color);
    glutSolidSphere(radius*r_table[atype_mat[0]]/3.5, ditail, ditail/2);
    glPopMatrix();
    glDisable(GL_LIGHTING);
    medium_font(d0-0.08,d1-.04,d2,"Na");
    small_font(d0+0.08,d1+.04,d2,"+");

    glEnable(GL_LIGHTING);
    d0 += 0.5;
    glPushMatrix();
    glTranslated(d0,d1,d2);
    particle_color[0] = color_table[atype_mat[1]][0]+d3;
    particle_color[1] = color_table[atype_mat[1]][1]+d3/3;
    particle_color[2] = color_table[atype_mat[1]][2]+d3/3;
    particle_color[3] = color_table[atype_mat[1]][3];
    glMaterialfv(GL_FRONT, GL_AMBIENT,particle_color);
    particle_color[0] = color_table[atype_mat[1]][0]+d3/4;
    particle_color[1] = color_table[atype_mat[1]][1]+d3/12;
    particle_color[2] = color_table[atype_mat[1]][2]+d3/12;
    particle_color[3] = color_table[atype_mat[1]][3];
    glMaterialfv(GL_FRONT, GL_DIFFUSE,particle_color);
    glutSolidSphere(radius*r_table[atype_mat[1]]/3.5, ditail, ditail/2);
    glPopMatrix();
    glDisable(GL_LIGHTING);
    medium_font(d0-0.06,d1-.04,d2,"Cl");
    small_font(d0+0.07,d1+.04,d2,"-");
  }

#ifdef SUBWIN
  if(vflg >= 1){
    mag = 1;
    glLineWidth(1.0);
    glNormal3d(0.0,0.0,1.0);
    /*
    glMaterialfv(GL_FRONT,GL_AMBIENT_AND_DIFFUSE
		 ,moji_c[(int)(clear_color+.5)]);
    glRasterPos3d(mag*.1+sub_x, mag*.8+sub_y,d2);
    glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, 'T');
    */
    /*
    if(temp < temp_ymax){
      glBegin(GL_LINE_STRIP);
      glVertex3d(sub_x,    temp/temp_ymax*mag+sub_y,d2);
      glVertex3d(sub_x+mag,temp/temp_ymax*mag+sub_y,d2);
      glEnd();
    }
    */
    glColor4fv(line[0]);
    glBegin(GL_LINE_STRIP);
    for(i = 0; i < p_count; i++){
      glVertex3d((double)i/100*mag+sub_x,
		 (double)temp_data[i]/temp_ymax*mag+sub_y,d2);
    }
    glEnd();
    glColor4fv(waku);
    glNormal3d(0,0,1);
    glBegin(GL_LINE_LOOP);
    glVertex3d(sub_x,sub_y,d2);
    glVertex3d(sub_x+mag,sub_y,d2);
    glVertex3d(sub_x+mag,sub_y+mag,d2);
    glVertex3d(sub_x,sub_y+mag,d2);
    glEnd();

    color[3] = 1.0;
    d0 = 1081;
    if(d0 < temp_ymax){
      color[0] = 0.2; color[1] = 0.2; color[2] = 0.8;
      glColor4fv(color);
      glBegin(GL_LINE_STRIP);
      glVertex3d(sub_x,    d0/temp_ymax*mag+sub_y,d2);
      glVertex3d(sub_x+mag,d0/temp_ymax*mag+sub_y,d2);
      glEnd();
      small_font(sub_x+mag+0.02,d0/temp_ymax*mag+sub_y-0.04,d2,"MP");
      small_font(sub_x-0.33,    d0/temp_ymax*mag+sub_y-0.04,d2,"1081K");
    }

    d1 = 1738;
    if(d1 < temp_ymax){
      color[0] = 0.6; color[1] = 0.2; color[2] = 0.2;
      glColor4fv(color);
      glBegin(GL_LINE_STRIP);
      glVertex3d(sub_x,    d1/temp_ymax*mag+sub_y,d2);
      glVertex3d(sub_x+mag,d1/temp_ymax*mag+sub_y,d2);
      glEnd();
      small_font(sub_x+mag+0.02,d1/temp_ymax*mag+sub_y-0.04,d2,"BP");
      small_font(sub_x-0.34,    d1/temp_ymax*mag+sub_y-0.04,d2,"1738K");
    }

    if(temp < temp_ymax){
      color[0] = 0.9; color[1] = 0.9; color[2] = 0.9;
      glColor4fv(color);
      glBegin(GL_LINE_STRIP);
      glVertex3d(sub_x,    temp/temp_ymax*mag+sub_y,d2);
      glVertex3d(sub_x+mag,temp/temp_ymax*mag+sub_y,d2);
      glEnd();
    }
  }
#endif

  if(save_flg == 1){

    int x_pixel,y_pixel;
    int x_len,x_add;
    char char_buf;
    char out_name[256];
    FILE *fp;

    x_pixel = glutGet(GLUT_WINDOW_WIDTH);
    y_pixel = glutGet(GLUT_WINDOW_HEIGHT);

    if(x_pixel > X_PIX_SIZE || y_pixel > Y_PIX_SIZE){
      printf("Window size is too large!!\n");
    } else {

      if((x_pixel*3) % 4 != 0){
        x_len = (x_pixel*3/4 + 1)*4;
        x_add = x_len - x_pixel*3;
      } else {
        x_len = x_pixel*3;
        x_add = 0;
      }

      glReadPixels(0, 0, x_pixel, y_pixel, GL_RGB, GL_UNSIGNED_BYTE, pix);

      for(i = 0; i < y_pixel; i++){
        for(j = 0; j < x_pixel*3; j += 3){
          char_buf = pix[i*x_len+j];
          pix[i*x_len+j] = pix[i*x_len+j+2];
          pix[i*x_len+j+2] = char_buf;
        }
      }

      bmp_header.bfType[0] = 'B';
      bmp_header.bfType[1] = 'M';
      bmp_header.bfSize = 14+sizeof(bmp_info)
        +sizeof(GLubyte)*x_len*y_pixel;

      bmp_header.bfReserved1 = 0;
      bmp_header.bfReserved2 = 0;
      bmp_header.bfOffBits = 14+sizeof(bmp_info);

      bmp_info.biSize = sizeof(bmp_info);
      bmp_info.biWidth = x_pixel;
      bmp_info.biHeight = y_pixel;
      bmp_info.biPlanes = 1;
      bmp_info.biBitCount = 24;
      bmp_info.biCompression = 0;
      bmp_info.biSizeImage = sizeof(GLubyte)*x_len*y_pixel;
      bmp_info.biXPixPerMeter = 0;
      bmp_info.biYPixPerMeter = 0;
      bmp_info.biClrUsed = 0;
      bmp_info.biClrImporant = 0;

      sprintf(out_name,"a%05d.bmp",file_num++);
      printf("save %s\n",out_name);

      if((fp = fopen(out_name,"wb")) == NULL){
        printf("file open error\n");
        exit(1);
      }
      fwrite(bmp_header.bfType, sizeof(char),2,fp);
      fwrite(&bmp_header.bfSize, sizeof(bmp_header.bfSize),1,fp);
      fwrite(&bmp_header.bfReserved1, sizeof(bmp_header.bfReserved1),1,fp);
      fwrite(&bmp_header.bfReserved2, sizeof(bmp_header.bfReserved2),1,fp);
      fwrite(&bmp_header.bfOffBits, sizeof(bmp_header.bfOffBits),1,fp);

      fwrite(&bmp_info, sizeof(bmp_info),1,fp);
      fwrite(pix,sizeof(char),x_len*y_pixel,fp);

      fclose(fp);
    }
  }

  glDisable(GL_LIGHT0);
  glDisable(GL_CULL_FACE);
}
void display(void)
{
#if STEREO == 1
  glDrawBuffer(GL_BACK_LEFT);
  single_display(-1);

  glDrawBuffer(GL_BACK_RIGHT);
  single_display(1);
#else
  single_display(eye_pos);
#endif
  glutSwapBuffers();
  /*
  if(run_flg == 1)
    glutIdleFunc(md_run);
  else
    glutIdleFunc(NULL);
  */
}
void reshape(int w, int h)
{
  glViewport(0, 0, (GLsizei)w, (GLsizei)h);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective(30.0, (double)w / (double)h, 1.0, 800.0);
  glMatrixMode(GL_MODELVIEW);
}
void mouse(int button, int state, int x, int y)
{
  switch (button) {
  case GLUT_LEFT_BUTTON:
    if (state == GLUT_DOWN) {
      mpos[0] = x;
      mpos[1] = y;
      mouse_l = 1;
    }
    if (state == GLUT_UP) {
      mouse_l = 0;
    }
    break;
  case GLUT_MIDDLE_BUTTON:
    if (state == GLUT_DOWN) {
      mpos[0] = x;
      mpos[1] = y;
      mouse_m = 1;
    }
    if (state == GLUT_UP) {
      mouse_m = 0;
    }
    break;
  case GLUT_RIGHT_BUTTON:
    if (state == GLUT_DOWN) {
      mpos[0] = x;
      mpos[1] = y;
      mouse_r = 1;
    }
    if (state == GLUT_UP) {
      mouse_r = 0;
    }
    break;
  default:
    break;
  }
}
void motion(int x, int y)
{
  double d0;
  double len = 10;

  len = eye_len;

  if(mouse_l == 1 && mouse_m == 1){
    trans[0] += (double)(y-mpos[1])*len/150;
    angle[0] = -(double)(x-mpos[0])*0.2;
  } else  if(mouse_m == 1 || (mouse_l == 1 && mouse_r == 1)){
    trans[1] += (double)(x-mpos[0])*len*.001;
    trans[2] -= (double)(y-mpos[1])*len*.001;
  } else if(mouse_r == 1){
    trans[0] -= (double)(y-mpos[1])*len/150;
    angle[0] =  (double)(x-mpos[0])*0.2;
  } else if(mouse_l == 1){
    d0 = len/50;
    if(d0 > 1.0) d0 = 1.0;
    angle[1] = (double)(y-mpos[1])*d0;
    angle[2] = (double)(x-mpos[0])*d0;
  }
  if(mouse_l == 1 || mouse_m == 1 || mouse_r == 1){
    mpos[0] = x;
    mpos[1] = y;
    glutPostRedisplay();
  }
	 
}
#endif
void keyboard(unsigned char key, int x, int y)
{
  int i,j,k;
  int i0,i1,i2;
  double d0,d1,d2,d3,d4,d5;
  double ang0,ang1,ang2,ang3;

  int c;
  int cf[6];
  double cfr[6];
  double cp[18];

  double l = 2.285852;

  l /= 2;

#ifdef SOCK_ON
  /*
  if(sc_flg == 2){
    sock_write(s_sock,(int*)&key,1);
    sock_write(s_sock,m_matrix,sizeof(double)*16);
    sock_write(s_sock,i_matrix,sizeof(double)*16);
    sock_write(s_sock,trans,sizeof(double)*3);
  }
  if(sc_flg == 1){
    sock_read(s_sock,m_matrix,sizeof(double)*16);
    sock_read(s_sock,i_matrix,sizeof(double)*16);
    sock_read(s_sock,trans,sizeof(double)*3);
  }
  */
  if(sc_flg == 2){
    sock_send_int(s_sock,(int*)&key,1);
    sock_send_double(s_sock,m_matrix,16);
    sock_send_double(s_sock,i_matrix,16);
    sock_send_double(s_sock,trans,3);
  }
  if(sc_flg == 1){
    sock_recv_double(s_sock,m_matrix,16);
    sock_recv_double(s_sock,i_matrix,16);
    sock_recv_double(s_sock,trans,3);
  }
#endif

  if(key == '?'){
    printf("!   : initilize\n");
    printf("q,Q : quit program\n");
    printf("i,I : (print information of postion and angle)\n");
    printf("p,P : (set cliping area)\n");
    printf("a,A : auto mode on/off\n");
    printf("k,K : simulatin cell display on/off\n");
    printf("W   : start output bmp file\n");
    printf("c,C : change backgrand color\n");
    printf("r   : make radius small\n");
    printf("R   : make radius large\n");
    printf("d   : make ditail donw\n");
    printf("D   : make ditail up\n");
    printf("v,V : varbose on/off\n");
    printf("s   : md_step--\n");
    printf("S   : md_step++\n");
    printf("t,T : temp += 100\n");
    printf("g,G : temp -= 100\n");
    printf("y,Y : temp += 10\n");
    printf("h,H : temp -= 10\n");
    printf("z,Z : stop/restart\n");
    printf("0-9 : chage particle number\n");
    printf("n   : create a new positive ion\n");
    printf("m   : create a new negative ion\n");
    printf("N   : create 4 new ions\n");
    printf("M   : create 27 new ions\n");
    printf("SP  : shoot new ion(s)\n");
  }

  if(key == ' ' && c_flg != C_STEP && start_vl <= 0){
    grape_flg = (grape_flg == 0 ? 1:0);
  }
  if(key == '!'){
    if(sc_flg != 1){
#ifdef GL_ON
      glLoadIdentity();
      glGetDoublev(GL_MODELVIEW_MATRIX,m_matrix);
      glGetDoublev(GL_MODELVIEW_MATRIX,i_matrix);
#ifdef SUBWIN
      p_count = 0;
      temp_ymax = 2000;
      for(i = 0; i < DATA_NUM; i++)
	temp_data[i] = 0;
#endif
#endif
      trans[0] = 0;
      trans[1] = 0;
      trans[2] = 0;
    }
    c_flg = 0;
    c_num = 0;
    m_clock = 0;
    set_cd(0);
  }

  if((key >= '1' && key <= '9') && c_flg == 0){
    if(sc_flg != 1){
#ifdef GL_ON
      glLoadIdentity();
      glGetDoublev(GL_MODELVIEW_MATRIX,m_matrix);
      glGetDoublev(GL_MODELVIEW_MATRIX,i_matrix);
#ifdef SUBWIN
      p_count = 0;
      temp_ymax = 2000;
      for(i = 0; i < DATA_NUM; i++)
	temp_data[i] = 0;
#endif
#endif
      trans[0] = 0;
      trans[1] = 0;
      trans[2] = 0;
    }
    np = key-'0';
    npx = np;
    npy = np;
    npz = np;
    c_flg = 0;
    c_num = 0;
    m_clock = 0;
    set_cd(0);
  }

  if(key == 'q' || key == 'Q'){
    if(kflg == 1)
      fclose(fp);
    exit(0);
  }
#ifdef INFO
  if(key == 'i' || key == 'I'){
    printf("(%f,%f,%f)\n"
	   ,trans[0]-trans0[0],trans[1]-trans0[1],trans[2]-trans0[2]);
    printf("(");
    for(i = 0; i < 4; i++){
      for(j = 0; j < 4; j++){
	if(i == 0 && j == 0)
	  printf("%f",m_matrix[i*4+j]-matrix0[i*4+j]);
	else
	  printf(",%f",m_matrix[i*4+j]-matrix0[i*4+j]);
      }
    }
    printf(")\n");
    for(i = 0; i < 3; i++){
      trans0[i] = trans[i];
    }
    for(i = 0; i < 16; i++)
      matrix0[i] = m_matrix[i];
  }
#endif
  if(key == 'f' || key == 'F') bond_flg = bond_flg == 0 ? 1:0;
#ifdef GL_ON
  if(key == 'p' || key == 'P'){
    if(clip_flg == 4) clip_flg = 0; else clip_flg++;
  }
  if(key == 'a' || key == 'A') auto_flg = auto_flg == 0 ? 1:0;
  if(key == 'k' || key == 'K') kabe_flg = kabe_flg == 0 ? 1:0;
  if(key == 'W') save_flg = save_flg == 0 ? 1:0;
  if(key == 'c' || key == 'C') clear_color = clear_color == 0 ? 1:0;
  if(key == 'r' && (int)(radius*10+.5) > 1) {radius -= .1;}
  if(key == 'R') {radius += .1; }
  if(key == 'd' && ditail > 5)  {ditail -= 1; }
  if(key == 'D' && ditail < 20) {ditail += 1; }
  if(key == 'v' || key == 'V'){
    if(vflg != 0)
      vflg--;
    else
#ifdef LAP_TIME
      vflg = 3;
#else
      vflg = 1;
#endif
  }
#endif
#if defined(MDGRAPE3) || defined(VTGRAPE)
  if(key == 's' && md_step > 10){ md_step -= 10; md_stepf = 10;}
  if(key == 'S'){ md_step += 10; md_stepf = 10;}
#else
  if(key == 's' && md_step > 1){ md_step -= 1; md_stepf = 10;}
  if(key == 'S'){ md_step += 1; md_stepf = 10;}
#endif
  if(key == 't' || key == 'T'){
    temp += 100;
    rtemp = temp / epsv * kb;
  }
  if(key == 'g' || key == 'G'){
    if(temp > 100){
      temp -= 100;
      rtemp = temp / epsv * kb;
    }
  }
  if(key == 'h' || key == 'H'){
    if(temp > 10){
      temp -= 10;
      rtemp = temp / epsv * kb;
    }
  }
  if(key == 'y' || key == 'Y'){
    temp += 10;
    rtemp = temp / epsv * kb;
  }
  if(key == 'z' || key == 'Z'){
    run_flg *= -1;
#ifdef GL_ON
    if(sc_flg == 0){
      if(run_flg == 1)
        glutIdleFunc(md_run);
      else
        glutIdleFunc(NULL);
    } else {
      if(run_flg == 1)
        glutIdleFunc(sock_md_run);
      else
	glutIdleFunc(NULL);
    }
#endif
  }
  /*
  if(key == '0')
    eye_pos = -1;
  if(key == '-')
    eye_pos = 0;
  if(key == '=')
    eye_pos = 1;
  */
  /*
#ifdef GL_ON
  if((key >= '1' && key <= '5') && c_flg == 0){
    drow_flg[key-'1'] = (drow_flg[key-'1'] == 1 ? 0:1);
  }
#endif
  */
  if(key >= '0' && key <= '9' && c_flg == C_STEP){
    start_vl = .3*(key-'0'+1)/10*delt/2e-15;
    velp_flg = 1;
  }
  if((key == 'N' || key == 'M' || key == 'B' ||
      key == 'n' || key == 'm' || key == 'b') && c_flg == 0){

    c_flg = 1;
    w_add = s_add = 0;

    if(key == 'b' || key == 'B'){
      c_num = w_site;
      w_add = 1;
    } else if(key == 'N'){
      c_num = 4;
      s_add = 4;
      r = 3;
    } else if(key == 'M'){
      c_num = 27;
      s_add = 27;
      r = 9;
    } else {
      c_num = 1;
      s_add = 1;
      r = 1;
    }

    d0 = (i_matrix[0]*(-trans[0])+
          i_matrix[4]*(-trans[1])+
          i_matrix[8]*(-trans[2]));
    d1 = (i_matrix[1]*(-trans[0])+
          i_matrix[5]*(-trans[1])+
          i_matrix[9]*(-trans[2]));
    d2 = (i_matrix[2]*(-trans[0])+
          i_matrix[6]*(-trans[1])+
          i_matrix[10]*(-trans[2]));
    d0 += sideh[0];
    d1 += sideh[1];
    d2 += sideh[2];

    d3 = (i_matrix[0]*(-trans[0]+eye_len-10)+
          i_matrix[4]*(-trans[1])+
          i_matrix[8]*(-trans[2]));
    d4 = (i_matrix[1]*(-trans[0]+eye_len-10)+
          i_matrix[5]*(-trans[1])+
          i_matrix[9]*(-trans[2]));
    d5 = (i_matrix[2]*(-trans[0]+eye_len-10)+
          i_matrix[6]*(-trans[1])+
          i_matrix[10]*(-trans[2]));
    d3 += sideh[0];
    d4 += sideh[1];
    d5 += sideh[2];
    t_cd[0] = d3-d0;
    t_cd[1] = d4-d1;
    t_cd[2] = d5-d2;
    /*
    for(i = 0; i < 4; i++){
      for(j = 0; j < 4; j++){
        printf("%f ",i_matrix[i*4+j]);
      }
      printf("\n");
    }
    */
    /*
    printf("%f %f %f %f  %f %f %f\n",trans[0],trans[1],trans[2],eye_len
           ,d3,d4,d5);
    */

    if(d3 < side[0]-r && d3 > r &&
       d4 < side[1]-r && d4 > r &&
       d5 < side[2]-r && d5 > r){
    } else {

      for(i = 0; i < 6; i++){
        cf[i] = 0;
        cfr[i] = -1;
      }

      i0 = 0;      /* x > */
      cp[i0]   = side[0]-r;
      cp[i0+1] = (cp[i0]-d3)/(d3-d0)*(d4-d1) + d4;
      cp[i0+2] = (cp[i0]-d3)/(d3-d0)*(d5-d2) + d5;
      if(cp[i0+1] > r && cp[i0+1] < side[1]-r &&
         cp[i0+2] > r && cp[i0+2] < side[2]-r) cf[i0/3] = 1;
      i0 += 3;    /* < x */
      cp[i0]   = r;
      cp[i0+1] = (cp[i0]-d3)/(d3-d0)*(d4-d1) + d4;
      cp[i0+2] = (cp[i0]-d3)/(d3-d0)*(d5-d2) + d5;
      if(cp[i0+1] > r && cp[i0+1] < side[1]-r &&
         cp[i0+2] > r && cp[i0+2] < side[1]-r) cf[i0/3] = 1;

      i0 += 3;    /* y > */
      cp[i0+1]   = side[1]-r;
      cp[i0]   = (cp[i0+1]-d4)/(d4-d1)*(d3-d0) + d3;
      cp[i0+2] = (cp[i0+1]-d4)/(d4-d1)*(d5-d2) + d5;
      if(cp[i0]   > r && cp[i0]   < side[0]-r &&
         cp[i0+2] > r && cp[i0+2] < side[2]-r) cf[i0/3] = 1;
      i0 += 3;    /* < y */
      cp[i0+1]   = r;
      cp[i0]   = (cp[i0+1]-d4)/(d4-d1)*(d3-d0) + d3;
      cp[i0+2] = (cp[i0+1]-d4)/(d4-d1)*(d5-d2) + d5;
      if(cp[i0]   > r && cp[i0]   < side[0]-r &&
         cp[i0+2] > r && cp[i0+2] < side[2]-r) cf[i0/3] = 1;

      i0 += 3;   /* z > */
      cp[i0+2]   = side[2]-r;
      cp[i0]   = (cp[i0+2]-d5)/(d5-d2)*(d3-d0) + d3;
      cp[i0+1] = (cp[i0+2]-d5)/(d5-d2)*(d4-d1) + d4;
      if(cp[i0]   > r && cp[i0]   < side[0]-r &&
         cp[i0+1] > r && cp[i0+1] < side[1]-r) cf[i0/3] = 1;
      i0 += 3;
      cp[i0+2]   = r;
      cp[i0]   = (cp[i0+2]-d5)/(d5-d2)*(d3-d0) + d3;
      cp[i0+1] = (cp[i0+2]-d5)/(d5-d2)*(d4-d1) + d4;
      if(cp[i0]   > r && cp[i0]   < side[0]-r &&
         cp[i0+1] > r && cp[i0+1] < side[1]-r) cf[i0/3] = 1;

      for(i = 0; i < 6; i++){
        if(cf[i] == 1){
          cfr[i] = sqrt((cp[i*3]  -d3)*(cp[i*3]  -d3)+
                        (cp[i*3+1]-d4)*(cp[i*3+1]-d4)+
                        (cp[i*3+2]-d5)*(cp[i*3+2]-d5));
        }
      }
      d0 = 10000;
      c = -1;
      for(i = 0; i < 6; i++){
        if(cf[i] == 1 && d0 > cfr[i]){
          d0 = cfr[i];
          c = i;
        }
      }
      if(c == -1) c_num = 0;
      c *= 3;
      d3 = cp[c];
      d4 = cp[c+1];
      d5 = cp[c+2];
    }

    if(key == 'b' || key == 'B'){
      for(i = 0; i < c_num*3/w_site; i+=3*w_site){
        cd[n3+i]   = d3;
        cd[n3+i+1] = d4;
        cd[n3+i+2] = d5;
        atype[n1+i/3] = 2;

        ang0 = PI/2 * (key == 'b' ? 1:-1);
        ang1 = 0; 
        ang2 = 0;
        ang[w_num*4  ] = sin(ang1/2)*sin((ang2-ang0)/2);
        ang[w_num*4+1] = sin(ang1/2)*cos((ang2-ang0)/2);
        ang[w_num*4+2] = cos(ang1/2)*sin((ang2+ang0)/2);
        ang[w_num*4+3] = cos(ang1/2)*cos((ang2+ang0)/2);

        w_index[w_num] = n3+i;
        w_info[n1] =  (n3+i)/3+1;
        for(k = 1; k < w_site; k++){
          w_info[n1+k] =  (n3+i)/3;
          atype[n1+i/3+k] = 2+k;
        }
        ang0 = ang[w_num*4  ];
        ang1 = ang[w_num*4+1];
        ang2 = ang[w_num*4+2];
        ang3 = ang[w_num*4+3];
        for(k = 0; k < w_site-1; k++){
          d0 = m_cdx[k]*(-ang0*ang0+ang1*ang1-ang2*ang2+ang3*ang3)
              +m_cdy[k]*(-2)*(ang0*ang1+ang2*ang3)
              +m_cdz[k]*( 2)*(ang1*ang2-ang0*ang3);
          d1 = m_cdx[k]*  2 *(ang2*ang3-ang0*ang1)
              +m_cdy[k]*( ang0*ang0-ang1*ang1-ang2*ang2+ang3*ang3)
              +m_cdz[k]*(-2)*(ang0*ang2+ang1*ang3);
          d2 = m_cdx[k]*  2 *(ang1*ang2+ang0*ang3)
              +m_cdy[k]*  2 *(ang1*ang3-ang0*ang2)
              +m_cdz[k]*(-ang0*ang0-ang1*ang1+ang2*ang2+ang3*ang3);
          cd[(k+1)*3  +n3+i] = cd[n3+i  ] + d0;
          cd[(k+1)*3+1+n3+i] = cd[n3+i+1] + d1;
          cd[(k+1)*3+2+n3+i] = cd[n3+i+2] + d2;
        }
      }
    } else {

      /*
      for(i = 0; i < c_num*3; i+=3){
        cd[n3+i]   = d3;
        cd[n3+i+1] = d4;
        cd[n3+i+2] = d5;
        if(key == 'n' || key == 'N')
          atype[n1+i/3] = 0;
        else if(key == 'm' || key == 'M')
          atype[n1+i/3] = 1;
      }
      */
      if(c_num == 1){
        cd[n3]   = d3;
        cd[n3+1] = d4;
        cd[n3+2] = d5;
        atype[n1] = ((key == 'n') ? 0:1);
      } else if(c_num == 4){
        i = 0;
        l *= 1.1;
        for(i0 = 0; i0 < 2; i0++){
          for(i1 = 0; i1 < 2; i1++){
            d0 = i_matrix[4]*(l*(i0*2-1))
                +i_matrix[8]*(l*(i1*2-1));
            d1 = i_matrix[5]*(l*(i0*2-1))
                +i_matrix[9]*(l*(i1*2-1));
            d2 = i_matrix[6]*(l*(i0*2-1))
                +i_matrix[10]*(l*(i1*2-1));
            cd[n3+i]   = d0+d3;
            cd[n3+i+1] = d1+d4;
            cd[n3+i+2] = d2+d5;
            atype[n1+i/3] = (i0+i1) %2;
            i += 3;
          }
        }
      } else if(c_num == 27){
        i = 0;
        l *= 1.2;
        for(i0 = 0; i0 < 3; i0++){
          for(i1 = 0; i1 < 3; i1++){
            for(i2 = 0; i2 < 3; i2++){
              d0 = i_matrix[0]*(l*2*(i0-1))
                  +i_matrix[4]*(l*2*(i1-1))
                  +i_matrix[8]*(l*2*(i2-1));
              d1 = i_matrix[1]*(l*2*(i0-1))
                  +i_matrix[5]*(l*2*(i1-1))
                  +i_matrix[9]*(l*2*(i2-1));
              d2 = i_matrix[2]*(l*2*(i0-1))
                  +i_matrix[6]*(l*2*(i1-1))
                  +i_matrix[10]*(l*2*(i2-1));
              cd[n3+i]   = d0+d3;
              cd[n3+i+1] = d1+d4;
              cd[n3+i+2] = d2+d5;
              atype[n1+i/3] = (i0+i1+i2) %2;
              i += 3;
            }
          }
        }
      }
    }
    /*
    d0 = (i_matrix[0]*(-trans[0])+
          i_matrix[4]*(-trans[1])+
          i_matrix[8]*(-trans[2]));
    d1 = (i_matrix[1]*(-trans[0])+
          i_matrix[5]*(-trans[1])+
          i_matrix[9]*(-trans[2]));
    d2 = (i_matrix[2]*(-trans[0])+
          i_matrix[6]*(-trans[1])+
          i_matrix[10]*(-trans[2]));

    d0 += sideh[0];
    d1 += sideh[1];
    d2 += sideh[2];

    d3 = d4 = d5 = 0;
    for(i = 0; i < 3*c_num; i += 3){
      d3 += cd[n3+i];
      d4 += cd[n3+i+1];
      d5 += cd[n3+i+2];
    }
    d3 /= c_num;
    d4 /= c_num;
    d5 /= c_num;

    t_cd[0] = d3-d0;
    t_cd[1] = d4-d1;
    t_cd[2] = d5-d2;
    */
  }
  if(key == ' ' && c_flg == C_STEP && start_vl > 0){

    w_num += w_add;
    w_num3 = w_num*3;
    s_num += s_add;
    s_num3 = s_num*3;
    ws_num = w_num + s_num;
    ws_num3= ws_num*3;

    n1 = s_num + w_num*w_site;
    n3 = n1*3;
    tscale = 1. / 3. /((double)(s_num + w_num*2) - 1);

    r = sqrt(t_cd[0]*t_cd[0] + t_cd[1]*t_cd[1] + t_cd[2]*t_cd[2]);
    d3 = 0;
    for(i = 0; i < c_num*3; i+=3){
      vl[n3-3-i]   = -t_cd[0]/r *start_vl;
      vl[n3-3-i+1] = -t_cd[1]/r *start_vl;
      vl[n3-3-i+2] = -t_cd[2]/r *start_vl;
      d3 += (vl[n3-3-i]*vl[n3-3-i]+vl[n3-2-i]*vl[n3-2-i]+vl[n3-1-i]*vl[n3-1-i])
        *a_mass[atype_mat[atype[(n3-3-i)/3]]];
    }
    d3 *= tscale/hsq;
    rtemp += d3;
    temp = rtemp * epsv /kb;

    c_flg = 0;
    c_num = 0;
    velp_flg = 0;
    start_vl = -1;

    /*
    for(i = 0; i < n3; i += 3)
      printf("%d %d %f %f %f\n",i/3,atype[i/3],cd[i],cd[i+1],cd[i+2]);
    */
    /*
    for(i = 0; i < w_num; i++)
      printf("%d %d\n",i,w_index[i]);
    for(i = 0; i <n1; i++)
      printf("%d %d\n",i,w_info[i]);
    */
  }

#ifdef GL_ON
  if(sc_flg != 1)
    glutPostRedisplay();
#endif
}
int main(int argc, char **argv)
{
  int i,j,k;
  int i0,i1;
  double d0;
  char sbuf[50];
  char tt_name[256];

#ifdef SOCK_ON
  int loop_flg;

  char port_num[10];
  char host_name[50];
  int s_connected = 1;
  char s_buffer[1024];
  char s_port_num[10];
  int s_port = -1;
  struct sigaction s_act, s_oldact;
  int s_mode;
#endif

#ifdef SOCK_ON
  if(argc == 1){
    printf("MD simulation program \"claret\" by using OpenGL ");
    printf("Copyright (C) 2001 Koishi\n");
    printf("Usage: server mode cras%02d -s [port num]\n",(int)(VER*100));
    printf("       client mode cras%02d -c [host name] [port num]\n"
           ,(int)(VER*100));
    exit(0);
  }
#endif
  np = 15; // !!!
  temp = 300;
  for(i = 1; i < argc; i++){
    if(argv[i][0] == '-' && argv[i][1] == 'a'){
      auto_flg = 1;
      for(j = 0; tt_name[j]=argv[i+1][j]; j++);
      if(argc > i+2)
	i+=2;
      else break;
    }
#ifdef SOCK_ON
    if(argv[i][0] == '-' && argv[i][1] == 's'){
      sc_flg = 1;
      for(j = 0; port_num[j]=argv[i+1][j]; j++);
      host_name[0] = 0;
      i++;
    }
    if(argv[i][0] == '-' && argv[i][1] == 'c'){
      sc_flg = 2;
      for(j = 0; host_name[j]=argv[i+1][j]; j++);
      for(j = 0; port_num[j]=argv[i+2][j]; j++);
      i+=2;
    }
#endif
    if(argv[i][0] == 'm' || argv[i][0] == 'M'){
      j = 0;
      while(sbuf[j]=argv[i][j+1])
        j++;
      if(!(timemx = atoi(sbuf))) timemx =-1;
    }
    if(argv[i][0] == 's' || argv[i][0] == 'S'){
      j = 0;
      while(sbuf[j]=argv[i][j+1])
        j++;
      if(!(sys_num = atoi(sbuf))) sys_num = 0;
    }
    if(argv[i][0] == 'n' || argv[i][0] == 'N'){
      if(argv[i][1] == 'x' || argv[i][1] == 'X'){
        for(j = 0; sbuf[j]=argv[i][j+2]; j++);
        if(!(npx = atof(sbuf))) npx = 2;
      } else if(argv[i][1] == 'y' || argv[i][1] == 'Y'){
        for(j = 0; sbuf[j]=argv[i][j+2]; j++);
        if(!(npy = atof(sbuf))) npy = 2;
      } else if(argv[i][1] == 'z' || argv[i][1] == 'Z'){
        for(j = 0; sbuf[j]=argv[i][j+2]; j++);
        if(!(npz = atof(sbuf))) npz = 2;
      } else if(argv[i][1] == 'n' || argv[i][1] == 'N'){
        for(j = 0; sbuf[j]=argv[i][j+2]; j++);
        if(!(nn = atof(sbuf))) nn = 0;
      } else if(argv[i][1] == 'w' || argv[i][1] == 'W'){
        for(j = 0; sbuf[j]=argv[i][j+2]; j++);
        if(!(nw = atof(sbuf))) nw = 0;
      } else {
        for(j = 0; sbuf[j]=argv[i][j+1]; j++);
        if(!(np = atof(sbuf))) np = 2;
        npx = np;
        npy = np;
        npz = np;
      }
    }
    if(argv[i][0] == 't' || argv[i][0] == 'T'){
      printf("temp check %s\n",argv[i]);
      tflg = 1;
      j = 0;
      while(sbuf[j]=argv[i][j+1])
        j++;
      if(!(temp = atof(sbuf))) temp = 100;
    }
    if(argv[i][0] == 'k' || argv[i][0] == 'K'){
      kflg = 1;
      j = 0;
      while(k_file[j]=argv[i][j+1])
        j++;
      k_file[j] = 0;
      strcat(k_file,".dat");
    }
  }

#ifdef SOCK_ON
#ifdef SWAP_ENDIAN
  if((send_buf_int = malloc(10 * sizeof(int)*10)) == NULL){
    printf("memory error\n");
    exit(1);
  }
  if((send_buf_double = malloc(10*sizeof(double))) == NULL){
    printf("memory error\n");
    exit(1);
  }
#endif
  if(sc_flg == 1){
    printf("server mode: open port %s\n",port_num);

    ignore_pipe();
    sigemptyset(&s_act.sa_mask);
    s_act.sa_flags = 0;
    s_act.sa_handler = sig_chld;
    sigaction(SIGCHLD, &s_act, &s_oldact);

    s_port = atoport(port_num, "tcp");
    if (s_port == -1) {
      fprintf(stderr,"Unable to find service: %s\n",port_num);
      exit(EXIT_FAILURE);
    }
    s_sock = get_connection(SOCK_STREAM, s_port, &listensock);
    printf("get_connection\n");
    connectsock = s_sock;
    sock_puts(s_sock,"Welcome to the MD server.\n");

    sock_recv_int(s_sock,&sys_num,1);
    sock_recv_int(s_sock,&np,1);
    printf("np %d\n",np);
    sock_recv_int(s_sock,&npx,1);
    sock_recv_int(s_sock,&npy,1);
    sock_recv_int(s_sock,&npz,1);
    sock_recv_int(s_sock,&nn,1);
    sock_recv_int(s_sock,&nw,1);
    sock_recv_double(s_sock, &temp, 1);
    printf("temp %f\n",temp);
    sock_recv_int(s_sock, &tflg, 1);
    sock_recv_int(s_sock, &auto_flg, 1);
  }
  if(sc_flg == 2){
    printf("client mode: try to make connection %s %s\n",host_name,port_num);

    ignore_pipe();
    s_sock = make_connection(port_num, SOCK_STREAM, host_name);
    if (s_sock == -1) {
      fprintf(stderr,"make_connection failed.\n");
      return -1;
    }
    sock_gets(s_sock,s_buffer,sizeof(s_buffer));
    printf("%s\n",s_buffer);

    sock_send_int(s_sock,&sys_num,1);
    sock_send_int(s_sock,&np,1);
    sock_send_int(s_sock,&npx,1);
    sock_send_int(s_sock,&npy,1);
    sock_send_int(s_sock,&npz,1);
    sock_send_int(s_sock,&nn,1);
    sock_send_int(s_sock,&nw,1);
    sock_send_double(s_sock,&temp,1);
    sock_send_int(s_sock,&tflg,1);
    sock_send_int(s_sock,&auto_flg,1);

    grape_flg = 1;
  }
#endif
  if(auto_flg == 1){
    make_time_table(tt_name);
  }
  /*
  init_MD();
  keep_mem(s_num,w_num*w_site);
  set_cd(1);
  for(i =0; i < 40; i++)
    md_run();
  exit(0);
  */

#if defined(SUBWIN) && defined(GL_ON)
  sub_x = 1.5;
  sub_y = 1.5;
  temp_ymax = 2000;
#endif

#ifdef GL_ON
  if(sc_flg == 0 || sc_flg == 2){
    glutInit(&argc, argv);
#if STEREO == 1
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH | GLUT_STEREO);
    glutInitWindowSize(1100, 800);
    glutInitWindowPosition(0, 0);
    glutCreateWindow(argv[0]);
    glutFullScreen();
#else
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
    glutInitWindowSize(512, 512);
    glutInitWindowPosition(100, 0);
    glutCreateWindow(argv[0]);
    // glutFullScreen(); // !!!
#endif

    init();
  }
#endif

  init_MD();
  /*  keep_mem(s_num+20,(w_num+20)*w_site);*/
  keep_mem(S_NUM_MAX,W_NUM_MAX*w_site);
  set_cd(1);

#ifdef SOCK_ON
#ifdef SWAP_ENDIAN
  free(send_buf_int);
  free(send_buf_double);
  if((send_buf_int = malloc((s_num+w_num*w_site+140) * sizeof(int)*10)) == NULL){
    printf("memory error\n");
    exit(1);
  }
  if((send_buf_double = malloc((s_num+w_num*w_site+140)*3*sizeof(double))) == NULL){
    printf("memory error\n");
    exit(1);
  }
#endif

  if(sc_flg == 1){
    loop_flg = 1;
    while(loop_flg){
      sock_recv_int(s_sock, &s_mode, 1);

      switch(s_mode){
      case 'q':
      case 'Q':
        loop_flg = 0;
        break;
      case 1:
        md_run();
        cd[n3+c_num*3] = mtemp;
        cd[n3+c_num*3+1] = m_clock;
#ifdef LAP_TIME
        cd[n3+c_num*3+2] = md_time0;
        cd[n3+c_num*3+3] = md_time;
        sock_send_double(s_sock,cd,(n3+c_num*3+4));
#else
        sock_send_double(s_sock,cd,(n3+c_num*3+2));
#endif
        if(c_flg != 0){
          if(c_flg+md_step <= C_STEP) c_flg += md_step; else c_flg = C_STEP;
        }

	sock_send_double(s_sock,vl,n3);
	if(bond_flg == 1){
	  sock_send_int(s_sock,nig_num,n1);
	  sock_send_int(s_sock,(int*)nig_data,n1*6);
	}
        break;
      default:
        keyboard(s_mode,i,j);
      }
    }
    close(s_sock);
    return 0;
  }
#endif

#ifdef LAP_TIME
#if defined(_WIN32) && !defined(__CYGWIN__)
  disp_time = (double)timeGetTime()/1000.;
#elif defined(MAC)
  disp_time = (double)clock()/60.;
#else
  gettimeofday(&time_v,NULL);
  disp_time = (time_v.tv_sec + time_v.tv_usec / 1000000.0);
#endif
#endif

#ifdef GL_ON
  glutDisplayFunc(display);
  glutReshapeFunc(reshape);
  glutMouseFunc(mouse);
  glutMotionFunc(motion);
  glutKeyboardFunc(keyboard);
  if(sc_flg == 0){
    if(run_flg == 1)
      glutIdleFunc(md_run);
    else
      glutIdleFunc(NULL);
  } else {
    if(run_flg == 1)
      glutIdleFunc(sock_md_run);
    else
      glutIdleFunc(NULL);
  }
  glutMainLoop();
#else

  init_MD();
  while(1) {
    md_run();
    printf("%.1fGflops\n"
#if defined(MDGRAPE3) || defined(VTGRAPE)
    ,(double)n1*(double)n1*78/(md_time-md_time0)*1e-9);
#else
    ,(double)n1*(double)n1/2*40/(md_time-md_time0)*1e-9);
#endif
  }
#endif
  return 0;
}
void make_time_table(char *file_name)
{
  int i,j;
  int num;
  int i0,i1,i2;
  int sfrm,efrm,slope,mode;
  double d0,d1,d2;
  double d[16],h[16],g[16];
  double x,y,z;
  double hx,hy,hz;
  double gx,gy,gz;
  char **buf;
  char key[10];
  char com[10];
  FILE *fp;

  if((tt = malloc((TIMETABLE_MAX) * sizeof(TIMETABLE))) == NULL){
    printf("memory error\n");
    exit(1);
  }

  if((buf = malloc((TIMETABLE_MAX) * sizeof(char*))) == NULL){
    printf("memory error\n");
    exit(1);
  }
  for(i = 0; i < TIMETABLE_MAX; i++)
    if((buf[i] = malloc((245) * sizeof(char))) == NULL){
      printf("memory error\n");
      exit(1);
    }

  for(i = 0; i < TIMETABLE_MAX; i++){
    for(j = 0; j < 3; j++)
      tt[i].mouse[j] = 0;
    for(j = 0; j < 3; j++)
      tt[i].move[j] = 0;
    for(j = 0; j < 3; j++)
      tt[i].rot[j] = 0;
    tt[i].command = 0;
    tt[i].temp = 0;
    for(j = 0; j < 16; j++)
      tt[i].matrix[j] = 0;
  }

  if(sc_flg != 1){
    printf("%s\n",file_name);
    if((fp = fopen(file_name,"rt")) == NULL){
      printf("file open error %s\n",file_name);
      exit(1);
    }
    fgets(buf[0],256,fp);
  }
#ifdef SOCK_ON
  if(sc_flg == 2){
    sock_send_char(s_sock,buf[0],256);
  }
  if(sc_flg == 1){
    sock_recv_char(s_sock,buf[0],256);
  }
#endif

  strsrc2(buf[0],"h",&delt);
  strsrc2(buf[0],"n",&d0);
  np = (int)(d0+.5);
  strsrc2(buf[0],"t",&temp);
  strsrc2(buf[0],"ditail",&d0);
#ifdef GL_ON
  ditail = (int)(d0+.5);
#endif
  strsrc2(buf[0],"md_step",&d0);
  md_step = (int)(d0+.5);
  strsrc2(buf[0],"vflg",&d0);
  vflg = (int)(d0+.5);
  strsrc2(buf[0],"temp_type",&d0);
  temp_unit_type = (int)(d0+.5);
  strcpy(temp_unit[0],"K");
  strcpy(temp_unit[1],"C");

  if(sc_flg != 1)
    for(num = 0; fgets(buf[num],256,fp) != NULL; num++);

  /*
  printf("%d\n",num);
  for(i = 0; i < num; i++){
    printf("%s\n",buf[i]);
  }
  exit(0);
  */

#ifdef SOCK_ON
  if(sc_flg == 2){
    sock_send_int(s_sock,&num,1);
  }
  if(sc_flg == 1){
    sock_recv_int(s_sock,&num,1);
  }
#endif

  for(i0 = 0; i0 < num; i0++){
#ifdef SOCK_ON
    if(sc_flg == 2){
      sock_send_char(s_sock,buf[i0],256);
    }
    if(sc_flg == 1){
      sock_recv_char(s_sock,buf[i0],256);
    }
#endif
    if(buf[i0][0] == '#') continue;
    if(buf[i0][0] == 'c'){
      sscanf(buf[i0],"%s %s %d",com,key,&sfrm);
      /*
      printf("%s %s %d\n",com,key,sfrm);
       */
      if(strcmp(key,"SP"))
	tt[sfrm].command = key[0];
      else
	tt[sfrm].command = ' ';
    }
    if(buf[i0][0] == 'm'){
      sscanf(buf[i0],"%s %d (%lf,%lf,%lf) %d %d %d",com,&mode
	     ,&x,&y,&z,&sfrm,&efrm,&slope);
      /*
      printf("%s %d (%lf,%lf,%lf) %d %d %d\n",com,mode
	     ,x,y,z,sfrm,efrm,slope);
      */
      hx = x/(efrm-sfrm-slope);
      hy = y/(efrm-sfrm-slope);
      hz = z/(efrm-sfrm-slope);
      if(slope != 0){
	gx = hx/(double)slope;
	gy = hy/(double)slope;
	gz = hz/(double)slope;
      } else {
	gx = hx;
	gy = hy;
	gz = hz;
      }
      for(i = sfrm; i < sfrm+slope; i++){
	tt[i].move[0] += gx*(i-sfrm);
	tt[i].move[1] += gy*(i-sfrm);
	tt[i].move[2] += gz*(i-sfrm);
      }
      for(i = sfrm+slope; i < efrm-slope; i++){
	tt[i].move[0] += hx;
	tt[i].move[1] += hy;
	tt[i].move[2] += hz;
      }
      for(i = efrm-slope; i < efrm; i++){
	tt[i].move[0] += -gx*(i-efrm+slope)+hx;
	tt[i].move[1] += -gy*(i-efrm+slope)+hy;
	tt[i].move[2] += -gz*(i-efrm+slope)+hz;
      }
    }
    if(buf[i0][0] == 'r'){
      sscanf(buf[i0],"%s %d (%lf,%lf,%lf) %d %d %d",com,&mode
	     ,&x,&y,&z,&sfrm,&efrm,&slope);
      /*
      printf("%s %d (%lf,%lf,%lf) %d %d %d\n",com,mode
	     ,x,y,z,sfrm,efrm,slope);
      */
      hx = x/(efrm-sfrm-slope);
      hy = y/(efrm-sfrm-slope);
      hz = z/(efrm-sfrm-slope);
      if(slope != 0){
	gx = hx/(double)slope;
	gy = hy/(double)slope;
	gz = hz/(double)slope;
      } else {
	gx = hx;
	gy = hy;
	gz = hz;
      }
      for(i = sfrm; i < sfrm+slope; i++){
	tt[i].mouse[0] = 1;
	tt[i].rot[0] += gx*(i-sfrm);
	tt[i].rot[1] += gy*(i-sfrm);
	tt[i].rot[2] += gz*(i-sfrm);
      }
      for(i = sfrm+slope; i < efrm-slope; i++){
	tt[i].mouse[0] = 1;
	tt[i].rot[0] += hx;
	tt[i].rot[1] += hy;
	tt[i].rot[2] += hz;
      }
      for(i = efrm-slope; i < efrm; i++){
	tt[i].mouse[0] = 1;
	tt[i].rot[0] += -gx*(i-efrm+slope)+hx;
	tt[i].rot[1] += -gy*(i-efrm+slope)+hy;
	tt[i].rot[2] += -gz*(i-efrm+slope)+hz;
      }
    }
    if(buf[i0][0] == 'a'){
      sscanf(buf[i0],"%s %d (%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf) %d %d %d"
	     ,com,&mode
	     ,&d[0],&d[1],&d[2],&d[3],&d[4],&d[5],&d[6],&d[7],&d[8],&d[9]
	     ,&d[10],&d[11],&d[12],&d[13],&d[14],&d[15]
	     ,&sfrm,&efrm,&slope);
      /*
      for(i = 0; i < 4; i++){
	for(j = 0; j < 4; j++){
	  if(i == j) d[i*4+j] -= 1.0;
	}
      }
      */
      /*
      for(i = 0; i < 4; i++){
	for(j = 0; j < 4; j++){
	  printf("% f ",d[i*4+j]);
	}
	printf("\n");
      }
      */

      for(i = 0; i < 16; i++)
	h[i] = d[i]/(efrm-sfrm-slope);
      if(slope != 0){
	for(i = 0; i < 16; i++)
	  g[i] = h[i]/(double)slope;
      } else {
	for(i = 0; i < 16; i++)
	  g[i] = h[i];
      }
      for(i = sfrm; i < sfrm+slope; i++){
	for(j = 0; j < 16; j++)
	  tt[i].matrix[j] += g[j]*(i-sfrm);
      }
      for(i = sfrm+slope; i < efrm-slope; i++){
	for(j = 0; j < 16; j++)
	  tt[i].matrix[j] += h[j];
      }
      for(i = efrm-slope; i < efrm; i++){
	for(j = 0; j < 16; j++)
	  tt[i].matrix[j] += -g[j]*(i-efrm+slope)+h[j];
      }
    }
    if(buf[i0][0] == 't'){
      sscanf(buf[i0],"%s %lf %d %d",com,&x,&sfrm,&efrm);
      y = x/(double)(efrm-sfrm);
      /*
      printf("%f %f %d %d\n",x,y,sfrm,efrm);
      exit(0);
      */
      for(i = sfrm; i < efrm; i++)
	tt[i].temp += y;
    }
  }
#if 0
  d0 = d1 = d2 = 0;
  for(i = 0; i < 3600; i++){
    printf("%d % f % f % f % f % f % f\n",i
	   ,tt[i].move[0],tt[i].move[1],tt[i].move[2]
	   ,tt[i].rot[0],tt[i].rot[1],tt[i].rot[2]);
    /*
    d0 += tt[i].move[0];
    d1 += tt[i].move[1];
    d2 += tt[i].move[2];
    */
    d0 += tt[i].rot[0];
    d1 += tt[i].rot[1];
    d2 += tt[i].rot[2];
    /*  printf("%f %f %f\n",d0,d1,d2);*/
  }
  exit(0);
#endif
  if(sc_flg != 1){
    fclose(fp);
  }
}
void set_cd(int ini_m2)
{
  int i,j,k,c;
  int i0,i1,i2,i3,i4,i5,i10;
  double d0,d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12;
  double ang0,ang1,ang2,ang3;

  if(ini_m2 == 0){
    temp = ini_temp;
    rtemp = temp / epsv * kb;
    if(sys_num == 0){
      n1 = np*np*np*8;
    } else if(sys_num == 1 || sys_num == 10){
      n1 = np*np*np*4*w_site;
    } else if(sys_num == 2){
#if ZERO_P == 1
      n1 = (npx*npy*npz*8+npy*npz*4+(npy-1)*npz*8)*w_site;
#else
      n1 = np*np*np*8*w_site;
#endif
    } else if(sys_num == 3){
      n1 = np*np*np*8*w_site;
    } else if(sys_num == 4){
      if(np*np*np*8-nn*2 >= 0)
        n1 = (np*np*np*8-nn*2)*w_site+nn*2;
      else {
        printf("nn is too large!\n");
        exit(0);
      }
    } else if(sys_num == 5){
      if(np*np*np*8-nw >= 0)
        n1 = (np*np*np*8-nw)+nw*w_site;
      else {
        printf("nw is too large!\n");
        exit(0);
      }
    } else if(sys_num == 6){
      if(np-nw >= 0){
        if(nw > 0)
          n1 = np*np*np*8+(np*np*np*8-(np-nw)*(np-nw)*(np-nw)*8)*(w_site-1);
        else
          n1 = np*np*np*8;
      } else {
        printf("nw is too large!\n");
        exit(0);
      }
    }
    n2 = n1 * 2;
    n3 = n1 * 3;

    if(sys_num == 0){
      w_num = 0;
      w_num3 = 0;
      s_num = n1;
      s_num3 = n3;
    } else if(sys_num >= 1){
      w_num = n1/w_site;
      w_num3= w_num*3;
      s_num = 0;
      s_num3= 0;
      if(sys_num == 4){
        w_num = np*np*np*8-nn*2;
        w_num3= w_num*3;
        s_num = nn*2;
        s_num3= s_num*3;
      }
      if(sys_num == 5){
        w_num = nw;
        w_num3= w_num*3;
        s_num = np*np*np*8-nw;
        s_num3= s_num*3;
      }
      if(sys_num == 6){
        if(nw > 0)
          w_num = np*np*np*8-(np-nw)*(np-nw)*(np-nw)*8;
        else
          w_num = 0;
        w_num3= w_num*3;
        s_num = np*np*np*8-w_num;
        s_num3= s_num*3;
      }
    }
    ws_num = w_num+s_num;
    ws_num3= ws_num*3;
    
    tscale = 1. / 3. /((double)(s_num + w_num*2) - 1);
  }

  if(sys_num == 0){
    side[0] = pow(8 / nden, 1./3.) * np;
    side0 = side[0];
    side[1] = side[0];
    side[2] = side[0];
    fccset2(np,side[0],cd);      /* set fcc */
    for(i = 0; i < s_num3/2; i++)
      cd[i+s_num3/2] = cd[i];
    for(i = 0;i < s_num3/2; i += 3){
      cd[i] += side[0] / np / 2.;
      if(cd[i] < 0)       cd[i] += side[0];
      if(cd[i] > side[0]) cd[i] -= side[0];
    }
    for(i = 0; i < s_num/2; i++)
      atype[i] = 0;
    for(i = s_num/2; i < s_num; i++)
      atype[i] = 1;

  } else if (sys_num == 1){
    strcpy(keiname,"water");
    side[0] = pow(4./nden,1./3.)*npx;
    side[1] = pow(4./nden,1./3.)*npy;
    side[2] = pow(4./nden,1./3.)*npz;

    for(i = 0; i < w_num; i++){
      w_index[i] = i*3;
      w_rindex[i] = i;
    }
    for(i = 0; i < w_num; i++){
      w_info[i] = i*(w_site-1)+w_num;
    }
    for(i = w_num; i < n1; i++){
      w_info[i] = (i-w_num)/(w_site-1);
    }
    /*
  for(i = 0; i < w_num3; i++)
    printf("%d %d\n",i,w_info[i]);
*/
    fccset_w(side);

    for(i0 = 0; i0 < w_num; i0++){
      i = w_index[i0]/3;
      atype[i] = 2;
      for(j = 0; j < w_site-1; j++)
        atype[w_info[i]+j] = j+3;
    }
    /*
    for(i = 0; i < n3; i += 3)
      printf("%d %d %f %f %f\n",i/3,atype[i/3],cd[i],cd[i+1],cd[i+2]);
    */
    for(i = 0; i < w_num*4; i += 4){
      ang0 = ((double)rand()/(double)RAND_MAX)*360*PI/180;
      ang1 = ((double)rand()/(double)RAND_MAX)*360*PI/180;
      ang2 = ((double)rand()/(double)RAND_MAX)*360*PI/180;
      ang[i  ] = sin(ang1/2)*sin((ang2-ang0)/2);
      ang[i+1] = sin(ang1/2)*cos((ang2-ang0)/2);
      ang[i+2] = cos(ang1/2)*sin((ang2+ang0)/2);
      ang[i+3] = cos(ang1/2)*cos((ang2+ang0)/2);
      angh[i  ] = ang[i  ];
      angh[i+1] = ang[i+1];
      angh[i+2] = ang[i+2];
      angh[i+3] = ang[i+3];
    }
  } else if (sys_num == 10){
    strcpy(keiname,"water");
    side[0] = pow(4./nden,1./3.)*npx;
    side[1] = pow(4./nden,1./3.)*npy;
    side[2] = pow(4./nden,1./3.)*npz;

#if 0

    for(i = 0; i < w_num; i++){
      w_index[i] = i*3;
    }
    for(i = 0; i < w_num; i++){
      w_info[i] = i*(w_site-1)+w_num;
    }
    for(i = w_num; i < n1; i++){
      w_info[i] = (i-w_num)/(w_site-1);
    }
    /*
  for(i = 0; i < w_num3; i++)
    printf("%d %d\n",i,w_info[i]);
*/

    fccset_w(side);

    for(i0 = 0; i0 < w_num; i0++){
      i = w_index[i0]/3;
      atype[i] = 2;
      for(j = 0; j < w_site-1; j++)
        atype[w_info[i]+j] = j+3;
    }

    for(i = 0; i < w_num3; i += 3){
      if(
	 (cd[i+2] < side[0]/2+side[0]/np &&
	  cd[i+2] > side[0]/2+side[0]/np-side[0]/np/2) || 
	 (cd[i+2] > side[0]/2-side[0]/np &&
	  cd[i+2] < side[0]/2-side[0]/np+side[0]/np/2)
	 ){
	atype[i/3] = 8;
      }
    }
#else

    for(i = 0; i < w_num; i++){
      w_index[i] = i*3;
    }
    fccset_w(side);
    s_num = 0;
    for(i = 0; i < w_num3; i += 3){
      if((cd[i]   > side[0]/np/2*(int)((np-2)/3*2) && cd[i]   < side[0]-side[0]/np/2*(int)((np-2)/3*2) &&
	  cd[i+1] > side[0]/np/2*(int)((np-2)/3*2) && cd[i+1] < side[0]-side[0]/np/2*(int)((np-2)/3*2)
	  ) &&
	 ((cd[i+2] < side[0]/2+side[0]/np/2*3 &&
	   cd[i+2] > side[0]/2+side[0]/np-side[0]/np/2) || 
	  (cd[i+2] > side[0]/2-side[0]/np/2*3 &&
	   cd[i+2] < side[0]/2-side[0]/np+side[0]/np/2))
	 ){
	atype[i/3] = 8;
	s_num++;
      } else {
	atype[i/3] = 2;
      }
    }
    s_num3 = s_num*3;
    w_num -= s_num;
    w_num3 = w_num*3;
    ws_num = w_num + s_num;
    ws_num3 = w_num3 + s_num3;

#ifdef GL_ON
    clip[0][0] = -1.0;
    clip[0][1] =  0.0;
    clip[0][2] =  0.0;
    clip[0][3] =  side[0]-side[0]/np/2*(int)((np-2)/3*2)-side[0]/2;
    clip[1][0] =  1.0;
    clip[1][1] =  0.0;
    clip[1][2] =  0.0;
    clip[1][3] =  clip[0][3];

    clip[2][0] =  0.0;
    clip[2][1] = -1.0;
    clip[2][2] =  0.0;
    clip[2][3] =  side[0]-side[0]/np/2*(int)((np-2)/3*2)-side[0]/2;
    clip[3][0] =  0.0;
    clip[3][1] =  1.0;
    clip[3][2] =  0.0;
    clip[3][3] =  clip[2][3];

    clip[4][0] =  0.0;
    clip[4][1] =  0.0;
    clip[4][2] = -1.0;
    clip[4][3] =  side[0]/2+side[0]/np-side[0]/np/2-side[0]/2;
    clip[5][0] =  0.0;
    clip[5][1] =  0.0;
    clip[5][2] =  1.0;
    clip[5][3] =  side[0]/2+side[0]/np/2*3-side[0]/2;
#endif

    for(i = s_num+w_num; i < n1; i += w_site-1){
      for(j = 0; j < w_site-1; j++)
        atype[i+j] = 3+j;
    }
    /*
    for(i = 0; i < n3; i += 3)
      printf("%d %d %f %f %f\n",i/3,atype[i/3],cd[i],cd[i+1],cd[i+2]);
    */

    for(i = 0; i < n1; i++)
      w_info[i] = -1;
    c = 0;
    for(i = 0; i < s_num+w_num; i++)
      if(atype[i] == 2){
        w_index[c/(w_site-1)] = i*3;
        w_info[i] = c+w_num+s_num;
        for(j = 0; j < w_site-1; j++)
          w_info[c+w_num+s_num+j] = i;
        c += w_site-1;
      }

    n1 = s_num + w_num*w_site;
    n2 = n1*2;
    n3 = n1*3;

    /*
    for(i = 0; i < w_num; i++)
      printf("%d %d %d\n",i,w_index[i]/3,atype[w_index[i]/3]);
    for(i = 0; i < n1; i++)
      printf("%d %d\n",i,w_info[i]);
    exit(0);
    */
    /*
    i0 = 0;
    for(i = 0; i < w_num; i++){
      if(atype[w_index[i]/3] == 8){
	i0++;
      } else if(i0 != 0){
	w_index[i-i0] = w_index[i];
      }
    }
    w_num -= i0;
    for(i = 0; i < w_num; i++)
      printf("%d %d %d\n",i,w_index[i]/3,atype[w_index[i]/3]);
    w_num3 = w_num*3;
    */
    /*
    for(i = 0; i < n3; i += 3)
      printf("%d %d %f %f %f\n",i/3,atype[i/3],cd[i],cd[i+1],cd[i+2]);
    exit(0);
    */
#endif

    for(i = 0; i < w_num*4; i += 4){
      ang0 = ((double)rand()/(double)RAND_MAX)*360*PI/180;
      ang1 = ((double)rand()/(double)RAND_MAX)*360*PI/180;
      ang2 = ((double)rand()/(double)RAND_MAX)*360*PI/180;
      ang[i  ] = sin(ang1/2)*sin((ang2-ang0)/2);
      ang[i+1] = sin(ang1/2)*cos((ang2-ang0)/2);
      ang[i+2] = cos(ang1/2)*sin((ang2+ang0)/2);
      ang[i+3] = cos(ang1/2)*cos((ang2+ang0)/2);
      angh[i  ] = ang[i  ];
      angh[i+1] = ang[i+1];
      angh[i+2] = ang[i+2];
      angh[i+3] = ang[i+3];
    }
  } else if(sys_num == 2){

    strcpy(keiname,"water");
    for(i = 0; i < w_num; i++)
      w_index[i] = i*3;

    for(i = 0; i < w_num; i++)
      w_info[i] = i*(w_site-1)+w_num;

    for(i = w_num; i < n1; i++)
      w_info[i] = (i-w_num)/(w_site-1);

    for(i0 = 0; i0 < w_num; i0++){
      i = w_index[i0]/3;
      atype[i] = 2;
      for(j = 0; j < w_site-1; j++)
        atype[w_info[i]+j] = j+3;
    }
    /*
    for(i0 = 0; i0 < w_num; i0++){
      printf("%d %d\n",i0,w_index[i0]);
    }
    exit(0);
    */
    /*
    for(i0 = 0; i0 < n1; i0++)
      printf("%d %d %d\n",i0,atype[i0],w_info[i0]);
    exit(0);
    */
    ice_set(side);

#if ZERO_P == 1

    c = npx*npy*npz*8*3;
    for(i = 0; i < npx*npy*npz*8*3; i += 3){
      if(cd[i] < side[0]/npx/2){
        cd[c]   = cd[i]  +side[0];
        cd[c+1] = cd[i+1];
        cd[c+2] = cd[i+2];
        i0 = w_index[i/3]/3*4;
        ang[c/3*4]   = ang[i0];
        ang[c/3*4+1] = ang[i0+1];
        ang[c/3*4+2] = ang[i0+2];
        ang[c/3*4+3] = ang[i0+3];
        c += 3;
      }
    }
    c = (npx*npy*npz*8+npy*npz*4)*3;
    for(k = 0; k < npz ; k++){
      for(i = 0; i < npy-1 ; i++){
        for(j = 0; j < 4*3; j += 3){
          cd[c]   = cd[j]   - side[0]/npx;
          cd[c+1] = cd[j+1] + (side[1]/npy)*(j >= 6 ? 1:0) + side[1]/npy*i;
          cd[c+2] = cd[j+2] + side[2]/npz*k;
          i0 = w_index[j/3]/3*4;
          ang[c/3*4]   = ang[i0];
          ang[c/3*4+1] = ang[i0+1];
          ang[c/3*4+2] = ang[i0+2];
          ang[c/3*4+3] = ang[i0+3];
          c += 3;
        }
        for(j = 0; j < 4*3; j += 3){
          cd[c]   = cd[j]   + side[0];
          cd[c+1] = cd[j+1] + (side[1]/npy)*(j >= 6 ? 1:0) + side[1]/npy*i;
          cd[c+2] = cd[j+2] + side[2]/npz*k;
          i0 = w_index[j/3]/3*4;
          ang[c/3*4]   = ang[i0];
          ang[c/3*4+1] = ang[i0+1];
          ang[c/3*4+2] = ang[i0+2];
          ang[c/3*4+3] = ang[i0+3];
          c += 3;
        }
      }
    }
    /*
    for(i = 0; i < n3; i += 3)
      printf("%d %d %f %f %f\n",i/3,atype[i/3],cd[i],cd[i+1],cd[i+2]);
    */
    /*
    for(i = 0; i < n1*4; i += 4)
      printf("%d %d %f %f %f %f\n",i/4,atype[i/4]
             ,ang[i],ang[i+1],ang[i+2],ang[i+3]);
    */
    for(i0 = 0; i0 < w_num; i0++){
      i = w_index[i0];
      j = i0*4;
      c = w_info[i/3]*3;
      ang0 = ang[j  ];
      ang1 = ang[j+1];
      ang2 = ang[j+2];
      ang3 = ang[j+3];
      for(k = 0; k < w_site-1; k++){
        d0 = m_cdx[k]*(-ang0*ang0+ang1*ang1-ang2*ang2+ang3*ang3)
            +m_cdy[k]*(-2)*(ang0*ang1+ang2*ang3)
            +m_cdz[k]*( 2)*(ang1*ang2-ang0*ang3);
        d1 = m_cdx[k]*  2 *(ang2*ang3-ang0*ang1)
            +m_cdy[k]*( ang0*ang0-ang1*ang1-ang2*ang2+ang3*ang3)
            +m_cdz[k]*(-2)*(ang0*ang2+ang1*ang3);
        d2 = m_cdx[k]*  2 *(ang1*ang2+ang0*ang3)
            +m_cdy[k]*  2 *(ang1*ang3-ang0*ang2)
            +m_cdz[k]*(-ang0*ang0-ang1*ang1+ang2*ang2+ang3*ang3);
        cd[k*3+c  ] = cd[i  ] + d0;
        cd[k*3+c+1] = cd[i+1] + d1;
        cd[k*3+c+2] = cd[i+2] + d2;
      }
    }
    /*
    for(i = 0; i < n3; i += 3)
      printf("%d %d %f %f %f\n",i/3,atype[i/3],cd[i],cd[i+1],cd[i+2]);
    exit(0);
    */
    side[0] += side[0]/npx/2;

#ifdef GL_ON
    /*
    glPushMatrix();
    glLoadIdentity();
    glRotatef( 90,0.0,1.0,0.0);
    glGetDoublev(GL_MODELVIEW_MATRIX, m_matrix);
    glPopMatrix();
    */

    ini_flg = 1;
    mouse_l = 1;
    angle[1] = 90;

#endif

#endif

  } else if(sys_num == 3){
    strcpy(keiname,"water");
    side[0] = pow(8./nden,1./3.)*npx;
    side[1] = pow(8./nden,1./3.)*npy;
    side[2] = pow(8./nden,1./3.)*npz;

    for(i = 0; i < w_num; i++){
      w_index[i] = i*3;
    }
    for(i = 0; i < w_num; i++){
      w_info[i] = i*(w_site-1)+w_num;
    }
    for(i = w_num; i < n1; i++){
      w_info[i] = (i-w_num)/(w_site-1);
    }

    for(i0 = 0; i0 < w_num; i0++){
      i = w_index[i0]/3;
      atype[i] = 2;
      for(j = 0; j < w_site-1; j++)
        atype[w_info[i]+j] = j+3;
    }

    ice_set2(side);

    /*
      for(i = 0; i < n3; i += 3)
      printf("%d %d %f %f %f\n",i/3,atype[i/3],cd[i],cd[i+1],cd[i+2]);
      exit(0);
    */
  } else if(sys_num == 4){
    strcat(keiname,"-water");
    side[0] = pow(8./nden,1./3.)*npx;
    side[1] = pow(8./nden,1./3.)*npy;
    side[2] = pow(8./nden,1./3.)*npz;

    for(i = 0; i < s_num+w_num; i++)
      atype[i] = 2;
    for(i = s_num+w_num; i < n1; i += w_site-1){
      for(j = 0; j < w_site-1; j++)
        atype[i+j] = 3+j;
    }

    c = 0;
    while(c < s_num){
      i0 = (int)((double)rand()/(double)RAND_MAX*(s_num+w_num));
      if(atype[i0] == 2){
        if(c % 2 == 0) atype[i0] = 0; else atype[i0] = 1;
        c++;
      }
    }
    /*
  for(i = 0; i < s_num+w_num; i++)
    printf("%d %d\n",i,atype[i]);
  exit(0);
  */
    for(i = 0; i < n1; i++)
      w_info[i] = -1;
    c = 0;
    for(i = 0; i < s_num+w_num; i++)
      if(atype[i] == 2){
        w_index[c/(w_site-1)] = i*3;
        w_info[i] = c+w_num+s_num;
        for(j = 0; j < w_site-1; j++)
          w_info[c+w_num+s_num+j] = i;
        c += w_site-1;
      }

    /*
  for(i = 0; i < w_num; i++)
    printf("%d %d\n",i,w_index[i]/3);
  for(i = 0; i < n1; i++){
    printf("%d %d % d\n",i,atype[i],w_info[i]);
  }
  exit(0);
  */
    ice_set2(side);
    /*
      for(i = 0; i < n3; i += 3)
      printf("%d %d %f %f %f\n",i/3,atype[i/3],cd[i],cd[i+1],cd[i+2]);
      exit(0);
    */

  }else if (sys_num == 5){

    side[0] = pow(8 / nden, 1./3.) * np;
    side[1] = side[0];
    side[2] = side[0];

    for(i = 0; i < (n1-w_num*(w_site-1))/2; i++)
      atype[i] = 0;
    for(i = (n1-w_num*(w_site-1))/2; i < n1-w_num*(w_site-1); i++)
      atype[i] = 1;

    fccset2(np,side[0],cd);     /* set fcc */
    for(i = 0; i < (n1-w_num*(w_site-1))*3/2; i++)
      cd[i+(n1-w_num*(w_site-1))*3/2] = cd[i];
    for(i = 0; i < (n1-w_num*(w_site-1))*3/2; i += 3){
      cd[i] += side[0] / np / 2.;
      if(cd[i] < 0)       cd[i] += side[0];
      if(cd[i] > side[0]) cd[i] -= side[0];
    }
    for(i = (n1-w_num*(w_site-1))*3; i < n3; i++)
      cd[i] = 0;

    c = 0;
    while(c < w_num){
      i0 = (int)((double)rand()/(double)RAND_MAX*(s_num+w_num));
      if(atype[i0] == (c%2)){
        atype[i0] = 2;
        for(i = 0; i < w_site-1; i++)
          atype[s_num+w_num+c*(w_site-1)+i] = 3+i;
        c++;
      }
    }
/*
    for(i = 0; i < n3; i += 3)
      printf("%d %d %f %f %f\n",i/3,atype[i/3],cd[i],cd[i+1],cd[i+2]);
    exit(0);
*/
    
    for(i = 0; i < n1; i++)
      w_info[i] = -1;
    c = 0;
    for(i = 0; i < s_num+w_num; i++)
      if(atype[i] == 2){
        w_index[c/(w_site-1)] = i*3;
        w_info[i] = c+w_num+s_num;
        for(j = 0; j < w_site-1; j++)
          w_info[c+w_num+s_num+j] = i;
        c += w_site-1;
      }
/*
    for(i = 0; i < w_num; i++)
      printf("%d %d\n",i,w_index[i]/3);
    for(i = 0; i < n1; i++){
      printf("%d %d % d\n",i,atype[i],w_info[i]);
    }
    exit(0);
*/
    for(i = 0; i < w_num*4; i += 4){
      ang0 = ((double)rand()/(double)RAND_MAX)*360*PI/180;
      ang1 = ((double)rand()/(double)RAND_MAX)*360*PI/180;
      ang2 = ((double)rand()/(double)RAND_MAX)*360*PI/180;
      ang[i  ] = sin(ang1/2)*sin((ang2-ang0)/2);
      ang[i+1] = sin(ang1/2)*cos((ang2-ang0)/2);
      ang[i+2] = cos(ang1/2)*sin((ang2+ang0)/2);
      ang[i+3] = cos(ang1/2)*cos((ang2+ang0)/2);
      angh[i  ] = ang[i  ];
      angh[i+1] = ang[i+1];
      angh[i+2] = ang[i+2];
      angh[i+3] = ang[i+3];
    }


  }else if (sys_num == 6){

    side[0] = pow(8 / nden, 1./3.) * np;
    side[1] = side[0];
    side[2] = side[0];

    for(i = 0; i < (n1-w_num*(w_site-1))/2; i++)
      atype[i] = 0;
    for(i = (n1-w_num*(w_site-1))/2; i < n1-w_num*(w_site-1); i++)
      atype[i] = 1;

    fccset2(np,side[0],cd);     /* set fcc */
    for(i = 0; i < (n1-w_num*(w_site-1))*3/2; i++)
      cd[i+(n1-w_num*(w_site-1))*3/2] = cd[i];
    for(i = 0; i < (n1-w_num*(w_site-1))*3/2; i += 3){
      cd[i] += side[0] / np / 2.;
      if(cd[i] < 0)       cd[i] += side[0];
      if(cd[i] > side[0]) cd[i] -= side[0];
    }
    for(i = (n1-w_num*(w_site-1))*3; i < n3; i++)
      cd[i] = 0;

    d0 = side[0]/(npx*2)*nw;
    d1 = side[1]/(npy*2)*nw;
    d2 = side[2]/(npz*2)*nw;

    for(i = 0; i < n3; i += 3){
      if(cd[i]   < d0 || cd[i]   > side[0]-d0) atype[i/3] = 2;
      if(cd[i+1] < d1 || cd[i+1] > side[1]-d1) atype[i/3] = 2;
      if(cd[i+2] < d2 || cd[i+2] > side[2]-d2) atype[i/3] = 2;
      /*
      printf("%d %d %f %f %f\n",i/3,atype[i/3],cd[i],cd[i+1],cd[i+2]);
      */
    }
    
    for(i = 0; i < n1; i++)
      w_info[i] = -1;
    c = 0;
    for(i = 0; i < s_num+w_num; i++)
      if(atype[i] == 2){
        w_index[c/(w_site-1)] = i*3;
        w_info[i] = c+w_num+s_num;
        for(j = 0; j < w_site-1; j++){
          atype[c+w_num+s_num+j] = 3+j;
          w_info[c+w_num+s_num+j] = i;
        }
        c += w_site-1;
      }
    /*
    for(i = 0; i < w_num; i++)
      printf("%d %d\n",i,w_index[i]/3);
    for(i = 0; i < n1; i++){
      printf("%d %d % d\n",i,atype[i],w_info[i]);
    }
    exit(0);
    */
    for(i = 0; i < w_num*4; i += 4){
      ang0 = ((double)rand()/(double)RAND_MAX)*360*PI/180;
      ang1 = ((double)rand()/(double)RAND_MAX)*360*PI/180;
      ang2 = ((double)rand()/(double)RAND_MAX)*360*PI/180;
      ang[i  ] = sin(ang1/2)*sin((ang2-ang0)/2);
      ang[i+1] = sin(ang1/2)*cos((ang2-ang0)/2);
      ang[i+2] = cos(ang1/2)*sin((ang2+ang0)/2);
      ang[i+3] = cos(ang1/2)*cos((ang2+ang0)/2);
      angh[i  ] = ang[i  ];
      angh[i+1] = ang[i+1];
      angh[i+2] = ang[i+2];
      angh[i+3] = ang[i+3];
    }

  }

#if ZERO_P == 1
  /*
  for(i = 0; i < n3; i += 3){
    cd[i]   += 4.5*side[0];
    cd[i+1] += 4.5*side[1];
    cd[i+2] += 4.5*side[2];
  }
  for(i = 0; i < 3; i++)
  side[i] *= 10;
  */
  for(i = 0; i < n3; i += 3){
    cd[i]   += 2*side[0];
    cd[i+1] += 2*side[1];
    cd[i+2] += 2*side[2];
  }
  for(i = 0; i < 3; i++)
    side[i] *= 5;
#endif

  alpha = oalpha / side[0];
  alpha2 = alpha*alpha;
  ial2si2 = 1. / (alpha*alpha*side[0]*side[0]);

  for(i = 0; i < 3; i++){
    sideh[i] = side[i] *.5;
    iside[i] = 1./side[i];
  }
  /*
  printf("%f %f %f\n",side[0],side[1],side[2]);
  printf("%f %f %f\n",sideh[0],sideh[1],sideh[2]);
  for(i = 0; i < n3; i += 3)
    printf("%d %d %f %f %f\n",i/3,atype[i/3],cd[i],cd[i+1],cd[i+2]);

  d0 = 0; d1 = 0; d2 = 0;
  for(i = 0; i < (w_num+s_num)*3; i += 3){
    d0 += cd[i];
    d1 += cd[i+1];
    d2 += cd[i+2];
  }
  printf("center % f % f % f\n"
         ,d0/(w_num+s_num)
         ,d1/(w_num+s_num)
         ,d2/(w_num+s_num));

  printf("center % f % f % f\n"
         ,d0/(w_num+s_num)-sideh[0]
         ,d1/(w_num+s_num)-sideh[1]
         ,d2/(w_num+s_num)-sideh[2]);

  exit(0);
  */
  for(i = 0; i < KNUM+2; i++)
    atype_num[i] = 0;
  for(i = 0; i < n1; i++){
    atype_num[atype_mat[atype[i]]]++;
  }
  for(i = 0; i < n3; i++)
    vl[i] = 0;
  velset6(rtemp,h,tscale,knum,s_num*3+w_num*3);

  d6 = d7 = d8 = 0;
  for(i = 0; i < ws_num3; i += 3){
    d6 += cd[i];
    d7 += cd[i+1];
    d8 += cd[i+2];
  }
  d6 /= ws_num;
  d7 /= ws_num;
  d8 /= ws_num;
  for(i = 0; i < ws_num3; i += 3){
    cd[i]   -= d6;
    cd[i+1] -= d7;
    cd[i+2] -= d8;
  }
  /*
  d0 = d1 = d2 = 0;
  for(i = 0; i < ws_num3; i += 3){
    d0 += (cd[i+1]*vl[i+2] - cd[i+2]*vl[i+1])*a_mass[atype_mat[atype[i/3]]];
    d1 += (cd[i+2]*vl[i  ] - cd[i  ]*vl[i+2])*a_mass[atype_mat[atype[i/3]]];
    d2 += (cd[i  ]*vl[i+1] - cd[i+1]*vl[i  ])*a_mass[atype_mat[atype[i/3]]];
  }
  printf("% f % f % f\n",d0,d1,d2);
  */

  d3 = d4 = d5 = 0;
  for(i = 0; i < ws_num3; i += 3){ /* calculae moment of inertia */
    d3 += (cd[i+1]*cd[i+1]+cd[i+2]*cd[i+2])*a_mass[atype_mat[atype[i/3]]];
    d4 += (cd[i]  *cd[i]  +cd[i+2]*cd[i+2])*a_mass[atype_mat[atype[i/3]]];
    d5 += (cd[i]  *cd[i]  +cd[i+1]*cd[i+1])*a_mass[atype_mat[atype[i/3]]];
  }
  d0 = d1 = d2 = 0;
  for(i = 0; i < ws_num3; i += 3){ /* calculate angular velocity */
    d0 +=(cd[i+1]*vl[i+2]-cd[i+2]*vl[i+1])*a_mass[atype_mat[atype[i/3]]]/d3;
    d1 +=(cd[i+2]*vl[i  ]-cd[i  ]*vl[i+2])*a_mass[atype_mat[atype[i/3]]]/d4;
    d2 +=(cd[i  ]*vl[i+1]-cd[i+1]*vl[i  ])*a_mass[atype_mat[atype[i/3]]]/d5;
  }
  for(i = 0; i < ws_num3; i += 3){
    vl[i]   -= d1*cd[i+2] - d2*cd[i+1];
    vl[i+1] -= d2*cd[i  ] - d0*cd[i+2];
    vl[i+2] -= d0*cd[i+1] - d1*cd[i  ];
  }
  for(i = 0; i < ws_num3; i += 3){
    cd[i]   += d6;
    cd[i+1] += d7;
    cd[i+2] += d8;
  }

  /*
  d0 = d1 = d2 = 0;
  for(i = 0; i < ws_num3; i += 3){
    d0 += (cd[i+1]*vl[i+2] - cd[i+2]*vl[i+1])*a_mass[atype_mat[atype[i/3]]];
    d1 += (cd[i+2]*vl[i  ] - cd[i  ]*vl[i+2])*a_mass[atype_mat[atype[i/3]]];
    d2 += (cd[i  ]*vl[i+1] - cd[i+1]*vl[i  ])*a_mass[atype_mat[atype[i/3]]];
  }
  printf("% f % f % f\n",d0,d1,d2);
  */

  /*
  d0 = d1 = d2 = 0;
  for(i = 0; i < n3; i += 3){
    printf("%d %d %f %f %f % f % f % f\n",i/3,atype[i/3]
           ,cd[i],cd[i+1],cd[i+2],vl[i],vl[i+1],vl[i+2]);
    d0 += (cd[i+1]*vl[i+2] - cd[i+2]*vl[i+1])*a_mass[atype_mat[atype[i/3]]];
    d1 += (cd[i+2]*vl[i  ] - cd[i  ]*vl[i+2])*a_mass[atype_mat[atype[i/3]]];
    d2 += (cd[i  ]*vl[i+1] - cd[i+1]*vl[i  ])*a_mass[atype_mat[atype[i/3]]];
  }
  printf("% f % f % f\n",d0,d1,d2);

  d0 = d1 = d2 = 0;
  for(i = 0; i < n3; i += 3){
    d0 += vl[i];
    d1 += vl[i+1];
    d2 += vl[i+2];
  }
  printf("% f % f % f\n",d0,d1,d2);
  */

  ekin = 0;
  for(i = 0; i < n3; i += 3){
    ekin += (vl[i  ]*vl[i  ] +
              vl[i+1]*vl[i+1] +
              vl[i+2]*vl[i+2])*a_mass[atype_mat[atype[i/3]]];
  }
  ekin /= hsq;
  mtemp = tscale * ekin;

  d0 = sqrt(rtemp / mtemp);
  for(i = 0; i < n3; i++){
    vl[i] *= d0;
  }
  ekin = 0;
  for(i = 0; i < n3; i += 3){
    ekin += (vl[i  ]*vl[i  ] +
              vl[i+1]*vl[i+1] +
              vl[i+2]*vl[i+2])*a_mass[atype_mat[atype[i/3]]];
  }
  ekin /= hsq;
  mtemp = tscale * ekin;
  /*  printf("%f %f\n",mtemp*epsv/kb,ekin);*/

#if ALPHAC == 1
  avrstart = 1;
  for(i = 0; i < n3; i++)
    vl[i] = 0;
#endif

#if MDM == 0
  vecset();
#endif

#if ALPHAC == 1
  oalpha = 0.0;
#endif

#if MDM == 2

  if(sys_num == 0){
#if ZERO_P  == 1
    strcpy(f_table_name,"table/fncl_af.table");
    strcpy(p_table_name,"table/fncl_ap.table");
    side_min = 3.5*side[0];
    side_max = 6.5*side[0];
    i0 = KNUM+4;
    for(i = 0; i < i0*i0; i++){
      gscale[i] = 0;
      rscale[i] = 0;
    }
    gscale[0*i0+0] = 1;
    gscale[0*i0+1] = 1;
    gscale[1*i0+0] = 1;
    gscale[1*i0+1] = 1;
    rscale[0*i0+0] = 1;
    rscale[0*i0+1] = pow(2,21);
    rscale[1*i0+0] = pow(2,21);
    rscale[1*i0+1] = pow(2,42);
#endif
  } else if(sys_num == 1 || sys_num == 2 || sys_num == 3 || sys_num == 10){
#if SPC == 1
    strcpy(f_table_name,"table/fspcl_af.table");
    strcpy(p_table_name,"table/fspcl_ap.table");
    side_min = 0;
    side_max = my_max(side[0],my_max(side[1],side[2]));
    i0 = KNUM+4;
    for(i = 0; i < i0*i0; i++){
      gscale[i] = 0;
      rscale[i] = 0;
    }
    gscale[2*i0+2] = 1;
    gscale[2*i0+3] = 1;
    gscale[2*i0+4] = 1;
    gscale[2*i0+8] = 1;
    gscale[3*i0+2] = 1;
    gscale[3*i0+3] = 1;
    gscale[3*i0+4] = 1;
    gscale[3*i0+8] = 1;
    gscale[4*i0+2] = 1;
    gscale[4*i0+3] = 1;
    gscale[4*i0+4] = 1;
    gscale[4*i0+8] = 1;
    gscale[8*i0+2] = 1;
    gscale[8*i0+3] = 1;
    gscale[8*i0+4] = 1;
    gscale[8*i0+8] = 1;

    rscale[2*i0+2] = 1;
    rscale[2*i0+3] = pow(2,21);
    rscale[2*i0+4] = pow(2,21);
    rscale[2*i0+8] = 1;

    rscale[3*i0+2] = pow(2,21);
    rscale[3*i0+3] = pow(2,42);
    rscale[3*i0+4] = pow(2,42);
    rscale[3*i0+8] = pow(2,21);

    rscale[4*i0+2] = pow(2,21);
    rscale[4*i0+3] = pow(2,42);
    rscale[4*i0+4] = pow(2,42);
    rscale[4*i0+8] = pow(2,21);

    rscale[8*i0+2] = pow(2,21);
    rscale[8*i0+3] = pow(2,42);
    rscale[8*i0+4] = pow(2,42);
    rscale[8*i0+8] = pow(2,21);
#elif ST2 == 1
    strcpy(f_table_name,"table/fx_ljclf.table");
    strcpy(p_table_name,"table/fx_ljclp.table");

    side_min = 0;
    side_max = my_max(side[0],my_max(side[1],side[2]));

    i10 = KNUM+4;

    for(i = 0; i < i10; i++)
      for(j = 0; j < i10; j++){
        i0 = atype_mat[i];
        i1 = atype_mat[j];
        charge[i*i10+j] = z[i0]*z[i1];
        roffset[i*10+j] = pow(2,32);
      }

    for(i = 0; i < i10; i++)
      for(j = 0; j < i10; j++){
        d0 = wpa;
        d1 = wpc;
        if(d0 != 0){
          gscale[i*i10+j] = d1*d1/d0;
          rscale[i*i10+j] = pow(d1/d0,1.0/3.0);
        } else {
          gscale[i*i10+j] = 0;
          rscale[i*i10+j] = 0;
        }
      }

    for(i = 0; i < i10; i++)
      for(j = 0; j < i10; j++){
        d0 = wpa;
        d1 = wpc;
        if(d0 != 0){
          gscale2[i*i10+j] = 6.0*d1*pow(d1/d0,4.0/3.0);
          rscale2[i*i10+j] = pow(d1/d0,1.0/3.0);
        } else {
          gscale2[i*i10+j] = 0;
          rscale2[i*i10+j] = 0;
        }
      }
#elif TIP5P == 1
    strcpy(f_table_name,"table/fx_ljclf.table");
    strcpy(p_table_name,"table/fx_ljclp.table");

    side_min = 0;
    side_max = my_max(side[0],my_max(side[1],side[2]));

    i10 = KNUM+4;

    for(i = 0; i < i10; i++)
      for(j = 0; j < i10; j++){
        i0 = atype_mat[i];
        i1 = atype_mat[j];
        charge[i*i10+j] = z[i0]*z[i1];
        roffset[i*i10+j] = pow(2,32);
      }

    for(i = 0; i < i10; i++)
      for(j = 0; j < i10; j++){
        d0 = wpa;
        d1 = wpc;
        if(d0 != 0){
          gscale[i*i10+j] = d1*d1/d0;
          rscale[i*i10+j] = pow(d1/d0,1.0/3.0);
        } else {
          gscale[i*i10+j] = 0;
          rscale[i*i10+j] = 0;
        }
      }

    for(i = 0; i < i10; i++)
      for(j = 0; j < i10; j++){
        d0 = wpa;
        d1 = wpc;
        if(d0 != 0){
          gscale2[i*i10+j] = 6.0*d1*pow(d1/d0,4.0/3.0);
          rscale2[i*i10+j] = pow(d1/d0,1.0/3.0);
        } else {
          gscale2[i*i10+j] = 0;
          rscale2[i*i10+j] = 0;
        }
      }
#endif

  } else if(sys_num >= 4){

    strcpy(f_table_name,"table/fx_ljclf.table");
    strcpy(p_table_name,"table/fx_ljclp.table");

    side_min = 0;
    side_max = my_max(side[0],my_max(side[1],side[2]));

    i10 = KNUM+4;

    for(i = 0; i < i10; i++)
      for(j = 0; j < i10; j++){
        i0 = (i == 4 ? 3:i);
        i1 = (j == 4 ? 3:j);
        charge[i*i10+j] = z[i0]*z[i1];
        roffset[i*i10+j] = pow(2,32);
      }

    for(i = 0; i < i10; i++)
      for(j = 0; j < i10; j++){
	if(i <= 4 && j <= 4){
	  i0 = (i == 4 ? 3:i);
	  i1 = (j == 4 ? 3:j);
	  d0 = as_a[i0][i1];
	  d1 = as_c[i0][i1];
	  if(d0 != 0){
	    gscale[i*i10+j] = d1*d1/d0;
	    rscale[i*i10+j] = pow(d1/d0,1.0/3.0);
	  } else {
	    gscale[i*i10+j] = 0;
	    rscale[i*i10+j] = 0;
	  }
        } else {
          gscale[i*i10+j] = 0;
          rscale[i*i10+j] = 0;
        }
      }

    for(i = 0; i < i10; i++)
      for(j = 0; j < i10; j++){
	if(i <= 4 && j <= 4){
	  i0 = (i == 4 ? 3:i);
	  i1 = (j == 4 ? 3:j);
	  d0 = as_a[i0][i1];
	  d1 = as_c[i0][i1];
	  if(d0 != 0){
	    gscale2[i*i10+j] = 6.0*d1*pow(d1/d0,4.0/3.0);
	    rscale2[i*i10+j] = pow(d1/d0,1.0/3.0);
	  } else {
	    gscale2[i*i10+j] = 0;
	    rscale2[i*i10+j] = 0;
	  }
        } else {
          gscale2[i*i10+j] = 0;
          rscale2[i*i10+j] = 0;
        }
      }
    /*
    exit(0);
	  for(i = 0;i < KNUM+4; i++){
	    for(j = 0;j < KNUM+4; j++){
	      printf("%d %d % f % f\n",i,j
		     ,rscale2[i*(KNUM+4)+j],gscale2[i*(KNUM+4)+j]);
	    }
	  }
	  exit(0);
*/
  }
#ifndef VTGRAPE
  if(ini_m2 == 1 && sc_flg != 2){
    /*    printf("before m2_allocate_unit\n");*/
    mu = m2_allocate_unit(f_table_name,M2_FORCE,side_min,side_max,NULL_INT);
    m2_set_charge_matrix(mu, gscale, KNUM+4, KNUM+4);
    m2_set_rscale_matrix(mu, rscale, KNUM+4, KNUM+4);
  }
#endif
#endif

#if T_CONST == 1
  xs = 0;
  /*  lq = rtemp*(n3-3)*1e-2;*/
#endif

  phir_corr = 0;
  i0 = 2; i1 = 3; r = bond[0];
/*    phir_corr += (2.*zz[i0][i1]*erfc(alpha * r)/ r)*w_num;*/
  phir_corr -= (2.*zz[i0][i1]/ r)*w_num;

  i0 = 3; i1 = 3; r = bond[0]*sin(hoh_deg/2./180.*PI)*2;
/*    phir_corr += (zz[i0][i1]*erfc(alpha * r)/ r)*w_num;*/
  phir_corr -= (zz[i0][i1]/ r)*w_num;

  for(i0 = 0; i0 < w_num; i0++){
    i = w_index[i0];
    j = i0*4;
    c = w_info[i/3]*3;
    ang0 = ang[j  ];
    ang1 = ang[j+1];
    ang2 = ang[j+2];
    ang3 = ang[j+3];
    for(k = 0; k < w_site-1; k++){
      d0 = m_cdx[k]*(-ang0*ang0+ang1*ang1-ang2*ang2+ang3*ang3)
          +m_cdy[k]*(-2)*(ang0*ang1+ang2*ang3)
          +m_cdz[k]*( 2)*(ang1*ang2-ang0*ang3);
      d1 = m_cdx[k]*  2 *(ang2*ang3-ang0*ang1)
          +m_cdy[k]*( ang0*ang0-ang1*ang1-ang2*ang2+ang3*ang3)
          +m_cdz[k]*(-2)*(ang0*ang2+ang1*ang3);
      d2 = m_cdx[k]*  2 *(ang1*ang2+ang0*ang3)
          +m_cdy[k]*  2 *(ang1*ang3-ang0*ang2)
          +m_cdz[k]*(-ang0*ang0-ang1*ang1+ang2*ang2+ang3*ang3);
      cd[k*3+c  ] = cd[i  ] + d0;
      cd[k*3+c+1] = cd[i+1] + d1;
      cd[k*3+c+2] = cd[i+2] + d2;
    }
  }

  d0 = my_max(sideh[0],my_max(sideh[1],sideh[2]))/10;
  eye_len = d0/tan(15./180.*PI)*1.6*2;
  ini_temp = temp;

  for(i = 0; i < w_num3; i++){
    agv[i] = 0;
    agvh[i] = 0;
    trq[i] = 0;
  }
  for(i = 0; i <n3; i++){
    fc[i] = 0;
  }
  /*
  for(i = 0; i < n3; i+=3){
    printf("%d %d %f %f %f\n",i/3,atype[i/3],cd[i],cd[i+1],cd[i+2]);
  }
  */
}
void md_run()
{
  int i,j,k,c;
  int i0,i1,i2,i3,i4,i5;
  double d0,d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12;

  double agv0,agv1,agv2;
  double ang0,ang1,ang2,ang3;
  int md_loop;

  double dphir;


  for(md_loop = 0; md_loop < md_step; md_loop++){

    m_clock++;

#ifdef C_MASS
    for(i = 0; i < n3; i += 3){
      if(atype[i/3] == 2){
	ang0 = ang[w_rindex[i/3]*4];
	ang1 = ang[w_rindex[i/3]*4+1];
	ang2 = ang[w_rindex[i/3]*4+2];
	ang3 = ang[w_rindex[i/3]*4+3];

	d0 = center_mass*(-2)*(ang0*ang1+ang2*ang3);
	d1 = center_mass*( ang0*ang0-ang1*ang1-ang2*ang2+ang3*ang3);
	d2 = center_mass*  2 *(ang1*ang3-ang0*ang2);

	cd[i]   -= d0;
	cd[i+1] -= d1;
	cd[i+2] -= d2;

      }
    }
#endif

      for(i = 0; i < n3; i++){ /* update coordinations */
        if(atype[i/3] <= 2 && atype[i/3] != 8){
#if T_CONST == 1
          vl[i] = (vl[i]*(1-xs)+fc[i])/(1+xs);
#else
          vl[i]   += fc[i];
#endif
          cd[i]   += vl[i];
        }
      }
      /*
      for(i = 0; i < ws_num3; i += 3){
        printf("%2d %d %f %f %f % f % f % f\n",i/3,atype[i/3]
               ,cd[i],cd[i+1],cd[i+2]
               ,vl[i],vl[i+1],vl[i+2]);
      }
      */
      /*
      for(i = 0; i < n3; i += 3){
        printf("%2d %d %f %f %f % f % f % f\n",i/3,atype[i/3]
               ,cd[i],cd[i+1],cd[i+2]
               ,vl[i],vl[i+1],vl[i+2]);
      }
      exit(0);
      */
      /*
      d3 = d4 = d5 = 0;
      for(i0 = 0; i0 < 3; i0++){
        d0 = 0; d1 = 0; d2 = 0;
        for(i = 0; i < n3; i += 3){
          if(atype[i/3] == i0){
            d0 += vl[i]  *a_mass[atype_mat[atype[i/3]]];
            d1 += vl[i+1]*a_mass[atype_mat[atype[i/3]]];
            d2 += vl[i+2]*a_mass[atype_mat[atype[i/3]]];
          }
        }
        printf("%d % f % f % f\n",i0,d0,d1,d2);
        d3 += d0;
        d4 += d1;
        d5 += d2;
      }
      printf("total % f % f % f\n",d3,d4,d5);
      d0 = 0; d1 = 0; d2 = 0;
      d3 = d4 = d5 = 0;
      for(i = 0; i < (w_num+s_num)*3; i += 3){
        d0 += cd[i];
        d1 += cd[i+1];
        d2 += cd[i+2];
        if(atype[i/3] == 2){
          d3 += fc[i]  *(a_mass[2]+2*a_mass[3]);
          d4 += fc[i+1]*(a_mass[2]+2*a_mass[3]);
          d5 += fc[i+2]*(a_mass[2]+2*a_mass[3]);
        } else if(atype[i/3] == 0 || atype[i/3] == 1){
          d3 += fc[i]*a_mass[atype_mat[atype[i/3]]];
          d4 += fc[i+1]*a_mass[atype_mat[atype[i/3]]];
          d5 += fc[i+2]*a_mass[atype_mat[atype[i/3]]];
        }
      }
      printf("center % f % f % f  % e % e % e\n"
             ,d0/(w_num+s_num)-sideh[0]
             ,d1/(w_num+s_num)-sideh[1]
             ,d2/(w_num+s_num)-sideh[2]
             ,d3/hsq,d4/hsq,d5/hsq
             );
      */
      for(i = 0; i < w_num3; i++){ /* half update anglusr velocity  */
        agv[i]   = agvh[i]   + trq[i]  *0.5;
      }
      for(i=0, j=0; i < w_num3; i+=3, j+=4){ /* coordination transformation */
        agv0 = agv[i  ];
        agv1 = agv[i+1];
        agv2 = agv[i+2];
        
        ang0 = ang[j  ];
        ang1 = ang[j+1];
        ang2 = ang[j+2];
        ang3 = ang[j+3];
        agvp[i  ] =(agv0     *(-ang0*ang0+ang1*ang1-ang2*ang2+ang3*ang3)
                    +agv1*  2 *( ang2*ang3-ang0*ang1)
                    +agv2*  2 *( ang1*ang2+ang0*ang3)) /moi[0];
        agvp[i+1] =(agv0*(-2)*( ang0*ang1+ang2*ang3)
                    +agv1     *( ang0*ang0-ang1*ang1-ang2*ang2+ang3*ang3)
                    +agv2*  2 *( ang1*ang3-ang0*ang2)) /moi[1];
        agvp[i+2] =(agv0*  2 *( ang1*ang2-ang0*ang3)
                    +agv1*(-2)*( ang0*ang2+ang1*ang3)
                    +agv2     *(-ang0*ang0-ang1*ang1+ang2*ang2+ang3*ang3))/moi[2];
      }
      for(i=0, j=0; i < w_num3; i+=3, j+=4){ /* half update of angle */
        ang0 = ang[j  ];
        ang1 = ang[j+1];
        ang2 = ang[j+2];
        ang3 = ang[j+3];
        angh[j]   = ang0+(-ang2*agvp[i]-ang3*agvp[i+1]+ang1*agvp[i+2])*0.25;
        angh[j+1] = ang1+( ang3*agvp[i]-ang2*agvp[i+1]-ang0*agvp[i+2])*0.25;
        angh[j+2] = ang2+( ang0*agvp[i]+ang1*agvp[i+1]+ang3*agvp[i+2])*0.25;
        angh[j+3] = ang3+(-ang1*agvp[i]+ang0*agvp[i+1]-ang2*agvp[i+2])*0.25;
      }
      
      for(i = 0; i < w_num*4; i += 4){ /* correction of angle */
        ang0 = angh[i  ];
        ang1 = angh[i+1];
        ang2 = angh[i+2];
        ang3 = angh[i+3];
        angh[i]  /= sqrt(ang0*ang0+ang1*ang1+ang2*ang2+ang3*ang3);
        angh[i+1]/= sqrt(ang0*ang0+ang1*ang1+ang2*ang2+ang3*ang3);
        angh[i+2]/= sqrt(ang0*ang0+ang1*ang1+ang2*ang2+ang3*ang3);
        angh[i+3]/= sqrt(ang0*ang0+ang1*ang1+ang2*ang2+ang3*ang3);
      }
      
      for(i = 0; i < w_num3; i++){ /* update of anglular velocity */
#if T_CONST == 1
        agvh[i] = (agvh[i]*(1-xs)+trq[i])/(1+xs);
#else
        agvh[i]   += trq[i];
#endif
      }
      
      for(i=0, j=0; i < w_num3; i+=3, j+=4){ /* coordination transformation */
        agv0 = agvh[i  ];
        agv1 = agvh[i+1];
        agv2 = agvh[i+2];
        ang0 = angh[j  ];
        ang1 = angh[j+1];
        ang2 = angh[j+2];
        ang3 = angh[j+3];
        agvph[i  ]=(agv0     *(-ang0*ang0+ang1*ang1-ang2*ang2+ang3*ang3)
                    +agv1*  2 *( ang2*ang3-ang0*ang1)
                    +agv2*  2 *( ang1*ang2+ang0*ang3)) /moi[0];
        agvph[i+1]=(agv0*(-2)*( ang0*ang1+ang2*ang3)
                    +agv1     *( ang0*ang0-ang1*ang1-ang2*ang2+ang3*ang3)
                    +agv2*  2 *( ang1*ang3-ang0*ang2)) /moi[1];
        agvph[i+2]=(agv0*  2 *( ang1*ang2-ang0*ang3)
                    +agv1*(-2)*( ang0*ang2+ang1*ang3)
                    +agv2     *(-ang0*ang0-ang1*ang1+ang2*ang2+ang3*ang3))/moi[2];
      }
      for(i=0, j=0; i < w_num3; i+=3, j+=4){ /* update of angle */
        ang0 = angh[j  ];
        ang1 = angh[j+1];
        ang2 = angh[j+2];
        ang3 = angh[j+3];
        ang[j]  += (-ang2*agvph[i] - ang3*agvph[i+1] + ang1*agvph[i+2])*0.5;
        ang[j+1]+= ( ang3*agvph[i] - ang2*agvph[i+1] - ang0*agvph[i+2])*0.5;
        ang[j+2]+= ( ang0*agvph[i] + ang1*agvph[i+1] + ang3*agvph[i+2])*0.5;
        ang[j+3]+= (-ang1*agvph[i] + ang0*agvph[i+1] - ang2*agvph[i+2])*0.5;
      }
      for(i = 0; i < w_num*4; i += 4){ /* correction of angle */
        ang0 = ang[i  ];
        ang1 = ang[i+1];
        ang2 = ang[i+2];
        ang3 = ang[i+3];
        ang[i]  /= sqrt(ang0*ang0+ang1*ang1+ang2*ang2+ang3*ang3);
        ang[i+1]/= sqrt(ang0*ang0+ang1*ang1+ang2*ang2+ang3*ang3);
        ang[i+2]/= sqrt(ang0*ang0+ang1*ang1+ang2*ang2+ang3*ang3);
        ang[i+3]/= sqrt(ang0*ang0+ang1*ang1+ang2*ang2+ang3*ang3);
      }
      
#ifdef C_MASS
    for(i = 0; i < n3; i += 3){
      if(atype[i/3] == 2){
	ang0 = ang[w_rindex[i/3]*4];
	ang1 = ang[w_rindex[i/3]*4+1];
	ang2 = ang[w_rindex[i/3]*4+2];
	ang3 = ang[w_rindex[i/3]*4+3];

	d0 = center_mass*(-2)*(ang0*ang1+ang2*ang3);
	d1 = center_mass*( ang0*ang0-ang1*ang1-ang2*ang2+ang3*ang3);
	d2 = center_mass*  2 *(ang1*ang3-ang0*ang2);

	cd[i]   += d0;
	cd[i+1] += d1;
	cd[i+2] += d2;

      }
    }
#endif

      for(i0 = 0; i0 < w_num; i0++){
        i = w_index[i0];
        j = i0*4;
        c = w_info[i/3]*3;
        ang0 = ang[j  ];
        ang1 = ang[j+1];
        ang2 = ang[j+2];
        ang3 = ang[j+3];
        /*      printf("%d % f % f % f % f\n",j/4,ang0,ang1,ang2,ang3);*/
        for(k = 0; k < w_site-1; k++){
          d0 = m_cdx[k]*(-ang0*ang0+ang1*ang1-ang2*ang2+ang3*ang3)
              +m_cdy[k]*(-2)*(ang0*ang1+ang2*ang3)
              +m_cdz[k]*( 2)*(ang1*ang2-ang0*ang3);
          d1 = m_cdx[k]*  2 *(ang2*ang3-ang0*ang1)
              +m_cdy[k]*( ang0*ang0-ang1*ang1-ang2*ang2+ang3*ang3)
              +m_cdz[k]*(-2)*(ang0*ang2+ang1*ang3);
          d2 = m_cdx[k]*  2 *(ang1*ang2+ang0*ang3)
              +m_cdy[k]*  2 *(ang1*ang3-ang0*ang2)
              +m_cdz[k]*(-ang0*ang0-ang1*ang1+ang2*ang2+ang3*ang3);
          cd[k*3+c  ] = cd[i  ] + d0;
          cd[k*3+c+1] = cd[i+1] + d1;
          cd[k*3+c+2] = cd[i+2] + d2;
        }
      }
#if ZERO_P == 0
      for(i = 0; i < n3; i += 3){
        if (cd[i]   < 0)       cd[i]   += side[0];
        if (cd[i]   > side[0]) cd[i]   -= side[0];
        if (cd[i+1] < 0)       cd[i+1] += side[1];
        if (cd[i+1] > side[1]) cd[i+1] -= side[1];
        if (cd[i+2] < 0)       cd[i+2] += side[2];
        if (cd[i+2] > side[2]) cd[i+2] -= side[2];
      }
#else
      for(i = 0; i < n3; i += 3){
        if(atype[i/3] <= 2){
#if 1
          if (cd[i]   < 0 || cd[i]   > side[0]) vl[i] *= -1;
          if (cd[i+1] < 0 || cd[i+1] > side[1]) vl[i+1] *= -1;
          if (cd[i+2] < 0 || cd[i+2] > side[2]) vl[i+2] *= -1;
#else
	  int revflag[3]={0,0,0},revcount=0,r;
          if (cd[i]   < 0 || cd[i]   >= side[0]){
	    vl[i] *= -1;
	    revflag[0]=1;
	  }
          if (cd[i+1] < 0 || cd[i+1] >= side[1]){
	    vl[i+1] *= -1;
	    revflag[1]=1;
	  }
          if (cd[i+2] < 0 || cd[i+2] >= side[2]){
	    vl[i+2] *= -1;
	    revflag[2]=1;
	  }
	  for(r=0;r<3;r++) if(revflag[r]!=0) revcount++;

	  //	  if(i==0) printf("\n");
	  {
	    static int i_bak=-1;
	    if(revcount>=2 /*|| i==i_bak*/){
	      printf("i=%d revc=%d revf=%d %d %d cd=%e %e %e vl=%e %e %e\n",
		     i/3,revcount,revflag[0],revflag[1],revflag[2],cd[i],cd[i+1],cd[i+2],vl[i],vl[i+1],vl[i+2]);
	      //	    printf("particle %d revcount=%d revflag=%d %d %d cd=%9.3e %9.3e %9.3e vl=%9.3e %9.3e %9.3e\n",
	      //		   i/3,revcount,revflag[0],revflag[1],revflag[2],cd[i],cd[i+1],cd[i+2],vl[i],vl[i+1],vl[i+2]);
	      i_bak=i;
	    }
	  }
	  if(fc[i]*fc[i]+fc[i+1]*fc[i+1]+fc[i+2]*fc[i+2]>1000.0f){
	    printf(" too-large-force i=%d fc=%e %e %e\n",i/3,fc[i],fc[i+1],fc[i+2]);
	  }
#endif
        }
	else{
	  printf("atye[%d] is %d\n",i/3,atype[i/3]);
	}
      }
#endif
      /*****************  calculation of force  *********************/
      
#ifdef LAP_TIME
#if defined(_WIN32) && !defined(__CYGWIN__)
    md_time0 = (double)timeGetTime()/1000.;
#elif defined(MAC)
    md_time0 = (double)clock()/60.;
#else
    gettimeofday(&time_v,NULL);
    md_time0 = (time_v.tv_sec + time_v.tv_usec / 1000000.0);
#endif
#endif

      for(i = 0;i < 2; i++){
        phi[i] = 0;
      }
      phir = 0;
      for(i = 0;i < n3; i++){
        fc[i] = 0;
        iphi[i] = 0;
      }
      vir = 0;

      /*#if MDM == 0*/
      for(i = 0;i < s_num3; i++){
        fc[i] = 0;
      }

      if(grape_flg == 1){
#if MDM == 2 && !defined(VTGRAPE)
	if(sys_num >= 4 && sys_num < 10){
	  if(pcun == 0){
	  } else {

	    m2_set_types(mu, atype, n1);
	    m2_set_positions(mu, (double(*)[3])cd, n1);
	    m2_set_pipeline_types(mu, atype, n1);
	    m2_set_charge_matrix(mu, charge, KNUM+4, KNUM+4);
	    m2_set_rscale_matrix(mu, roffset, KNUM+4, KNUM+4);
	    printf("before calc force\n");
	    m2_calculate_forces(mu, (double(*)[3])cd, n1, (double(*)[3])fc);
	    printf("after calc force\n");

	    m2_set_pipeline_types(mu, atype, n1);
	    m2_set_charge_matrix(mu, gscale2, KNUM+4, KNUM+4);
	    m2_set_rscale_matrix(mu, rscale2, KNUM+4, KNUM+4);
	    m2_calculate_forces(mu,(double(*)[3])cd,n1,(double(*)[3])fcc);
	    for(i = 0; i < n3; i++)
	      fc[i] += fcc[i];
	  }
	} else {
	  if(pcun == 0){
	  } else {

#if SPC == 1
	    m2_set_types(mu, atype, n1);
	    m2_set_positions(mu, (double(*)[3])cd, n1);
	    m2_set_pipeline_types(mu, atype, n1);
	    m2_calculate_forces(mu, (double(*)[3])cd, n1, (double(*)[3])fc);
#elif ST2 == 1 || TIP5P == 1

	    if(sys_num == 10){
	      m2_set_types(mu, atype, n1);
	      m2_set_positions(mu, (double(*)[3])cd, n1);
	      m2_set_pipeline_types(mu, atype, n1);
	      m2_set_charge_matrix(mu, charge, 9, 9);
	      m2_set_rscale_matrix(mu, roffset, 9, 9);
	      m2_calculate_forces(mu, (double(*)[3])cd, n1, (double(*)[3])fc);

	      m2_set_types(mu, atype, w_num+s_num);
	      m2_set_positions(mu, (double(*)[3])cd, w_num+s_num);
	      m2_set_charge_matrix(mu, gscale2, 9, 9);
	      m2_set_rscale_matrix(mu, rscale2, 9, 9);
	      m2_set_pipeline_types(mu, atype, w_num+s_num);
	      m2_calculate_forces(mu,(double(*)[3])cd,w_num+s_num,(double(*)[3])fcc);
	      for(i = 0; i < w_num3+s_num3; i++)
		fc[i] += fcc[i];
	    } else if(sys_num == 0){

#if 1
	      /*
	      gettimeofday(&time_v,NULL);
	      d0 = (time_v.tv_sec + time_v.tv_usec / 1000000.0);
	      */

#if defined(MDGRAPE3) || defined(VTGRAPE)
	      //	      printf("m2_set_neighbor_ratius is not supported\n");
#else
	      if(bond_flg == 1)
		m2_set_neighbor_radius(mu, neighbor_radius);
	      else
		m2_set_neighbor_radius(mu, 0.0);
#endif
#endif
	      /*
	      gettimeofday(&time_v,NULL);
	      d1 = (time_v.tv_sec + time_v.tv_usec / 1000000.0);
	      */

	      m2_set_types(mu, atype, n1);
	      m2_set_positions(mu, (double(*)[3])cd, n1);
	      m2_set_pipeline_types(mu, atype, n1);
	      m2_calculate_forces(mu, (double(*)[3])cd, n1, (double(*)[3])fc);

#if 1
	      /*
	      gettimeofday(&time_v,NULL);
	      d2 = (time_v.tv_sec + time_v.tv_usec / 1000000.0);
	      */

	      if(bond_flg == 1){
#if defined(MDGRAPE3) || defined(VTGRAPE)
		//		fprintf(stderr,"m2_get_neighbors is not supported\n");
		for(i=0;i<n1;i++) nli[i]=0;
#else
		nig = m2_get_neighbors(mu,nli,n1);
	      /*	      
	      for(i = 0; i < n1; i++){
		nig_num[i] = nli[i] - (i == 0 ? 0:nli[i-1])-1;
		for(j = 1; j < nli[i] - (i == 0 ? 0:nli[i-1]); j++)
		  nig_data[i][j-1]=nig[(i==0?0:nli[i-1])+j];
	      }
	      */

		for(i = 0; i < n1; i++)
		  nig_num[i] = 0;
		i0 = 0;
		for(i = 0; i < n1; i++){
		  /*
		  printf("%d %d %d\n",i,nli[i]-i0,nli[i]);
		  */
		  for(j = 0; j < nli[i]-i0; j++){
		    /*
		    if(j < 6 && i != nig[i0+j])
		      printf("(%d %d %d)o ",nig[i0+j],j,nig_num[i]);
		    else
		      printf("(%d %d %d)x ",nig[i0+j],j,nig_num[i]);
		    */
		    if(nig_num[i] < 6 && i != nig[i0+j]){
		      nig_data[i*6+nig_num[i]] = nig[i0+j];
		      nig_num[i]++;
		    }
		  }
		  /*
		  printf("\n");
		  */
		  i0 = nli[i];
		}
		/*
		i0 = 0;
		for(i = 0; i < n1; i++){
		  if(nli[i] - i0 -1 <= 6)
		    nig_num[i] = nli[i] - i0 -1;
		  for(j = i0+1; j < nli[i]; j++)
		    if(j-i0-1 <= 6)
		      nig_data[i*6+j-i0-1]=nig[j];
		  i0 = nli[i];
		}
		*/
		free(nig);
#endif
	      }
#endif
	      /*
	      gettimeofday(&time_v,NULL);
	      d3 = (time_v.tv_sec + time_v.tv_usec / 1000000.0);
	      printf("%d %f %f %f  %f\n",bond_flg,d1-d0,d2-d1,d3-d2,d3-d0);
	      */

	    } else {
	      m2_set_types(mu, atype, n1);
	      m2_set_positions(mu, (double(*)[3])cd, n1);
	      m2_set_pipeline_types(mu, atype, n1);
	      m2_set_charge_matrix(mu, charge, 9, 9);
	      m2_set_rscale_matrix(mu, roffset, 9, 9);
	      m2_calculate_forces(mu, (double(*)[3])cd, n1, (double(*)[3])fc);

	      m2_set_types(mu, atype, w_num);
	      m2_set_positions(mu, (double(*)[3])cd, w_num);
	      m2_set_charge_matrix(mu, gscale2, 9, 9);
	      m2_set_rscale_matrix(mu, rscale2, 9, 9);
	      m2_set_pipeline_types(mu, atype, w_num);
	      m2_calculate_forces(mu,(double(*)[3])cd,w_num,(double(*)[3])fcc);

	      for(i = 0; i < w_num3; i++)
		fc[i] += fcc[i];
	    }
#endif
	  }
	}

	for(i0 = 0; i0 < w_num; i0++){
	  i = w_index[i0];
	  for(i1 = 0; i1 < w_site-1; i1++){
	    j = (w_info[i/3]+i1)*3;
	    fc[i]   += fc[j];
	    fc[i+1] += fc[j+1];
	    fc[i+2] += fc[j+2];
	  }
	}
#elif MDM == 2 && defined(VTGRAPE)
	double zz2[2][2],center[3];
	int ii,jj;
	static n3_bak=0;
	if(n3!=n3_bak){
#if 1
	  if(n3_bak!=0) MR3free();
	  MR3init();
#else
	  if(n3_bak==0) MR3init();
#endif
	  n3_bak=n3;
	}
	for(ii=0;ii<2;ii++) for(jj=0;jj<2;jj++) zz2[ii][jj]=zz[ii][jj];
	MR3calcnacl(cd,n3/3,atype,2,(double *)pol,(double *)sigm,
		    (double *)ipotro,(double *)pc,(double *)pd,
		    (double *)zz2,8,
		    side[0],
		    //		    side[0]*2.0,
		    0,fc);

#if 0 // find large force
	{
	  int i,j,flag=0,targeti=-1,targeti2=-1;
	  static double *fcbak=NULL,*cdbak=NULL,*vlbak=NULL;
	  if(fcbak==NULL && (fcbak=(double *)malloc(sizeof(double)*10000*3))==NULL){
	    fprintf(stderr,"** error : can't malloc fcbak **\n");
	    exit(1);
	  }
	  if(cdbak==NULL && (cdbak=(double *)malloc(sizeof(double)*10000*3))==NULL){
	    fprintf(stderr,"** error : can't malloc cdbak **\n");
	    exit(1);
	  }
	  if(vlbak==NULL && (vlbak=(double *)malloc(sizeof(double)*10000*3))==NULL){
	    fprintf(stderr,"** error : can't malloc vlbak **\n");
	    exit(1);
	  }
	  for(i=0;i<n3/3;i++){
	    if(fc[i*3]*fc[i*3]+fc[i*3+1]*fc[i*3+1]+fc[i*3+2]*fc[i*3+2]>1000.0){
	      printf(" cras.c: too-large-force i=%d fc=%e %e %e\n",i,fc[i*3],fc[i*3+1],fc[i*3+2]);
	      printf("                             fcbk=%e %e %e\n",fcbak[i*3],fcbak[i*3+1],fcbak[i*3+2]);
	      printf("         cd  =%e %e %e vl  =%e %e %e\n",cd[i*3],cd[i*3+1],cd[i*3+2],vl[i*3],vl[i*3+1],vl[i*3+2]);
	      printf("         cdbk=%e %e %e vlbk=%e %e %e\n",cdbak[i*3],cdbak[i*3+1],cdbak[i*3+2],vlbak[i*3],vlbak[i*3+1],vlbak[i*3+2]);
	      flag=1;
	      if(targeti>=0){
		targeti2=i;
		printf("         distance target1=%d and %d is %e, bk=%e\n",targeti,targeti2,
		       sqrt((cd[targeti*3]-cd[targeti2*3])*(cd[targeti*3]-cd[targeti2*3])+
			    (cd[targeti*3+1]-cd[targeti2*3+1])*(cd[targeti*3+1]-cd[targeti2*3+1])+
			    (cd[targeti*3+2]-cd[targeti2*3+2])*(cd[targeti*3+2]-cd[targeti2*3+2])),
		       sqrt((cdbak[targeti*3]-cdbak[targeti2*3])*(cdbak[targeti*3]-cdbak[targeti2*3])+
			    (cdbak[targeti*3+1]-cdbak[targeti2*3+1])*(cdbak[targeti*3+1]-cdbak[targeti2*3+1])+
			    (cdbak[targeti*3+2]-cdbak[targeti2*3+2])*(cdbak[targeti*3+2]-cdbak[targeti2*3+2])));
	      }
	      else           targeti=i;
	    }
	  }
	  memcpy(fcbak,fc,sizeof(double)*n3);
	  memcpy(cdbak,cd,sizeof(double)*n3);
	  memcpy(vlbak,vl,sizeof(double)*n3);
	  if(flag){
	    m3_set_debugflag(m3_get_unit(),flag,targeti,targeti2,0);
	    bzero(fc,sizeof(double)*n3);
	    MR3calcnacl(cd,n3/3,atype,2,(double *)pol,(double *)sigm,
			(double *)ipotro,(double *)pc,(double *)pd,
			(double *)zz2,8,
			side[0],
			0,fc);
	    printf(" targeti=%d cd=%e %e %e\n",targeti,cd[targeti*3],cd[targeti*3+1],cd[targeti*3+2]);
	    printf(" targeti2=%d cd=%e %e %e\n",targeti2,cd[targeti2*3],cd[targeti2*3+1],cd[targeti2*3+2]);
#if 0
	    printf(" large-force is scaled down\n");
	    fc[targeti*3]*=fc[targeti*3+1]*=fc[targeti*3+2]*=0.01;
	    fc[targeti2*3]*=fc[targeti2*3+1]*=fc[targeti2*3+2]*=0.01;
#endif
	    for(i=0;i<n3/3;i++){
	      if(i==targeti){
		for(j=0;j<n3/3;j++){
		  double r=0.0;
		  int k;
		  for(k=0;k<3;k++) r+=(cd[i*3+k]-cd[j*3+k])*(cd[i*3+k]-cd[j*3+k]);
		  r=sqrt(r);
		  if(r<0.5){
		    printf("    targeti=%d j=%d r=%e side=%e %e %e\n",targeti,j,r,side[0],side[1],side[2]);
		  }
		}
	      }
	      if(fc[i*3]*fc[i*3]+fc[i*3+1]*fc[i*3+1]+fc[i*3+2]*fc[i*3+2]>1000.0){
		printf(" cras 2: too-large-force i=%d fc=%e %e %e\n",i,fc[i*3],fc[i*3+1],fc[i*3+2]);
		flag=1;
#if 0 // force limit to 10
		printf(" large-force is limited to 10\n");
		if(fc[i*3]>10.0) fc[i*3]=10.0; if(fc[i*3]<-10.0) fc[i*3]=-10.0;
		if(fc[i*3+1]>10.0) fc[i*3+1]=10.0; if(fc[i*3+1]<-10.0) fc[i*3+1]=-10.0;
		if(fc[i*3+2]>10.0) fc[i*3+2]=10.0; if(fc[i*3+2]<-10.0) fc[i*3+2]=-10.0;
#endif
	      }
	    }
	    //	    exit(1);
	  }
	}
#endif

#if 1	// remove offset
	for(jj=0;jj<3;jj++) center[jj]=0.0;
	for(ii=0;ii<n3;ii+=3) for(jj=0;jj<3;jj++) center[jj]+=fc[ii+jj];
	for(jj=0;jj<3;jj++) center[jj]/=n3/3;
	//	printf("center=%e %e %e\n",center[0],center[1],center[2]);
	for(ii=0;ii<n3;ii+=3) for(jj=0;jj<3;jj++) fc[ii+jj]-=center[jj];
#endif
#endif

      } else {

	for(i = 0; i < n1; i++)
	  nig_num[i] = 0;
#if defined(VTGRAPE) && 0
	static int ini=0;
	double zz2[2][2];
	int ii,jj;
	for(ii=0;ii<2;ii++) for(jj=0;jj<2;jj++) zz2[ii][jj]=zz[ii][jj];
	if(ini==0){
	  printf("VTGRAPE is defined in cras36.c\n");
	  MR3init();
	  ini=1;
	}
	MR3calcnacl(cd,n3/3,atype,2,(double *)pol,(double *)sigm,
		    (double *)ipotro,(double *)pc,(double *)pd,
		    (double *)zz2,8,side[0],0,fc);
	//	ini++; if(ini<4) for(i=0;i<6;i++) printf("vtgrape fc[%d]=%e\n",i,fc[i]);
#else // else of VTGRAPE
	for(i = 0; i < n3; i += 3){
	  i0 = atype_mat[atype[i/3]];
	  for(j = i + 3; j < n3; j += 3){
	    d0 = cd[i    ] - cd[j    ]; 
	    d1 = cd[i + 1] - cd[j + 1];
	    d2 = cd[i + 2] - cd[j + 2];

#if ZERO_P == 0      
	    if (d0 < -sideh[0]) d0 += side[0];
	    if (d0 >  sideh[0]) d0 -= side[0];
	    if (d1 < -sideh[1]) d1 += side[1];
	    if (d1 >  sideh[1]) d1 -= side[1];
	    if (d2 < -sideh[2]) d2 += side[2];
	    if (d2 >  sideh[2]) d2 -= side[2];
#endif
      
	    rd = d0 * d0 + d1 * d1 + d2 * d2; 
	    r = sqrt(rd);
	    inr = 1./r;
	  
	    i1 = atype_mat[atype[j/3]];

	    if(bond_flg == 1)
	      if(r < neighbor_radius && 
		 atype_mat[atype[i/3]] != atype_mat[atype[j/3]]){
		if(nig_num[i/3] < 6)
		  nig_data[i/3*6+nig_num[i/3]++] = j/3;
		if(nig_num[j/3] < 6)
		  nig_data[j/3*6+nig_num[j/3]++] = i/3;
	      }

	    d7 = phir;

	    if(sys_num >= 4 && sys_num < 10){
#if ZERO_P == 1
	      phir += zz[i0][i1]/r;
	      dphir = zz[i0][i1]/(r*rd);
	      if(i0+i1 < 5){
		phir += (as_a[i0][i1]*pow(inr,12)-
			 as_c[i0][i1]*pow(inr,6));
		dphir +=(12.0*as_a[i0][i1]*pow(inr,14)-
			 6.0 *as_c[i0][i1]*pow(inr,8));
	      }
#else
	      d6 = erfc(alpha * r);
	      phi[0] += zz[i0][i1]*d6/ r;
	      dphir   = zz[i0][i1]
		*(2*alpha*ISPI*r*exp(-alpha2*rd) + d6)/(r*rd);
	      if(i0+i1 < 5){
		phi[0] += (as_a[i0][i1]*pow(inr,12)-
			   as_c[i0][i1]*pow(inr,6));
		dphir +=(12.0*as_a[i0][i1]*pow(inr,14)-
			 6.0 *as_c[i0][i1]*pow(inr,8));
	      }
#endif
	    } else {
#if ZERO_P == 1
	      if(i0 < 2 && i1 < 2){
		d3 = pb*pol[i0][i1]*exp((sigm[i0][i1]-r)*ipotro[i0][i1]);
		/*
		phir += (d3 - pc[i0][i1]*pow(inr,6)
			 -  pd[i0][i1]*pow(inr,8)
			 + inr*zz[i0][i1]);
		*/
		dphir = (d3 * ipotro[i0][i1] * inr
			 - 6 * pc[i0][i1]*pow(inr,8)
			 - 8 * pd[i0][i1]*pow(inr,10)
			 + inr*inr*inr*zz[i0][i1]);


	      } else  if(i0 > 1 && i1 > 1){
		dphir = zz[i0][i1]/(r*rd);
		/*
		phir += zz[i0][i1]/r;
		iphi[i] += zz[i0][i1]/ r;
		iphi[j] += zz[i0][i1]/ r;
		*/
	      }

	      if(i0 == 2 && i1 == 2){
		dphir += 12.0*wpa*pow(rd,-7.0)-6.0*wpc*pow(rd,-4.0);
		/*
		phir  += wpa*pow(rd,-6.0)-wpc*pow(rd,-3.0);
		iphi[i] += wpa*pow(rd,-6.0)-wpc*pow(rd,-3.0);
		iphi[j] += wpa*pow(rd,-6.0)-wpc*pow(rd,-3.0);
		*/
	      }

#else
	      if(i0 < 2 && i1 < 2){
		d3 = pb*pol[i0][i1]*exp((sigm[i0][i1]-r)*ipotro[i0][i1]);

		phi[0] += (d3 - pc[i0][i1]*pow(inr,6)
			   -    pd[i0][i1]*pow(inr,8));

		dphir = (d3 * ipotro[i0][i1] * inr
			 - 6 * pc[i0][i1]*pow(inr,8)
			 - 8 * pd[i0][i1]*pow(inr,10));

		d4 = erfc(alpha * r);
		phi[0] += d4*inr*zz[i0][i1];
		dphir += (2*alpha*ISPI*r*exp(-alpha2*rd) + d4)/(r*rd)
		  *zz[i0][i1];
	      } else if(i0 > 1 && i1 > 1){
		d6 = erfc(alpha * r);
		phi[0] += zz[i0][i1]*d6/ r;
		dphir   = zz[i0][i1]
		  *(2*alpha*ISPI*r*exp(-alpha2*rd) + d6)/(r*rd);
		iphi[i] += zz[i0][i1]*d6/ r;
		iphi[j] += zz[i0][i1]*d6/ r;
	      }
	      if(i0 == 2 && i1 == 2){
		phi[0]  += wpa*pow(rd,-6.0)-wpc*pow(rd,-3.0);
		dphir += 12.0*wpa*pow(rd,-7.0)-6.0*wpc*pow(rd,-4.0);
		iphi[i] += wpa*pow(rd,-6.0)-wpc*pow(rd,-3.0);
		iphi[j] += wpa*pow(rd,-6.0)-wpc*pow(rd,-3.0);
	      }

#endif
	    }

	    vir -= rd * dphir;

	    d3 = d0 * dphir;
	    d4 = d1 * dphir;
	    d5 = d2 * dphir;

#if 1
	    fc[i  ] += d3;
	    fc[i+1] += d4;
	    fc[i+2] += d5;
	    fc[j  ] -= d3;
	    fc[j+1] -= d4;
	    fc[j+2] -= d5;
#endif

	    if(i0 >= 3){
	      i2 = w_info[i/3]*3;
	      fc[i2  ] += d3;
	      fc[i2+1] += d4;
	      fc[i2+2] += d5;
	    }
	    if(i1 >= 3){
	      i3 = w_info[j/3]*3;
	      fc[i3  ] -= d3;
	      fc[i3+1] -= d4;
	      fc[i3+2] -= d5;
	    }

	  }
	}
#if 0 // for comparing host and VTGRAPE
	static int ini=0;
	if(ini==0){
	  printf("VTGRAPE is not defined but called\n");
	  MR3init();
	}
	if(ini<2){
	  double zz2[2][2];
	  int ii,jj;
	  for(ii=0;ii<2;ii++) for(jj=0;jj<2;jj++) zz2[ii][jj]=zz[ii][jj];
	  for(i=n3-9;i<n3;i++) printf("host fc[%d]=%e\n",i,fc[i]);
	  for(i=0;i<n3;i++) fc[i]=0.0;
	  MR3calcnacl(cd,n3/3,atype,2,(double *)pol,(double *)sigm,
		      (double *)ipotro,(double *)pc,(double *)pd,
		      (double *)zz2,8,side[0],0,fc);
	  for(i=n3-9;i<n3;i++) printf("vtgrape fc[%d]=%e\n",i,fc[i]);
	}
	ini++;
#endif
#endif // end of VTGRAPE
      }

#if 1
      /*#define CHECK*/
#if defined(MDGRAPE3) || defined(VTGRAPE)
      if(bond_flg == 1 && grape_flg==0){
#else
      if(bond_flg == 1){
#endif
	/*
	for(i = 0; i < n1; i++){
	  printf("%d %d (%f %f %f)\n",i,nig_num[i],cd[i*3],cd[i*3+1],cd[i*3+2]);
	  for(j = 0; j < nig_num[i]; j++)
	    printf("%d ",nig_data[i*6+j]);
	  printf("\n");
	}
	exit(0);
	*/
	for(i = 0; i < n1; i++){
	  i0 = 0;
	  for(j = 0; j < nig_num[i]; j++){
	    d0 = cd[nig_data[i*6+j]*3]  -cd[i*3];
	    d1 = cd[nig_data[i*6+j]*3+1]-cd[i*3+1];
	    d2 = cd[nig_data[i*6+j]*3+2]-cd[i*3+2];
	    for(k = j+1; k < nig_num[i]; k++){
	      d3 = cd[nig_data[i*6+k]*3]  -cd[i*3];
	      d4 = cd[nig_data[i*6+k]*3+1]-cd[i*3+1];
	      d5 = cd[nig_data[i*6+k]*3+2]-cd[i*3+2];
	      d6 = acos(fabs((d0*d3 + d1*d4 + d2*d5)
			     /sqrt(d0*d0+d1*d1+d2*d2)
			     /sqrt(d3*d3+d4*d4+d5*d5)))/PI*180;
	      if(d6 < min_angle || d6 > max_angle) i0++;
	    }
	  }
	  if(i0 < 4 || i0 != nig_num[i]*(nig_num[i]-1)/2){
#if 0
	    printf("%d %d (%f %f %f):",i,nig_num[i],cd[i*3],cd[i*3+1],cd[i*3+2]);
	    for(j = 0; j < nig_num[i]; j++)
	      printf("%d ",nig_data[i*6+j]);
	    printf("\n");
	    for(j = 0; j < nig_num[i]; j++)
	      printf("% f % f % f  % f % f % f %f\n"
		     ,cd[nig_data[i*6+j]*3]
		     ,cd[nig_data[i*6+j]*3+1]
		     ,cd[nig_data[i*6+j]*3+2]
		     ,cd[nig_data[i*6+j]*3]  -cd[i*3]
		     ,cd[nig_data[i*6+j]*3+1]-cd[i*3+1]
		     ,cd[nig_data[i*6+j]*3+2]-cd[i*3+2]
		     ,sqrt(pow(cd[nig_data[i*6+j]*3]  -cd[i*3]  ,2)+
			   pow(cd[nig_data[i*6+j]*3+1]-cd[i*3+1],2)+
			   pow(cd[nig_data[i*6+j]*3+2]-cd[i*3+2],2))
		     );
	    i0 = 0;
	    for(j = 0; j < nig_num[i]; j++){
	      d0 = cd[nig_data[i*6+j]*3]  -cd[i*3];
	      d1 = cd[nig_data[i*6+j]*3+1]-cd[i*3+1];
	      d2 = cd[nig_data[i*6+j]*3+2]-cd[i*3+2];
	      for(k = j+1; k < nig_num[i]; k++){
		d3 = cd[nig_data[i*6+k]*3]  -cd[i*3];
		d4 = cd[nig_data[i*6+k]*3+1]-cd[i*3+1];
		d5 = cd[nig_data[i*6+k]*3+2]-cd[i*3+2];
		d6 = acos(fabs((d0*d3 + d1*d4 + d2*d5)
			       /sqrt(d0*d0+d1*d1+d2*d2)
			       /sqrt(d3*d3+d4*d4+d5*d5)))/PI*180;
		if(d6 < min_angle || d6 > max_angle) i0++;
		printf("%d %d (% f % f % f)(% f % f % f) %f %f %d\n",j,k
		       ,d0,d1,d2,d3,d4,d5
		       ,fabs((d0*d3+d1*d4+d2*d5)/sqrt(d0*d0+d1*d1+d2*d2)/sqrt(d3*d3+d4*d4+d5*d5))
		       ,acos(fabs((d0*d3 + d1*d4 + d2*d5)
				  /sqrt(d0*d0+d1*d1+d2*d2)
				  /sqrt(d3*d3+d4*d4+d5*d5)))/PI*180
		       ,i0
		       );
	      }
	    }
	    keyboard('z',0,0);
#endif
	    nig_num[i] = 0;
	  } else {
#if 0
	    printf("%d %d (%f %f %f):",i,nig_num[i],cd[i*3],cd[i*3+1],cd[i*3+2]);
	    for(j = 0; j < nig_num[i]; j++)
	      printf("%d ",nig_data[i*6+j]);
	    printf("\n");
	    for(j = 0; j < nig_num[i]; j++)
	      printf("% f % f % f  % f % f % f %f\n"
		     ,cd[nig_data[i*6+j]*3]
		     ,cd[nig_data[i*6+j]*3+1]
		     ,cd[nig_data[i*6+j]*3+2]
		     ,cd[nig_data[i*6+j]*3]  -cd[i*3]
		     ,cd[nig_data[i*6+j]*3+1]-cd[i*3+1]
		     ,cd[nig_data[i*6+j]*3+2]-cd[i*3+2]
		     ,sqrt(pow(cd[nig_data[i*6+j]*3]  -cd[i*3]  ,2)+
			   pow(cd[nig_data[i*6+j]*3+1]-cd[i*3+1],2)+
			   pow(cd[nig_data[i*6+j]*3+2]-cd[i*3+2],2))
		     );
	    i0 = 0;
	    for(j = 0; j < nig_num[i]; j++){
	      d0 = cd[nig_data[i*6+j]*3]  -cd[i*3];
	      d1 = cd[nig_data[i*6+j]*3+1]-cd[i*3+1];
	      d2 = cd[nig_data[i*6+j]*3+2]-cd[i*3+2];
	      for(k = j+1; k < nig_num[i]; k++){
		d3 = cd[nig_data[i*6+k]*3]  -cd[i*3];
		d4 = cd[nig_data[i*6+k]*3+1]-cd[i*3+1];
		d5 = cd[nig_data[i*6+k]*3+2]-cd[i*3+2];
		d6 = acos(fabs((d0*d3 + d1*d4 + d2*d5)
			       /sqrt(d0*d0+d1*d1+d2*d2)
			       /sqrt(d3*d3+d4*d4+d5*d5)))/PI*180;
		if(d6 < min_angle || d6 > max_angle) i0++;
		printf("%d %d (% f % f % f)(% f % f % f) %f %f %d\n",j,k
		       ,d0,d1,d2,d3,d4,d5
		       ,fabs((d0*d3+d1*d4+d2*d5)/sqrt(d0*d0+d1*d1+d2*d2)/sqrt(d3*d3+d4*d4+d5*d5))
		       ,acos(fabs((d0*d3 + d1*d4 + d2*d5)
				  /sqrt(d0*d0+d1*d1+d2*d2)
				  /sqrt(d3*d3+d4*d4+d5*d5)))/PI*180
		       ,i0
		       );
	      }
	    }
	    keyboard('z',0,0);
#endif
	  }
	}

	for(i = 0; i < n1; i++){
	  if(nig_num[i] < 6){
	    i0 = 0;
	    for(j = 0; j < nig_num[i]; j++){
	      if(nig_num[nig_data[i*6+j]] >= 4) i0++;
	    }
	    if(i0 == 0) nig_num[i] = 0;
	    /*
	    else {
	      printf("%d %d %d\n",i,nig_num[i],i0);
	      for(j = 0; j < nig_num[i]; j++){
		printf("(%d %d)",nig_data[i*6+j],nig_num[nig_data[i*6+j]]);
	      }
	      printf("\n");
	    }
	    */
	  }
	}
#if 0
	i0 = 0;
	for(i = 0; i < n1; i++){
	  i0 += nig_num[i];
	}
	if(i0 != 0){
	  for(i = 0; i < n1; i++){
	    printf("%d %d: ",i,nig_num[i]);
	    i1 = 0;
	    for(j = 0; j < nig_num[i]; j++){
	      if(nig_num[nig_data[i*6+j]] > 3) i1++;
	      printf("(%d %d)",nig_data[i*6+j],nig_num[nig_data[i*6+j]]);
	    }
	    printf(" -> %d\n",i1);
	  }
	  keyboard('z',0,0);
	}
#endif

      }
      /*
      for(i = 0; i < n1; i++)
	printf("%d %d (%f %f %f)\n",i,nig_num[i],cd[i*3],cd[i*3+1],cd[i*3+2]);
      exit(0);
      */
#endif

#if ZERO_P == 0  
      for(k = 0;k < vmax; k++){
        d0 = 0;
        d1 = 0;
        d2 = iside[0]*iside[1]*iside[2]/(pow(vecn[k][1]*iside[0],2)+
                                         pow(vecn[k][2]*iside[1],2)+
                                         pow(vecn[k][3]*iside[2],2))
          *exp(-PI2/alpha2*(pow(vecn[k][1]*iside[0],2)+
                            pow(vecn[k][2]*iside[1],2)+
                            pow(vecn[k][3]*iside[2],2)));
        d3 = 4 * d2;
        d2 *= IPI;

        for(i = 0;i < n3; i += 3){
          d5 = (vecn[k][1]*cd[i]  *iside[0]+
                vecn[k][2]*cd[i+1]*iside[1]+
                vecn[k][3]*cd[i+2]*iside[2]);
          d6 = z[atype_mat[atype[i/3]]];
          d0 += cos(PIT*d5)*d6;
          d1 += sin(PIT*d5)*d6;
        }

        phi[1] += d2 * (d0*d0 + d1*d1);
        /*
          printf("%d % d % d % d %e %e\n"
          ,k,vecn[k][1],vecn[k][2],vecn[k][3],phi[1]
          , d2 * (d0*d0 + d1*d1));
        */
        for(i = 0;i < n3; i += 3){
          d5 = (vecn[k][1]*cd[i]  *iside[0]+
                vecn[k][2]*cd[i+1]*iside[1]+
                vecn[k][3]*cd[i+2]*iside[2]);
          d4 = d3*(d0*sin(PIT*d5)-d1*cos(PIT*d5));
          d6 = z[atype_mat[atype[i/3]]];
          fc[i  ] += d4 * vecn[k][1]*d6*iside[0];
          fc[i+1] += d4 * vecn[k][2]*d6*iside[1];
          fc[i+2] += d4 * vecn[k][3]*d6*iside[2];
          if(atype_mat[atype[i/3]] >= 3){
            i2 = w_info[i/3]*3;
            fc[i2  ] += d4 * vecn[k][1]*d6*iside[0];
            fc[i2+1] += d4 * vecn[k][2]*d6*iside[1];
            fc[i2+2] += d4 * vecn[k][3]*d6*iside[2];
          }
        }
      }
      phir = phi[0] + phi[1] + phi[2] + phir_corr;
      /*
        d0 = 0;
        for(i0 = 0; i0 < w_site; i0++){
        for(i1 = 0; i1 < w_num; i1++){
        if(i0 == 0)
        i = w_index[i1];
        else
        i = (w_info[w_index[i1]/3]+i0-1)*3;
        printf("host %3d % e % e % e % e\n"
        ,i/3
        ,fc[i],fc[i+1],fc[i+2],iphi[i]);
        d0 += iphi[i];
        }
        }
        printf("%f %f %f %f %f\n",phir,phi[0],phi[1],phi[2],d0/2);
*/
#endif

#if 0      
#if ZERO_P == 1
      for(i = 0; i < n3; i += 3){
        if(atype[i/3] <= 2){
        /*
        printf("%d %d %f %f %f % f % f % f\n",i/3,atype[i/3],
               cd[i],cd[i+1],cd[i+2],
               vl[i],vl[i+1],vl[i+2]);
        */
          for(j = 0; j < 3; j++){
            d0 = cd[i+j];
            rd = d0*d0;
            if(rd < 16.0){
              dphir = pow(rd,-7.);
              d3 = d0 * dphir;
              fc[i+j] += d3;
              if(atype_mat[atype[i/3]] >= 3){
                i2 = w_info[i/3]*3;
                fc[i2+j] += d3;
              }
            }
          }
          for(j = 0; j < 3; j++){
            d0 = cd[i+j] - side[j];
            rd = d0*d0;
            if(rd < 16.0){
              dphir = pow(rd,-7.);
              d3 = d0 * dphir;
              fc[i+j] += d3;
              if(atype_mat[atype[i/3]] >= 3){
                i2 = w_info[i/3]*3;
                fc[i2+j] += d3;
              }
            }
          }
        }
      }
#endif
#endif


#ifdef LAP_TIME
#if defined(_WIN32) && !defined(__CYGWIN__)
    md_time = (double)timeGetTime()/1000.;
#elif defined(MAC)
    md_time = (double)clock()/60.;
#else
    gettimeofday(&time_v,NULL);
    md_time = (time_v.tv_sec + time_v.tv_usec / 1000000.0);
#endif
#endif

#ifdef C_MASS
    for(i0 = 0; i0 < w_num; i0++){
      i = w_index[i0];
      j = i0*4;
      c = w_info[i/3]*3;
      ang0 = ang[j  ];
      ang1 = ang[j+1];
      ang2 = ang[j+2];
      ang3 = ang[j+3];

      trq[i0*3] = trq[i0*3+1] = trq[i0*3+2] = 0;

      d10 = (center_mass*(-2)*(ang0*ang1+ang2*ang3));
      d11 = (center_mass*( ang0*ang0-ang1*ang1-ang2*ang2+ang3*ang3));
      d12 = (center_mass*  2 *(ang1*ang3-ang0*ang2));
      trq[i0*3  ] += d11*fc[i+2] - d12*fc[i+1];
      trq[i0*3+1] += d12*fc[i  ] - d10*fc[i+2];
      trq[i0*3+2] += d10*fc[i+1] - d11*fc[i  ];
      for(k = 0; k < w_site-1; k++){
	d0 = (m_cdx[k]*(-ang0*ang0+ang1*ang1-ang2*ang2+ang3*ang3)+
	      (m_cdy[k]+center_mass)*(-2)*(ang0*ang1+ang2*ang3)+
	      m_cdz[k]*( 2)*(ang1*ang2-ang0*ang3));
	d1 = (m_cdx[k]*  2 *(ang2*ang3-ang0*ang1)+
	      (m_cdy[k]+center_mass)*( ang0*ang0-ang1*ang1-ang2*ang2+ang3*ang3)+
	      m_cdz[k]*(-2)*(ang0*ang2+ang1*ang3));
	d2 = (m_cdx[k]*  2 *(ang1*ang2+ang0*ang3)+
	      (m_cdy[k]+center_mass)*  2 *(ang1*ang3-ang0*ang2)+
	      m_cdz[k]*(-ang0*ang0-ang1*ang1+ang2*ang2+ang3*ang3));
	trq[i0*3  ] += d1*fc[k*3+c+2] - d2*fc[k*3+c+1];
	trq[i0*3+1] += d2*fc[k*3+c  ] - d0*fc[k*3+c+2];
	trq[i0*3+2] += d0*fc[k*3+c+1] - d1*fc[k*3+c  ];
      }
    }
#else
      for(i0 = 0; i0 < w_num; i0++){
        i = w_index[i0];
        j = i0*4;
        c = w_info[i/3]*3;
        ang0 = ang[j  ];
        ang1 = ang[j+1];
        ang2 = ang[j+2];
        ang3 = ang[j+3];
        trq[i0*3] = trq[i0*3+1] = trq[i0*3+2] = 0;
        for(k = 0; k < w_site-1; k++){
          d0 = m_cdx[k]*(-ang0*ang0+ang1*ang1-ang2*ang2+ang3*ang3)
              +m_cdy[k]*(-2)*(ang0*ang1+ang2*ang3)
              +m_cdz[k]*( 2)*(ang1*ang2-ang0*ang3);
          d1 = m_cdx[k]*  2 *(ang2*ang3-ang0*ang1)
              +m_cdy[k]*(ang0*ang0-ang1*ang1-ang2*ang2+ang3*ang3)
              +m_cdz[k]*(-2)*(ang0*ang2+ang1*ang3);
          d2 = m_cdx[k]*2*(ang1*ang2+ang0*ang3)
              +m_cdy[k]*2*(ang1*ang3-ang0*ang2)
              +m_cdz[k]*(-ang0*ang0-ang1*ang1+ang2*ang2+ang3*ang3);
          trq[i0*3  ] += d1*fc[k*3+c+2] - d2*fc[k*3+c+1];
          trq[i0*3+1] += d2*fc[k*3+c  ] - d0*fc[k*3+c+2];
          trq[i0*3+2] += d0*fc[k*3+c+1] - d1*fc[k*3+c  ];
          /*
          printf("%d %d %d %d % e % e % e  % e % e % e\n",i0,k,c,w_info[i/3]
                 ,trq[i0*3],trq[i0*3+1],trq[i0*3+2]
                 ,fc[k*3+c],fc[k*3+c+1],fc[k*3+c+2]);
          */
        }
      }
#endif

      /*
      for(i = 0; i < w_num3; i += 3){
        printf("%3d % e % e % e\n",i/3,trq[i],trq[i+1],trq[i+2]);
      }
      */

      for(i = 0; i < n3; i++){
        if(atype[i/3] == 2)
          fc[i] *= hsq/(a_mass[2]+2*a_mass[3]);
        else if(atype[i/3] == 0 || atype[i/3] == 1)
          fc[i] *= hsq/a_mass[atype_mat[atype[i/3]]];
      }

      for(i = 0; i < w_num3; i++)
        trq[i] *= hsq;
      
      ekin1 = 0;
      ekin2 = 0;
      for(i = 0; i < n3; i += 3){
        ekin1 += (vl[i  ]*vl[i  ] +
                  vl[i+1]*vl[i+1] +
                  vl[i+2]*vl[i+2])*a_mass[atype_mat[atype[i/3]]];
      }
      for(i = 0; i < w_num3; i += 3){
        ekin2 += (moi[0]*agvph[i  ]*agvph[i  ]+
                  moi[1]*agvph[i+1]*agvph[i+1]+
                  moi[2]*agvph[i+2]*agvph[i+2]);
      }
      
      ekin1 /= hsq;
      ekin2 /= hsq;
      
      ekin = ekin1 + ekin2;

#if V_SCALE == 1      
      /*      if(m_clock < 1000){*/     /* velocity scaling */
      if(1){    /* velocity scaling */
        if(m_clock % 50 == 0 || mtemp > rtemp*3){
          mtemp = tscale * ekin;
          d0 = sqrt(rtemp / mtemp);
          for(i = 0; i < n3; i++){
            vl[i] *= d0;
          }
          for(i = 0; i < w_num3; i++){
            agvph[i]*= d0;
          }
          for(i = 0; i < w_num3; i += 3){
            agv0 = agvph[i  ]*moi[0];
            agv1 = agvph[i+1]*moi[1];
            agv2 = agvph[i+2]*moi[2];
            ang0 = angh[i/3*4  ];
            ang1 = angh[i/3*4+1];
            ang2 = angh[i/3*4+2];
            ang3 = angh[i/3*4+3];
            agvh[i]  = agv0*(-ang0*ang0+ang1*ang1-ang2*ang2+ang3*ang3)
              + agv1*(-2)*(ang0*ang1+ang2*ang3)
              + agv2*( 2)*(ang1*ang2-ang0*ang3);
            agvh[i+1]= agv0*( 2)*(ang2*ang3-ang0*ang1)
              + agv1*(ang0*ang0-ang1*ang1-ang2*ang2+ang3*ang3)
              + agv2*(-2)*(ang0*ang2+ang1*ang3);
            agvh[i+2]= agv0*( 2)*(ang1*ang2+ang0*ang3)
              + agv1*( 2)*(ang1*ang3-ang0*ang2)
              + agv2*(-ang0*ang0-ang1*ang1+ang2*ang2+ang3*ang3);
          }
          ekin1 = 0;
          ekin2 = 0;
          for(i = 0; i < n3; i += 3){
            ekin1 += (vl[i  ]*vl[i  ]+
                      vl[i+1]*vl[i+1]+
                      vl[i+2]*vl[i+2])*a_mass[atype_mat[atype[i/3]]];
          }
          for(i = 0; i < w_num3; i += 3){
            ekin2 += (moi[0]*agvph[i  ]*agvph[i  ]+
                      moi[1]*agvph[i+1]*agvph[i+1]+
                      moi[2]*agvph[i+2]*agvph[i+2]);
          }
          ekin1 /= hsq;
          ekin2 /= hsq;

          ekin = ekin1 + ekin2;
        }
      }
#endif
      mtemp = tscale * ekin;
#if 0
      printf("mtemp=%e temp=%e\n",mtemp,temp);
#endif
      mpres  = nden / 3. * (ekin - vir) / (s_num+w_num);
#if T_CONST == 1
      xs += (mtemp - rtemp)/ lq * hsq *.5;
#endif
      /*
      printf("%4d %f %f %f %f %f %f\n",m_clock,mtemp*epsv/kb
             ,(ekin/2.+phir)*erdp/(double)(s_num/2+w_num)
             ,ekin/2.*erdp/(double)(s_num/2+w_num)
             ,phir*erdp/(double)(s_num/2+w_num)
             ,ekin1/(ekin1+ekin2), ekin2/(ekin1+ekin2));
      */
  }  

#if defined(SUBWIN) && defined(GL_ON)
  if((b_clock % 10) == 0){
    if(p_count < DATA_NUM){
      temp_data[p_count] = (int)(mtemp*epsv/kb);
      p_count++;
    } else {
      for(i = 0; i < DATA_NUM-1; i++){
	temp_data[i] = temp_data[i+1];
      }
      temp_data[DATA_NUM-1] = (int)(mtemp*epsv/kb);
    }

    temp_max = 0;
    for(i = 0; i < DATA_NUM; i++){
      if(temp_data[i] > temp_max) temp_max=temp_data[i];
    }
    if(temp_ymax < temp_max){ temp_ymax = temp_max*1.5;}
    if(temp_ymax > temp_max*2 && temp_ymax/2 > 2000){ temp_ymax /= 2;}

  }
#endif

#ifdef GL_ON
  if(sc_flg != 1)
    glutPostRedisplay();
#endif

  if(auto_flg == 1){
#ifdef GL_ON
    mouse_l = tt[b_clock].mouse[0];
    mouse_m = tt[b_clock].mouse[1];
    mouse_r = tt[b_clock].mouse[2];
    for(i = 0; i < 3; i++){
      trans[i] += tt[b_clock].move[i];
      angle[i] = tt[b_clock].rot[i];
    }
#endif
    if(sc_flg != 1){
      if(tt[b_clock].command != 0)
	keyboard(tt[b_clock].command,0,0);
    }
    temp += tt[b_clock].temp;
    rtemp = temp / epsv * kb;
    for(i = 0; i < 16; i++)
      m_matrix[i] += tt[b_clock].matrix[i];
    /*
    if(b_clock == 10)
      angle[2] = 1;
    if(b_clock == 100)
      mouse_l = 1;
    if(b_clock >= 110 && b_clock <= 200)
      trans[0] -= 1;
    if(b_clock >= 210 && b_clock <= 300){
      temp += 10;
      rtemp = temp / epsv * kb;
    }
    if(b_clock >= 310 && b_clock <= 410){
      temp -= 10;
      rtemp = temp / epsv * kb;
    }
    if(b_clock == 410)
      keyboard('n',0,0);
    if(b_clock == 520)
      keyboard('9',0,0);
    if(b_clock == 521)
      keyboard(' ',0,0);
    */
  }
  /*
  printf("%d %d %d %d % f % f % f\n",b_clock,mouse_l,mouse_m,mouse_r
	 ,angle[0],angle[1],angle[2]);
  */
  b_clock++;
}
void sock_md_run()
{
#ifdef SOCK_ON
  int md_loop;
  int i;
  int i0,i1,i2,i3,i4,i5;
  char s_buffer[1024];
  char sbuf[1024];


  i0 = 1;

  sock_send_int(s_sock,&i0,1);


#ifdef LAP_TIME
  sock_recv_double(s_sock,cd,(n3+4+c_num*3));
#else
  sock_recv_double(s_sock,cd,(n3+2+c_num*3));
#endif

  mtemp = cd[n3+c_num*3];
  m_clock = cd[n3+c_num*3+1];
#ifdef LAP_TIME
  md_time0 = cd[n3+c_num*3+2];
  md_time  = cd[n3+c_num*3+3];
#endif

  sock_recv_double(s_sock,vl,n3);
  if(bond_flg == 1){
    sock_recv_int(s_sock,nig_num,n1);
    sock_recv_int(s_sock,(int*)nig_data,n1*6);
  }

#if defined(SUBWIN) && defined(GL_ON)
  if((b_clock % 10) == 0){
    if(p_count < DATA_NUM){
      temp_data[p_count] = (int)(mtemp*epsv/kb);
      p_count++;
    } else {
      for(i = 0; i < DATA_NUM-1; i++){
	temp_data[i] = temp_data[i+1];
      }
      temp_data[DATA_NUM-1] = (int)(mtemp*epsv/kb);
    }

    temp_max = 0;
    for(i = 0; i < DATA_NUM; i++){
      if(temp_data[i] > temp_max) temp_max=temp_data[i];
    }
    if(temp_ymax < temp_max){ temp_ymax = temp_max*1.5;}
    if(temp_ymax > temp_max*2 && temp_ymax/2 > 2000){ temp_ymax /= 2;}

  }
#endif
#ifdef GL_ON
  glutPostRedisplay();
#endif
  if(auto_flg == 1){
#ifdef GL_ON
    mouse_l = tt[b_clock].mouse[0];
    mouse_m = tt[b_clock].mouse[1];
    mouse_r = tt[b_clock].mouse[2];
    for(i = 0; i < 3; i++){
      trans[i] += tt[b_clock].move[i];
      angle[i] = tt[b_clock].rot[i];
    }
#endif
    if(tt[b_clock].command != 0)
      keyboard(tt[b_clock].command,0,0);
    temp += tt[b_clock].temp;
    rtemp = temp / epsv * kb;
    for(i = 0; i < 16; i++)
      m_matrix[i] += tt[b_clock].matrix[i];
  }
  b_clock++;
#endif
}
void init_MD(void)
{
  double d0,d1,d2;
  int i,j;

  srand( 1 );
  /*  srand( ( unsigned )time( NULL ));*/
  
  mass = a_mass[0]/avo*1e-3;

  for(i = 1; i < 4; i++){
    a_mass[i] /= a_mass[0];
  }
  a_mass[0] = 1.0;

  epsj  = epsv*1.60219e-19;
  crdp = sigma * 1e+10;
  tmrdp = sqrt(mass / epsj) * sigma;
  erdp = epsv * 2.30492e+1;       /* for calculate energy(kcal/mol) */

  keiname[0] = 0;

  if(sys_num == 0){
    if(tflg == 0)
      temp  = 300;
    delt = 2.0e-15;
  } else {
    if(tflg == 0)
      temp  = 293;
    delt = .5e-15;
  }

  rtemp = temp / epsv * kb;
  h     = delt / tmrdp;
  hsq   = h * h;

  atype_mat[0] = 0; /* Na */
  atype_mat[1] = 1; /* Cl */
  z[0] = 1.0;
  z[1] =-1.0;

#if SPC == 1
  wpa = 629.4/2.30492e+1/epsv*1e+3;
  wpc = 625.5/2.30492e+1/epsv;
  z[2] = -.82; z[3] = 0.41; z[4] = 0.0;
  bond[0] = 1.0;  bond[1] = 0.0;
  hoh_deg = 109.47;
  w_site = 3;
  atype_mat[2] = 2; /* O  */
  atype_mat[3] = 3; /* H1 */
  atype_mat[4] = 3; /* H2 */
  atype_mat[8] = 2; /* O  */
#endif
#if ST2  == 1
  d0 = 7.575e-2;
  d1 = 3.1;
  wpa = d0*4*pow(d1,12)/2.30492e+1/epsv;
  wpc = d0*4*pow(d1, 6)/2.30492e+1/epsv;
  z[2] = 0; z[3] = 0.2357; z[4] =-0.2357;
  bond[0] = 1.0;  bond[1] = 0.8;

  wpa = 629.4/2.30492e+1/epsv*1e+3;
  wpc = 625.5/2.30492e+1/epsv;
  z[2] = 0; z[3] = 0.41; z[4] =-0.41;
  bond[0] = 1.0;  bond[1] = 0.4;

  hoh_deg = 109.28;
  w_site = 5;
  atype_mat[2] = 2; /* O  */
  atype_mat[3] = 3; /* H1 */
  atype_mat[4] = 3; /* H2 */
  atype_mat[5] = 4; /* L1 */
  atype_mat[6] = 4; /* L2 */
#endif
#if TIP5P  == 1
  d0 = 0.16;
  d1 = 3.12;
  wpa = d0*4*pow(d1,12)/2.30492e+1/epsv;
  wpc = d0*4*pow(d1, 6)/2.30492e+1/epsv;
  z[2] = 0; z[3] = 0.241; z[4] =-0.241;
  bond[0] = 0.9572;  bond[1] = 0.7;

  hoh_deg = 104.52;
  w_site = 5;
  atype_mat[2] = 2; /* O  */
  atype_mat[3] = 3; /* H1 */
  atype_mat[4] = 3; /* H2 */
  atype_mat[5] = 4; /* L1 */
  atype_mat[6] = 4; /* L2 */
  atype_mat[8] = 2; /* O  */
#endif

  for(i = 0; i < KNUM+4; i++)
    for(j = 0; j < KNUM+4; j++){
      zz[i][j] = z[i]*z[j];
    }

  pb = 0.338e-19/epsj;
  for(i = 0; i < 2; i++){
    for(j = 0; j < 2; j++){
      pc[i][j] = 0;
      pd[i][j] = 0;
      pol[i][j] = 0;
      sigm[i][j] = 0;
      ipotro[i][j] = 0;
    }
  }
  potpar5(1,-1,1,-1,keiname);

  for(i = 0;i < 2; i++){
    for(j = 0;j < 2; j++){
      pc[i][j] *= 1e-79/epsj/pow(sigma,6);
      pd[i][j] *= 1e-99/epsj/pow(sigma,8);
    }
  }

  /* 0:Na 1:Cl 2:O 3:H */
  as_s[0][0] = 2.443; as_s[0][1] = 2.796; as_s[0][2] = 2.72; as_s[0][3] =1.310;
  as_s[1][0] = 2.796; as_s[1][1] = 3.487; as_s[1][2] = 3.55; as_s[1][3] =2.140;
  as_s[2][0] = 2.72;  as_s[2][1] = 3.55;  as_s[2][2] = 3.156;as_s[2][3] =0.0;
  as_s[3][0] = 1.310; as_s[3][1] = 2.140; as_s[3][2] = 0.0;  as_s[3][3] =0.0;

  as_e[0][0]=0.11913; as_e[0][1]= 0.3526;as_e[0][2]=0.56014;as_e[0][3]=0.56014;
  as_e[1][0]=0.3526;  as_e[1][1]=0.97906;as_e[1][2]=1.50575;as_e[1][3]=1.50575;
  as_e[2][0]=0.56014; as_e[2][1]=1.50575;as_e[2][2]=0.65020;as_e[2][3]=0.0;
  as_e[3][0]=0.56014; as_e[3][1]=1.50575;as_e[3][2] = 0.0;  as_e[3][3]=0.0;

  for(i = 0; i < 4; i++)
    for(j = 0; j < 4; j++){
      as_e[i][j] *= 4.*1000. * 1.0364272e-5 /epsv;
      as_a[i][j] = as_e[i][j]*pow(as_s[i][j],12);
      as_c[i][j] = as_e[i][j]*pow(as_s[i][j], 6);
    }

  if(sys_num == 0){
    n1 = np*np*np*8;
  } else if(sys_num == 1 || sys_num == 10){
    n1 = np*np*np*4*w_site;
  } else if(sys_num == 2){
#if ZERO_P == 1
    n1 = (npx*npy*npz*8+npy*npz*4+(npy-1)*npz*8)*w_site;
#else
    n1 = np*np*np*8*w_site;
#endif
  } else if(sys_num == 3){
    n1 = np*np*np*8*w_site;
  } else if(sys_num == 4){
    if(np*np*np*8-nn*2 >= 0)
      n1 = (np*np*np*8-nn*2)*w_site+nn*2;
    else {
      printf("nn is too large!\n");
      exit(0);
    }
  } else if(sys_num == 5){
    if(np*np*np*8-nw >= 0)
      n1 = (np*np*np*8-nw)+nw*w_site;
    else {
      printf("nw is too large!\n");
      exit(0);
    }
  } else if(sys_num == 6){
    if(np-nw >= 0){
      if(nw > 0)
        n1 = np*np*np*8+(np*np*np*8-(np-nw)*(np-nw)*(np-nw)*8)*(w_site-1);
      else
        n1 = np*np*np*8;
    } else {
      printf("nw is too large!\n");
      exit(0);
    }
  }

  n2 = n1 * 2;
  n3 = n1 * 3;

  m_cdx[0] =  bond[0]*sin(hoh_deg/2/180*PI);
  m_cdy[0] = -bond[0]*cos(hoh_deg/2/180*PI);
  m_cdz[0] = 0;
  m_cdx[1] = -bond[0]*sin(hoh_deg/2/180*PI);
  m_cdy[1] = -bond[0]*cos(hoh_deg/2/180*PI);
  m_cdz[1] = 0;
#if ST2 == 1
  m_cdx[2] =  0;
  m_cdy[2] =  bond[1]*cos(hoh_deg/2/180*PI);
  m_cdz[2] =  bond[1]*sin(hoh_deg/2/180*PI);
  m_cdx[3] =  0;
  m_cdy[3] =  bond[1]*cos(hoh_deg/2/180*PI);
  m_cdz[3] = -bond[1]*sin(hoh_deg/2/180*PI);
#endif
#if TIP5P == 1
  d0 = 109.47;
  m_cdx[2] =  0;
  m_cdy[2] =  bond[1]*cos(d0/2/180*PI);
  m_cdz[2] =  bond[1]*sin(d0/2/180*PI);
  m_cdx[3] =  0;
  m_cdy[3] =  bond[1]*cos(d0/2/180*PI);
  m_cdz[3] = -bond[1]*sin(d0/2/180*PI);
#endif

  d0 = d1 = d2= 0;
  for(i = 0; i < 2; i++){
    d0 += m_cdx[i]*a_mass[3];
    d1 += m_cdy[i]*a_mass[3];
    d2 += m_cdz[i]*a_mass[3];
  }
  center_mass = -d1/(a_mass[2]+a_mass[3]*2.0);

#ifdef C_MASS
  moi[0] = (a_mass[2]*center_mass*center_mass +
	    a_mass[3]*(m_cdy[0]+center_mass)*(m_cdy[0]+center_mass)*2.0);
  moi[1] = a_mass[3]*m_cdx[0]*m_cdx[0]*2.0;
  d0 = sqrt(bond[0]*bond[0]+center_mass*center_mass
	    -2.0*bond[0]*center_mass*cos(hoh_deg/2/180*PI));
  moi[2] = a_mass[2]*center_mass*center_mass+a_mass[3]*d0*d0*2.0;
#else
  moi[0] = 2*a_mass[3]*m_cdy[0]*m_cdy[0];
  moi[1] = 2*a_mass[3]*m_cdx[0]*m_cdx[0];
  moi[2] = 2*a_mass[3]*bond[0]*bond[0];
#endif

  if(sys_num == 0){
    w_site = 1;
    nden = mass_den3(1,-1,1,-1,0,temp);
    tscale = 1. / 3. /((double)n1 - 1);
    w_num = 0;
    w_num3 = 0;
    s_num = n1;
    s_num3 = n3;

    vmax = VMAX;
    oalpha = 6;
    if(np >= 8){
      vmax = 462;
      oalpha = 8.6;
    }
    if(np == 7){
      vmax = 462;
      oalpha = 8.2;
    }
    if(np == 6){
      vmax = 40;
      oalpha = 4.0;
      vmax = 462;
      oalpha = 7.6;
    }
    if(np == 5){
      vmax = 46;
      oalpha = 4.4;
      vmax = 462;
      oalpha = 6.9;
    }
    if(np == 4){
      vmax = 101;
      oalpha = 5.2;
    }
    if(np == 3){
      vmax = 309;
      oalpha = 5.6;
    }

  } else if(sys_num >= 1){

    vmax = VMAX;
    oalpha = 1.5;
    nden = nden_set(temp-273)/((a_mass[2]+a_mass[3]*2)*mass)*1e-27;
    tscale = 1. / 3. /((double)n1/w_site*2 - 1);

    w_num = n1/w_site;
    w_num3= w_num*3;
    s_num = 0;
    s_num3= 0;

    if(sys_num == 4){
      w_num = np*np*np*8-nn*2;
      w_num3= w_num*3;
      s_num = nn*2;
      s_num3= s_num*3;
      printf("np %d nn %d w_num %d s_num %d n1 %d\n",np,nn,w_num,s_num,n1);
    }
    if(sys_num == 5){
      w_num = nw;
      w_num3= w_num*3;
      s_num = np*np*np*8-nw;
      s_num3= s_num*3;
      nden = mass_den3(1,-1,1,-1,0,temp);
      printf("np %d nn %d w_num %d s_num %d n1 %d\n",np,nn,w_num,s_num,n1);
    }
    if(sys_num == 6){
      if(nw > 0)
        w_num = np*np*np*8-(np-nw)*(np-nw)*(np-nw)*8;
      else
        w_num = 0;
      w_num3= w_num*3;
      s_num = np*np*np*8-w_num;
      s_num3= s_num*3;
      nden = mass_den3(1,-1,1,-1,0,temp);
      printf("np %d nw %d w_num %d s_num %d n1 %d\n",np,nw,w_num,s_num,n1);
    }
  }
  ws_num = w_num+s_num;
  ws_num3= ws_num*3;

  tscale = 1. / 3. /((double)(s_num + w_num*2) - 1);
}
void keep_mem(int num_s, int num_w)
{
  int i,j;
  int add = 100;
  
  if((nli = (long*)malloc(20000 * sizeof(long))) == NULL){
    printf("memory error\n");
    exit(1);
  }

  if((nig_num = (int*)malloc((num_s+num_w+add) * sizeof(int))) == NULL){
    printf("memory error\n");
    exit(1);
  }
  if((nig_data = (int*)malloc((num_s+num_w+add)*6 * sizeof(int))) == NULL){
    printf("memory error\n");
    exit(1);
  }
  /*
  if((nig_data = (int**)malloc((num_s+num_w+add) * sizeof(int*))) == NULL){
    printf("memory error\n");
    exit(1);
  }
  for(i = 0; i < (num_s+num_w+add); i++)
    if((nig_data[i] = malloc(6 * sizeof(int))) == NULL){
      printf("memory error\n");
      exit(1);
    }
  */

  /*
  if((nig_data = (int**)malloc(sizeof(int*))) == NULL){
    printf("memory error\n");
    exit(1);
  }
  if((nig_data[0] = (int*)malloc((num_s+num_w+add)*6 * sizeof(int))) == NULL){
    printf("memory error\n");
    exit(1);
  }
  for(i = 1; i < (num_s+num_w+add); i++)
    nig_data[i] = nig_data[0] + i*6;
  */
  /*
  printf("%d\n",nig_data);
  printf("%d\n",nig_data[0]);
  for(i = 0; i < (num_s+num_w+add); i++){
    for(j = 0; j < 6; j++){
      printf("(%d %d %d)",i,j,&nig_data[i][j]);
    }
    printf("\n");
  }
  */


  if((atype = malloc((num_s+num_w+add) * sizeof(int))) == NULL){
    printf("memory error\n");
    exit(1);
  }
  if((cd = malloc((num_s+num_w+add) * 3 * sizeof(double))) == NULL){
    printf("memory error\n");
    exit(1);
  }
  if((vl = malloc((num_s+num_w+add) * 3 * sizeof(double))) == NULL){
    printf("memory error\n");
    exit(1);
  }
  if((fc = malloc((num_s+num_w+add) * 3 * sizeof(double))) == NULL){
    printf("memory error\n");
    exit(1);
  }
  if((fcc = malloc((num_s+num_w+add) * 3 * sizeof(double))) == NULL){
    printf("memory error\n");
    exit(1);
  }
  if((iphi = malloc((num_s+num_w+add) * 3 * sizeof(double))) == NULL){
    printf("memory error\n");
    exit(1);
  }
  if((ang = malloc((num_w+add) * 4 * sizeof(double))) == NULL){
    printf("memory error\n");
    exit(1);
  }
  if((agv = malloc((num_w+add) * 3 * sizeof(double))) == NULL){
    printf("memory error\n");
    exit(1);
  }
  if((agvp = malloc((num_w+add) * 3 * sizeof(double))) == NULL){
    printf("memory error\n");
    exit(1);
  }
  if((angh = malloc((num_w+add) * 4 * sizeof(double))) == NULL){
    printf("memory error\n");
    exit(1);
  }
  if((agvh = malloc((num_w+add) * 3 * sizeof(double))) == NULL){
    printf("memory error\n");
    exit(1);
  }
  if((agvph = malloc((num_w+add) * 3 * sizeof(double))) == NULL){
    printf("memory error\n");
    exit(1);
  }
  if((trq = malloc((num_w+add) * 3 * sizeof(double))) == NULL){
    printf("memory error\n");
    exit(1);
  }
  if((w_index = malloc((num_w+add)/w_site * sizeof(int))) == NULL){
    printf("memory error\n");
    exit(1);
  }
  if((w_rindex = malloc((num_s+num_w+add) * 3 * sizeof(double))) == NULL){
    printf("memory error\n");
    exit(1);
  }
  if((w_info = malloc((num_s+num_w+add) * sizeof(int))) == NULL){
    printf("memory error\n");
    exit(1);
  }

  if((erfct = malloc((EFT+1) * sizeof(float))) == NULL){
    printf("memory error\n");
    exit(1);
  }
  for(i = 0;i < VMAX; i++)
    if((vecn[i] = malloc(4 * sizeof(int))) == NULL){
      printf("memory error\n");
      exit(1);
    }
#ifdef GL_ON
  if((pix=(GLubyte*)malloc((X_PIX_SIZE+3)*(Y_PIX_SIZE)*3*sizeof(GLubyte)))==NULL){
    printf("memory error\n");
    exit(1);
  }
#endif

}
void fccset_w(double* side)
{
  int     i,j,k,c,i0;
  double  px,py,pz;
  double  l;

  l = side[0] / (npx * 2);

  for(i = 0;i < npz; i++)
    for(j = 0;j < npy; j++)
      for(k = 0;k < npx; k++){
        px = k * 2 * l;
        py = j * 2 * l;
        pz = i * 2 * l;
        c  = 4*3*(i*npx*npy + j*npx + k);
        cd[c    ] = px;
        cd[c  +1] = py;
        cd[c  +2] = pz;
        cd[c+3  ] = px + l;
        cd[c+3+1] = py + l;
        cd[c+3+2] = pz;
        cd[c+6  ] = px + l;
        cd[c+6+1] = py;
        cd[c+6+2] = pz + l;
        cd[c+9  ] = px;
        cd[c+9+1] = py + l;
        cd[c+9+2] = pz + l;
      }

  for(i0 = 0; i0 < w_num; i0++){
    i = w_index[i0];
    cd[i]   += l / 2.;
    cd[i+1] += l / 2.;
    cd[i+2] += l / 2.;
  }

}
void velset6(double tref,double dh,double tscale,int knum,int num)
{
  int i,j,k,c;
  double u1,u2,v1,v2,s,d;
  double spx=0,spy=0,spz=0;
  double ekin,ts,sc;

  c = 0;
  i = 0;
  while(c < num){
    u1 = (double)rand() / (double)RAND_MAX;
    u2 = (double)rand() / (double)RAND_MAX;
    v1 = 2.*u1-1.;
    v2 = 2.*u2-1.;
    s  = v1*v1 + v2*v2;
    if( s < 1 ){
      while(atype[i/3] > 2) i++;
      vl[i++] = v1*(double)sqrt((-2*log(s))/s);
      c++;
      while(atype[i/3] > 2) i++;
      if(i < n3){
        vl[i++] = v2*(double)sqrt((-2*log(s))/s);
        c++;
      }
    }
  }
/*
  for(i = 0; i < n3; i += 3)
    printf("%d % f % f % f\n",i/3,vl[i],vl[i+1],vl[i+2]);
  exit(0);
*/
  for(k = 0; k < knum; k++){
    j = 0;
    spx = spy = spz = 0;
    for(i = 0; i < n3; i += 3){
      if(atype[i/3] == k){
        spx += vl[i  ];
        spy += vl[i+1];
        spz += vl[i+2];
        j++;
      }
    }
    if(j != 0){
      spx /= j;
      spy /= j;
      spz /= j;
    }
    for(i = 0;i < n3; i += 3){
      if(atype[i/3] == k){
        vl[i  ] -= spx;
        vl[i+1] -= spy;
        vl[i+2] -= spz;
      }
    }
  }

  ekin = 0;
  for(i = 0; i < n3; i += 3){
    ekin += (vl[i]*vl[i] + vl[i+1]*vl[i+1] + vl[i+2]*vl[i+2])
      *a_mass[atype_mat[atype[i/3]]];
  }
  /*
     for(i = 0; i < n3; i+=3)
     printf("%d %d % f % f % f\n",i/3,atype[i/3],vl[i],vl[i+1],vl[i+2]);
     exit(0);
     */

  ts = tscale * ekin;
  sc = sqrt( tref / ts );
  sc *= dh;
  for(i = 0;i < n3; i += 3){
    vl[i  ] *= sc;
    vl[i+1] *= sc;
    vl[i+2] *= sc;
  }
}
#if 0
void velset6(double tref,double dh,double tscale,int knum)
{
  int i,j,k,c;
  double u1,u2,v1,v2,s,d;
  double spx=0,spy=0,spz=0;
  double ekin,ts,sc;

  c = 0;
  i = 0;
  while(c < n3/w_site){
    u1 = (double)rand() / (double)RAND_MAX;
    u2 = (double)rand() / (double)RAND_MAX;
    v1 = 2.*u1-1.;
    v2 = 2.*u2-1.;
    s  = v1*v1 + v2*v2;
    if( s < 1 ){
      vl[i  ] = v1*(double)sqrt((-2*log(s))/s);
      if((i%3) == 2) i += 3*(w_site-1)+1; else i++;
      if(i < n3){
        vl[i] = v2*(double)sqrt((-2*log(s))/s);
        if((i%3) == 2) i += 3*(w_site-1)+1; else i++;
      }
      c += 2;
    }
  }

  for(k = 0; k < knum; k++){
    j = 0;
    spx = spy = spz = 0;
    for(i = 0; i < n3; i += 3){
      if(atype[i/3] == k){
        spx += vl[i  ];
        spy += vl[i+1];
        spz += vl[i+2];
        j++;
      }
    }
    if(j != 0){
      spx /= j;
      spy /= j;
      spz /= j;
    }
    for(i = 0;i < n3; i += 3){
      if(atype[i/3] == k){
        vl[i  ] -= spx;
        vl[i+1] -= spy;
        vl[i+2] -= spz;
      }
    }
  }

  ekin = 0;
  for(i = 0; i < n3; i += 3){
    ekin += (vl[i]*vl[i] + vl[i+1]*vl[i+1] + vl[i+2]*vl[i+2])
      *a_mass[atype_mat[atype[i/3]]];
  }
  /*
     for(i = 0; i < n3; i+=3)
     printf("%d %d % f % f % f\n",i/3,atype[i/3],vl[i],vl[i+1],vl[i+2]);
     exit(0);
     */

  ts = tscale * ekin;
  sc = sqrt( tref / ts );
  sc *= dh;
  for(i = 0;i < n3; i += 3){
    vl[i  ] *= sc;
    vl[i+1] *= sc;
    vl[i+2] *= sc;
  }
}
#endif
#define VNN 9
#define VM (VNN*2+1)
#define VM3 VM*VM*VM
void vecset()
{
  int i,j,k,c;
  static int vec[VM3][4];
  
  c = 0;
  for(i = -VNN;i < VNN+1; i++)
    for(j = -VNN;j < VNN+1; j++)
      for(k = -VNN;k < VNN+1; k++){
        vec[c][0] = i*i + j*j + k*k;
        vec[c][1] = i;
        vec[c][2] = j;
        vec[c][3] = k;
        c++;
      }
/*
  for(i = 0; i < c; i++)
    printf("%d %4d %4d %4d %4d\n",i,vec[i][0],vec[i][1],vec[i][2],vec[i][3]);
*/
  c = 0;
  for(i = 1;i < 82; i++){
    for(j = (VM3-1)/2+1;j < VM3; j++)
      if(vec[j][0] == i && c < VMAX){
        vecn[c][0] = vec[j][0];
        vecn[c][1] = vec[j][1];
        vecn[c][2] = vec[j][2];
        vecn[c][3] = vec[j][3];
        c++;
      }
  }
/*
  for(i = 0; i < c; i++)
    printf("%d %4d %4d %4d %4d\n"
           ,i,vecn[i][0],vecn[i][1],vecn[i][2],vecn[i][3]);
  exit(0);
*/
}
void mitoa(int c,char str[],int len)
{
  int i,keta;
  
  for(i = len-1;i >= 0; i--){
    keta = (int)(c / pow(10.,(double)i));
    c -= keta * pow(10.,(double)i);
    str[len-1-i] = keta + '0';
  }
  str[len] = 0;
}
void potpar5(int xp,int xp2,int xm,int xm2, char keiname[]){

  char gpname[4][3] = {"Li", "Na", "K",  "Rb"};
  char gmname[4][3] = {"F",  "Cl", "Br", "I"};

  int nip[4] = {2, 8, 8, 8};
  int nim[4] = {8, 8, 8, 8};

  double sigmp[4] = {.816, 1.17, 1.463, 1.587}; /* 0:Li 1:Na 2:K  3:Rb */
  double sigmm[4] = {1.179, 1.585, 1.716, 1.907}; /* 0:F  1:Cl 2:Br 3:I  */
  /*  F     Cl    Br    I  */
  double rho[4][4] = {.299, .342, .353, .430, /* Li */
                        .330, .317, .340, .386, /* Na */
                        .338, .337, .335, .355, /* K  */
                        .328, .318, .335, .337}; /* Rb */
  double cpp[4] = {0.073, 1.68, 24.3, 59.4};
  double cmm[4][4] = {14.5, 111.0, 185.1, 378.0,
                        16.5, 116.0, 196.0, 392.0,
                        18.6, 124.5, 206.0, 403.0,
                        18.9, 130.0, 215.0, 428.0};
  double cpm[4][4] = { 0.8,  2.0,  2.5,   3.3,
                         4.5, 11.2, 13.0,  19.1,
                         19.5, 48.0, 60.0,  82.0,
                         31.0, 79.0, 99.0, 135.0};

  double dpp[4] = { 0.03, 0.8, 24.0, 82.0};
  double dmm[4][4] = {17, 223, 423, 1060,
                        20, 233, 450, 1100,
                        22, 250, 470, 1130,
                        23, 260, 490, 1200};
  double dpm[4][4] = { 0.6,   2.4,   3.3,   5.3,
                         3.8,  13.9,  19.0,  31.0,
                         21.0,  73.0,  99.0, 156.0,
                         40.0, 134.0, 180.0, 280.0};

  if(xp > 3 || xm > 3){
    printf("error\n");
    exit(1);
  }

  strcpy(keiname,gpname[xp]);
  strcat(keiname,gmname[xm]);
  if(xp2 != -1){
    strcat(keiname,gpname[xp2]);
    strcat(keiname,gmname[xm2]);
  }

  pc[1][1] = cmm[xp][xm];
  pc[1][0] = cpm[xp][xm];
  pc[0][1] = cpm[xp][xm];
  pc[0][0] = cpp[xp];
  if(xp2 != -1){
    pc[1][3] = sqrt(cmm[xp][xm]*cmm[xp2][xm2]);
    pc[3][1] = sqrt(cmm[xp][xm]*cmm[xp2][xm2]);
    pc[3][3] = cmm[xp2][xm2];
    pc[0][3] = cpm[xp][xm2];
    pc[3][0] = cpm[xp][xm2];
    pc[1][2] = cpm[xp2][xm];
    pc[2][1] = cpm[xp2][xm];
    pc[2][3] = cpm[xp2][xm2];
    pc[3][2] = cpm[xp2][xm2];
    pc[0][2] = sqrt(cpp[xp]*cpp[xp2]);
    pc[2][0] = sqrt(cpp[xp]*cpp[xp2]);
    pc[2][2] = cpp[xp2];
  }

  pd[1][1] = dmm[xp][xm];
  pd[0][1] = dpm[xp][xm];
  pd[1][0] = dpm[xp][xm];
  pd[0][0] = dpp[xp];
  if(xp2 != -1){
    pd[1][3] = sqrt(dmm[xp][xm]*dmm[xp2][xm2]);
    pd[3][1] = sqrt(dmm[xp][xm]*dmm[xp2][xm2]);
    pd[3][3] = dmm[xp2][xm2];
    pd[0][3] = dpm[xp][xm2];
    pd[3][0] = dpm[xp][xm2];
    pd[2][1] = dpm[xp2][xm];
    pd[1][2] = dpm[xp2][xm];
    pd[2][3] = dpm[xp2][xm2];
    pd[3][2] = dpm[xp2][xm2];
    pd[0][2] = sqrt(dpp[xp]*dpp[xp2]);
    pd[2][0] = sqrt(dpp[xp]*dpp[xp2]);
    pd[2][2] = dpp[xp2];
  }

  ipotro[1][1] = 1./rho[xp][xm];
  ipotro[0][1] = 1./rho[xp][xm];
  ipotro[1][0] = 1./rho[xp][xm];
  ipotro[0][0] = 1./rho[xp][xm];
  if(xp2 != -1){
    ipotro[1][3] = 1./((rho[xp][xm]+rho[xp2][xm2])/2);
    ipotro[3][1] = 1./((rho[xp][xm]+rho[xp2][xm2])/2);
    ipotro[3][3] = 1./rho[xp2][xm2];
    ipotro[0][3] = 1./rho[xp][xm2];
    ipotro[3][0] = 1./rho[xp][xm2];
    ipotro[2][1] = 1./rho[xp2][xm];
    ipotro[1][2] = 1./rho[xp2][xm];
    ipotro[2][3] = 1./rho[xp2][xm2];
    ipotro[3][2] = 1./rho[xp2][xm2];
    ipotro[0][2] = 1./((rho[xp][xm]+rho[xp2][xm2])/2);
    ipotro[2][0] = 1./((rho[xp][xm]+rho[xp2][xm2])/2);
    ipotro[2][2] = 1./rho[xp2][xm2];
  }

  sigm[1][1] = sigmm[xm]*2;
  sigm[0][1] = sigmp[xp] + sigmm[xm];
  sigm[1][0] = sigmp[xp] + sigmm[xm];
  sigm[0][0] = sigmp[xp]*2;
  if(xp2 != -1){
    sigm[1][3] = sigmm[xm]+sigmm[xm2];
    sigm[3][1] = sigmm[xm]+sigmm[xm2];
    sigm[3][3] = sigmm[xm2]*2;
    sigm[0][3] = sigmp[xp] + sigmm[xm2];
    sigm[3][0] = sigmp[xp] + sigmm[xm2];
    sigm[1][2] = sigmp[xp2] + sigmm[xm];
    sigm[2][1] = sigmp[xp2] + sigmm[xm];
    sigm[2][3] = sigmp[xp2] + sigmm[xm2];
    sigm[3][2] = sigmp[xp2] + sigmm[xm2];
    sigm[0][2] = sigmp[xp]+sigmp[xp2];
    sigm[2][0] = sigmp[xp]+sigmp[xp2];
    sigm[2][2] = sigmp[xp2]*2;
  }

  pol[1][1] = -1./nim[xm] - 1./nim[xm] + 1;
  pol[0][1] =  1./nip[xp] - 1./nim[xm] + 1;
  pol[1][0] =  1./nip[xp] - 1./nim[xm] + 1;
  pol[0][0] =  1./nip[xp] + 1./nip[xp] + 1;
  if(xp2 != -1){
    pol[1][3] =  -1./nim[xm] - 1./nim[xm2] + 1;
    pol[3][1] =  -1./nim[xm] - 1./nim[xm2] + 1;
    pol[3][3] =  -1./nim[xm2] - 1./nim[xm2] + 1;
    pol[0][3] =  1./nip[xp] - 1./nim[xm2] + 1;
    pol[3][0] =  1./nip[xp] - 1./nim[xm2] + 1;
    pol[1][2] =  1./nip[xp2] - 1./nim[xm] + 1;
    pol[2][1] =  1./nip[xp2] - 1./nim[xm] + 1;
    pol[2][3] =  1./nip[xp2] - 1./nim[xm2] + 1;
    pol[3][2] =  1./nip[xp2] - 1./nim[xm2] + 1;
    pol[0][2] =  1./nip[xp] + 1./nip[xp2] + 1;
    pol[2][0] =  1./nip[xp] + 1./nip[xp2] + 1;
    pol[2][2] =  1./nip[xp2] + 1./nip[xp2] + 1;
  }
}
void fccset2(int lnp,double lside,double cod[])
{
  int     i,j,k,c;
  double  px,py,pz;
  double  l;

  l = lside / (lnp * 2);

  for(i = 0;i < lnp; i++)
    for(j = 0;j < lnp; j++)
      for(k = 0;k < lnp; k++){
        px = k * 2 * l;
        py = j * 2 * l;
        pz = i * 2 * l;
        c  = 4*3*(i*lnp*lnp + j*lnp + k);
        cod[c    ] = px;
        cod[c  +1] = py;
        cod[c  +2] = pz;
        cod[c+3  ] = px + l;
        cod[c+3+1] = py + l;
        cod[c+3+2] = pz;
        cod[c+6  ] = px + l;
        cod[c+6+1] = py;
        cod[c+6+2] = pz + l;
        cod[c+9  ] = px;
        cod[c+9+1] = py + l;
        cod[c+9+2] = pz + l;
      }
  for(i = 0;i < n3; i++)
    cod[i] += l / 2.;
}
double mass_den3(int xp, int xp2, int xm, int xm2, double comp, double temp)
{
  double nden1, nden2, nden;

  double p_atomnum[4] = {6.941, 22.989768, 39.0983, 85.4678}; /* atomic weight + */
  double m_atomnum[4] = {18.9984032, 35.4527, 79.904, 126.90447};

  /*  F       Cl      Br      I   */
  double a[4][4] = {2.3768, 1.8842, 3.0658, 3.7902, /* Li */ /* density */
                      2.655,  2.1393, 3.1748, 3.6274, /* Na */
                      2.6464, 2.1359, 2.9583, 3.3594, /* K  */
                      0.0000, 3.1210, 3.7390, 3.9449}; /* Rb */
  double b[4][4] = {0.4902, 0.4328, 0.6520, 0.9176,
                      0.560,  0.5430, 0.9169, 0.9491,
                      0.6515, 0.5831, 0.8253, 0.9557,
                      0.0000, 0.8832, 1.0718, 1.1435};

  double a_s[4][4] = {0,    0,     0,    0, /* density of solid */
                        0,    2.168, 0,    0,
                        0,    1.985, 0,    0,
                        0,    0,     0,    0};
  double b_s[4][4] = {0,    0,     0,    0,
                        0,    1.267, 0,    0,
                        0,    .5459, 0,    0,
                        0,    0,     0,    0};
  double c_s[4][4] = {0,    0,     0,    0,
                        0,    1.754, 0,    0,
                        0,    1.836, 0,    0,
                        0,    0,     0,    0};

  if(xp2 == -1){
    return((a[xp][xm] - b[xp][xm]*1e-3*temp)/((a_mass[0]+a_mass[1])/2*mass)*1e-27);
  } else {
    nden1 = (a[xp][xm] - b[xp][xm]*1e-3*temp)/((a_mass[0]+a_mass[1])/2*mass)*1e-27;
    nden2 = (a[xp2][xm2] - b[xp2][xm2]*1e-3*temp)/((a_mass[2]+a_mass[3])/2*mass)*1e-27;
    return( nden1*(100-comp)/100 + nden2*comp/100);
  }
}
double nden_set(double tmp)
{
  int i;
  double den[] = {   0.99984, 0.99990, 0.99994, 0.99996, 0.99997,
                       0.99996, 0.99994, 0.99990, 0.99985, 0.99978,
                       0.99970, 0.99961, 0.99949, 0.99938, 0.99924,
                       0.99910, 0.99894, 0.99877, 0.99860, 0.99841,
                       0.99820, 0.99799, 0.99777, 0.99754, 0.99730,
                       0.99704, 0.99678, 0.99651, 0.99623, 0.99594,
                       0.99565, 0.99534, 0.99503, 0.99470, 0.99437,
                       0.99403, 0.99368, 0.99333, 0.99297, 0.99259,
                       0.99222, 0.99183, 0.99144, 0.99104, 0.99063,
                       0.99021, 0.98979, 0.98936, 0.98893, 0.98849,
                       0.98804, 0.98758, 0.98715, 0.98665, 0.98618,
                       0.98570, 0.98521, 0.98471, 0.98422, 0.98371,
                       0.98320, 0.98268, 0.98216, 0.98163, 0.98110,
                       0.98055, 0.98001, 0.97946, 0.97890, 0.97834,
                       0.97777, 0.97720, 0.97662, 0.97603, 0.97544,
                       0.97485, 0.97425, 0.97364, 0.97303, 0.97242,
                       0.97180, 0.97117, 0.97054, 0.96991, 0.96927,
                       0.96862, 0.96797, 0.96731, 0.96665, 0.96600,
                       0.96532, 0.96465, 0.96379, 0.96328, 0.96259,
                       0.96190, 0.96120, 0.96050, 0.95979, 0.95906};

  if(temp >= 273 && temp <= 373)
    return(den[(int)(tmp+.5)]);
  else 
    return 0.917;
}
void ice_set(double *side)
{
  int i,j,k;
  int i0,i1,i2,i3;
  int c;
  double a;
  double s3,s6;
  double ang0,ang1,ang2,ang3;
  double d0,d1,d2,d3,d4,d5;

  s3 = sqrt(3.0);
  s6 = sqrt(6.0);
  a = 4.52;

  d0 = a;
  d1 = a*s3;
  d2 = a*s6/3.0*2.0;

  if(nden > 0)
    a /= pow(nden/(8.0/(d0*d1*d2)),1./3.);
  else
    printf("You must set number of density");

  for(i = 0; i < n3; i++)
    cd[i] = 0;
  for(i = 0; i < n1*4; i++)
    ang[i] = 0;

  i = 0;
  cd[i] = a*.5; cd[i+1] = s3/6.*a; cd[i+2] = s6/12.*a;
  atype[i/3] = 2;
  i += 3;
  cd[i] = a*.5; cd[i+1] = s3/6.*a; cd[i+2] = s6/3.*a;
  atype[i/3] = 2;
  i += 3;
  cd[i] = a*.5; cd[i+1] =-s3/2.*a; cd[i+2] = 0;
  atype[i/3] = 2;
  i += 3;
  cd[i] = a*.5; cd[i+1] =-s3/2.*a; cd[i+2] = (s6/3.+s6/12.)*a;
  atype[i/3] = 2;
  i += 3;
  cd[i] = 0; cd[i+1] = 0; cd[i+2] = 0;
  atype[i/3] = 2;
  i += 3;
  cd[i] = 0; cd[i+1] = 0; cd[i+2] = (s6/3.+s6/12.)*a;
  atype[i/3] = 2;
  i += 3;
  cd[i] = 0; cd[i+1] =-s3/3.*a; cd[i+2] = s6/12.*a;
  atype[i/3] = 2;
  i += 3;
  cd[i] = 0; cd[i+1] =-s3/3.*a; cd[i+2] = s6/3.*a;
  atype[i/3] = 2;

  c = 8*3;
  for(i0 = 0; i0 < npz; i0++)
    for(i1 = 0; i1 < npy; i1++)
      for(i2 = 0; i2 < npx; i2++)
        if(i0 != 0 || i1 != 0 || i2 != 0)
          for(i3 = 0; i3 < 24; i3 += 3){
            cd[c]   = cd[i3]  +i2*a;
            cd[c+1] = cd[i3+1]+i1*a*s3;
            cd[c+2] = cd[i3+2]+i0*a*s6/3*2;
            atype[c/3] = 2;
            /*
               printf("%d 0 % f % f % f % f % f % f %d %d %d\n"
               ,c/3,cd[c],cd[c+1],cd[c+2]
               ,cd[i3],cd[i3+1],cd[i3+2],i0,i1,i2);
            */
            c += 3;
          }

  side[0] = a*npx;
  side[1] = a*s3*npy;
  side[2] = a*s6/3.0*2.0*npz;

  for(i = 0; i < 8*4; i += 4){
    ang0 = 120./180.*PI;
    ang1 = (90.-54.74)/180.*PI;
    ang2 = 0;
    ang[i  ] = sin(ang1/2)*sin((ang2-ang0)/2);
    ang[i+1] = sin(ang1/2)*cos((ang2-ang0)/2);
    ang[i+2] = cos(ang1/2)*sin((ang2+ang0)/2);
    ang[i+3] = cos(ang1/2)*cos((ang2+ang0)/2);
  }

  i = 0*4;
  ang0 = 120./180.*PI;
  ang1 = (90.-54.74)/180.*PI;
  ang2 = 0;
  ang[i  ] = sin(ang1/2)*sin((ang2-ang0)/2);
  ang[i+1] = sin(ang1/2)*cos((ang2-ang0)/2);
  ang[i+2] = cos(ang1/2)*sin((ang2+ang0)/2);
  ang[i+3] = cos(ang1/2)*cos((ang2+ang0)/2);
  i = 1*4;
  ang0 = 30./180.*PI;
  ang1 = 90./180.*PI;
  ang2 = -109.47/2/180*PI;
  ang[i  ] = sin(ang1/2)*sin((ang2-ang0)/2);
  ang[i+1] = sin(ang1/2)*cos((ang2-ang0)/2);
  ang[i+2] = cos(ang1/2)*sin((ang2+ang0)/2);
  ang[i+3] = cos(ang1/2)*cos((ang2+ang0)/2);
  i = 2*4;
  ang0 = 30./180.*PI;
  ang1 = 90./180.*PI;
  ang2 = 109.47/2/180*PI;
  ang[i  ] = sin(ang1/2)*sin((ang2-ang0)/2);
  ang[i+1] = sin(ang1/2)*cos((ang2-ang0)/2);
  ang[i+2] = cos(ang1/2)*sin((ang2+ang0)/2);
  ang[i+3] = cos(ang1/2)*cos((ang2+ang0)/2);
  i = 3*4;
  ang0 = 60./180.*PI;
  ang1 = (90.-54.74)/180.*PI;
  ang2 = 0;
  ang[i  ] = sin(ang1/2)*sin((ang2-ang0)/2);
  ang[i+1] = sin(ang1/2)*cos((ang2-ang0)/2);
  ang[i+2] = cos(ang1/2)*sin((ang2+ang0)/2);
  ang[i+3] = cos(ang1/2)*cos((ang2+ang0)/2);
  i = 4*4;
  ang0 = 30./180.*PI;
  ang1 = 90./180.*PI;
  ang2 = 109.47/2/180*PI;
  ang[i  ] = sin(ang1/2)*sin((ang2-ang0)/2);
  ang[i+1] = sin(ang1/2)*cos((ang2-ang0)/2);
  ang[i+2] = cos(ang1/2)*sin((ang2+ang0)/2);
  ang[i+3] = cos(ang1/2)*cos((ang2+ang0)/2);
  i = 5*4;
  ang0 = -60./180.*PI;
  ang1 = (90.-54.74)/180.*PI;
  ang2 = 0;
  ang[i  ] = sin(ang1/2)*sin((ang2-ang0)/2);
  ang[i+1] = sin(ang1/2)*cos((ang2-ang0)/2);
  ang[i+2] = cos(ang1/2)*sin((ang2+ang0)/2);
  ang[i+3] = cos(ang1/2)*cos((ang2+ang0)/2);
  i = 6*4;
  ang0 = 120./180.*PI;
  ang1 = (90.-54.74)/180.*PI;
  ang2 = 0;
  ang[i  ] = sin(ang1/2)*sin((ang2-ang0)/2);
  ang[i+1] = sin(ang1/2)*cos((ang2-ang0)/2);
  ang[i+2] = cos(ang1/2)*sin((ang2+ang0)/2);
  ang[i+3] = cos(ang1/2)*cos((ang2+ang0)/2);
  i = 7*4;
  ang0 = -30./180.*PI;
  ang1 = 90./180.*PI;
  ang2 = 109.47/2/180*PI;
  ang[i  ] = sin(ang1/2)*sin((ang2-ang0)/2);
  ang[i+1] = sin(ang1/2)*cos((ang2-ang0)/2);
  ang[i+2] = cos(ang1/2)*sin((ang2+ang0)/2);
  ang[i+3] = cos(ang1/2)*cos((ang2+ang0)/2);
        
  c = 32;
  for(i0 = 0; i0 < npz; i0++)
    for(i1 = 0; i1 < npy; i1++)
      for(i2 = 0; i2 < npx; i2++)
        if(i0 != 0 || i1 != 0 || i2 != 0)
          for(i3 = 0; i3 < 32; i3++){
            ang[c++] = ang[i3];
          }
/*
  for(i = 0, j = 0; i < n3; i += 3*w_site, j += 4){
    ang0 = ang[j  ];
    ang1 = ang[j+1];
    ang2 = ang[j+2];
    ang3 = ang[j+3];
    for(k = 0; k < w_site-1; k++){
      d0 = m_cdx[k]*(-ang0*ang0+ang1*ang1-ang2*ang2+ang3*ang3)
          +m_cdy[k]*(-2)*(ang0*ang1+ang2*ang3)
          +m_cdz[k]*( 2)*(ang1*ang2-ang0*ang3);
      d1 = m_cdx[k]*  2 *(ang2*ang3-ang0*ang1)
          +m_cdy[k]*( ang0*ang0-ang1*ang1-ang2*ang2+ang3*ang3)
          +m_cdz[k]*(-2)*(ang0*ang2+ang1*ang3);
      d2 = m_cdx[k]*  2 *(ang1*ang2+ang0*ang3)
          +m_cdy[k]*  2 *(ang1*ang3-ang0*ang2)
          +m_cdz[k]*(-ang0*ang0-ang1*ang1+ang2*ang2+ang3*ang3);
      cd[(k+1)*3+i  ] = cd[i  ] + d0;
      cd[(k+1)*3+i+1] = cd[i+1] + d1;
      cd[(k+1)*3+i+2] = cd[i+2] + d2;
      atype[k+1+i/3] = k+3;
    }
  }
*/
    for(i0 = 0; i0 < w_num; i0++){
      i = w_index[i0];
      j = i0*4;
      c = w_info[i/3]*3;
      ang0 = ang[j  ];
      ang1 = ang[j+1];
      ang2 = ang[j+2];
      ang3 = ang[j+3];
      for(k = 0; k < w_site-1; k++){
        d0 = m_cdx[k]*(-ang0*ang0+ang1*ang1-ang2*ang2+ang3*ang3)
            +m_cdy[k]*(-2)*(ang0*ang1+ang2*ang3)
            +m_cdz[k]*( 2)*(ang1*ang2-ang0*ang3);
        d1 = m_cdx[k]*  2 *(ang2*ang3-ang0*ang1)
            +m_cdy[k]*( ang0*ang0-ang1*ang1-ang2*ang2+ang3*ang3)
            +m_cdz[k]*(-2)*(ang0*ang2+ang1*ang3);
        d2 = m_cdx[k]*  2 *(ang1*ang2+ang0*ang3)
            +m_cdy[k]*  2 *(ang1*ang3-ang0*ang2)
            +m_cdz[k]*(-ang0*ang0-ang1*ang1+ang2*ang2+ang3*ang3);
        cd[k*3+c  ] = cd[i  ] + d0;
        cd[k*3+c+1] = cd[i+1] + d1;
        cd[k*3+c+2] = cd[i+2] + d2;
      }
    }
    /*
  for(i = 0; i < n3; i += 3)
    printf("%d %d %f %f %f\n",i/3,atype[i/3],cd[i],cd[i+1],cd[i+2]);
  exit(0);
*/
  d0 = 100;  d1 = 0;
  d2 = 100;  d3 = 0;
  d4 = 100;  d5 = 0;
  for(i = 0; i <  n3; i+=3){
    /*    printf("%d %d % f % f % f\n",i/3,atype[i/3],cd[i],cd[i+1],cd[i+2]);*/
    if(cd[i]   < d0) d0 = cd[i];
    if(cd[i]   > d1) d1 = cd[i];
    if(cd[i+1] < d2) d2 = cd[i+1];
    if(cd[i+1] > d3) d3 = cd[i+1];
    if(cd[i+2] < d4) d4 = cd[i+2];
    if(cd[i+2] > d5) d5 = cd[i+2];
  }
  /*
     printf("%f %f %f  %f %f %f  %f %f %f\n",d0,d1,d1-d0,d2,d3,d3-d2,d4,d5,d5-d4);
     printf("%f %f %f\n",side[0]-(d1-d0),side[1]-(d3-d2),side[2]-(d5-d4));
     */
  for(i = 0; i <  n3; i+=3){
    cd[i]   += (side[0]-(d1-d0))/2.-d0;
    cd[i+1] += (side[1]-(d3-d2))/2.-d2;
    cd[i+2] += (side[2]-(d5-d4))/2.-d4;
    /*    printf("%d %d % f % f % f\n",i/3,atype[i/3],cd[i],cd[i+1],cd[i+2]);*/
  }

}
void ice_set2(double* side)
{
  int i,j,k;
  int i0,i1,i2,i3;
  int c;
  double ang0,ang1,ang2,ang3;
  double d0,d1,d2,d3,d4,d5;

  double l;

  l = side[0] / (npx * 2);

  cd[0]   = 0;
  cd[1]   = 0;
  cd[2]   = 0;
  cd[3  ] = l;
  cd[3+1] = l;
  cd[3+2] = 0;
  cd[6  ] = l;
  cd[6+1] = 0;
  cd[6+2] = l;
  cd[9  ] = 0;
  cd[9+1] = l;
  cd[9+2] = l;

  for(i = 4*3; i < 8*3; i++)
    cd[i] = cd[i-4*3] + l/2;

  for(i = 0; i < 8*3; i++)
    cd[i] += l/4;

  c = 8*3;
  for(i0 = 0; i0 < npz; i0++)
    for(i1 = 0; i1 < npy; i1++)
      for(i2 = 0; i2 < npx; i2++)
        if(i0 != 0 || i1 != 0 || i2 != 0)
          for(i3 = 0; i3 < 24; i3 += 3){
            cd[c]   = cd[i3]  +(double)i2*side[0]/npx;
            cd[c+1] = cd[i3+1]+(double)i1*side[1]/npy;
            cd[c+2] = cd[i3+2]+(double)i0*side[2]/npz;
/*
            printf("%d %d %d %f %f %f %f %f %f\n",i2,i1,i0
                   ,cd[c],cd[c+1],cd[c+2]
                   ,cd[i3],cd[i3+1],cd[i3+2]
                   ,(double)i2*side[0]/npx
                   ,(double)i1*side[1]/npy
                   ,(double)i0*side[2]/npz
                   );
*/
/*          atype[c/3] = 2;*/
            c += 3;
          }
  /*
  for(i = 0; i < n3; i += 3)
    printf("%d %d %f %f %f\n",i/3,atype[i/3],cd[i],cd[i+1],cd[i+2]);
  exit(0);
  */
/*
  for(i = 0; i < w_num3/2; i += 3){
    cd[i]   -= side[0]/npx/8;
    cd[i+1] -= side[1]/npy/8;
    cd[i+2] -= side[2]/npz/8;
    cd[i+w_num3/2]   = cd[i]  + side[0]/npx/4;
    cd[i+w_num3/2+1] = cd[i+1]+ side[1]/npy/4;
    cd[i+w_num3/2+2] = cd[i+2]+ side[1]/npy/4;
    if (cd[i+w_num3/2]   < 0)       cd[i+w_num3/2]   += side[0];
    if (cd[i+w_num3/2]   > side[0]) cd[i+w_num3/2]   -= side[0];
    if (cd[i+w_num3/2+1] < 0)       cd[i+w_num3/2+1] += side[1];
    if (cd[i+w_num3/2+1] > side[1]) cd[i+w_num3/2+1] -= side[1];
    if (cd[i+w_num3/2+2] < 0)       cd[i+w_num3/2+2] += side[2];
    if (cd[i+w_num3/2+2] > side[2]) cd[i+w_num3/2+2] -= side[2];
  }
*/

  c = 0;
  for(i = 0; i < n1; i++){
    if(atype[i] == 2){
      if((i % 8) < 4){
        ang0 =-45*PI/180;
        ang1 = 90*PI/180;
        ang2 = 0*PI/180;
      } else {
        ang0 = 45*PI/180;
        ang1 = 90*PI/180;
        ang2 = 0*PI/180;
      }
      ang[c  ] = sin(ang1/2)*sin((ang2-ang0)/2);
      ang[c+1] = sin(ang1/2)*cos((ang2-ang0)/2);
      ang[c+2] = cos(ang1/2)*sin((ang2+ang0)/2);
      ang[c+3] = cos(ang1/2)*cos((ang2+ang0)/2);
      angh[c  ] = ang[c  ];
      angh[c+1] = ang[c+1];
      angh[c+2] = ang[c+2];
      angh[c+3] = ang[c+3];
      c += 4;
    }
  }

  /*
  for(i = 0; i < 4*4; i += 4){
    ang0 =-45*PI/180;
    ang1 = 90*PI/180;
    ang2 = 0*PI/180;
    ang[i  ] = sin(ang1/2)*sin((ang2-ang0)/2);
    ang[i+1] = sin(ang1/2)*cos((ang2-ang0)/2);
    ang[i+2] = cos(ang1/2)*sin((ang2+ang0)/2);
    ang[i+3] = cos(ang1/2)*cos((ang2+ang0)/2);
    angh[i  ] = ang[i  ];
    angh[i+1] = ang[i+1];
    angh[i+2] = ang[i+2];
    angh[i+3] = ang[i+3];
  }
  for(i = 4*4; i < 8*4; i += 4){
    ang0 = 45*PI/180;
    ang1 = 90*PI/180;
    ang2 = 0*PI/180;
    ang[i  ] = sin(ang1/2)*sin((ang2-ang0)/2);
    ang[i+1] = sin(ang1/2)*cos((ang2-ang0)/2);
    ang[i+2] = cos(ang1/2)*sin((ang2+ang0)/2);
    ang[i+3] = cos(ang1/2)*cos((ang2+ang0)/2);
    angh[i  ] = ang[i  ];
    angh[i+1] = ang[i+1];
    angh[i+2] = ang[i+2];
    angh[i+3] = ang[i+3];
  }

  c = 32;
  for(i0 = 0; i0 < npz; i0++)
    for(i1 = 0; i1 < npy; i1++)
      for(i2 = 0; i2 < npx; i2++)
        if(i0 != 0 || i1 != 0 || i2 != 0)
          for(i3 = 0; i3 < 32; i3++){
            ang[c++] = ang[i3];
          }
  */
/*
  for(i = 0; i < w_num*4; i += 4)
    printf("%d % f % f % f\n",i/4,ang[i],ang[i+1],ang[i+2],ang[i+3]);
  exit(0);
*/
}
int strsrc2(char str[],char key[], double *d)
{
  int i;
  int len;
  char *buf;
  char val[256];

  i = 0;
  while(key[i++]);
  len = i - 1;
  if((buf = strstr(str,key)) == NULL)
    return(0);
  i = 0;
  while((val[i] = (buf+len)[i]) != ' ' && (val[i] = (buf+len)[i]) != 0)
    i++;
  val[i] = 0;

  *d = atof(val);
  return(1);
}


