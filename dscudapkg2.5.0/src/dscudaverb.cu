#include <stdio.h>
#include <string.h>
#include <vector_types.h>
#include "dscuda.h"
#include "dscudaverb.h"

static dscudaVerbHist *verbHists = NULL;
static int verbHistNum = 0;
static int verbHistMax = 0;
int verbModuleId;

typedef enum {
  DSCVMethodNone = 0,
  DSCVMethodSetDevice,
  DSCVMethodMemcpyH2D,
  DSCVMethodMemcpyD2D,
  DSCVMethodMemcpyD2H,
  DSCVMethodMemcpyToSymbolH2D,
  DSCVMethodMemcpyToSymbolD2D,
  DSCVMethodLaunchKernel,

  DSCVMethodConfigureCall,
  DSCVMethodSetupArgument,
  DSCVMethodSetupArgumentOfTypeP,
  DSCVMethodDscudaLaunchWrapper,

  DSCVMethodEnd
} DSCVMethod;

//stubs for store/release args, and recall functions
static void *(*storeArgsStub[DSCVMethodEnd])(void *);
static void (*releaseArgsStub[DSCVMethodEnd])(void *);
static void (*recallStub[DSCVMethodEnd])(void *);

#define DSCUDAVERB_SET_STUBS(mthd) \
  storeArgsStub[DSCVMethod ## mthd] = store ## mthd; \
  releaseArgsStub[DSCVMethod ## mthd] = release ## mthd; \
  recallStub[DSCVMethod ## mthd] = recall ## mthd;

#define DSCUDAVERB_SET_ARGS(mthd) \
  cuda ## mthd ## Args *argsrc; \
  argsrc = (cuda ## mthd ## Args *)argp;

#define DSCUDAVERB_STORE_ARGS(mthd) \
  DSCUDAVERB_SET_ARGS(mthd); \
  cuda ## mthd ## Args *argdst; \
  argdst = (cuda ## mthd ## Args *)malloc(sizeof(cuda ## mthd ## Args)); \
  *argdst = *(cuda ## mthd ## Args *)argp;


//mapping RPCfunctionID to DSCUDAVerbMethodID
static DSCVMethod funcID2DSCVMethod(int funcID) {
  switch (funcID) {
    case RCMethodSetDevice:
      return DSCVMethodSetDevice;
    case RCMethodMemcpyH2D:
      return DSCVMethodMemcpyH2D;
    case RCMethodMemcpyD2D:
      return DSCVMethodMemcpyD2D;
    case RCMethodMemcpyD2H:
      return DSCVMethodMemcpyD2H;
    case RCMethodDscudaMemcpyToSymbolH2D:
      return DSCVMethodMemcpyToSymbolH2D;
    case RCMethodDscudaMemcpyToSymbolD2D:
      return DSCVMethodMemcpyToSymbolD2D;
    case RCMethodDscudaLaunchKernel:
      return DSCVMethodLaunchKernel;
    case RCMethodConfigureCall:
      return DSCVMethodConfigureCall;
    case RCMethodSetupArgument:
      return DSCVMethodSetupArgument;
    case RCMethodSetupArgumentOfTypeP:
      return DSCVMethodSetupArgumentOfTypeP;
    case RCMethodDscudaLaunch:
      return DSCVMethodDscudaLaunchWrapper;
    default:
      WARN(0, "funcID2DSCVMethod:invalud funcID (=%d)\n", funcID);
      return DSCVMethodNone;
  }
}

//stubs for store args
static void *
storeSetDevice(void *argp) {
  WARN(3, "add hist cudaSetDevice\n");
  DSCUDAVERB_STORE_ARGS(SetDevice); 
  return argdst;
}

static void *
storeMemcpyH2D(void *argp) {
  WARN(3, "add hist cudaMemcpyH2D\n");
  DSCUDAVERB_STORE_ARGS(Memcpy);
  argdst->src = malloc(argsrc->count + 1);
  memcpy(argdst->src, (const void *)argsrc->src, argsrc->count);
  return argdst;
}

static void *
storeMemcpyD2D(void *argp) {
  WARN(3, "add hist cudaMemcpyD2D\n");
  DSCUDAVERB_STORE_ARGS(Memcpy);
  return argdst;
}

static void *
storeMemcpyD2H(void *argp) {
  WARN(3, "add hist cudaMemcpyD2H\n");
  DSCUDAVERB_STORE_ARGS(Memcpy);
  return argdst;
}

static void *
storeMemcpyToSymbolH2D(void *argp) {
  WARN(3, "add hist cudaMemcpyToSymbolH2D\n");
  DSCUDAVERB_STORE_ARGS(MemcpyToSymbol);

  int nredundancy = dscudaNredundancy();
  argdst->moduleid = (int *)malloc(sizeof(int) * nredundancy);
  memcpy(argdst->moduleid, argsrc->moduleid, sizeof(int) * nredundancy);

  argdst->symbol = (char *)malloc(sizeof(char) * (strlen(argsrc->symbol) + 1));
  argdst->src = malloc(argsrc->count);

  strcpy(argdst->symbol, argsrc->symbol);
  memcpy(argdst->src, argsrc->src, argsrc->count);

  return argdst;
}

static void *
storeMemcpyToSymbolD2D(void *argp) {
  WARN(3, "add hist cudaMemcpyToSymbolD2D\n");
  DSCUDAVERB_STORE_ARGS(MemcpyToSymbol);

  int nredundancy = dscudaNredundancy();
  argdst->moduleid = (int *)malloc(sizeof(int) * nredundancy);
  memcpy(argdst->moduleid, argsrc->moduleid, sizeof(int) * nredundancy);

  argdst->symbol = (char *)malloc(sizeof(char) * (strlen(argsrc->symbol) + 1));
  strcpy(argdst->symbol, argsrc->symbol);

  return argdst;
}

static void *
storeLaunchKernel(void *argp)
{
  WARN(3, "add hist IbvLaunchKernel\n");
  DSCUDAVERB_STORE_ARGS(LaunchKernel);

  int nredundancy = dscudaNredundancy();
  argdst->moduleid = (int *)malloc(sizeof(int) * nredundancy);
  memcpy(argdst->moduleid, argsrc->moduleid, sizeof(int) * nredundancy);

  argdst->kname = (char *)malloc(sizeof(char) * strlen(argsrc->kname) + 1);
  strcpy(argdst->kname, argsrc->kname);

  argdst->gdim = (int *)malloc(sizeof(dim3));
  argdst->bdim = (int *)malloc(sizeof(dim3));
  memcpy(argdst->gdim, argsrc->gdim, sizeof(dim3));
  memcpy(argdst->bdim, argsrc->bdim, sizeof(dim3));

  int narg = argsrc->narg;
  RCArg *argbuf = (RCArg *)malloc(sizeof(RCArg) * narg);
  memcpy(argbuf, argsrc->arg, sizeof(RCArg) * narg);
  argdst->arg = argbuf;

  return argdst;
}

static void *
storeConfigureCall(void *argp)
{
  WARN(3, "add hist cudaConfigureCall\n");
  DSCUDAVERB_STORE_ARGS(ConfigureCall);

  argdst->gdim = (int *)malloc(sizeof(int) * 3);
  argdst->bdim = (int *)malloc(sizeof(int) * 3);
  memcpy(argdst->gdim, argsrc->gdim, sizeof(int) * 3);
  memcpy(argdst->bdim, argsrc->bdim, sizeof(int) * 3);

  return argdst;
}

static void *
storeSetupArgument(void *argp)
{
  WARN(3, "add hist cudaSetupArgument\n");
  DSCUDAVERB_STORE_ARGS(SetupArgument);

  argdst->arg = (void *)malloc(argsrc->size);
  memcpy(argdst->arg, argsrc->arg, argsrc->size);

  return argdst;
}

static void *
storeSetupArgumentOfTypeP(void *argp)
{
  WARN(3, "add hist cudaSetupArgumentOfTypeP\n");
  DSCUDAVERB_STORE_ARGS(SetupArgumentOfTypeP);

  //  argdst->arg = (void *)malloc(argsrc->size);
  //  memcpy(argdst->arg, argsrc->arg, argsrc->size);

  return argdst;
}

static void *
storeDscudaLaunchWrapper(void *argp)
{
  WARN(3, "add hist dscudaLaunchWrapper\n");
  DSCUDAVERB_STORE_ARGS(DscudaLaunchWrapper);

  argdst->key = (char *)malloc(strlen(argsrc->key) + 1);
  strncpy(argdst->key, argsrc->key, strlen(argsrc->key) + 1);

  return argdst;
}


//stubs for release args
static void
releaseSetDevice(void *argp) {
  DSCUDAVERB_SET_ARGS(SetDevice);
  free(argsrc);
}

static void
releaseMemcpyH2D(void *argp) {
  DSCUDAVERB_SET_ARGS(Memcpy);
  free(argsrc->src);
  free(argsrc);
}

static void
releaseMemcpyD2D(void *argp) {
  DSCUDAVERB_SET_ARGS(Memcpy);
  free(argsrc);
}

static void
releaseMemcpyD2H(void *argp) {
  DSCUDAVERB_SET_ARGS(Memcpy);
  free(argsrc);
}

static void
releaseMemcpyToSymbolH2D(void *argp) {
  DSCUDAVERB_SET_ARGS(MemcpyToSymbol);
  free(argsrc->moduleid);
  free(argsrc->symbol);
  free(argsrc->src);
  free(argsrc);
}

static void
releaseMemcpyToSymbolD2D(void *argp) {
  DSCUDAVERB_SET_ARGS(MemcpyToSymbol);
  free(argsrc->moduleid);
  free(argsrc->symbol);
  free(argsrc);

}

static void
releaseLaunchKernel(void *argp) {
  DSCUDAVERB_SET_ARGS(LaunchKernel);
  free(argsrc->moduleid);
  free(argsrc->kname);
  free(argsrc->gdim);
  free(argsrc->bdim);
  free(argsrc->arg);
  free(argsrc);
}

static void
releaseConfigureCall(void *argp) {
  DSCUDAVERB_SET_ARGS(ConfigureCall);
  free(argsrc->gdim);
  free(argsrc->bdim);
}

static void
releaseSetupArgument(void *argp) {
  DSCUDAVERB_SET_ARGS(SetupArgument);
  free(argsrc->arg);
}

static void
releaseSetupArgumentOfTypeP(void *argp) {
  DSCUDAVERB_SET_ARGS(SetupArgumentOfTypeP);
  //  free(argsrc->arg);
}

static void
releaseDscudaLaunchWrapper(void *argp) {
  DSCUDAVERB_SET_ARGS(DscudaLaunchWrapper);
  free(argsrc->key);
}

//stubs for recall
static void
recallSetDevice(void *argp) {
  DSCUDAVERB_SET_ARGS(SetDevice);
  WARN(3, "recall cudaSetDevice\n");
  cudaSetDevice(argsrc->device);
}

static void
recallMemcpyH2D(void *argp) {
  DSCUDAVERB_SET_ARGS(Memcpy);
  WARN(3, "recall cudaMemcpyH2D\n");
  cudaMemcpy(argsrc->dst, argsrc->src, argsrc->count, cudaMemcpyHostToDevice);
}

static void
recallMemcpyD2D(void *argp) {
  DSCUDAVERB_SET_ARGS(Memcpy);
  WARN(3, "recall cudaMemcpyD2D\n");
  cudaMemcpy(argsrc->dst, argsrc->src, argsrc->count, cudaMemcpyDeviceToDevice);
}

static void
recallMemcpyD2H(void *argp) {
  DSCUDAVERB_SET_ARGS(Memcpy);
  WARN(3, "recall cudaMemcpyD2H\n");
  cudaMemcpy(argsrc->dst, argsrc->src, argsrc->count, cudaMemcpyDeviceToHost);
}

static void
recallMemcpyToSymbolH2D(void *argp) {
  DSCUDAVERB_SET_ARGS(MemcpyToSymbol);
  WARN(3, "recall cudaMemcpyToSymbolH2D\n");
  dscudaMemcpyToSymbolWrapper(argsrc->moduleid, argsrc->symbol, argsrc->src, argsrc->count, argsrc->offset, cudaMemcpyHostToDevice);
}

static void
recallMemcpyToSymbolD2D(void *argp) {
  DSCUDAVERB_SET_ARGS(MemcpyToSymbol);
  WARN(3, "recall cudaMemcpyToSymbolD2D\n");
  dscudaMemcpyToSymbolWrapper(argsrc->moduleid, argsrc->symbol, argsrc->src, argsrc->count, argsrc->offset, cudaMemcpyDeviceToDevice);
}

static void
recallLaunchKernel(void *argp) {
  DSCUDAVERB_SET_ARGS(LaunchKernel);
  WARN(3, "recall IbvLaunchKernel\n");
  dscudaLaunchKernelWrapper(argsrc->moduleid, argsrc->kid, argsrc->kname, argsrc->gdim, argsrc->bdim, argsrc->smemsize, argsrc->stream, argsrc->narg, argsrc->arg);
}

static void
recallConfigureCall(void *argp) {
  DSCUDAVERB_SET_ARGS(ConfigureCall);
  WARN(3, "recall cudaConfigureCall\n");
  dim3 gdim, bdim;
  gdim.x = argsrc->gdim[0];
  gdim.y = argsrc->gdim[1];
  gdim.z = argsrc->gdim[2];
  bdim.x = argsrc->bdim[0];
  bdim.y = argsrc->bdim[1];
  bdim.z = argsrc->bdim[2];
  cudaConfigureCall(gdim, bdim, argsrc->smemsize, (cudaStream_t)argsrc->stream);
}

static void
recallSetupArgument(void *argp) {
  DSCUDAVERB_SET_ARGS(SetupArgument);
  WARN(3, "recall cudaSetupArgument\n");
  cudaSetupArgument(argsrc->arg, argsrc->size, argsrc->offset);
}

static void
recallSetupArgumentOfTypeP(void *argp) {
  DSCUDAVERB_SET_ARGS(SetupArgumentOfTypeP);
  WARN(3, "recall dscudaSetupArgumentOfTypeP\n");
  cudaSetupArgumentOfTypeP(argsrc->arg, argsrc->size, argsrc->offset);
}

static void
recallDscudaLaunchWrapper(void *argp) {
  DSCUDAVERB_SET_ARGS(DscudaLaunchWrapper);
  WARN(3, "recall dscudaLaunchWrapper\n");
  dscudaLaunchWrapper(argsrc->kadrp, argsrc->key);
}

//initialize redundant unit
void dscudaVerbInit(void) {
  int i;

  memset(storeArgsStub, 0, sizeof(DSCVMethod) * DSCVMethodEnd);
  memset(releaseArgsStub, 0, sizeof(DSCVMethod) * DSCVMethodEnd);
  memset(recallStub, 0, sizeof(DSCVMethod) * DSCVMethodEnd);

  DSCUDAVERB_SET_STUBS(SetDevice);
  DSCUDAVERB_SET_STUBS(MemcpyH2D);
  DSCUDAVERB_SET_STUBS(MemcpyD2D);
  DSCUDAVERB_SET_STUBS(MemcpyD2H);
  DSCUDAVERB_SET_STUBS(MemcpyToSymbolH2D);
  DSCUDAVERB_SET_STUBS(MemcpyToSymbolD2D);
  DSCUDAVERB_SET_STUBS(LaunchKernel);
  DSCUDAVERB_SET_STUBS(ConfigureCall);
  DSCUDAVERB_SET_STUBS(SetupArgument);
  DSCUDAVERB_SET_STUBS(SetupArgumentOfTypeP);
  DSCUDAVERB_SET_STUBS(DscudaLaunchWrapper);

  for (i = 1; i < DSCVMethodEnd; i++) {
    if (!storeArgsStub[i]) {
      fprintf(stderr, "dscudaVerbInit: storeArgsStub[%d] is not initialized.\n", i);
      exit(1);
    }
    if (!releaseArgsStub[i]) {
      fprintf(stderr, "dscudaVerbInit: releaseArgsStub[%d] is not initialized.\n", i);
      exit(1);
    }
    if (!recallStub[i]) {
      fprintf(stderr, "dscudaVerbInit: recallStub[%d] is not initialized.\n", i);
      exit(1);
    }
  }
}



void dscudaVerbAddHist(int funcID, void *argp) {
  if (verbHistNum == verbHistMax) {
    verbHistMax += DSCUDAVERB_HISTMAX_GROWSIZE;
    verbHists = (dscudaVerbHist *)realloc(verbHists, sizeof(dscudaVerbHist) * verbHistMax);
  }

  verbHists[verbHistNum].args = (storeArgsStub[funcID2DSCVMethod(funcID)])(argp);
  verbHists[verbHistNum].funcID = funcID;
  verbHistNum++;

  /*
  if (funcID == dscudaLaunchKernelId) {
    if (dscudaRemoteCallType() == RC_REMOTECALL_TYPE_IBV) { // IBV
      verbHists[verbHistNum].args = *(storeIbvCudaLaunchKernel(&argp));
    }
    else { // RPC
      RCargs *LKargs = &(argp.rpcCudaLaunchKernelArgs.args);
      RCarg *args2;

      args2 = (RCarg *)malloc(sizeof(RCarg) * LKargs->RCargs_len);
      memcpy(args2, LKargs->RCargs_val, sizeof(RCarg) * LKargs->RCargs_len);
      LKargs->RCargs_val = args2;
      verbHists[verbHistNum].args = argp;
    }
  }
  else {
    verbHists[verbHistNum].args = argp;
  }
  */

  //WARN(3, "%dth function history added\n", verbHistNum);
  return;
}

void dscudaVerbClearHist(void) {
  int i;
  if (verbHists) {
    for (i = 0; i < verbHistNum; i++) {
      (releaseArgsStub[funcID2DSCVMethod(verbHists[i].funcID)])(verbHists[i].args);
    }

    //free(verbHists);
    //verbHists = NULL;
  }
  verbHistNum = 0;

  WARN(3, "function history cleared\n");
  return;
}

void dscudaVerbRecallHist(void) {
  int i;

  dscudaSetAutoVerb(0); //this must be removed in the future

  WARN(1, "recalling functions...\n");
  for (i = 0; i < verbHistNum; i++) {
    (recallStub[funcID2DSCVMethod(verbHists[i].funcID)])(verbHists[i].args);
  }
  dscudaSetAutoVerb(1);

  WARN(1, "recalling done.\n");
}
