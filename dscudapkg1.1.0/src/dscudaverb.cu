#include <string.h>
#include "dscuda.h"
#include "dscudarpc.h"
#include "dscudaverb.h"

static verbHist *verbHists = NULL;
static int verbHistNum = 0;
static int verbHistMax = 0;

static cudaArgs
storeIbvCudaLaunchKernel(cudaArgs *casrcp)
{
    cudaArgs cadst;
    ibvCudaLaunchKernelArgsType *argsrcp = &casrcp->ibvCudaLaunchKernelArgs;
    ibvCudaLaunchKernelArgsType *argdstp = &cadst.ibvCudaLaunchKernelArgs;
    int narg = argsrcp->narg;
    IbvArg *ibvargbuf = (IbvArg *)malloc(sizeof(IbvArg) * narg);
    *argdstp = *argsrcp;
    memcpy(ibvargbuf, argsrcp->arg, sizeof(IbvArg) * narg);
    argdstp->arg = ibvargbuf;

    return cadst;
}

static void
releaseIbvCudaLaunchKernel(cudaArgs *cudaargsp)
{
    free(cudaargsp->ibvCudaLaunchKernelArgs.arg);
}

void verbAddHist(int funcID, cudaArgs args) {
    if (verbHistNum == verbHistMax) {
        verbHistMax += 10;
        verbHists = (verbHist *)realloc(verbHists, sizeof(verbHist) * verbHistMax);
    }

    if (funcID == dscudaLaunchKernelId) {
        if (dscudaRemoteCallType() == RC_REMOTECALL_TYPE_IBV) { // IBV
            args = storeIbvCudaLaunchKernel(&args);

            //            ibvCudaLaunchKernelArgsType *p = &args.ibvCudaLaunchKernelArgs;
            //            fprintf(stderr, "moduleid:%p\n", p->moduleid);
            //            fprintf(stderr, "moduleid[0]:%d\n", p->moduleid[0]);
        }
        else { // RPC
            RCargs *LKargs = &(args.rpcCudaLaunchKernelArgs.args);
            RCarg *args2;

            args2 = (RCarg *)malloc(sizeof(RCarg) * LKargs->RCargs_len);
            memcpy(args2, LKargs->RCargs_val, sizeof(RCarg) * LKargs->RCargs_len);
            LKargs->RCargs_val = args2;
        }
    }
    else {
      verbHists[verbHistNum].args = args;
    }

    verbHists[verbHistNum].funcID = funcID;
    verbHistNum++;

    //WARN(3, "%dth function history added\n", verbHistNum);
    return;
}

void verbClearHist(void) {
    verbHistNum = verbHistMax = 0;
    if (verbHists) {
        free(verbHists);
        verbHists = NULL;
    }

    WARN(3, "function history cleared\n");
    return;
}

void verbRecallHist(void) {
    int i;
    int *moduleid;

    dscudaSetAutoVerb(0); // avoid recursive invocation of verbAddHist.
    WARN(1, "\nIllegal return value has detected\nRecalling functions...\n");

    for (i = 0; i < verbHistNum; i++) {
        switch (verbHists[i].funcID) {
            {
              case dscudaSetDeviceId:
                WARN(2, "recall cudaSetDevice\n");
                cudaSetDeviceArgsType args = verbHists[i].args.cudaSetDeviceArgs;
                cudaSetDevice(args.device);
                break;
            }

            {
              case dscudaGetDevicePropertiesId:
                WARN(2, "recall cudaGetDeviceProperties [not implemented yet]\n");
                cudaGetDevicePropertiesArgsType args = verbHists[i].args.cudaGetDevicePropertiesArgs;
                //cudaGetDeviceProperties(args.prop, args.device);
                break;
            }

#if 0 // do nothing.
            {
              case dscudaMallocId:
                WARN(2, "recall cudaMalloc\n");
                cudaMallocArgsType args = verbHists[i].args.cudaMallocArgs;
                cudaMalloc(args.devPtr, args.size);
                break;
            }

            {
              case dscudaFreeId:
                WARN(2, "recall cudaFree\n");
                cudaFreeArgsType args = verbHists[i].args.cudaFreeArgs;
                cudaFree(args.devPtr);
                break;
            }

#endif

            {
              case dscudaMemcpyH2DId:
                WARN(2, "recall cudaMemcpy H2D\n");
                cudaMemcpyArgsType args = verbHists[i].args.cudaMemcpyArgs;
                cudaMemcpy(args.dst, args.src, args.count, args.kind);
                break;
            }
            {
              case dscudaMemcpyD2DId:
                WARN(2, "recall cudaMemcpy D2D\n");
                cudaMemcpyArgsType args = verbHists[i].args.cudaMemcpyArgs;
                cudaMemcpy(args.dst, args.src, args.count, args.kind);
                break;
            }

            {
              case dscudaMemcpyD2HId:
                WARN(2, "recall cudaMemcpy D2H\n");
                cudaMemcpyArgsType args = verbHists[i].args.cudaMemcpyArgs;
                cudaMemcpy(args.dst, args.src, args.count, args.kind);
                break;
            }

            {
              case dscudaMemcpyToSymbolH2DId:
                WARN(2, "recall MemcpyToSymbol\n");
                cudaMemcpyToSymbolArgsType args = verbHists[i].args.cudaMemcpyToSymbolArgs;
                dscudaMemcpyToSymbolWrapper(moduleid, args.symbol, args.src, args.count, args.offset, args.kind);
                break;
            }

            {
              case dscudaLoadModuleId:
                WARN(2, "recall cudaLoadModule\n");
                cudaLoadModuleArgsType args = verbHists[i].args.cudaLoadModuleArgs;
                moduleid = dscudaLoadModule(args.srcname);
                break;
            }

            {
              case dscudaLaunchKernelId:
                WARN(2, "recall cudaLaunchKernel\n");
                if (dscudaRemoteCallType() == RC_REMOTECALL_TYPE_IBV) {
                    ibvCudaLaunchKernelArgsType args = verbHists[i].args.ibvCudaLaunchKernelArgs;

                    fprintf(stderr, ">>>>>>>>>> mid[0]:%d  narg:%d  smemsize:%d\n",
                            args.moduleid[0], args.narg, args.smemsize);
                    fprintf(stderr, ">>>>>>>>>> kid:%d     kname:%s\n", args.kid, args.kname);
                    fprintf(stderr, ">>>>>>>>>> gdim:%d %d %d   bdim:%d %d %d\n",
                            args.gdim[0], args.gdim[1], args.gdim[2],
                            args.bdim[0], args.bdim[1], args.bdim[2]);
                    for (int iarg = 0; iarg < args.narg; iarg++) {
                        IbvArg *a = args.arg + iarg;
                        fprintf(stderr, ">>>>>>>>>> arg[%d] off:%d  size:%d     ",
                                iarg, a->offset, a->size);
                        switch (a->type) {
                          case dscudaArgTypeP:
                            fprintf(stderr, "type:ptr  val:%p\n", a->val.pointerval);
                            break;
                          case dscudaArgTypeI:
                            fprintf(stderr, "type:int  val:%d\n", a->val.intval);
                            break;
                          case dscudaArgTypeF:
                            fprintf(stderr, "type:float val:%f\n", a->val.floatval);
                            break;
                          case dscudaArgTypeV:
                            fprintf(stderr, "type:custom val:%d\n", (int *)a->val.customval);
                            break;
                        }
                    }

                    ibvDscudaLaunchKernelWrapper(args.moduleid, args.kid, args.kname,
                                                 args.gdim, args.bdim, args.smemsize, args.stream,
                                                 args.narg, args.arg);
                    releaseIbvCudaLaunchKernel(&verbHists[i].args);
                }
                else {
                    rpcCudaLaunchKernelArgsType args = verbHists[i].args.rpcCudaLaunchKernelArgs;
                    rpcDscudaLaunchKernelWrapper(args.moduleid, args.kid, args.kname,
                                                 args.gdim, args.bdim, args.smemsize, args.stream,
                                                 args.args);
                    free(args.args.RCargs_val);
                }
                break;
            }
        }
    }

    dscudaSetAutoVerb(1);

    WARN(1, "done.\n");
}
 
