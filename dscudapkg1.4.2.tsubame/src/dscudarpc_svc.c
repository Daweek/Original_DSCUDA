/*
 * Please do not edit this file.
 * It was generated using rpcgen.
 */

#include "dscudarpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <rpc/pmap_clnt.h>
#include <string.h>
#include <memory.h>
#include <sys/socket.h>
#include <netinet/in.h>

#ifndef SIG_PF
#define SIG_PF void(*)(int)
#endif

static dscudaResult *
_dscudathreadexitid_1 (void  *argp, struct svc_req *rqstp)
{
	return (dscudathreadexitid_1_svc(rqstp));
}

static dscudaResult *
_dscudathreadsynchronizeid_1 (void  *argp, struct svc_req *rqstp)
{
	return (dscudathreadsynchronizeid_1_svc(rqstp));
}

static dscudaResult *
_dscudathreadsetlimitid_1 (dscudathreadsetlimitid_1_argument *argp, struct svc_req *rqstp)
{
	return (dscudathreadsetlimitid_1_svc(argp->limit, argp->value, rqstp));
}

static dscudaThreadGetLimitResult *
_dscudathreadgetlimitid_1 (int  *argp, struct svc_req *rqstp)
{
	return (dscudathreadgetlimitid_1_svc(*argp, rqstp));
}

static dscudaResult *
_dscudathreadsetcacheconfigid_1 (int  *argp, struct svc_req *rqstp)
{
	return (dscudathreadsetcacheconfigid_1_svc(*argp, rqstp));
}

static dscudaThreadGetCacheConfigResult *
_dscudathreadgetcacheconfigid_1 (void  *argp, struct svc_req *rqstp)
{
	return (dscudathreadgetcacheconfigid_1_svc(rqstp));
}

static dscudaResult *
_dscudagetlasterrorid_1 (void  *argp, struct svc_req *rqstp)
{
	return (dscudagetlasterrorid_1_svc(rqstp));
}

static dscudaResult *
_dscudapeekatlasterrorid_1 (void  *argp, struct svc_req *rqstp)
{
	return (dscudapeekatlasterrorid_1_svc(rqstp));
}

static dscudaGetErrorStringResult *
_dscudageterrorstringid_1 (int  *argp, struct svc_req *rqstp)
{
	return (dscudageterrorstringid_1_svc(*argp, rqstp));
}

static dscudaGetDeviceResult *
_dscudagetdeviceid_1 (void  *argp, struct svc_req *rqstp)
{
	return (dscudagetdeviceid_1_svc(rqstp));
}

static dscudaGetDeviceCountResult *
_dscudagetdevicecountid_1 (void  *argp, struct svc_req *rqstp)
{
	return (dscudagetdevicecountid_1_svc(rqstp));
}

static dscudaGetDevicePropertiesResult *
_dscudagetdevicepropertiesid_1 (int  *argp, struct svc_req *rqstp)
{
	return (dscudagetdevicepropertiesid_1_svc(*argp, rqstp));
}

static dscudaDriverGetVersionResult *
_dscudadrivergetversionid_1 (void  *argp, struct svc_req *rqstp)
{
	return (dscudadrivergetversionid_1_svc(rqstp));
}

static dscudaRuntimeGetVersionResult *
_dscudaruntimegetversionid_1 (void  *argp, struct svc_req *rqstp)
{
	return (dscudaruntimegetversionid_1_svc(rqstp));
}

static dscudaResult *
_dscudasetdeviceid_1 (int  *argp, struct svc_req *rqstp)
{
	return (dscudasetdeviceid_1_svc(*argp, rqstp));
}

static dscudaResult *
_dscudasetdeviceflagsid_1 (u_int  *argp, struct svc_req *rqstp)
{
	return (dscudasetdeviceflagsid_1_svc(*argp, rqstp));
}

static dscudaChooseDeviceResult *
_dscudachoosedeviceid_1 (RCbuf  *argp, struct svc_req *rqstp)
{
	return (dscudachoosedeviceid_1_svc(*argp, rqstp));
}

static dscudaResult *
_dscudadevicesynchronize_1 (void  *argp, struct svc_req *rqstp)
{
	return (dscudadevicesynchronize_1_svc(rqstp));
}

static dscudaResult *
_dscudadevicereset_1 (void  *argp, struct svc_req *rqstp)
{
	return (dscudadevicereset_1_svc(rqstp));
}

static dscudaStreamCreateResult *
_dscudastreamcreateid_1 (void  *argp, struct svc_req *rqstp)
{
	return (dscudastreamcreateid_1_svc(rqstp));
}

static dscudaResult *
_dscudastreamdestroyid_1 (RCstream  *argp, struct svc_req *rqstp)
{
	return (dscudastreamdestroyid_1_svc(*argp, rqstp));
}

static dscudaResult *
_dscudastreamsynchronizeid_1 (RCstream  *argp, struct svc_req *rqstp)
{
	return (dscudastreamsynchronizeid_1_svc(*argp, rqstp));
}

static dscudaResult *
_dscudastreamqueryid_1 (RCstream  *argp, struct svc_req *rqstp)
{
	return (dscudastreamqueryid_1_svc(*argp, rqstp));
}

static dscudaResult *
_dscudastreamwaiteventid_1 (dscudastreamwaiteventid_1_argument *argp, struct svc_req *rqstp)
{
	return (dscudastreamwaiteventid_1_svc(argp->stream, argp->event, argp->flags, rqstp));
}

static dscudaEventCreateResult *
_dscudaeventcreateid_1 (void  *argp, struct svc_req *rqstp)
{
	return (dscudaeventcreateid_1_svc(rqstp));
}

static dscudaEventCreateResult *
_dscudaeventcreatewithflagsid_1 (u_int  *argp, struct svc_req *rqstp)
{
	return (dscudaeventcreatewithflagsid_1_svc(*argp, rqstp));
}

static dscudaResult *
_dscudaeventdestroyid_1 (RCevent  *argp, struct svc_req *rqstp)
{
	return (dscudaeventdestroyid_1_svc(*argp, rqstp));
}

static dscudaEventElapsedTimeResult *
_dscudaeventelapsedtimeid_1 (dscudaeventelapsedtimeid_1_argument *argp, struct svc_req *rqstp)
{
	return (dscudaeventelapsedtimeid_1_svc(argp->start, argp->end, rqstp));
}

static dscudaResult *
_dscudaeventrecordid_1 (dscudaeventrecordid_1_argument *argp, struct svc_req *rqstp)
{
	return (dscudaeventrecordid_1_svc(argp->event, argp->stream, rqstp));
}

static dscudaResult *
_dscudaeventsynchronizeid_1 (RCevent  *argp, struct svc_req *rqstp)
{
	return (dscudaeventsynchronizeid_1_svc(*argp, rqstp));
}

static dscudaResult *
_dscudaeventqueryid_1 (RCevent  *argp, struct svc_req *rqstp)
{
	return (dscudaeventqueryid_1_svc(*argp, rqstp));
}

static void *
_dscudalaunchkernelid_1 (dscudalaunchkernelid_1_argument *argp, struct svc_req *rqstp)
{
	return (dscudalaunchkernelid_1_svc(argp->moduleid, argp->kid, argp->kname, argp->gdim, argp->bdim, argp->smemsize, argp->stream, argp->args, rqstp));
}

static dscudaLoadModuleResult *
_dscudaloadmoduleid_1 (dscudaloadmoduleid_1_argument *argp, struct svc_req *rqstp)
{
	return (dscudaloadmoduleid_1_svc(argp->ipaddr, argp->pid, argp->mname, argp->image, rqstp));
}

static dscudaFuncGetAttributesResult *
_dscudafuncgetattributesid_1 (dscudafuncgetattributesid_1_argument *argp, struct svc_req *rqstp)
{
	return (dscudafuncgetattributesid_1_svc(argp->moduleid, argp->kname, rqstp));
}

static dscudaMallocResult *
_dscudamallocid_1 (RCsize  *argp, struct svc_req *rqstp)
{
	return (dscudamallocid_1_svc(*argp, rqstp));
}

static dscudaResult *
_dscudafreeid_1 (RCadr  *argp, struct svc_req *rqstp)
{
	return (dscudafreeid_1_svc(*argp, rqstp));
}

static dscudaMemcpyH2HResult *
_dscudamemcpyh2hid_1 (dscudamemcpyh2hid_1_argument *argp, struct svc_req *rqstp)
{
	return (dscudamemcpyh2hid_1_svc(argp->dst, argp->src, argp->count, rqstp));
}

static dscudaResult *
_dscudamemcpyh2did_1 (dscudamemcpyh2did_1_argument *argp, struct svc_req *rqstp)
{
	return (dscudamemcpyh2did_1_svc(argp->dst, argp->src, argp->count, rqstp));
}

static dscudaMemcpyD2HResult *
_dscudamemcpyd2hid_1 (dscudamemcpyd2hid_1_argument *argp, struct svc_req *rqstp)
{
	return (dscudamemcpyd2hid_1_svc(argp->src, argp->count, rqstp));
}

static dscudaResult *
_dscudamemcpyd2did_1 (dscudamemcpyd2did_1_argument *argp, struct svc_req *rqstp)
{
	return (dscudamemcpyd2did_1_svc(argp->dst, argp->src, argp->count, rqstp));
}

static dscudaMemcpyAsyncH2HResult *
_dscudamemcpyasynch2hid_1 (dscudamemcpyasynch2hid_1_argument *argp, struct svc_req *rqstp)
{
	return (dscudamemcpyasynch2hid_1_svc(argp->dst, argp->src, argp->count, argp->stream, rqstp));
}

static dscudaResult *
_dscudamemcpyasynch2did_1 (dscudamemcpyasynch2did_1_argument *argp, struct svc_req *rqstp)
{
	return (dscudamemcpyasynch2did_1_svc(argp->dst, argp->src, argp->count, argp->stream, rqstp));
}

static dscudaMemcpyAsyncD2HResult *
_dscudamemcpyasyncd2hid_1 (dscudamemcpyasyncd2hid_1_argument *argp, struct svc_req *rqstp)
{
	return (dscudamemcpyasyncd2hid_1_svc(argp->src, argp->count, argp->stream, rqstp));
}

static dscudaResult *
_dscudamemcpyasyncd2did_1 (dscudamemcpyasyncd2did_1_argument *argp, struct svc_req *rqstp)
{
	return (dscudamemcpyasyncd2did_1_svc(argp->dst, argp->src, argp->count, argp->stream, rqstp));
}

static dscudaResult *
_dscudamemcpytosymbolh2did_1 (dscudamemcpytosymbolh2did_1_argument *argp, struct svc_req *rqstp)
{
	return (dscudamemcpytosymbolh2did_1_svc(argp->moduleid, argp->symbol, argp->src, argp->count, argp->offset, rqstp));
}

static dscudaResult *
_dscudamemcpytosymbold2did_1 (dscudamemcpytosymbold2did_1_argument *argp, struct svc_req *rqstp)
{
	return (dscudamemcpytosymbold2did_1_svc(argp->moduleid, argp->symbol, argp->src, argp->count, argp->offset, rqstp));
}

static dscudaMemcpyFromSymbolD2HResult *
_dscudamemcpyfromsymbold2hid_1 (dscudamemcpyfromsymbold2hid_1_argument *argp, struct svc_req *rqstp)
{
	return (dscudamemcpyfromsymbold2hid_1_svc(argp->moduleid, argp->symbol, argp->count, argp->offset, rqstp));
}

static dscudaResult *
_dscudamemcpyfromsymbold2did_1 (dscudamemcpyfromsymbold2did_1_argument *argp, struct svc_req *rqstp)
{
	return (dscudamemcpyfromsymbold2did_1_svc(argp->moduleid, argp->dst, argp->symbol, argp->count, argp->offset, rqstp));
}

static dscudaResult *
_dscudamemsetid_1 (dscudamemsetid_1_argument *argp, struct svc_req *rqstp)
{
	return (dscudamemsetid_1_svc(argp->dst, argp->value, argp->count, rqstp));
}

static dscudaHostAllocResult *
_dscudahostallocid_1 (dscudahostallocid_1_argument *argp, struct svc_req *rqstp)
{
	return (dscudahostallocid_1_svc(argp->size, argp->flags, rqstp));
}

static dscudaMallocHostResult *
_dscudamallochostid_1 (RCsize  *argp, struct svc_req *rqstp)
{
	return (dscudamallochostid_1_svc(*argp, rqstp));
}

static dscudaResult *
_dscudafreehostid_1 (RCadr  *argp, struct svc_req *rqstp)
{
	return (dscudafreehostid_1_svc(*argp, rqstp));
}

static dscudaHostGetDevicePointerResult *
_dscudahostgetdevicepointerid_1 (dscudahostgetdevicepointerid_1_argument *argp, struct svc_req *rqstp)
{
	return (dscudahostgetdevicepointerid_1_svc(argp->pHost, argp->flags, rqstp));
}

static dscudaHostGetFlagsResult *
_dscudahostgetflagsid_1 (RCadr  *argp, struct svc_req *rqstp)
{
	return (dscudahostgetflagsid_1_svc(*argp, rqstp));
}

static dscudaMallocArrayResult *
_dscudamallocarrayid_1 (dscudamallocarrayid_1_argument *argp, struct svc_req *rqstp)
{
	return (dscudamallocarrayid_1_svc(argp->desc, argp->width, argp->height, argp->flags, rqstp));
}

static dscudaResult *
_dscudafreearrayid_1 (RCadr  *argp, struct svc_req *rqstp)
{
	return (dscudafreearrayid_1_svc(*argp, rqstp));
}

static dscudaMemcpyToArrayH2HResult *
_dscudamemcpytoarrayh2hid_1 (dscudamemcpytoarrayh2hid_1_argument *argp, struct svc_req *rqstp)
{
	return (dscudamemcpytoarrayh2hid_1_svc(argp->dst, argp->wOffset, argp->hOffset, argp->src, argp->count, rqstp));
}

static dscudaResult *
_dscudamemcpytoarrayh2did_1 (dscudamemcpytoarrayh2did_1_argument *argp, struct svc_req *rqstp)
{
	return (dscudamemcpytoarrayh2did_1_svc(argp->dst, argp->wOffset, argp->hOffset, argp->src, argp->count, rqstp));
}

static dscudaMemcpyToArrayD2HResult *
_dscudamemcpytoarrayd2hid_1 (dscudamemcpytoarrayd2hid_1_argument *argp, struct svc_req *rqstp)
{
	return (dscudamemcpytoarrayd2hid_1_svc(argp->wOffset, argp->hOffset, argp->src, argp->count, rqstp));
}

static dscudaResult *
_dscudamemcpytoarrayd2did_1 (dscudamemcpytoarrayd2did_1_argument *argp, struct svc_req *rqstp)
{
	return (dscudamemcpytoarrayd2did_1_svc(argp->dst, argp->wOffset, argp->hOffset, argp->src, argp->count, rqstp));
}

static dscudaMallocPitchResult *
_dscudamallocpitchid_1 (dscudamallocpitchid_1_argument *argp, struct svc_req *rqstp)
{
	return (dscudamallocpitchid_1_svc(argp->width, argp->height, rqstp));
}

static dscudaMemcpy2DToArrayH2HResult *
_dscudamemcpy2dtoarrayh2hid_1 (dscudamemcpy2dtoarrayh2hid_1_argument *argp, struct svc_req *rqstp)
{
	return (dscudamemcpy2dtoarrayh2hid_1_svc(argp->dst, argp->wOffset, argp->hOffset, argp->src, argp->spitch, argp->width, argp->height, rqstp));
}

static dscudaResult *
_dscudamemcpy2dtoarrayh2did_1 (dscudamemcpy2dtoarrayh2did_1_argument *argp, struct svc_req *rqstp)
{
	return (dscudamemcpy2dtoarrayh2did_1_svc(argp->dst, argp->wOffset, argp->hOffset, argp->srcbuf, argp->spitch, argp->width, argp->height, rqstp));
}

static dscudaMemcpy2DToArrayD2HResult *
_dscudamemcpy2dtoarrayd2hid_1 (dscudamemcpy2dtoarrayd2hid_1_argument *argp, struct svc_req *rqstp)
{
	return (dscudamemcpy2dtoarrayd2hid_1_svc(argp->wOffset, argp->hOffset, argp->src, argp->spitch, argp->width, argp->height, rqstp));
}

static dscudaResult *
_dscudamemcpy2dtoarrayd2did_1 (dscudamemcpy2dtoarrayd2did_1_argument *argp, struct svc_req *rqstp)
{
	return (dscudamemcpy2dtoarrayd2did_1_svc(argp->dst, argp->wOffset, argp->hOffset, argp->src, argp->spitch, argp->width, argp->height, rqstp));
}

static dscudaMemcpy2DH2HResult *
_dscudamemcpy2dh2hid_1 (dscudamemcpy2dh2hid_1_argument *argp, struct svc_req *rqstp)
{
	return (dscudamemcpy2dh2hid_1_svc(argp->dst, argp->dpitch, argp->src, argp->spitch, argp->width, argp->height, rqstp));
}

static dscudaResult *
_dscudamemcpy2dh2did_1 (dscudamemcpy2dh2did_1_argument *argp, struct svc_req *rqstp)
{
	return (dscudamemcpy2dh2did_1_svc(argp->dst, argp->dpitch, argp->src, argp->spitch, argp->width, argp->height, rqstp));
}

static dscudaMemcpy2DD2HResult *
_dscudamemcpy2dd2hid_1 (dscudamemcpy2dd2hid_1_argument *argp, struct svc_req *rqstp)
{
	return (dscudamemcpy2dd2hid_1_svc(argp->dpitch, argp->src, argp->spitch, argp->width, argp->height, rqstp));
}

static dscudaResult *
_dscudamemcpy2dd2did_1 (dscudamemcpy2dd2did_1_argument *argp, struct svc_req *rqstp)
{
	return (dscudamemcpy2dd2did_1_svc(argp->dst, argp->dpitch, argp->src, argp->spitch, argp->width, argp->height, rqstp));
}

static dscudaResult *
_dscudamemset2did_1 (dscudamemset2did_1_argument *argp, struct svc_req *rqstp)
{
	return (dscudamemset2did_1_svc(argp->dst, argp->pitch, argp->value, argp->width, argp->height, rqstp));
}

static dscudaResult *
_dscudamemcpytosymbolasynch2did_1 (dscudamemcpytosymbolasynch2did_1_argument *argp, struct svc_req *rqstp)
{
	return (dscudamemcpytosymbolasynch2did_1_svc(argp->moduleid, argp->symbol, argp->src, argp->count, argp->offset, argp->stream, rqstp));
}

static dscudaResult *
_dscudamemcpytosymbolasyncd2did_1 (dscudamemcpytosymbolasyncd2did_1_argument *argp, struct svc_req *rqstp)
{
	return (dscudamemcpytosymbolasyncd2did_1_svc(argp->moduleid, argp->symbol, argp->src, argp->count, argp->offset, argp->stream, rqstp));
}

static dscudaMemcpyFromSymbolAsyncD2HResult *
_dscudamemcpyfromsymbolasyncd2hid_1 (dscudamemcpyfromsymbolasyncd2hid_1_argument *argp, struct svc_req *rqstp)
{
	return (dscudamemcpyfromsymbolasyncd2hid_1_svc(argp->moduleid, argp->symbol, argp->count, argp->offset, argp->stream, rqstp));
}

static dscudaResult *
_dscudamemcpyfromsymbolasyncd2did_1 (dscudamemcpyfromsymbolasyncd2did_1_argument *argp, struct svc_req *rqstp)
{
	return (dscudamemcpyfromsymbolasyncd2did_1_svc(argp->moduleid, argp->dst, argp->symbol, argp->count, argp->offset, argp->stream, rqstp));
}

static dscudaCreateChannelDescResult *
_dscudacreatechanneldescid_1 (dscudacreatechanneldescid_1_argument *argp, struct svc_req *rqstp)
{
	return (dscudacreatechanneldescid_1_svc(argp->x, argp->y, argp->z, argp->w, argp->f, rqstp));
}

static dscudaGetChannelDescResult *
_dscudagetchanneldescid_1 (RCadr  *argp, struct svc_req *rqstp)
{
	return (dscudagetchanneldescid_1_svc(*argp, rqstp));
}

static dscudaBindTextureResult *
_dscudabindtextureid_1 (dscudabindtextureid_1_argument *argp, struct svc_req *rqstp)
{
	return (dscudabindtextureid_1_svc(argp->moduleid, argp->texname, argp->devPtr, argp->size, argp->texbuf, rqstp));
}

static dscudaBindTexture2DResult *
_dscudabindtexture2did_1 (dscudabindtexture2did_1_argument *argp, struct svc_req *rqstp)
{
	return (dscudabindtexture2did_1_svc(argp->moduleid, argp->texname, argp->devPtr, argp->width, argp->height, argp->pitch, argp->texbuf, rqstp));
}

static dscudaResult *
_dscudabindtexturetoarrayid_1 (dscudabindtexturetoarrayid_1_argument *argp, struct svc_req *rqstp)
{
	return (dscudabindtexturetoarrayid_1_svc(argp->moduleid, argp->texname, argp->array, argp->texbuf, rqstp));
}

static dscudaResult *
_dscudaunbindtextureid_1 (RCtexture  *argp, struct svc_req *rqstp)
{
	return (dscudaunbindtextureid_1_svc(*argp, rqstp));
}

static dscufftPlanResult *
_dscufftplan3did_1 (dscufftplan3did_1_argument *argp, struct svc_req *rqstp)
{
	return (dscufftplan3did_1_svc(argp->nx, argp->ny, argp->nz, argp->type, rqstp));
}

static dscufftResult *
_dscufftdestroyid_1 (u_int  *argp, struct svc_req *rqstp)
{
	return (dscufftdestroyid_1_svc(*argp, rqstp));
}

static dscufftResult *
_dscufftexecc2cid_1 (dscufftexecc2cid_1_argument *argp, struct svc_req *rqstp)
{
	return (dscufftexecc2cid_1_svc(argp->plan, argp->idata, argp->odata, argp->direction, rqstp));
}

void
dscuda_prog_1(struct svc_req *rqstp, register SVCXPRT *transp)
{
	union {
		dscudathreadsetlimitid_1_argument dscudathreadsetlimitid_1_arg;
		int dscudathreadgetlimitid_1_arg;
		int dscudathreadsetcacheconfigid_1_arg;
		int dscudageterrorstringid_1_arg;
		int dscudagetdevicepropertiesid_1_arg;
		int dscudasetdeviceid_1_arg;
		u_int dscudasetdeviceflagsid_1_arg;
		RCbuf dscudachoosedeviceid_1_arg;
		RCstream dscudastreamdestroyid_1_arg;
		RCstream dscudastreamsynchronizeid_1_arg;
		RCstream dscudastreamqueryid_1_arg;
		dscudastreamwaiteventid_1_argument dscudastreamwaiteventid_1_arg;
		u_int dscudaeventcreatewithflagsid_1_arg;
		RCevent dscudaeventdestroyid_1_arg;
		dscudaeventelapsedtimeid_1_argument dscudaeventelapsedtimeid_1_arg;
		dscudaeventrecordid_1_argument dscudaeventrecordid_1_arg;
		RCevent dscudaeventsynchronizeid_1_arg;
		RCevent dscudaeventqueryid_1_arg;
		dscudalaunchkernelid_1_argument dscudalaunchkernelid_1_arg;
		dscudaloadmoduleid_1_argument dscudaloadmoduleid_1_arg;
		dscudafuncgetattributesid_1_argument dscudafuncgetattributesid_1_arg;
		RCsize dscudamallocid_1_arg;
		RCadr dscudafreeid_1_arg;
		dscudamemcpyh2hid_1_argument dscudamemcpyh2hid_1_arg;
		dscudamemcpyh2did_1_argument dscudamemcpyh2did_1_arg;
		dscudamemcpyd2hid_1_argument dscudamemcpyd2hid_1_arg;
		dscudamemcpyd2did_1_argument dscudamemcpyd2did_1_arg;
		dscudamemcpyasynch2hid_1_argument dscudamemcpyasynch2hid_1_arg;
		dscudamemcpyasynch2did_1_argument dscudamemcpyasynch2did_1_arg;
		dscudamemcpyasyncd2hid_1_argument dscudamemcpyasyncd2hid_1_arg;
		dscudamemcpyasyncd2did_1_argument dscudamemcpyasyncd2did_1_arg;
		dscudamemcpytosymbolh2did_1_argument dscudamemcpytosymbolh2did_1_arg;
		dscudamemcpytosymbold2did_1_argument dscudamemcpytosymbold2did_1_arg;
		dscudamemcpyfromsymbold2hid_1_argument dscudamemcpyfromsymbold2hid_1_arg;
		dscudamemcpyfromsymbold2did_1_argument dscudamemcpyfromsymbold2did_1_arg;
		dscudamemsetid_1_argument dscudamemsetid_1_arg;
		dscudahostallocid_1_argument dscudahostallocid_1_arg;
		RCsize dscudamallochostid_1_arg;
		RCadr dscudafreehostid_1_arg;
		dscudahostgetdevicepointerid_1_argument dscudahostgetdevicepointerid_1_arg;
		RCadr dscudahostgetflagsid_1_arg;
		dscudamallocarrayid_1_argument dscudamallocarrayid_1_arg;
		RCadr dscudafreearrayid_1_arg;
		dscudamemcpytoarrayh2hid_1_argument dscudamemcpytoarrayh2hid_1_arg;
		dscudamemcpytoarrayh2did_1_argument dscudamemcpytoarrayh2did_1_arg;
		dscudamemcpytoarrayd2hid_1_argument dscudamemcpytoarrayd2hid_1_arg;
		dscudamemcpytoarrayd2did_1_argument dscudamemcpytoarrayd2did_1_arg;
		dscudamallocpitchid_1_argument dscudamallocpitchid_1_arg;
		dscudamemcpy2dtoarrayh2hid_1_argument dscudamemcpy2dtoarrayh2hid_1_arg;
		dscudamemcpy2dtoarrayh2did_1_argument dscudamemcpy2dtoarrayh2did_1_arg;
		dscudamemcpy2dtoarrayd2hid_1_argument dscudamemcpy2dtoarrayd2hid_1_arg;
		dscudamemcpy2dtoarrayd2did_1_argument dscudamemcpy2dtoarrayd2did_1_arg;
		dscudamemcpy2dh2hid_1_argument dscudamemcpy2dh2hid_1_arg;
		dscudamemcpy2dh2did_1_argument dscudamemcpy2dh2did_1_arg;
		dscudamemcpy2dd2hid_1_argument dscudamemcpy2dd2hid_1_arg;
		dscudamemcpy2dd2did_1_argument dscudamemcpy2dd2did_1_arg;
		dscudamemset2did_1_argument dscudamemset2did_1_arg;
		dscudamemcpytosymbolasynch2did_1_argument dscudamemcpytosymbolasynch2did_1_arg;
		dscudamemcpytosymbolasyncd2did_1_argument dscudamemcpytosymbolasyncd2did_1_arg;
		dscudamemcpyfromsymbolasyncd2hid_1_argument dscudamemcpyfromsymbolasyncd2hid_1_arg;
		dscudamemcpyfromsymbolasyncd2did_1_argument dscudamemcpyfromsymbolasyncd2did_1_arg;
		dscudacreatechanneldescid_1_argument dscudacreatechanneldescid_1_arg;
		RCadr dscudagetchanneldescid_1_arg;
		dscudabindtextureid_1_argument dscudabindtextureid_1_arg;
		dscudabindtexture2did_1_argument dscudabindtexture2did_1_arg;
		dscudabindtexturetoarrayid_1_argument dscudabindtexturetoarrayid_1_arg;
		RCtexture dscudaunbindtextureid_1_arg;
		dscufftplan3did_1_argument dscufftplan3did_1_arg;
		u_int dscufftdestroyid_1_arg;
		dscufftexecc2cid_1_argument dscufftexecc2cid_1_arg;
	} argument;
	char *result;
	xdrproc_t _xdr_argument, _xdr_result;
	char *(*local)(char *, struct svc_req *);

	switch (rqstp->rq_proc) {
	case NULLPROC:
		(void) svc_sendreply (transp, (xdrproc_t) xdr_void, (char *)NULL);
		return;

	case dscudaThreadExitId:
		_xdr_argument = (xdrproc_t) xdr_void;
		_xdr_result = (xdrproc_t) xdr_dscudaResult;
		local = (char *(*)(char *, struct svc_req *)) _dscudathreadexitid_1;
		break;

	case dscudaThreadSynchronizeId:
		_xdr_argument = (xdrproc_t) xdr_void;
		_xdr_result = (xdrproc_t) xdr_dscudaResult;
		local = (char *(*)(char *, struct svc_req *)) _dscudathreadsynchronizeid_1;
		break;

	case dscudaThreadSetLimitId:
		_xdr_argument = (xdrproc_t) xdr_dscudathreadsetlimitid_1_argument;
		_xdr_result = (xdrproc_t) xdr_dscudaResult;
		local = (char *(*)(char *, struct svc_req *)) _dscudathreadsetlimitid_1;
		break;

	case dscudaThreadGetLimitId:
		_xdr_argument = (xdrproc_t) xdr_int;
		_xdr_result = (xdrproc_t) xdr_dscudaThreadGetLimitResult;
		local = (char *(*)(char *, struct svc_req *)) _dscudathreadgetlimitid_1;
		break;

	case dscudaThreadSetCacheConfigId:
		_xdr_argument = (xdrproc_t) xdr_int;
		_xdr_result = (xdrproc_t) xdr_dscudaResult;
		local = (char *(*)(char *, struct svc_req *)) _dscudathreadsetcacheconfigid_1;
		break;

	case dscudaThreadGetCacheConfigId:
		_xdr_argument = (xdrproc_t) xdr_void;
		_xdr_result = (xdrproc_t) xdr_dscudaThreadGetCacheConfigResult;
		local = (char *(*)(char *, struct svc_req *)) _dscudathreadgetcacheconfigid_1;
		break;

	case dscudaGetLastErrorId:
		_xdr_argument = (xdrproc_t) xdr_void;
		_xdr_result = (xdrproc_t) xdr_dscudaResult;
		local = (char *(*)(char *, struct svc_req *)) _dscudagetlasterrorid_1;
		break;

	case dscudaPeekAtLastErrorId:
		_xdr_argument = (xdrproc_t) xdr_void;
		_xdr_result = (xdrproc_t) xdr_dscudaResult;
		local = (char *(*)(char *, struct svc_req *)) _dscudapeekatlasterrorid_1;
		break;

	case dscudaGetErrorStringId:
		_xdr_argument = (xdrproc_t) xdr_int;
		_xdr_result = (xdrproc_t) xdr_dscudaGetErrorStringResult;
		local = (char *(*)(char *, struct svc_req *)) _dscudageterrorstringid_1;
		break;

	case dscudaGetDeviceId:
		_xdr_argument = (xdrproc_t) xdr_void;
		_xdr_result = (xdrproc_t) xdr_dscudaGetDeviceResult;
		local = (char *(*)(char *, struct svc_req *)) _dscudagetdeviceid_1;
		break;

	case dscudaGetDeviceCountId:
		_xdr_argument = (xdrproc_t) xdr_void;
		_xdr_result = (xdrproc_t) xdr_dscudaGetDeviceCountResult;
		local = (char *(*)(char *, struct svc_req *)) _dscudagetdevicecountid_1;
		break;

	case dscudaGetDevicePropertiesId:
		_xdr_argument = (xdrproc_t) xdr_int;
		_xdr_result = (xdrproc_t) xdr_dscudaGetDevicePropertiesResult;
		local = (char *(*)(char *, struct svc_req *)) _dscudagetdevicepropertiesid_1;
		break;

	case dscudaDriverGetVersionId:
		_xdr_argument = (xdrproc_t) xdr_void;
		_xdr_result = (xdrproc_t) xdr_dscudaDriverGetVersionResult;
		local = (char *(*)(char *, struct svc_req *)) _dscudadrivergetversionid_1;
		break;

	case dscudaRuntimeGetVersionId:
		_xdr_argument = (xdrproc_t) xdr_void;
		_xdr_result = (xdrproc_t) xdr_dscudaRuntimeGetVersionResult;
		local = (char *(*)(char *, struct svc_req *)) _dscudaruntimegetversionid_1;
		break;

	case dscudaSetDeviceId:
		_xdr_argument = (xdrproc_t) xdr_int;
		_xdr_result = (xdrproc_t) xdr_dscudaResult;
		local = (char *(*)(char *, struct svc_req *)) _dscudasetdeviceid_1;
		break;

	case dscudaSetDeviceFlagsId:
		_xdr_argument = (xdrproc_t) xdr_u_int;
		_xdr_result = (xdrproc_t) xdr_dscudaResult;
		local = (char *(*)(char *, struct svc_req *)) _dscudasetdeviceflagsid_1;
		break;

	case dscudaChooseDeviceId:
		_xdr_argument = (xdrproc_t) xdr_RCbuf;
		_xdr_result = (xdrproc_t) xdr_dscudaChooseDeviceResult;
		local = (char *(*)(char *, struct svc_req *)) _dscudachoosedeviceid_1;
		break;

	case dscudaDeviceSynchronize:
		_xdr_argument = (xdrproc_t) xdr_void;
		_xdr_result = (xdrproc_t) xdr_dscudaResult;
		local = (char *(*)(char *, struct svc_req *)) _dscudadevicesynchronize_1;
		break;

	case dscudaDeviceReset:
		_xdr_argument = (xdrproc_t) xdr_void;
		_xdr_result = (xdrproc_t) xdr_dscudaResult;
		local = (char *(*)(char *, struct svc_req *)) _dscudadevicereset_1;
		break;

	case dscudaStreamCreateId:
		_xdr_argument = (xdrproc_t) xdr_void;
		_xdr_result = (xdrproc_t) xdr_dscudaStreamCreateResult;
		local = (char *(*)(char *, struct svc_req *)) _dscudastreamcreateid_1;
		break;

	case dscudaStreamDestroyId:
		_xdr_argument = (xdrproc_t) xdr_RCstream;
		_xdr_result = (xdrproc_t) xdr_dscudaResult;
		local = (char *(*)(char *, struct svc_req *)) _dscudastreamdestroyid_1;
		break;

	case dscudaStreamSynchronizeId:
		_xdr_argument = (xdrproc_t) xdr_RCstream;
		_xdr_result = (xdrproc_t) xdr_dscudaResult;
		local = (char *(*)(char *, struct svc_req *)) _dscudastreamsynchronizeid_1;
		break;

	case dscudaStreamQueryId:
		_xdr_argument = (xdrproc_t) xdr_RCstream;
		_xdr_result = (xdrproc_t) xdr_dscudaResult;
		local = (char *(*)(char *, struct svc_req *)) _dscudastreamqueryid_1;
		break;

	case dscudaStreamWaitEventId:
		_xdr_argument = (xdrproc_t) xdr_dscudastreamwaiteventid_1_argument;
		_xdr_result = (xdrproc_t) xdr_dscudaResult;
		local = (char *(*)(char *, struct svc_req *)) _dscudastreamwaiteventid_1;
		break;

	case dscudaEventCreateId:
		_xdr_argument = (xdrproc_t) xdr_void;
		_xdr_result = (xdrproc_t) xdr_dscudaEventCreateResult;
		local = (char *(*)(char *, struct svc_req *)) _dscudaeventcreateid_1;
		break;

	case dscudaEventCreateWithFlagsId:
		_xdr_argument = (xdrproc_t) xdr_u_int;
		_xdr_result = (xdrproc_t) xdr_dscudaEventCreateResult;
		local = (char *(*)(char *, struct svc_req *)) _dscudaeventcreatewithflagsid_1;
		break;

	case dscudaEventDestroyId:
		_xdr_argument = (xdrproc_t) xdr_RCevent;
		_xdr_result = (xdrproc_t) xdr_dscudaResult;
		local = (char *(*)(char *, struct svc_req *)) _dscudaeventdestroyid_1;
		break;

	case dscudaEventElapsedTimeId:
		_xdr_argument = (xdrproc_t) xdr_dscudaeventelapsedtimeid_1_argument;
		_xdr_result = (xdrproc_t) xdr_dscudaEventElapsedTimeResult;
		local = (char *(*)(char *, struct svc_req *)) _dscudaeventelapsedtimeid_1;
		break;

	case dscudaEventRecordId:
		_xdr_argument = (xdrproc_t) xdr_dscudaeventrecordid_1_argument;
		_xdr_result = (xdrproc_t) xdr_dscudaResult;
		local = (char *(*)(char *, struct svc_req *)) _dscudaeventrecordid_1;
		break;

	case dscudaEventSynchronizeId:
		_xdr_argument = (xdrproc_t) xdr_RCevent;
		_xdr_result = (xdrproc_t) xdr_dscudaResult;
		local = (char *(*)(char *, struct svc_req *)) _dscudaeventsynchronizeid_1;
		break;

	case dscudaEventQueryId:
		_xdr_argument = (xdrproc_t) xdr_RCevent;
		_xdr_result = (xdrproc_t) xdr_dscudaResult;
		local = (char *(*)(char *, struct svc_req *)) _dscudaeventqueryid_1;
		break;

	case dscudaLaunchKernelId:
		_xdr_argument = (xdrproc_t) xdr_dscudalaunchkernelid_1_argument;
		_xdr_result = (xdrproc_t) xdr_void;
		local = (char *(*)(char *, struct svc_req *)) _dscudalaunchkernelid_1;
		break;

	case dscudaLoadModuleId:
		_xdr_argument = (xdrproc_t) xdr_dscudaloadmoduleid_1_argument;
		_xdr_result = (xdrproc_t) xdr_dscudaLoadModuleResult;
		local = (char *(*)(char *, struct svc_req *)) _dscudaloadmoduleid_1;
		break;

	case dscudaFuncGetAttributesId:
		_xdr_argument = (xdrproc_t) xdr_dscudafuncgetattributesid_1_argument;
		_xdr_result = (xdrproc_t) xdr_dscudaFuncGetAttributesResult;
		local = (char *(*)(char *, struct svc_req *)) _dscudafuncgetattributesid_1;
		break;

	case dscudaMallocId:
		_xdr_argument = (xdrproc_t) xdr_RCsize;
		_xdr_result = (xdrproc_t) xdr_dscudaMallocResult;
		local = (char *(*)(char *, struct svc_req *)) _dscudamallocid_1;
		break;

	case dscudaFreeId:
		_xdr_argument = (xdrproc_t) xdr_RCadr;
		_xdr_result = (xdrproc_t) xdr_dscudaResult;
		local = (char *(*)(char *, struct svc_req *)) _dscudafreeid_1;
		break;

	case dscudaMemcpyH2HId:
		_xdr_argument = (xdrproc_t) xdr_dscudamemcpyh2hid_1_argument;
		_xdr_result = (xdrproc_t) xdr_dscudaMemcpyH2HResult;
		local = (char *(*)(char *, struct svc_req *)) _dscudamemcpyh2hid_1;
		break;

	case dscudaMemcpyH2DId:
		_xdr_argument = (xdrproc_t) xdr_dscudamemcpyh2did_1_argument;
		_xdr_result = (xdrproc_t) xdr_dscudaResult;
		local = (char *(*)(char *, struct svc_req *)) _dscudamemcpyh2did_1;
		break;

	case dscudaMemcpyD2HId:
		_xdr_argument = (xdrproc_t) xdr_dscudamemcpyd2hid_1_argument;
		_xdr_result = (xdrproc_t) xdr_dscudaMemcpyD2HResult;
		local = (char *(*)(char *, struct svc_req *)) _dscudamemcpyd2hid_1;
		break;

	case dscudaMemcpyD2DId:
		_xdr_argument = (xdrproc_t) xdr_dscudamemcpyd2did_1_argument;
		_xdr_result = (xdrproc_t) xdr_dscudaResult;
		local = (char *(*)(char *, struct svc_req *)) _dscudamemcpyd2did_1;
		break;

	case dscudaMemcpyAsyncH2HId:
		_xdr_argument = (xdrproc_t) xdr_dscudamemcpyasynch2hid_1_argument;
		_xdr_result = (xdrproc_t) xdr_dscudaMemcpyAsyncH2HResult;
		local = (char *(*)(char *, struct svc_req *)) _dscudamemcpyasynch2hid_1;
		break;

	case dscudaMemcpyAsyncH2DId:
		_xdr_argument = (xdrproc_t) xdr_dscudamemcpyasynch2did_1_argument;
		_xdr_result = (xdrproc_t) xdr_dscudaResult;
		local = (char *(*)(char *, struct svc_req *)) _dscudamemcpyasynch2did_1;
		break;

	case dscudaMemcpyAsyncD2HId:
		_xdr_argument = (xdrproc_t) xdr_dscudamemcpyasyncd2hid_1_argument;
		_xdr_result = (xdrproc_t) xdr_dscudaMemcpyAsyncD2HResult;
		local = (char *(*)(char *, struct svc_req *)) _dscudamemcpyasyncd2hid_1;
		break;

	case dscudaMemcpyAsyncD2DId:
		_xdr_argument = (xdrproc_t) xdr_dscudamemcpyasyncd2did_1_argument;
		_xdr_result = (xdrproc_t) xdr_dscudaResult;
		local = (char *(*)(char *, struct svc_req *)) _dscudamemcpyasyncd2did_1;
		break;

	case dscudaMemcpyToSymbolH2DId:
		_xdr_argument = (xdrproc_t) xdr_dscudamemcpytosymbolh2did_1_argument;
		_xdr_result = (xdrproc_t) xdr_dscudaResult;
		local = (char *(*)(char *, struct svc_req *)) _dscudamemcpytosymbolh2did_1;
		break;

	case dscudaMemcpyToSymbolD2DId:
		_xdr_argument = (xdrproc_t) xdr_dscudamemcpytosymbold2did_1_argument;
		_xdr_result = (xdrproc_t) xdr_dscudaResult;
		local = (char *(*)(char *, struct svc_req *)) _dscudamemcpytosymbold2did_1;
		break;

	case dscudaMemcpyFromSymbolD2HId:
		_xdr_argument = (xdrproc_t) xdr_dscudamemcpyfromsymbold2hid_1_argument;
		_xdr_result = (xdrproc_t) xdr_dscudaMemcpyFromSymbolD2HResult;
		local = (char *(*)(char *, struct svc_req *)) _dscudamemcpyfromsymbold2hid_1;
		break;

	case dscudaMemcpyFromSymbolD2DId:
		_xdr_argument = (xdrproc_t) xdr_dscudamemcpyfromsymbold2did_1_argument;
		_xdr_result = (xdrproc_t) xdr_dscudaResult;
		local = (char *(*)(char *, struct svc_req *)) _dscudamemcpyfromsymbold2did_1;
		break;

	case dscudaMemsetId:
		_xdr_argument = (xdrproc_t) xdr_dscudamemsetid_1_argument;
		_xdr_result = (xdrproc_t) xdr_dscudaResult;
		local = (char *(*)(char *, struct svc_req *)) _dscudamemsetid_1;
		break;

	case dscudaHostAllocId:
		_xdr_argument = (xdrproc_t) xdr_dscudahostallocid_1_argument;
		_xdr_result = (xdrproc_t) xdr_dscudaHostAllocResult;
		local = (char *(*)(char *, struct svc_req *)) _dscudahostallocid_1;
		break;

	case dscudaMallocHostId:
		_xdr_argument = (xdrproc_t) xdr_RCsize;
		_xdr_result = (xdrproc_t) xdr_dscudaMallocHostResult;
		local = (char *(*)(char *, struct svc_req *)) _dscudamallochostid_1;
		break;

	case dscudaFreeHostId:
		_xdr_argument = (xdrproc_t) xdr_RCadr;
		_xdr_result = (xdrproc_t) xdr_dscudaResult;
		local = (char *(*)(char *, struct svc_req *)) _dscudafreehostid_1;
		break;

	case dscudaHostGetDevicePointerId:
		_xdr_argument = (xdrproc_t) xdr_dscudahostgetdevicepointerid_1_argument;
		_xdr_result = (xdrproc_t) xdr_dscudaHostGetDevicePointerResult;
		local = (char *(*)(char *, struct svc_req *)) _dscudahostgetdevicepointerid_1;
		break;

	case dscudaHostGetFlagsID:
		_xdr_argument = (xdrproc_t) xdr_RCadr;
		_xdr_result = (xdrproc_t) xdr_dscudaHostGetFlagsResult;
		local = (char *(*)(char *, struct svc_req *)) _dscudahostgetflagsid_1;
		break;

	case dscudaMallocArrayId:
		_xdr_argument = (xdrproc_t) xdr_dscudamallocarrayid_1_argument;
		_xdr_result = (xdrproc_t) xdr_dscudaMallocArrayResult;
		local = (char *(*)(char *, struct svc_req *)) _dscudamallocarrayid_1;
		break;

	case dscudaFreeArrayId:
		_xdr_argument = (xdrproc_t) xdr_RCadr;
		_xdr_result = (xdrproc_t) xdr_dscudaResult;
		local = (char *(*)(char *, struct svc_req *)) _dscudafreearrayid_1;
		break;

	case dscudaMemcpyToArrayH2HId:
		_xdr_argument = (xdrproc_t) xdr_dscudamemcpytoarrayh2hid_1_argument;
		_xdr_result = (xdrproc_t) xdr_dscudaMemcpyToArrayH2HResult;
		local = (char *(*)(char *, struct svc_req *)) _dscudamemcpytoarrayh2hid_1;
		break;

	case dscudaMemcpyToArrayH2DId:
		_xdr_argument = (xdrproc_t) xdr_dscudamemcpytoarrayh2did_1_argument;
		_xdr_result = (xdrproc_t) xdr_dscudaResult;
		local = (char *(*)(char *, struct svc_req *)) _dscudamemcpytoarrayh2did_1;
		break;

	case dscudaMemcpyToArrayD2HId:
		_xdr_argument = (xdrproc_t) xdr_dscudamemcpytoarrayd2hid_1_argument;
		_xdr_result = (xdrproc_t) xdr_dscudaMemcpyToArrayD2HResult;
		local = (char *(*)(char *, struct svc_req *)) _dscudamemcpytoarrayd2hid_1;
		break;

	case dscudaMemcpyToArrayD2DId:
		_xdr_argument = (xdrproc_t) xdr_dscudamemcpytoarrayd2did_1_argument;
		_xdr_result = (xdrproc_t) xdr_dscudaResult;
		local = (char *(*)(char *, struct svc_req *)) _dscudamemcpytoarrayd2did_1;
		break;

	case dscudaMallocPitchId:
		_xdr_argument = (xdrproc_t) xdr_dscudamallocpitchid_1_argument;
		_xdr_result = (xdrproc_t) xdr_dscudaMallocPitchResult;
		local = (char *(*)(char *, struct svc_req *)) _dscudamallocpitchid_1;
		break;

	case dscudaMemcpy2DToArrayH2HId:
		_xdr_argument = (xdrproc_t) xdr_dscudamemcpy2dtoarrayh2hid_1_argument;
		_xdr_result = (xdrproc_t) xdr_dscudaMemcpy2DToArrayH2HResult;
		local = (char *(*)(char *, struct svc_req *)) _dscudamemcpy2dtoarrayh2hid_1;
		break;

	case dscudaMemcpy2DToArrayH2DId:
		_xdr_argument = (xdrproc_t) xdr_dscudamemcpy2dtoarrayh2did_1_argument;
		_xdr_result = (xdrproc_t) xdr_dscudaResult;
		local = (char *(*)(char *, struct svc_req *)) _dscudamemcpy2dtoarrayh2did_1;
		break;

	case dscudaMemcpy2DToArrayD2HId:
		_xdr_argument = (xdrproc_t) xdr_dscudamemcpy2dtoarrayd2hid_1_argument;
		_xdr_result = (xdrproc_t) xdr_dscudaMemcpy2DToArrayD2HResult;
		local = (char *(*)(char *, struct svc_req *)) _dscudamemcpy2dtoarrayd2hid_1;
		break;

	case dscudaMemcpy2DToArrayD2DId:
		_xdr_argument = (xdrproc_t) xdr_dscudamemcpy2dtoarrayd2did_1_argument;
		_xdr_result = (xdrproc_t) xdr_dscudaResult;
		local = (char *(*)(char *, struct svc_req *)) _dscudamemcpy2dtoarrayd2did_1;
		break;

	case dscudaMemcpy2DH2HId:
		_xdr_argument = (xdrproc_t) xdr_dscudamemcpy2dh2hid_1_argument;
		_xdr_result = (xdrproc_t) xdr_dscudaMemcpy2DH2HResult;
		local = (char *(*)(char *, struct svc_req *)) _dscudamemcpy2dh2hid_1;
		break;

	case dscudaMemcpy2DH2DId:
		_xdr_argument = (xdrproc_t) xdr_dscudamemcpy2dh2did_1_argument;
		_xdr_result = (xdrproc_t) xdr_dscudaResult;
		local = (char *(*)(char *, struct svc_req *)) _dscudamemcpy2dh2did_1;
		break;

	case dscudaMemcpy2DD2HId:
		_xdr_argument = (xdrproc_t) xdr_dscudamemcpy2dd2hid_1_argument;
		_xdr_result = (xdrproc_t) xdr_dscudaMemcpy2DD2HResult;
		local = (char *(*)(char *, struct svc_req *)) _dscudamemcpy2dd2hid_1;
		break;

	case dscudaMemcpy2DD2DId:
		_xdr_argument = (xdrproc_t) xdr_dscudamemcpy2dd2did_1_argument;
		_xdr_result = (xdrproc_t) xdr_dscudaResult;
		local = (char *(*)(char *, struct svc_req *)) _dscudamemcpy2dd2did_1;
		break;

	case dscudaMemset2DId:
		_xdr_argument = (xdrproc_t) xdr_dscudamemset2did_1_argument;
		_xdr_result = (xdrproc_t) xdr_dscudaResult;
		local = (char *(*)(char *, struct svc_req *)) _dscudamemset2did_1;
		break;

	case dscudaMemcpyToSymbolAsyncH2DId:
		_xdr_argument = (xdrproc_t) xdr_dscudamemcpytosymbolasynch2did_1_argument;
		_xdr_result = (xdrproc_t) xdr_dscudaResult;
		local = (char *(*)(char *, struct svc_req *)) _dscudamemcpytosymbolasynch2did_1;
		break;

	case dscudaMemcpyToSymbolAsyncD2DId:
		_xdr_argument = (xdrproc_t) xdr_dscudamemcpytosymbolasyncd2did_1_argument;
		_xdr_result = (xdrproc_t) xdr_dscudaResult;
		local = (char *(*)(char *, struct svc_req *)) _dscudamemcpytosymbolasyncd2did_1;
		break;

	case dscudaMemcpyFromSymbolAsyncD2HId:
		_xdr_argument = (xdrproc_t) xdr_dscudamemcpyfromsymbolasyncd2hid_1_argument;
		_xdr_result = (xdrproc_t) xdr_dscudaMemcpyFromSymbolAsyncD2HResult;
		local = (char *(*)(char *, struct svc_req *)) _dscudamemcpyfromsymbolasyncd2hid_1;
		break;

	case dscudaMemcpyFromSymbolAsyncD2DId:
		_xdr_argument = (xdrproc_t) xdr_dscudamemcpyfromsymbolasyncd2did_1_argument;
		_xdr_result = (xdrproc_t) xdr_dscudaResult;
		local = (char *(*)(char *, struct svc_req *)) _dscudamemcpyfromsymbolasyncd2did_1;
		break;

	case dscudaCreateChannelDescId:
		_xdr_argument = (xdrproc_t) xdr_dscudacreatechanneldescid_1_argument;
		_xdr_result = (xdrproc_t) xdr_dscudaCreateChannelDescResult;
		local = (char *(*)(char *, struct svc_req *)) _dscudacreatechanneldescid_1;
		break;

	case dscudaGetChannelDescId:
		_xdr_argument = (xdrproc_t) xdr_RCadr;
		_xdr_result = (xdrproc_t) xdr_dscudaGetChannelDescResult;
		local = (char *(*)(char *, struct svc_req *)) _dscudagetchanneldescid_1;
		break;

	case dscudaBindTextureId:
		_xdr_argument = (xdrproc_t) xdr_dscudabindtextureid_1_argument;
		_xdr_result = (xdrproc_t) xdr_dscudaBindTextureResult;
		local = (char *(*)(char *, struct svc_req *)) _dscudabindtextureid_1;
		break;

	case dscudaBindTexture2DId:
		_xdr_argument = (xdrproc_t) xdr_dscudabindtexture2did_1_argument;
		_xdr_result = (xdrproc_t) xdr_dscudaBindTexture2DResult;
		local = (char *(*)(char *, struct svc_req *)) _dscudabindtexture2did_1;
		break;

	case dscudaBindTextureToArrayId:
		_xdr_argument = (xdrproc_t) xdr_dscudabindtexturetoarrayid_1_argument;
		_xdr_result = (xdrproc_t) xdr_dscudaResult;
		local = (char *(*)(char *, struct svc_req *)) _dscudabindtexturetoarrayid_1;
		break;

	case dscudaUnbindTextureId:
		_xdr_argument = (xdrproc_t) xdr_RCtexture;
		_xdr_result = (xdrproc_t) xdr_dscudaResult;
		local = (char *(*)(char *, struct svc_req *)) _dscudaunbindtextureid_1;
		break;

	case dscufftPlan3dId:
		_xdr_argument = (xdrproc_t) xdr_dscufftplan3did_1_argument;
		_xdr_result = (xdrproc_t) xdr_dscufftPlanResult;
		local = (char *(*)(char *, struct svc_req *)) _dscufftplan3did_1;
		break;

	case dscufftDestroyId:
		_xdr_argument = (xdrproc_t) xdr_u_int;
		_xdr_result = (xdrproc_t) xdr_dscufftResult;
		local = (char *(*)(char *, struct svc_req *)) _dscufftdestroyid_1;
		break;

	case dscufftExecC2CId:
		_xdr_argument = (xdrproc_t) xdr_dscufftexecc2cid_1_argument;
		_xdr_result = (xdrproc_t) xdr_dscufftResult;
		local = (char *(*)(char *, struct svc_req *)) _dscufftexecc2cid_1;
		break;

	default:
		svcerr_noproc (transp);
		return;
	}
	memset ((char *)&argument, 0, sizeof (argument));
	if (!svc_getargs (transp, (xdrproc_t) _xdr_argument, (caddr_t) &argument)) {
		svcerr_decode (transp);
		return;
	}
	result = (*local)((char *)&argument, rqstp);
	if (result != NULL && !svc_sendreply(transp, (xdrproc_t) _xdr_result, result)) {
		svcerr_systemerr (transp);
	}
	if (!svc_freeargs (transp, (xdrproc_t) _xdr_argument, (caddr_t) &argument)) {
		fprintf (stderr, "%s", "unable to free arguments");
		exit (1);
	}
	return;
}
