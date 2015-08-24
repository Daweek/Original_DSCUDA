#pragma once
#include <algorithm>
#include "kernel.h"
#include "warpscan.h"

#define MEM_PER_WARP (4096 * WARP_SIZE)
#define IF(x) (-(int)(x))

namespace {
  __device__ __forceinline__
  int ringAddr(const int i) {
    return i & (MEM_PER_WARP - 1);
  }

  __device__ __forceinline__
  bool applyMAC(const fvec3 sourceCenter,
		const float MAC,
		const CellData sourceData,
		const fvec3 targetCenter,
		const fvec3 targetSize) {
    fvec3 dX = abs(targetCenter - sourceCenter) - targetSize;
    dX += abs(dX);
    dX *= 0.5f;
    const float R2 = norm(dX);
    return R2 < fabsf(MAC) || sourceData.nbody() < 3;
  }

  __device__
  void approxAcc(fvec4 acc_i[2],
		 const fvec3 pos_i[2],
		 const int cellIdx,
		 const float EPS2) {
    fvec4 M4[NVEC4];
    float M[4*NVEC4];
    const fvec4 Xj = tex1Dfetch(texCellCenter, cellIdx);
    if (cellIdx >= 0) {
#pragma unroll
      for (int i=0; i<NVEC4; i++) M4[i] = tex1Dfetch(texMultipole, NVEC4*cellIdx+i);
    } else {
#pragma unroll
      for (int i=0; i<NVEC4; i++) M4[i] = 0.0f;
    }
    for (int j=0; j<WARP_SIZE; j++) {
      const fvec3 pos_j(__shfl(Xj[0],j), __shfl(Xj[1],j), __shfl(Xj[2],j));
#pragma unroll
      for (int i=0; i<NVEC4; i++) {
        M[4*i+0] = __shfl(M4[i][0], j);
        M[4*i+1] = __shfl(M4[i][1], j);
        M[4*i+2] = __shfl(M4[i][2], j);
        M[4*i+3] = __shfl(M4[i][3], j);
      }
      for (int k=0; k<2; k++)
	acc_i[k] = M2P(acc_i[k], pos_i[k], pos_j, *(fvecP*)M, EPS2);
    }
  }

  __device__
  void approxAcc(fvec4 acc_i[2],
		 fvec4 M4[NVEC4],
		 float M[4*NVEC4],
                 const fvec3 pos_i[2],
                 const int cellIdx,
                 const float EPS2) {
    const fvec4 Xj = tex1Dfetch(texCellCenter, cellIdx);
    for (int j=0; j<WARP_SIZE; j++) {
      const fvec3 pos_j(__shfl(Xj[0],j), __shfl(Xj[1],j), __shfl(Xj[2],j));
      const int cellIdxWarp = __shfl(cellIdx,j);
      if (cellIdxWarp >= 0) {
#pragma unroll
	for (int i=0; i<NVEC4; i++) M4[i] = tex1Dfetch(texMultipole, NVEC4*cellIdxWarp+i);
      } else {
#pragma unroll
	for (int i=0; i<NVEC4; i++) M4[i] = 0.0f;
      }
#pragma unroll
      for (int i=0; i<NVEC4; i++) {
        M[4*i+0] = M4[i][0];
        M[4*i+1] = M4[i][1];
        M[4*i+2] = M4[i][2];
        M[4*i+3] = M4[i][3];
      }
      for (int k=0; k<2; k++)
        acc_i[k] = M2P(acc_i[k], pos_i[k], pos_j, *(fvecP*)M, EPS2);
    }
  }

  __device__
  uint2 traverseWarp(fvec4 * acc_i,
		     fvec4 * M4,
		     float * M,
		     const fvec3 pos_i[2],
		     const fvec3 targetCenter,
		     const fvec3 targetSize,
		     const float EPS2,
		     const int2 rootRange,
		     volatile int * tempQueue,
		     int * cellQueue) {
    const int laneIdx = threadIdx.x & (WARP_SIZE-1);

    uint2 counters = {0,0};
    int approxQueue, directQueue;

    for (int root=rootRange.x; root<rootRange.y; root+=WARP_SIZE)
      if (root + laneIdx < rootRange.y)
	cellQueue[ringAddr(root - rootRange.x + laneIdx)] = root + laneIdx;
    int numSources = rootRange.y - rootRange.x;
    int newSources = 0;
    int oldSources = 0;
    int sourceOffset = 0;
    int approxOffset = 0;
    int bodyOffset = 0;

    while (numSources > 0) {                                    // While there are source cells to traverse
      const int sourceIdx = sourceOffset + laneIdx;             //  Source cell index of current lane
      const int sourceQueue = cellQueue[ringAddr(oldSources + sourceIdx)];//  Global source cell index in queue
      const fvec4 MAC = tex1Dfetch(texCellCenter, sourceQueue); //  Source cell center + MAC
      const fvec3 sourceCenter(MAC[0],MAC[1],MAC[2]);           //  Source cell center
      const CellData sourceData = tex1Dfetch(texCell, sourceQueue);//  Source cell data
      const bool isNode = sourceData.isNode();                  //  Is non-leaf cell
      const bool isClose = applyMAC(sourceCenter, MAC[3], sourceData, targetCenter, targetSize);//  Is too close for MAC
      const bool isSource = sourceIdx < numSources;             //  Source index is within bounds

      // Split
      const bool isSplit = isNode && isClose && isSource;       //  Source cell must be split
      const int childBegin = sourceData.child();                //  First child cell
      const int numChild = sourceData.nchild() & IF(isSplit);   //  Number of child cells (masked by split flag)
      const int numChildScan = inclusiveScanInt(numChild);      //  Inclusive scan of numChild
      const int numChildLane = numChildScan - numChild;         //  Exclusive scan of numChild
      const int numChildWarp = __shfl(numChildScan, WARP_SIZE-1);//  Total numChild of current warp
      sourceOffset += min(WARP_SIZE, numSources - sourceOffset);//  Increment source offset
      if (numChildWarp + numSources - sourceOffset > MEM_PER_WARP)//  If cell queue overflows
	return make_uint2(0xFFFFFFFF,0xFFFFFFFF);               //  Exit kernel
      int childIdx = oldSources + numSources + newSources + numChildLane;//  Child index of current lane
      for (int i=0; i<numChild; i++)                            //  Loop over child cells for each lane
	cellQueue[ringAddr(childIdx + i)] = childBegin + i;	//   Queue child cells
      newSources += numChildWarp;                               //  Increment source cell count for next loop

      // Approx
      const bool isApprox = !isClose && isSource;               //  Source cell can be used for M2P
      const uint approxBallot = __ballot(isApprox);             //  Gather approx flags
      const int numApproxLane = __popc(approxBallot & lanemask_lt());//  Exclusive scan of approx flags
      const int numApproxWarp = __popc(approxBallot);           //  Total isApprox for current warp
      int approxIdx = approxOffset + numApproxLane;             //  Approx cell index of current lane
      tempQueue[laneIdx] = approxQueue;                         //  Fill queue with remaining sources for approx
      if (isApprox && approxIdx < WARP_SIZE)                    //  If approx flag is true and index is within bounds
	tempQueue[approxIdx] = sourceQueue;                     //   Fill approx queue with current sources
      if (approxOffset + numApproxWarp >= WARP_SIZE) {          //  If approx queue is larger than the warp size
#if WARP_PER_CELL
	approxAcc(acc_i, M4, M, pos_i, tempQueue[laneIdx], EPS2);//  Call M2P kernel
#else
	approxAcc(acc_i, pos_i, tempQueue[laneIdx], EPS2);      //   Call M2P kernel
#endif
	approxOffset -= WARP_SIZE;                              //   Decrement approx queue size
	approxIdx = approxOffset + numApproxLane;               //   Update approx index using new queue size
	if (isApprox && approxIdx >= 0)                         //   If approx flag is true and index is within bounds
	  tempQueue[approxIdx] = sourceQueue;                   //    Fill approx queue with current sources
	counters.x += WARP_SIZE;                                //   Increment M2P counter
      }                                                         //  End if for approx queue size
      approxQueue = tempQueue[laneIdx];                         //  Free temp queue for use in direct
      approxOffset += numApproxWarp;                            //  Increment approx queue offset

      // Direct
      const bool isLeaf = !isNode;                              //  Is leaf cell
      bool isDirect = isClose && isLeaf && isSource;            //  Source cell can be used for P2P
      const int bodyBegin = sourceData.body();                  //  First body in cell
      const int numBodies = sourceData.nbody() & IF(isDirect);  //  Number of bodies in cell
      const int numBodiesScan = inclusiveScanInt(numBodies);    //  Inclusive scan of numBodies
      int numBodiesLane = numBodiesScan - numBodies;            //  Exclusive scan of numBodies
      int numBodiesWarp = __shfl(numBodiesScan, WARP_SIZE-1);   //  Total numBodies of current warp
      int tempOffset = 0;                                       //  Initialize temp queue offset
      while (numBodiesWarp > 0) {                               //  While there are bodies to process
	tempQueue[laneIdx] = 1;                                 //   Initialize body queue
	if (isDirect && (numBodiesLane < WARP_SIZE)) {          //   If direct flag is true and index is within bounds
	  isDirect = false;                                     //    Set flag as processed
	  tempQueue[numBodiesLane] = -1-bodyBegin;              //    Put body in queue
	}                                                       //   End if for direct flag
        const int bodyQueue = inclusiveSegscanInt(tempQueue[laneIdx], tempOffset);//  Inclusive segmented scan of temp queue
        tempOffset = __shfl(bodyQueue, WARP_SIZE-1);            //   Last lane has the temp queue offset
	if (numBodiesWarp >= WARP_SIZE) {                       //   If warp is full of bodies
	  const fvec4 pos = tex1Dfetch(texBody, bodyQueue);     //    Load position of source bodies
	  for (int j=0; j<WARP_SIZE; j++) {                     //    Loop over the warp size
	    const fvec3 pos_j(__shfl(pos[0], j),                //     Get source x value from lane j
			      __shfl(pos[1], j),                //     Get source y value from lane j
			      __shfl(pos[2], j));               //     Get source z value from lane j
	    const float q_j = __shfl(pos[3], j);                //     Get source w value from lane j
#pragma unroll                                                  //     Unroll loop
	    for (int k=0; k<2; k++)                             //     Loop over two targets
	      acc_i[k] = P2P(acc_i[k], pos_i[k], pos_j, q_j, EPS2);  //      Call P2P kernel
	  }                                                     //    End loop over the warp size
	  numBodiesWarp -= WARP_SIZE;                           //    Decrement body queue size
	  numBodiesLane -= WARP_SIZE;                           //    Derecment lane offset of body index
	  counters.y += WARP_SIZE;                              //    Increment P2P counter
	} else {                                                //   If warp is not entirely full of bodies
	  int bodyIdx = bodyOffset + laneIdx;                   //    Body index of current lane
	  tempQueue[laneIdx] = directQueue;                     //    Initialize body queue with saved values
	  if (bodyIdx < WARP_SIZE)                              //    If body index is less than the warp size
	    tempQueue[bodyIdx] = bodyQueue;                     //     Push bodies into queue
	  bodyOffset += numBodiesWarp;                          //    Increment body queue offset
	  if (bodyOffset >= WARP_SIZE) {                        //    If this causes the body queue to spill
	    const fvec4 pos = tex1Dfetch(texBody, tempQueue[laneIdx]);//  Load position of source bodies
	    for (int j=0; j<WARP_SIZE; j++) {                   //     Loop over the warp size
  	      const fvec3 pos_j(__shfl(pos[0], j),              //     Get source x value from lane j
	  			__shfl(pos[1], j),              //     Get source y value from lane j
				__shfl(pos[2], j));             //     Get source z value from lane j
	      const float q_j = __shfl(pos[3], j);              //     Get source w value from lane j
#pragma unroll                                                  //     Unroll loop
	      for (int k=0; k<2; k++)                           //     Loop over two targets
		acc_i[k] = P2P(acc_i[k], pos_i[k], pos_j, q_j, EPS2);// Call P2P kernel
	    }                                                   //     End loop over the warp size
	    bodyOffset -= WARP_SIZE;                            //     Decrement body queue size
	    bodyIdx -= WARP_SIZE;                               //     Decrement body index of current lane
	    if (bodyIdx >= 0)                                   //     If body index is valid
	      tempQueue[bodyIdx] = bodyQueue;                   //      Push bodies into queue
	    counters.y += WARP_SIZE;                            //     Increment P2P counter
	  }                                                     //    End if for body queue spill
	  directQueue = tempQueue[laneIdx];                     //    Free temp queue for use in approx
	  numBodiesWarp = 0;                                    //    Reset numBodies of current warp
	}                                                       //   End if for warp full of bodies
      }                                                         //  End while loop for bodies to process
      if (sourceOffset >= numSources) {                         //  If the current level is done
	oldSources += numSources;                               //   Update finished source size
	numSources = newSources;                                //   Update current source size
	sourceOffset = newSources = 0;                          //   Initialize next source size and offset
      }                                                         //  End if for level finalization
    }                                                           // End while for source cells to traverse
    if (approxOffset > 0) {                                     // If there are leftover approx cells
#if WARP_PER_CELL
      approxAcc(acc_i, M4, M, pos_i, laneIdx < approxOffset ? approxQueue : -1, EPS2);// Call M2P kernel
#else
      approxAcc(acc_i, pos_i, laneIdx < approxOffset ? approxQueue : -1, EPS2);// Call M2P kernel
#endif
      counters.x += approxOffset;                               //  Increment M2P counter
      approxOffset = 0;                                         //  Reset offset for approx
    }                                                           // End if for leftover approx cells
    if (bodyOffset > 0) {                                       // If there are leftover direct cells
      const int bodyQueue = laneIdx < bodyOffset ? directQueue : -1;// Get body index
      const fvec4 pos = bodyQueue >= 0 ? tex1Dfetch(texBody, bodyQueue) :// Load position of source bodies
	make_float4(0.0f, 0.0f, 0.0f, 0.0f);                    //  With padding for invalid lanes
      for (int j=0; j<WARP_SIZE; j++) {                         //  Loop over the warp size
	const fvec3 pos_j(__shfl(pos[0], j),                    //   Get source x value from lane j
			  __shfl(pos[1], j),                    //   Get source y value from lane j
			  __shfl(pos[2], j));                   //   Get source z value from lane j
	const float q_j = __shfl(pos[3], j);                    //   Get source w value from lane j
#pragma unroll                                                  //   Unroll loop
	for (int k=0; k<2; k++)                                 //   Loop over two targets
	  acc_i[k] = P2P(acc_i[k], pos_i[k], pos_j, q_j, EPS2); //    Call P2P kernel
      }                                                         //  End loop over the warp size
      counters.y += bodyOffset;                                 //  Increment P2P counter
      bodyOffset = 0;                                           //  Reset offset for direct
    }                                                           // End if for leftover direct cells
    return counters;                                            // Return M2P & P2P counters
  }

  __device__ uint64_t     sumP2PGlob = 0;
  __device__ unsigned int maxP2PGlob = 0;
  __device__ uint64_t     sumM2PGlob = 0;
  __device__ unsigned int maxM2PGlob = 0;

  __global__ __launch_bounds__(NTHREAD, 4)
    void traverse(const int numTargets,
		  const int images,
		  const float EPS2,
		  const float cycle,
		  const int2 * levelRange,
		  const fvec4 * bodyPos,
		  fvec4 * bodyAcc,
		  const int2 * targetRange,
		  int * globalPool) {
    const int laneIdx = threadIdx.x & (WARP_SIZE-1);
    const int warpIdx = threadIdx.x >> WARP_SIZE2;
    const int NWARP2 = NTHREAD2 - WARP_SIZE2;
    __shared__ int sharedPool[NTHREAD];
    __shared__ float4 sharedM4[NVEC4 << NWARP2];
    __shared__ float sharedM[NVEC4 << (NWARP2 + 2)];
    int * tempQueue = sharedPool + WARP_SIZE * warpIdx;
    int * cellQueue = globalPool + MEM_PER_WARP * ((blockIdx.x<<NWARP2) + warpIdx);
    fvec4 * M4 = reinterpret_cast<fvec4*>(sharedM4 + NVEC4 * warpIdx);
    float * M = sharedM + 4 * NVEC4 * warpIdx;
    while (1) {
      int targetIdx = 0;
      if (laneIdx == 0)
        targetIdx = atomicAdd(&counterGlob, 1);
      targetIdx = __shfl(targetIdx, 0, WARP_SIZE);
      if (targetIdx >= numTargets) return;

      const int2 target = targetRange[targetIdx];
      const int bodyBegin = target.x;
      const int bodyEnd   = target.x+target.y;
      fvec3 pos_i[2], pos_p[2];
      for (int i=0; i<2; i++) {
        const int bodyIdx = min(bodyBegin+i*WARP_SIZE+laneIdx, bodyEnd-1);
	pos_i[i] = make_fvec3(fvec4(bodyPos[bodyIdx]));
      }
      fvec3 Xmin = pos_i[0];
      fvec3 Xmax = Xmin;
      for (int i=0; i<2; i++)
	getMinMax(Xmin, Xmax, pos_i[i]);
      Xmin[0] = __shfl(Xmin[0],0);
      Xmin[1] = __shfl(Xmin[1],0);
      Xmin[2] = __shfl(Xmin[2],0);
      Xmax[0] = __shfl(Xmax[0],0);
      Xmax[1] = __shfl(Xmax[1],0);
      Xmax[2] = __shfl(Xmax[2],0);
      const fvec3 targetCenter = (Xmax+Xmin) * 0.5f;
      const fvec3 targetSize = (Xmax-Xmin) * 0.5f;
      fvec4 acc_i[2] = {0.0f, 0.0f};
      fvec3 Xperiodic = 0.0f;
      int numP2P = 0, numM2P = 0;
      for (int ix=-images; ix<=images; ix++) {
	for (int iy=-images; iy<=images; iy++) {
	  for (int iz=-images; iz<=images; iz++) {
	    Xperiodic[0] = ix * cycle;
	    Xperiodic[1] = iy * cycle;
	    Xperiodic[2] = iz * cycle;
	    const fvec3 targetPeriodic = targetCenter - Xperiodic;
	    for (int i=0; i<2; i++)
	      pos_p[i] = pos_i[i] - Xperiodic;
	    const uint2 counters = traverseWarp(acc_i, M4, M, pos_p, targetPeriodic, targetSize,
						EPS2, levelRange[1], tempQueue, cellQueue);
	    assert(!(counters.x == 0xFFFFFFFF && counters.y == 0xFFFFFFFF));
	    numM2P += counters.x;
	    numP2P += counters.y;
	  }
	}
      }
      int maxP2P = numP2P;
      int sumP2P = 0;
      int maxM2P = numM2P;
      int sumM2P = 0;
      const int bodyIdx = bodyBegin + laneIdx;
      for (int i=0; i<2; i++) {
	if (i*WARP_SIZE + bodyIdx < bodyEnd) {
	  sumM2P += numM2P;
	  sumP2P += numP2P;
	}
      }
#pragma unroll
      for (int i=0; i<WARP_SIZE2; i++) {
	maxP2P  = max(maxP2P, __shfl_xor(maxP2P, 1<<i));
	sumP2P += __shfl_xor(sumP2P, 1<<i);
	maxM2P  = max(maxM2P, __shfl_xor(maxM2P, 1<<i));
	sumM2P += __shfl_xor(sumM2P, 1<<i);
      }
      if (laneIdx == 0) {
	atomicMax(&maxP2PGlob, maxP2P);
	atomicAdd((unsigned long long*)&sumP2PGlob, (unsigned long long)sumP2P);
	atomicMax(&maxM2PGlob, maxM2P);
	atomicAdd((unsigned long long*)&sumM2PGlob, (unsigned long long)sumM2P);
      }
      for (int i=0; i<2; i++) {
	if (bodyIdx + i * WARP_SIZE < bodyEnd)
	  bodyAcc[i*WARP_SIZE + bodyIdx] = acc_i[i];
      }
    }
  }

  __global__
  void directKernel(const int numSource,
		    const int images,
		    const float EPS2,
		    const float cycle,
		    fvec4 * bodyAcc) {
    const int laneIdx = threadIdx.x & (WARP_SIZE-1);
    const int numChunk = (numSource - 1) / gridDim.x + 1;
    const int numWarpChunk = (numChunk - 1) / WARP_SIZE + 1;
    const int blockOffset = blockIdx.x * numChunk;
    fvec4 pos = tex1Dfetch(texBody, threadIdx.x);
    const fvec3 pos_i(pos[0],pos[1],pos[2]);
    kvec4 acc;
    fvec3 Xperiodic = 0.0f;
    for (int jb=0; jb<numWarpChunk; jb++) {
      const int sourceIdx = min(blockOffset+jb*WARP_SIZE+laneIdx, numSource-1);
      pos = tex1Dfetch(texBody, sourceIdx);
      if (sourceIdx >= numSource) pos[3] = 0;
      for (int ix=-images; ix<=images; ix++) {
	for (int iy=-images; iy<=images; iy++) {
	  for (int iz=-images; iz<=images; iz++) {
	    Xperiodic[0] = ix * cycle;
	    Xperiodic[1] = iy * cycle;
	    Xperiodic[2] = iz * cycle;
	    for (int j=0; j<WARP_SIZE; j++) {
	      const fvec3 pos_j(__shfl(pos[0],j),__shfl(pos[1],j),__shfl(pos[2],j));
	      const float q_j = __shfl(pos[3],j);
	      fvec3 dX = pos_j - pos_i - Xperiodic;
	      const float R2 = norm(dX) + EPS2;
	      const float invR = rsqrtf(R2);
	      const float invR2 = invR * invR;
	      const float invR1 = q_j * invR;
	      dX *= invR1 * invR2;
	      acc[0] -= invR1;
	      acc[1] += dX[0];
	      acc[2] += dX[1];
	      acc[3] += dX[2];
	    }
	  }
	}
      }
    }
    const int targetIdx = blockIdx.x * blockDim.x + threadIdx.x;
    bodyAcc[targetIdx] = fvec4(acc[0],acc[1],acc[2],acc[3]);
  }
}

class Traversal {
public:
  fvec4 approx(const int numTargets,
	       const int images,
	       const float eps,
	       const float cycle,
	       cudaVec<fvec4> & bodyPos,
	       cudaVec<fvec4> & bodyPos2,
	       cudaVec<fvec4> & bodyAcc,
	       cudaVec<int2> & targetRange,
	       cudaVec<CellData> & sourceCells,
	       cudaVec<fvec4> & sourceCenter,
	       cudaVec<fvec4> & Multipole,
	       cudaVec<int2> & levelRange) {
    const int NWARP = 1 << (NTHREAD2 - WARP_SIZE2);
    const int NBLOCK = (numTargets - 1) / NTHREAD + 1;
    const int poolSize = MEM_PER_WARP * NWARP * NBLOCK;
    const int numBodies = bodyPos.size();
    const int numSources = sourceCells.size();
    sourceCells.bind(texCell);
    sourceCenter.bind(texCellCenter);
    Multipole.bind(texMultipole);
    bodyPos.bind(texBody);
    cudaVec<int> globalPool(poolSize);
    cudaDeviceSynchronize();
    const double t0 = get_time();
    CUDA_SAFE_CALL(cudaFuncSetCacheConfig(&traverse, cudaFuncCachePreferL1));
    traverse<<<NBLOCK,NTHREAD>>>(numTargets, images, eps*eps, cycle, levelRange.d(),
				 bodyPos2.d(), bodyAcc.d(),
				 targetRange.d(), globalPool.d());
    kernelSuccess("traverse");
    const double dt = get_time() - t0;

    uint64_t sumP2P, sumM2P;
    unsigned int maxP2P, maxM2P;
    CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&sumP2P, sumP2PGlob, sizeof(uint64_t)));
    CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&maxP2P, maxP2PGlob, sizeof(unsigned int)));
    CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&sumM2P, sumM2PGlob, sizeof(uint64_t)));
    CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&maxM2P, maxM2PGlob, sizeof(unsigned int)));
    fvec4 interactions;
    interactions[0] = sumP2P * 1.0 / numBodies;
    interactions[1] = maxP2P;
    interactions[2] = sumM2P * 1.0 / numBodies;
    interactions[3] = maxM2P;
    float flops = (interactions[0] * 20 + interactions[2] * 2 * pow(P,3)) * numBodies / dt / 1e12;
    fprintf(stdout,"Traverse             : %.7f s (%.7f TFlops)\n",dt,flops);

    sourceCells.unbind(texCell);
    sourceCenter.unbind(texCellCenter);
    Multipole.unbind(texMultipole);
    bodyPos.unbind(texBody);
    return interactions;
  }

  void direct(const int numTarget,
	      const int numBlock,
	      const int images,
	      const float eps,
	      const float cycle,
	      cudaVec<fvec4> & bodyPos2,
	      cudaVec<fvec4> & bodyAcc2) {
    const int numBodies = bodyPos2.size();
    bodyPos2.bind(texBody);
    directKernel<<<numBlock,numTarget>>>(numBodies, images, eps*eps, cycle, bodyAcc2.d());
    bodyPos2.unbind(texBody);
    cudaDeviceSynchronize();
  }
};
