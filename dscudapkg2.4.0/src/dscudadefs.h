#ifndef _DSCUDADEFS_H
#define _DSCUDADEFS_H

#define RC_NSERVERMAX  4    // max # of servers per node.
#define RC_NDEVICEMAX  4    // max # of GPU devices per node.
#define RC_NP2PMAX    (RC_NSERVERMAX * RC_NSERVERMAX)    // max # of P2P connections per node.
#define RC_NREDUNDANCYMAX 4 // max redundancy per server.
#define RC_NVDEVMAX 64      // max # of virtual devices per client.
#define RC_NPTHREADMAX 64   // max # of pthreads which use virtual devices.

#define RC_BUFSIZE (1024*1024) // size (in byte) of send/receive buffers for socket.
#define RC_NKMODULEMAX 128  // max # of kernel modules to be stored.
#define RC_NKFUNCMAX   128  // max # of kernel functions to be stored.
#define RC_NSYMBOLMAX   128  // max # of global symbols (defined in the device code) to be stored.
#define RC_NKARGMAX    64   // max # of arguments of a kernel.
#define RC_KARGLEN     64   // max size (in byte) for one argument of a kernel.
#define RC_KMODULENAMELEN 128   // max length of a kernel-module name.
#define RC_KNAMELEN      128   // max length of a kernel-function name.
#define RC_KMODULEIMAGELEN (1024*1024*2)   // max length of a kernel-image (approximately the size of .ptx file).
#define RC_SNAMELEN       512   // max length of a symbol name.

#define RC_CACHE_MODULE (1) // set 1 for practical use. set 0 to disable module caching mechanism, just for debugging.
#define RC_CLIENT_CACHE_LIFETIME (30) // period (in second) for a module sent by a client is cached. should be shorter enough than RC_SERVER_CACHE_LIFETIME.
#define RC_SERVER_CACHE_LIFETIME (RC_CLIENT_CACHE_LIFETIME+30) // period (in second) for a module loaded by dscudasvr is cached.

#define RC_SUPPORT_PAGELOCK (0)  // set 1 if cudaMallocHost(), cudaMemcpyAsync(), cudaFreeHost() are truly implemented, i.e., with page-locked memory.
#define RC_SUPPORT_STREAM (0)
#define RC_SUPPORT_CONCURRENT_EXEC (0)

#define RC_DAEMON_IP_PORT  (65432)
#define RC_SERVER_IP_PORT  (RC_DAEMON_IP_PORT+1)

#define RC_CACHE_IPCMEM (0) // set 1 for better performance of inter process memory copy. for now this does not work fine so set 0.

#endif //  _DSCUDADEFS_H
