#define SETUP_IBV_PACKET_BUF(mthd, vid, rid)                            \
        IbvConnection *conn = (IbvConnection *)Cmid[vid][rid]->context; \
        IbvMemcpyH2DInvokeHdr *spkt = (IbvMemcpyH2DInvokeHdr *)conn->rdma_local_region; \
        IbvMemcpyH2DReturnHdr *rpkt = (IbvMemcpyH2DReturnHdr *)conn->rdma_remote_region; \
        int spktsize = sizeof(IbvMemcpyH2DInvokeHdr);                \
        rdmaWaitReadyToKickoff(conn);                                   \
        spkt->method = RCMethodMemcpyH2D;
