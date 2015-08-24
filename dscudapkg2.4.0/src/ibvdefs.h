#ifndef IBVDEFS_H
#define IBVDEFS_H

#ifndef TCP_ONLY

#include <rdma/rdma_cma.h>

#define TEST_NZ(x) do { if ( (x)) {perror(#x " failed."); exit(EXIT_FAILURE); } } while (0)
#define TEST_Z(x)  do { if (!(x)) {perror(#x " failed."); exit(EXIT_FAILURE); } } while (0)

// RDMA buffer
#define RC_NWR_PER_POST (16) // max # of work requests in a single post.

// #define RC_SGE_SIZE (1024 * 1024 * 2) // size per segment.
#define RC_SGE_SIZE (1024 * 1024 * 16) // size per segment.

#define RC_WR_MAX (RC_NWR_PER_POST * 16) // max # of work requests stored in QP.

#define RC_RDMA_BUF_SIZE (RC_NWR_PER_POST * RC_SGE_SIZE) // size of the rdma buf.


#if RC_RDMA_BUF_SIZE  < RC_KMODULEIMAGELEN
#error "RC_RDMA_BUF_SIZE too small."
// you can reduce RC_KMODULEIMAGELEN if you know your .ptx files are small enough.
#endif

#define RC_SERVER_IBV_CQ_SIZE (RC_WR_MAX)
#define RC_CLIENT_IBV_CQ_SIZE (65536)

//#define RC_CLIENT_IBV_CQ_SIZE (RC_WR_MAX)

#define RC_IBV_IP_PORT_BASE  (65432)
#define RC_IBV_TIMEOUT (500)  // in milli second.

struct message {
    struct ibv_mr mr[RC_NWR_PER_POST];
};

enum rdma_state_t {
    STATE_INIT,
    STATE_READY,
    STATE_BUSY,
};

typedef struct {
    // IB Verb resources.
    struct rdma_cm_id *id;
    struct ibv_qp *qp;
    struct ibv_context *ibvctx;
    struct ibv_pd *pd;
    struct ibv_cq *cq;
    struct ibv_comp_channel *comp_channel;

    // message buf.
    struct message *recv_msg;
    struct message *send_msg;

    // rdma buf.
    char *rdma_local_region;
    char *rdma_remote_region;

    // MR for message buf.
    struct ibv_mr *recv_mr;
    struct ibv_mr *send_mr;
    struct ibv_mr peer_mr[RC_NWR_PER_POST];

    // MR for rdma buf.
    struct ibv_mr *rdma_local_mr[RC_NWR_PER_POST];
    struct ibv_mr *rdma_remote_mr[RC_NWR_PER_POST];

    // misc.
    pthread_t cq_poller_thread;
    int connected;
    enum rdma_state_t rdma_state;
    int rdma_nreq_pending;
    int rdma_imm_recvd;
    pthread_mutex_t rdma_mutex;
    pthread_mutex_t inuse_mutex;
    void (*on_completion_handler)(struct ibv_wc *);
} IbvConnection;

void rdmaBuildConnection(struct rdma_cm_id *id, bool is_server);
void rdmaBuildParams(struct rdma_conn_param *params);
void rdmaDestroyConnection(IbvConnection *conn);
void rdmaWaitEvent(struct rdma_event_channel *ec, rdma_cm_event_type et, int (*handler)(struct rdma_cm_id *id));
void rdmaWaitRDMAImmRecvd(IbvConnection *conn);
void rdmaWaitReadyToKickoff(IbvConnection *conn);
void rdmaWaitReadyToDisconnect(IbvConnection *conn);
void rdmaKickoff(IbvConnection *conn, int length);
void rdmaPipelinedKickoff(IbvConnection *conn, int length, char *payload_buf, char *payload_src, int payload_size, int chunk_size);
void rdmaImmKickoff(IbvConnection *conn, int my_lkey, void *src, int peer_rkey, void *dst, int length);
struct ibv_mr *rdmaRegisterMR(struct ibv_pd *pd, void *adr, int length, int is_send);
void rdmaPostReceive(IbvConnection *conn);
void rdmaSendMr(IbvConnection *conn);
int dscudaMyServerId(void);

#endif // TCP_ONLY

#endif // IBVDEFS_H
