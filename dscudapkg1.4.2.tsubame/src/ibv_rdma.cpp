#include "ibv_rdma.h"

static void check_msgsanity(struct ibv_wc *wc);
static void print_rdma_state(enum rdma_state_t state);
static void build_verbs(IbvConnection *conn, struct ibv_context *verbs, bool is_server);
static void build_qp_attr(IbvConnection *conn, struct ibv_qp_init_attr *qp_attr);
static void *poll_cq(void *);
static void post_recv_msg(IbvConnection *conn);
static void on_completion(struct ibv_wc *wc, bool is_server);
static void register_memory(IbvConnection *conn);
static void post_send_msg(IbvConnection *conn);
static void kickoff_rdma_with_offset(intptr_t offset, IbvConnection *conn, int length);

static void (*OnCompletionHandler)(struct ibv_wc *) = NULL;

static pthread_mutex_t RdmaMutex = PTHREAD_MUTEX_INITIALIZER;

/*
 * private funcitons
 */

static void
check_msgsanity(struct ibv_wc *wc, bool is_server)
{
#if 1
    if (wc->status != IBV_WC_SUCCESS) {
        WARN(0, "%s\n", ibv_wc_status_str(wc->status));
    }
#else

    IbvConnection *conn = (IbvConnection *)(uintptr_t)wc->wr_id;
    int exit_on_err = false;

    switch (wc->opcode) {
      case IBV_WC_RECV:
        WARN(0, "completion RECV message");
        exit_on_err = true;
        break;
      case IBV_WC_SEND:
        WARN(0, "completion SEND message");
        exit_on_err = true;
        break;
      case IBV_WC_RDMA_WRITE:
        WARN(0, "completion RDMA_WRITE");
        break;
      case IBV_WC_RDMA_READ:
        WARN(0, "completion RDMA_READ");
        break;
      default:
        WARN(0, "completion of unknown type : 0x%x  status : %s\n",
                wc->opcode, ibv_wc_status_str(wc->status));
        exit(1);
    }

    if (wc->status != IBV_WC_SUCCESS) {
        WARN(0, " NG. %s\n", ibv_wc_status_str(wc->status));
        if (exit_on_err) {
            exit(1);
        }
    }
    else {
        WARN(0, " OK\n");
    }
#endif
}


static void
print_rdma_state(enum rdma_state_t state)
{
    WARN(0, "rdma state : ");
    switch (state) {
      case STATE_INIT:
        WARN(0, "INIT");
        break;
      case STATE_READY:
        WARN(0, "READY");
        break;
      case STATE_BUSY:
        WARN(0, "BUSY");
        break;
      default:
        WARN(0, "unknown");
    }
    WARN(0, "\n");
}

static void
build_verbs(IbvConnection *conn, struct ibv_context *verbs, bool is_server)
{
    conn->ibvctx = verbs;
    TEST_Z(conn->pd = ibv_alloc_pd(conn->ibvctx));
    TEST_Z(conn->comp_channel = ibv_create_comp_channel(conn->ibvctx));
    int cqe = is_server ? RC_SERVER_IBV_CQ_SIZE : RC_CLIENT_IBV_CQ_SIZE;
    TEST_Z(conn->cq = ibv_create_cq(conn->ibvctx, cqe, NULL, conn->comp_channel, 0));
    TEST_NZ(ibv_req_notify_cq(conn->cq, 0));
    TEST_NZ(pthread_create(&conn->cq_poller_thread, NULL, poll_cq, conn));
}

static void
build_qp_attr(IbvConnection *conn, struct ibv_qp_init_attr *qp_attr)
{
    memset(qp_attr, 0, sizeof(*qp_attr));

    qp_attr->send_cq = conn->cq;
    qp_attr->recv_cq = conn->cq;
    qp_attr->qp_type = IBV_QPT_RC;

    qp_attr->cap.max_send_wr = RC_WR_MAX;
    qp_attr->cap.max_recv_wr = RC_WR_MAX;
    qp_attr->cap.max_send_sge = 1;
    qp_attr->cap.max_recv_sge = 1;
}

static void *
poll_cq(void *ctx)
{
    struct ibv_cq *cq;
    struct ibv_wc wc;
    IbvConnection *conn = (IbvConnection *)ctx;
    int ne;
    static int cnt = 0;

    while (1) {
        TEST_NZ(ibv_get_cq_event(conn->comp_channel, &cq, &ctx));
        ibv_ack_cq_events(cq, 1);
        TEST_NZ(ibv_req_notify_cq(cq, 0));

        do {
            ne = ibv_poll_cq(cq, 1, &wc);
            if (ne < 0) {
                WARN(0, "ibv_poll_cq() failed.\n");
                exit(EXIT_FAILURE);
            }
            if (ne == 0) {
                continue; // escape from this do block.
            }
            if (wc.status != IBV_WC_SUCCESS) {
                WARN(0, "ibv_poll_cq() got WC with status %d, %s.\n",
                        wc.status, ibv_wc_status_str(wc.status));
                //                exit(1);
            }
            cnt++;
            //             WARN(0, "%d\n", cnt);
            //            WARN(0, "poll WCs.\n", cnt);
            (OnCompletionHandler)(&wc);
        } while (ne);
    }

    return NULL;
}

static void
on_completion(struct ibv_wc *wc, bool is_server)
{
    int i;
    IbvConnection *conn = (IbvConnection *)(uintptr_t)wc->wr_id;

    //    check_msgsanity(wc, is_server);

    switch (wc->opcode) {
      case IBV_WC_RECV:
        for (i = 0; i < RC_NWR_PER_POST; i++) {
            memcpy(conn->peer_mr + i, conn->recv_msg->mr + i, sizeof(struct ibv_mr));
            WARN(4, "peer_mr[%d]  adr:%lx  rkey:%x  lkey:%x\n",
                 i, conn->recv_msg->mr[i].addr, conn->recv_msg->mr[i].rkey,
                 conn->recv_msg->mr[i].lkey);
        }

        // server receives MR before sending ours, so send ours back.
        if (is_server) {
            rdmaSendMr(conn);
        }
        pthread_mutex_lock(&RdmaMutex);
        conn->rdma_state = STATE_READY;
        pthread_mutex_unlock(&RdmaMutex);
        post_recv_msg(conn); // rearm for next message.
        break;

      case IBV_WC_SEND:
        // nop.
        break;

      case IBV_WC_RDMA_WRITE:
        pthread_mutex_lock(&RdmaMutex);
        switch (conn->rdma_state) {
          case STATE_READY:
            break;
          case STATE_BUSY: // kickoff_rdma_with_offset(offset, ...) done.
            conn->rdma_nreq_pending--;
            if (conn->rdma_nreq_pending == 0) {
                conn->rdma_state = STATE_READY;
            }
            break;
          default:
            WARN(0, "invalid conn->rdma_state:%d\n", conn->rdma_state);
            exit(1);
        }
        pthread_mutex_unlock(&RdmaMutex);
        break;

      default:
        WARN(0, "completion with unexpected opcode : 0x%x\n", wc->opcode);
        exit(1);
    }
}

static void
register_memory(IbvConnection *conn)
{
    int i;

    conn->send_msg = (message *)malloc(sizeof(struct message));
    conn->recv_msg = (message *)malloc(sizeof(struct message));
    conn->rdma_local_region = (char *)calloc(RC_RDMA_BUF_SIZE, 1);
    conn->rdma_remote_region = (char *)calloc(RC_RDMA_BUF_SIZE, 1);
    pthread_mutex_lock(&RdmaMutex);
    conn->rdma_nreq_pending = 0;
    pthread_mutex_unlock(&RdmaMutex);

    TEST_Z(conn->send_mr = ibv_reg_mr(conn->pd, 
                                      conn->send_msg, 
                                      sizeof(struct message), 
                                      IBV_ACCESS_LOCAL_WRITE));

    TEST_Z(conn->recv_mr = ibv_reg_mr(conn->pd, 
                                      conn->recv_msg, 
                                      sizeof(struct message), 
                                      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE ));

    for (i = 0; i < RC_NWR_PER_POST; i++) {
        TEST_Z(conn->rdma_local_mr[i] = ibv_reg_mr(conn->pd, 
                                                   conn->rdma_local_region + RC_SGE_SIZE * i,
                                                   RC_SGE_SIZE, 
                                                   IBV_ACCESS_LOCAL_WRITE));

        TEST_Z(conn->rdma_remote_mr[i] = ibv_reg_mr(conn->pd, 
                                                    conn->rdma_remote_region + RC_SGE_SIZE * i,
                                                    RC_SGE_SIZE, 
                                                    IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE));
        WARN(4, "set  remote_mr[%d]  adr:%lx  rkey:%x  lkey:%x\n",
             i, conn->rdma_remote_mr[i]->addr, conn->rdma_remote_mr[i]->rkey,
             conn->rdma_remote_mr[i]->lkey);
    }
}

static void
post_recv_msg(IbvConnection *conn)
{
    struct ibv_recv_wr wr, *bad_wr = NULL;
    struct ibv_sge sge;

    wr.wr_id = (uintptr_t)conn;
    wr.next = NULL;
    wr.sg_list = &sge;
    wr.num_sge = 1;

    sge.addr = (uintptr_t)conn->recv_msg;
    sge.length = sizeof(struct message);
    sge.lkey = conn->recv_mr->lkey;

    WARN(4, "post_recv_msg adr:%lx  length:%d  lkey:%x\n",
         sge.addr, sge.length, sge.lkey);

    TEST_NZ(ibv_post_recv(conn->qp, &wr, &bad_wr));
    WARN(3, "posted a recv WR.\n");
}

static void
post_send_msg(IbvConnection *conn)
{
    struct ibv_send_wr wr, *bad_wr = NULL;
    struct ibv_sge sge;
    memset(&wr, 0, sizeof(wr));

    wr.wr_id = (uintptr_t)conn;
    wr.opcode = IBV_WR_SEND;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.send_flags = IBV_SEND_SIGNALED;

    sge.addr = (uintptr_t)conn->send_msg;
    sge.length = sizeof(struct message);
    sge.lkey = conn->send_mr->lkey;

    while (!conn->connected);


    WARN(4, "post_send_msg adr:%x  length:%d  lkey:%x\n",
         sge.addr, sge.length, sge.lkey);
    {
        int i;
        struct message *mp = conn->send_msg;
        for (i = 0; i < RC_NWR_PER_POST; i++) {
            WARN(4, "send remote_mr[%d]  adr:%lx  rkey:%x  lkey:%x\n",
                 i, mp->mr[i].addr, mp->mr[i].rkey, mp->mr[i].lkey);
        }
    }

    TEST_NZ(ibv_post_send(conn->qp, &wr, &bad_wr));
    WARN(3, "posted a send WR.\n");
}

static void
kickoff_rdma_with_offset(intptr_t offset, IbvConnection *conn, int length)
{
    int i, i_start, i_end, off;
    struct ibv_send_wr wr_list[RC_NWR_PER_POST], *bad_wr = NULL;
    struct ibv_sge sge_list[RC_NWR_PER_POST];

    // calculate the 1st & last segment.
    for (i = 0, off = 0; ; i++, off += RC_SGE_SIZE) {
        if (RC_NWR_PER_POST <= i) {
            WARN(0, "kickoff_rmda_with_offset(): offset (=%d) exceeds RC_RDMA_BUF_SIZE (=%d).\n",
                 offset, RC_RDMA_BUF_SIZE);
            exit(1);
        }
        if (offset < off + RC_SGE_SIZE) break;
    }
    i_start = i;

    for ( ; ; i++, off += RC_SGE_SIZE) {
        if (RC_NWR_PER_POST < i) {
            WARN(0, "kickoff_rmda_with_offset(): offset + length (=%d) exceeds RC_RDMA_BUF_SIZE (=%d).\n",
                 offset + length, RC_RDMA_BUF_SIZE);
            exit(1);
        }
        if (length + offset < off) break;
    }
    i_end = i;

    WARN(4, "i_start:%d  i_end:%d  offset:%d  length:%d\n",
         i_start, i_end, offset, length);

    // configure all segments in between the 1st & last.
    // each segment is associated with one work request.
    memset(wr_list, 0, sizeof(wr_list));
    for (i = i_start; i < i_end; i++) {
        wr_list[i].wr_id = (uintptr_t)conn;
        if (i == i_end - 1) {
            wr_list[i].next = NULL;
        }
        else {
            wr_list[i].next = wr_list + i + 1;
        }
        wr_list[i].opcode = IBV_WR_RDMA_WRITE;
        wr_list[i].sg_list = sge_list + i;
        wr_list[i].num_sge = 1;
        wr_list[i].send_flags = IBV_SEND_SIGNALED;
        wr_list[i].wr.rdma.remote_addr = (uintptr_t)conn->peer_mr[i].addr;
        wr_list[i].wr.rdma.rkey = conn->peer_mr[i].rkey;

        sge_list[i].lkey = conn->rdma_local_mr[i]->lkey;
        sge_list[i].addr = (uintptr_t)conn->rdma_local_region + RC_SGE_SIZE * i;
        sge_list[i].length = RC_SGE_SIZE;

        if (i == i_start) {
            wr_list[i].wr.rdma.remote_addr += offset - RC_SGE_SIZE * i;
            sge_list[i].addr += offset - RC_SGE_SIZE * i;
            sge_list[i].length = RC_SGE_SIZE * (i + 1) - offset;
        }
        if (i == i_end - 1) {
            sge_list[i].length -= RC_SGE_SIZE * (i + 1) - (length + offset);
        }
        WARN(4, "radr:0x%lx  rkey:0x%x  ladr:0x%lx  lkey:0x%x  length:%d\n",
             wr_list[i].wr.rdma.remote_addr, wr_list[i].wr.rdma.rkey, 
             sge_list[i].addr, sge_list[i].lkey, sge_list[i].length);
             
    }
    conn->rdma_nreq_pending += i_end - i_start;
    TEST_NZ(ibv_post_send(conn->qp, wr_list + i_start, &bad_wr));
}


/*
 * public funcitons
 */

void
rdmaBuildConnection(struct rdma_cm_id *id, bool is_server)
{
    IbvConnection *conn;
    struct ibv_qp_init_attr qp_attr;

    id->context = conn = (IbvConnection *)malloc(sizeof(IbvConnection));

    build_verbs(conn, id->verbs, is_server);
    build_qp_attr(conn, &qp_attr);

    TEST_NZ(rdma_create_qp(id, conn->pd, &qp_attr));

    conn->id = id;
    conn->qp = id->qp;

    conn->connected = 0;

    register_memory(conn);
    post_recv_msg(conn);
}

void
rdmaBuildParams(struct rdma_conn_param *params)
{
    memset(params, 0, sizeof(*params));

    params->initiator_depth = params->responder_resources = 1;
    params->rnr_retry_count = 7; /* infinite retry */
}

void
rdmaDestroyConnection(IbvConnection *conn)
{
    int i;

    rdma_destroy_qp(conn->id);

    ibv_dereg_mr(conn->send_mr);
    ibv_dereg_mr(conn->recv_mr);

    for (i = 0; i < RC_NWR_PER_POST; i++) {
        ibv_dereg_mr(conn->rdma_local_mr[i]);
        ibv_dereg_mr(conn->rdma_remote_mr[i]);
    }

    free(conn->send_msg);
    free(conn->recv_msg);
    free(conn->rdma_local_region);
    free(conn->rdma_remote_region);

    rdma_destroy_id(conn->id);

    free(conn);
}

void
rdmaWaitReadyToKickoff(IbvConnection *conn)
{
    while (conn->rdma_state != STATE_READY) {
        // nop.
    }
}

void
rdmaWaitReadyToDisconnect(IbvConnection *conn)
{
    while (conn->rdma_state != STATE_READY) {
        // nop.
    }
}

void
rdmaOnCompletionServer(struct ibv_wc *wc)
{
    on_completion(wc, true);
}

void
rdmaOnCompletionClient(struct ibv_wc *wc)
{
    on_completion(wc, false);
}
void
rdmaSetOnCompletionHandler(void (*handler)(struct ibv_wc *))
{
    OnCompletionHandler = handler;
}


void
rdmaSendMr(IbvConnection *conn)
{
    int i;
    struct ibv_mr **sp = conn->rdma_remote_mr;
    struct message *dp = conn->send_msg;

    for (i = 0; i < RC_NWR_PER_POST; i++) {
        memcpy(conn->send_msg->mr + i, conn->rdma_remote_mr[i], sizeof(struct ibv_mr));
        WARN(4,
             "copy\n"
             "  rdma_remote_mr[%d]  adr:%lx  rkey:%x  lkey:%x ->\n"
             "    send_msg->mr[%d]  adr:%lx  rkey:%x  lkey:%x\n",
             i, sp[i]->addr, sp[i]->rkey, sp[i]->lkey,
             i, dp->mr[i].addr, dp->mr[i].rkey, dp->mr[i].lkey);

    }

    post_send_msg(conn);
}


void
rdmaWaitEvent(struct rdma_event_channel *ec, rdma_cm_event_type et, int (*handler)(struct rdma_cm_id *id))
{
    struct rdma_cm_event *ep = NULL;
    while (rdma_get_cm_event(ec, &ep) == 0) {
        struct rdma_cm_event event_copy;

        memcpy(&event_copy, ep, sizeof(*ep));
        rdma_ack_cm_event(ep);
        if (et == event_copy.event) {
            if (handler) {
                handler(event_copy.id);
            }
            break;
        }
        else {
            WARN(0, "got an event %s, not what expected (%s). "
                    "event.status:%d\n",
                    rdma_event_str(event_copy.event), rdma_event_str(et),
                    event_copy.status);
            switch (event_copy.event) {
              case RDMA_CM_EVENT_REJECTED:
                WARN(0, "RDMA connection rejected. The peer may be down.\n");
                break;
              default:
                break;
            }
            // 
            // infiniband/cm.h ib_cm_rej_reason
            // IB_CM_REJ_INVALID_SERVICE_ID		= 8,

            exit(EXIT_FAILURE);
        }
    }    
}

void
rdmaKickoff(IbvConnection *conn, int length)
{
    uintptr_t at;
    int offset = sizeof(RCMethod);

    pthread_mutex_lock(&RdmaMutex);
    conn->rdma_state = STATE_BUSY;
    kickoff_rdma_with_offset(offset, conn, length - offset);
    kickoff_rdma_with_offset(0, conn, offset);
    pthread_mutex_unlock(&RdmaMutex);
}

void
rdmaPipelinedKickoff(IbvConnection *conn, int length,
                     char *payload_buf, char *payload_src, int payload_size)
{
    const int size_d = 200000; // pipelined size

    int size = size_d;
    int offset = sizeof(RCMethod); 
    int hdrsize = length - payload_size;
    char *pbuf = payload_buf;
    char *psrc = payload_src;
    int psize = (offset + size_d) - hdrsize;

    pthread_mutex_lock(&RdmaMutex);
    conn->rdma_nreq_pending = 0;
    conn->rdma_state = STATE_BUSY;
    for (int i = 0; offset < length; i++) {
        if (length - offset < size_d) {
            size = length - offset;
            psize = payload_size;
        }
        if (0 < i) {
            psize = size;
        }
        memcpy(pbuf, psrc, psize);
        pbuf += psize;
        psrc += psize;

        kickoff_rdma_with_offset(offset, conn, size);
        offset += size_d;
    }
    kickoff_rdma_with_offset(0, conn, sizeof(RCMethod));
    pthread_mutex_unlock(&RdmaMutex);
}
