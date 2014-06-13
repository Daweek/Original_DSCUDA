#include "ibv_rdma.h"

static void (*OnCompletionHandler)(struct ibv_wc *) = NULL;

static void build_verbs(IbvConnection *conn, struct ibv_context *verbs);
static void build_qp_attr(IbvConnection *conn, struct ibv_qp_init_attr *qp_attr);
static void *poll_cq(void *);
static void post_receives(IbvConnection *conn);
static void register_memory(IbvConnection *conn);

static void send_done(IbvConnection *conn);
static void send_message(IbvConnection *conn);

static void on_completion(struct ibv_wc *wc, bool is_server);

void build_connection(struct rdma_cm_id *id)
{
    IbvConnection *conn;
    struct ibv_qp_init_attr qp_attr;

    id->context = conn = (IbvConnection *)malloc(sizeof(IbvConnection));

    build_verbs(conn, id->verbs);
    build_qp_attr(conn, &qp_attr);

    TEST_NZ(rdma_create_qp(id, conn->pd, &qp_attr));

    conn->id = id;
    conn->qp = id->qp;

    conn->connected = 0;

    register_memory(conn);
    post_receives(conn);
}

static void build_verbs(IbvConnection *conn, struct ibv_context *verbs)
{
    conn->ibvctx = verbs;
    TEST_Z(conn->pd = ibv_alloc_pd(conn->ibvctx));
    TEST_Z(conn->comp_channel = ibv_create_comp_channel(conn->ibvctx));
    TEST_Z(conn->cq = ibv_create_cq(conn->ibvctx, 10, NULL, conn->comp_channel, 0)); /* cqe=10 is arbitrary */
    TEST_NZ(ibv_req_notify_cq(conn->cq, 0));

    TEST_NZ(pthread_create(&conn->cq_poller_thread, NULL, poll_cq, conn));
}


void build_params(struct rdma_conn_param *params)
{
    memset(params, 0, sizeof(*params));

    params->initiator_depth = params->responder_resources = 1;
    params->rnr_retry_count = 7; /* infinite retry */
}

static void
build_qp_attr(IbvConnection *conn, struct ibv_qp_init_attr *qp_attr)
{
    memset(qp_attr, 0, sizeof(*qp_attr));

    qp_attr->send_cq = conn->cq;
    qp_attr->recv_cq = conn->cq;
    qp_attr->qp_type = IBV_QPT_RC;

    qp_attr->cap.max_send_wr = 10;
    qp_attr->cap.max_recv_wr = 10;
    qp_attr->cap.max_send_sge = 1;
    qp_attr->cap.max_recv_sge = 1;
}

void
destroy_connection(IbvConnection *conn)
{
    rdma_destroy_qp(conn->id);

    ibv_dereg_mr(conn->send_mr);
    ibv_dereg_mr(conn->recv_mr);
    ibv_dereg_mr(conn->rdma_local_mr);
    ibv_dereg_mr(conn->rdma_remote_mr);

    free(conn->send_msg);
    free(conn->recv_msg);
    free(conn->rdma_local_region);
    free(conn->rdma_remote_region);

    rdma_destroy_id(conn->id);

    free(conn);
}

char *
msgtype2str(int type) {
    static char buf[128];
    switch (type) {
      case MSG_MR:
        sprintf(buf, "%s", "MSG_MR");
        break;
      case MSG_DONE:
        sprintf(buf, "%s", "MSG_DONE");
        break;
      default:
        fprintf(stderr, "unknown type.\n");
        exit(EXIT_FAILURE);
    }
    return buf;
}

void wait_ready_to_rdma(IbvConnection *conn)
{
    while (conn->rdma_state != STATE_READY) {
        // nop.
    }
}

void wait_ready_to_disconnect(IbvConnection *conn)
{
    while (conn->rdma_state != STATE_READY) {
        // nop.
    }
}


static void check_msgsanity(struct ibv_wc *wc)
{
    return; // !!!

    IbvConnection *conn = (IbvConnection *)(uintptr_t)wc->wr_id;
    int exit_on_err = false;

    switch (wc->opcode) {
      case IBV_WC_RECV:
        fprintf(stderr, "completion RECV %s", msgtype2str(conn->recv_msg->type));
        exit_on_err = true;
        break;
      case IBV_WC_SEND:
        fprintf(stderr, "completion SEND %s", msgtype2str(conn->send_msg->type));
        exit_on_err = true;
        break;
      case IBV_WC_RDMA_WRITE:
        fprintf(stderr, "completion RDMA_WRITE");
        break;
      case IBV_WC_RDMA_READ:
        fprintf(stderr, "completion RDMA_READ");
        break;
      default:
        fprintf(stderr, "completion of unknown type : 0x%x", wc->opcode);
        exit(1);
    }

    if (wc->status != IBV_WC_SUCCESS) {
        fprintf(stderr, " NG\n");
        if (exit_on_err) {
            exit(1);
        }
    }
    else {
        fprintf(stderr, " OK\n");
    }
}


static void
print_rdma_state(enum rdma_state_t state)
{
    fprintf(stderr, "rdma state : ");
    switch (state) {
      case STATE_INIT:
        fprintf(stderr, "INIT");
        break;
      case STATE_READY:
        fprintf(stderr, "READY");
        break;
      case STATE_BUSY0:
        fprintf(stderr, "BUSY0");
        break;
      case STATE_BUSY1:
        fprintf(stderr, "BUSY1");
        break;
      default:
        fprintf(stderr, "unknown");
    }
    fprintf(stderr, "\n");
}

void on_completion_server(struct ibv_wc *wc)
{
    on_completion(wc, true);
}

void on_completion_client(struct ibv_wc *wc)
{
    on_completion(wc, false);
}

static void
on_completion(struct ibv_wc *wc, bool is_server)
{
    IbvConnection *conn = (IbvConnection *)(uintptr_t)wc->wr_id;

    check_msgsanity(wc);

    switch (wc->opcode) {
      case IBV_WC_RECV:
        switch (conn->recv_msg->type) {
          case MSG_MR:
            memcpy(&conn->peer_mr, &conn->recv_msg->data.mr, sizeof(conn->peer_mr));
            // server receives MR before sending ours, so send ours back.
            if (is_server) {
                send_mr(conn);
            }
            conn->rdma_state = STATE_READY;
            break;
          case MSG_DONE:
            break;
          default:
            fprintf(stderr, "received a message of unknown type.\n");
            exit(EXIT_FAILURE);
        }
        post_receives(conn); // rearm for next message.
        break;

      case IBV_WC_SEND:
        // nop.
        break;

      case IBV_WC_RDMA_WRITE:
        switch (conn->rdma_state) {
          case STATE_READY:
            break;
          case STATE_BUSY0:
            conn->rdma_state = STATE_BUSY1;
            break;
          case STATE_BUSY1:
            conn->rdma_state = STATE_READY;
            break;
          default:
            fprintf(stderr, "invalid conn->rdma_state:%d\n", conn->rdma_state);
            exit(1);
        }
        break;

      default:
        fprintf(stderr, "completion with unexpected opcode : 0x%x\n", wc->opcode);
        exit(1);
    }
}

void
set_on_completion_handler(void (*handler)(struct ibv_wc *))
{
    OnCompletionHandler = handler;
}

void *
poll_cq(void *ctx)
{
    struct ibv_cq *cq;
    struct ibv_wc wc;
    IbvConnection *conn = (IbvConnection *)ctx;

    while (1) {
        TEST_NZ(ibv_get_cq_event(conn->comp_channel, &cq, &ctx));
        ibv_ack_cq_events(cq, 1);
        TEST_NZ(ibv_req_notify_cq(cq, 0));

        while (ibv_poll_cq(cq, 1, &wc)) {
            (OnCompletionHandler)(&wc);
        }
    }

    return NULL;
}

void post_receives(IbvConnection *conn)
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

    TEST_NZ(ibv_post_recv(conn->qp, &wr, &bad_wr));
}

void register_memory(IbvConnection *conn)
{
    conn->send_msg = (message *)malloc(sizeof(struct message));
    conn->recv_msg = (message *)malloc(sizeof(struct message));

    conn->rdma_local_region = (char *)calloc(RDMA_BUFFER_SIZE, 1);
    conn->rdma_remote_region = (char *)calloc(RDMA_BUFFER_SIZE, 1);

    TEST_Z(conn->send_mr = ibv_reg_mr(
                                      conn->pd, 
                                      conn->send_msg, 
                                      sizeof(struct message), 
                                      IBV_ACCESS_LOCAL_WRITE));

    TEST_Z(conn->recv_mr = ibv_reg_mr(
                                      conn->pd, 
                                      conn->recv_msg, 
                                      sizeof(struct message), 
                                      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE ));

    TEST_Z(conn->rdma_local_mr = ibv_reg_mr(
                                            conn->pd, 
                                            conn->rdma_local_region, 
                                            RDMA_BUFFER_SIZE, 
                                            IBV_ACCESS_LOCAL_WRITE));

    TEST_Z(conn->rdma_remote_mr = ibv_reg_mr(
                                             conn->pd, 
                                             conn->rdma_remote_region, 
                                             RDMA_BUFFER_SIZE, 
                                             IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE));
}

void
send_mr(IbvConnection *conn)
{
    conn->send_msg->type = MSG_MR;
    memcpy(&conn->send_msg->data.mr, conn->rdma_remote_mr, sizeof(struct ibv_mr));
    send_message(conn);
}

static void
send_done(IbvConnection *conn)
{
    conn->send_msg->type = MSG_DONE;
    send_message(conn);
}

static void
send_message(IbvConnection *conn)
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

    TEST_NZ(ibv_post_send(conn->qp, &wr, &bad_wr));
    WARN(3, "sent %s.\n", msgtype2str(conn->send_msg->type));
}

void
wait_event(struct rdma_event_channel *ec, int et, int (*handler)(struct rdma_cm_id *id))
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
            fprintf(stderr, "got an event (=%d) not what expected (=%d). "
                    "event.status:%d\n",
                    event_copy.event, et, event_copy.status);
            // 
            // infiniband/cm.h ib_cm_rej_reason
            // IB_CM_REJ_INVALID_SERVICE_ID		= 8,

            exit(EXIT_FAILURE);
        }
    }    
}

static void
kickoff_rdma_with_offset(uintptr_t offset, IbvConnection *conn, int length)
{
    struct ibv_send_wr wr, *bad_wr = NULL;
    struct ibv_sge sge;

    memset(&wr, 0, sizeof(wr));

    wr.wr_id = (uintptr_t)conn;
    wr.opcode = IBV_WR_RDMA_WRITE;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.send_flags = IBV_SEND_SIGNALED;
    wr.wr.rdma.remote_addr = (uintptr_t)conn->peer_mr.addr + offset;
    wr.wr.rdma.rkey = conn->peer_mr.rkey;

    sge.addr = (uintptr_t)conn->rdma_local_region + offset;
    sge.length = length;
    sge.lkey = conn->rdma_local_mr->lkey;

    TEST_NZ(ibv_post_send(conn->qp, &wr, &bad_wr));
}

void
kickoff_rdma(IbvConnection *conn, int length)
{
    uintptr_t at;
    conn->rdma_state = STATE_BUSY0;
    int offset = sizeof(RCMethod);
    kickoff_rdma_with_offset(offset, conn, length - offset);
    kickoff_rdma_with_offset(0, conn, offset);
}

/*
 *
 * t0 : time of day (in second) the last time this function is called.
 * returns the number of seconds passed since *t0.
 */
double
RCgetCputime(double *t0)
{
    struct timeval t;
    double tnow, dt;

    gettimeofday(&t, NULL);
    tnow = t.tv_sec + t.tv_usec/1000000.0;
    dt = tnow - *t0;
    *t0 = tnow;
    return dt;
}
