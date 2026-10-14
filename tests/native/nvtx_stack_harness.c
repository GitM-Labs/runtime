/* Drives cupti_core.c's NVTX range stack and node naming through on_callback,
 * with the CUPTI/CUDA entry points it links against stubbed out. Built and run
 * by tests/test_graph_identity_emitters.py when a C compiler and CUDA headers
 * are available. Exit status 0 = all checks passed. */
#ifdef _WIN32 /* the harness also runs on Windows dev boxes; buffers are not exercised */
#include <malloc.h>
#include <stddef.h>
static int posix_memalign(void **p, size_t align, size_t n) {
    *p = _aligned_malloc(n, align);
    return *p ? 0 : 12;
}
#endif
#include "cupti_core.c"

#include <stdio.h>

/* ---- stubs ---- */
CUptiResult CUPTIAPI cuptiActivityRegisterCallbacks(CUpti_BuffersCallbackRequestFunc a,
                                                    CUpti_BuffersCallbackCompleteFunc b) {
    (void)a; (void)b; return CUPTI_SUCCESS; }
CUptiResult CUPTIAPI cuptiActivityEnable(CUpti_ActivityKind k) { (void)k; return CUPTI_SUCCESS; }
CUptiResult CUPTIAPI cuptiActivityDisable(CUpti_ActivityKind k) { (void)k; return CUPTI_SUCCESS; }
CUptiResult CUPTIAPI cuptiActivityFlushAll(uint32_t f) { (void)f; return CUPTI_SUCCESS; }
CUptiResult CUPTIAPI cuptiActivityFlushPeriod(uint32_t t) { (void)t; return CUPTI_SUCCESS; }
CUptiResult CUPTIAPI cuptiActivityGetNextRecord(uint8_t *b, size_t s, CUpti_Activity **r) {
    (void)b; (void)s; (void)r; return CUPTI_ERROR_MAX_LIMIT_REACHED; }
CUptiResult CUPTIAPI cuptiGetTimestamp(uint64_t *t) { *t = 0; return CUPTI_SUCCESS; }
CUptiResult CUPTIAPI cuptiGetResultString(CUptiResult r, const char **s) { (void)r; *s = "x"; return CUPTI_SUCCESS; }
cudaError_t CUDARTAPI cudaGetDeviceCount(int *n) { *n = 0; return cudaSuccess; }
CUptiResult CUPTIAPI cuptiSubscribe(CUpti_SubscriberHandle *s, CUpti_CallbackFunc f, void *u) {
    (void)f; (void)u; *s = (CUpti_SubscriberHandle)1; return CUPTI_SUCCESS; }
CUptiResult CUPTIAPI cuptiUnsubscribe(CUpti_SubscriberHandle s) { (void)s; return CUPTI_SUCCESS; }
CUptiResult CUPTIAPI cuptiEnableDomain(uint32_t e, CUpti_SubscriberHandle s, CUpti_CallbackDomain d) {
    (void)e; (void)s; (void)d; return CUPTI_SUCCESS; }
CUptiResult CUPTIAPI cuptiEnableCallback(uint32_t e, CUpti_SubscriberHandle s,
                                         CUpti_CallbackDomain d, CUpti_CallbackId c) {
    (void)e; (void)s; (void)d; (void)c; return CUPTI_SUCCESS; }
CUptiResult CUPTIAPI cuptiGetGraphNodeId(CUgraphNode n, uint64_t *id) {
    *id = (uint64_t)(uintptr_t)n; return CUPTI_SUCCESS; }

/* ---- drivers ---- */
static void nvtx(CUpti_CallbackId cbid, const void *params, const void *ret) {
    CUpti_NvtxData d = {"f", params, ret};
    on_callback(NULL, CUPTI_CB_DOMAIN_NVTX, cbid, &d);
}
static void push_a(const char *m) {
    nvtxRangePushA_params p = {m};
    nvtx(CUPTI_CBID_NVTX_nvtxRangePushA, &p, NULL);
}
static void push_domain(uintptr_t dom, nvtxEventAttributes_t *a) {
    nvtxDomainRangePushEx_params p = {(nvtxDomainHandle_t)dom, {a}};
    nvtx(CUPTI_CBID_NVTX_nvtxDomainRangePushEx, &p, NULL);
}
static void pop_default(void) { nvtxRangePop_params p = {0}; nvtx(CUPTI_CBID_NVTX_nvtxRangePop, &p, NULL); }
static void pop_domain(uintptr_t dom) {
    nvtxDomainRangePop_params p = {(nvtxDomainHandle_t)dom};
    nvtx(CUPTI_CBID_NVTX_nvtxDomainRangePop, &p, NULL);
}

static char g_last[GITM_NAME_MAX + 1];
static void sink(const gitm_record *r, void *u) { (void)u; strcpy(g_last, r->name); }
static const char *create_node(uintptr_t id) {
    CUpti_GraphData g;
    memset(&g, 0, sizeof g);
    g.node = (CUgraphNode)id;
    g.nodeType = CU_GRAPH_NODE_TYPE_KERNEL;
    CUpti_ResourceData rd;
    memset(&rd, 0, sizeof rd);
    rd.resourceDescriptor = &g;
    g_last[0] = '\x01';
    on_callback(NULL, CUPTI_CB_DOMAIN_RESOURCE, CUPTI_CBID_RESOURCE_GRAPHNODE_CREATED, &rd);
    return g_last;
}

static int failures = 0;
#define CHECK(cond, msg) do { if (!(cond)) { printf("FAIL: %s\n", msg); failures++; } } while (0)

int main(void) {
    gitm_set_sink(sink, NULL);
    node_map_start();
    const uintptr_t OTHER = 0x5000;
    nvtxEventAttributes_t a;
    memset(&a, 0, sizeof a);
    a.messageType = NVTX_MESSAGE_TYPE_ASCII;
    a.message.ascii = "L1/mlp_down";

    /* cross-domain pop order: the default pop must remove the default range */
    push_a("L0/qkv_proj");
    push_domain(OTHER, &a);
    pop_default();
    CHECK(strcmp(create_node(0x10), "L1/mlp_down") == 0, "default pop removed the other domain");
    pop_domain(OTHER);
    CHECK(strcmp(create_node(0x11), "") == 0, "stack not empty after both pops");

    /* registered string */
    nvtxDomainRegisterStringA_params rp = {(nvtxDomainHandle_t)OTHER, "L7/attn_out_proj"};
    nvtxStringHandle_t h = (nvtxStringHandle_t)(uintptr_t)0xABC0;
    nvtx(CUPTI_CBID_NVTX_nvtxDomainRegisterStringA, &rp, &h);
    nvtxEventAttributes_t r;
    memset(&r, 0, sizeof r);
    r.messageType = NVTX_MESSAGE_TYPE_REGISTERED;
    r.message.registered = h;
    push_domain(OTHER, &r);
    CHECK(strcmp(create_node(0x12), "L7/attn_out_proj") == 0, "registered string not resolved");
    pop_domain(OTHER);

    /* overflow: while a dropped push is open the innermost range is unknown */
    for (int i = 0; i < RANGE_STACK_MAX; i++) push_a("L2/qkv_proj");
    push_a("L3/mlp_down"); /* dropped */
    CHECK(strcmp(create_node(0x20), "") == 0, "named a node while a dropped range was open");
    pop_default(); /* closes the dropped one, not a stored one */
    CHECK(strcmp(create_node(0x21), "L2/qkv_proj") == 0, "pop closed a stored range first");
    CHECK(tls_nvtx.depth == RANGE_STACK_MAX, "stored range removed by the dropped push's pop");
    for (int i = 0; i < RANGE_STACK_MAX; i++) pop_default();
    CHECK(strcmp(create_node(0x22), "") == 0, "stack not empty after unwinding");

    /* repeat registration of one handle keeps one entry */
    for (int i = 0; i < 1000; i++) nvtx(CUPTI_CBID_NVTX_nvtxDomainRegisterStringA, &rp, &h);
    int entries = 0;
    for (int b = 0; b < REG_BUCKETS; b++)
        for (reg_node *n = g_reg[b]; n; n = n->next) entries += n->handle == (uintptr_t)h;
    CHECK(entries == 1, "repeat registrations grew the table");

    /* a new session starts empty */
    push_a("L9/stale");
    node_map_stop();
    node_map_start();
    CHECK(strcmp(create_node(0x13), "") == 0, "range from an earlier session survived");

    printf(failures ? "%d failure(s)\n" : "ok\n", failures);
    return failures != 0;
}
