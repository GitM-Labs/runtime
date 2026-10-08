/*
 * cupti_core — see cupti_core.h.
 *
 * Struct versions are pinned to CUDA 12.x/13.x (Kernel9 / Memcpy5 / Sync). If the
 * deployed CUPTI drops a versioned struct name the compile fails loudly — bump the
 * version in the cast, never guess offsets.
 */

#include "cupti_core.h"

#include <cuda_runtime.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

/* The capture-time node map needs the NVTX callback parameter structs. Built
 * without the NVTX headers, the collector still works; replayed kernels then
 * keep range_op=None, as before the map existed. */
#if defined(__has_include)
#if __has_include(<nvtx3/nvToolsExt.h>) && __has_include(<nvtx3/nvToolsExtSync.h>) && \
    __has_include(<generated_nvtx_meta.h>)
#include <nvtx3/nvToolsExt.h>
#include <nvtx3/nvToolsExtSync.h>
#include <generated_nvtx_meta.h>
#define GITM_HAVE_NODE_MAP 1
#endif
#endif

#define BUF_SIZE (32 * 1024 * 1024)   /* 32 MiB activity buffers */
#define BUF_ALIGN 32

static gitm_sink_fn   g_sink = NULL;
static void          *g_sink_user = NULL;
static gitm_buffer_fn g_buffer_hook = NULL;
static void          *g_buffer_user = NULL;
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
static int g_enabled = 0;

/* Enable CONCURRENT_KERNEL only, not also CUPTI_ACTIVITY_KIND_KERNEL. Enabling
 * both yields two records per kernel, and the duplicate set comes back with
 * zeroed timestamps (verified on an A100 / CUDA 13). CONCURRENT_KERNEL is the
 * correct kind for async workloads and carries valid start/end. */
static const CUpti_ActivityKind ENABLED_KINDS[] = {
    CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL,
    CUPTI_ACTIVITY_KIND_MEMCPY,
    CUPTI_ACTIVITY_KIND_SYNCHRONIZATION,
};
#define N_ENABLED_KINDS (sizeof(ENABLED_KINDS) / sizeof(ENABLED_KINDS[0]))

/* Opt-in via GITM_TRACE_NVTX, because the cost is real and is not proportional
 * to what it buys on an uninstrumented run.
 *
 * RUNTIME emits one record per CUDA API call — on a decode step that is a
 * multiple of the kernel count, not a fraction of it — and MARKER adds two per
 * NVTX range. Together they roughly triple buffer pressure and add host-side
 * interception to every launch. That is the overhead the with/without-NVTX
 * throughput comparison exists to measure.
 *
 * What they buy: an anonymous `nvjet_sm90_tst_*` GEMM becomes resolvable to a
 * layer and an op. Its name carries no projection, so name matching cannot
 * recover that identity and no amount of vocabulary work will. The chain is
 * kernel.correlation_id -> RUNTIME record -> enclosing MARKER range. */
static const CUpti_ActivityKind NVTX_KINDS[] = {
    CUPTI_ACTIVITY_KIND_RUNTIME,
    /* DRIVER as well as RUNTIME, and this is not redundancy. cuBLAS, cuBLASLt
     * and CUTLASS launch through the *driver* API (cuLaunchKernel), not the
     * runtime one, so their correlation records arrive under this kind. With
     * only RUNTIME enabled, a B200 smoke test resolved the elementwise kernels
     * inside an NVTX range and left every GEMM unattributed — which is exactly
     * backwards, since an `nvjet_*` GEMM carries no projection in its name and
     * is the kernel correlation exists to identify. */
    CUPTI_ACTIVITY_KIND_DRIVER,
    CUPTI_ACTIVITY_KIND_MARKER,
};
#define N_NVTX_KINDS (sizeof(NVTX_KINDS) / sizeof(NVTX_KINDS[0]))

int gitm_nvtx_enabled(void) {
    const char *v = getenv("GITM_TRACE_NVTX");
    return v && *v && strcmp(v, "0") != 0;
}

void gitm_set_sink(gitm_sink_fn sink, void *user) {
    pthread_mutex_lock(&g_lock);
    g_sink = sink;
    g_sink_user = user;
    pthread_mutex_unlock(&g_lock);
}

void gitm_set_buffer_hook(gitm_buffer_fn hook, void *user) {
    pthread_mutex_lock(&g_lock);
    g_buffer_hook = hook;
    g_buffer_user = user;
    pthread_mutex_unlock(&g_lock);
}

static void copy_name(gitm_record *r, const char *name) {
    if (!name) { r->name[0] = '\0'; return; }
    strncpy(r->name, name, GITM_NAME_MAX);
    r->name[GITM_NAME_MAX] = '\0';
}

/* Decode one CUPTI record into a normalized gitm_record and hand it to the sink.
 * Called with g_lock held. */
static void ingest(CUpti_Activity *rec) {
    gitm_record r;
    memset(&r, 0, sizeof(r));

    switch (rec->kind) {
        case CUPTI_ACTIVITY_KIND_KERNEL:
        case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
            CUpti_ActivityKernel9 *k = (CUpti_ActivityKernel9 *)rec;
            r.kind = GITM_REC_KERNEL;
            copy_name(&r, k->name);
            r.start_ns = k->start;
            r.end_ns = k->end;
            r.device_id = k->deviceId;
            r.context_id = k->contextId;
            r.stream_id = k->streamId;
            r.correlation_id = k->correlationId;
            r.grid[0] = k->gridX; r.grid[1] = k->gridY; r.grid[2] = k->gridZ;
            r.block[0] = k->blockX; r.block[1] = k->blockY; r.block[2] = k->blockZ;
            r.static_shared_mem = k->staticSharedMemory;
            r.dynamic_shared_mem = k->dynamicSharedMemory;
            r.registers_per_thread = k->registersPerThread;
            r.graph_id = k->graphId;
            r.graph_node_id = k->graphNodeId;
            break;
        }
        case CUPTI_ACTIVITY_KIND_MEMCPY: {
            CUpti_ActivityMemcpy5 *m = (CUpti_ActivityMemcpy5 *)rec;
            r.kind = GITM_REC_MEMCPY;
            r.start_ns = m->start;
            r.end_ns = m->end;
            r.device_id = m->deviceId;
            r.context_id = m->contextId;
            r.stream_id = m->streamId;
            r.correlation_id = m->correlationId;
            r.copy_kind = m->copyKind;
            r.bytes = m->bytes;
            break;
        }
        case CUPTI_ACTIVITY_KIND_SYNCHRONIZATION: {
            CUpti_ActivitySynchronization *s = (CUpti_ActivitySynchronization *)rec;
            r.kind = GITM_REC_SYNC;
            r.start_ns = s->start;
            r.end_ns = s->end;
            r.context_id = s->contextId;
            r.stream_id = s->streamId;
            r.correlation_id = s->correlationId;
            r.sync_type = s->type;
            break;
        }
        case CUPTI_ACTIVITY_KIND_RUNTIME:
        case CUPTI_ACTIVITY_KIND_DRIVER: {
            /* The host-side CUDA API call. Both kinds decode through
             * CUpti_ActivityAPI — the struct is shared — and both are needed:
             * the runtime API covers torch's own launches, the driver API covers
             * cuBLAS/cuBLASLt/CUTLASS. Its correlation_id is the same one the
             * kernel record carries, and its thread_id is what makes range
             * containment safe — a launch on one thread must never be attributed
             * to a range pushed on another. */
            CUpti_ActivityAPI *a = (CUpti_ActivityAPI *)rec;
            r.kind = GITM_REC_RUNTIME;
            r.start_ns = a->start;
            r.end_ns = a->end;
            r.correlation_id = a->correlationId;
            r.thread_id = a->threadId;
            break;
        }
        case CUPTI_ACTIVITY_KIND_MARKER: {
            /* One NVTX push or pop, not a range: CUpti_ActivityMarker2 carries a
             * single `timestamp`, and the header states the name "will be NULL
             * for an end marker". Pairing the two into a range needs state, so
             * the C side stays stateless and emits both; _cupti_decode.py joins
             * them on marker_id and takes the name from the start.
             *
             * flags is a bitfield (START = 1<<1, END = 1<<2) and the bits can
             * combine with the SYNC_* flags, so it must be masked, never
             * compared for equality.
             *
             * thread_id lives in a union whose valid member is selected by
             * objectKind. Reading `pt` under any other kind would return a
             * device or context id silently typed as a thread. */
            CUpti_ActivityMarker2 *m = (CUpti_ActivityMarker2 *)rec;
            r.kind = GITM_REC_MARKER;
            r.start_ns = m->timestamp;
            r.end_ns = m->timestamp;
            r.marker_id = m->id;
            if (m->flags & CUPTI_ACTIVITY_FLAG_MARKER_END) {
                r.marker_flags = GITM_MARKER_END;
            } else {
                r.marker_flags = GITM_MARKER_START;
            }
            if (m->objectKind == CUPTI_ACTIVITY_OBJECT_THREAD) {
                r.thread_id = m->objectId.pt.threadId;
            }
            copy_name(&r, m->name);  /* NULL on an end marker; copy_name handles it */
            break;
        }
        default:
            return;  /* kinds GITM doesn't model */
    }

    if (g_sink) g_sink(&r, g_sink_user);
}

const char *gitm_node_kind_name(int kind) {
    switch (kind) {
        case GITM_NODE_KERNEL: return "kernel";
        case GITM_NODE_MEMCPY: return "memcpy";
        case GITM_NODE_MEMSET: return "memset";
        default:               return "other";
    }
}

#ifdef GITM_HAVE_NODE_MAP
/* ---- capture-time node map ("the Nsight approach", kernel_identity.md) ----
 *
 * MARKER activity arrives in buffers, too late to say which range was open
 * when a graph node was created. NVTX callbacks are synchronous on the pushing
 * thread, so a per-thread stack of names is current when the RESOURCE callback
 * reports a node created by stream capture on that thread. GRAPHNODE_CREATED
 * also fires for the copies cudaGraphInstantiate makes; those are skipped (as
 * NVIDIA's cuda_graphs_trace sample does), or the range open at instantiate
 * time would rename the captured nodes. */

#define RANGE_STACK_MAX 128

static CUpti_SubscriberHandle g_sub;
static int g_subscribed = 0;
/* Bumped on every node_map_start. Pops that happen while callbacks are off
 * are never seen, so a thread's stack from an earlier session would name the
 * next session's nodes; a thread resets the first time it is seen in a new one. */
static atomic_uint g_session = 0;

/* One stack per thread holding every domain's open ranges in push order; each
 * entry remembers its domain. Domains nest independently, so a pop removes the
 * innermost range of *its* domain, which need not be the top. A node is named
 * after the innermost range open in any domain. */
static __thread struct {
    struct {
        uintptr_t domain; /* 0: the default domain */
        char *name;
    } open[RANGE_STACK_MAX];
    int depth;
    int lost; /* pushes past RANGE_STACK_MAX; their pops consume this */
    int instantiating;
    unsigned session;
} tls_nvtx;

static void tls_sync(void) {
    unsigned now = atomic_load(&g_session);
    if (tls_nvtx.session == now) return;
    for (int d = 0; d < tls_nvtx.depth; d++) free(tls_nvtx.open[d].name);
    memset(&tls_nvtx, 0, sizeof tls_nvtx);
    tls_nvtx.session = now;
}

static void nvtx_push(uintptr_t domain, const char *msg) {
    if (tls_nvtx.depth >= RANGE_STACK_MAX) {
        tls_nvtx.lost++;
        return;
    }
    size_t n = msg ? strnlen(msg, GITM_NAME_MAX) : 0;
    char *copy = malloc(n + 1);
    if (copy) {
        if (n) memcpy(copy, msg, n);
        copy[n] = '\0';
    }
    tls_nvtx.open[tls_nvtx.depth].domain = domain;
    tls_nvtx.open[tls_nvtx.depth].name = copy;
    tls_nvtx.depth++;
}

static void nvtx_pop(uintptr_t domain) {
    for (int d = tls_nvtx.depth - 1; d >= 0; d--) {
        if (tls_nvtx.open[d].domain != domain) continue;
        free(tls_nvtx.open[d].name);
        memmove(&tls_nvtx.open[d], &tls_nvtx.open[d + 1],
                (size_t)(tls_nvtx.depth - d - 1) * sizeof tls_nvtx.open[0]);
        tls_nvtx.depth--;
        return;
    }
    if (tls_nvtx.lost > 0) tls_nvtx.lost--; /* the pop of an untracked push */
}

static const char *nvtx_top(void) {
    int d = tls_nvtx.depth;
    return d > 0 && tls_nvtx.open[d - 1].name ? tls_nvtx.open[d - 1].name : "";
}

/* nvtxDomainRegisterStringA handle -> copy of its string, so ranges pushed with
 * a registered message keep their name. Registration is rare; one lock. */
#define REG_BUCKETS 256
typedef struct reg_node {
    uintptr_t handle;
    struct reg_node *next;
    char name[];
} reg_node;
static reg_node *g_reg[REG_BUCKETS];
static pthread_mutex_t g_reg_lock = PTHREAD_MUTEX_INITIALIZER;

static void reg_put(uintptr_t handle, const char *str) {
    size_t n = str ? strnlen(str, GITM_NAME_MAX) : 0;
    reg_node *r = malloc(sizeof(reg_node) + n + 1);
    if (!r) return;
    r->handle = handle;
    if (n) memcpy(r->name, str, n);
    r->name[n] = '\0';
    pthread_mutex_lock(&g_reg_lock);
    r->next = g_reg[(handle >> 4) % REG_BUCKETS];
    g_reg[(handle >> 4) % REG_BUCKETS] = r;
    pthread_mutex_unlock(&g_reg_lock);
}

/* Registered strings live for the process, so the returned pointer stays valid. */
static const char *reg_get(uintptr_t handle) {
    const char *out = NULL;
    pthread_mutex_lock(&g_reg_lock);
    for (reg_node *r = g_reg[(handle >> 4) % REG_BUCKETS]; r; r = r->next)
        if (r->handle == handle) {
            out = r->name;
            break;
        }
    pthread_mutex_unlock(&g_reg_lock);
    return out;
}

static const char *attr_message(const nvtxEventAttributes_t *a) {
    if (!a) return NULL;
    if (a->messageType == NVTX_MESSAGE_TYPE_ASCII) return a->message.ascii;
    if (a->messageType == NVTX_MESSAGE_TYPE_REGISTERED)
        return reg_get((uintptr_t)a->message.registered);
    return NULL;
}

static int node_kind(CUgraphNodeType t) {
    switch (t) {
        case CU_GRAPH_NODE_TYPE_KERNEL: return GITM_NODE_KERNEL;
        case CU_GRAPH_NODE_TYPE_MEMCPY: return GITM_NODE_MEMCPY;
        case CU_GRAPH_NODE_TYPE_MEMSET: return GITM_NODE_MEMSET;
        default:                        return GITM_NODE_OTHER;
    }
}

static void emit_node(uint64_t id, uint64_t cloned_from, const char *name, int kind) {
    gitm_record r;
    memset(&r, 0, sizeof r);
    r.kind = GITM_REC_GRAPH_NODE;
    r.graph_node_id = id;
    r.cloned_from = cloned_from;
    r.node_kind = kind;
    copy_name(&r, name);
    pthread_mutex_lock(&g_lock);
    if (g_sink) g_sink(&r, g_sink_user);
    pthread_mutex_unlock(&g_lock);
}

static const CUpti_CallbackId RUNTIME_INSTANTIATE[] = {
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphInstantiate_v10000,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphInstantiateWithFlags_v11040,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphInstantiateWithParams_v12000,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphInstantiateWithParams_ptsz_v12000,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphInstantiate_v12000,
};
static const CUpti_CallbackId DRIVER_INSTANTIATE[] = {
    CUPTI_DRIVER_TRACE_CBID_cuGraphInstantiate,
    CUPTI_DRIVER_TRACE_CBID_cuGraphInstantiate_v2,
    CUPTI_DRIVER_TRACE_CBID_cuGraphInstantiateWithFlags,
    CUPTI_DRIVER_TRACE_CBID_cuGraphInstantiateWithParams,
    CUPTI_DRIVER_TRACE_CBID_cuGraphInstantiateWithParams_ptsz,
};
#define N_OF(a) (sizeof(a) / sizeof((a)[0]))

static void CUPTIAPI on_callback(void *user, CUpti_CallbackDomain dom, CUpti_CallbackId cbid,
                                 const void *cbdata) {
    (void)user;
    tls_sync();
    if (dom == CUPTI_CB_DOMAIN_NVTX) {
        const CUpti_NvtxData *nd = cbdata;
        const void *params = nd->functionParams;
        switch (cbid) {
            case CUPTI_CBID_NVTX_nvtxRangePushA:
                nvtx_push(0, ((const nvtxRangePushA_params *)params)->message);
                break;
            case CUPTI_CBID_NVTX_nvtxRangePushEx:
                nvtx_push(0, attr_message(((const nvtxRangePushEx_params *)params)->eventAttrib));
                break;
            case CUPTI_CBID_NVTX_nvtxRangePushW: /* unnamed, but keeps the stack balanced */
                nvtx_push(0, NULL);
                break;
            case CUPTI_CBID_NVTX_nvtxDomainRangePushEx: {
                const nvtxDomainRangePushEx_params *dp = params;
                nvtx_push((uintptr_t)dp->domain, attr_message(dp->core.eventAttrib));
                break;
            }
            case CUPTI_CBID_NVTX_nvtxRangePop:
                nvtx_pop(0);
                break;
            case CUPTI_CBID_NVTX_nvtxDomainRangePop:
                nvtx_pop((uintptr_t)((const nvtxDomainRangePop_params *)params)->domain);
                break;
            case CUPTI_CBID_NVTX_nvtxDomainRegisterStringA:
                if (nd->functionReturnValue)
                    reg_put((uintptr_t)*(const nvtxStringHandle_t *)nd->functionReturnValue,
                            ((const nvtxDomainRegisterStringA_params *)params)->string);
                break;
            default:
                break;
        }
    } else if (dom == CUPTI_CB_DOMAIN_RUNTIME_API || dom == CUPTI_CB_DOMAIN_DRIVER_API) {
        /* Only the instantiate callbacks are enabled in these domains; runtime
         * instantiate calls driver instantiate, hence a depth, not a flag. */
        if (((const CUpti_CallbackData *)cbdata)->callbackSite == CUPTI_API_ENTER)
            tls_nvtx.instantiating++;
        else if (tls_nvtx.instantiating > 0)
            tls_nvtx.instantiating--;
    } else if (dom == CUPTI_CB_DOMAIN_RESOURCE) {
        const CUpti_GraphData *g = ((const CUpti_ResourceData *)cbdata)->resourceDescriptor;
        uint64_t id = 0, orig = 0;
        if (!g || !g->node || cuptiGetGraphNodeId(g->node, &id) != CUPTI_SUCCESS) return;
        if (cbid != CUPTI_CBID_RESOURCE_GRAPHNODE_CREATED &&
            cbid != CUPTI_CBID_RESOURCE_GRAPHNODE_CLONED)
            return;
        /* Whether a replayed kernel reports the captured node or the copy
         * instantiate/clone made is not documented; a copy that names its
         * original becomes a link, so the decoder resolves either. */
        if (g->originalNode && cuptiGetGraphNodeId(g->originalNode, &orig) == CUPTI_SUCCESS &&
            orig != id) {
            emit_node(id, orig, "", node_kind(g->nodeType));
        } else if (cbid == CUPTI_CBID_RESOURCE_GRAPHNODE_CREATED && !tls_nvtx.instantiating) {
            emit_node(id, 0, nvtx_top(), node_kind(g->nodeType));
        }
    }
}

/* Tolerated like the NVTX activity kinds: another subscriber (Nsight) or an old
 * CUPTI costs the node map, never the trace. */
static void node_map_start(void) {
    if (g_subscribed || cuptiSubscribe(&g_sub, on_callback, NULL) != CUPTI_SUCCESS) return;
    g_subscribed = 1;
    atomic_fetch_add(&g_session, 1);
    cuptiEnableDomain(1, g_sub, CUPTI_CB_DOMAIN_NVTX);
    cuptiEnableCallback(1, g_sub, CUPTI_CB_DOMAIN_RESOURCE, CUPTI_CBID_RESOURCE_GRAPHNODE_CREATED);
    cuptiEnableCallback(1, g_sub, CUPTI_CB_DOMAIN_RESOURCE, CUPTI_CBID_RESOURCE_GRAPHNODE_CLONED);
    for (size_t i = 0; i < N_OF(RUNTIME_INSTANTIATE); i++)
        cuptiEnableCallback(1, g_sub, CUPTI_CB_DOMAIN_RUNTIME_API, RUNTIME_INSTANTIATE[i]);
    for (size_t i = 0; i < N_OF(DRIVER_INSTANTIATE); i++)
        cuptiEnableCallback(1, g_sub, CUPTI_CB_DOMAIN_DRIVER_API, DRIVER_INSTANTIATE[i]);
}

static void node_map_stop(void) {
    if (!g_subscribed) return;
    cuptiUnsubscribe(g_sub);
    g_subscribed = 0;
}
#else
static void node_map_start(void) {}
static void node_map_stop(void) {}
#endif

static void CUPTIAPI buffer_requested(uint8_t **buffer, size_t *size,
                                      size_t *maxNumRecords) {
    void *p = NULL;
    if (posix_memalign(&p, BUF_ALIGN, BUF_SIZE) != 0) p = NULL;
    *buffer = (uint8_t *)p;
    /* Zero on failure, not BUF_SIZE. Reporting a size for a NULL buffer invites
     * CUPTI to write into it; the previous code passed the allocation straight
     * through and would have handed over NULL with a 32 MiB size attached. An
     * allocation failure should cost records, not the traced process. */
    *size = p ? BUF_SIZE : 0;
    *maxNumRecords = 0;  /* fill as many as fit */
}

static void CUPTIAPI buffer_completed(CUcontext ctx, uint32_t streamId,
                                      uint8_t *buffer, size_t size, size_t validSize) {
    (void)ctx; (void)streamId; (void)size;
    CUpti_Activity *record = NULL;

    pthread_mutex_lock(&g_lock);
    if (g_buffer_hook) g_buffer_hook(g_buffer_user);
    if (validSize > 0) {
        for (;;) {
            CUptiResult st = cuptiActivityGetNextRecord(buffer, validSize, &record);
            if (st == CUPTI_SUCCESS) {
                ingest(record);
            } else {
                break;  /* MAX_LIMIT_REACHED = end of buffer; anything else, stop */
            }
        }
    }
    pthread_mutex_unlock(&g_lock);
    free(buffer);  /* posix_memalign's pointer — free() takes it directly */
}

CUptiResult gitm_cupti_start(void) {
    if (g_enabled) return CUPTI_SUCCESS;

    CUptiResult st = cuptiActivityRegisterCallbacks(buffer_requested, buffer_completed);
    if (st != CUPTI_SUCCESS) return st;

    for (size_t i = 0; i < N_ENABLED_KINDS; i++) {
        st = cuptiActivityEnable(ENABLED_KINDS[i]);
        if (st != CUPTI_SUCCESS) return st;
    }
    /* Tolerated, not required. A CUPTI that declines RUNTIME or MARKER still
     * produces a complete kernel trace — correlation is simply unavailable, and
     * every kernel falls back to name classification. Failing the whole capture
     * for it would trade a working trace for no trace. */
    if (gitm_nvtx_enabled()) {
        for (size_t i = 0; i < N_NVTX_KINDS; i++) {
            cuptiActivityEnable(NVTX_KINDS[i]);
        }
        node_map_start();
    }
    g_enabled = 1;
    return CUPTI_SUCCESS;
}

CUptiResult gitm_cupti_flush(void) {
    return cuptiActivityFlushAll(1 /* FORCE */);
}

CUptiResult gitm_cupti_stop(void) {
    if (!g_enabled) return CUPTI_SUCCESS;
    for (size_t i = 0; i < N_ENABLED_KINDS; i++) {
        cuptiActivityDisable(ENABLED_KINDS[i]);
    }
    if (gitm_nvtx_enabled()) {
        for (size_t i = 0; i < N_NVTX_KINDS; i++) {
            cuptiActivityDisable(NVTX_KINDS[i]);
        }
        node_map_stop();
    }
    g_enabled = 0;
    return gitm_cupti_flush();
}

CUptiResult gitm_cupti_set_flush_period(uint32_t ms) {
    return cuptiActivityFlushPeriod(ms);
}

uint64_t gitm_cupti_timestamp(void) {
    uint64_t ts = 0;
    if (cuptiGetTimestamp(&ts) != CUPTI_SUCCESS) return 0;
    return ts;
}

int gitm_cuda_device_count(void) {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess) return 0;
    return n;
}

const char *gitm_cupti_errstr(CUptiResult status) {
    const char *msg = NULL;
    cuptiGetResultString(status, &msg);
    return msg ? msg : "?";
}