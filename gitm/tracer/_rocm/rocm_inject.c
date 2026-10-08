/*
 * rocm_inject — rocprofiler-sdk collector, the AMD counterpart of cupti_inject.c.
 * Loaded into every HIP process via ROCP_TOOL_LIBRARIES; writes the same per-pid
 * JSONL shards ($GITM_TRACE_OUT.<pid>) while $GITM_TRACE_OUT.arm exists.
 *
 * Differences from CUPTI: kernel names come from code-object load callbacks
 * (dispatches carry only a kernel_id); rocTX has no range ids, so push/pop is a
 * per-thread stack here; grids arrive in work-items and are divided into blocks;
 * there are no sync records.
 *
 * With GITM_TRACE_NVTX set, three identity mechanisms (contract and validation
 * in gitm/distributed/correlate.py, design in docs/rocm_correlation.md):
 *   1. range stamps: the external-correlation request callback stamps every
 *      dispatch/copy/HIP-API record with the innermost rocTX range id on the
 *      enqueuing thread;
 *   2. replay stamps: inside hipGraphLaunch the stamp is (exec, ordinal);
 *   3. capture nodes: launches made while stream-capturing become graph_node
 *      records naming the open range; EndCapture + Instantiate* link capture to
 *      exec. Graph records are written unarmed: capture precedes any window.
 */

/* rocprofiler.h first: the HIP id headers assume the HIP types it declares. */
#include <rocprofiler-sdk/rocprofiler.h>
#include <rocprofiler-sdk/external_correlation.h>
#include <rocprofiler-sdk/hip/runtime_api_id.h>
#include <rocprofiler-sdk/marker/api_id.h>
#include <rocprofiler-sdk/registration.h>

#include <limits.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

#define GITM_NAME_MAX 1023          /* mirrors cupti_core.h */
#define BUF_SIZE (32 * 1024 * 1024)
#define DEFAULT_FLUSH_MS 100
#define ARM_CACHE_MS 50
#define MARKER_STACK_MAX 128
#define GRAPH_STACK_MAX 8
#define GITM_MARKER_START 0
#define GITM_MARKER_END 1

/* Node-id layout, mirrored by NODE_* in correlate.py. Ordinals are stored +1 so
 * ids are never 0 ("not a graph" downstream). */
#define GITM_NODE_FLAG (1ULL << 63)
#define GITM_NODE_SEQ_BITS 31
#define GITM_NODE_ORD_BITS 32
#define GITM_NODE_SEQ_MASK ((1ULL << GITM_NODE_SEQ_BITS) - 1)
#define GITM_NODE_ORD_MASK ((1ULL << GITM_NODE_ORD_BITS) - 1)
#define GITM_EXEC_UNTRACKED GITM_NODE_SEQ_MASK

static inline uint64_t node_id(uint64_t seq, uint64_t ordinal) {
    return ((seq & GITM_NODE_SEQ_MASK) << GITM_NODE_ORD_BITS) |
           ((ordinal + 1) & GITM_NODE_ORD_MASK);
}

static FILE *g_fp = NULL;
static char g_arm_path[PATH_MAX];
static int g_armed = 0;
static uint64_t g_armed_checked_ms = 0;
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;

static rocprofiler_context_id_t g_ctx = {0};    /* buffers + stamps */
static rocprofiler_context_id_t g_cb_ctx = {0}; /* synchronous callbacks */
static rocprofiler_buffer_id_t g_buffer = {0};
static int g_nvtx = 0;
/* CLR submits graph packets on the launching thread only under direct dispatch
 * (the Linux default). With AMD_DIRECT_DISPATCH=0 a worker thread writes them,
 * outside hipGraphLaunch's correlation scope, so no replay stamp can fire. */
static int g_direct_dispatch = 1;

static pthread_t g_flusher;
static int g_flusher_started = 0;
static atomic_int g_stop_flusher = 0;
static uint32_t g_flush_ms = DEFAULT_FLUSH_MS;

static atomic_ullong g_stamp_overflow = 0;
static atomic_ullong g_untracked_launch = 0;

/* ---- agents ------------------------------------------------------------- */

#define MAX_AGENTS 64
static struct {
    uint64_t handle;
    uint32_t ordinal;
    char name[64];
    char product[128];
} g_agents[MAX_AGENTS];
static int g_n_agents = 0;

static void copy_bounded(char *dst, size_t cap, const char *src) {
    size_t n = src ? strlen(src) : 0;
    if (n >= cap) n = cap - 1;
    if (n) memcpy(dst, src, n);
    dst[n] = '\0';
}

static rocprofiler_status_t
agent_iter(rocprofiler_agent_version_t version, const void **agents,
           size_t num_agents, void *user) {
    (void)version; (void)user;
    uint32_t ordinal = 0;
    for (size_t i = 0; i < num_agents && g_n_agents < MAX_AGENTS; i++) {
        const rocprofiler_agent_v0_t *a = agents[i];
        if (a->type != ROCPROFILER_AGENT_TYPE_GPU) continue;
        g_agents[g_n_agents].handle = a->id.handle;
        g_agents[g_n_agents].ordinal = ordinal++;
        copy_bounded(g_agents[g_n_agents].name, sizeof(g_agents[0].name), a->name);
        copy_bounded(g_agents[g_n_agents].product, sizeof(g_agents[0].product),
                     a->product_name);
        g_n_agents++;
    }
    return ROCPROFILER_STATUS_SUCCESS;
}

static uint32_t agent_ordinal(rocprofiler_agent_id_t id) {
    for (int i = 0; i < g_n_agents; i++)
        if (g_agents[i].handle == id.handle) return g_agents[i].ordinal;
    return 0;
}

/* ---- kernel_id -> {name, vgpr} ------------------------------------------ */

#define KMAP_BUCKETS 4096
typedef struct kmap_node {
    uint64_t kernel_id;
    uint32_t vgprs;
    struct kmap_node *next;
    char name[];
} kmap_node;
static kmap_node *g_kmap[KMAP_BUCKETS];

static void kmap_put(uint64_t kernel_id, const char *name, uint32_t vgprs) {
    size_t len = name ? strlen(name) : 0;
    if (len > GITM_NAME_MAX) len = GITM_NAME_MAX;
    kmap_node *n = malloc(sizeof(kmap_node) + len + 1);
    if (!n) return;
    n->kernel_id = kernel_id;
    n->vgprs = vgprs;
    memcpy(n->name, name ? name : "", len);
    n->name[len] = '\0';
    size_t b = kernel_id % KMAP_BUCKETS;
    n->next = g_kmap[b];
    g_kmap[b] = n; /* newest first: a reloaded id resolves to the new symbol */
}

static const kmap_node *kmap_get(uint64_t kernel_id) {
    for (kmap_node *n = g_kmap[kernel_id % KMAP_BUCKETS]; n; n = n->next)
        if (n->kernel_id == kernel_id) return n;
    return NULL;
}

/* ---- u64 -> (a, b) maps, guarded by g_lock ------------------------------ */

#define UMAP_BUCKETS 1024
typedef struct umap_node {
    uint64_t key, a, b;
    struct umap_node *next;
} umap_node;
typedef struct {
    umap_node *buckets[UMAP_BUCKETS];
} umap;

static umap g_hostfn; /* host stub address -> kernel_id */
static umap g_graphs; /* hipGraph_t -> (capture seq, node count); seq 0 = untrusted */
static umap g_execs;  /* hipGraphExec_t -> exec seq */

static size_t umap_bucket(uint64_t key) {
    return (size_t)((key >> 4) ^ (key >> 20)) % UMAP_BUCKETS;
}

static void umap_put(umap *m, uint64_t key, uint64_t a, uint64_t b) {
    size_t i = umap_bucket(key);
    for (umap_node *n = m->buckets[i]; n; n = n->next) {
        if (n->key == key) { /* pointer reuse: newest binding wins */
            n->a = a;
            n->b = b;
            return;
        }
    }
    umap_node *n = malloc(sizeof(umap_node));
    if (!n) return;
    *n = (umap_node){key, a, b, m->buckets[i]};
    m->buckets[i] = n;
}

static void umap_del(umap *m, uint64_t key) {
    for (umap_node **pp = &m->buckets[umap_bucket(key)]; *pp; pp = &(*pp)->next) {
        if ((*pp)->key == key) {
            umap_node *dead = *pp;
            *pp = dead->next;
            free(dead);
            return;
        }
    }
}

static void umap_remove(umap *m, uint64_t key) {
    pthread_mutex_lock(&g_lock);
    umap_del(m, key);
    pthread_mutex_unlock(&g_lock);
}

static const umap_node *umap_get(const umap *m, uint64_t key) {
    for (umap_node *n = m->buckets[umap_bucket(key)]; n; n = n->next)
        if (n->key == key) return n;
    return NULL;
}

static int umap_lookup(umap *m, uint64_t key, uint64_t *a, uint64_t *b) {
    pthread_mutex_lock(&g_lock);
    const umap_node *n = umap_get(m, key);
    if (n) {
        if (a) *a = n->a;
        if (b) *b = n->b;
    }
    pthread_mutex_unlock(&g_lock);
    return n != NULL;
}

static void umap_store(umap *m, uint64_t key, uint64_t a, uint64_t b) {
    pthread_mutex_lock(&g_lock);
    umap_put(m, key, a, b);
    pthread_mutex_unlock(&g_lock);
}

/* ---- shard writing ------------------------------------------------------ */

static void write_json_string(FILE *fp, const char *s) {
    fputc('"', fp);
    for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
        switch (*p) {
            case '"':  fputs("\\\"", fp); break;
            case '\\': fputs("\\\\", fp); break;
            case '\n': fputs("\\n", fp);  break;
            case '\r': fputs("\\r", fp);  break;
            case '\t': fputs("\\t", fp);  break;
            default:
                if (*p < 0x20) fprintf(fp, "\\u%04x", *p);
                else fputc(*p, fp);
        }
    }
    fputc('"', fp);
}

static uint64_t now_coarse_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000u + (uint64_t)(ts.tv_nsec / 1000000u);
}

/* Under g_lock; stat()s at most once per ARM_CACHE_MS. */
static void refresh_armed(void) {
    uint64_t now = now_coarse_ms();
    if (now - g_armed_checked_ms < ARM_CACHE_MS) return;
    g_armed_checked_ms = now;
    g_armed = (access(g_arm_path, F_OK) == 0);
}

/* -> CUpti_ActivityMemcpyKind, which _cupti_decode._COPY_KIND speaks. */
static int copy_kind_cupti(rocprofiler_memory_copy_operation_t op) {
    switch (op) {
        case ROCPROFILER_MEMORY_COPY_HOST_TO_DEVICE:   return 1;
        case ROCPROFILER_MEMORY_COPY_DEVICE_TO_HOST:   return 2;
        case ROCPROFILER_MEMORY_COPY_DEVICE_TO_DEVICE: return 8;
        case ROCPROFILER_MEMORY_COPY_HOST_TO_HOST:     return 9;
        default:                                       return 0;
    }
}

typedef struct {
    uint64_t range_id, graph_id, graph_node_id;
} stamp_t;

static stamp_t decode_stamp(uint64_t ext) {
    stamp_t s = {0, 0, 0};
    if (ext & GITM_NODE_FLAG) {
        s.graph_id = (ext >> GITM_NODE_ORD_BITS) & GITM_NODE_SEQ_MASK;
        if (ext & GITM_NODE_ORD_MASK) s.graph_node_id = ext & ~GITM_NODE_FLAG;
    } else {
        s.range_id = ext;
    }
    return s;
}

static int is_graph_launch_op(rocprofiler_tracing_operation_t op) {
    return op == ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphLaunch ||
           op == ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphLaunch_spt;
}

/* ---- buffer callback ---------------------------------------------------- */

static void
buffer_cb(rocprofiler_context_id_t context, rocprofiler_buffer_id_t buffer_id,
          rocprofiler_record_header_t **headers, size_t num_headers,
          void *user_data, uint64_t drop_count) {
    (void)context; (void)buffer_id; (void)user_data;

    pthread_mutex_lock(&g_lock);
    refresh_armed();
    if (!g_fp || !g_armed) {
        pthread_mutex_unlock(&g_lock);
        return;
    }
    if (drop_count > 0)
        fprintf(g_fp, "{\"kind\":\"meta\",\"dropped_records\":%llu}\n",
                (unsigned long long)drop_count);

    for (size_t i = 0; i < num_headers; i++) {
        rocprofiler_record_header_t *h = headers[i];
        if (h->category != ROCPROFILER_BUFFER_CATEGORY_TRACING) continue;

        if (h->kind == ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH) {
            rocprofiler_buffer_tracing_kernel_dispatch_record_t *k = h->payload;
            const rocprofiler_kernel_dispatch_info_t *di = &k->dispatch_info;
            const kmap_node *sym = kmap_get(di->kernel_id);
            stamp_t st = decode_stamp(k->correlation_id.external.value);
            uint32_t wx = di->workgroup_size.x, wy = di->workgroup_size.y,
                     wz = di->workgroup_size.z;
            fputs("{\"kind\":\"kernel\",\"name\":", g_fp);
            write_json_string(g_fp, sym ? sym->name : "");
            fprintf(g_fp,
                    ",\"start_ns\":%llu,\"end_ns\":%llu,\"device_id\":%u,"
                    "\"context_id\":0,\"stream_id\":%llu,\"correlation_id\":%llu,"
                    "\"grid\":[%u,%u,%u],\"block\":[%u,%u,%u],"
                    "\"static_shared_mem\":%u,\"dynamic_shared_mem\":0,"
                    "\"registers_per_thread\":%u,\"kernel_id\":%llu,"
                    "\"thread_id\":%llu,\"range_id\":%llu,"
                    "\"graph_id\":%llu,\"graph_node_id\":%llu}\n",
                    (unsigned long long)k->start_timestamp,
                    (unsigned long long)k->end_timestamp,
                    agent_ordinal(di->agent_id),
                    (unsigned long long)di->queue_id.handle,
                    (unsigned long long)k->correlation_id.internal,
                    wx ? di->grid_size.x / wx : 0, wy ? di->grid_size.y / wy : 0,
                    wz ? di->grid_size.z / wz : 0, wx, wy, wz,
                    di->group_segment_size, sym ? sym->vgprs : 0,
                    (unsigned long long)di->kernel_id,
                    (unsigned long long)k->thread_id,
                    (unsigned long long)st.range_id,
                    (unsigned long long)st.graph_id,
                    (unsigned long long)st.graph_node_id);
        } else if (h->kind == ROCPROFILER_BUFFER_TRACING_MEMORY_COPY) {
            rocprofiler_buffer_tracing_memory_copy_record_t *m = h->payload;
            stamp_t st = decode_stamp(m->correlation_id.external.value);
            uint32_t dev = m->operation == ROCPROFILER_MEMORY_COPY_HOST_TO_DEVICE
                               ? agent_ordinal(m->dst_agent_id)
                               : agent_ordinal(m->src_agent_id);
            fprintf(g_fp,
                    "{\"kind\":\"memcpy\",\"copy_kind\":%d,\"bytes\":%llu,"
                    "\"start_ns\":%llu,\"end_ns\":%llu,\"device_id\":%u,"
                    "\"context_id\":0,\"stream_id\":0,\"correlation_id\":%llu,"
                    "\"thread_id\":%llu,\"range_id\":%llu,"
                    "\"graph_id\":%llu,\"graph_node_id\":%llu}\n",
                    copy_kind_cupti(m->operation), (unsigned long long)m->bytes,
                    (unsigned long long)m->start_timestamp,
                    (unsigned long long)m->end_timestamp, dev,
                    (unsigned long long)m->correlation_id.internal,
                    (unsigned long long)m->thread_id,
                    (unsigned long long)st.range_id,
                    (unsigned long long)st.graph_id,
                    (unsigned long long)st.graph_node_id);
        } else if (h->kind == ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API) {
            /* graph_launch lets the decoder keep a replayed kernel the ordinal
             * stamp missed off this record's range. */
            rocprofiler_buffer_tracing_hip_api_record_t *a = h->payload;
            fprintf(g_fp,
                    "{\"kind\":\"runtime\",\"start_ns\":%llu,\"end_ns\":%llu,"
                    "\"correlation_id\":%llu,\"thread_id\":%llu,"
                    "\"range_id\":%llu,\"graph_launch\":%d}\n",
                    (unsigned long long)a->start_timestamp,
                    (unsigned long long)a->end_timestamp,
                    (unsigned long long)a->correlation_id.internal,
                    (unsigned long long)a->thread_id,
                    (unsigned long long)decode_stamp(a->correlation_id.external.value)
                        .range_id,
                    is_graph_launch_op(a->operation));
        }
    }
    fflush(g_fp);
    pthread_mutex_unlock(&g_lock);
}

/* ---- per-thread state --------------------------------------------------- */

static atomic_uint g_marker_seq = 1;
static atomic_ullong g_capture_seq = 0;
static atomic_ullong g_exec_seq = 0;

static __thread struct {
    uint32_t ids[MARKER_STACK_MAX];
    char *names[MARKER_STACK_MAX];
    int depth;
} tls_ranges;

/* A capture covers its origin stream plus every stream that waits on an event
 * recorded inside it (how forks join). Launches on any other stream are not
 * nodes and must not take an ordinal. Past the set sizes the capture is marked
 * untrusted: its replays are left unnamed rather than guessed. */
#define CAPTURE_STREAMS_MAX 32
#define CAPTURE_EVENTS_MAX 128
static __thread struct {
    int active, untrusted;
    uint64_t seq, n;
    uint64_t streams[CAPTURE_STREAMS_MAX];
    uint64_t events[CAPTURE_EVENTS_MAX];
    int n_streams, n_events;
} tls_capture;

static int set_has(const uint64_t *set, int n, uint64_t v) {
    for (int i = 0; i < n; i++)
        if (set[i] == v) return 1;
    return 0;
}

static void set_add(uint64_t *set, int *n, int cap, uint64_t v) {
    if (set_has(set, *n, v)) return;
    if (*n < cap) set[(*n)++] = v;
    else tls_capture.untrusted = 1;
}

static __thread struct {
    uint64_t exec[GRAPH_STACK_MAX], ordinal[GRAPH_STACK_MAX];
    int depth;
} tls_graph;

static int range_top(void) {
    int d = tls_ranges.depth;
    return (d > 0 && d <= MARKER_STACK_MAX) ? d - 1 : -1;
}

/* ---- the stamp ---------------------------------------------------------- */

static int
stamp_cb(rocprofiler_thread_id_t thread_id, rocprofiler_context_id_t context_id,
         rocprofiler_external_correlation_id_request_kind_t kind,
         rocprofiler_tracing_operation_t operation, uint64_t internal_corr_id,
         rocprofiler_user_data_t *external, void *data) {
    (void)thread_id; (void)context_id; (void)operation;
    (void)internal_corr_id; (void)data;

    int d = tls_graph.depth;
    if (d > 0 && d <= GRAPH_STACK_MAX &&
        (kind == ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_KERNEL_DISPATCH ||
         kind == ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_MEMORY_COPY)) {
        uint64_t exec = tls_graph.exec[d - 1];
        uint64_t ord = tls_graph.ordinal[d - 1]++;
        if (ord + 1 > GITM_NODE_ORD_MASK || exec > GITM_NODE_SEQ_MASK) {
            /* still a replay, with no node */
            atomic_fetch_add(&g_stamp_overflow, 1);
            external->value =
                GITM_NODE_FLAG | ((exec & GITM_NODE_SEQ_MASK) << GITM_NODE_ORD_BITS);
        } else {
            external->value = GITM_NODE_FLAG | node_id(exec, ord);
        }
        return 0;
    }
    int t = range_top();
    external->value = t < 0 ? 0 : tls_ranges.ids[t];
    return 0;
}

/* ---- structural records (written armed or not) -------------------------- */

typedef struct {
    uint32_t x, y, z;
} dims_t;

static void emit_graph_node(const char *node_kind, uint64_t kernel_id,
                            const dims_t *grid, const dims_t *block) {
    uint64_t ord = tls_capture.n++;
    int t = range_top();
    pthread_mutex_lock(&g_lock);
    if (g_fp) {
        fprintf(g_fp,
                "{\"kind\":\"graph_node\",\"graph_node_id\":%llu,"
                "\"capture_id\":%llu,\"node_kind\":\"%s\",\"kernel_id\":%llu,"
                "\"name\":",
                (unsigned long long)(GITM_NODE_FLAG | node_id(tls_capture.seq, ord)),
                (unsigned long long)tls_capture.seq, node_kind,
                (unsigned long long)kernel_id);
        write_json_string(g_fp, t < 0 || !tls_ranges.names[t] ? "" : tls_ranges.names[t]);
        if (grid && block)
            fprintf(g_fp, ",\"grid\":[%u,%u,%u],\"block\":[%u,%u,%u]", grid->x,
                    grid->y, grid->z, block->x, block->y, block->z);
        fputs("}\n", g_fp);
    }
    pthread_mutex_unlock(&g_lock);
}

static void emit_graph_exec(uint64_t exec_seq, uint64_t capture_seq,
                            uint64_t n_nodes, int captured) {
    pthread_mutex_lock(&g_lock);
    if (g_fp) {
        if (captured)
            fprintf(g_fp,
                    "{\"kind\":\"graph_exec\",\"graph_id\":%llu,"
                    "\"capture_id\":%llu,\"n_nodes\":%llu}\n",
                    (unsigned long long)exec_seq, (unsigned long long)capture_seq,
                    (unsigned long long)n_nodes);
        else /* built with the graph API: nothing to name its nodes */
            fprintf(g_fp, "{\"kind\":\"graph_exec\",\"graph_id\":%llu,"
                          "\"capture_id\":0}\n",
                    (unsigned long long)exec_seq);
        fflush(g_fp);
    }
    pthread_mutex_unlock(&g_lock);
}

/* ---- HIP API callbacks: capture and graph launches ---------------------- */

static uint32_t div0(uint32_t a, uint32_t b) { return b ? a / b : 0; }

static uint64_t hostfn_kid(const void *fn) {
    uint64_t kid = 0;
    umap_lookup(&g_hostfn, (uint64_t)(uintptr_t)fn, &kid, NULL);
    return kid;
}

#define DIM3(d) ((dims_t){(d).x, (d).y, (d).z})

/* One captured launch/copy/memset -> one graph node, if its stream belongs to
 * the capture. Geometry in blocks, the unit dispatch records are normalized to. */
static void capture_node(rocprofiler_tracing_operation_t op,
                         const rocprofiler_hip_api_args_t *a) {
    const char *kind = "kernel";
    hipStream_t stream = NULL;
    uint64_t kid = 0;
    dims_t g = {0, 0, 0}, b = {0, 0, 0};
    int dims = 1;
    switch (op) {
        case ROCPROFILER_HIP_RUNTIME_API_ID_hipLaunchKernel:
            stream = a->hipLaunchKernel.stream;
            kid = hostfn_kid(a->hipLaunchKernel.function_address);
            g = DIM3(a->hipLaunchKernel.numBlocks);
            b = DIM3(a->hipLaunchKernel.dimBlocks);
            break;
        case ROCPROFILER_HIP_RUNTIME_API_ID_hipLaunchKernel_spt:
            stream = a->hipLaunchKernel_spt.stream;
            kid = hostfn_kid(a->hipLaunchKernel_spt.function_address);
            g = DIM3(a->hipLaunchKernel_spt.numBlocks);
            b = DIM3(a->hipLaunchKernel_spt.dimBlocks);
            break;
        case ROCPROFILER_HIP_RUNTIME_API_ID_hipExtLaunchKernel:
            stream = a->hipExtLaunchKernel.stream;
            kid = hostfn_kid(a->hipExtLaunchKernel.function_address);
            g = DIM3(a->hipExtLaunchKernel.numBlocks);
            b = DIM3(a->hipExtLaunchKernel.dimBlocks);
            break;
        case ROCPROFILER_HIP_RUNTIME_API_ID_hipLaunchCooperativeKernel:
            stream = a->hipLaunchCooperativeKernel.stream;
            kid = hostfn_kid(a->hipLaunchCooperativeKernel.func);
            g = DIM3(a->hipLaunchCooperativeKernel.gridDim);
            b = DIM3(a->hipLaunchCooperativeKernel.blockDimX);
            break;
        case ROCPROFILER_HIP_RUNTIME_API_ID_hipLaunchKernelExC: {
            const hipLaunchConfig_t *c = a->hipLaunchKernelExC.config;
            if (!c) return; /* no stream to attribute it to */
            stream = c->stream;
            kid = hostfn_kid(a->hipLaunchKernelExC.fPtr);
            g = DIM3(c->gridDim);
            b = DIM3(c->blockDim);
            break;
        }
        /* hipFunction_t launches (Triton, hipBLASLt, AITER asm) have no
         * kernel_id mapping; they validate on geometry alone. */
        case ROCPROFILER_HIP_RUNTIME_API_ID_hipModuleLaunchKernel:
            stream = a->hipModuleLaunchKernel.stream;
            g = (dims_t){a->hipModuleLaunchKernel.gridDimX, a->hipModuleLaunchKernel.gridDimY,
                         a->hipModuleLaunchKernel.gridDimZ};
            b = (dims_t){a->hipModuleLaunchKernel.blockDimX, a->hipModuleLaunchKernel.blockDimY,
                         a->hipModuleLaunchKernel.blockDimZ};
            break;
        case ROCPROFILER_HIP_RUNTIME_API_ID_hipExtModuleLaunchKernel:
            /* global size is in work-items, like the dispatch record */
            stream = a->hipExtModuleLaunchKernel.stream;
            b = (dims_t){a->hipExtModuleLaunchKernel.localWorkSizeX,
                         a->hipExtModuleLaunchKernel.localWorkSizeY,
                         a->hipExtModuleLaunchKernel.localWorkSizeZ};
            g = (dims_t){div0(a->hipExtModuleLaunchKernel.globalWorkSizeX, b.x),
                         div0(a->hipExtModuleLaunchKernel.globalWorkSizeY, b.y),
                         div0(a->hipExtModuleLaunchKernel.globalWorkSizeZ, b.z)};
            break;
        case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyAsync:
            kind = "memcpy", dims = 0, stream = a->hipMemcpyAsync.stream;
            break;
        case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyDtoDAsync:
            kind = "memcpy", dims = 0, stream = a->hipMemcpyDtoDAsync.stream;
            break;
        case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyDtoHAsync:
            kind = "memcpy", dims = 0, stream = a->hipMemcpyDtoHAsync.stream;
            break;
        case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyHtoDAsync:
            kind = "memcpy", dims = 0, stream = a->hipMemcpyHtoDAsync.stream;
            break;
        case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpy2DAsync:
            kind = "memcpy", dims = 0, stream = a->hipMemcpy2DAsync.stream;
            break;
        case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyPeerAsync:
            kind = "memcpy", dims = 0, stream = a->hipMemcpyPeerAsync.stream;
            break;
        case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemsetAsync:
            kind = "memset", dims = 0, stream = a->hipMemsetAsync.stream;
            break;
        case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemsetD8Async:
            kind = "memset", dims = 0, stream = a->hipMemsetD8Async.stream;
            break;
        case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemsetD16Async:
            kind = "memset", dims = 0, stream = a->hipMemsetD16Async.stream;
            break;
        case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemsetD32Async:
            kind = "memset", dims = 0, stream = a->hipMemsetD32Async.stream;
            break;
        case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemset2DAsync:
            kind = "memset", dims = 0, stream = a->hipMemset2DAsync.stream;
            break;
        default:
            return;
    }
    if (!set_has(tls_capture.streams, tls_capture.n_streams, (uint64_t)(uintptr_t)stream))
        return; /* another stream: not a node of this capture */
    emit_graph_node(kind, kid, dims ? &g : NULL, dims ? &b : NULL);
}

/* Bind an executable to a fresh exec seq and the capture its graph came from
 * (instantiate, or hipGraphExecUpdate swapping in another graph). */
static void link_exec(hipGraphExec_t exec, hipGraph_t graph) {
    if (!exec) return;
    uint64_t exec_seq = atomic_fetch_add(&g_exec_seq, 1) + 1;
    uint64_t cap = 0, n = 0;
    int captured = umap_lookup(&g_graphs, (uint64_t)(uintptr_t)graph, &cap, &n) && cap;
    umap_store(&g_execs, (uint64_t)(uintptr_t)exec, exec_seq, 0);
    emit_graph_exec(exec_seq, cap, n, captured);
}

static void link_instantiate(hipGraphExec_t *pexec, hipGraph_t graph) {
    if (pexec) link_exec(*pexec, graph);
}

static void hip_api_cb(rocprofiler_callback_tracing_record_t record) {
    const rocprofiler_callback_tracing_hip_api_data_t *d = record.payload;
    const rocprofiler_hip_api_args_t *a = &d->args;
    rocprofiler_tracing_operation_t op = record.operation;
    int ok = d->retval.hipError_t_retval == hipSuccess;

    if (record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER) {
        if (is_graph_launch_op(op)) {
            hipGraphExec_t exec = op == ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphLaunch
                                      ? a->hipGraphLaunch.graphExec
                                      : a->hipGraphLaunch_spt.graphExec;
            uint64_t seq = GITM_EXEC_UNTRACKED;
            if (!umap_lookup(&g_execs, (uint64_t)(uintptr_t)exec, &seq, NULL))
                atomic_fetch_add(&g_untracked_launch, 1);
            int dp = tls_graph.depth++;
            if (dp < GRAPH_STACK_MAX) {
                tls_graph.exec[dp] = seq;
                tls_graph.ordinal[dp] = 0;
            }
        } else if (tls_capture.active) {
            capture_node(op, a);
        }
        return;
    }

    switch (op) {
        case ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphLaunch:
        case ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphLaunch_spt:
            if (tls_graph.depth > 0) tls_graph.depth--;
            return;
        case ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamBeginCapture:
        case ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamBeginCaptureToGraph:
            if (ok && !tls_capture.active) {
                hipStream_t origin = op == ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamBeginCapture
                                         ? a->hipStreamBeginCapture.stream
                                         : a->hipStreamBeginCaptureToGraph.stream;
                tls_capture.active = 1;
                tls_capture.untrusted = 0;
                tls_capture.seq = atomic_fetch_add(&g_capture_seq, 1) + 1;
                tls_capture.n = 0;
                tls_capture.n_streams = tls_capture.n_events = 0;
                set_add(tls_capture.streams, &tls_capture.n_streams, CAPTURE_STREAMS_MAX,
                        (uint64_t)(uintptr_t)origin);
            }
            return;
        case ROCPROFILER_HIP_RUNTIME_API_ID_hipEventRecord:
        case ROCPROFILER_HIP_RUNTIME_API_ID_hipEventRecordWithFlags: {
            hipStream_t st = op == ROCPROFILER_HIP_RUNTIME_API_ID_hipEventRecord
                                 ? a->hipEventRecord.stream
                                 : a->hipEventRecordWithFlags.stream;
            hipEvent_t ev = op == ROCPROFILER_HIP_RUNTIME_API_ID_hipEventRecord
                                ? a->hipEventRecord.event
                                : a->hipEventRecordWithFlags.event;
            if (ok && tls_capture.active &&
                set_has(tls_capture.streams, tls_capture.n_streams, (uint64_t)(uintptr_t)st))
                set_add(tls_capture.events, &tls_capture.n_events, CAPTURE_EVENTS_MAX,
                        (uint64_t)(uintptr_t)ev);
            return;
        }
        case ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamWaitEvent:
            /* a stream waiting on a captured event joins the capture (a fork) */
            if (ok && tls_capture.active &&
                set_has(tls_capture.events, tls_capture.n_events,
                        (uint64_t)(uintptr_t)a->hipStreamWaitEvent.event))
                set_add(tls_capture.streams, &tls_capture.n_streams, CAPTURE_STREAMS_MAX,
                        (uint64_t)(uintptr_t)a->hipStreamWaitEvent.stream);
            return;
        case ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamEndCapture:
            if (tls_capture.active && ok && a->hipStreamEndCapture.pGraph &&
                *a->hipStreamEndCapture.pGraph)
                /* an untrusted capture is stored as seq 0: never projected */
                umap_store(&g_graphs, (uint64_t)(uintptr_t)*a->hipStreamEndCapture.pGraph,
                           tls_capture.untrusted ? 0 : tls_capture.seq, tls_capture.n);
            tls_capture.active = 0;
            return;
        case ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphDestroy:
            /* a reused pointer must not inherit this capture */
            if (ok) umap_remove(&g_graphs, (uint64_t)(uintptr_t)a->hipGraphDestroy.graph);
            return;
        case ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphExecDestroy:
            if (ok) umap_remove(&g_execs, (uint64_t)(uintptr_t)a->hipGraphExecDestroy.graphExec);
            return;
        case ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphExecUpdate:
            /* the executable now runs another graph's nodes: new exec seq */
            if (ok) link_exec(a->hipGraphExecUpdate.hGraphExec, a->hipGraphExecUpdate.hGraph);
            return;
        case ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphInstantiate:
            if (ok) link_instantiate(a->hipGraphInstantiate.pGraphExec,
                                     a->hipGraphInstantiate.graph);
            return;
        case ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphInstantiateWithFlags:
            if (ok) link_instantiate(a->hipGraphInstantiateWithFlags.pGraphExec,
                                     a->hipGraphInstantiateWithFlags.graph);
            return;
        case ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphInstantiateWithParams:
            if (ok) link_instantiate(a->hipGraphInstantiateWithParams.pGraphExec,
                                     a->hipGraphInstantiateWithParams.graph);
            return;
        default:
            return;
    }
}

static const rocprofiler_tracing_operation_t k_hip_ops[] = {
    ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamBeginCapture,
    ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamBeginCaptureToGraph,
    ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamEndCapture,
    ROCPROFILER_HIP_RUNTIME_API_ID_hipEventRecord,
    ROCPROFILER_HIP_RUNTIME_API_ID_hipEventRecordWithFlags,
    ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamWaitEvent,
    ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphDestroy,
    ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphExecDestroy,
    ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphExecUpdate,
    ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphInstantiate,
    ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphInstantiateWithFlags,
    ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphInstantiateWithParams,
    ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphLaunch,
    ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphLaunch_spt,
    ROCPROFILER_HIP_RUNTIME_API_ID_hipLaunchKernel,
    ROCPROFILER_HIP_RUNTIME_API_ID_hipLaunchKernel_spt,
    ROCPROFILER_HIP_RUNTIME_API_ID_hipExtLaunchKernel,
    ROCPROFILER_HIP_RUNTIME_API_ID_hipLaunchCooperativeKernel,
    ROCPROFILER_HIP_RUNTIME_API_ID_hipLaunchKernelExC,
    ROCPROFILER_HIP_RUNTIME_API_ID_hipModuleLaunchKernel,
    ROCPROFILER_HIP_RUNTIME_API_ID_hipExtModuleLaunchKernel,
    ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyAsync,
    ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyDtoDAsync,
    ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyDtoHAsync,
    ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyHtoDAsync,
    ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpy2DAsync,
    ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyPeerAsync,
    ROCPROFILER_HIP_RUNTIME_API_ID_hipMemsetAsync,
    ROCPROFILER_HIP_RUNTIME_API_ID_hipMemsetD8Async,
    ROCPROFILER_HIP_RUNTIME_API_ID_hipMemsetD16Async,
    ROCPROFILER_HIP_RUNTIME_API_ID_hipMemsetD32Async,
    ROCPROFILER_HIP_RUNTIME_API_ID_hipMemset2DAsync,
};

/* ---- code objects + rocTX markers --------------------------------------- */

static void emit_marker(const char *name, uint32_t marker_id, int flags,
                        uint64_t thread_id) {
    rocprofiler_timestamp_t ts = 0;
    rocprofiler_get_timestamp(&ts);
    pthread_mutex_lock(&g_lock);
    refresh_armed();
    if (g_fp && g_armed) {
        fputs("{\"kind\":\"marker\",\"name\":", g_fp);
        write_json_string(g_fp, name ? name : "");
        fprintf(g_fp,
                ",\"timestamp_ns\":%llu,\"marker_id\":%u,\"marker_flags\":%d,"
                "\"thread_id\":%llu}\n",
                (unsigned long long)ts, marker_id, flags,
                (unsigned long long)thread_id);
    }
    pthread_mutex_unlock(&g_lock);
}

static char *dup_bounded(const char *s) {
    size_t n = s ? strlen(s) : 0;
    if (n > GITM_NAME_MAX) n = GITM_NAME_MAX;
    char *out = malloc(n + 1);
    if (!out) return NULL;
    if (n) memcpy(out, s, n);
    out[n] = '\0';
    return out;
}

static void
callback_cb(rocprofiler_callback_tracing_record_t record,
            rocprofiler_user_data_t *user_data, void *callback_data) {
    (void)user_data; (void)callback_data;

    if (record.kind == ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API) {
        hip_api_cb(record);
        return;
    }

    if (record.kind == ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT) {
        if (record.phase != ROCPROFILER_CALLBACK_PHASE_LOAD) return;
        if (record.operation == ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER) {
            rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t
                *d = record.payload;
            pthread_mutex_lock(&g_lock);
            kmap_put(d->kernel_id, d->kernel_name, d->arch_vgpr_count);
            pthread_mutex_unlock(&g_lock);
        } else if (record.operation ==
                   ROCPROFILER_CODE_OBJECT_HOST_KERNEL_SYMBOL_REGISTER) {
            rocprofiler_callback_tracing_code_object_host_kernel_symbol_register_data_t
                *d = record.payload;
            umap_store(&g_hostfn, d->host_function.value, d->kernel_id, 0);
        }
        return;
    }

    if (record.kind != ROCPROFILER_CALLBACK_TRACING_MARKER_CORE_API ||
        record.phase != ROCPROFILER_CALLBACK_PHASE_ENTER)
        return;
    rocprofiler_callback_tracing_marker_api_data_t *d = record.payload;

    if (record.operation == ROCPROFILER_MARKER_CORE_API_ID_roctxRangePushA) {
        uint32_t id = atomic_fetch_add(&g_marker_seq, 1);
        if (tls_ranges.depth < MARKER_STACK_MAX) {
            tls_ranges.ids[tls_ranges.depth] = id;
            tls_ranges.names[tls_ranges.depth] = dup_bounded(d->args.roctxRangePushA.message);
        }
        tls_ranges.depth++; /* past the cap: emitted, untracked, pop pairs nothing */
        emit_marker(d->args.roctxRangePushA.message, id, GITM_MARKER_START,
                    record.thread_id);
    } else if (record.operation == ROCPROFILER_MARKER_CORE_API_ID_roctxRangePop) {
        if (tls_ranges.depth <= 0) return;
        tls_ranges.depth--;
        if (tls_ranges.depth >= MARKER_STACK_MAX) return;
        free(tls_ranges.names[tls_ranges.depth]);
        tls_ranges.names[tls_ranges.depth] = NULL;
        emit_marker(NULL, tls_ranges.ids[tls_ranges.depth], GITM_MARKER_END,
                    record.thread_id);
    } else if (record.operation == ROCPROFILER_MARKER_CORE_API_ID_roctxMarkA) {
        /* both halves at once: a zero-length range */
        uint32_t id = atomic_fetch_add(&g_marker_seq, 1);
        emit_marker(d->args.roctxMarkA.message, id, GITM_MARKER_START, record.thread_id);
        emit_marker(NULL, id, GITM_MARKER_END, record.thread_id);
    }
}

/* ---- flush thread ------------------------------------------------------- */

static void emit_stamp_faults(void) {
    unsigned long long over = atomic_exchange(&g_stamp_overflow, 0);
    unsigned long long untracked = atomic_exchange(&g_untracked_launch, 0);
    if (!over && !untracked) return;
    pthread_mutex_lock(&g_lock);
    if (g_fp)
        fprintf(g_fp,
                "{\"kind\":\"meta\",\"graph_stamp_overflow\":%llu,"
                "\"graph_untracked_launch\":%llu}\n",
                over, untracked);
    pthread_mutex_unlock(&g_lock);
}

static void *flusher_main(void *arg) {
    (void)arg;
    while (!atomic_load(&g_stop_flusher)) {
        usleep((useconds_t)g_flush_ms * 1000u);
        rocprofiler_flush_buffer(g_buffer);
        emit_stamp_faults();
    }
    return NULL;
}

/* ---- lifecycle ---------------------------------------------------------- */

/* In-band provenance, unarmed: gitm.tracer.vendor reads it. */
static void emit_collector_meta(void) {
    pthread_mutex_lock(&g_lock);
    if (g_fp) {
        fprintf(g_fp,
                "{\"kind\":\"meta\",\"collector\":\"rocprofiler-sdk\","
                "\"sdk_version\":\"%d.%d.%d\",\"identity\":%d,"
                "\"direct_dispatch\":%d,\"agents\":[",
                ROCPROFILER_VERSION_MAJOR, ROCPROFILER_VERSION_MINOR,
                ROCPROFILER_VERSION_PATCH, g_nvtx, g_direct_dispatch);
        for (int i = 0; i < g_n_agents; i++) {
            fprintf(g_fp, "%s{\"ordinal\":%u,\"name\":", i ? "," : "",
                    g_agents[i].ordinal);
            write_json_string(g_fp, g_agents[i].name);
            fputs(",\"product\":", g_fp);
            write_json_string(g_fp, g_agents[i].product);
            fputc('}', g_fp);
        }
        fputs("]}\n", g_fp);
        fflush(g_fp);
    }
    pthread_mutex_unlock(&g_lock);
}

static int tool_init(rocprofiler_client_finalize_t fini, void *tool_data) {
    (void)fini; (void)tool_data;

    rocprofiler_query_available_agents(ROCPROFILER_AGENT_INFO_VERSION_0, agent_iter,
                                       sizeof(rocprofiler_agent_v0_t), NULL);
    emit_collector_meta();

    if (rocprofiler_create_context(&g_ctx) != ROCPROFILER_STATUS_SUCCESS ||
        rocprofiler_create_context(&g_cb_ctx) != ROCPROFILER_STATUS_SUCCESS ||
        rocprofiler_create_buffer(g_ctx, BUF_SIZE, BUF_SIZE / 2,
                                  ROCPROFILER_BUFFER_POLICY_LOSSLESS, buffer_cb, NULL,
                                  &g_buffer) != ROCPROFILER_STATUS_SUCCESS)
        return -1;

    rocprofiler_configure_buffer_tracing_service(
        g_ctx, ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH, NULL, 0, g_buffer);
    rocprofiler_configure_buffer_tracing_service(
        g_ctx, ROCPROFILER_BUFFER_TRACING_MEMORY_COPY, NULL, 0, g_buffer);
    rocprofiler_configure_callback_tracing_service(
        g_cb_ctx, ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT, NULL, 0, callback_cb, NULL);

    if (g_nvtx) {
        rocprofiler_configure_buffer_tracing_service(
            g_ctx, ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API, NULL, 0, g_buffer);
        rocprofiler_configure_callback_tracing_service(
            g_cb_ctx, ROCPROFILER_CALLBACK_TRACING_MARKER_CORE_API, NULL, 0,
            callback_cb, NULL);
        /* separate context: never competes with the buffered HIP API service */
        rocprofiler_configure_callback_tracing_service(
            g_cb_ctx, ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API, k_hip_ops,
            sizeof(k_hip_ops) / sizeof(k_hip_ops[0]), callback_cb, NULL);
        static const rocprofiler_external_correlation_id_request_kind_t kinds[] = {
            ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_KERNEL_DISPATCH,
            ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_MEMORY_COPY,
            ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_HIP_RUNTIME_API,
        };
        rocprofiler_configure_external_correlation_id_request_service(
            g_ctx, kinds, sizeof(kinds) / sizeof(kinds[0]), stamp_cb, NULL);
    }

    rocprofiler_callback_thread_t cb_thread;
    if (rocprofiler_create_callback_thread(&cb_thread) == ROCPROFILER_STATUS_SUCCESS)
        rocprofiler_assign_callback_thread(g_buffer, cb_thread);

    if (rocprofiler_start_context(g_cb_ctx) != ROCPROFILER_STATUS_SUCCESS ||
        rocprofiler_start_context(g_ctx) != ROCPROFILER_STATUS_SUCCESS)
        return -1;

    const char *ms = getenv("GITM_TRACE_FLUSH_MS");
    if (ms && *ms) g_flush_ms = (uint32_t)strtoul(ms, NULL, 10);
    if (g_flush_ms == 0) g_flush_ms = DEFAULT_FLUSH_MS;
    if (pthread_create(&g_flusher, NULL, flusher_main, NULL) == 0) g_flusher_started = 1;
    return 0;
}

static void tool_fini(void *tool_data) {
    (void)tool_data;
    if (g_flusher_started) {
        atomic_store(&g_stop_flusher, 1);
        pthread_join(g_flusher, NULL);
        g_flusher_started = 0;
    }
    rocprofiler_stop_context(g_ctx);
    rocprofiler_stop_context(g_cb_ctx);
    rocprofiler_flush_buffer(g_buffer);
    emit_stamp_faults();
    pthread_mutex_lock(&g_lock);
    if (g_fp) {
        fclose(g_fp);
        g_fp = NULL;
    }
    pthread_mutex_unlock(&g_lock);
}

rocprofiler_tool_configure_result_t *
rocprofiler_configure(uint32_t version, const char *runtime_version,
                      uint32_t priority, rocprofiler_client_id_t *id) {
    (void)version; (void)runtime_version; (void)priority;
    id->name = "gitm";

    /* Can't open the output: decline, and the process runs untraced. */
    const char *out = getenv("GITM_TRACE_OUT");
    if (!out || !*out) return NULL;

    char shard[PATH_MAX];
    if (snprintf(shard, sizeof(shard), "%s.%d", out, (int)getpid()) >= (int)sizeof(shard) ||
        snprintf(g_arm_path, sizeof(g_arm_path), "%s.arm", out) >= (int)sizeof(g_arm_path))
        return NULL;

    g_fp = fopen(shard, "a");
    if (!g_fp) return NULL;
    setvbuf(g_fp, NULL, _IOFBF, 1 << 20);

    const char *nvtx = getenv("GITM_TRACE_NVTX");
    g_nvtx = nvtx && *nvtx && strcmp(nvtx, "0") != 0;
    const char *dd = getenv("AMD_DIRECT_DISPATCH");
    g_direct_dispatch = !(dd && strcmp(dd, "0") == 0);

    static rocprofiler_tool_configure_result_t result = {
        sizeof(rocprofiler_tool_configure_result_t), tool_init, tool_fini, NULL};
    return &result;
}
