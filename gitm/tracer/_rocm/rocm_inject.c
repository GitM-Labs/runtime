/*
 * rocm_inject — rocprofiler-sdk collection for processes we don't control.
 * The AMD counterpart of cupti_inject.c; emits the SAME per-pid JSONL shards.
 *
 * Built as libgitm_rocm_inject.so: a plain shared library with NO libpython.
 * Point the ROCm runtime at it with
 *
 *     export ROCP_TOOL_LIBRARIES=/abs/path/to/libgitm_rocm_inject.so
 *     export GITM_TRACE_OUT=/abs/path/to/trace.jsonl
 *
 * and rocprofiler-register (linked into the HIP/HSA runtimes since ROCm 6.2)
 * dlopens it inside every process that initializes HIP — parent AND children —
 * calling rocprofiler_configure() before a single kernel runs. Like
 * CUDA_INJECTION64_PATH it is an ordinary environment variable inherited across
 * fork/spawn, which is what lets it see a vLLM/SGLang EngineCore child the
 * parent interpreter cannot instrument.
 *
 * Differences from the CUPTI collector that are structural, not omissions:
 *
 *   * Kernel names are NOT on the dispatch record. rocprofiler-sdk hands them
 *     out once, at code-object load, via the CODE_OBJECT callback service; the
 *     dispatch record carries only a kernel_id. We keep a kernel_id -> name map
 *     and resolve at emit time. A dispatch whose load callback we missed (tool
 *     attached late) emits an empty name and decodes as <anonymous>.
 *   * There is no NVTX_INJECTION64_PATH analog and none is needed. rocTX
 *     markers (what torch.cuda.nvtx.range_push compiles to on ROCm) are
 *     delivered by the MARKER_CORE_API service of this same tool (given the
 *     roctx shim on the emitting side — see roctx_shim.c).
 *   * roctx has no per-range id, so push/pop pairing is a per-thread stack in
 *     this library (NVTX gave us marker ids for free). Ids are synthesized from
 *     one atomic counter; the JSONL marker halves look identical to CUPTI's and
 *     _cupti_decode.pair_markers joins them unchanged.
 *   * grid on a dispatch record is in WORK-ITEMS (HSA convention), not blocks.
 *     We divide by the workgroup size so KernelEvent.grid_* means the same
 *     thing on both vendors.
 *   * No SYNCHRONIZATION activity kind exists; AMD traces carry no sync
 *     records. Consumers already tolerate their absence.
 *   * registers_per_thread reports the kernel's arch VGPR count — the AMD
 *     occupancy-limiting analog — captured from the code-object symbol.
 *
 * Kernel identity under GITM_TRACE_NVTX (the correlation arm). Three AMD-native
 * mechanisms, contract in gitm/distributed/correlate.py:
 *
 *   1. Range stamps. rocprofiler-sdk's external-correlation request service
 *      asks this tool — synchronously, on the enqueuing thread — for a value to
 *      put in every dispatch, copy and HIP API record's correlation_id.external.
 *      We answer with the id of the innermost rocTX range open on that thread,
 *      so an eager kernel arrives already joined to its range ("range_id"). The
 *      HIP-API ("runtime") records stay on as the containment cross-check.
 *   2. Graph replay stamps. ROCm 7.2 dispatch records have no graph node id.
 *      Following the sdk's own recipe (callback_tracing.h, HIP graph domain —
 *      which itself only exists after 7.2, so we build it from HIP API
 *      callbacks that do), hipGraphLaunch ENTER/EXIT maintains a per-thread
 *      (exec, ordinal) stack, and the request callback stamps dispatches and
 *      copies enqueued inside a launch with the next ordinal instead of a range.
 *   3. Capture-time projection. While a thread is stream-capturing, every
 *      launch/copy/memset API call it makes becomes a graph node; we emit a
 *      "graph_node" record naming the range open at that moment, with a
 *      signature (kernel_id, geometry, node kind) the decoder validates the
 *      replay against. hipStreamEndCapture + hipGraphInstantiate* link the
 *      capture to the executable ("graph_exec").
 *
 * Graph records are STRUCTURAL: the engine captures its graphs at startup, long
 * before any capture window arms, and nothing else can ever name a replayed
 * kernel. They are therefore written whether or not the window is armed, and
 * injection.read_shards exempts them from windowing. Everything else follows
 * the arming protocol: records are written only while $GITM_TRACE_OUT.arm
 * exists, checked at most once per ARM_CACHE_MS so marker callbacks (which
 * arrive one by one, not in buffers) don't stat() per range.
 *
 * Same durability model as the CUPTI side: JSONL streamed as buffers complete
 * plus a periodic flush thread (GITM_TRACE_FLUSH_MS, default 100), so a
 * SIGKILLed child loses at most one period. No signal handlers installed, for
 * the same reason as the CUPTI side: EngineCore owns its SIGTERM path.
 *
 * Verify on hardware before relying on graph identity (docs/rocm.md lists the
 * exact checks): that 7.2's request callback fires once per dispatch inside
 * hipGraphLaunch on the launching thread (the decoder refuses a replay whose
 * ordinals repeat, so a "no" degrades to range_op=None, never to a wrong op).
 */

/* rocprofiler.h first: the HIP id/arg headers below assume the HIP runtime
 * types it pulls in (via hip/api_args.h) are already declared. */
#include <rocprofiler-sdk/rocprofiler.h>
#include <rocprofiler-sdk/external_correlation.h>
#include <rocprofiler-sdk/hip/runtime_api_id.h>
#include <rocprofiler-sdk/marker/api_id.h> /* ROCPROFILER_MARKER_CORE_API_ID_* */
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
#define BUF_SIZE (32 * 1024 * 1024) /* 32 MiB, matches the CUPTI collector */
#define DEFAULT_FLUSH_MS 100
#define ARM_CACHE_MS 50
#define MARKER_STACK_MAX 128
#define GRAPH_STACK_MAX 8

/* GITM_MARKER_START/END from cupti_core.h — the JSONL contract, not CUPTI's. */
#define GITM_MARKER_START 0
#define GITM_MARKER_END 1

/* Graph-node id layout. Mirrors NODE_* in gitm/distributed/correlate.py, which
 * tests the two against each other (tests/test_rocm_collector_contract.py).
 *   bit 63      capture flag (node ids) / graph flag (external stamps)
 *   bits 32..62 exec or capture sequence number, from 1
 *   bits 0..31  ordinal + 1 (0 = unknown)
 * Ids are nonzero by construction: 0 means "not a graph launch" everywhere
 * downstream (CUPTI's convention). */
#define GITM_NODE_FLAG (1ULL << 63)
#define GITM_NODE_SEQ_BITS 31
#define GITM_NODE_ORD_BITS 32
#define GITM_NODE_SEQ_MASK ((1ULL << GITM_NODE_SEQ_BITS) - 1)
#define GITM_NODE_ORD_MASK ((1ULL << GITM_NODE_ORD_BITS) - 1)
/* exec seq for a hipGraphLaunch of an executable we never saw instantiated:
 * still a graph replay (so never given the launch's range as its op), but no
 * node can be named. */
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

static rocprofiler_context_id_t g_ctx = {0};    /* buffer services + stamps */
static rocprofiler_context_id_t g_cb_ctx = {0}; /* synchronous callbacks */
static rocprofiler_buffer_id_t g_buffer = {0};
static int g_nvtx = 0; /* GITM_TRACE_NVTX: identity collection on */

static pthread_t g_flusher;
static int g_flusher_started = 0;
static atomic_int g_stop_flusher = 0;
static uint32_t g_flush_ms = DEFAULT_FLUSH_MS;

static atomic_ullong g_stamp_overflow = 0;   /* ordinals past the id layout */
static atomic_ullong g_untracked_launch = 0; /* launches of unknown execs */

/* ---- agent handle -> logical GPU ordinal ------------------------------- */

#define MAX_AGENTS 64
static struct {
    uint64_t handle;
    uint32_t ordinal;
    char name[64];     /* gfx target, e.g. gfx950 */
    char product[128]; /* marketing name, e.g. AMD Instinct MI355X */
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
    (void)version;
    (void)user;
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
    for (int i = 0; i < g_n_agents; i++) {
        if (g_agents[i].handle == id.handle) return g_agents[i].ordinal;
    }
    return 0; /* CPU agent or unknown: same fallback the decoder tolerates */
}

/* ---- kernel_id -> {name, vgpr} map ------------------------------------- */

#define KMAP_BUCKETS 4096
typedef struct kmap_node {
    uint64_t kernel_id;
    uint32_t vgprs;
    struct kmap_node *next;
    char name[]; /* NUL-terminated, truncated to GITM_NAME_MAX */
} kmap_node;

static kmap_node *g_kmap[KMAP_BUCKETS];

static void kmap_put(uint64_t kernel_id, const char *name, uint32_t vgprs) {
    size_t len = name ? strlen(name) : 0;
    if (len > GITM_NAME_MAX) len = GITM_NAME_MAX;
    kmap_node *n = malloc(sizeof(kmap_node) + len + 1);
    if (!n) return; /* a dispatch of this kernel decodes as <anonymous> */
    n->kernel_id = kernel_id;
    n->vgprs = vgprs;
    memcpy(n->name, name ? name : "", len);
    n->name[len] = '\0';
    size_t b = kernel_id % KMAP_BUCKETS;
    n->next = g_kmap[b];
    g_kmap[b] = n; /* newest first: a reloaded id resolves to the new symbol */
}

static const kmap_node *kmap_get(uint64_t kernel_id) {
    for (kmap_node *n = g_kmap[kernel_id % KMAP_BUCKETS]; n; n = n->next) {
        if (n->kernel_id == kernel_id) return n;
    }
    return NULL;
}

/* ---- u64 -> u64 maps: host function -> kernel_id, graph -> capture,
 *      exec -> exec seq. Small, insert-mostly, guarded by g_lock. -------- */

#define UMAP_BUCKETS 1024
typedef struct umap_node {
    uint64_t key;
    uint64_t a, b;
    struct umap_node *next;
} umap_node;
typedef struct {
    umap_node *buckets[UMAP_BUCKETS];
} umap;

static umap g_hostfn;  /* host function address -> kernel_id */
static umap g_graphs;  /* hipGraph_t -> (capture seq, node count) */
static umap g_execs;   /* hipGraphExec_t -> exec seq */

static size_t umap_bucket(uint64_t key) {
    return (size_t)((key >> 4) ^ (key >> 20)) % UMAP_BUCKETS;
}

static void umap_put(umap *m, uint64_t key, uint64_t a, uint64_t b) {
    size_t i = umap_bucket(key);
    for (umap_node *n = m->buckets[i]; n; n = n->next) {
        if (n->key == key) { /* pointer reuse: the newest binding wins */
            n->a = a;
            n->b = b;
            return;
        }
    }
    umap_node *n = malloc(sizeof(umap_node));
    if (!n) return; /* lookups miss; the decoder refuses rather than guesses */
    n->key = key;
    n->a = a;
    n->b = b;
    n->next = m->buckets[i];
    m->buckets[i] = n;
}

static const umap_node *umap_get(const umap *m, uint64_t key) {
    for (umap_node *n = m->buckets[umap_bucket(key)]; n; n = n->next) {
        if (n->key == key) return n;
    }
    return NULL;
}

/* ---- shard writing (same JSON contract as cupti_inject.c) --------------- */

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

/* Refresh the armed flag, at most once per ARM_CACHE_MS. Called under g_lock.
 * Buffer callbacks land every few MiB so the cache barely matters there; it
 * exists for marker callbacks, which arrive per-range. */
static void refresh_armed(void) {
    uint64_t now = now_coarse_ms();
    if (now - g_armed_checked_ms < ARM_CACHE_MS) return;
    g_armed_checked_ms = now;
    g_armed = (access(g_arm_path, F_OK) == 0);
}

/* rocprofiler_memory_copy_operation_t -> CUpti_ActivityMemcpyKind ints, which
 * are what _cupti_decode._COPY_KIND speaks. Values from cupti_activity.h. */
static int copy_kind_cupti(rocprofiler_memory_copy_operation_t op) {
    switch (op) {
        case ROCPROFILER_MEMORY_COPY_HOST_TO_DEVICE:   return 1; /* HTOD */
        case ROCPROFILER_MEMORY_COPY_DEVICE_TO_HOST:   return 2; /* DTOH */
        case ROCPROFILER_MEMORY_COPY_DEVICE_TO_DEVICE: return 8; /* DTOD */
        case ROCPROFILER_MEMORY_COPY_HOST_TO_HOST:     return 9; /* HTOH */
        default:                                       return 0; /* UNKNOWN */
    }
}

/* Decode an external stamp into (range_id, graph_id, graph_node_id). */
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

/* ---- buffer callback: kernel dispatch / memory copy / HIP API ----------- */

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
    if (drop_count > 0) {
        /* The counter the CUPTI side never had: LOSSLESS should keep this at
         * zero, but if the sdk ever drops records, say so in-band so a lossy
         * trace is distinguishable from a quiet one. */
        fprintf(g_fp, "{\"kind\":\"meta\",\"dropped_records\":%llu}\n",
                (unsigned long long)drop_count);
    }

    for (size_t i = 0; i < num_headers; i++) {
        rocprofiler_record_header_t *h = headers[i];
        if (h->category != ROCPROFILER_BUFFER_CATEGORY_TRACING) continue;

        if (h->kind == ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH) {
            rocprofiler_buffer_tracing_kernel_dispatch_record_t *k = h->payload;
            const kmap_node *sym = kmap_get(k->dispatch_info.kernel_id);
            stamp_t st = decode_stamp(k->correlation_id.external.value);
            /* grid arrives in work-items; normalize to CUDA-style blocks. */
            uint32_t wx = k->dispatch_info.workgroup_size.x;
            uint32_t wy = k->dispatch_info.workgroup_size.y;
            uint32_t wz = k->dispatch_info.workgroup_size.z;
            uint32_t gx = wx ? k->dispatch_info.grid_size.x / wx : 0;
            uint32_t gy = wy ? k->dispatch_info.grid_size.y / wy : 0;
            uint32_t gz = wz ? k->dispatch_info.grid_size.z / wz : 0;
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
                    agent_ordinal(k->dispatch_info.agent_id),
                    (unsigned long long)k->dispatch_info.queue_id.handle,
                    (unsigned long long)k->correlation_id.internal,
                    gx, gy, gz, wx, wy, wz,
                    k->dispatch_info.group_segment_size,
                    sym ? sym->vgprs : 0,
                    (unsigned long long)k->dispatch_info.kernel_id,
                    (unsigned long long)k->thread_id,
                    (unsigned long long)st.range_id,
                    (unsigned long long)st.graph_id,
                    (unsigned long long)st.graph_node_id);
        } else if (h->kind == ROCPROFILER_BUFFER_TRACING_MEMORY_COPY) {
            rocprofiler_buffer_tracing_memory_copy_record_t *m = h->payload;
            stamp_t st = decode_stamp(m->correlation_id.external.value);
            /* device_id: the GPU endpoint — dst for H2D, src otherwise. */
            uint32_t dev =
                (m->operation == ROCPROFILER_MEMORY_COPY_HOST_TO_DEVICE)
                    ? agent_ordinal(m->dst_agent_id)
                    : agent_ordinal(m->src_agent_id);
            fprintf(g_fp,
                    "{\"kind\":\"memcpy\",\"copy_kind\":%d,\"bytes\":%llu,"
                    "\"start_ns\":%llu,\"end_ns\":%llu,\"device_id\":%u,"
                    "\"context_id\":0,\"stream_id\":0,\"correlation_id\":%llu,"
                    "\"thread_id\":%llu,\"range_id\":%llu,"
                    "\"graph_id\":%llu,\"graph_node_id\":%llu}\n",
                    copy_kind_cupti(m->operation),
                    (unsigned long long)m->bytes,
                    (unsigned long long)m->start_timestamp,
                    (unsigned long long)m->end_timestamp,
                    dev, (unsigned long long)m->correlation_id.internal,
                    (unsigned long long)m->thread_id,
                    (unsigned long long)st.range_id,
                    (unsigned long long)st.graph_id,
                    (unsigned long long)st.graph_node_id);
        } else if (h->kind == ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API) {
            /* The host-side launch record of the correlation chain — the
             * CUPTI RUNTIME/DRIVER analog. HIP has no runtime/driver split:
             * hipBLASLt and torch both launch through the HIP runtime, so one
             * service covers what needed two kinds on NVIDIA. graph_launch
             * marks a hipGraphLaunch, so a replayed kernel the ordinal stamp
             * missed is still known to be a replay and never takes this
             * record's range as its op. */
            rocprofiler_buffer_tracing_hip_api_record_t *a = h->payload;
            fprintf(g_fp,
                    "{\"kind\":\"runtime\",\"start_ns\":%llu,\"end_ns\":%llu,"
                    "\"correlation_id\":%llu,\"thread_id\":%llu,"
                    "\"range_id\":%llu,\"graph_launch\":%d}\n",
                    (unsigned long long)a->start_timestamp,
                    (unsigned long long)a->end_timestamp,
                    (unsigned long long)a->correlation_id.internal,
                    (unsigned long long)a->thread_id,
                    (unsigned long long)decode_stamp(
                        a->correlation_id.external.value).range_id,
                    is_graph_launch_op(a->operation));
        }
    }
    fflush(g_fp);
    pthread_mutex_unlock(&g_lock);
}

/* ---- per-thread state: rocTX range stack, capture, graph launches ------- */

static atomic_uint g_marker_seq = 1;
static atomic_ullong g_capture_seq = 0;
static atomic_ullong g_exec_seq = 0;

static __thread struct {
    uint32_t ids[MARKER_STACK_MAX];
    char *names[MARKER_STACK_MAX]; /* owned copies; the emitter's may die */
    int depth;
} tls_ranges;

static __thread struct {
    int active;
    uint64_t seq;
    uint64_t n; /* nodes recorded so far */
} tls_capture;

static __thread struct {
    uint64_t exec[GRAPH_STACK_MAX];
    uint64_t ordinal[GRAPH_STACK_MAX];
    int depth;
} tls_graph;

static uint32_t top_range_id(void) {
    int d = tls_ranges.depth;
    if (d <= 0 || d > MARKER_STACK_MAX) return 0;
    return tls_ranges.ids[d - 1];
}

static const char *top_range_name(void) {
    int d = tls_ranges.depth;
    if (d <= 0 || d > MARKER_STACK_MAX) return "";
    return tls_ranges.names[d - 1] ? tls_ranges.names[d - 1] : "";
}

/* ---- external correlation: the stamp ------------------------------------ */

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
            /* Still flagged as a replay, with no node: the decoder keeps it
             * off the launch range and counts it unresolved. */
            atomic_fetch_add(&g_stamp_overflow, 1);
            external->value = GITM_NODE_FLAG |
                              ((exec & GITM_NODE_SEQ_MASK) << GITM_NODE_ORD_BITS);
        } else {
            external->value = GITM_NODE_FLAG | node_id(exec, ord);
        }
        return 0;
    }
    external->value = top_range_id();
    return 0;
}

/* ---- structural records (written armed or not) -------------------------- */

static void emit_graph_node(uint64_t capture_seq, uint64_t ordinal,
                            const char *node_kind, uint64_t kernel_id,
                            int has_dims, uint32_t gx, uint32_t gy, uint32_t gz,
                            uint32_t bx, uint32_t by, uint32_t bz) {
    pthread_mutex_lock(&g_lock);
    if (g_fp) {
        fprintf(g_fp,
                "{\"kind\":\"graph_node\",\"graph_node_id\":%llu,"
                "\"capture_id\":%llu,\"node_kind\":\"%s\",\"kernel_id\":%llu,"
                "\"name\":",
                (unsigned long long)(GITM_NODE_FLAG | node_id(capture_seq, ordinal)),
                (unsigned long long)capture_seq, node_kind,
                (unsigned long long)kernel_id);
        write_json_string(g_fp, top_range_name());
        if (has_dims) {
            fprintf(g_fp, ",\"grid\":[%u,%u,%u],\"block\":[%u,%u,%u]", gx, gy,
                    gz, bx, by, bz);
        }
        fputs("}\n", g_fp);
    }
    pthread_mutex_unlock(&g_lock);
}

static void emit_graph_exec(uint64_t exec_seq, uint64_t capture_seq,
                            uint64_t n_nodes, int known) {
    pthread_mutex_lock(&g_lock);
    if (g_fp) {
        if (known) {
            fprintf(g_fp,
                    "{\"kind\":\"graph_exec\",\"graph_id\":%llu,"
                    "\"capture_id\":%llu,\"n_nodes\":%llu}\n",
                    (unsigned long long)exec_seq,
                    (unsigned long long)capture_seq,
                    (unsigned long long)n_nodes);
        } else {
            /* Built through the graph API rather than captured: no node is
             * named by a range, so no capture link and no node count. */
            fprintf(g_fp, "{\"kind\":\"graph_exec\",\"graph_id\":%llu,"
                          "\"capture_id\":0}\n",
                    (unsigned long long)exec_seq);
        }
        fflush(g_fp);
    }
    pthread_mutex_unlock(&g_lock);
}

/* ---- HIP API callbacks: capture and graph launch tracking ---------------- */

static uint64_t hostfn_kernel_id(const void *fn) {
    pthread_mutex_lock(&g_lock);
    const umap_node *n = umap_get(&g_hostfn, (uint64_t)(uintptr_t)fn);
    uint64_t id = n ? n->a : 0;
    pthread_mutex_unlock(&g_lock);
    return id;
}

static uint32_t div_or_zero(uint32_t a, uint32_t b) { return b ? a / b : 0; }

/* One captured launch/copy/memset -> one graph node. Geometry is recorded in
 * blocks, the unit the dispatch record is normalized to. */
static void capture_node(rocprofiler_tracing_operation_t op,
                         const rocprofiler_hip_api_args_t *a) {
    uint64_t ord = tls_capture.n++;
    uint64_t seq = tls_capture.seq;
    switch (op) {
        case ROCPROFILER_HIP_RUNTIME_API_ID_hipLaunchKernel:
            emit_graph_node(seq, ord, "kernel",
                            hostfn_kernel_id(a->hipLaunchKernel.function_address), 1,
                            a->hipLaunchKernel.numBlocks.x, a->hipLaunchKernel.numBlocks.y,
                            a->hipLaunchKernel.numBlocks.z, a->hipLaunchKernel.dimBlocks.x,
                            a->hipLaunchKernel.dimBlocks.y, a->hipLaunchKernel.dimBlocks.z);
            return;
        case ROCPROFILER_HIP_RUNTIME_API_ID_hipLaunchKernel_spt:
            emit_graph_node(seq, ord, "kernel",
                            hostfn_kernel_id(a->hipLaunchKernel_spt.function_address), 1,
                            a->hipLaunchKernel_spt.numBlocks.x,
                            a->hipLaunchKernel_spt.numBlocks.y,
                            a->hipLaunchKernel_spt.numBlocks.z,
                            a->hipLaunchKernel_spt.dimBlocks.x,
                            a->hipLaunchKernel_spt.dimBlocks.y,
                            a->hipLaunchKernel_spt.dimBlocks.z);
            return;
        case ROCPROFILER_HIP_RUNTIME_API_ID_hipExtLaunchKernel:
            emit_graph_node(seq, ord, "kernel",
                            hostfn_kernel_id(a->hipExtLaunchKernel.function_address), 1,
                            a->hipExtLaunchKernel.numBlocks.x,
                            a->hipExtLaunchKernel.numBlocks.y,
                            a->hipExtLaunchKernel.numBlocks.z,
                            a->hipExtLaunchKernel.dimBlocks.x,
                            a->hipExtLaunchKernel.dimBlocks.y,
                            a->hipExtLaunchKernel.dimBlocks.z);
            return;
        case ROCPROFILER_HIP_RUNTIME_API_ID_hipLaunchCooperativeKernel:
            emit_graph_node(seq, ord, "kernel",
                            hostfn_kernel_id(a->hipLaunchCooperativeKernel.func), 1,
                            a->hipLaunchCooperativeKernel.gridDim.x,
                            a->hipLaunchCooperativeKernel.gridDim.y,
                            a->hipLaunchCooperativeKernel.gridDim.z,
                            a->hipLaunchCooperativeKernel.blockDimX.x,
                            a->hipLaunchCooperativeKernel.blockDimX.y,
                            a->hipLaunchCooperativeKernel.blockDimX.z);
            return;
        case ROCPROFILER_HIP_RUNTIME_API_ID_hipLaunchKernelExC: {
            const hipLaunchConfig_t *c = a->hipLaunchKernelExC.config;
            if (c) {
                emit_graph_node(seq, ord, "kernel",
                                hostfn_kernel_id(a->hipLaunchKernelExC.fPtr), 1,
                                c->gridDim.x, c->gridDim.y, c->gridDim.z,
                                c->blockDim.x, c->blockDim.y, c->blockDim.z);
            } else {
                emit_graph_node(seq, ord, "kernel",
                                hostfn_kernel_id(a->hipLaunchKernelExC.fPtr), 0, 0,
                                0, 0, 0, 0, 0);
            }
            return;
        }
        /* Module launches (Triton, hipBLASLt/Tensile, AITER asm kernels) take
         * a hipFunction_t, which no code-object callback maps to a kernel_id:
         * these nodes are validated on geometry alone. */
        case ROCPROFILER_HIP_RUNTIME_API_ID_hipModuleLaunchKernel:
            emit_graph_node(seq, ord, "kernel", 0, 1,
                            a->hipModuleLaunchKernel.gridDimX,
                            a->hipModuleLaunchKernel.gridDimY,
                            a->hipModuleLaunchKernel.gridDimZ,
                            a->hipModuleLaunchKernel.blockDimX,
                            a->hipModuleLaunchKernel.blockDimY,
                            a->hipModuleLaunchKernel.blockDimZ);
            return;
        case ROCPROFILER_HIP_RUNTIME_API_ID_hipExtModuleLaunchKernel: {
            /* Global work size is in work-items (HSA), like the dispatch
             * record; divide exactly as buffer_cb does. */
            uint32_t lx = a->hipExtModuleLaunchKernel.localWorkSizeX;
            uint32_t ly = a->hipExtModuleLaunchKernel.localWorkSizeY;
            uint32_t lz = a->hipExtModuleLaunchKernel.localWorkSizeZ;
            emit_graph_node(seq, ord, "kernel", 0, 1,
                            div_or_zero(a->hipExtModuleLaunchKernel.globalWorkSizeX, lx),
                            div_or_zero(a->hipExtModuleLaunchKernel.globalWorkSizeY, ly),
                            div_or_zero(a->hipExtModuleLaunchKernel.globalWorkSizeZ, lz),
                            lx, ly, lz);
            return;
        }
        case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyAsync:
        case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyDtoDAsync:
        case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyDtoHAsync:
        case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyHtoDAsync:
        case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpy2DAsync:
        case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyPeerAsync:
            emit_graph_node(seq, ord, "memcpy", 0, 0, 0, 0, 0, 0, 0, 0);
            return;
        case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemsetAsync:
        case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemsetD8Async:
        case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemsetD16Async:
        case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemsetD32Async:
        case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemset2DAsync:
            emit_graph_node(seq, ord, "memset", 0, 0, 0, 0, 0, 0, 0, 0);
            return;
        default:
            tls_capture.n--; /* not a node-producing call */
            return;
    }
}

static int hip_ok(const rocprofiler_callback_tracing_hip_api_data_t *d) {
    return d->retval.hipError_t_retval == hipSuccess;
}

static void link_instantiate(hipGraphExec_t *pexec, hipGraph_t graph) {
    if (!pexec || !*pexec) return;
    uint64_t exec_seq = atomic_fetch_add(&g_exec_seq, 1) + 1;
    pthread_mutex_lock(&g_lock);
    const umap_node *cap = umap_get(&g_graphs, (uint64_t)(uintptr_t)graph);
    uint64_t cap_seq = cap ? cap->a : 0, n = cap ? cap->b : 0;
    umap_put(&g_execs, (uint64_t)(uintptr_t)*pexec, exec_seq, 0);
    pthread_mutex_unlock(&g_lock);
    emit_graph_exec(exec_seq, cap_seq, n, cap != NULL);
}

static void hip_api_cb(rocprofiler_callback_tracing_record_t record) {
    const rocprofiler_callback_tracing_hip_api_data_t *d = record.payload;
    const rocprofiler_hip_api_args_t *a = &d->args;
    rocprofiler_tracing_operation_t op = record.operation;

    if (record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER) {
        if (is_graph_launch_op(op)) {
            hipGraphExec_t exec = op == ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphLaunch
                                      ? a->hipGraphLaunch.graphExec
                                      : a->hipGraphLaunch_spt.graphExec;
            pthread_mutex_lock(&g_lock);
            const umap_node *e = umap_get(&g_execs, (uint64_t)(uintptr_t)exec);
            uint64_t seq = e ? e->a : GITM_EXEC_UNTRACKED;
            pthread_mutex_unlock(&g_lock);
            if (!e) atomic_fetch_add(&g_untracked_launch, 1);
            int dp = tls_graph.depth;
            if (dp < GRAPH_STACK_MAX) {
                tls_graph.exec[dp] = seq;
                tls_graph.ordinal[dp] = 0;
            }
            tls_graph.depth++;
        } else if (tls_capture.active) {
            capture_node(op, a);
        }
        return;
    }

    /* EXIT */
    if (is_graph_launch_op(op)) {
        if (tls_graph.depth > 0) tls_graph.depth--;
        return;
    }
    switch (op) {
        case ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamBeginCapture:
        case ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamBeginCaptureToGraph:
            if (hip_ok(d) && !tls_capture.active) {
                /* A per-thread flag, not a stream match: event-forked side
                 * streams are captured too, from this same thread. */
                tls_capture.active = 1;
                tls_capture.seq = atomic_fetch_add(&g_capture_seq, 1) + 1;
                tls_capture.n = 0;
            }
            return;
        case ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamEndCapture:
            if (tls_capture.active) {
                if (hip_ok(d) && a->hipStreamEndCapture.pGraph &&
                    *a->hipStreamEndCapture.pGraph) {
                    pthread_mutex_lock(&g_lock);
                    umap_put(&g_graphs,
                             (uint64_t)(uintptr_t)*a->hipStreamEndCapture.pGraph,
                             tls_capture.seq, tls_capture.n);
                    pthread_mutex_unlock(&g_lock);
                }
                tls_capture.active = 0;
            }
            return;
        case ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphInstantiate:
            if (hip_ok(d))
                link_instantiate(a->hipGraphInstantiate.pGraphExec,
                                 a->hipGraphInstantiate.graph);
            return;
        case ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphInstantiateWithFlags:
            if (hip_ok(d))
                link_instantiate(a->hipGraphInstantiateWithFlags.pGraphExec,
                                 a->hipGraphInstantiateWithFlags.graph);
            return;
        case ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphInstantiateWithParams:
            if (hip_ok(d))
                link_instantiate(a->hipGraphInstantiateWithParams.pGraphExec,
                                 a->hipGraphInstantiateWithParams.graph);
            return;
        default:
            return;
    }
}

/* The HIP API calls hip_api_cb needs. Filtering at configuration keeps the
 * per-call cost off every other API (hipGetDevice, hipStreamQuery, ...). */
static const rocprofiler_tracing_operation_t k_hip_ops[] = {
    ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamBeginCapture,
    ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamBeginCaptureToGraph,
    ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamEndCapture,
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

/* ---- callback tracing: code objects (names) + roctx markers ------------- */

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
    if (!out) return NULL; /* the range is named "" in capture nodes */
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
        if (record.operation ==
            ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER) {
            rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t
                *d = record.payload;
            pthread_mutex_lock(&g_lock);
            kmap_put(d->kernel_id, d->kernel_name, d->arch_vgpr_count);
            pthread_mutex_unlock(&g_lock);
        } else if (record.operation ==
                   ROCPROFILER_CODE_OBJECT_HOST_KERNEL_SYMBOL_REGISTER) {
            /* hipLaunchKernel names its kernel by host stub address; this is
             * the only map from that address to the kernel_id a dispatch
             * reports, which is what validates a replayed kernel against the
             * node it was captured as. */
            rocprofiler_callback_tracing_code_object_host_kernel_symbol_register_data_t
                *d = record.payload;
            pthread_mutex_lock(&g_lock);
            umap_put(&g_hostfn, d->host_function.value, d->kernel_id, 0);
            pthread_mutex_unlock(&g_lock);
        }
        return;
    }

    if (record.kind != ROCPROFILER_CALLBACK_TRACING_MARKER_CORE_API) return;
    if (record.phase != ROCPROFILER_CALLBACK_PHASE_ENTER) return;
    rocprofiler_callback_tracing_marker_api_data_t *d = record.payload;

    if (record.operation == ROCPROFILER_MARKER_CORE_API_ID_roctxRangePushA) {
        uint32_t id = atomic_fetch_add(&g_marker_seq, 1);
        if (tls_ranges.depth < MARKER_STACK_MAX) {
            tls_ranges.ids[tls_ranges.depth] = id;
            tls_ranges.names[tls_ranges.depth] =
                dup_bounded(d->args.roctxRangePushA.message);
        }
        /* Past MARKER_STACK_MAX the push is emitted but not tracked; its pop
         * pairs with nothing and pair_markers drops the half, which is the
         * documented behavior for unpaired halves. 128 deep is ~4x anything
         * vLLM layerwise instrumentation produces. */
        tls_ranges.depth++;
        emit_marker(d->args.roctxRangePushA.message, id, GITM_MARKER_START,
                    record.thread_id);
    } else if (record.operation == ROCPROFILER_MARKER_CORE_API_ID_roctxRangePop) {
        if (tls_ranges.depth <= 0) return; /* pop with no push: ignore */
        tls_ranges.depth--;
        if (tls_ranges.depth >= MARKER_STACK_MAX) return; /* untracked push */
        free(tls_ranges.names[tls_ranges.depth]);
        tls_ranges.names[tls_ranges.depth] = NULL;
        emit_marker(NULL, tls_ranges.ids[tls_ranges.depth], GITM_MARKER_END,
                    record.thread_id);
    } else if (record.operation == ROCPROFILER_MARKER_CORE_API_ID_roctxMarkA) {
        /* A point marker: emit both halves at one timestamp so it decodes as
         * a zero-length range rather than an unpaired half. */
        uint32_t id = atomic_fetch_add(&g_marker_seq, 1);
        emit_marker(d->args.roctxMarkA.message, id, GITM_MARKER_START,
                    record.thread_id);
        emit_marker(NULL, id, GITM_MARKER_END, record.thread_id);
    }
}

/* ---- flush thread ------------------------------------------------------- */

static void emit_stamp_faults(void) {
    unsigned long long over = atomic_exchange(&g_stamp_overflow, 0);
    unsigned long long untracked = atomic_exchange(&g_untracked_launch, 0);
    if (!over && !untracked) return;
    pthread_mutex_lock(&g_lock);
    if (g_fp) {
        fprintf(g_fp,
                "{\"kind\":\"meta\",\"graph_stamp_overflow\":%llu,"
                "\"graph_untracked_launch\":%llu}\n",
                over, untracked);
    }
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

/* ---- tool lifecycle ----------------------------------------------------- */

/* In-band provenance, written once at init whether or not armed: which
 * collector produced this shard and on which GPUs. gitm.tracer.vendor reads it
 * as the strongest evidence of a trace's vendor. */
static void emit_collector_meta(void) {
    pthread_mutex_lock(&g_lock);
    if (g_fp) {
        fprintf(g_fp,
                "{\"kind\":\"meta\",\"collector\":\"rocprofiler-sdk\","
                "\"sdk_version\":\"%d.%d.%d\",\"identity\":%d,\"agents\":[",
                ROCPROFILER_VERSION_MAJOR, ROCPROFILER_VERSION_MINOR,
                ROCPROFILER_VERSION_PATCH, g_nvtx);
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

    rocprofiler_query_available_agents(
        ROCPROFILER_AGENT_INFO_VERSION_0, agent_iter,
        sizeof(rocprofiler_agent_v0_t), NULL);
    emit_collector_meta();

    if (rocprofiler_create_context(&g_ctx) != ROCPROFILER_STATUS_SUCCESS)
        return -1;
    if (rocprofiler_create_context(&g_cb_ctx) != ROCPROFILER_STATUS_SUCCESS)
        return -1;
    if (rocprofiler_create_buffer(g_ctx, BUF_SIZE, BUF_SIZE / 2,
                                  ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                  buffer_cb, NULL,
                                  &g_buffer) != ROCPROFILER_STATUS_SUCCESS)
        return -1;

    rocprofiler_configure_buffer_tracing_service(
        g_ctx, ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH, NULL, 0, g_buffer);
    rocprofiler_configure_buffer_tracing_service(
        g_ctx, ROCPROFILER_BUFFER_TRACING_MEMORY_COPY, NULL, 0, g_buffer);
    /* Kernel names are useless without this: dispatch records carry only ids.
     * Host symbols ride along for the capture-time kernel_id. */
    rocprofiler_configure_callback_tracing_service(
        g_cb_ctx, ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT, NULL, 0,
        callback_cb, NULL);

    if (g_nvtx) {
        /* Correlation records, gated exactly like the CUPTI side: HIP API
         * records are a multiple of the kernel count on a decode step, and
         * that cost is what the with/without comparison measures. */
        rocprofiler_configure_buffer_tracing_service(
            g_ctx, ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API, NULL, 0,
            g_buffer);
        rocprofiler_configure_callback_tracing_service(
            g_cb_ctx, ROCPROFILER_CALLBACK_TRACING_MARKER_CORE_API, NULL, 0,
            callback_cb, NULL);
        /* Capture + graph-launch tracking, synchronous on the calling thread,
         * on its own context so it never competes with the buffered HIP API
         * service above for the same domain. */
        rocprofiler_configure_callback_tracing_service(
            g_cb_ctx, ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API, k_hip_ops,
            sizeof(k_hip_ops) / sizeof(k_hip_ops[0]), callback_cb, NULL);
        /* The stamp. Only records of g_ctx get one, which is all we emit. */
        static const rocprofiler_external_correlation_id_request_kind_t kinds[] = {
            ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_KERNEL_DISPATCH,
            ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_MEMORY_COPY,
            ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_HIP_RUNTIME_API,
        };
        rocprofiler_configure_external_correlation_id_request_service(
            g_ctx, kinds, sizeof(kinds) / sizeof(kinds[0]), stamp_cb, NULL);
    }

    /* Deliver buffer callbacks on our own thread, not the runtime's. */
    rocprofiler_callback_thread_t cb_thread;
    if (rocprofiler_create_callback_thread(&cb_thread) ==
        ROCPROFILER_STATUS_SUCCESS) {
        rocprofiler_assign_callback_thread(g_buffer, cb_thread);
    }

    if (rocprofiler_start_context(g_cb_ctx) != ROCPROFILER_STATUS_SUCCESS)
        return -1;
    if (rocprofiler_start_context(g_ctx) != ROCPROFILER_STATUS_SUCCESS)
        return -1;

    const char *ms = getenv("GITM_TRACE_FLUSH_MS");
    if (ms && *ms) g_flush_ms = (uint32_t)strtoul(ms, NULL, 10);
    if (g_flush_ms == 0) g_flush_ms = DEFAULT_FLUSH_MS;
    if (pthread_create(&g_flusher, NULL, flusher_main, NULL) == 0)
        g_flusher_started = 1;
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
    rocprofiler_flush_buffer(g_buffer); /* drains into buffer_cb -> g_fp */
    emit_stamp_faults();
    pthread_mutex_lock(&g_lock);
    if (g_fp) {
        fflush(g_fp);
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

    /* Same dormancy contract as InitializeInjection(): a tracer that cannot
     * open its output has no business degrading the workload it rode into.
     * Returning NULL declines registration and the process runs untraced. */
    const char *out = getenv("GITM_TRACE_OUT");
    if (!out || !*out) return NULL;

    char shard[PATH_MAX];
    if (snprintf(shard, sizeof(shard), "%s.%d", out, (int)getpid()) >=
        (int)sizeof(shard))
        return NULL;
    if (snprintf(g_arm_path, sizeof(g_arm_path), "%s.arm", out) >=
        (int)sizeof(g_arm_path))
        return NULL;

    g_fp = fopen(shard, "a");
    if (!g_fp) return NULL;
    setvbuf(g_fp, NULL, _IOFBF, 1 << 20);

    const char *nvtx = getenv("GITM_TRACE_NVTX");
    g_nvtx = nvtx && *nvtx && strcmp(nvtx, "0") != 0;

    static rocprofiler_tool_configure_result_t result = {
        sizeof(rocprofiler_tool_configure_result_t), tool_init, tool_fini,
        NULL};
    return &result;
}
