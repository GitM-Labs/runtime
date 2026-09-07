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
 *     delivered by the MARKER_CORE_API service of this same tool — the
 *     two-mechanism split that bit us on the B200 does not exist here.
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
 * Same arming protocol as the CUPTI side: records are written only while
 * $GITM_TRACE_OUT.arm exists, checked at most once per ARM_CACHE_MS so marker
 * callbacks (which arrive one by one, not in buffers) don't stat() per range.
 * Same durability model: JSONL streamed as buffers complete plus a periodic
 * flush thread (GITM_TRACE_FLUSH_MS, default 100), so a SIGKILLed child loses
 * at most one period. No signal handlers installed, for the same reason as the
 * CUPTI side: EngineCore owns its SIGTERM path.
 */

#include <rocprofiler-sdk/marker/api_id.h> /* ROCPROFILER_MARKER_CORE_API_ID_* */
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

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

/* GITM_MARKER_START/END from cupti_core.h — the JSONL contract, not CUPTI's. */
#define GITM_MARKER_START 0
#define GITM_MARKER_END 1

static FILE *g_fp = NULL;
static char g_arm_path[PATH_MAX];
static int g_armed = 0;
static uint64_t g_armed_checked_ms = 0;
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;

static rocprofiler_context_id_t g_ctx = {0};
static rocprofiler_buffer_id_t g_buffer = {0};
static int g_nvtx = 0; /* GITM_TRACE_NVTX: HIP-API + marker collection on */

static pthread_t g_flusher;
static int g_flusher_started = 0;
static atomic_int g_stop_flusher = 0;
static uint32_t g_flush_ms = DEFAULT_FLUSH_MS;

/* ---- agent handle -> logical GPU ordinal ------------------------------- */

#define MAX_AGENTS 64
static struct {
    uint64_t handle;
    uint32_t ordinal;
} g_agents[MAX_AGENTS];
static int g_n_agents = 0;

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
                    "\"registers_per_thread\":%u}\n",
                    (unsigned long long)k->start_timestamp,
                    (unsigned long long)k->end_timestamp,
                    agent_ordinal(k->dispatch_info.agent_id),
                    (unsigned long long)k->dispatch_info.queue_id.handle,
                    (unsigned long long)k->correlation_id.internal,
                    gx, gy, gz, wx, wy, wz,
                    k->dispatch_info.group_segment_size,
                    sym ? sym->vgprs : 0);
        } else if (h->kind == ROCPROFILER_BUFFER_TRACING_MEMORY_COPY) {
            rocprofiler_buffer_tracing_memory_copy_record_t *m = h->payload;
            /* device_id: the GPU endpoint — dst for H2D, src otherwise. */
            uint32_t dev =
                (m->operation == ROCPROFILER_MEMORY_COPY_HOST_TO_DEVICE)
                    ? agent_ordinal(m->dst_agent_id)
                    : agent_ordinal(m->src_agent_id);
            fprintf(g_fp,
                    "{\"kind\":\"memcpy\",\"copy_kind\":%d,\"bytes\":%llu,"
                    "\"start_ns\":%llu,\"end_ns\":%llu,\"device_id\":%u,"
                    "\"context_id\":0,\"stream_id\":0,\"correlation_id\":%llu}\n",
                    copy_kind_cupti(m->operation),
                    (unsigned long long)m->bytes,
                    (unsigned long long)m->start_timestamp,
                    (unsigned long long)m->end_timestamp,
                    dev, (unsigned long long)m->correlation_id.internal);
        } else if (h->kind == ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API) {
            /* The host-side launch record of the correlation chain — the
             * CUPTI RUNTIME/DRIVER analog. HIP has no runtime/driver split:
             * hipBLASLt and torch both launch through the HIP runtime, so one
             * service covers what needed two kinds on NVIDIA. */
            rocprofiler_buffer_tracing_hip_api_record_t *a = h->payload;
            fprintf(g_fp,
                    "{\"kind\":\"runtime\",\"start_ns\":%llu,\"end_ns\":%llu,"
                    "\"correlation_id\":%llu,\"thread_id\":%llu}\n",
                    (unsigned long long)a->start_timestamp,
                    (unsigned long long)a->end_timestamp,
                    (unsigned long long)a->correlation_id.internal,
                    (unsigned long long)a->thread_id);
        }
    }
    fflush(g_fp);
    pthread_mutex_unlock(&g_lock);
}

/* ---- callback tracing: code objects (names) + roctx markers ------------- */

static atomic_uint g_marker_seq = 1;
static __thread struct {
    uint32_t ids[MARKER_STACK_MAX];
    int depth;
} tls_ranges;

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

static void
callback_cb(rocprofiler_callback_tracing_record_t record,
            rocprofiler_user_data_t *user_data, void *callback_data) {
    (void)user_data; (void)callback_data;

    if (record.kind == ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT) {
        if (record.operation !=
            ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER)
            return;
        if (record.phase != ROCPROFILER_CALLBACK_PHASE_LOAD) return;
        rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t
            *d = record.payload;
        pthread_mutex_lock(&g_lock);
        kmap_put(d->kernel_id, d->kernel_name, d->arch_vgpr_count);
        pthread_mutex_unlock(&g_lock);
        return;
    }

    if (record.kind != ROCPROFILER_CALLBACK_TRACING_MARKER_CORE_API) return;
    if (record.phase != ROCPROFILER_CALLBACK_PHASE_ENTER) return;
    rocprofiler_callback_tracing_marker_api_data_t *d = record.payload;

    if (record.operation == ROCPROFILER_MARKER_CORE_API_ID_roctxRangePushA) {
        uint32_t id = atomic_fetch_add(&g_marker_seq, 1);
        if (tls_ranges.depth < MARKER_STACK_MAX) {
            tls_ranges.ids[tls_ranges.depth] = id;
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

static void *flusher_main(void *arg) {
    (void)arg;
    while (!atomic_load(&g_stop_flusher)) {
        usleep((useconds_t)g_flush_ms * 1000u);
        rocprofiler_flush_buffer(g_buffer);
    }
    return NULL;
}

/* ---- tool lifecycle ----------------------------------------------------- */

static int tool_init(rocprofiler_client_finalize_t fini, void *tool_data) {
    (void)fini; (void)tool_data;

    rocprofiler_query_available_agents(
        ROCPROFILER_AGENT_INFO_VERSION_0, agent_iter,
        sizeof(rocprofiler_agent_v0_t), NULL);

    if (rocprofiler_create_context(&g_ctx) != ROCPROFILER_STATUS_SUCCESS)
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
    /* Kernel names are useless without this: dispatch records carry only ids. */
    rocprofiler_configure_callback_tracing_service(
        g_ctx, ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT, NULL, 0,
        callback_cb, NULL);

    if (g_nvtx) {
        /* Correlation records, gated exactly like the CUPTI side: HIP API
         * records are a multiple of the kernel count on a decode step, and
         * that cost is what the with/without comparison measures. */
        rocprofiler_configure_buffer_tracing_service(
            g_ctx, ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API, NULL, 0,
            g_buffer);
        rocprofiler_configure_callback_tracing_service(
            g_ctx, ROCPROFILER_CALLBACK_TRACING_MARKER_CORE_API, NULL, 0,
            callback_cb, NULL);
    }

    /* Deliver buffer callbacks on our own thread, not the runtime's. */
    rocprofiler_callback_thread_t cb_thread;
    if (rocprofiler_create_callback_thread(&cb_thread) ==
        ROCPROFILER_STATUS_SUCCESS) {
        rocprofiler_assign_callback_thread(g_buffer, cb_thread);
    }

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
    rocprofiler_flush_buffer(g_buffer); /* drains into buffer_cb -> g_fp */
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
