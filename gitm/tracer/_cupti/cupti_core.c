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
#include <stdlib.h>
#include <string.h>

#define BUF_SIZE (8 * 1024 * 1024)   /* 8 MiB activity buffers */
#define BUF_ALIGN 8

#define ALIGN_BUFFER(p, a) \
    (((uintptr_t)(p) % (a)) ? ((p) + (a) - ((uintptr_t)(p) % (a))) : (p))

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

static void CUPTIAPI buffer_requested(uint8_t **buffer, size_t *size,
                                      size_t *maxNumRecords) {
    uint8_t *raw = (uint8_t *)malloc(BUF_SIZE + BUF_ALIGN);
    *buffer = (uint8_t *)ALIGN_BUFFER(raw, BUF_ALIGN);
    *size = BUF_SIZE;
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
    free(buffer);  /* matches malloc in buffer_requested (aligned within) */
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