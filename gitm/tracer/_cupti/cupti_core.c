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