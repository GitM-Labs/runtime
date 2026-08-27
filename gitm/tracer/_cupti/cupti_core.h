/*
 * cupti_core — CUPTI Activity ingestion, with no opinion about where records go.
 *
 * Shared by the two build targets that need it:
 *
 *   cupti_shim.c    -> _cupti_shim<EXT>.so   CPython extension; sink appends to
 *                                            an in-memory list, marshalled on stop().
 *   cupti_inject.c  -> libgitm_inject.so     plain .so with no libpython, loaded
 *                                            into EVERY CUDA process by the driver
 *                                            via $CUDA_INJECTION64_PATH; sink writes
 *                                            JSONL straight to a per-pid file.
 *
 * The injection target is what lets us trace vLLM: its V1 engine runs the model in
 * a child process (EngineCore), so a tracer that only ever loads in the parent
 * interpreter sees zero kernels. The driver dlopens this library in the child at
 * CUDA init and calls InitializeInjection(), which is the only way in without
 * making vLLM disable its own process model.
 *
 * Everything unsafe lives here: buffer management, walking records with
 * cuptiActivityGetNextRecord, and reading fields off the real CUPTI structs whose
 * layout the compiler resolves against the installed cupti_activity.h. Callers only
 * ever see the normalized gitm_record below.
 *
 * Threading: CUPTI invokes buffer_completed on its own internal threads. The core
 * holds a mutex across a whole buffer and calls the sink under it, so sinks may be
 * written as if single-threaded — but they must not call back into Python without
 * taking the GIL, and they must not block.
 */

#ifndef GITM_CUPTI_CORE_H
#define GITM_CUPTI_CORE_H

#include <cupti.h>
#include <stdint.h>

/* CuTeDSL and CUTLASS encode tensor layouts into the symbol, so a single 
 * FlashInfer GDN or grouped-GEMM instantiation runs hundred characters and two
 * variants differ only in a late template argument are identical under a short cap
 * they merge into one identity in KernelBreakdown and their time is pooled.
 */
#define GITM_NAME_MAX 1023

#define GITM_REC_KERNEL 0
#define GITM_REC_MEMCPY 1
#define GITM_REC_SYNC 2
/* The correlation chain. A kernel carries only a correlation_id; recovering the
 * op and layer it belongs to needs the host-side launch that issued it (RUNTIME)
 * and the innermost NVTX range enclosing that launch (MARKER). See
 * gitm/distributed/correlate.py for the three-record contract.
 *
 * Both are OFF unless GITM_TRACE_NVTX is set: RUNTIME emits one record per CUDA
 * API call, which on a decode step is a multiple of the kernel count, and the
 * pair roughly triples buffer pressure. Correlation is worth that; ordinary
 * capture is not. */
#define GITM_REC_RUNTIME 3
#define GITM_REC_MARKER 4

/* CUpti_ActivityMarker2 arrives as two records per range — a START and an END
 * sharing an `id`. Pairing them needs state; the C side stays stateless and
 * emits both, and _cupti_decode.py joins them on marker_id. */
#define GITM_MARKER_START 0
#define GITM_MARKER_END 1

/** Normalized record - only the fields GITM consumes.
 * Field-for-field contract documented in gitm/tracer/_cupti_decode.py
 * both sinks emit exactly these keys so one decoder serves both targets. */
typedef struct {
    int      kind;
    char     name[GITM_NAME_MAX + 1];
    uint64_t start_ns;
    uint64_t end_ns;
    uint32_t device_id;
    uint32_t context_id;
    uint32_t stream_id;
    uint32_t correlation_id;
    int32_t  grid[3];
    int32_t  block[3];
    int32_t  static_shared_mem;
    int32_t  dynamic_shared_mem;
    int32_t  registers_per_thread;
    int      copy_kind;
    uint64_t bytes;
    int      sync_type;
    /* Correlation. `thread_id` is the host thread that made the call, and is the
     * secondary key range containment is matched on — a kernel launched from one
     * thread must never be attributed to a range pushed on another. Both fields
     * are zero on kernel/memcpy/sync records. */
    uint32_t thread_id;
    uint32_t marker_id;
    int      marker_flags;
} gitm_record;

/** Whether RUNTIME/MARKER collection is on (GITM_TRACE_NVTX). Exposed so both
 * sinks and the build can report the mode a capture actually ran in — a trace
 * without correlation records looks identical to one whose ranges never fired. */
int gitm_nvtx_enabled(void);

/** Called once per decoded record, under the core's lock. */
typedef void (*gitm_sink_fn)(const gitm_record *rec, void *user);

/* Called once at the top of each completed CUPTI buffer, under the core's lock,
 * before any records from that buffer are sunk. The injection sink uses this to
 * refresh its armed/disarmed state with one stat() per buffer instead of one per
 * kernel. */
typedef void (*gitm_buffer_fn)(void *user);

void gitm_set_sink(gitm_sink_fn sink, void *user);
void gitm_set_buffer_hook(gitm_buffer_fn hook, void *user);

/* Register callbacks and enable the activity kinds GITM models. Idempotent. */
CUptiResult gitm_cupti_start(void);

/* Disable the activity kinds and force a final flush. Idempotent. */
CUptiResult gitm_cupti_stop(void);

/* Force a flush of all pending buffers (records arrive via the sink). */
CUptiResult gitm_cupti_flush(void);

/* Ask CUPTI to flush completed buffers every ms milliseconds.
 *
 * Load-bearing for injection: the parent process closes its capture window while
 * the child still holds partially-filled buffers, and the parent cannot reach into
 * the child to force a flush. A periodic flush bounds how much of the tail is
 * still in flight when the window closes; capture.py then waits out one period
 * before merging. */
CUptiResult gitm_cupti_set_flush_period(uint32_t ms);

/* A timestamp in the SAME clock domain as the activity records' start/end.
 *
 * This is what makes cross-process merging correct. CUPTI normalizes activity
 * timestamps to one driver-global clock, so a bound read in the parent is directly
 * comparable to a kernel timestamp recorded in a child — unlike wall-clock, which
 * is per-process. capture() reads this at window open/close and filters merged
 * shards to the window, which is how the 80s of engine build and CUDA-graph capture
 * that precede the decode stay out of the trace. */
uint64_t gitm_cupti_timestamp(void);

int gitm_cuda_device_count(void);
const char *gitm_cupti_errstr(CUptiResult status);

#endif /* GITM_CUPTI_CORE_H */