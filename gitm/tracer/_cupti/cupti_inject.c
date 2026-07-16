/*
 * cupti_inject — CUPTI collection for processes we don't control.
 *
 * Built as libgitm_inject.so: a plain shared library with NO libpython. Point the
 * CUDA driver at it with
 *
 *     export CUDA_INJECTION64_PATH=/abs/path/to/libgitm_inject.so
 *     export GITM_TRACE_OUT=/abs/path/to/trace.jsonl
 *
 * and the driver dlopens it inside every process that initializes CUDA — the
 * parent AND any child it forks or spawns — calling InitializeInjection() before
 * that process runs a single kernel. Because CUDA_INJECTION64_PATH is an ordinary
 * environment variable it is inherited across fork/spawn, which is the whole point:
 * vLLM's V1 engine runs the model in a separate EngineCore process, so a tracer
 * that only loads in the parent interpreter captures nothing. This is the same
 * mechanism nsys uses to follow children, and it needs no cooperation from vLLM —
 * its process model is untouched.
 *
 * Each process writes its own shard, $GITM_TRACE_OUT.<pid>, one JSON record per
 * line in exactly the dict shape gitm/tracer/_cupti_decode.py already consumes.
 * capture() merges the shards. Per-pid files because parent and child both load
 * this library and appending both to one file would interleave partial lines.
 *
 * Durability: records are written from the sink as CUPTI hands buffers back,
 * rather than accumulated and dumped at exit. A child that is SIGKILLed — or that
 * dies in its own shutdown path, which vLLM's EngineCore has been observed to do —
 * still leaves everything up to the last flush on disk. atexit() catches the tail
 * on a clean exit. We deliberately install NO signal handler: EngineCore installs
 * its own SIGTERM handler for orderly shutdown, and clobbering it from a library
 * the driver injected underneath the application would be a hostile thing to do.
 *
 * Arming: capture() creates $GITM_TRACE_OUT.arm on window open and removes it on
 * close, and this library only writes while it exists. Without that gate every
 * CUDA process on the box would stream records for its entire lifetime — for vLLM
 * that means ~80s of weight load, torch.compile and CUDA-graph capture before the
 * decode even starts, and an unbounded file on a 24h run. The check is one stat()
 * per completed buffer (8 MiB), not per kernel.
 */

#include "cupti_core.h"

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

#define DEFAULT_FLUSH_MS 100

static FILE *g_fp = NULL;
static char  g_arm_path[PATH_MAX];
static int   g_armed = 0;

/* JSON-escape a kernel name. Mangled C++ template names routinely carry quotes,
 * backslashes and control bytes; an unescaped one would corrupt the shard and take
 * the whole trace down at parse time. */
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

/* One stat() per completed buffer, refreshing the armed flag for the records that
 * buffer is about to yield. */
static void refresh_armed(void *user) {
    (void)user;
    g_armed = (access(g_arm_path, F_OK) == 0);
}

static void file_sink(const gitm_record *r, void *user) {
    (void)user;
    if (!g_fp || !g_armed) return;

    if (r->kind == GITM_REC_KERNEL) {
        fputs("{\"kind\":\"kernel\",\"name\":", g_fp);
        write_json_string(g_fp, r->name);
        fprintf(g_fp,
                ",\"start_ns\":%llu,\"end_ns\":%llu,\"device_id\":%u,\"context_id\":%u,"
                "\"stream_id\":%u,\"correlation_id\":%u,\"grid\":[%d,%d,%d],"
                "\"block\":[%d,%d,%d],\"static_shared_mem\":%d,"
                "\"dynamic_shared_mem\":%d,\"registers_per_thread\":%d}\n",
                (unsigned long long)r->start_ns, (unsigned long long)r->end_ns,
                r->device_id, r->context_id, r->stream_id, r->correlation_id,
                r->grid[0], r->grid[1], r->grid[2],
                r->block[0], r->block[1], r->block[2],
                r->static_shared_mem, r->dynamic_shared_mem, r->registers_per_thread);
    } else if (r->kind == GITM_REC_MEMCPY) {
        fprintf(g_fp,
                "{\"kind\":\"memcpy\",\"copy_kind\":%d,\"bytes\":%llu,"
                "\"start_ns\":%llu,\"end_ns\":%llu,\"device_id\":%u,\"context_id\":%u,"
                "\"stream_id\":%u,\"correlation_id\":%u}\n",
                r->copy_kind, (unsigned long long)r->bytes,
                (unsigned long long)r->start_ns, (unsigned long long)r->end_ns,
                r->device_id, r->context_id, r->stream_id, r->correlation_id);
    } else {
        fprintf(g_fp,
                "{\"kind\":\"sync\",\"sync_type\":%d,"
                "\"start_ns\":%llu,\"end_ns\":%llu,\"device_id\":%u,\"context_id\":%u,"
                "\"stream_id\":%u,\"correlation_id\":%u}\n",
                r->sync_type,
                (unsigned long long)r->start_ns, (unsigned long long)r->end_ns,
                r->device_id, r->context_id, r->stream_id, r->correlation_id);
    }
}

static void finish(void) {
    if (!g_fp) return;
    gitm_cupti_stop();   /* disable + force flush; the sink drains into g_fp */
    fflush(g_fp);
    fclose(g_fp);
    g_fp = NULL;
}

/* The driver's entry point. Must return non-zero; returning 0 tells CUDA the
 * injection failed. We always report success — a tracer that cannot open its
 * output has no business taking down the workload it was injected into, so on any
 * error we simply stay dormant. */
int InitializeInjection(void) {
    static int initialized = 0;
    if (initialized) return 1;
    initialized = 1;

    const char *out = getenv("GITM_TRACE_OUT");
    if (!out || !*out) return 1;  /* not a GITM run — stay dormant */

    char shard[PATH_MAX];
    if (snprintf(shard, sizeof(shard), "%s.%d", out, (int)getpid()) >= (int)sizeof(shard)) {
        return 1;
    }
    if (snprintf(g_arm_path, sizeof(g_arm_path), "%s.arm", out) >= (int)sizeof(g_arm_path)) {
        return 1;
    }

    g_fp = fopen(shard, "a");
    if (!g_fp) return 1;
    setvbuf(g_fp, NULL, _IOFBF, 1 << 20);

    gitm_set_sink(file_sink, NULL);
    gitm_set_buffer_hook(refresh_armed, NULL);

    if (gitm_cupti_start() != CUPTI_SUCCESS) {
        fclose(g_fp);
        g_fp = NULL;
        return 1;
    }

    const char *ms = getenv("GITM_TRACE_FLUSH_MS");
    gitm_cupti_set_flush_period(ms && *ms ? (uint32_t)strtoul(ms, NULL, 10)
                                          : (uint32_t)DEFAULT_FLUSH_MS);

    atexit(finish);
    return 1;
}