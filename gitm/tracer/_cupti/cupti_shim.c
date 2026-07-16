/*
 * cupti_shim — the CPython face of cupti_core.
 *
 * Why a C shim instead of ctypes: the CUPTI activity record structs
 * (CUpti_ActivityKernel*, CUpti_ActivityMemcpy*, ...) change layout between
 * CUDA versions. Letting the compiler read the installed cupti_activity.h
 * means field offsets are always correct — never hand-mirrored in Python where
 * a wrong pad would silently corrupt every kernel timing. The core copies only
 * primitives into a normalized C record; this file turns those into dicts, and
 * gitm.tracer._cupti_decode turns the dicts into validated trace events.
 *
 * Threading: CUPTI may invoke the buffer-completed callback on its own internal
 * threads during cuptiActivityFlushAll. We therefore do NO Python work in the
 * sink — records are appended to a C array the core already guards with its
 * mutex — and only build Python objects in stop(), on the calling thread,
 * holding the GIL.
 *
 * This target traces only the process it is imported into. For vLLM, whose V1
 * engine runs the model in a child process, use the injection target instead
 * (cupti_inject.c). See cupti_core.h.
 *
 * Build: python -m gitm.tracer._cupti.build  (needs CUPTI headers + libcupti).
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "cupti_core.h"

#include <stdlib.h>
#include <string.h>

/* Growable record store, appended from the core's sink (called under the core's
 * lock, so no additional locking here). */
static gitm_record *g_records = NULL;
static size_t g_count = 0;
static size_t g_cap = 0;

static void store_sink(const gitm_record *rec, void *user) {
    (void)user;
    if (g_count == g_cap) {
        size_t ncap = g_cap ? g_cap * 2 : 4096;
        gitm_record *n = (gitm_record *)realloc(g_records, ncap * sizeof(gitm_record));
        if (!n) return;  /* drop the record rather than abort the workload */
        g_records = n;
        g_cap = ncap;
    }
    g_records[g_count++] = *rec;
}

static PyObject *set_cupti_error(CUptiResult st, const char *where) {
    PyErr_Format(PyExc_RuntimeError, "CUPTI %s failed: %s", where, gitm_cupti_errstr(st));
    return NULL;
}

static PyObject *py_start(PyObject *self, PyObject *args) {
    (void)self; (void)args;
    g_count = 0;  /* fresh trace */
    gitm_set_sink(store_sink, NULL);
    CUptiResult st = gitm_cupti_start();
    if (st != CUPTI_SUCCESS) return set_cupti_error(st, "start");
    Py_RETURN_NONE;
}

static PyObject *rec_to_dict(const gitm_record *r) {
    if (r->kind == GITM_REC_KERNEL) {
        return Py_BuildValue(
            "{s:s, s:s, s:K, s:K, s:I, s:I, s:I, s:I, s:[iii], s:[iii], s:i, s:i, s:i}",
            "kind", "kernel", "name", r->name,
            "start_ns", (unsigned long long)r->start_ns,
            "end_ns", (unsigned long long)r->end_ns,
            "device_id", r->device_id, "context_id", r->context_id,
            "stream_id", r->stream_id, "correlation_id", r->correlation_id,
            "grid", r->grid[0], r->grid[1], r->grid[2],
            "block", r->block[0], r->block[1], r->block[2],
            "static_shared_mem", r->static_shared_mem,
            "dynamic_shared_mem", r->dynamic_shared_mem,
            "registers_per_thread", r->registers_per_thread);
    } else if (r->kind == GITM_REC_MEMCPY) {
        return Py_BuildValue(
            "{s:s, s:i, s:K, s:K, s:K, s:I, s:I, s:I, s:I}",
            "kind", "memcpy", "copy_kind", r->copy_kind,
            "bytes", (unsigned long long)r->bytes,
            "start_ns", (unsigned long long)r->start_ns,
            "end_ns", (unsigned long long)r->end_ns,
            "device_id", r->device_id, "context_id", r->context_id,
            "stream_id", r->stream_id, "correlation_id", r->correlation_id);
    } else {
        return Py_BuildValue(
            "{s:s, s:i, s:K, s:K, s:I, s:I, s:I, s:I}",
            "kind", "sync", "sync_type", r->sync_type,
            "start_ns", (unsigned long long)r->start_ns,
            "end_ns", (unsigned long long)r->end_ns,
            "device_id", r->device_id, "context_id", r->context_id,
            "stream_id", r->stream_id, "correlation_id", r->correlation_id);
    }
}

static PyObject *py_stop(PyObject *self, PyObject *args) {
    (void)self; (void)args;
    CUptiResult st = gitm_cupti_stop();
    if (st != CUPTI_SUCCESS) return set_cupti_error(st, "stop");

    PyObject *list = PyList_New((Py_ssize_t)g_count);
    if (!list) return NULL;
    for (size_t i = 0; i < g_count; i++) {
        PyObject *d = rec_to_dict(&g_records[i]);
        if (!d) { Py_DECREF(list); return NULL; }
        PyList_SET_ITEM(list, (Py_ssize_t)i, d);  /* steals ref */
    }
    g_count = 0;
    return list;
}

static PyObject *py_device_count(PyObject *self, PyObject *args) {
    (void)self; (void)args;
    return PyLong_FromLong(gitm_cuda_device_count());
}

/* Exposed so capture() can bound its window in the SAME clock domain as the
 * activity records — the only way to tell which records from an injected child
 * process fall inside the capture window. Safe to call without start(): reading
 * the CUPTI clock does not register callbacks, so the parent can call this while
 * the injection library owns collection. */
static PyObject *py_timestamp(PyObject *self, PyObject *args) {
    (void)self; (void)args;
    return PyLong_FromUnsignedLongLong((unsigned long long)gitm_cupti_timestamp());
}

static PyMethodDef methods[] = {
    {"start", py_start, METH_NOARGS, "Enable CUPTI activity collection."},
    {"stop", py_stop, METH_NOARGS, "Flush and return the record dicts."},
    {"device_count", py_device_count, METH_NOARGS, "Number of CUPTI devices."},
    {"timestamp", py_timestamp, METH_NOARGS, "CUPTI clock now (activity time base)."},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef moddef = {
    PyModuleDef_HEAD_INIT, "_cupti_shim",
    "CUPTI activity collection shim for GITM.", -1, methods,
    NULL, NULL, NULL, NULL,
};
PyMODINIT_FUNC PyInit__cupti_shim(void) {
    return PyModule_Create(&moddef);
}