/* roctx64 -> rocprofiler-sdk-roctx forwarding shim (LD_PRELOAD).
 *
 * PyTorch's ROCm build links torch.cuda.nvtx to the LEGACY roctx
 * (libroctx64.so, roctracer lineage). rocprofiler-sdk's marker tracing
 * service only sees ranges routed through rocprofiler-register — which the
 * sdk's own librocprofiler-sdk-roctx.so does and libroctx64 does NOT (verified
 * on ROCm 7.2.3: a libroctx64 push produces no marker record while an sdk push
 * does). So every torch/vLLM layerwise range lands in the void, and arm-C
 * captures come back with range_op null on all kernels — silently.
 *
 * Preloading this shim interposes the legacy entry points and forwards them to
 * the sdk implementation, resolved by dlopen/dlsym (NOT by linking, which
 * would resolve the same-named symbols back to us). Only needed in processes
 * that EMIT markers (the serving engine, under GITM_TRACE_NVTX=1); the
 * collector side is unaffected.
 *
 * Build: python -m gitm.tracer._rocm.build (produces libgitm_roctx_shim.so
 * beside the tool). Deploy: LD_PRELOAD=<path>/libgitm_roctx_shim.so on the
 * correlation arm only — a range nobody collects still costs throughput.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdint.h>
#include <stddef.h>

typedef uint64_t roctx_range_id_t;

static void *sdk;
static int (*sdk_push)(const char *);
static int (*sdk_pop)(void);
static void (*sdk_mark)(const char *);
static roctx_range_id_t (*sdk_start)(const char *);
static void (*sdk_stop)(roctx_range_id_t);

__attribute__((constructor)) static void gitm_roctx_shim_init(void) {
    /* RTLD_LOCAL: the sdk library's own roctx symbols must not enter the
     * global scope, or the next lookup of roctxRangePushA could bind to it
     * directly and bypass nothing — fine — or bind us to ourselves — fatal. */
    sdk = dlopen("librocprofiler-sdk-roctx.so", RTLD_NOW | RTLD_LOCAL);
    if (!sdk)
        return; /* every entry point degrades to the legacy no-op */
    sdk_push = (int (*)(const char *))dlsym(sdk, "roctxRangePushA");
    sdk_pop = (int (*)(void))dlsym(sdk, "roctxRangePop");
    sdk_mark = (void (*)(const char *))dlsym(sdk, "roctxMarkA");
    sdk_start = (roctx_range_id_t(*)(const char *))dlsym(sdk, "roctxRangeStartA");
    sdk_stop = (void (*)(roctx_range_id_t))dlsym(sdk, "roctxRangeStop");
}

int roctxRangePushA(const char *msg) { return sdk_push ? sdk_push(msg) : 0; }
int roctxRangePop(void) { return sdk_pop ? sdk_pop() : 0; }
void roctxMarkA(const char *msg) {
    if (sdk_mark)
        sdk_mark(msg);
}
roctx_range_id_t roctxRangeStartA(const char *msg) {
    return sdk_start ? sdk_start(msg) : 0;
}
void roctxRangeStop(roctx_range_id_t id) {
    if (sdk_stop)
        sdk_stop(id);
}
