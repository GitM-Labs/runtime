"""Which GPU vendor a host runs, and which vendor's collector produced a trace.

Everything vendor-specific in the tracer forks on this answer: which injection
hook ``run_env`` renders (``CUDA_INJECTION64_PATH`` vs ``ROCP_TOOL_LIBRARIES``),
which clock bounds the capture window, which kernel-name dialect the taxonomy
has to read. Getting it wrong is silent in every case — the wrong hook loads
nothing, the wrong clock windows out the whole trace, the wrong dialect files
every hipBLASLt GEMM as ``other``.

The answer is assembled from **evidence**, each item naming its source, rather
than from the first probe that returns something. Two reasons:

* Probes disagree on real machines. A workstation with an AMD APU and an NVIDIA
  card shows both vendors on the PCI bus; a ROCm container on an NVIDIA host
  carries ``torch.version.hip`` with no AMD device behind it. A first-match
  answer is whichever probe happens to run first.
* The strength of a probe matters. A loaded compute driver (kfd GPU nodes,
  ``/proc/driver/nvidia``) says the runtime can reach a device; a PCI vendor id
  says only that one is plugged in; a torch build says only what software is
  installed. :func:`classify_host` lets the strongest tier decide and reports a
  conflict *within* that tier instead of resolving it by accident.

For traces, the strongest evidence is in-band: the ROCm collector writes a
``{"kind":"meta","collector":"rocprofiler-sdk",...}`` record at tool init.
Kernel-name dialect is the fallback for captures that predate it — Tensile's
``Cijk_`` GEMMs, ROCclr's ``__amd_rocclr_*`` blits and CK's ``ck_tile`` exist
only on AMD, cuBLAS's ``nvjet``/``sm90_xmma`` families only on NVIDIA.

Nothing here initializes a GPU runtime. Every probe is a file read or a lookup
of an already-imported module, so it is safe while an injected collector owns
the process, and on Windows (no ``/sys``) it degrades to "no evidence".
"""

from __future__ import annotations

import os
import re
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

Vendor = Literal["nvidia", "amd"]

#: PCI vendor ids (``/sys/bus/pci/devices/*/vendor``).
PCI_VENDOR = {"0x10de": "nvidia", "0x1002": "amd"}

#: PCI class prefixes that are GPUs or compute accelerators: 0x03xx display
#: controllers (consumer cards, and MI-series parts, which report 0x0380) and
#: 0x12xx processing accelerators. Excludes AMD's chipset/USB functions, which
#: share vendor 0x1002 and would otherwise read as a GPU on every Ryzen board.
_GPU_CLASS_PREFIXES = ("0x03", "0x12")

#: Evidence strength. The strongest tier present decides; a split *inside* it
#: is a conflict.
STRENGTH = {"driver": 3, "trace": 3, "pci": 2, "device_name": 2, "dialect": 1, "software": 1}


@dataclass(frozen=True)
class Evidence:
    """One probe's answer, with where it came from and how much it weighs."""

    vendor: Vendor
    tier: str  # key of STRENGTH
    source: str
    detail: str = ""

    @property
    def strength(self) -> int:
        return STRENGTH[self.tier]


@dataclass(frozen=True)
class Classification:
    """The decided vendor (``None`` when undecidable) and everything behind it."""

    vendor: Vendor | None
    evidence: tuple[Evidence, ...] = field(default_factory=tuple)
    #: True when the deciding tier holds both vendors. ``vendor`` is then None:
    #: the caller has to choose, because nothing here can.
    conflict: bool = False

    @property
    def decided_by(self) -> str | None:
        """The tier that decided, or ``None`` when nothing did."""
        if self.vendor is None:
            return None
        return max((e for e in self.evidence if e.vendor == self.vendor),
                   key=lambda e: e.strength).tier

    def explain(self) -> str:
        if not self.evidence:
            return "no vendor evidence"
        rows = [f"  [{e.tier}] {e.vendor}: {e.source}" + (f" ({e.detail})" if e.detail else "")
                for e in sorted(self.evidence, key=lambda e: -e.strength)]
        head = (f"vendor={self.vendor}" if self.vendor else
                "vendor undecided (conflict)" if self.conflict else "vendor undecided")
        return "\n".join([head, *rows])


def decide(evidence: Iterable[Evidence]) -> Classification:
    """Let the strongest tier present decide; a split inside it is a conflict."""
    ev = tuple(evidence)
    if not ev:
        return Classification(None, ev)
    top = max(e.strength for e in ev)
    vendors = {e.vendor for e in ev if e.strength == top}
    if len(vendors) == 1:
        return Classification(vendors.pop(), ev)
    return Classification(None, ev, conflict=True)


# ── host ────────────────────────────────────────────────────────────────────


def _read(p: Path) -> str | None:
    try:
        return p.read_text().strip()
    except OSError:
        return None


def _kfd_gpu_nodes(root: Path) -> int:
    nodes = root / "sys/class/kfd/kfd/topology/nodes"
    if not nodes.is_dir():
        return 0
    n = 0
    for node in nodes.iterdir():
        gpu_id = _read(node / "gpu_id")
        if gpu_id not in (None, "", "0"):
            n += 1
    return n


def _pci_gpus(root: Path) -> dict[str, int]:
    devices = root / "sys/bus/pci/devices"
    out: dict[str, int] = {}
    if not devices.is_dir():
        return out
    for dev in devices.iterdir():
        vendor = PCI_VENDOR.get((_read(dev / "vendor") or "").lower())
        klass = (_read(dev / "class") or "").lower()
        if vendor and klass.startswith(_GPU_CLASS_PREFIXES):
            out[vendor] = out.get(vendor, 0) + 1
    return out


_AMD_NAME = re.compile(r"\b(amd|instinct|radeon|mi\d{3}[a-z]*)\b", re.I)
_NVIDIA_NAME = re.compile(
    r"\b(nvidia|tesla|geforce|quadro|rtx|[abgh]\d{2,3}[a-z]*|l\d{1,2}[a-z]*|gb\d{3}|gh\d{3})\b",
    re.I)


def vendor_of_device_name(name: str | None) -> Vendor | None:
    """``"AMD Instinct MI355X"`` -> amd, ``"NVIDIA H200"`` -> nvidia, else None.

    The AMD pattern is tested first: no AMD product name matches the NVIDIA
    SKU shapes, but ``MI300X`` is close enough to ``[a-z]\\d{3}`` that the order
    is not left to chance.
    """
    if not name:
        return None
    if _AMD_NAME.search(name):
        return "amd"
    if _NVIDIA_NAME.search(name):
        return "nvidia"
    return None


def host_evidence(
    *,
    root: str | Path = "/",
    modules: Mapping[str, object] | None = None,
    device_name: str | None = None,
) -> list[Evidence]:
    """Every vendor signal this host gives without initializing a GPU runtime.

    ``root`` relocates the filesystem probes (tests point it at a fake tree).
    ``modules`` defaults to :data:`sys.modules`; torch is consulted only if it
    is already imported, because importing it here would cost seconds and, on
    ROCm, load HIP into a process an injected collector may be watching.
    """
    r = Path(root)
    ev: list[Evidence] = []

    kfd = _kfd_gpu_nodes(r)
    if kfd:
        ev.append(Evidence("amd", "driver", "kfd topology", f"{kfd} GPU node(s)"))
    nv_version = _read(r / "proc/driver/nvidia/version")
    if nv_version is not None or (r / "dev/nvidiactl").exists():
        ev.append(Evidence("nvidia", "driver", "nvidia kernel driver",
                           (nv_version or "/dev/nvidiactl").splitlines()[0][:80]))

    for vendor, n in sorted(_pci_gpus(r).items()):
        ev.append(Evidence(vendor, "pci", "PCI display/accelerator functions", f"{n}"))  # type: ignore[arg-type]

    named = vendor_of_device_name(device_name)
    if named:
        ev.append(Evidence(named, "device_name", "device name", device_name or ""))

    mods = sys.modules if modules is None else modules
    torch = mods.get("torch")
    version = getattr(torch, "version", None)
    if getattr(version, "hip", None):
        ev.append(Evidence("amd", "software", "torch.version.hip", str(version.hip)))
    elif getattr(version, "cuda", None):
        ev.append(Evidence("nvidia", "software", "torch.version.cuda", str(version.cuda)))
    return ev


def classify_host(**kw) -> Classification:
    """:func:`decide` over :func:`host_evidence`."""
    return decide(host_evidence(**kw))


# ── trace ───────────────────────────────────────────────────────────────────

#: Kernel-name fragments that exist on one vendor only. Lower-case substrings.
#: Each is a library's own naming convention, not a guess about what a kernel
#: does, which is what makes the absence of overlap dependable:
#:
#: * AMD — Tensile/hipBLASLt solution names (``Cijk_Ailk_Bljk_...``), ROCclr's
#:   blit kernels (``__amd_rocclr_copyBuffer``/``fillBufferAligned``), Composable
#:   Kernel (``ck_tile``, ``ck::kernel_gemm_xdl``), vLLM's ROCm skinny GEMMs
#:   (``wvSplitK``, ``LLGemm1``) and AITER (``aiter::``, ``_ZN5aiter``).
#: * NVIDIA — cuBLAS's JIT families (``nvjet_``), SASS-arch-tagged cuBLAS and
#:   CUTLASS kernels (``sm80_xmma``, ``ampere_*gemm``, ``sm90_``), NCCL's
#:   ``ncclKernel``/CUDA-only device helpers. RCCL keeps NCCL's
#:   ``ncclDevKernel`` names, so that prefix is NOT evidence either way.
_AMD_DIALECT = ("cijk_", "__amd_rocclr", "ck_tile", "ck::", "_zn2ck", "_zn7ck_tile",
                "wvsplitk", "llgemm1", "aiter::", "_zn5aiter", "gfx9", "gfx12")
_NVIDIA_DIALECT = ("nvjet_", "xmma", "ampere_", "volta_", "turing_", "hopper_",
                   "sm80_", "sm86_", "sm89_", "sm90_", "sm100_", "sm120_", "cutlass::",
                   "_zn7cutlass", "s16816gemm", "s1688gemm", "cublaslt", "cudnn")


def name_dialect(name: str) -> Vendor | None:
    n = (name or "").lower()
    if any(k in n for k in _AMD_DIALECT):
        return "amd"
    if any(k in n for k in _NVIDIA_DIALECT):
        return "nvidia"
    return None


def trace_evidence(records: Iterable[Mapping]) -> list[Evidence]:
    """Vendor evidence carried by collector records (raw dicts or decoded events).

    The collector's own ``meta`` record decides when present. Otherwise kernel
    names vote, and a dialect counts only with a 4:1 majority over the other —
    a mixed vote means stale shards from another run, or a name list that has
    grown a cross-vendor needle, and either way it is not an answer.
    """
    ev: list[Evidence] = []
    votes = {"amd": 0, "nvidia": 0}
    for r in records:
        get = r.get if isinstance(r, Mapping) else (lambda k, _r=r: getattr(_r, k, None))
        kind = get("kind")
        if kind == "meta" and get("collector"):
            collector = str(get("collector"))
            vendor = ("amd" if "rocprofiler" in collector else
                      "nvidia" if "cupti" in collector else None)
            if vendor:
                ev.append(Evidence(vendor, "trace", "collector meta record", collector))
        elif kind == "kernel":
            v = name_dialect(get("name") or "")
            if v:
                votes[v] += 1
    a, n = votes["amd"], votes["nvidia"]
    if a and a >= 4 * n:
        ev.append(Evidence("amd", "dialect", "kernel-name dialect", f"{a} amd vs {n} nvidia"))
    elif n and n >= 4 * a:
        ev.append(Evidence("nvidia", "dialect", "kernel-name dialect", f"{n} nvidia vs {a} amd"))
    return ev


def classify_trace(records: Iterable[Mapping]) -> Classification:
    return decide(trace_evidence(records))


def env_vendor_override(env: Mapping[str, str] | None = None) -> Vendor | None:
    """``GITM_VENDOR=amd|nvidia`` — the operator's answer when evidence conflicts."""
    raw = (os.environ if env is None else env).get("GITM_VENDOR", "").strip().lower()
    return raw if raw in ("amd", "nvidia") else None  # type: ignore[return-value]
