"""Which GPU vendor a host runs, and which vendor's collector produced a trace.

Evidence is ranked: a loaded compute driver beats a PCI id beats a device name
beats a torch build. The strongest tier present decides; a split inside it is a
conflict (``vendor=None``), not a guess. Nothing here initializes a GPU runtime.
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

PCI_VENDOR = {"0x10de": "nvidia", "0x1002": "amd"}
#: Display (0x03xx, incl. MI-series 0x0380) and accelerator (0x12xx) classes —
#: not AMD's chipset functions, which share vendor 0x1002.
_GPU_CLASS_PREFIXES = ("0x03", "0x12")
STRENGTH = {"driver": 3, "trace": 3, "pci": 2, "device_name": 2, "dialect": 1, "software": 1}


@dataclass(frozen=True)
class Evidence:
    vendor: Vendor
    tier: str
    source: str
    detail: str = ""

    @property
    def strength(self) -> int:
        return STRENGTH[self.tier]


@dataclass(frozen=True)
class Classification:
    vendor: Vendor | None
    evidence: tuple[Evidence, ...] = field(default_factory=tuple)
    conflict: bool = False

    @property
    def decided_by(self) -> str | None:
        if self.vendor is None:
            return None
        return max((e for e in self.evidence if e.vendor == self.vendor),
                   key=lambda e: e.strength).tier

    def explain(self) -> str:
        head = (f"vendor={self.vendor}" if self.vendor else
                "vendor undecided (conflict)" if self.conflict else "vendor undecided")
        return "\n".join([head, *(f"  [{e.tier}] {e.vendor}: {e.source} {e.detail}".rstrip()
                                  for e in sorted(self.evidence, key=lambda e: -e.strength))])


def decide(evidence: Iterable[Evidence]) -> Classification:
    ev = tuple(evidence)
    if not ev:
        return Classification(None, ev)
    top = max(e.strength for e in ev)
    vendors = {e.vendor for e in ev if e.strength == top}
    if len(vendors) == 1:
        return Classification(vendors.pop(), ev)
    return Classification(None, ev, conflict=True)


def _read(p: Path) -> str | None:
    try:
        return p.read_text().strip()
    except OSError:
        return None


_AMD_NAME = re.compile(r"\b(amd|instinct|radeon|mi\d{3}[a-z]*)\b", re.I)
_NVIDIA_NAME = re.compile(
    r"\b(nvidia|tesla|geforce|quadro|rtx|[abgh]\d{2,3}[a-z]*|l\d{1,2}[a-z]*|gb\d{3}|gh\d{3})\b",
    re.I)


def vendor_of_device_name(name: str | None) -> Vendor | None:
    if not name:
        return None
    if _AMD_NAME.search(name):  # first: MI300X would also fit an NVIDIA SKU shape
        return "amd"
    if _NVIDIA_NAME.search(name):
        return "nvidia"
    return None


def host_evidence(*, root: str | Path = "/", modules: Mapping[str, object] | None = None,
                  device_name: str | None = None) -> list[Evidence]:
    """``root`` relocates the filesystem probes; torch is read only if already imported."""
    r = Path(root)
    ev: list[Evidence] = []

    nodes = r / "sys/class/kfd/kfd/topology/nodes"
    kfd = sum(_read(n / "gpu_id") not in (None, "", "0") for n in nodes.iterdir()) \
        if nodes.is_dir() else 0
    if kfd:
        ev.append(Evidence("amd", "driver", "kfd topology", f"{kfd} GPU node(s)"))
    nv = _read(r / "proc/driver/nvidia/version")
    if nv is not None or (r / "dev/nvidiactl").exists():
        ev.append(Evidence("nvidia", "driver", "nvidia kernel driver",
                           (nv or "/dev/nvidiactl").splitlines()[0][:80]))

    pci: dict[str, int] = {}
    devices = r / "sys/bus/pci/devices"
    for dev in devices.iterdir() if devices.is_dir() else ():
        vendor = PCI_VENDOR.get((_read(dev / "vendor") or "").lower())
        if vendor and (_read(dev / "class") or "").lower().startswith(_GPU_CLASS_PREFIXES):
            pci[vendor] = pci.get(vendor, 0) + 1
    ev += [Evidence(v, "pci", "PCI GPU functions", str(n)) for v, n in sorted(pci.items())]  # type: ignore[arg-type]

    named = vendor_of_device_name(device_name)
    if named:
        ev.append(Evidence(named, "device_name", "device name", device_name or ""))

    version = getattr((sys.modules if modules is None else modules).get("torch"), "version", None)
    if getattr(version, "hip", None):
        ev.append(Evidence("amd", "software", "torch.version.hip", str(version.hip)))
    elif getattr(version, "cuda", None):
        ev.append(Evidence("nvidia", "software", "torch.version.cuda", str(version.cuda)))
    return ev


def classify_host(**kw) -> Classification:
    return decide(host_evidence(**kw))


#: Name fragments unique to one vendor's libraries: Tensile/hipBLASLt, ROCclr
#: blits, CK, vLLM ROCm skinny GEMMs and AITER; cuBLAS/CUTLASS SASS families.
#: RCCL keeps NCCL's ``ncclDevKernel`` names, so that is evidence of neither.
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


def trace_evidence(records: Iterable) -> list[Evidence]:
    """From raw record dicts or decoded events. The collector's ``meta`` record
    decides; otherwise kernel names vote and need a 4:1 majority."""
    ev: list[Evidence] = []
    votes = {"amd": 0, "nvidia": 0}
    for r in records:
        get = r.get if isinstance(r, Mapping) else (lambda k, _r=r: getattr(_r, k, None))
        if get("kind") == "meta" and get("collector"):
            c = str(get("collector"))
            vendor = "amd" if "rocprofiler" in c else "nvidia" if "cupti" in c else None
            if vendor:
                ev.append(Evidence(vendor, "trace", "collector meta record", c))
        elif get("kind") == "kernel" and (v := name_dialect(get("name") or "")):
            votes[v] += 1
    a, n = votes["amd"], votes["nvidia"]
    if a and a >= 4 * n:
        ev.append(Evidence("amd", "dialect", "kernel-name dialect", f"{a} vs {n}"))
    elif n and n >= 4 * a:
        ev.append(Evidence("nvidia", "dialect", "kernel-name dialect", f"{n} vs {a}"))
    return ev


def classify_trace(records: Iterable) -> Classification:
    return decide(trace_evidence(records))


def env_vendor_override(env: Mapping[str, str] | None = None) -> Vendor | None:
    """``GITM_VENDOR=amd|nvidia``."""
    raw = (os.environ if env is None else env).get("GITM_VENDOR", "").strip().lower()
    return raw if raw in ("amd", "nvidia") else None  # type: ignore[return-value]
