"""Vendor evidence: the strongest tier decides, a split inside it is a conflict."""

from __future__ import annotations

import types

import pytest

from gitm.tracer import injection
from gitm.tracer import vendor as V


def _tree(tmp_path, *, kfd_gpus=0, kfd_cpus=1, nvidia_driver=False, pci=()):
    nodes = tmp_path / "sys/class/kfd/kfd/topology/nodes"
    i = 0
    for _ in range(kfd_cpus):
        (nodes / str(i)).mkdir(parents=True)
        (nodes / str(i) / "gpu_id").write_text("0\n")
        i += 1
    for g in range(kfd_gpus):
        (nodes / str(i)).mkdir(parents=True)
        (nodes / str(i) / "gpu_id").write_text(f"{4000 + g}\n")
        i += 1
    if nvidia_driver:
        (tmp_path / "proc/driver/nvidia").mkdir(parents=True)
        (tmp_path / "proc/driver/nvidia/version").write_text(
            "NVRM version: NVIDIA UNIX x86_64 Kernel Module  580.65.06\n")
    for j, (vendor_id, klass) in enumerate(pci):
        # Real entries are 0000:BB:DD.F; the probe never parses the name, and
        # Windows test hosts reject ":" in paths.
        d = tmp_path / f"sys/bus/pci/devices/pci{j:02x}"
        d.mkdir(parents=True)
        (d / "vendor").write_text(vendor_id + "\n")
        (d / "class").write_text(klass + "\n")
    return tmp_path


NO_MODULES: dict = {}


def test_an_empty_host_has_no_evidence_and_no_vendor(tmp_path):
    c = V.classify_host(root=_tree(tmp_path), modules=NO_MODULES)
    assert c.vendor is None and not c.conflict and c.evidence == ()


def test_kfd_gpu_nodes_are_amd_driver_evidence(tmp_path):
    c = V.classify_host(root=_tree(tmp_path, kfd_gpus=8), modules=NO_MODULES)
    assert c.vendor == "amd" and c.decided_by == "driver"


def test_kfd_cpu_nodes_alone_are_not_a_gpu(tmp_path):
    """CPU sockets appear as kfd nodes with gpu_id 0 on every ROCm host."""
    c = V.classify_host(root=_tree(tmp_path, kfd_gpus=0, kfd_cpus=2), modules=NO_MODULES)
    assert c.vendor is None


def test_the_nvidia_kernel_driver_is_nvidia_driver_evidence(tmp_path):
    c = V.classify_host(root=_tree(tmp_path, nvidia_driver=True), modules=NO_MODULES)
    assert c.vendor == "nvidia" and c.decided_by == "driver"


def test_an_apu_beside_an_nvidia_card_is_decided_by_the_driver_tier(tmp_path):
    """The workstation case: both vendors on the PCI bus."""
    root = _tree(tmp_path, nvidia_driver=True,
                 pci=[("0x1002", "0x030000"), ("0x10de", "0x030000")])
    c = V.classify_host(root=root, modules=NO_MODULES)
    assert c.vendor == "nvidia" and not c.conflict
    assert {e.vendor for e in c.evidence if e.tier == "pci"} == {"amd", "nvidia"}


def test_both_compute_drivers_loaded_is_a_conflict_not_a_guess(tmp_path):
    c = V.classify_host(root=_tree(tmp_path, kfd_gpus=1, nvidia_driver=True),
                        modules=NO_MODULES)
    assert c.vendor is None and c.conflict
    assert "conflict" in c.explain()


def test_amd_chipset_functions_are_not_gpus(tmp_path):
    """Vendor 0x1002 also owns USB/SMBus functions on every Ryzen board."""
    root = _tree(tmp_path, pci=[("0x1002", "0x0c0330"), ("0x1002", "0x0c0500")])
    assert V.classify_host(root=root, modules=NO_MODULES).vendor is None


def test_mi_series_accelerators_count_under_either_gpu_class(tmp_path):
    root = _tree(tmp_path, pci=[("0x1002", "0x038000"), ("0x1002", "0x120000")])
    c = V.classify_host(root=root, modules=NO_MODULES)
    assert c.vendor == "amd" and c.decided_by == "pci"


def test_a_torch_build_is_only_software_evidence(tmp_path):
    torch = types.SimpleNamespace(version=types.SimpleNamespace(hip="7.2.53211", cuda=None))
    alone = V.classify_host(root=_tree(tmp_path / "a"), modules={"torch": torch})
    assert alone.vendor == "amd" and alone.decided_by == "software"
    outranked = V.classify_host(root=_tree(tmp_path / "b", nvidia_driver=True),
                                modules={"torch": torch})
    assert outranked.vendor == "nvidia" and not outranked.conflict


@pytest.mark.parametrize("name,vendor", [
    ("AMD Instinct MI355X", "amd"),
    ("AMD Instinct MI300X", "amd"),
    ("AMD Radeon Graphics", "amd"),
    ("NVIDIA H200", "nvidia"),
    ("NVIDIA B200", "nvidia"),
    ("NVIDIA GB200", "nvidia"),
    ("NVIDIA L4", "nvidia"),
    ("NVIDIA GeForce RTX 4090", "nvidia"),
    ("Tesla V100-SXM2-16GB", "nvidia"),
    ("", None),
    ("Intel Data Center GPU Max 1550", None),
])
def test_device_names(name, vendor):
    assert V.vendor_of_device_name(name) == vendor


def test_env_override_wins_over_conflict(monkeypatch):
    monkeypatch.setenv("GITM_VENDOR", "amd")
    monkeypatch.setattr(V, "classify_host", lambda **kw: pytest.fail("not consulted"))
    assert injection.detect_vendor() == "amd"


def test_detect_vendor_keeps_the_nvidia_default_on_an_empty_host(monkeypatch):
    monkeypatch.delenv("GITM_VENDOR", raising=False)
    monkeypatch.setattr(V, "classify_host", lambda **kw: V.Classification(None))
    assert injection.detect_vendor() == "nvidia"


def test_detect_vendor_warns_on_conflict_and_says_how_to_choose(monkeypatch):
    monkeypatch.delenv("GITM_VENDOR", raising=False)
    ev = (V.Evidence("amd", "driver", "kfd"), V.Evidence("nvidia", "driver", "nv"))
    monkeypatch.setattr(V, "classify_host", lambda **kw: V.Classification(None, ev, True))
    with pytest.warns(RuntimeWarning, match="GITM_VENDOR"):
        assert injection.detect_vendor() == "nvidia"


def test_detect_vendor_follows_amd_driver_evidence(monkeypatch):
    monkeypatch.delenv("GITM_VENDOR", raising=False)
    monkeypatch.setattr(V, "classify_host", lambda **kw: V.decide(
        [V.Evidence("amd", "driver", "kfd")]))
    assert injection.detect_vendor() == "amd"
    assert injection.run_env("/tmp/t.jsonl", vendor=None)[injection.ENV_ROCP]


# ── traces ──────────────────────────────────────────────────────────────────


def test_the_rocm_collector_meta_record_decides_a_trace():
    recs = [{"kind": "meta", "collector": "rocprofiler-sdk"},
            *[{"kind": "kernel", "name": "nvjet_sm90_tst_x"} for _ in range(50)]]
    c = V.classify_trace(recs)
    assert c.vendor == "amd" and c.decided_by == "trace"


def test_kernel_dialect_decides_a_trace_without_meta():
    amd = [{"kind": "kernel", "name": n} for n in (
        "Cijk_Alik_Bljk_BBS_BH_MT256x256x64", "__amd_rocclr_copyBuffer",
        "_ZN5aiter19fmha_fwd_hd128_bf16E", "wvSplitK_hf_sml_",
        "ncclDevKernel_Generic_4(ncclDevKernelArgsStorage<4096ul>)")]
    nv = [{"kind": "kernel", "name": n} for n in (
        "nvjet_sm90_tst_128x8_64x12_4x1_v_bz_TNT", "sm90_xmma_gemm_bf16bf16_bf16f32",
        "ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_tn",
        "ncclDevKernel_AllReduce_Sum_bf16_RING_LL")]
    assert V.classify_trace(amd).vendor == "amd"
    assert V.classify_trace(nv).vendor == "nvidia"


def test_rccl_keeps_ncclDevKernel_so_it_is_no_evidence_either_way():
    assert V.name_dialect("ncclDevKernel_Generic_4(x)") is None
    assert V.name_dialect("ncclDevKernel_AllReduce_Sum_bf16_RING_LL") is None


def test_a_mixed_dialect_trace_is_not_an_answer():
    recs = ([{"kind": "kernel", "name": "Cijk_A"}] * 3
            + [{"kind": "kernel", "name": "nvjet_sm90_x"}] * 2)
    assert V.classify_trace(recs).vendor is None


def test_decoded_events_classify_too():
    from gitm.tracer.schema import KernelEvent

    evs = [KernelEvent(name="Cijk_Alik_Bljk_BBS", start_ns=0, end_ns=1, stream_id=0,
                       device_id=0) for _ in range(4)]
    assert V.classify_trace(evs).vendor == "amd"


def test_emulated_collector_output_classifies_to_its_vendor():
    from gitm.optimizer.mechanism_fixtures import Scenario, generate
    from gitm.tracer.emulate import EmulationConfig, emulate, launches_from_fixture

    fx = generate(Scenario(n_steps=1))[0]
    la = launches_from_fixture(fx)
    assert V.classify_trace(emulate(la, EmulationConfig("amd")).records).vendor == "amd"
    assert V.classify_trace(emulate(la, EmulationConfig("nvidia")).records).vendor == "nvidia"
