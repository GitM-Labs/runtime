"""The last SKU fallback: the kfd topology, for the AMD box torch did not name.

After the torch fallback landed, the MI355X pod still priced every roofline
floor against A100, because NVML gave no name and neither did torch. The kfd
topology in sysfs names the device without either. These tests point
``_KFD_NODES`` at a fake tree built from a real dump.
"""

from __future__ import annotations

import os
from types import SimpleNamespace as NS

import pytest

import gitm.planner.context as ctx

# Node 5 of rocprofiler-sdk's kfd topology test fixture, verbatim: an MI350X
# (device_id 30112 = 0x75A0, gfx_target_version 90500 = gfx950).
# https://github.com/ROCm/rocm-systems/blob/923570cefcb446b0b09119950fdc495312083c29/projects/rocprofiler-sdk/source/lib/rocprofiler-sdk/tests/data/topology/nodes/5/properties
MI350X_PROPERTIES = """\
cpu_cores_count 0
simd_count 1024
mem_banks_count 1
caches_count 0
io_links_count 1
p2p_links_count 0
cpu_core_id_base 0
simd_id_base 2147487744
max_waves_per_simd 8
lds_size_in_kb 160
gds_size_in_kb 0
num_gws 64
wave_front_size 64
array_count 32
simd_arrays_per_engine 1
cu_per_simd_array 9
simd_per_cu 4
max_slots_scratch_cu 32
gfx_target_version 90500
vendor_id 4098
device_id 30112
location_id 33536
domain 0
drm_render_minor 0
hive_id 0
num_sdma_engines 2
num_sdma_xgmi_engines 14
num_sdma_queues_per_engine 8
num_cp_queues 24
cwsr_size 22687744
ctl_stack_size 12288
max_engine_clk_fcompute 2200
local_mem_size 0
fw_version 36
capability 2889327232
capability2 1
debug_prop 1494
sdma_fw_version 12
unique_id 5176797314141101417
num_xcc 8
max_engine_clk_ccompute 2450
"""

# Node 0 of the same fixture, a CPU socket (gpu_id 0), abbreviated.
CPU_PROPERTIES = """\
cpu_cores_count 24
simd_count 0
gfx_target_version 0
vendor_id 0
device_id 0
"""

MI355X_DEVICE_ID = 0x75A3


def _engine(world_size: int):
    return NS(vllm_config=NS(parallel_config=NS(world_size=world_size)))


def _gpu_properties(device_id: int) -> str:
    """The MI350X dump with only its PCI device id changed."""
    return MI350X_PROPERTIES.replace("device_id 30112", f"device_id {device_id}")


def _node(root, index: int, gpu_id: int, properties: str) -> None:
    node = root / str(index)
    node.mkdir()
    (node / "gpu_id").write_text(f"{gpu_id}\n")
    (node / "properties").write_text(properties)


@pytest.fixture
def kfd(tmp_path, monkeypatch):
    """A kfd tree holding one CPU node, as every real one does."""
    root = tmp_path / "nodes"
    root.mkdir()
    _node(root, 0, 0, CPU_PROPERTIES)
    monkeypatch.setattr(ctx, "_KFD_NODES", root)
    return root


def _mi355x_nodes(kfd, gpus: int = 8) -> None:
    for i in range(1, gpus + 1):
        _node(kfd, i, 1000 + i, _gpu_properties(MI355X_DEVICE_ID))


# ── reading the topology ────────────────────────────────────────────────────
def test_an_mi355x_is_named_and_priced_as_one(kfd):
    _mi355x_nodes(kfd)

    name, count = ctx._query_amd()

    assert (name, count) == ("AMD Instinct MI355X", 8)
    peak = ctx.peak_for_sku(name)
    assert peak is not None
    assert (peak.peak_flops, peak.peak_bw_bytes_s) == (2500e12, 8000e9)
    assert ctx.hardware_spec_for(peak).arch == "cdna4"


def test_an_mi350x_is_named_but_not_priced_as_an_mi355x(kfd):
    """Same gfx950, different part. Only the device id separates them, and
    MI350X has no catalogue entry, so it must not borrow MI355X's."""
    _node(kfd, 1, 55290, MI350X_PROPERTIES)

    name, count = ctx._query_amd()

    assert (name, count) == ("AMD Instinct MI350X", 1)
    assert ctx.peak_for_sku(name) is None


def test_only_mi355x_names_reach_the_mi355x_entry():
    """The catalogue has MI355X and no other gfx950 part, so every other name
    in the table must miss rather than borrow MI355X's peaks."""
    for device, name in ctx._AMD_DEVICE_NAMES.items():
        matched = ctx.peak_for_sku(name) is not None
        assert matched == ("MI355X" in name), (hex(device), name)


def test_an_unlisted_device_is_named_visibly_and_matches_nothing(kfd):
    """A wrong match is silent; a miss names what the box was."""
    _node(kfd, 1, 1001, _gpu_properties(0x75AF))

    name, _ = ctx._query_amd()

    assert name == "AMD gfx950 device id 30127"
    assert ctx.peak_for_sku(name) is None


@pytest.mark.parametrize("device", [0xA100, 0xB200, 0xB300, 0x4000, 0xFFFF])
def test_no_fallback_name_can_contain_a_catalogue_key(kfd, device):
    """`peak_for_sku` matches substrings, and a hex id such as 0xa100 spelled
    "A100" priced an unknown AMD part as an A100."""
    _node(kfd, 1, 1001, _gpu_properties(device))

    name, _ = ctx._query_amd()

    assert ctx.peak_for_sku(name) is None, name
    assert all(key.lower() not in name.lower() for key in ctx._PEAKS), name


def test_a_node_that_does_not_say_amd_is_counted_but_not_named(kfd):
    """kfd only lists AMD GPUs, so vendor_id 4098 (0x1002) is a sanity check:
    a node whose properties do not say so names nothing."""
    _node(kfd, 1, 1001, _gpu_properties(MI355X_DEVICE_ID).replace("vendor_id 4098", ""))

    assert ctx._query_amd() == (None, 1)


def test_cpu_nodes_are_not_gpus(kfd):
    assert ctx._query_amd() == (None, 0)


def test_an_unreadable_nodes_directory_is_no_answer(kfd):
    """Another probe that must not raise out of build_planner_context."""
    _mi355x_nodes(kfd)
    kfd.chmod(0)
    try:
        if os.access(kfd, os.R_OK):
            pytest.skip("permissions are not enforced for this user")
        assert ctx._query_amd() == (None, None)
    finally:
        kfd.chmod(0o755)


def test_no_kfd_is_no_answer(tmp_path, monkeypatch):
    monkeypatch.setattr(ctx, "_KFD_NODES", tmp_path / "absent")
    assert ctx._query_amd() == (None, None)


def test_a_node_the_container_cannot_open_is_not_counted(kfd):
    """kfd answers EPERM for GPUs outside the container's device cgroup. A
    directory where the file should be raises the same OSError family."""
    _mi355x_nodes(kfd, gpus=2)
    (kfd / "3").mkdir()
    (kfd / "3" / "gpu_id").mkdir()

    assert ctx._query_amd() == ("AMD Instinct MI355X", 2)


def test_node_order_is_numeric(kfd):
    """The name comes from the first GPU node; "10" must not sort before "2"."""
    for i in range(2, 11):
        device = MI355X_DEVICE_ID if i == 2 else 0x75AF
        _node(kfd, i, 1000 + i, _gpu_properties(device))

    assert ctx._query_amd() == ("AMD Instinct MI355X", 9)


# ── its place in build_planner_context ──────────────────────────────────────
def _no_nvml_no_torch(monkeypatch):
    monkeypatch.delenv("GITM_GPU_SKU", raising=False)
    monkeypatch.setattr(ctx, "_query_nvml", lambda: (None, None))
    monkeypatch.setattr(ctx, "_query_torch", lambda: (None, None))


def test_the_pod_case_resolves_to_mi355x(kfd, monkeypatch):
    """The MI355X pod: NVML absent, torch gave no name."""
    _no_nvml_no_torch(monkeypatch)
    _mi355x_nodes(kfd)

    pctx = ctx.build_planner_context(None, workload="vllm-decode")

    assert pctx.sku == "AMD Instinct MI355X"
    assert pctx.peak is not None and pctx.peak.peak_bw_bytes_s == 8000e9
    assert pctx.num_gpus == 8


def test_the_run_world_size_still_beats_the_kfd_count(kfd, monkeypatch):
    _no_nvml_no_torch(monkeypatch)
    _mi355x_nodes(kfd)

    assert ctx.build_planner_context(_engine(1)).num_gpus == 1


def _kfd_must_not_be_asked(monkeypatch):
    def asked():
        raise AssertionError("kfd read although the SKU was already known")

    monkeypatch.setattr(ctx, "_query_amd", asked)


def test_nvml_wins_and_kfd_is_not_asked(monkeypatch):
    monkeypatch.delenv("GITM_GPU_SKU", raising=False)
    monkeypatch.setattr(ctx, "_query_nvml", lambda: ("NVIDIA H200", 2))
    _kfd_must_not_be_asked(monkeypatch)

    assert ctx.build_planner_context(None).sku == "NVIDIA H200"


def test_torch_wins_and_kfd_is_not_asked(monkeypatch):
    monkeypatch.delenv("GITM_GPU_SKU", raising=False)
    monkeypatch.setattr(ctx, "_query_nvml", lambda: (None, None))
    monkeypatch.setattr(ctx, "_query_torch", lambda: ("AMD Instinct MI355X", 8))
    _kfd_must_not_be_asked(monkeypatch)

    assert ctx.build_planner_context(None).sku == "AMD Instinct MI355X"


def _torch_says(monkeypatch, name, count):
    monkeypatch.delenv("GITM_GPU_SKU", raising=False)
    monkeypatch.setattr(ctx, "_query_nvml", lambda: (None, None))
    monkeypatch.setattr(ctx, "_query_torch", lambda: (name, count))


@pytest.mark.parametrize("generic", ["AMD Radeon Graphics", ""])
def test_a_torch_name_with_no_peak_does_not_block_kfd(kfd, monkeypatch, generic):
    """ROCm torch can name a part generically, or not at all. Either is an
    answer to "what is it called" and no answer to "what are its peaks"."""
    _torch_says(monkeypatch, generic, 8)
    _mi355x_nodes(kfd)

    pctx = ctx.build_planner_context(None)

    assert pctx.sku == "AMD Instinct MI355X"
    assert pctx.peak is not None and pctx.peak.peak_bw_bytes_s == 8000e9


def test_a_generic_torch_name_is_kept_when_kfd_has_no_peak_either(kfd, monkeypatch):
    """Nothing prices the box, so keep the most specific human name: torch's
    marketing name over kfd's id-based one."""
    _torch_says(monkeypatch, "AMD Radeon Graphics", 1)
    _node(kfd, 1, 1001, MI350X_PROPERTIES)

    pctx = ctx.build_planner_context(None)

    assert pctx.sku == "AMD Radeon Graphics"
    assert pctx.peak is None


def test_kfd_names_the_box_when_torch_gave_an_empty_name(kfd, monkeypatch):
    _torch_says(monkeypatch, "", 1)
    _node(kfd, 1, 1001, MI350X_PROPERTIES)

    assert ctx.build_planner_context(None).sku == "AMD Instinct MI350X"


def test_an_empty_torch_name_and_no_kfd_is_no_sku(tmp_path, monkeypatch):
    monkeypatch.setattr(ctx, "_KFD_NODES", tmp_path / "absent")
    _torch_says(monkeypatch, "", 1)

    assert ctx.build_planner_context(None).sku is None


def test_an_nvml_name_with_no_peak_is_kept_when_kfd_has_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(ctx, "_KFD_NODES", tmp_path / "absent")
    monkeypatch.delenv("GITM_GPU_SKU", raising=False)
    monkeypatch.setattr(ctx, "_query_nvml", lambda: ("NVIDIA RTX 4090", 1))

    assert ctx.build_planner_context(None).sku == "NVIDIA RTX 4090"


def test_an_override_with_no_peak_is_still_the_override(monkeypatch):
    """The override is an instruction, not a guess: it stays even when NVML
    names a catalogued part."""
    monkeypatch.setenv("GITM_GPU_SKU", "SomeFutureGPU-9000")
    monkeypatch.setattr(ctx, "_query_nvml", lambda: ("NVIDIA H200", 2))
    _kfd_must_not_be_asked(monkeypatch)

    assert ctx.build_planner_context(_engine(2)).sku == "SomeFutureGPU-9000"


def test_kfd_counts_the_box_when_nothing_else_does(kfd, monkeypatch):
    """GITM_GPU_SKU names the part, so kfd is not needed for that, but with no
    engine and no NVML or torch count it is the only count there is."""
    monkeypatch.setenv("GITM_GPU_SKU", "MI355X")
    monkeypatch.setattr(ctx, "_query_nvml", lambda: (None, None))
    monkeypatch.setattr(ctx, "_query_torch", lambda: (None, None))
    _mi355x_nodes(kfd)

    pctx = ctx.build_planner_context(None)

    assert (pctx.sku, pctx.num_gpus) == ("MI355X", 8)
    assert pctx.gate.has_collective is True


def test_a_known_world_size_does_not_ask_kfd(monkeypatch):
    monkeypatch.setenv("GITM_GPU_SKU", "MI355X")
    monkeypatch.setattr(ctx, "_query_nvml", lambda: (None, None))
    _kfd_must_not_be_asked(monkeypatch)

    assert ctx.build_planner_context(_engine(1)).num_gpus == 1


def test_a_torch_count_is_asked_before_kfd(monkeypatch):
    monkeypatch.setenv("GITM_GPU_SKU", "MI355X")
    monkeypatch.setattr(ctx, "_query_nvml", lambda: (None, None))
    monkeypatch.setattr(ctx, "_query_torch", lambda: ("AMD Instinct MI355X", 4))
    _kfd_must_not_be_asked(monkeypatch)

    assert ctx.build_planner_context(None).num_gpus == 4
