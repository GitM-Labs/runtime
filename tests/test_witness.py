"""Witness checks on synthetic events. Nothing here has run on a real capture."""

from __future__ import annotations

import json
import os
import random
from collections import Counter

import pytest

from gitm.importers._common import LightKernel, LightMemcpy, LightSync
from gitm.optimizer import primitives as prim
from gitm.optimizer import witness
from gitm.optimizer.metrics import HardwarePeak, compute_metrics
from gitm.optimizer.witness import Header, account
from gitm.tracer._cupti_decode import decode_record
from gitm.tracer.capture import write_trace_jsonl
from gitm.tracer.schema import KernelEvent, MemcpyEvent, SyncEvent, Trace

MS = 1_000_000
S = 1_000_000_000
CUPTI = Header(source="cupti", device_count=1, vendor="nvidia")

#: Why a row may be ``cannot_tell``.
REASONS = ("channel_missing", "invalid_capture", "undivided_group", "too_short",
           "no_passive_witness", "not_built")


def K(start, end, name="gemm", *, dev=0, pid=None, stream=0):
    return KernelEvent(start_ns=start, end_ns=end, stream_id=stream, device_id=dev, name=name,
                       pid=pid)


def M(start, end, *, dev=0, pid=None, src="host", dst="device"):
    return MemcpyEvent(start_ns=start, end_ns=end, stream_id=0, device_id=dev, bytes=1, src=src,
                       dst=dst, pid=pid)


def Y(start, end, *, dev=0, pid=None, kind="stream"):
    return SyncEvent(start_ns=start, end_ns=end, stream_id=0, device_id=dev, sync_kind=kind,
                     pid=pid)


def steady(n=1000, *, period=2 * MS, busy=1 * MS, t0=0, **kw):
    """``n`` kernels of ``busy`` ns every ``period`` ns: a 50% busy window by default."""
    return [K(t0 + i * period, t0 + i * period + busy, **kw) for i in range(n)]


def rows(report, row_id):
    return [r for r in report.results if r["id"] == row_id]


def one(report, row_id):
    (row,) = rows(report, row_id)
    return row


def closure(report):
    """Check the account closes wherever it claims to; returns how many partitions that was."""
    checked = 0
    for info in report.partitions:
        if not info["closure_asserted"]:
            continue
        mine = [r for r in report.results
                if r["partition"] == {"pid": info["pid"], "device_id": info["device_id"]}
                and r["id"] in ("10", "11", "G1", "remainder")]
        assert len(mine) == 4
        total = info["busy_s"] + sum(r["seconds"] for r in mine)
        assert total == pytest.approx(info["window_s"], rel=1e-9)
        checked += 1
    return checked


# ── gap rows ─────────────────────────────────────────────────────────────────


def test_rows_10_11_g1_fire_and_account_closes():
    ev = steady(1000)  # 2 s window, 1 s busy, 1 ms gaps
    ev += [Y(i * 2 * MS + 1 * MS, i * 2 * MS + 1_100_000) for i in range(0, 1000, 2)]  # 50 ms
    ev += [M(i * 2 * MS + 1_500_000, i * 2 * MS + 1_600_000) for i in range(0, 1000, 2)]  # 50 ms
    # 3999 launch-scale gaps of 10 us after the steady part.
    t = 2000 * MS
    for _ in range(4000):
        ev.append(K(t, t + 10_000, stream=1))
        t += 20_000
    rep = account(ev, CUPTI)
    assert one(rep, "10")["state"] == "fired"
    assert one(rep, "10")["seconds"] == pytest.approx(0.05)
    assert one(rep, "10")["by_sync_kind"] == {"stream": pytest.approx(0.05)}
    assert one(rep, "11")["state"] == "fired"
    assert one(rep, "11")["by_direction"] == {"host->device": pytest.approx(0.05)}
    assert one(rep, "G1")["state"] == "fired"
    assert one(rep, "G1")["seconds"] == pytest.approx(0.03999)
    assert one(rep, "remainder")["state"] is None
    for r in (one(rep, "10"), one(rep, "11"), one(rep, "G1")):
        assert r["completeness"] == "unknown" and r["price_kind"] == "associated"
        assert r["thresholds_version"] == prim.THRESHOLDS_VERSION
    assert closure(rep) == 1
    assert rep.run_presence["10"] == "fired"


def test_rows_do_not_fire_on_a_clean_window():
    rep = account(steady(1000), CUPTI)
    for row_id in ("10", "11", "G1"):
        assert one(rep, row_id)["state"] == "not_fired"
        assert "candidate_levers" not in one(rep, row_id)
    assert one(rep, "remainder")["seconds"] == pytest.approx(0.999)
    assert closure(rep) == 1


def test_gap_shares_match_stall_breakdown():
    ev = steady(600) + [Y(1 * MS, 1_200_000), M(3 * MS, 3_400_000), K(1200 * MS - 5000, 1200 * MS)]
    trace = Trace(workload_id="w", fingerprint="f", run_id="r", device_count=1, vendor="nvidia",
                  captured_at_ns=0, duration_ns=1200 * MS, events=ev)
    b = compute_metrics(trace, HardwarePeak("T", 1e14, 1e10)).stall_breakdown
    rep = account(ev, CUPTI)
    assert one(rep, "11")["share"] == pytest.approx(b["transfer_bound"])
    assert one(rep, "10")["share"] == pytest.approx(b["sync_wait"])
    assert one(rep, "G1")["share"] == pytest.approx(b["launch_latency"])
    assert one(rep, "remainder")["share"] == pytest.approx(b["idle"])


@pytest.mark.parametrize("seed", range(25))
def test_closure_and_order_independence_on_random_events(seed):
    rng = random.Random(seed)
    ev, t = [], 0
    while t < 1500 * MS:
        length = rng.choice([5_000, 200_000, 3 * MS])
        ev.append(K(t, t + length))
        t += length + rng.choice([0, 4_000, 15_000, 30_000, 2 * MS])
    for _ in range(200):
        s = rng.randrange(t)
        ev.append(Y(s, s + rng.randrange(1, 3 * MS), kind=rng.choice(["stream", "event"])))
        s = rng.randrange(t)
        ev.append(M(s, s + rng.randrange(1, 3 * MS), dst=rng.choice(["device", "host"])))
    rep = account(ev, CUPTI)
    closure(rep)
    shuffled = ev[:]
    rng.shuffle(shuffled)
    again = account(shuffled, CUPTI)
    assert [dict(r, thresholds_version=None) for r in again.results] == [
        dict(r, thresholds_version=None) for r in rep.results]


def test_row_10_reports_sync_records_that_sit_over_busy_time():
    ev = [K(0, 1500 * MS), Y(100 * MS, 600 * MS)]
    r = one(account(ev, CUPTI), "10")
    assert r["state"] == "not_fired" and r["seconds"] == 0
    assert r["sync_record_s"] == pytest.approx(0.5) and r["sync_record_count"] == 1


def test_too_short_window():
    rep = account(steady(100), CUPTI)  # 0.2 s
    for row_id in ("10", "11", "G1", "22", "25"):
        r = one(rep, row_id)
        assert (r["state"], r["reason"]) == ("cannot_tell", "too_short")
        assert r["what"] and r["why"] and r["next"]
    assert one(rep, "remainder")["seconds"] is None


# ── capture validation ───────────────────────────────────────────────────────


@pytest.mark.parametrize("events, header, needle", [
    ([], CUPTI, "no events"),
    ([Y(0, 2 * S)], CUPTI, "no kernel event"),
    (steady(1000), Header(source="none"), "source is none"),
    (steady(1000) + [K(3 * S, 3 * S - 5)], CUPTI, "end before they start"),
    (steady(1000) + steady(1000, pid=7, dev=1), CUPTI, "some do not"),
])
def test_unusable_trace_gives_one_row_per_unit(events, header, needle):
    rep = account(events, header)
    for row_id in ("10", "11", "22", "25", "G1"):
        r = one(rep, row_id)
        assert (r["state"], r["reason"]) == ("cannot_tell", "invalid_capture")
        assert r["partition"]["all"] is True and needle in r["why"]
    assert {one(rep, g)["reason"] for g in ("G2", "G3", "G4")} == {"not_built"}
    assert one(rep, "8")["reason"] == "undivided_group"
    assert rep.partitions == []
    # A row this build does not evaluate keeps its own reason on an unusable trace.
    assert one(rep, "3")["reason"] == "not_built"


def test_overlap_on_one_stream_invalidates_that_partition_only():
    ev = steady(1000, pid=1, dev=0) + [K(500, 900, pid=1, dev=0)] + steady(1000, pid=2, dev=1)
    rep = account(ev, CUPTI)
    by_dev = {r["partition"]["device_id"]: r for r in rows(rep, "10")}
    assert by_dev[0]["reason"] == "invalid_capture" and "overlap" in by_dev[0]["why"]
    assert by_dev[1]["state"] == "not_fired"
    assert {r["partition"]["device_id"]: r["reason"] for r in rows(rep, "22")} == {
        0: "invalid_capture", 1: "channel_missing"}  # two devices, no collective names


def test_every_id_appears_once_per_instance_of_its_scope():
    ev = steady(1000, pid=1, dev=0) + steady(1000, pid=2, dev=0) + steady(1000, pid=1, dev=1)
    rep = account(ev, CUPTI)
    counts = Counter(r["id"] for r in rep.results)
    for p in prim.REGISTRY.values():
        if p.baseline:
            assert p.id not in counts
        else:
            assert counts[p.id] == {"partition": 3, "device": 2, "run": 1}[p.scope], p.id
    assert counts["remainder"] == 3
    assert [b["id"] for b in rep.baselines] == ["7", "16"]
    for r in rep.results:
        if r["state"] == "cannot_tell":
            assert r["reason"] in REASONS and r["what"] and r["why"] and r["next"], r["id"]


def test_unbuilt_groups_members_and_fixed_rows():
    rep = account(steady(1000), CUPTI)
    for g in ("G2", "G3", "G4"):
        r = one(rep, g)
        assert (r["state"], r["reason"], r["seconds"]) == ("cannot_tell", "not_built", None)
    for m in ("6", "8", "9", "12", "13", "14", "15", "17", "23"):
        assert one(rep, m)["reason"] == "undivided_group"
    assert one(rep, "1")["reason"] == one(rep, "21")["reason"] == "no_passive_witness"
    for fixed in ("18", "19", "20", "24"):  # row 24: this header has no drop count
        assert one(rep, fixed)["reason"] == "channel_missing"


# ── shared devices and records without a pid ─────────────────────────────────


def test_shared_device_carries_the_assumption():
    ev = steady(1000, pid=1) + steady(1000, pid=2, t0=1 * MS)
    rep = account(ev, CUPTI)
    assert len(rep.partitions) == 2
    for row_id in ("10", "11", "G1", "remainder"):
        for r in rows(rep, row_id):
            assert r["assumptions"] == ["other_pid_active_on_device"]
    assert "another traced process" in rows(rep, "remainder")[0]["name"]
    # Pids are pooled on a device row, so another process's collectives count there too.
    assert one(rep, "22")["assumptions"] == ["other_pid_active_on_device"]
    closure(rep)


def test_pidless_records_on_a_shared_device_make_four_rows_cannot_tell():
    ev = steady(1000, pid=1) + steady(1000, pid=2, t0=1 * MS) + [M(5 * MS, 6 * MS)]
    rep = account(ev, Header(source="nsys-import"))
    for row_id in ("10", "11", "G1"):
        for r in rows(rep, row_id):
            assert (r["state"], r["reason"]) == ("cannot_tell", "invalid_capture")
            assert "carry no pid" in r["why"]
    for r in rows(rep, "remainder"):
        assert r["seconds"] is None and r["reason"] == "invalid_capture"
    assert one(rep, "22")["state"] == "not_fired"


def test_pidless_record_follows_the_only_pid_on_its_device():
    ev = steady(1000, pid=1) + [M(i * 2 * MS + 1 * MS, i * 2 * MS + 1_100_000) for i in range(500)]
    rep = account(ev, CUPTI)
    assert one(rep, "11")["state"] == "fired"
    assert one(rep, "11")["seconds"] == pytest.approx(0.05)


def test_cupti_sync_record_follows_its_pid_not_its_device():
    ev = steady(1000, pid=1, dev=1)
    ev += [Y(i * 2 * MS + 1 * MS, i * 2 * MS + 1_100_000, pid=1, dev=0) for i in range(500)]
    r = one(account(ev, CUPTI), "10")
    assert r["partition"] == {"pid": 1, "device_id": 1}
    assert r["state"] == "fired" and r["seconds"] == pytest.approx(0.05)
    # Another source keeps the recorded device, where this pid has no kernels: the
    # records match no partition, so the gap rows cannot be trusted.
    other = account(ev, Header(source="nsys-import"))
    assert other.records_unplaced == 500
    assert one(other, "10")["reason"] == one(other, "11")["reason"] == "invalid_capture"


def test_cupti_sync_for_a_pid_on_two_devices():
    ev = steady(1000, pid=1, dev=0) + steady(1000, pid=1, dev=1)
    ev += [Y(1 * MS, 1_500_000, pid=1), M(3 * MS, 3_500_000, pid=1, dev=1)]
    rep = account(ev, CUPTI)
    for row_id in ("10", "G1"):
        for r in rows(rep, row_id):
            assert (r["state"], r["reason"]) == ("cannot_tell", "invalid_capture")
            assert "carry no device" in r["why"]
    assert all(r["seconds"] is None for r in rows(rep, "remainder"))
    row11 = {r["partition"]["device_id"]: r for r in rows(rep, "11")}
    assert row11[0]["seconds"] == 0 and row11[1]["seconds"] == pytest.approx(0.0005)
    assert not any(i["closure_asserted"] for i in rep.partitions)


# ── rows 22 and 25 ───────────────────────────────────────────────────────────


def collectives(name, *, overlapped, n=500):
    """50 ms of ``name`` kernels on stream 1, inside the gaps of ``steady`` or over its kernels."""
    off = 100_000 if overlapped else 1_100_000
    return [K(i * 2 * MS + off, i * 2 * MS + off + 100_000, name, stream=1) for i in range(n)]


@pytest.mark.parametrize("name", ["cross_device_reduce_1stage", "cross_device_reduce_2stage",
                                  "ncclDevKernel_AllReduce_Sum_f32"])
def test_row_22_counts_exposed_all_reduce_on_a_single_device_header(name):
    rep = account(steady(1000) + collectives(name, overlapped=False), CUPTI)
    r = one(rep, "22")
    assert r["state"] == "fired" and r["seconds"] == pytest.approx(0.05)
    assert r["scope"] == "device" and r["partition"] == {"pid": None, "device_id": 0}
    assert r["candidate_levers"] == [] and "no lever" in r["levers_note"]
    assert one(rep, "25")["state"] == "not_fired"


def test_row_22_overlapped_by_compute_does_not_fire():
    r = one(account(steady(1000) + collectives("ncclDevKernel_AllReduce", overlapped=True), CUPTI),
            "22")
    assert r["state"] == "not_fired" and r["seconds"] == 0 and r["n_kernels"] == 500
    assert r["completeness"] == "unknown"


def test_row_25_counts_all_to_all_only_and_lists_one_lever():
    rep = account(steady(1000) + collectives("ncclDevKernel_AllToAll", overlapped=False), CUPTI)
    r = one(rep, "25")
    assert r["state"] == "fired" and r["seconds"] == pytest.approx(0.05)
    assert [v["name"] for v in r["candidate_levers"]] == ["enable_dbo"]
    assert r["candidate_levers"][0]["note"] == "applicability not checked"
    assert one(rep, "22")["seconds"] == 0


def test_broadcast_to_is_in_neither_and_the_two_classes_do_not_hide_each_other():
    ev = steady(1000) + collectives("ncclDevKernel_AllReduce", overlapped=False)
    ev += [K(k.start_ns, k.end_ns, "ncclDevKernel_AllToAll", stream=2)
           for k in collectives("x", overlapped=False)]
    ev += [K(i * 2 * MS + 1_500_000, i * 2 * MS + 1_600_000, "elementwise_kernel<broadcast_to>",
             stream=3) for i in range(500)]
    rep = account(ev, CUPTI)
    assert one(rep, "22")["seconds"] == pytest.approx(0.05)
    assert one(rep, "25")["seconds"] == pytest.approx(0.05)


def test_no_collective_names_on_single_and_multi_device_headers():
    single = account(steady(1000), CUPTI)
    assert one(single, "22")["state"] == "not_fired"
    assert one(single, "22")["completeness"] == "unknown"
    multi = account(steady(1000), Header(source="cupti", device_count=2))
    for row_id in ("22", "25"):
        r = one(multi, row_id)
        assert (r["state"], r["reason"]) == ("cannot_tell", "channel_missing")
        assert "no collective kernel names recognised" in r["what"]


# ── levers ───────────────────────────────────────────────────────────────────


def test_fired_group_lists_levers_under_their_members():
    ev, t = [], 0
    while t < 1100 * MS:
        ev.append(K(t, t + 10_000))
        t += 20_000
    r = one(account(ev, CUPTI), "G1")
    assert r["state"] == "fired"
    by_member = {m["member"]: m for m in r["candidate_levers"]}
    assert set(by_member) == {"8", "23"}
    assert [v["name"] for v in by_member["8"]["levers"]] == ["cuda_graphs_enable",
                                                             "max_seq_len_to_capture_dynamic"]
    assert by_member["8"]["levers"][0] == {"name": "cuda_graphs_enable", "review": None,
                                           "note": "applicability not checked"}
    assert by_member["23"]["levers"] == []
    assert "cannot tell which member" in r["levers_note"]


# ── importer events, files, command ──────────────────────────────────────────


def test_importer_event_classes_without_a_pid_attribute():
    ev = [LightKernel("gemm", i * 2 * MS, i * 2 * MS + MS) for i in range(1000)]
    ev += [LightSync(i * 2 * MS + MS, i * 2 * MS + 1_100_000, 0, 0, "stream") for i in range(500)]
    ev += [LightMemcpy(i * 2 * MS + 1_500_000, i * 2 * MS + 1_600_000, 0, 0, 1, "host", "device")
           for i in range(500)]
    rep = account(ev, Header(source="nsys-import"))
    assert one(rep, "10")["state"] == one(rep, "11")["state"] == "fired"
    assert one(rep, "10")["partition"] == {"pid": None, "device_id": 0}
    assert one(rep, "22")["completeness"] == "unknown"


@pytest.mark.parametrize("source", ["cupti", "rocprof", "nsys-import", "torch-import"])
def test_trace_rows_are_not_complete_without_a_drop_count(source):
    rep = account(steady(1000), Header(source=source))
    # Row 11 with no memcpy record at all is cannot_tell on nsys and rocprof, which has no
    # completeness.
    seen = {one(rep, i)["completeness"] for i in ("11", "G1", "22")}
    assert seen == ({None, "unknown"} if source in ("nsys-import", "rocprof") else {"unknown"})


def write_run(tmp_path, events, **files):
    trace = Trace(workload_id="w", fingerprint="f", run_id="r", device_count=1, vendor="nvidia",
                  captured_at_ns=0, duration_ns=max(e.end_ns for e in events), events=events)
    write_trace_jsonl(tmp_path / "trace.jsonl", trace)
    for name, text in files.items():
        (tmp_path / name).write_text(text)
    return tmp_path


@pytest.mark.parametrize("source", ["rocprof", "torch-import"])
def test_row_10_is_channel_missing_on_sources_with_no_sync_records(source):
    rep = account(steady(1000, pid=1), Header(source=source))
    r = one(rep, "10")
    assert (r["state"], r["reason"]) == ("cannot_tell", "channel_missing")
    assert source in r["why"] and r["seconds"] is None
    assert one(rep, "G1")["state"] == "not_fired"
    # No memcpy record at all: rocprof may not have recorded them, the torch importer builds them.
    assert one(rep, "11")["state"] == ("cannot_tell" if source == "rocprof" else "not_fired")


def test_row_10_on_an_nsys_import_needs_a_sync_record_somewhere():
    without = account(steady(1000), Header(source="nsys-import"))
    assert one(without, "10")["reason"] == "channel_missing"
    with_one = account(steady(1000) + [Y(MS, MS + 100)], Header(source="nsys-import"))
    assert one(with_one, "10")["state"] == "not_fired"
    assert one(account(steady(1000), CUPTI), "10")["state"] == "not_fired"


def test_rows_22_and_25_count_the_devices_seen_not_only_the_header():
    ev = steady(1000, pid=1, dev=0) + steady(1000, pid=2, dev=1)
    rep = account(ev, Header(source="cupti", device_count=1))
    for r in rows(rep, "22") + rows(rep, "25"):
        assert (r["state"], r["reason"]) == ("cannot_tell", "channel_missing")
        assert "2 devices" in r["why"]


@pytest.mark.parametrize("source", ["None", "CUPTI", "nsys"])
def test_unknown_source_is_not_trusted(source, tmp_path):
    rep = account(steady(1000), Header(source=source))
    r = one(rep, "10")
    assert (r["reason"], r["partition"]["all"]) == ("invalid_capture", True)
    assert "source" in r["why"]
    run_dir = write_run(tmp_path, steady(1000))
    trace = run_dir / "trace.jsonl"
    trace.write_text(trace.read_text().replace('"source": "cupti"', f'"source": "{source}"'))
    with pytest.raises(witness.WitnessInputError, match="source"):
        witness.run(run_dir)


def test_pidless_cupti_sync_on_a_shared_device_makes_four_rows_cannot_tell():
    ev = steady(1000, pid=1) + steady(1000, pid=2, t0=MS) + [Y(5 * MS, 6 * MS)]
    ev += [M(i * 2 * MS + 1_100_000, i * 2 * MS + 1_900_000, pid=1) for i in range(500)]
    rep = account(ev, CUPTI)
    for row_id in ("10", "11", "G1"):
        for r in rows(rep, row_id):
            assert (r["state"], r["reason"]) == ("cannot_tell", "invalid_capture")
            assert "carry no pid" in r["why"]
    assert all(r["seconds"] is None for r in rows(rep, "remainder"))


def test_pidless_cupti_sync_with_two_single_device_pids_names_the_missing_pid():
    ev = steady(1000, pid=1, dev=0) + steady(1000, pid=2, dev=1) + [Y(5 * MS, 6 * MS)]
    for r in rows(account(ev, CUPTI), "10"):
        assert (r["state"], r["reason"]) == ("cannot_tell", "invalid_capture")
        assert "neither a pid" in r["why"] and "several devices" not in r["why"]


# ── the window is the kernel span ────────────────────────────────────────────


def test_a_stray_record_far_from_the_kernels_does_not_dilute_shares():
    ev = steady(5000, pid=1)  # 10 s
    ev += [Y(i * 2 * MS + MS, i * 2 * MS + 1_200_000, pid=1) for i in range(5000)]  # 10%
    clean = one(account(ev, CUPTI), "10")
    stray = account(ev + [M(-600 * S, -600 * S + MS, pid=1), Y(900 * S, 901 * S, pid=1)], CUPTI)
    assert clean["state"] == one(stray, "10")["state"] == "fired"
    assert one(stray, "10")["share"] == pytest.approx(clean["share"])
    assert stray.partitions[0]["window_s"] == pytest.approx(9.999)
    assert one(stray, "11")["seconds"] == 0
    # 4999 syncs of 0.2 ms inside the span; the 1 s stray one is clipped away.
    assert one(stray, "10")["sync_record_s"] == pytest.approx(0.9998)
    assert one(stray, "10")["sync_record_count"] == 4999
    closure(stray)


def test_row_10_on_a_short_window_without_a_sync_channel_names_the_channel():
    r = one(account(steady(100, pid=1), Header(source="rocprof")), "10")
    assert (r["state"], r["reason"]) == ("cannot_tell", "channel_missing")


def test_written_files_honour_the_umask(tmp_path):
    from gitm.optimizer import separation

    old = os.umask(0o022)
    try:
        witness._write_atomic(tmp_path / "witness.json", "{}")
        separation._write_atomic(tmp_path / "decision.json", "{}")
    finally:
        os.umask(old)
    assert {(tmp_path / n).stat().st_mode & 0o777 for n in ("witness.json", "decision.json")} == {
        0o644}


# ── context and raw sync type ────────────────────────────────────────────────


def test_two_contexts_may_each_have_the_same_stream_number():
    a = [K(i * 2 * MS, i * 2 * MS + MS, pid=1).model_copy(update={"context_id": 1})
         for i in range(1000)]
    b = [K(i * 2 * MS + 500_000, i * 2 * MS + 1_500_000, pid=1).model_copy(
        update={"context_id": 2}) for i in range(1000)]
    rep = account(a + b, CUPTI)
    assert one(rep, "10")["state"] == "not_fired"  # overlapping in time, but not on one stream
    same_context = [e.model_copy(update={"context_id": 1}) for e in b]
    assert one(account(a + same_context, CUPTI), "10")["reason"] == "invalid_capture"
    # With no context reported, the stream number alone is the key, as before.
    plain = [e.model_copy(update={"context_id": None}) for e in a + b]
    assert one(account(plain, CUPTI), "10")["reason"] == "invalid_capture"


def test_row_10_splits_by_the_collectors_own_sync_type(tmp_path):
    ev = steady(1000, pid=1)
    for i in range(500):
        t = i * 4 * MS + MS
        ev.append(Y(t, t + 100_000, pid=1).model_copy(update={"sync_type": 2}))
        ev.append(Y(t + 2 * MS, t + 2 * MS + 300_000, pid=1).model_copy(update={"sync_type": 3}))
    r = one(account(ev, CUPTI), "10")
    assert r["by_sync_kind"] == {"stream_synchronize": pytest.approx(0.1497, rel=0.01),
                                 "stream_wait_event": pytest.approx(0.05, rel=0.01)}
    assert r["seconds"] == pytest.approx(sum(r["by_sync_kind"].values()))
    # Through the file: the fields survive trace.jsonl.
    from_file = one(witness.run(write_run(tmp_path, ev)), "10")
    assert from_file["by_sync_kind"] == r["by_sync_kind"]
    # A record without the raw type falls back to the folded kind.
    assert set(one(account(steady(1000) + [Y(MS, MS + 100_000)], CUPTI), "10")[
        "by_sync_kind"]) == {"stream"}


def test_sync_labels_cover_every_raw_type_and_keep_an_unknown_one():
    ev = steady(1000, pid=1)
    for i, raw in enumerate((1, 4, 9, True)):
        t = i * 4 * MS + MS
        ev.append(Y(t, t + 100_000, pid=1, kind="device").model_copy(update={"sync_type": raw}))
    assert set(one(account(ev, CUPTI), "10")["by_sync_kind"]) == {
        "event_synchronize", "context_synchronize", "type_9", "device"}


# ── a missing record type is not zero time ───────────────────────────────────


@pytest.mark.parametrize("source", ["torch-import", "rocprof", "nsys-import"])
def test_g1_and_the_remainder_say_when_sync_records_cannot_be_seen(source):
    # 10 us gaps: all launch-scale, and all of it could sit under sync records the source drops.
    ev = [K(i * 20_000, i * 20_000 + 10_000) for i in range(100_000)]
    rep = account(ev, Header(source=source, device_count=1, vendor="nvidia"))
    g1 = one(rep, "G1")
    assert g1["state"] == "fired" and "no_sync_records_in_source" in g1["assumptions"]
    assert g1["status"] == "not_identifiable_from_this_data"
    assert "no_sync_records_in_source" in one(rep, "remainder")["assumptions"]
    # With the channel present the same gaps are an estimate.
    assert one(account(ev, CUPTI), "G1")["status"] == "estimated"


def test_row_11_on_an_nsys_import_needs_a_memcpy_record_somewhere():
    nsys = Header(source="nsys-import", device_count=1, vendor="nvidia")
    without = account(steady(1000) + [Y(MS, MS + 100)], nsys)
    r = one(without, "11")
    assert (r["state"], r["reason"], r["seconds"]) == ("cannot_tell", "channel_missing", None)
    for row_id in ("10", "G1", "remainder"):
        assert "no_memcpy_records_in_source" in one(without, row_id)["assumptions"]
    with_one = account(steady(1000) + [M(MS, MS + 100), Y(MS, MS + 100)], nsys)
    assert one(with_one, "11")["state"] == "not_fired"
    assert one(with_one, "G1")["assumptions"] == []


# ── windows, contexts and boundaries ─────────────────────────────────────────


def test_a_zero_length_kernel_far_away_does_not_stretch_the_window():
    ev = steady(1000) + [Y(i * 2 * MS + MS, i * 2 * MS + MS + 50_000) for i in range(1000)]
    plain = account(ev, CUPTI)
    far = account(ev + [K(20 * S, 20 * S, "noop")], CUPTI)
    assert one(plain, "10")["state"] == "fired"
    assert one(far, "10")["state"] == "fired"
    assert far.partitions[0]["window_s"] == plain.partitions[0]["window_s"]
    assert one(far, "remainder")["seconds"] == one(plain, "remainder")["seconds"]
    assert closure(far) == 1


def test_a_partition_of_zero_length_kernels_only_is_invalid():
    rep = account([K(0, 0), K(2 * S, 2 * S)], CUPTI)
    for row_id in ("10", "11", "G1", "22", "25"):
        assert (one(rep, row_id)["state"], one(rep, row_id)["reason"]) == (
            "cannot_tell", "invalid_capture"), row_id


def ctx_kernel(start, end, ctx, *, dev=0, pid=1):
    return KernelEvent(start_ns=start, end_ns=end, stream_id=0, device_id=dev, name="gemm",
                       pid=pid, context_id=ctx)


def ctx_sync(start, end, ctx, *, pid=1):
    return SyncEvent(start_ns=start, end_ns=end, stream_id=0, device_id=0, sync_kind="stream",
                     pid=pid, context_id=ctx)


def test_a_cupti_sync_from_a_context_with_no_kernels_is_not_placed_by_pid():
    ev = [ctx_kernel(i * 2 * MS, i * 2 * MS + MS, 1) for i in range(1000)]
    ev += [ctx_sync(i * 2 * MS + MS, i * 2 * MS + 2 * MS, 2) for i in range(999)]
    rep = account(ev, CUPTI)
    for row_id in ("10", "G1"):
        r = one(rep, row_id)
        assert (r["state"], r["reason"]) == ("cannot_tell", "invalid_capture"), row_id
    assert "context" in one(rep, "10")["why"]
    # Row 11 does not depend on where a sync record belongs.
    assert one(rep, "11")["state"] == "not_fired"


def test_a_cupti_sync_follows_its_context_to_one_of_its_pids_devices():
    ev = [ctx_kernel(i * 2 * MS, i * 2 * MS + MS, 1, dev=0) for i in range(1000)]
    ev += [ctx_kernel(i * 2 * MS, i * 2 * MS + MS, 2, dev=1) for i in range(1000)]
    ev += [ctx_sync(i * 2 * MS + MS, i * 2 * MS + MS + 100_000, 2) for i in range(999)]
    rep = account(ev, Header(source="cupti", device_count=2, vendor="nvidia"))
    by_dev = {r["partition"]["device_id"]: r for r in rows(rep, "10")}
    assert by_dev[1]["state"] == "fired" and by_dev[1]["seconds"] == pytest.approx(0.0999)
    assert by_dev[0]["state"] == "not_fired" and by_dev[0]["seconds"] == 0


def test_a_share_of_exactly_the_threshold_fires():
    # 10,280,000 / 1,028,000,000 is exactly 1%, and 0.009999999999999998 as a float quotient.
    ev = [K(0, 500 * MS), M(500 * MS, 510_280_000), K(510_280_000, 1_028_000_000)]
    r = one(account(ev, CUPTI), "11")
    assert r["seconds"] == pytest.approx(0.01028)
    assert r["state"] == "fired"
    below = [K(0, 500 * MS), M(500 * MS, 510_279_999), K(510_279_999, 1_028_000_000 - 1)]
    assert one(account(below, CUPTI), "11")["state"] == "not_fired"


def test_a_window_of_exactly_the_minimum_is_evaluated():
    assert one(account([K(0, 490 * MS), K(500 * MS, S)], CUPTI), "10")["state"] == "not_fired"
    assert one(account([K(0, 490 * MS), K(500 * MS, S - 1)], CUPTI), "10")["reason"] == "too_short"


def test_abutting_kernels_on_one_stream_are_not_an_overlap():
    rep = account([K(i * MS, (i + 1) * MS) for i in range(1500)], CUPTI)
    assert one(rep, "10")["state"] == "not_fired"
    assert closure(rep) == 1


def test_run_presence_prefers_cannot_tell_over_not_fired():
    ev = steady(1000, pid=1, dev=0) + [K(500, 900, pid=1, dev=0)] + steady(1000, pid=2, dev=1)
    for events in (ev, list(reversed(ev))):
        assert account(events, CUPTI).run_presence["10"] == "cannot_tell"


def test_a_zero_length_kernel_does_not_split_an_idle_gap():
    ev = [K(0, 600 * MS), K(600 * MS + 30_000, 600 * MS + 30_000), K(600 * MS + 45_000, 1200 * MS)]
    rep = account(ev, CUPTI)
    assert one(rep, "G1")["seconds"] == 0
    assert one(rep, "remainder")["seconds"] == pytest.approx(45e-6)


# ── what a file can force ────────────────────────────────────────────────────


def test_many_distinct_labels_are_folded_and_the_split_still_adds_up():
    ev = steady(1000)
    ev += [SyncEvent(start_ns=i * 2 * MS + MS, end_ns=i * 2 * MS + MS + 10_000, stream_id=0,
                     device_id=0, sync_kind="stream", sync_type=100 + i) for i in range(1000)]
    r = one(account(ev, CUPTI), "10")
    split = r["by_sync_kind"]
    assert len(split) == witness._MAX_LABELS + 1
    other = [k for k in split if k.startswith("other (")]
    assert other == [f"other ({1000 - witness._MAX_LABELS} labels)"]
    assert sum(split.values()) == pytest.approx(r["seconds"])


def test_a_label_as_long_as_a_line_is_cut_in_the_output():
    ev = steady(1000) + [witness._Event("memcpy", MS, MS + 100_000, 0, 0, None, "x" * 5000)]
    (label,) = one(account(ev, CUPTI), "11")["by_direction"]
    assert len(label) < 100


def test_more_partitions_than_any_capture_has_is_a_fault_not_a_row_each():
    ev = [K(i * 10, i * 10 + 5, dev=i) for i in range(witness._MAX_PARTITIONS + 1)]
    rep = account(ev, CUPTI)
    assert rep.partitions == []
    r = one(rep, "10")
    assert (r["state"], r["reason"]) == ("cannot_tell", "invalid_capture")
    assert "partitions" in r["why"]


@pytest.mark.parametrize("extra", [{"device_id": True}, {"device_id": "1"}, {"stream_id": 1.9},
                                   {"pid": 7.9}, {"context_id": "3"}, {"start_ns": 0.5}])
def test_a_wrong_typed_field_in_an_event_line_exits_2(tmp_path, capsys, extra):
    run_dir = write_run(tmp_path, steady(1000))
    line = {"kind": "kernel", "name": "k", "start_ns": 10**12, "end_ns": 10**12 + 5,
            "stream_id": 0, "device_id": 0, **extra}
    with (run_dir / "trace.jsonl").open("a") as fh:
        fh.write(json.dumps(line) + "\n")
    assert witness.main([str(run_dir)]) == 2
    assert "did not run" in capsys.readouterr().err


@pytest.mark.parametrize("missing", ["device_id", "stream_id"])
def test_an_event_line_without_a_device_or_stream_exits_2(tmp_path, capsys, missing):
    run_dir = write_run(tmp_path, steady(1000))
    line = {"kind": "memcpy", "src": "host", "dst": "device", "start_ns": MS, "end_ns": 2 * MS,
            "stream_id": 0, "device_id": 0}
    del line[missing]
    with (run_dir / "trace.jsonl").open("a") as fh:
        fh.write(json.dumps(line) + "\n")
    assert witness.main([str(run_dir)]) == 2
    assert "did not run" in capsys.readouterr().err


def test_an_empty_trace_file_exits_2(tmp_path, capsys):
    (tmp_path / "trace.jsonl").write_text("")
    assert witness.main([str(tmp_path)]) == 2
    assert "no trace header" in capsys.readouterr().err


@pytest.mark.parametrize("device_count", [1.9, "2", True])
def test_a_header_device_count_that_is_not_an_integer_exits_2(tmp_path, capsys, device_count):
    run_dir = write_run(tmp_path, steady(1000))
    lines = (run_dir / "trace.jsonl").read_text().splitlines()
    header = json.loads(lines[0])
    header["_header"]["device_count"] = device_count
    (run_dir / "trace.jsonl").write_text("\n".join([json.dumps(header), *lines[1:]]) + "\n")
    assert witness.main([str(run_dir)]) == 2
    assert "no trace header" in capsys.readouterr().err


def test_a_zero_length_kernel_never_makes_the_result_depend_on_event_order():
    # Same start as a real kernel on its stream, and one inside a real kernel.
    ev = [K(0, 500 * MS), K(500 * MS + 10, 2 * S), K(500 * MS + 10, 500 * MS + 10, "marker"),
          K(S, S, "marker")]
    for events in (ev, list(reversed(ev)), [ev[2], ev[0], ev[3], ev[1]]):
        rep = account(events, CUPTI)
        assert one(rep, "G1")["state"] == "not_fired", [e.name for e in events]
        assert closure(rep) == 1


def test_a_zero_length_kernel_far_away_does_not_stretch_the_device_window():
    ev = steady(1000) + collectives("ncclDevKernel_AllReduce_Sum_f32", overlapped=False)
    plain = one(account(ev, CUPTI), "22")
    far = one(account(ev + [K(50 * S, 50 * S, "noop")], CUPTI), "22")
    assert plain["state"] == "fired"
    assert (far["state"], far["share"], far["device_window_s"]) == (
        plain["state"], plain["share"], plain["device_window_s"])


def test_a_stray_sync_from_an_untraced_context_only_matters_inside_the_window():
    ev = [ctx_kernel(i * 2 * MS, i * 2 * MS + MS, 1) for i in range(1000)]
    ev += [ctx_sync(i * 2 * MS + MS, i * 2 * MS + MS + 100_000, 1) for i in range(999)]
    expected = one(account(ev, CUPTI), "10")["seconds"]
    for stray in (ctx_sync(100 * S, 100 * S + 1, 2), ctx_sync(MS, MS, 2)):
        r = one(account(ev + [stray], CUPTI), "10")
        assert (r["state"], r["seconds"]) == ("fired", expected)
    inside = one(account(ev + [ctx_sync(MS, MS + 50, 2)], CUPTI), "10")
    assert (inside["state"], inside["reason"]) == ("cannot_tell", "invalid_capture")


# ── records that cover no gap decide nothing ─────────────────────────────────


def test_a_pidless_cupti_sync_from_another_context_is_not_placed():
    for pid in (1, None):
        ev = [ctx_kernel(i * 2 * MS, i * 2 * MS + MS, 5, pid=pid) for i in range(1000)]
        r = one(account(ev + [ctx_sync(MS, 1900 * MS, 99, pid=None)], CUPTI), "10")
        assert (r["state"], r["reason"]) == ("cannot_tell", "invalid_capture"), pid
        ok = one(account(ev + [ctx_sync(MS, MS + 400_000, 5, pid=None)], CUPTI), "10")
        assert ok["state"] == "not_fired" and ok["seconds"] == pytest.approx(0.0004), pid


@pytest.mark.parametrize("stray", [M(100 * S, 101 * S, dev=5), M(500 * MS, 500 * MS, dev=5),
                                   Y(100 * S, 101 * S, dev=5, pid=7)])
def test_an_unplaceable_record_outside_every_gap_changes_nothing(stray):
    plain = account(steady(1000, pid=1), CUPTI)
    rep = account(steady(1000, pid=1) + [stray], CUPTI)
    for row_id in ("10", "11", "G1", "remainder"):
        assert one(rep, row_id)["state"] == one(plain, row_id)["state"], row_id
        assert one(rep, row_id)["seconds"] == one(plain, row_id)["seconds"], row_id
    assert rep.records_unplaced == 1
    inside = account(steady(1000, pid=1) + [M(MS, MS + 50, dev=5)], CUPTI)
    assert one(inside, "11")["reason"] == "invalid_capture"


def test_a_contextless_sync_for_a_pid_on_two_devices_only_matters_inside_a_window():
    ev = steady(1000, pid=1, dev=0) + steady(1000, pid=1, dev=1)
    for stray in (Y(100 * S, 101 * S, pid=1), Y(MS, MS, pid=1)):
        rep = account(ev + [stray], Header(source="cupti", device_count=2, vendor="nvidia"))
        assert {r["state"] for r in rows(rep, "10")} == {"not_fired"}
    rep = account(ev + [Y(MS, MS + 50, pid=1)], Header(source="cupti", device_count=2,
                                                      vendor="nvidia"))
    assert {r["reason"] for r in rows(rep, "10")} == {"invalid_capture"}


def test_the_label_fold_cannot_overwrite_a_real_label_and_long_labels_are_not_added_twice():
    ev = steady(1000)
    ev += [SyncEvent(start_ns=i * 2 * MS + MS, end_ns=i * 2 * MS + MS + 1000, stream_id=0,
                     device_id=0, sync_kind="stream", sync_type=100 + i % 40) for i in range(400)]
    real = "other (9 labels)"  # 41 labels, 32 kept: the name the fold of the other 9 would take
    ev += [witness._Event("sync", (500 + i) * 2 * MS + MS, (500 + i) * 2 * MS + MS + 1000, 0, 0,
                          None, real) for i in range(50)]
    r = one(account(ev, CUPTI), "10")
    assert r["by_sync_kind"][real] == pytest.approx(50e-6)
    assert r["by_sync_kind"][real + "*"] == pytest.approx(90e-6)
    assert sum(r["by_sync_kind"].values()) == pytest.approx(r["seconds"])
    # Two labels that share their first 80 characters, over the same gap.
    twins = [witness._Event("memcpy", MS, MS + 30_000, 0, 0, None, "x" * 90 + tail)
             for tail in ("a", "b")]
    r = one(account(steady(1000) + twins, CUPTI), "11")
    assert sum(r["by_direction"].values()) == pytest.approx(r["seconds"]) == pytest.approx(30e-6)


# ── branches the fixes added ─────────────────────────────────────────────────


def test_a_cupti_sync_with_a_context_follows_its_pid_when_kernels_name_none():
    ev = steady(1000, pid=1) + [ctx_sync(i * 2 * MS + MS, i * 2 * MS + MS + 100_000, 9)
                                for i in range(999)]
    r = one(account(ev, CUPTI), "10")
    assert r["state"] == "fired" and r["seconds"] == pytest.approx(0.0999)


def test_a_sync_naming_another_pids_context_is_not_placed_there():
    ev = [ctx_kernel(i * 2 * MS, i * 2 * MS + MS, 1, dev=0, pid=1) for i in range(1000)]
    ev += [ctx_kernel(i * 2 * MS, i * 2 * MS + MS, 2, dev=1, pid=2) for i in range(1000)]
    ev += [ctx_sync(i * 2 * MS + MS, i * 2 * MS + MS + 100_000, 2, pid=1) for i in range(999)]
    by_pid = {r["partition"]["pid"]: r for r in rows(account(ev, CUPTI), "10")}
    assert by_pid[1]["reason"] == "invalid_capture" and by_pid[2]["seconds"] == 0


def test_a_context_with_kernels_on_two_devices_cannot_place_a_sync():
    ev = [ctx_kernel(i * 2 * MS, i * 2 * MS + MS, 1, dev=d) for d in (0, 1) for i in range(1000)]
    rep = account(ev + [ctx_sync(MS, MS + 100_000, 1)],
                  Header(source="cupti", device_count=2, vendor="nvidia"))
    for r in rows(rep, "10"):
        assert r["reason"] == "invalid_capture" and "carry no device" in r["why"]


def test_a_stray_sync_that_ends_before_the_window_changes_nothing():
    ev = [ctx_kernel(S + i * 2 * MS, S + i * 2 * MS + MS, 1) for i in range(1000)]
    ev += [ctx_sync(S + i * 2 * MS + MS, S + i * 2 * MS + MS + 100_000, 1) for i in range(999)]
    r = one(account(ev + [ctx_sync(0, S, 2)], CUPTI), "10")
    assert r["state"] == "fired" and r["seconds"] == pytest.approx(0.0999)


def test_the_sync_split_leaves_out_time_already_under_a_copy():
    ev = steady(1000) + [Y(MS, MS + 400_000), M(MS + 100_000, MS + 200_000)]
    r = one(account(ev, CUPTI), "10")
    assert r["seconds"] == pytest.approx(0.0003)
    assert sum(r["by_sync_kind"].values()) == pytest.approx(r["seconds"])


def test_the_label_fold_keeps_the_labels_with_most_records():
    ev = steady(1000)
    ev += [SyncEvent(start_ns=i * 2 * MS + MS, end_ns=i * 2 * MS + MS + 1000, stream_id=0,
                     device_id=0, sync_kind="stream", sync_type=100 + i) for i in range(40)]
    ev += [SyncEvent(start_ns=(100 + i) * 2 * MS + MS, end_ns=(100 + i) * 2 * MS + MS + 1000,
                     stream_id=0, device_id=0, sync_kind="stream", sync_type=999)
           for i in range(5)]
    split = one(account(ev, CUPTI), "10")["by_sync_kind"]
    assert split["type_999"] == pytest.approx(5e-6)  # sorts last by name, kept by count


def test_exactly_the_partition_cap_is_still_accounted():
    ev = [K(i * 10, i * 10 + 5, dev=i) for i in range(witness._MAX_PARTITIONS)]
    assert len(account(ev, CUPTI).partitions) == witness._MAX_PARTITIONS


def test_a_wrong_typed_end_time_exits_2(tmp_path, capsys):
    run_dir = write_run(tmp_path, steady(1000))
    with (run_dir / "trace.jsonl").open("a") as fh:
        fh.write(json.dumps({"kind": "kernel", "name": "k", "start_ns": 10**12, "end_ns": 0.5,
                             "stream_id": 0, "device_id": 0}) + "\n")
    assert witness.main([str(run_dir)]) == 2
    assert "did not run" in capsys.readouterr().err


def test_a_short_nsys_window_without_memcpy_records_names_the_channel():
    rep = account(steady(100) + [Y(MS, MS + 100)],
                  Header(source="nsys-import", device_count=1, vendor="nvidia"))
    assert one(rep, "11")["reason"] == "channel_missing"
    assert one(rep, "10")["reason"] == "too_short"


@pytest.mark.parametrize("source", ["rocprof", "torch-import"])
def test_sync_records_under_a_source_that_has_none_are_not_used(source):
    header = Header(source=source, device_count=1, vendor="nvidia")
    plain = account(steady(1000), header)
    with_syncs = account(steady(1000) + [Y(i * 2 * MS + MS, i * 2 * MS + 2 * MS)
                                         for i in range(999)], header)
    for row_id in ("G1", "remainder"):
        assert one(with_syncs, row_id)["seconds"] == one(plain, row_id)["seconds"]


def test_a_pidless_record_outside_every_gap_changes_nothing_on_a_shared_device():
    ev = steady(1000, pid=1) + steady(1000, pid=2, stream=1)
    plain = account(ev, CUPTI)
    for stray in (M(100 * S, 101 * S), M(MS, MS), Y(100 * S, 101 * S), Y(MS, MS)):
        rep = account(ev + [stray], CUPTI)
        for a, b in zip(rows(plain, "11") + rows(plain, "10"), rows(rep, "11") + rows(rep, "10"),
                        strict=True):
            assert (a["state"], a["seconds"]) == (b["state"], b["seconds"])
    for inside in (M(MS, MS + 50), Y(MS, MS + 50)):
        rep = account(ev + [inside], CUPTI)
        assert {r["reason"] for r in rows(rep, "11")} == {"invalid_capture"}


def test_the_reason_for_an_unplaceable_sync_does_not_depend_on_event_order():
    ev = [ctx_kernel(i * 2 * MS, i * 2 * MS + MS, 1 + d, dev=d) for d in (0, 1)
          for i in range(1000)]
    strays = [ctx_sync(MS, MS + 50, 99),
              SyncEvent(start_ns=3 * MS, end_ns=3 * MS + 50, stream_id=0, device_id=0,
                        sync_kind="stream", pid=1)]
    header = Header(source="cupti", device_count=2, vendor="nvidia")
    a = [r["why"] for r in rows(account(ev + strays, header), "10")]
    b = [r["why"] for r in rows(account(ev + strays[::-1], header), "10")]
    assert a == b and len(set(a)) == 1


def test_a_trace_row_says_why_it_cannot_be_ruled_out():
    r = one(account(steady(1000), CUPTI), "11")
    assert r["state"] == "not_fired"
    assert "carries no dropped-record count" in r["status_detail"]


# ── the command, end to end ──────────────────────────────────────────────────


def test_end_to_end_from_record_dicts(tmp_path):
    recs = []
    for i in range(1000):
        t = i * 2 * MS
        recs.append({"kind": "kernel", "name": "gemm", "start_ns": t, "end_ns": t + MS,
                     "stream_id": 7, "device_id": 0, "pid": 42})
        recs.append({"kind": "memcpy", "start_ns": t + MS, "end_ns": t + 1_100_000,
                     "stream_id": 7, "device_id": 0, "bytes": 64, "copy_kind": 1, "pid": 42})
        # Both stream synchronization types; the shim leaves their device at 0.
        recs.append({"kind": "sync", "start_ns": t + 1_200_000, "end_ns": t + 1_300_000,
                     "sync_type": 2 + i % 2, "pid": 42})
    events = [decode_record(r) for r in recs]
    rep = witness.run(write_run(tmp_path, events))
    assert one(rep, "10")["state"] == one(rep, "11")["state"] == "fired"
    # 999 gaps inside the kernel span; the records after the last kernel are outside it.
    assert one(rep, "10")["seconds"] == one(rep, "11")["seconds"] == pytest.approx(0.0999)
    assert one(rep, "10")["partition"] == {"pid": 42, "device_id": 0}
    assert not [r for r in rep.results if r["reason"] == "invalid_capture"]
    assert rep.inputs == {"found": ["trace.jsonl"], "missing": []}
    assert closure(rep) == 1


def test_command_writes_strict_json_that_loads(tmp_path, capsys):
    run_dir = write_run(tmp_path, steady(1000) + [Y(MS, 300 * MS)])
    assert witness.main([str(run_dir)]) == 0
    out = capsys.readouterr().out
    assert out.startswith("Capture precondition: the server was started under gitm's injection")
    assert "fired" in out.splitlines()[1] and "never run on a real capture" in out
    for name in ("gap time under a synchronization record", "unexplained idle"):
        assert name in out
    # The /metrics rows are in the list and not evaluated, so the default table leaves them out.
    assert "request queue time" not in out
    text = (run_dir / "witness.json").read_text()
    data = json.loads(text, parse_constant=lambda c: pytest.fail(f"non-strict JSON: {c}"))
    assert data["schema_version"] == witness.SCHEMA_VERSION
    assert data["thresholds_version"] == prim.THRESHOLDS_VERSION and data["generated_at"]
    assert {"window_start_ns", "window_end_ns", "window_s", "busy_s"} <= set(data["partitions"][0])
    assert {r["scope"] for r in data["results"]} == {"partition", "device", "run"}
    assert all({"pid", "device_id"} <= set(r["partition"]) for r in data["results"])
    assert [p.name for p in run_dir.iterdir() if p.name.endswith(".tmp")] == []
    assert witness.main([str(run_dir), "--json"]) == 0
    presence = json.loads(capsys.readouterr().out)["run_presence"]
    assert presence["10"] == "fired" and presence["3"] == "cannot_tell"


def test_data_problems_never_raise():
    assert set(witness.__all__) == {"Header", "Report", "WitnessInputError", "account", "run"}
    account([K(0, 5), K(3, 9), Y(2, 1)], Header(source="torch-import", device_count=0))


@pytest.mark.parametrize("row_id, existing_rule", [("2", "admission_backlog"),
                                                   ("3", "kv_cache_preemption"),
                                                   ("4", None), ("5", None)])
def test_the_metrics_rows_are_listed_and_not_evaluated(row_id, existing_rule):
    r = one(account(steady(1000), CUPTI), row_id)
    assert (r["state"], r["reason"]) == ("cannot_tell", "not_built")
    assert r["status"] == "not_identifiable_from_this_data"
    assert "/metrics counters are not read by this build" in r["why"]
    assert (existing_rule in r["why"]) if existing_rule else ("scheduler_attribution" not in r["why"])


# ── the attribution range: what a row's records alone explain, and what they could ──


WHOLE = Header(source="cupti", dropped_records=0)
NOT_ID = "not_identifiable_from_this_data"


def gap(i):
    """The i-th gap of ``steady()``: 1 ms after each 1 ms kernel."""
    return i * 2 * MS + MS, i * 2 * MS + 2 * MS


def test_without_overlap_the_range_is_the_point():
    rep = account(steady(1000) + [Y(*gap(i)) for i in range(300)]
                  + [M(*gap(i)) for i in range(300, 400)], WHOLE)
    for row_id, seconds in (("10", 0.3), ("11", 0.1)):
        r = one(rep, row_id)
        assert r["status"] == "estimated"
        assert r["interval"]["low_s"] == r["interval"]["high_s"] == r["seconds"] == seconds
    assert one(rep, "G1")["interval"] == {"low_s": 0.0, "high_s": 0.0, "low_share": 0.0,
                                          "high_share": 0.0, "low_fires": False,
                                          "high_fires": False}


def test_time_under_both_a_sync_and_a_memcpy_record_widens_both_rows():
    # 300 gaps under a sync record, 150 of them under a memcpy record too.
    rep = account(steady(1000) + [Y(*gap(i)) for i in range(300)]
                  + [M(*gap(i)) for i in range(150)], WHOLE)
    r10, r11 = one(rep, "10"), one(rep, "11")
    # Row 10: 0.15 s is under a sync record alone, 0.30 s under one at all.
    assert (r10["seconds"], r10["interval"]["low_s"], r10["interval"]["high_s"]) == (0.15, 0.15, 0.3)
    assert (r10["state"], r10["status"]) == ("fired", "estimated")
    # Row 11: every second of it is also under a sync record, so none is its alone.
    assert (r11["seconds"], r11["interval"]["low_s"], r11["interval"]["high_s"]) == (0.15, 0.0, 0.15)
    assert (r11["state"], r11["status"], r11["status_reason"]) == ("fired", NOT_ID, "confounded")
    assert r11["shares_with"] == ["10"] and r10["shares_with"] == ["11"]
    assert "row(s) 10 can" in r11["status_detail"] and "runtime API" in r11["next"]
    assert "range 0.1500 to 0.3000 s" in witness._render(rep)


def test_a_row_is_not_absent_while_its_records_cover_more_than_the_threshold():
    # 2% of the window is under both records. The split gives it all to row 11, and
    # row 10's own share is zero; its records still cover 2%.
    rep = account(steady(1000) + [r for i in range(40) for r in (Y(*gap(i)), M(*gap(i)))], WHOLE)
    r10 = one(rep, "10")
    assert (r10["state"], r10["seconds"], r10["interval"]["high_s"]) == ("not_fired", 0.0, 0.04)
    assert (r10["status"], r10["status_reason"]) == (NOT_ID, "confounded")
    # Under the threshold on both ends is still a rule-out.
    few = account(steady(1000) + [r for i in range(4) for r in (Y(*gap(i)), M(*gap(i)))], WHOLE)
    assert one(few, "10")["status"] == "absent"


def test_a_record_over_a_launch_scale_gap_is_claimed_by_both_rows():
    # 20 us gaps, 1500 of 2000 under a sync record: launch overhead or a sync?
    n, busy, short = 2000, MS, 20_000
    events = steady(n, period=busy + short, busy=busy)
    events += [Y(i * (busy + short) + busy, (i + 1) * (busy + short)) for i in range(1500)]
    rep = account(events, WHOLE)
    g1, r10 = one(rep, "G1"), one(rep, "10")
    # 1999 gaps of 20 us: 499 with no record over them, all of them launch-scale.
    assert (g1["interval"]["low_s"], g1["interval"]["high_s"]) == pytest.approx(
        (499 * 20e-6, 1999 * 20e-6))
    assert (g1["state"], g1["status"], g1["status_reason"]) == ("not_fired", NOT_ID, "confounded")
    assert (r10["interval"]["low_s"], r10["interval"]["high_s"]) == pytest.approx((0.0, 0.03))
    assert (r10["state"], r10["status"], r10["status_reason"]) == ("fired", NOT_ID, "confounded")
    assert (g1["shares_with"], r10["shares_with"]) == (["10"], ["G1"])


def test_the_point_always_sits_inside_the_range():
    rng = random.Random(7)
    events = steady(1000)
    for i in range(1000):
        a, b = gap(i)
        if rng.random() < 0.3:
            events.append(Y(a + rng.randrange(0, 400_000), b - rng.randrange(0, 400_000)))
        if rng.random() < 0.3:
            events.append(M(a + rng.randrange(0, 400_000), b - rng.randrange(0, 400_000)))
    rep = account(events, WHOLE)
    for row_id in ("10", "11", "G1", "22", "25"):
        r = one(rep, row_id)
        assert r["interval"]["low_s"] <= r["seconds"] <= r["interval"]["high_s"], row_id
