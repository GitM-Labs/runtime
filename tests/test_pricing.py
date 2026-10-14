"""The brief's statuses and where a price sits, on synthetic captures with a known injected amount."""

from __future__ import annotations

import json

import numpy as np
import pytest

from gitm.optimizer import pricing, witness, witness_fixtures
from gitm.optimizer import primitives as prim
from gitm.optimizer.witness import Header, account
from gitm.optimizer.witness_fixtures import Injection, generate
from gitm.tracer.schema import KernelEvent, SyncEvent

MS = 1_000_000
S = 1_000_000_000
CUPTI = Header(source="cupti")
NOT_ID = "not_identifiable_from_this_data"
ALL_THREE = dict(sync=Injection(0.3, 100_000), memcpy=Injection(0.2, 80_000),
                 all_reduce=Injection(0.1, 150_000))
# About 2.5%, 2% and 2.25% of the window: rows 10, 11 and 22 all fire.
FIRING = dict(sync=Injection(0.5, 100_000), memcpy=Injection(0.4, 100_000),
              all_reduce=Injection(0.3, 150_000))


def one(report, row_id):
    (row,) = [r for r in report.results if r["id"] == row_id]
    return row


# ── the price and where it sits ──────────────────────────────────────────────


def test_the_point_price_is_the_injected_total():
    for seed in range(5):
        f = generate(seed, **ALL_THREE)
        rep = account(f.events, CUPTI)
        for row_id in ("10", "11", "22"):
            r = one(rep, row_id)
            assert r["seconds"] == pytest.approx(f.realised_ns[row_id] / 1e9, rel=1e-9)
            # Injected records never overlap, so the range is the point.
            assert r["interval"]["low_s"] == r["interval"]["high_s"] == r["seconds"]


def test_slice_facts_say_where_the_time_sat():
    r = one(account(generate(0, **ALL_THREE).events, CUPTI), "10")
    f = r["slices"]
    assert (f["slice_s"], f["n"], f["with_time"]) == (1.0, 25, 25)
    assert 0 < f["max_slice_s"] < 0.1
    # The slices' own total leaves out the final partial slice, so it is at most the price.
    assert f["slice_sum_s"] <= r["seconds"] and f["slice_sum_s"] == pytest.approx(r["seconds"],
                                                                                rel=0.05)
    # Even: the busiest 3 of 25 slices hold about 3/25 of the time.
    assert f["busiest_tenth_share"] == pytest.approx(3 / 25, rel=0.25)
    assert r["slices_reason"] is None


def test_slice_facts_are_the_same_in_any_event_order():
    f = generate(3, **ALL_THREE)
    first = one(account(f.events, CUPTI), "10")["slices"]
    assert one(account(list(reversed(f.events)), CUPTI), "10")["slices"] == first


def test_slice_facts_direct():
    facts, reason = pricing._slice_facts(np.array([0.0, 4e8, 0.0, 6e8, 0, 0, 0, 0, 0, 0, 0]), 11 * S)
    assert reason is None
    assert facts == {"slice_s": 1.0, "n": 11, "with_time": 2, "max_slice_s": 0.6,
                     "slice_sum_s": 1.0, "busiest_tenth_share": 1.0}  # top 2 of 11 hold it all
    assert pricing._slice_facts(np.zeros(5), 5 * S)[0]["busiest_tenth_share"] is None
    assert pricing._slice_facts(np.zeros(0), 0) == (None, "window shorter than one slice")
    assert "window too long" in pricing._slice_facts(None, 200_000 * S)[1]


def test_slices_by_start_drops_the_final_partial_slice_and_keeps_gaps_whole():
    # 2.5 s window: a 1.2 s amount starting in slice 0 stays whole there.
    got = pricing._slices_by_start([0, int(1.9 * S), int(2.2 * S)], [1_200 * MS, 5, 7], 0,
                                   int(2.5 * S))
    assert got.tolist() == [1_200 * MS, 5]


def test_slices_by_overlap_cuts_an_interval_at_slice_boundaries():
    got = pricing._slices_by_overlap([(int(0.5 * S), int(2.25 * S)), (int(2.9 * S), int(3.4 * S))],
                                     0, int(3.5 * S))
    assert got.tolist() == [0.5 * S, 1.0 * S, 0.25 * S + 0.1 * S]


# ── the status table ─────────────────────────────────────────────────────────


def test_a_fired_priced_row_is_estimated_for_this_capture():
    # About 2.5%, 2% and 2.25% of the window.
    rep = account(generate(0, sync=Injection(0.5, 100_000), memcpy=Injection(0.4, 100_000),
                           all_reduce=Injection(0.3, 150_000)).events, CUPTI)
    for row_id in ("10", "11"):
        r = one(rep, row_id)
        assert (r["state"], r["status"], r["status_reason"]) == ("fired", "estimated", None)
        assert "this capture" in r["status_detail"]
        # The one assumption an estimated row may carry.
        assert r["assumptions"] == ["no_dropped_events"]


def test_a_collective_row_is_priced_and_never_estimated_on_a_name_match():
    # Rows 22 and 25 call a kernel a collective by a substring of its name.
    whole = Header(source="cupti", dropped_records=0)
    rep = account(generate(0, **FIRING).events, whole)
    r = one(rep, "22")
    assert (r["state"], r["status"], r["status_reason"]) == ("fired", NOT_ID, "structural")
    assert r["assumptions"] == ["collective_classified_by_name_only"]
    assert r["seconds"] == pytest.approx(generate(0, **FIRING).realised_ns["22"] / 1e9)
    assert "classifier" in r["next"]
    # A compute kernel whose name happens to match is the same row, on one device.
    stray = [KernelEvent(start_ns=i * 2 * MS + MS, end_ns=i * 2 * MS + MS + 200_000,
                         name="sgemm_custom_argmax", stream_id=3, device_id=0, pid=4242)
             for i in range(2000)]
    assert one(account(generate(0).events + stray, whole), "22")["status"] == NOT_ID
    # No match at all on one stated device: nothing to miss, so it can be ruled out.
    assert one(account(generate(0).events, whole), "22")["status"] == "absent"


def test_a_short_window_is_estimated_all_the_same():
    # 2 s: the time under the witness is arithmetic on the records, however short the window.
    f = generate(0, window_s=2.0, sync=Injection(0.5, 100_000))
    r = one(account(f.events, CUPTI), "10")
    assert (r["state"], r["status"]) == ("fired", "estimated")
    assert r["seconds"] == pytest.approx(f.realised_ns["10"] / 1e9) and r["slices"]["n"] <= 2


def kernels(seconds: int) -> list:
    return [KernelEvent(start_ns=i * 2 * MS, end_ns=i * 2 * MS + MS, stream_id=0, device_id=0,
                        name="gemm") for i in range(seconds * 500)]


def sync(start_ns: int, length_ns: int) -> SyncEvent:
    return SyncEvent(start_ns=start_ns, end_ns=start_ns + length_ns, stream_id=0, device_id=0,
                     sync_kind="stream")


def test_time_concentrated_in_two_slices_is_estimated_and_the_facts_show_it():
    # 120 s of kernels; the only sync time is two 0.7 s stalls at 6 s and 8 s.
    at = (6 * S, 8 * S)
    ev = [k for k in kernels(120) if not any(a <= k.start_ns < a + 700 * MS for a in at)]
    r = one(account(ev + [sync(a, 700 * MS) for a in at], CUPTI), "10")
    assert (r["state"], r["status"]) == ("fired", "estimated")
    assert r["seconds"] == pytest.approx(1.4, rel=0.01)
    f = r["slices"]
    assert (f["with_time"], f["n"]) == (2, 119)
    assert f["max_slice_s"] == pytest.approx(0.7, rel=0.01) and f["busiest_tenth_share"] == 1.0


def test_a_gap_longer_than_a_slice_shows_as_a_slice_over_its_length():
    ev = [k for k in kernels(60) if not 5 * S <= k.start_ns < 8 * S] + [sync(5 * S, 3 * S)]
    r = one(account(ev, CUPTI), "10")
    assert r["status"] == "estimated" and r["slices"]["max_slice_s"] == pytest.approx(3.0, rel=0.01)
    assert r["slices"]["with_time"] == 1


def test_not_fired_rows_are_not_absent_without_a_drop_count():
    rep = account(generate(0).events, CUPTI)
    for row_id in ("10", "11", "G1", "22", "25"):
        r = one(rep, row_id)
        assert (r["state"], r["status"], r["status_reason"]) == ("not_fired", NOT_ID,
                                                                 "structural")
        # A row that is not identifiable says what would settle it.
        assert "dropped_records" in r["next"]
    assert not [r for r in rep.results if r.get("status") == "absent"]


def test_zero_dropped_records_makes_a_not_fired_row_absent():
    rep = account(generate(0).events, Header(source="cupti", dropped_records=0))
    for row_id in ("10", "11", "G1", "22", "25"):
        r = one(rep, row_id)
        assert (r["state"], r["status"], r["completeness"]) == ("not_fired", "absent",
                                                                "complete")
        assert r["fire_at"] == prim.FIRE_AT_WINDOW_SHARE and r["assumptions"] == []
        assert "this capture" in r["status_detail"]


def test_zero_dropped_records_discharges_the_assumption_on_an_estimated_row():
    rep = account(generate(0, **FIRING).events, Header(source="cupti", dropped_records=0))
    for row_id in ("10", "11"):
        r = one(rep, row_id)
        assert (r["status"], r["assumptions"]) == ("estimated", [])


def test_dropped_records_leave_every_trace_row_not_identifiable():
    lossy = Header(source="cupti", dropped_records=3)
    fired = account(generate(0, **FIRING).events, lossy)
    quiet = account(generate(0).events, lossy)
    for rep, state in ((fired, "fired"), (quiet, "not_fired")):
        for row_id in ("10", "11"):
            r = one(rep, row_id)
            assert (r["state"], r["status"], r["status_reason"]) == (state, NOT_ID,
                                                                     "structural")
            assert "3 record(s) were dropped" in r["status_detail"]
            assert "GITM_TRACE_FLUSH_MS" in r["next"]
            # The time is still shown; it is the status that is withheld.
            assert r["seconds"] is not None
    assert "no_dropped_events" not in one(fired, "10")["assumptions"]


def test_a_blocker_other_than_drops_keeps_a_row_from_absent():
    whole = Header(source="cupti", dropped_records=0)
    # Another traced pid on the device: the split of the gap time is unverified.
    shared = generate(0).events + generate(1, pid=7).events
    for r in account(shared, whole).results:
        assert not (r["id"] in ("10", "11", "G1") and r.get("status") == "absent"), r["id"]
    # Several devices: a collective the classifier does not know would read as compute.
    two = account(generate(0).events + generate(1, device_id=1).events,
                  Header(source="cupti", device_count=2, dropped_records=0))
    assert {r["status"] for r in two.results if r["id"] in ("22", "25")} == {NOT_ID}
    # Row 24 does not check misroute.
    quiet = account(generate(0).events, whole)
    assert (one(quiet, "24")["state"], one(quiet, "24")["status"]) == ("not_fired", NOT_ID)
    assert "misrouted" in one(quiet, "24")["status_detail"]


@pytest.mark.parametrize("device_count, missing", [(0, ()), (-3, ()), (1, ("device_count",))])
def test_an_unknown_device_count_keeps_rows_22_and_25_from_absent(device_count, missing):
    # capture records 0 devices when its probe fails: that is not "one device".
    rep = account(generate(0).events, Header(source="cupti", dropped_records=0,
                                             device_count=device_count, missing=missing))
    assert {one(rep, i)["status"] for i in ("22", "25")} == {NOT_ID}


@pytest.mark.parametrize("source", ["nsys-import", "torch-import"])
def test_an_imported_trace_is_never_absent_even_with_a_stated_count(source):
    rep = account(generate(0).events, Header(source=source, dropped_records=0))
    assert not [r["id"] for r in rep.results if r.get("status") == "absent"]
    assert one(rep, "24")["state"] == "cannot_tell"


def test_rocprof_with_no_memcpy_record_cannot_rule_out_row_11():
    # A collector whose memcpy service never started reports zero drops truthfully.
    rep = account(generate(0).events, Header(source="rocprof", dropped_records=0))
    r = one(rep, "11")
    assert (r["state"], r["reason"]) == ("cannot_tell", "channel_missing")


@pytest.mark.parametrize("stated", [-2, 1.0, 0.0, True, False, "0"])
def test_an_in_memory_header_value_that_is_not_a_count_is_no_count(stated):
    rep = account(generate(0, window_s=2.0).events, Header(dropped_records=stated))
    assert one(rep, "24")["state"] == "cannot_tell"
    assert not [r for r in rep.results if r.get("status") == "absent"]


def test_the_next_step_answers_the_reason_the_row_is_not_identifiable():
    whole = Header(source="cupti", dropped_records=0)
    shared = account(generate(0).events + generate(1, pid=7).events, whole)
    assert {r["next"] for r in shared.results if r["id"] in ("10", "11", "G1")} == {
        "trace one process per device"}
    # The same rows with no count: sharing the device is still the stated reason.
    uncounted = account(generate(0).events + generate(1, pid=7).events, CUPTI)
    assert "dropped_records" not in next(r for r in uncounted.results if r["id"] == "G1")["next"]


def test_row_25_is_not_absent_on_two_devices_when_only_all_reduce_matched():
    whole = Header(source="cupti", device_count=2, dropped_records=0)
    ev = (generate(0, all_reduce=Injection(0.01, 150_000)).events
          + generate(1, device_id=1, all_reduce=Injection(0.01, 150_000)).events)
    rows = [r for r in account(ev, whole).results if r["id"] == "25"]
    assert len(rows) == 2
    for r in rows:
        assert (r["state"], r["status"]) == ("not_fired", NOT_ID)
        assert r["assumptions"] == ["collective_classified_by_name_only"]


def test_a_complete_rocprof_capture_can_rule_out():
    rep = account(generate(0, memcpy=Injection(0.01, 80_000)).events,
                  Header(source="rocprof", dropped_records=0))
    assert one(rep, "11")["status"] == one(rep, "22")["status"] == "absent"
    assert one(rep, "24")["state"] == "not_fired" and one(rep, "10")["reason"] == "channel_missing"
    assert one(rep, "G1")["status"] == NOT_ID


def test_next_on_fired_rows_with_no_price():
    ev = generate(0, window_s=2.0).events
    r = one(account(ev, Header(dropped_records=5)), "24")
    assert r["next"] == witness._LOSSY_NEXT and "not known" in r["price_reason"]


def test_row_24_reads_the_drop_count():
    events = generate(0, window_s=2.0).events
    none, lossy = account(events, CUPTI), account(events, Header(dropped_records=5))
    assert (one(none, "24")["state"], one(none, "24")["reason"]) == ("cannot_tell",
                                                                    "channel_missing")
    r = one(lossy, "24")
    assert (r["state"], r["status"], r["status_reason"]) == ("fired", NOT_ID, "unpriced")
    assert r["observation"]["value"] == 5
    # An unusable capture still says what its header reports.
    assert one(account([], Header(dropped_records=5)), "24")["state"] == "fired"


@pytest.mark.parametrize("stated", [None, -2, 1.0, True, "0"])
def test_a_header_value_that_is_not_a_count_is_no_count(tmp_path, stated):
    witness_fixtures.write_run_dir(tmp_path, generate(0, window_s=2.0))
    path = tmp_path / "trace.jsonl"
    head, rest = path.read_text().split("\n", 1)
    header = json.loads(head)
    header["_header"]["dropped_records"] = stated
    path.write_text(json.dumps(header) + "\n" + rest)
    rep = witness.run(tmp_path)
    assert one(rep, "24")["state"] == "cannot_tell"
    assert not [r for r in rep.results if r.get("status") == "absent"]


def test_nothing_fires_on_a_clean_capture_at_any_kernel_length():
    # No injected cause, kernels from 0.25 ms to 4 ms: a load change must not look like one.
    for kernel_ns in (250_000, MS, 4 * MS):
        rep = account(generate(1, kernel_ns=kernel_ns).events, CUPTI)
        for row_id in ("10", "11", "G1", "22", "25"):
            assert one(rep, row_id)["state"] == "not_fired", (kernel_ns, row_id)
        assert not [r for r in rep.results if r.get("status") == "estimated"]


# ── output ───────────────────────────────────────────────────────────────────


def test_demo_capture_end_to_end(tmp_path, capsys):
    assert witness_fixtures.main(["demo", str(tmp_path)]) == 0
    capsys.readouterr()
    assert witness.main([str(tmp_path)]) == 0
    out = capsys.readouterr().out.rstrip()
    # The demo header reports zero dropped records, so all three statuses appear.
    assert "2 fired, 4 not fired, 3 cannot tell" in out
    assert "status: 2 estimated, 4 not_identifiable_from_this_data, 3 absent" in out
    assert "By form" in out and "(also observation distortion)" in out
    assert "where: in 120 of 120 one-second slices" in out

    data = json.loads((tmp_path / "witness.json").read_text())
    got = {r["id"]: (r["state"], r["status"]) for r in data["results"]
           if r["id"] in ("3", "10", "11", "24", "G1")}
    assert got == {"10": ("fired", "estimated"), "11": ("fired", "estimated"),
                   "G1": ("not_fired", "absent"),
                   "24": ("not_fired", "not_identifiable_from_this_data"),
                   "3": ("cannot_tell", "not_identifiable_from_this_data")}
    row10 = next(r for r in data["results"] if r["id"] == "10")
    assert row10["slices"]["n"] == 120


# ── long windows ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("days", [2, 365, 50_000])
def test_a_window_too_long_to_slice_gets_no_slices(days):
    far = days * 86_400 * S
    ev = kernels(2) + [KernelEvent(start_ns=far + k.start_ns, end_ns=far + k.end_ns, stream_id=0,
                                   device_id=0, name="gemm") for k in kernels(2)]
    rep = account(ev, CUPTI)
    for row_id in ("10", "11", "G1", "22"):
        r = one(rep, row_id)
        assert r["slices"] is None and "window too long" in r["slices_reason"]
    assert one(rep, "remainder")["share"] > 0.99


def test_slices_stop_exactly_past_the_maximum():
    # No allocation past the maximum: the answer is None, not a 50,000-day array.
    at = prim.MAX_SLICES * prim.SLICE_NS
    assert len(pricing._slices_by_start([0], [5], 0, at)) == prim.MAX_SLICES
    assert pricing._slices_by_start([0], [5], 0, at + prim.SLICE_NS) is None
    assert pricing._slices_by_overlap([(0, 5)], 0, at + prim.SLICE_NS) is None


def test_in_memory_timestamps_that_cannot_be_nanoseconds_are_an_invalid_capture():
    ev = kernels(2) + [KernelEvent(start_ns=2**70, end_ns=2**70 + MS, stream_id=0, device_id=0,
                                   name="gemm")]
    r = one(account(ev, CUPTI), "10")
    assert r["reason"] == "invalid_capture" and "cannot be nanoseconds" in r["why"]


# ── paths the generator's independent slices would not catch ─────────────────


def test_stall_ns_per_gap_places_each_amount_in_its_own_gap():
    from types import SimpleNamespace

    from gitm.optimizer.metrics import _stall_ns_per_gap as raw

    def _stall_ns_per_gap(*args):
        return [tuple(row) for row in raw(*args).tolist()]

    gaps = [(30_000, 100_000), (0, 10_000)]  # unsorted on purpose
    sync_rec = [SimpleNamespace(start_ns=5_000, end_ns=40_000)]
    copy_rec = [SimpleNamespace(start_ns=8_000, end_ns=32_000)]
    # (start, transfer, sync, launch, idle): the 10 us gap is launch-scale, the 70 us one is not.
    assert _stall_ns_per_gap(gaps, copy_rec, sync_rec) == [
        (0, 2_000, 3_000, 5_000, 0), (30_000, 2_000, 8_000, 0, 60_000)]
    assert _stall_ns_per_gap(gaps, [], []) == [(0, 0, 0, 10_000, 0), (30_000, 0, 0, 0, 70_000)]


def test_each_gap_row_slices_its_own_column():
    # Sync only in the first half, memcpy only in the second, launch-scale gaps throughout.
    f = generate(5, gap_ns=18_000, kernel_ns=MS)
    half = f.window_ns // 2
    ev = list(f.events)
    gaps = [(a.end_ns, b.start_ns) for a, b in zip(ev, ev[1:], strict=False)]
    ev += [sync(g0, 6_000) for g0, _ in gaps if g0 < half]
    ev += [witness_fixtures.MemcpyEvent(start_ns=g0, end_ns=g0 + 6_000, stream_id=0, device_id=0,
                                        bytes=1, src="host", dst="device", pid=4242)
           for g0, _ in gaps if g0 >= half]
    ev = [e if e.kind != "sync" else e.model_copy(update={"pid": 4242}) for e in ev]
    rep = account(ev, CUPTI)
    n = one(rep, "10")["slices"]["n"]
    for row_id in ("10", "11", "G1"):
        r = one(rep, row_id)
        assert r["slices"]["slice_sum_s"] == pytest.approx(r["seconds"], rel=0.05), row_id
    # Sync only in the first half, memcpy only in the second: each row's slices are its own.
    assert one(rep, "10")["slices"]["with_time"] <= n // 2 + 1
    assert one(rep, "11")["slices"]["with_time"] <= n // 2 + 1
    assert one(rep, "G1")["slices"]["with_time"] == n and one(rep, "G1")["state"] == "fired"


def test_row_11_keeps_its_slices_when_sync_records_cannot_be_placed():
    a = generate(1, memcpy=Injection(0.5, 100_000), pid=7, device_id=0)
    b = generate(2, memcpy=Injection(0.5, 100_000), pid=7, device_id=1)
    stray = SyncEvent(start_ns=S, end_ns=S + MS, stream_id=0, device_id=0, sync_kind="stream",
                      pid=7)
    rep = account(a.events + b.events + [stray], CUPTI)
    for fixture, device in ((a, 0), (b, 1)):
        r = next(r for r in rep.results if r["id"] == "11" and r["partition"]["device_id"] == device)
        assert r["seconds"] == pytest.approx(fixture.realised_ns["11"] / 1e9)
        assert r["slices"]["slice_sum_s"] == pytest.approx(r["seconds"], rel=0.05)
        assert r["status"] == "estimated"
    assert {r["reason"] for r in rep.results if r["id"] in ("10", "G1")} == {"invalid_capture"}


def test_a_shared_device_row_is_not_estimated():
    # Another traced process's kernel time may be sitting in this partition's gaps.
    a = generate(1, window_s=120.0, sync=Injection(0.5, 100_000), pid=1)
    b = generate(2, window_s=120.0, sync=Injection(0.5, 100_000), pid=2)
    shifted = [e.model_copy(update={"start_ns": e.start_ns + 300_000,
                                    "end_ns": e.end_ns + 300_000, "stream_id": 5})
               for e in b.events]
    mine = [r for r in account(a.events + shifted, CUPTI).results if r["id"] == "10"]
    assert len(mine) == 2
    for r in mine:
        assert r["state"] == "fired" and r["assumptions"] == ["other_pid_active_on_device"]
        assert (r["status"], r["status_reason"]) == (NOT_ID, "structural")
        assert "other_pid_active_on_device" in r["status_detail"]


def test_float_timestamps_in_memory_are_an_invalid_capture():
    ev = kernels(2) + [KernelEvent.model_construct(kind="kernel", start_ns=5e9, end_ns=6e9,
                                                   stream_id=0, device_id=0, name="gemm",
                                                   pid=None)]
    r = one(account(ev, CUPTI), "10")
    assert r["reason"] == "invalid_capture" and "cannot be nanoseconds" in r["why"]


def test_the_demo_refuses_to_overwrite_a_capture(tmp_path, capsys):
    (tmp_path / "trace.jsonl").write_text("REAL CAPTURE")
    assert witness_fixtures.main(["demo", str(tmp_path)]) == 2
    assert "nothing written" in capsys.readouterr().err
    assert (tmp_path / "trace.jsonl").read_text() == "REAL CAPTURE"


# ── release gate: load alone must not look like a cause ──────────────────────


@pytest.mark.parametrize("batch", [1, 8, 32])
def test_no_check_fires_on_a_clean_capture_at_any_batch(batch):
    f = witness_fixtures.varying_batch(batch)
    assert f.window_ns >= 25 * S
    rep = account(f.events, CUPTI)
    for row_id in ("10", "11", "G1", "22", "25"):
        assert one(rep, row_id)["state"] == "not_fired", row_id
    assert one(rep, "G1")["share"] <= 0.005  # short gaps are there, under half a percent
    assert (one(rep, "G2")["reason"], one(rep, "G4")["reason"]) == ("not_built", "not_built")
    for row_id in ("2", "3", "4", "5"):
        assert one(rep, row_id)["reason"] == "not_built"
    assert not [r["id"] for r in rep.results if r.get("status") in ("estimated", "absent")]
    assert one(rep, "G3")["reason"] == "not_built"


def test_an_estimated_row_carries_only_the_one_stated_assumption():
    rep = account(generate(1, window_s=120.0, sync=Injection(0.5, 100_000)).events, CUPTI)
    for r in rep.results:
        if r.get("status") == "estimated":
            assert r["assumptions"] == ["no_dropped_events"], r["id"]


def test_all_flag_prints_members_fixed_rows_and_baselines(tmp_path, capsys):
    witness_fixtures.write_run_dir(tmp_path, generate(0, window_s=2.0))
    assert witness.main([str(tmp_path)]) == 0
    short = capsys.readouterr().out
    assert "Group members" not in short and "Baselines" not in short
    assert witness.main([str(tmp_path), "--all"]) == 0
    full = capsys.readouterr().out
    assert full.startswith(short.rstrip())
    for needle in ("Group members", "framework-bound launch: reported as part of G1",
                   "Not evaluated in this build", "thermal throttling, power capping: "
                   "channel_missing", "request queue time: not_built",
                   "Baselines, not checked", "tile / wave quantization"):
        assert needle in full, needle


def test_every_cannot_tell_row_carries_its_reason_as_its_status_reason():
    rep = account(generate(0, window_s=2.0).events, CUPTI)
    for r in rep.results:
        if r["state"] == "cannot_tell":
            assert (r["status"], r["status_reason"]) == (NOT_ID, r["reason"]), r["id"]
        if r["state"] is not None:
            assert r["status"] in prim.STATUSES
    assert "status" not in one(rep, "remainder")
    # This header has no drop count, so nothing is complete and nothing is absent.
    assert not [r["id"] for r in rep.results if r.get("status") == "absent"]


def test_loss_of_unknown_size_is_lossy_not_an_assumption():
    # The collector said records were lost and could not count them: that is known loss.
    some = Header(source="cupti", dropped_records=-1)
    rep = account(generate(0, **FIRING).events, some)
    for row_id in ("10", "11"):
        r = one(rep, row_id)
        assert (r["state"], r["status"], r["completeness"]) == ("fired", NOT_ID, "lossy")
        assert "no_dropped_events" not in r["assumptions"] and r["seconds"] is not None
    r24 = one(rep, "24")
    assert (r24["state"], r24["status_reason"]) == ("fired", "unpriced")
    assert "count unknown" in witness._render(rep)
    assert not [r for r in account(generate(0).events, some).results
                if r.get("status") in ("absent", "estimated")]
