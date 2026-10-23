"""The cause registry: its shape, its registered constants, and its lever names."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from gitm.kernels.library import load_library
from gitm.optimizer import primitives as prim
from gitm.optimizer.metrics import _LAUNCH_LATENCY_NS


def test_registry_is_25_causes_and_4_groups():
    groups = [p for p in prim.REGISTRY.values() if p.is_group]
    causes = [p for p in prim.REGISTRY.values() if not p.is_group]
    assert [g.id for g in groups] == ["G1", "G2", "G3", "G4"]
    assert sorted(int(c.id) for c in causes) == list(range(1, 26))


def test_registry_is_read_only_and_get_takes_row_numbers():
    with pytest.raises(TypeError):
        prim.REGISTRY["26"] = prim.get("1")  # type: ignore[index]
    assert prim.get(10) is prim.get("10")
    with pytest.raises(KeyError):
        prim.get("G9")


def test_constants_read_back():
    # Registered values. A change here is a re-registration: bump THRESHOLDS_VERSION.
    assert prim.THRESHOLDS_VERSION == "2026-10-12.2"
    assert prim.NOT_BUILT == ("G2", "G3", "G4")
    assert (prim.SLICE_NS, prim.MAX_SLICES) == (1_000_000_000, 100_000)
    assert prim.STATUSES == ("estimated", "not_identifiable_from_this_data", "absent")
    assert prim.MIN_WINDOW_NS == 1_000_000_000
    # The 20 us in G1's witness text is the cut the sweep applies.
    assert _LAUNCH_LATENCY_NS == 20_000 and "20 us" in prim.get("G1").witness
    assert {i: prim.get(i).fire_at for i in ("10", "11", "22", "25", "G1")} == dict.fromkeys(
        ("10", "11", "22", "25", "G1"), 0.01)
    # The /metrics rows are listed and not evaluated, so they register no threshold.
    assert {prim.get(i).fire_at for i in ("2", "3", "4", "5")} == {None}
    assert {prim.get(i).fixed_reason for i in ("2", "3", "4", "5")} == {"not_built"}


def test_reported_names():
    assert prim.get("2").name == "request queue time"
    assert prim.get("4").name == "prefix-cache miss fraction"
    assert prim.get("5").name == "speculative-draft rejection fraction"
    assert prim.get("22").name == "exposed collective communication"
    assert prim.get("25").levers == ("enable_dbo",)
    for bit in ("SW_POWER_CAP", "HW_SLOWDOWN", "SW_THERMAL_SLOWDOWN", "HW_THERMAL_SLOWDOWN",
                "HW_POWER_BRAKE_SLOWDOWN"):
        assert bit in prim.get("20").witness


def test_groups_and_members_agree():
    for p in prim.REGISTRY.values():
        for m in p.members:
            assert prim.get(m).group == p.id
            assert prim.get(m).scope == p.scope
        if p.group:
            assert p.id in prim.get(p.group).members
        assert p.form in prim.FORMS
        assert p.scope in ("partition", "device", "run")
        assert p.fixed_reason in (None, "channel_missing", "no_passive_witness", "not_built")
        assert p.remedy


def test_every_lever_is_a_library_entry():
    library = {s.name for s in load_library()}
    for p in prim.REGISTRY.values():
        for lever in p.levers:
            assert lever in library, f"row {p.id} names lever {lever!r}, not in library.yaml"


def test_docs_table_rows_and_levers_match_the_registry():
    text = (Path(__file__).parent.parent / "docs" / "primitives-table.md").read_text()
    table = re.findall(r"^\| (\d+) \| [^|]+ \| [^|]+ \| ([^|]+) \| ([^|]+) \|$", text, flags=re.M)
    assert {row_id for row_id, _, _ in table} == {
        p.id for p in prim.REGISTRY.values() if not p.is_group}
    for row_id, use, passive in table:
        p = prim.get(row_id)
        assert tuple(re.findall(r"`([a-z_0-9]+)`", use)) == p.levers, row_id
        # A row this build cannot evaluate must not read as available without saying so.
        if p.fixed_reason and passive.strip().startswith("Yes"):
            assert "this build" in passive or "yet" in passive, row_id
