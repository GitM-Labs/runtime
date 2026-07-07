"""Tests for the dataset + reproducibility tooling.

Covers the HFT synthetic generator, the biotech/edge fetch smoke fixtures, the
shared smoke harness, and the reproducibility test. All GPU-free; the HFT tests
need pyarrow (the ``bench`` extra) and skip cleanly if it is absent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pyarrow", reason="pyarrow (bench extra) not installed")
import pyarrow.parquet as pq  # noqa: E402,I001


# --- HFT generator ----------------------------------------------------------


def _gen(out: Path, events=20_000, seed=42, per_file=8_000):
    from benchmarks.hft.generate import GenConfig, generate

    return generate(GenConfig(events=events, seed=seed, events_per_file=per_file), out)


def test_hft_generator_schema_and_sharding(tmp_path: Path):
    shards = _gen(tmp_path / "ds")
    assert len(shards) == 3  # 20k / 8k -> 8k, 8k, 4k
    tbl = pq.read_table(shards[0])
    assert tbl.column_names == ["ts_ns", "symbol_id", "side", "price", "size", "type"]
    assert str(tbl.schema.field("ts_ns").type) == "int64"
    assert str(tbl.schema.field("side").type) == "int8"


def test_hft_generator_is_deterministic(tmp_path: Path):
    import hashlib

    def digest(d: Path) -> list[str]:
        return [hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(d.glob("part-*.parquet"))]

    _gen(tmp_path / "a")
    _gen(tmp_path / "b")
    assert digest(tmp_path / "a") == digest(tmp_path / "b")


def test_hft_generator_timestamps_monotonic(tmp_path: Path):
    import numpy as np

    shards = _gen(tmp_path / "ds")
    last = -1
    for s in shards:  # global monotonicity across shards
        ts = pq.read_table(s)["ts_ns"].to_numpy()
        assert bool(np.all(np.diff(ts) > 0))
        assert ts[0] > last
        last = int(ts[-1])


def test_hft_first_row_sample(tmp_path: Path):
    from benchmarks.hft.generate import first_row_sample

    _gen(tmp_path / "ds")
    row = first_row_sample(tmp_path / "ds")
    assert set(row) == {"ts_ns", "symbol_id", "side", "price", "size", "type"}
    assert row["side"] in (0, 1)
    assert row["type"] in (0, 1, 2)


# --- biotech fetch ----------------------------------------------------------


def test_biotech_synth_proteins_bounded_and_deterministic():
    from benchmarks.biotech.fetch import LEN_MAX, LEN_MIN, synth_proteins

    a = synth_proteins(10, seed=7)
    b = synth_proteins(10, seed=7)
    assert [r.seq for r in a] == [r.seq for r in b]
    for r in a:
        assert LEN_MIN <= len(r.seq) <= LEN_MAX


def test_biotech_fasta_roundtrip_and_histogram(tmp_path: Path):
    from benchmarks.biotech.fetch import length_histogram, read_fasta, synth_proteins, write_fasta

    recs = synth_proteins(15, seed=1)
    path = write_fasta(recs, tmp_path / "p.fasta")
    back = read_fasta(path)
    assert [r.seq for r in back] == [r.seq for r in recs]
    hist = length_histogram(recs)
    assert sum(hist.values()) == 15


def test_biotech_write_smoke_produces_fasta_and_msas(tmp_path: Path):
    from benchmarks.biotech.fetch import write_smoke

    info = write_smoke(tmp_path / "bio", n=8)
    assert info["n_proteins"] == 8
    assert info["n_msas"] == 8
    assert Path(info["fasta"]).exists()
    assert len(list((tmp_path / "bio" / "msas").glob("*.a3m"))) == 8


# --- edge fetch -------------------------------------------------------------


def test_edge_write_smoke_and_keyframes(tmp_path: Path):
    from benchmarks.edge.fetch import verify_counts, write_smoke
    from gitm.bench import edge_manifest as em

    counts = write_smoke(tmp_path / "edge", n_scenes=2, frames_per_scene=3, n_kitti=4)
    assert counts["nuscenes_scenes"] == 2
    assert counts["kitti_frames"] == 4
    assert counts["nuscenes_keyframes"] == 6

    assert verify_counts(tmp_path / "edge")["kitti_frames"] == 4

    rows = em.build_manifest(tmp_path / "edge")
    assert sum(r.source == "nuscenes" for r in rows) == 6
    assert sum(r.source == "kitti" for r in rows) == 4


# --- smoke harness ----------------------------------------------------------


def test_smoke_harness_payload_is_valid_contract(tmp_path: Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_smoke_harness",
        Path(__file__).resolve().parent.parent / "benchmarks" / "_smoke_harness.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    cfg = Path(__file__).resolve().parent.parent / "benchmarks" / "hft" / "bench.toml"
    payload = mod.emit(cfg, seed=42, work_units=4)
    assert "metric_value" in payload
    row = payload["stall_breakdown"][0]
    total = row["cpu"] + row["data_stall"] + row["sync"] + row["gpu_active"]
    assert total == pytest.approx(1.0, abs=1e-6)


# --- reproducibility test ---------------------------------------------------


def test_reproduce_dataset_only_pass_and_tamper(tmp_path: Path):
    from gitm.bench import manifest as m
    from gitm.bench.reproduce import reproduce
    from gitm.bench.schema import BenchConfig

    stage = tmp_path / "stage"
    _gen(stage / "hft_smoke_seed42", events=8_000, per_file=8_000)
    manifest_path = tmp_path / "manifest.yaml"
    m.write_manifest(m.build_manifest(stage, "hft"), manifest_path)

    cfg = BenchConfig.from_toml(
        Path(__file__).resolve().parent.parent / "benchmarks" / "hft" / "bench.smoke.toml"
    )

    rep = reproduce(cfg, stage_dir=stage, manifest_path=manifest_path,
                    runs_dir=tmp_path / "runs", run_metric=False)
    assert rep.dataset_ok and rep.passed
    assert "skipped" in " ".join(rep.notes)

    # tamper a byte -> dataset no longer reproduces
    shard = next(stage.glob("**/part-*.parquet"))
    shard.write_bytes(shard.read_bytes() + b"x")
    rep2 = reproduce(cfg, stage_dir=stage, manifest_path=manifest_path,
                     runs_dir=tmp_path / "runs", run_metric=False)
    assert not rep2.dataset_ok and not rep2.passed


def test_reproduce_time_budget_exceeded(tmp_path: Path):
    from gitm.bench import manifest as m
    from gitm.bench.reproduce import reproduce
    from gitm.bench.schema import BenchConfig

    stage = tmp_path / "stage"
    _gen(stage / "seed42", events=8_000, per_file=8_000)
    manifest_path = tmp_path / "manifest.yaml"
    m.write_manifest(m.build_manifest(stage, "hft"), manifest_path)
    cfg = BenchConfig.from_toml(
        Path(__file__).resolve().parent.parent / "benchmarks" / "hft" / "bench.smoke.toml"
    )

    # a clock that always reports +120 min elapsed
    ticks = iter([0.0, 120 * 60.0])
    rep = reproduce(cfg, stage_dir=stage, manifest_path=manifest_path,
                    runs_dir=tmp_path / "runs", run_metric=False,
                    limit_minutes=60.0, now=lambda: next(ticks))
    assert rep.dataset_ok
    assert not rep.within_time and not rep.passed
