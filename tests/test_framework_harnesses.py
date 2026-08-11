"""Tests for the biotech (OpenFold) and edge (OpenPCDet) harness scaffolding.

The real inference needs a GPU box with the framework installed; here we inject
a fake Runner and validate everything around it — work-unit iteration, the warm
window cap, contract emission, metric aggregation, and the framework-absent
error path.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# --- biotech ----------------------------------------------------------------


class _FakeAFRunner:
    name = "fake-openfold"

    def __init__(self):
        self.seen = []

    def predict(self, record, msa_path):
        self.seen.append((record.header, msa_path))
        return {"plddt": 80.0 + len(record.seq) % 10}


def _write_biotech(stage: Path, n=5, lengths=(50, 100, 400, 200, 600)):
    from benchmarks.biotech.fetch import FastaRecord, write_fasta

    recs = [FastaRecord(header=f"p{i}", seq="A" * lengths[i]) for i in range(n)]
    write_fasta(recs, stage / "proteins_50k.fasta")
    msa = stage / "msas"
    msa.mkdir(parents=True, exist_ok=True)
    for r in recs:
        (msa / f"{r.header}.a3m").write_text(f">{r.header}\n{r.seq}\n")
    return recs


def test_biotech_select_proteins_filters_and_caps():
    from benchmarks.biotech.fetch import FastaRecord
    from benchmarks.biotech.harness import select_proteins

    recs = [FastaRecord(f"p{i}", "A" * L) for i, L in enumerate((50, 400, 100, 600, 200))]
    out = select_proteins(recs, max_len=384, warm=2)
    assert [len(r.seq) for r in out] == [50, 100]  # >384 dropped, capped at 2


def test_biotech_run_with_fake_runner(tmp_path: Path):
    from benchmarks.biotech.harness import run

    _write_biotech(tmp_path)
    runner = _FakeAFRunner()
    payload = run(tmp_path, seed=42, warm=10, max_len=384, runner=runner)
    # lengths <=384: 50,100,200 -> 3 structures
    assert payload["n_structures"] == 3
    assert payload["metric_value"] > 0
    assert payload["median_plddt"] is not None
    assert payload["harness_commit"].startswith("openfold-")
    # MSA paths were resolved and passed through
    assert all(mp is not None for _, mp in runner.seen)


def test_biotech_main_emits_contract(tmp_path: Path, capsys, monkeypatch):
    from benchmarks.biotech.harness import main

    _write_biotech(tmp_path)
    monkeypatch.setenv("GITM_BENCH_STAGE", str(tmp_path))
    rc = main(["--seed", "42", "--warm-proteins", "10"], runner=_FakeAFRunner())
    assert rc == 0
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["metric_value"] > 0
    assert payload["gpu_name"]  # "cpu" on CPU box, GPU name on a GPU box


def test_biotech_missing_fasta_raises(tmp_path: Path):
    from benchmarks.biotech.harness import run

    with pytest.raises(FileNotFoundError):
        run(tmp_path, seed=42, warm=10, max_len=384, runner=_FakeAFRunner())


def test_biotech_loader_raises_without_framework():
    from benchmarks.biotech.harness import load_openfold_runner

    with pytest.raises(RuntimeError, match="OpenFold"):
        load_openfold_runner(42)


@pytest.mark.parametrize(
    "timings",
    [
        [{"_t_featurize_s": 0.0, "_t_inference_s": 0.0, "_t_post_s": 0.0, "_t_total_s": 0.0}],
        [{"_t_featurize_s": 0.6, "_t_inference_s": 0.6, "_t_post_s": 0.0, "_t_total_s": 1.0}],
    ],
)
def test_biotech_stall_breakdown_refuses_invalid_timing(timings):
    from benchmarks.biotech.harness import _build_stall_phase

    with pytest.raises(RuntimeError, match="timing"):
        _build_stall_phase(timings, 1.0)


# --- edge -------------------------------------------------------------------


class _FakePCDetRunner:
    name = "fake-openpcdet"

    def __init__(self):
        self.count = 0

    def infer(self, frame, stage):
        self.count += 1
        return {"map": 0.5, "detections": 3}


def _write_edge_manifest(stage: Path, n=8):
    stage.mkdir(parents=True, exist_ok=True)
    with open(stage / "manifest.jsonl", "w") as fh:
        for i in range(n):
            fh.write(json.dumps({
                "scene_id": "scene-0", "frame_id": f"{i:06d}",
                "lidar_path": f"x/{i}.bin", "gt_path": f"y/{i}.txt", "source": "kitti",
            }) + "\n")


def test_edge_iter_frames_caps_at_warm(tmp_path: Path):
    from benchmarks.edge.harness import iter_frames

    _write_edge_manifest(tmp_path, n=10)
    frames = list(iter_frames(tmp_path / "manifest.jsonl", warm=4))
    assert len(frames) == 4
    assert frames[0]["frame_id"] == "000000"


def test_edge_run_with_fake_runner(tmp_path: Path):
    from benchmarks.edge.harness import run

    _write_edge_manifest(tmp_path, n=8)
    runner = _FakePCDetRunner()
    payload = run(tmp_path, warm=5, runner=runner)
    assert payload["n_frames"] == 5
    assert runner.count == 5
    assert payload["metric_value"] > 0
    assert payload["mean_map"] == pytest.approx(0.5)
    assert payload["harness_commit"].startswith("openpcdet-")


def test_edge_main_emits_contract(tmp_path: Path, capsys, monkeypatch):
    from benchmarks.edge.harness import main

    _write_edge_manifest(tmp_path, n=6)
    monkeypatch.setenv("GITM_BENCH_STAGE", str(tmp_path))
    rc = main(["--seed", "42", "--warm-frames", "5000"], runner=_FakePCDetRunner())
    assert rc == 0
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["n_frames"] == 6
    assert payload["gpu_name"]  # "cpu" on CPU box, GPU name on a GPU box


def test_edge_missing_manifest_raises(tmp_path: Path):
    from benchmarks.edge.harness import run

    with pytest.raises(FileNotFoundError):
        run(tmp_path, warm=10, runner=_FakePCDetRunner())


def test_edge_loader_raises_without_framework():
    from benchmarks.edge.harness import load_openpcdet_runner

    with pytest.raises(RuntimeError, match="OpenPCDet"):
        load_openpcdet_runner()


@pytest.mark.parametrize(
    "timings",
    [
        [{"_t_load_s": 0.0, "_t_preprocess_s": 0.0, "_t_inference_s": 0.0, "_t_postprocess_s": 0.0, "_t_total_s": 0.0}],
        [{"_t_load_s": 0.4, "_t_preprocess_s": 0.3, "_t_inference_s": 0.4, "_t_postprocess_s": 0.0, "_t_total_s": 1.0}],
    ],
)
def test_edge_stall_breakdown_refuses_invalid_timing(timings):
    from benchmarks.edge.harness import _build_stall_phase

    with pytest.raises(RuntimeError, match="timing"):
        _build_stall_phase(timings, 1.0)
