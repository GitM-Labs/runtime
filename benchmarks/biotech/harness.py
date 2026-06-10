"""Biotech baseline harness — AlphaFold2 inference via OpenFold v1.0.1.

The work unit is one protein, single-seed AF2 inference (5 recycles, 1 model),
length ≤ 384. The metric is ``structures_per_hour`` over a warm window of
proteins. Unlike the HFT harness there is no CPU equivalent of AF2 — OpenFold is
the workload — so this module is framework-integration code that runs only on a
GPU box with OpenFold + weights installed.

To keep the *harness scaffolding* (work-unit iteration, warm-window timing,
contract emission, plDDT aggregation, stall breakdown) testable without a GPU,
the per-protein inference is behind a small ``Runner`` seam:
:func:`load_openfold_runner` builds the real one; tests inject a fake. The runner
contract is one method — ``predict(record, msa_path) -> {"plddt": float, ...}``.
The real runner additionally returns ``_t_*`` per-phase timings so :func:`run`
can emit a coarse stall breakdown (data_stall = MSA load + featurize,
gpu_active = Evoformer + structure module, sync ≈ recycle barriers) without an
external profiler. The *kernel-level* split (and the concurrency/sync invariant)
comes from running this in-process under ``gitm.tracer.capture`` — see
``scripts/run_under_runtime.py``.

Prints the one-line harness contract on stdout: ``metric_value`` =
structures/hour, plus device info and median plDDT as an auxiliary sanity field.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Protocol

from benchmarks.biotech.fetch import FastaRecord, read_fasta

OPENFOLD_COMMIT = "v1.0.1"  # pinned; weight hashes pinned in datasets.md
MODEL_NAME = "model_1"      # single-model AF2 monomer; weights = params_model_1.npz
RECYCLES = 5                # max_recycling_iters (spec §2); fixed across seeds
MODELS = 1
WARMUP_DEFAULT = 3          # untimed forward passes before the measured window


class Runner(Protocol):
    """Per-protein inference seam. The real impl wraps an OpenFold model."""

    name: str

    def predict(self, record: FastaRecord, msa_path: Path | None) -> dict: ...


def load_openfold_runner(seed: int, *, recycles: int = RECYCLES, chunk_size: int | None = None):
    """Build the real OpenFold runner (pinned commit + weights). GPU-only."""
    try:
        import numpy as np  # type: ignore
        import openfold  # type: ignore  # noqa: F401
        import torch  # type: ignore
        from openfold.config import model_config  # type: ignore
        from openfold.data import data_pipeline, feature_pipeline  # type: ignore
        from openfold.model.model import AlphaFold  # type: ignore
        from openfold.utils.import_weights import import_jax_weights_  # type: ignore
    except Exception as exc:  # pragma: no cover - framework absent on laptop
        raise RuntimeError(
            "OpenFold/torch not importable — the biotech harness runs on a GPU "
            "box with OpenFold v1.0.1 installed (in the `openfold_env` conda env "
            "on the staging pod). The dataset + reproducibility loop is exercised "
            "via the CPU smoke harness instead."
        ) from exc

    # OpenFold has no seed_everything helper; the reference script just seeds
    # torch.
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False  # variable seq lengths -> autotune churn

    config = model_config(MODEL_NAME, train=False, low_prec=False)
    config.globals.chunk_size = chunk_size  # None = no Evoformer chunking (fastest; fits ≤384 on 32GB)
    config.data.common.max_recycling_iters = recycles
    config.data.predict.max_recycling_iters = recycles

    model = AlphaFold(config)

    weights_path = Path(
        os.environ.get("OPENFOLD_WEIGHTS", "/workspace/af2_data/params/params_model_1.npz")
    )
    if not weights_path.exists():
        raise FileNotFoundError(
            f"OpenFold weights not found at {weights_path}. Set OPENFOLD_WEIGHTS "
            "(on the staging pod the AF2 params live at "
            "/workspace/af2_data/params/params_model_1.npz)."
        )
    # Remap AF2 JAX params -> OpenFold torch modules. Must match MODEL_NAME.
    import_jax_weights_(model, str(weights_path), version=MODEL_NAME)
    model = model.eval().to(device)

    feat_pipeline = feature_pipeline.FeaturePipeline(config.data)
    data_proc = data_pipeline.DataPipeline(template_featurizer=None)

    class OpenFoldRunner:
        name = f"openfold-{OPENFOLD_COMMIT}"

        def predict(self, record: FastaRecord, msa_path: Path | None) -> dict:
            tag = record.header.split()[0]
            # process_fasta reads a real one-sequence FASTA + a per-sequence
            # alignment dir (precomputed .a3m/.sto). msa_path may be that dir, or
            # a single .a3m file whose parent we use.
            if msa_path is None:
                raise FileNotFoundError(f"no precomputed alignments for {tag}")
            alignment_dir = msa_path if msa_path.is_dir() else msa_path.parent

            tmp_fasta = Path(os.environ.get("TMPDIR", "/tmp")) / f"of_{os.getpid()}_{tag}.fasta"
            tmp_fasta.write_text(f">{tag}\n{record.seq}\n")
            try:
                t0 = time.perf_counter()
                feature_dict = data_proc.process_fasta(
                    fasta_path=str(tmp_fasta), alignment_dir=str(alignment_dir)
                )
                processed = feat_pipeline.process_features(feature_dict, mode="predict")
                batch = {k: torch.as_tensor(v, device=device) for k, v in processed.items()}
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t1 = time.perf_counter()  # data_stall boundary (MSA load + featurize + H2D)

                with torch.no_grad():
                    out = model(batch)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t2 = time.perf_counter()  # gpu_active boundary (Evoformer + structure, all recycles)

                plddt = out["plddt"].mean().item()  # OpenFold plddt is already 0–100
                t3 = time.perf_counter()  # postprocess / D2H
            finally:
                tmp_fasta.unlink(missing_ok=True)

            return {
                "plddt": plddt,
                "_t_featurize_s": t1 - t0,
                "_t_inference_s": t2 - t1,
                "_t_post_s": t3 - t2,
                "_t_total_s": t3 - t0,
            }

    return OpenFoldRunner()


def _msa_path(stage: Path, record: FastaRecord) -> Path | None:
    """Resolve a protein's precomputed alignments.

    Prefers a per-tag *directory* (OpenFold precompute layout:
    ``msas/<tag>/{uniref90_hits.sto, bfd_uniref_hits.a3m, ...}``); falls back to
    a single ``msas/<tag>.a3m`` (smoke/synthetic). Returns ``None`` if neither.
    """
    tag = record.header.split()[0]
    d = stage / "msas" / tag
    if d.is_dir():
        return d
    f = stage / "msas" / f"{tag}.a3m"
    return f if f.exists() else None


def select_proteins(records: list[FastaRecord], *, max_len: int, warm: int) -> list[FastaRecord]:
    """Length-filtered warm window, in file order (deterministic)."""
    eligible = [r for r in records if len(r.seq) <= max_len]
    return eligible[:warm]


def _build_stall_phase(timings: list[dict], wall_clock_s: float) -> dict:
    """Aggregate per-protein ``_t_*`` timings into one StallPhase-compatible dict.

    Coarse split: data_stall = MSA load + featurize + H2D, gpu_active = Evoformer
    + structure module (all recycles), sync ≈ the small D2H/postprocess tail.
    Recycle barriers run *inside* model() and are folded into gpu_active here; the
    true sync/concurrency fraction comes from the CUPTI trace under the runtime.
    """
    t_feat = sum(t["_t_featurize_s"] for t in timings)
    t_inf = sum(t["_t_inference_s"] for t in timings)
    t_post = sum(t["_t_post_s"] for t in timings)
    total = max(sum(t["_t_total_s"] for t in timings), 1e-9)

    data_stall = min(1.0, t_feat / total)
    gpu_active = min(1.0, t_inf / total)
    sync = min(1.0, t_post / total)
    cpu = max(0.0, 1.0 - data_stall - gpu_active - sync)

    return {
        "phase": "all",
        "cpu": round(cpu, 4),
        "data_stall": round(data_stall, 4),
        "sync": round(sync, 4),
        "gpu_active": round(gpu_active, 4),
        "throughput": len(timings) / wall_clock_s,
        "wall_clock_s": round(wall_clock_s, 3),
    }


def run(
    stage: Path,
    seed: int,
    *,
    warm: int,
    max_len: int,
    runner: Runner,
    warmup: int = 0,
) -> dict:
    """Run the warm window through ``runner`` and return the contract payload.

    ``warmup`` untimed forward passes (on the leading proteins) run first so
    first-call kernel JIT/autotune and allocator growth don't pollute the timed
    window — important for a stable ``structures_per_hour`` and a clean trace.
    """
    fasta = stage / "proteins_50k.fasta"
    if not fasta.exists():
        raise FileNotFoundError(f"missing {fasta} — run the biotech dataset pipeline first")

    proteins = select_proteins(read_fasta(fasta), max_len=max_len, warm=warm)
    if not proteins:
        raise RuntimeError(f"no proteins with length <= {max_len} in {fasta}")

    for r in proteins[: min(warmup, len(proteins))]:
        runner.predict(r, _msa_path(stage, r))

    plddts: list[float] = []
    timings: list[dict] = []
    t0 = time.perf_counter()
    for r in proteins:
        result = runner.predict(r, _msa_path(stage, r))
        if "plddt" in result:
            plddts.append(float(result["plddt"]))
        if "_t_total_s" in result:
            timings.append(result)
    elapsed = max(time.perf_counter() - t0, 1e-9)

    structures_per_hour = len(proteins) / elapsed * 3600.0
    payload: dict = {
        "metric_value": structures_per_hour,
        "n_structures": len(proteins),
        "median_plddt": statistics.median(plddts) if plddts else None,
        "harness_commit": f"openfold-{OPENFOLD_COMMIT}",
    }
    if timings:
        payload["stall_breakdown"] = [_build_stall_phase(timings, elapsed)]
    return payload


def _device_info() -> tuple[str, int]:
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0), torch.cuda.device_count()
    except Exception:
        pass
    return "cpu", 0


def main(argv: list[str] | None = None, *, runner: Runner | None = None) -> int:
    p = argparse.ArgumentParser(description="Biotech AF2 harness (OpenFold).")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--warm-proteins", type=int, default=1000)
    p.add_argument("--max-len", type=int, default=384)
    p.add_argument("--warmup", type=int, default=WARMUP_DEFAULT)
    p.add_argument("--stage", type=Path, default=None)
    args, _ = p.parse_known_args(argv)

    stage = args.stage or Path(os.environ.get("GITM_BENCH_STAGE", "."))
    # Injected fake runners (tests) skip warm-up; only the real path needs it.
    warmup = 0 if runner is not None else args.warmup
    runner = runner or load_openfold_runner(args.seed)
    gpu_name, device_count = _device_info()

    payload = run(
        stage, args.seed, warm=args.warm_proteins, max_len=args.max_len,
        runner=runner, warmup=warmup,
    )
    payload.update({"gpu_name": gpu_name, "device_count": device_count})

    print(f"[biotech harness:{getattr(runner, 'name', '?')}] "
          f"{payload['n_structures']} structures, "
          f"{payload['metric_value']:.1f} structures/hour")
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
