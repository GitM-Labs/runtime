#!/usr/bin/env python3
"""Scrub internal breadcrumbs from the runtime repo before sharing it externally.

Run from the repo root:

    python scrub.py            # edits + file/dir renames
    python scrub.py --remove-internal   # also git-rm the GTM/ops-only files

Every change lands in the working tree, so `git diff` is your review surface and
`git checkout .` is your undo. Nothing here makes a stubbed component claim to be
finished; it removes ticket IDs, sprint labels, dated/person/demo asides, and
defensive self-narration. Genuine roadmap gaps are documented honestly in
ROADMAP.md instead of scattered as apologies in comments.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# (relative path, exact old substring, new substring). Multi-line spans use \n.
EDITS: list[tuple[str, str, str]] = [
    # --- kernels / intervention library -------------------------------------
    (
        "gitm/kernels/library.yaml",
        "# and a safety gate. I2 curates entries in W2 (GITM-031 easy, GITM-032 medium,\n"
        "# GITM-033 harder); every entry carries review: null until Adit signs it off in\n"
        "# the GITM-010 library review. Expected deltas are pre-review estimates from the\n"
        "# cited source, not measured on our workloads — the replay engine and the\n"
        "# rollback-gated live apply produce the real numbers.",
        "# and a safety gate. Every entry carries review: null until it is signed off in\n"
        "# library review. Expected deltas are estimates from the cited source, not yet\n"
        "# measured on our own workloads. The replay engine and the rollback-gated live\n"
        "# apply produce the measured numbers.",
    ),
    ("gitm/kernels/library.yaml", "  # --- Tier 1: easy levers (GITM-031) ---------------------------------------",
     "  # --- Tier 1: low-effort levers --------------------------------------------"),
    ("gitm/kernels/library.yaml", "  # --- Tier 2: medium levers (GITM-032) -------------------------------------",
     "  # --- Tier 2: medium-effort levers -----------------------------------------"),
    ("gitm/kernels/library.yaml", "  # --- Tier 3: harder levers (GITM-033) -------------------------------------",
     "  # --- Tier 3: higher-effort levers -----------------------------------------"),
    (
        "gitm/kernels/__init__.py",
        "range with cited source, and safety gate. Every entry is reviewed by Adit\nbefore commit. I2 curates entries in W2 (GITM-031..033).",
        "range with cited source, and safety gate. Every entry is reviewed and signed\noff before it can be applied to a live workload.",
    ),
    ("gitm/kernels/spec.py", "    review: str | None = None  # Adit's review note when signed off",
     "    review: str | None = None  # reviewer sign-off note (None until reviewed)"),
    # --- planner -------------------------------------------------------------
    ("gitm/planner/roofline.py", "    land in a vendor catalogue at ``gitm/planner/catalogue.yaml`` (W2).",
     "    land in a vendor catalogue at ``gitm/planner/catalogue.yaml`` (roadmap)."),
    (
        "gitm/planner/graph.py",
        "Adit extends this Tue Day 2 (GITM-003) — current implementation is\nload-bearing v0, not a stub.",
        "v0 emits one decode step worth of nodes; multi-step and dependency-edge\nmodeling are on the roadmap.",
    ),
    # --- optimizer -----------------------------------------------------------
    ("gitm/optimizer/apply.py", "The vLLM/engine applicator (GITM-020) implements the same three methods.",
     "The live vLLM/engine applicator implements the same three methods (roadmap)."),
    ("gitm/optimizer/apply.py", "    Used by the embedded loop when no engine is attached (the W1 skeleton runs",
     "    Used by the embedded loop when no engine is attached (the loop runs"),
    ("gitm/optimizer/apply.py", "    a no-op rather than pretending — a live engine applicator (GITM-020) is the",
     "    a no-op rather than pretending. A live engine applicator is the"),
    (
        "gitm/optimizer/replay.py",
        "    fraction of trace time spent in ops the spec is applicable to. Adit\n    replaces this with a real replay engine in W2 (GITM-009).",
        "    fraction of trace time spent in ops the spec is applicable to. The\n    trace-driven replay engine that replaces this v0 is on the roadmap.",
    ),
    ("gitm/optimizer/dr.py", '"""Doubly-robust causal attribution (GITM-008), alongside Granger.',
     '"""Doubly-robust causal attribution, alongside Granger.'),
    ("gitm/optimizer/attribution.py", "Doubly-robust estimator lands alongside Granger in W2 (GITM-008).",
     "A doubly-robust estimator runs alongside Granger (see gitm/optimizer/dr.py)."),
    ("gitm/optimizer/replay_validation.py",
     '"""Validate the counterfactual replay engine against synthetic ground truth (GITM-009).',
     '"""Validate the counterfactual replay engine against synthetic ground truth.'),
    ("gitm/optimizer/qualification.py",
     "    The real fingerprint check lands W2 (GITM-010); this version routes the\n    plumbing end-to-end.",
     "    A richer fingerprint check is on the roadmap; this version routes the\n    plumbing end-to-end."),
    ("gitm/optimizer/multibasis.py", '"""Multi-basis anomaly confirmation (GITM-008).',
     '"""Multi-basis anomaly confirmation.'),
    # --- scheduler -----------------------------------------------------------
    ("gitm/scheduler/loop.py", "    violations = check_invariants(res)  # multi-basis confirmed (GITM-008)",
     "    violations = check_invariants(res)  # multi-basis confirmed"),
    ("gitm/scheduler/loop.py", "    dr_hypotheses = attribute_dr(res, graph)  # doubly-robust, corroborating (GITM-008)",
     "    dr_hypotheses = attribute_dr(res, graph)  # doubly-robust, corroborating"),
    ("gitm/scheduler/loop.py", "        # W1 skeleton: no live engine attached -> predict-only, unverified claims.",
     "        # No live engine attached -> predict-only, unverified claims."),
    ("gitm/scheduler/loop.py", "        # A live run passes an engine applicator here (GITM-020).",
     "        # A live run passes an engine applicator here."),
    # --- tracer --------------------------------------------------------------
    ("gitm/tracer/capture.py",
     "Capture overhead target: <5% of workload runtime\n    (W2). The W1 target is <10%.",
     "Capture overhead target: <10% of workload runtime today,\n    tightening to <5% on the roadmap."),
    ("gitm/tracer/capture.py", "    # best-effort instrumentation, not load-bearing for the workload.",
     "    # best-effort instrumentation, not critical to the workload."),
    # --- telemetry (AMD backend genuinely not implemented; keep honest) ------
    ("gitm/telemetry/backends/amd.py", "    def device_count(self) -> int:  # pragma: no cover - stub",
     "    def device_count(self) -> int:  # pragma: no cover - not yet implemented"),
    ("gitm/telemetry/backends/amd.py", '            gpu_uuid=f"amd-stub-{gpu_index}",',
     '            gpu_uuid=f"amd-{gpu_index}",'),
    ("gitm/telemetry/backends/amd.py", "    def close(self) -> None:  # pragma: no cover - stub",
     "    def close(self) -> None:  # pragma: no cover - not yet implemented"),
    # --- bench / harness demo + intern + person breadcrumbs ------------------
    ("gitm/bench/results.py", "the markdown artifact Jalon's Friday demo expects: the canonical table plus the",
     "the markdown artifact the benchmark suite emits: the canonical table plus the"),
    ("gitm/bench/reproduce.py", '"""The reproducibility test — intern-1\'s Friday deliverable, shared across benchmarks.',
     '"""The reproducibility test, shared across benchmarks.'),
    ("gitm/bench/reproduce.py", "and the Friday demo, rather than relying on a human reading log output.",
     "and CI, rather than relying on a human reading log output."),
    ("gitm/bench/profile.py", "    Pure: builds argv and the bundle skeleton without executing anything, so",
     "    Pure: builds argv and the bundle scaffold without executing anything, so"),
    ("gitm/bench/manifest.py", "This is the one-liner each dataset+reproducibility intern runs on Wednesday.",
     "This is the one-liner run to (re)generate each dataset manifest."),
    ("harness/fill_results.py", "> 2%. Flag Adit.",
     "> 2%. Flag for review."),
    # --- benchmarks (in-package) --------------------------------------------
    ("gitm/benchmarks/edge/workunit.py", "STAGE-TIMING CAVEAT (load-bearing for the stall breakdown):",
     "STAGE-TIMING CAVEAT (important for the stall breakdown):"),
    ("gitm/benchmarks/edge/baseline.py", '"near-saturated. Flag Adit / consider 500-frame fallback. "',
     '"near-saturated. Flag for review / consider 500-frame fallback. "'),
    ("gitm/benchmarks/kitti/workunit.py", '"  # Ask Adit for the checkpoint URL or pull from S3"',
     '"  # Set the checkpoint URL or pull from S3"'),
    ("gitm/benchmarks/kitti/baseline.py", 'f"Ask Adit for the S3 path, or set GITM_DATA_ROOT correctly."',
     'f"Set the S3 path, or set GITM_DATA_ROOT correctly."'),
    ("gitm/benchmarks/kitti/baseline.py", '"flag Adit: workload may be near-saturated. Consider the 500-frame fallback."',
     '"flag for review: workload may be near-saturated. Consider the 500-frame fallback."'),
    # --- top-level benchmarks data scripts + docs ---------------------------
    ("benchmarks/edge/nuscenes_source.py", "across machines (local dev box, GPU box, Friday clean-box re-run).",
     "across machines (local dev box, GPU box, clean-box re-run)."),
    ("benchmarks/edge/kitti_source.py", "across machines (local dev box, GPU box, Friday clean-box re-run).",
     "across machines (local dev box, GPU box, clean-box re-run)."),
    ("benchmarks/biotech/fetch.py", "# Pinned sources. Replace placeholders with the exact pinned URLs/commands.",
     "# Pinned sources. Fill in the exact pinned URLs/commands below."),
    ("benchmarks/biotech/fetch.py", "``{'50-99': 12, ...}`` — Tue deliverable.\"\"\"",
     "``{'50-99': 12, ...}``.\"\"\""),
    ("benchmarks/README.md", "    datasets.md          dataset description + seed protocol  (intern writes)",
     "    datasets.md          dataset description + seed protocol"),
    ("benchmarks/README.md", "    spec.md              4-section spec                       (intern writes)",
     "    spec.md              4-section spec"),
    ("benchmarks/README.md", "## Cross-cutting Friday deliverables (per benchmark)",
     "## Cross-cutting deliverables (per benchmark)"),
    ("benchmarks/hft/spec.md", "flag Adit same day", "flag for review same day"),
    ("benchmarks/biotech/spec.md", "flag Adit immediately", "flag for review immediately"),
    ("benchmarks/kitti/results.md", "message Adit before proceeding", "flag for review before proceeding"),
    ("benchmarks/kitti/spec.md", "If saturated, flag Adit same day", "If saturated, flag for review same day"),
    ("benchmarks/kitti/spec.md", "flag Adit immediately", "flag for review immediately"),
    ("benchmarks/edge/datasets_proposal.md", "> Author: Karthik — for review by Adit before adding to spec.",
     "> Draft proposal, for review before adding to the spec."),
    ("benchmarks/edge/datasets_proposal.md", "Also non-commercial only — verify with Adit before committing.",
     "Also non-commercial only; verify licensing before committing."),
    ("docs/invariants.md", "## Tier system (W2)", "## Tier system"),
    # --- stragglers (person/intern/defensive) -------------------------------
    ("tests/test_tracer_jsonl.py", "coverage of the load-bearing\nserialization contract.",
     "coverage of the\nserialization contract."),
    ("benchmarks/_smoke_harness.py",
     "Each benchmark's real work-unit (CUDA LOB kernels / OpenFold / OpenPCDet) is\nintern-2's deliverable and needs a GPU. This stand-in lets intern-1's\ndataset + reproducibility loop",
     "Each benchmark's real work-unit (CUDA LOB kernels / OpenFold / OpenPCDet)\nneeds a GPU. This stand-in lets the dataset + reproducibility loop"),
    ("benchmarks/README.md", "`make baseline` is the load-bearing reproducibility command:",
     "`make baseline` is the core reproducibility command:"),
    ("benchmarks/edge/spec.md", "> Owner: Karthik — baseline + profiling + spec doc.",
     "> Scope: baseline + profiling + spec doc."),
    # --- tests ---------------------------------------------------------------
    ("tests/test_smoke.py", '"""End-to-end smoke tests for the W1 skeleton.',
     '"""End-to-end smoke tests for the runtime loop.'),
    ("tests/test_bench_datasets.py", '"""Tests for intern-1\'s dataset + reproducibility tooling.',
     '"""Tests for the dataset + reproducibility tooling.'),
    ("tests/test_apply_rollback.py", '"""Tests for the rollback-gated intervention apply path (GITM-020/021).',
     '"""Tests for the rollback-gated intervention apply path.'),
    ("tests/test_apply_rollback.py", "from benchmarks.skeleton.measure_overhead import measure_overhead",
     "from benchmarks.overhead.measure_overhead import measure_overhead"),
]

# Edits applied AFTER renames, addressed by their NEW path.
EDITS_POST_RENAME: list[tuple[str, str, str]] = [
    ("scripts/run_on_real_trace.py", '"""Run the W2 runtime (monitor + attribution) on a REAL captured A100 trace.',
     '"""Run the runtime (monitor + attribution) on a real captured A100 trace.'),
    ("scripts/run_on_real_trace.py", "Unit tests (tests/test_w2_runtime.py) prove the algorithms are correct on",
     "Unit tests (tests/test_runtime_on_trace.py) prove the algorithms are correct on"),
    ("scripts/run_on_real_trace.py", 'with capture(out, workload_id="w2-real") as tr:',
     'with capture(out, workload_id="real-trace") as tr:'),
    ("scripts/run_on_real_trace.py", "    # Real stream-concurrency from the trace (was the hardcoded-0.0 stub).",
     "    # Real stream-concurrency computed from the trace stream IDs."),
    ("scripts/run_on_real_trace.py", 'print("PASS: the W2 runtime ran end-to-end on real A100 kernel data")',
     'print("PASS: the runtime ran end-to-end on real A100 kernel data")'),
    ("tests/test_runtime_on_trace.py", '"""Tests for the W2 runtime upgrades (GITM-008/009).',
     '"""Tests for the runtime monitor + attribution upgrades.'),
    ("benchmarks/overhead/measure_overhead.py", '"""Measure trace-capture overhead — GITM-017 (W1 target <10%, W2 target <5%).',
     '"""Measure trace-capture overhead (current target <10%, tightening to <5%).'),
    ("benchmarks/overhead/measure_overhead.py", "    python -m benchmarks.skeleton.measure_overhead --runs 3 --steps 100",
     "    python -m benchmarks.overhead.measure_overhead --runs 3 --steps 100"),
    ("benchmarks/overhead/measure_overhead.py", '          f"(W1 target <10%, W2 target <5%)")',
     '          f"(target <10%, tightening to <5%)")'),
    ("benchmarks/overhead/overhead.md", "# Trace-capture overhead — GITM-017", "# Trace-capture overhead"),
    ("benchmarks/overhead/overhead.md",
     "**Target:** W1 < 10 %, W2 < 5 % (after the buffered/async-I/O pass, GITM-018).",
     "**Target:** < 10 %, tightening to < 5 % after the buffered/async-I/O pass."),
    ("benchmarks/overhead/overhead.md", "python -m benchmarks.skeleton.measure_overhead --runs 3 --steps 100",
     "python -m benchmarks.overhead.measure_overhead --runs 3 --steps 100"),
    ("benchmarks/overhead/overhead.md", "in on the dev box before W1 sign-off.", "in on the dev box before sign-off."),
    ("benchmarks/overhead/overhead.md", "## W2 reduction (GITM-018)", "## Overhead reduction (roadmap)"),
    ("benchmarks/overhead/overhead.md",
     "If the W1 measurement exceeds 5 %, the hot path is the synchronous per-event",
     "If the measurement exceeds 5 %, the hot path is the synchronous per-event"),
    ("benchmarks/overhead/overhead.md", "JSONL write in [`capture.py`](../../gitm/tracer/capture.py). The W2 fix is",
     "JSONL write in [`capture.py`](../../gitm/tracer/capture.py). The planned fix is"),
    ("benchmarks/overhead/overhead.md", "The load-bearing A100 numbers must be filled",
     "The headline A100 numbers must be filled"),
]

RENAMES: list[tuple[str, str]] = [
    ("scripts/w2_on_real_trace.py", "scripts/run_on_real_trace.py"),
    ("tests/test_w2_runtime.py", "tests/test_runtime_on_trace.py"),
    ("benchmarks/skeleton", "benchmarks/overhead"),
]

# GTM / ops-only files: not runtime code, imported by nothing. Remove from any
# externally shared copy (move to a private repo). Only acted on with --remove-internal.
INTERNAL_ONLY: list[str] = [
    "skills",                 # GTM agent skills: context-store, status-loop, signal-scan
    "docs/scoring",           # lead-scoring contract
]


def _git(*args: str) -> None:
    subprocess.run(["git", *args], check=True)


def apply_edits(edits: list[tuple[str, str, str]]) -> None:
    misses = 0
    for rel, old, new in edits:
        p = Path(rel)
        if not p.exists():
            print(f"  MISS (no file): {rel}")
            misses += 1
            continue
        text = p.read_text(encoding="utf-8")
        if old not in text:
            print(f"  MISS (no match): {rel} :: {old[:60]!r}")
            misses += 1
            continue
        p.write_text(text.replace(old, new), encoding="utf-8")
    if misses:
        print(f"  ({misses} misses above, applied the rest)")


def main() -> int:
    if not Path(".git").exists():
        print("run from the repo root (no .git here)")
        return 1

    print("renaming W2-labeled files and the skeleton dir...")
    for old, new in RENAMES:
        if Path(old).exists():
            _git("mv", old, new)

    print("editing pre-rename paths...")
    apply_edits(EDITS)
    print("editing post-rename paths...")
    apply_edits(EDITS_POST_RENAME)

    if "--remove-internal" in sys.argv:
        print("removing GTM/ops-only files (--remove-internal)...")
        for rel in INTERNAL_ONLY:
            if Path(rel).exists():
                _git("rm", "-r", "-q", rel)
    else:
        print("\nNOT removed (run with --remove-internal, or git rm yourself):")
        for rel in INTERNAL_ONLY:
            print(f"  {rel}")

    print("\nDone. Review with:  git diff   (and git status for renames)")
    print("Undo everything with:  git checkout . && git clean -fd")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

