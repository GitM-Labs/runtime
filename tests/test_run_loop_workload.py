"""The loop must run a real workload and refuse to fake a result from nothing.

Covers the wiring added so ``gitm run`` actually drives a workload under the
tracer (instead of capturing an empty ``pass`` block) and the guard that reports
*no-data* rather than fabricating claims when the trace is empty.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

from .conftest import make_kernel, make_trace

# Digest of an empty kernel list — sha256(repr([]))[:16]. A real trace must not
# produce this; if it does, the workload didn't actually run.
EMPTY_DIGEST = "4f53cda18c2baa0c"


def test_no_data_guard_does_not_fabricate_claims(tmp_path: Path):
    """No GPU/shim and no registered runner → honest no-data, zero claims."""
    from gitm import optimize

    result = optimize(workload="vllm-decode", budget="1s", target=0.15, scratch=str(tmp_path))
    summary = result["summary"]

    assert summary["status"] == "no_data"
    assert summary["n_claims"] == 0
    assert summary["commit"] is False
    assert summary["diagnostic"]  # explains why nothing was measured
    assert Path(summary["report_path"]).exists()
    assert "NO DATA" in result["report_md"]


class _Runner:
    """A minimal workload_runner double — explicit about whether/what
    ``workload_id`` it carries, instead of monkey-patching a plain function
    (which silently loses the attribute if ever wrapped, e.g. functools.partial)."""

    def __init__(self, workload_id: object = "unset") -> None:
        # "unset" (the sentinel) means: don't set the attribute at all, so
        # getattr(..., "workload_id", None) exercises its own default path
        # rather than reading an attribute that happens to be None.
        if workload_id != "unset":
            self.workload_id = workload_id

    def __call__(self) -> dict:
        return {"events": 1}


def _capturing_fake_capture(captured: dict):
    """A fake ``capture()`` that records the ``workload_id`` it was called
    with, shared by the workload_id-relabeling tests below."""

    @contextmanager
    def fake_capture(out_path, *, workload_id="w", fingerprint="f", run_id=None):
        captured["workload_id"] = workload_id
        kernels = [make_kernel("k", start_ns=i * 100, end_ns=i * 100 + 90) for i in range(10)]
        yield make_trace(events=kernels, vendor="nvidia", run_id=run_id or "r")

    return fake_capture


def test_runner_workload_id_relabels_trace_when_workload_unset(tmp_path: Path, monkeypatch):
    """A factory-built runner's own ``workload_id`` (e.g. set by
    ``_vllm_decode_factory``, see #60) must relabel the trace/capture call
    instead of being silently discarded.

    Regression guard: ``run_loop`` computed its ``workload`` label from
    ``cfg.engine.workload_id`` before the runner (and ``cfg.engine``) were even
    resolved — the runner's own attribute was set on the wrong object at the
    wrong time and was never actually read, so a runner passed via
    ``workload_runner`` with no explicit ``cfg.workload`` always fell back to
    the hardcoded "vllm-decode" default regardless of what it actually was.
    """
    import gitm.scheduler.loop as loop

    captured: dict = {}
    monkeypatch.setattr(loop, "capture", _capturing_fake_capture(captured))
    monkeypatch.setattr(loop, "sync_device", lambda: None)

    from gitm import optimize

    optimize(budget="1s", scratch=str(tmp_path), workload_runner=_Runner("custom-workload"))

    assert captured["workload_id"] == "custom-workload"


def test_runner_workload_id_does_not_override_explicit_workload(tmp_path: Path, monkeypatch):
    """An explicit ``cfg.workload`` always wins over the runner's self-reported id."""
    import gitm.scheduler.loop as loop

    captured: dict = {}
    monkeypatch.setattr(loop, "capture", _capturing_fake_capture(captured))
    monkeypatch.setattr(loop, "sync_device", lambda: None)

    from gitm import optimize

    optimize(
        workload="hft", budget="1s", scratch=str(tmp_path),
        workload_runner=_Runner("custom-workload"),
    )

    assert captured["workload_id"] == "hft"


def test_runner_workload_id_none_falls_back_to_default(tmp_path: Path, monkeypatch):
    """A factory that didn't populate ``workload_id`` (explicit ``None``, a
    plausible real-world state) falls back to the guessed/default label
    rather than relabeling to "None"."""
    import gitm.scheduler.loop as loop

    captured: dict = {}
    monkeypatch.setattr(loop, "capture", _capturing_fake_capture(captured))
    monkeypatch.setattr(loop, "sync_device", lambda: None)

    from gitm import optimize

    optimize(budget="1s", scratch=str(tmp_path), workload_runner=_Runner(None))

    assert captured["workload_id"] == "vllm-decode"


def test_runner_without_workload_id_attribute_falls_back_to_default(
    tmp_path: Path, monkeypatch
):
    """A plain runner with no ``workload_id`` attribute at all (the common case
    for a directly-passed callable, not a registry factory) exercises the
    ``getattr(..., "workload_id", None)`` default path, not just the explicit
    ``None`` case above."""
    import gitm.scheduler.loop as loop

    captured: dict = {}
    monkeypatch.setattr(loop, "capture", _capturing_fake_capture(captured))
    monkeypatch.setattr(loop, "sync_device", lambda: None)

    from gitm import optimize

    optimize(budget="1s", scratch=str(tmp_path), workload_runner=_Runner())

    assert captured["workload_id"] == "vllm-decode"


def test_runner_runs_inside_capture_and_produces_real_trace(tmp_path: Path, monkeypatch):
    """An injected runner is invoked inside the capture window; with kernels in
    the trace the loop proceeds to real claims with a non-empty fingerprint."""
    import gitm.scheduler.loop as loop

    called = {"runner": False, "sync": False}

    # Fake capture: yields a populated nvidia trace so the guard passes and the
    # fingerprint reflects real kernels.
    @contextmanager
    def fake_capture(out_path, *, workload_id="w", fingerprint="f", run_id=None):
        kernels = [
            make_kernel(f"cudf_groupby_{i % 5}", start_ns=i * 100, end_ns=i * 100 + 90)
            for i in range(60)
        ]
        yield make_trace(events=kernels, vendor="nvidia", run_id=run_id or "r")

    monkeypatch.setattr(loop, "capture", fake_capture)
    monkeypatch.setattr(loop, "sync_device", lambda: called.__setitem__("sync", True))

    def runner():
        called["runner"] = True
        return {"events": 1_000}

    from gitm import optimize

    result = optimize(
        workload="hft", budget="1s", target=0.15, scratch=str(tmp_path), workload_runner=runner
    )
    summary = result["summary"]

    assert called["runner"], "runner must be invoked inside the capture window"
    assert called["sync"], "device must be synced so kernels land in the trace"
    assert summary["status"] == "ok"
    assert summary["fingerprint"].startswith("nvidia:")
    assert summary["fingerprint"] != f"nvidia:{EMPTY_DIGEST}"
    assert Path(summary["report_path"]).exists()


_VLLM_KNOBS = (
    "max_num_batched_tokens",
    "gpu_memory_utilization",
    "max_num_seqs",
    "scheduling_policy",
    "swap_space",
)


def _fake_capture_with_kernels(prefix: str):
    @contextmanager
    def fake_capture(out_path, *, workload_id="w", fingerprint="f", run_id=None):
        kernels = [
            make_kernel(f"{prefix}_{i % 4}", start_ns=i * 100, end_ns=i * 100 + 90 + (i % 9))
            for i in range(80)
        ]
        yield make_trace(events=kernels, vendor="nvidia", run_id=run_id or "r")

    return fake_capture


def test_non_vllm_workload_emits_measurement_not_vllm_claims(tmp_path: Path, monkeypatch):
    """HFT (no intervention library) must report real kernels, never vLLM knobs."""
    import gitm.scheduler.loop as loop

    monkeypatch.setattr(loop, "capture", _fake_capture_with_kernels("cudf_groupby_scan"))
    monkeypatch.setattr(loop, "sync_device", lambda: None)

    from gitm import optimize

    result = optimize(
        workload="hft-lob", budget="1s", scratch=str(tmp_path), workload_runner=lambda: {"events": 1}
    )
    summary, md = result["summary"], result["report_md"]

    assert summary["status"] == "ok"
    assert summary["mode"] == "measurement"
    assert summary["n_claims"] == 0
    assert "Measurement run" in md
    for knob in _VLLM_KNOBS:
        assert knob not in md, f"measurement report must not contain vLLM knob {knob!r}"


def test_vllm_workload_still_uses_intervention_path(tmp_path: Path, monkeypatch):
    """vllm-decode keeps the intervention/claims pipeline (the library applies)."""
    import gitm.scheduler.loop as loop

    monkeypatch.setattr(loop, "capture", _fake_capture_with_kernels("paged_attention"))
    monkeypatch.setattr(loop, "sync_device", lambda: None)

    from gitm import optimize

    result = optimize(
        workload="vllm-decode", budget="1s", scratch=str(tmp_path), workload_runner=lambda: {}
    )
    assert result["summary"]["mode"] == "intervention"


def test_cli_run_returns_nonzero_on_no_data(tmp_path: Path, capsys):
    """Automation must see a failure exit when a run measures nothing."""
    from gitm.cli import main

    rc = main(["run", "--workload", "vllm-decode", "--budget", "1s", "--scratch", str(tmp_path)])
    assert rc == 3


def test_hft_harness_importable_from_package():
    """The harness must ship in the wheel, i.e. be importable from the package."""
    from gitm.benchmarks.hft import harness

    assert harness.run_pipeline and harness.load_events and harness.select_backend


def test_hft_is_registered():
    from gitm.workloads import get_factory, registered

    assert "hft" in registered() and "hft-lob" in registered()
    assert get_factory("hft") is not None
    # vllm-decode now has a registered factory (Stream A task 1); an unknown id
    # still resolves to None.
    assert get_factory("vllm-decode") is not None
    assert get_factory("not-a-workload") is None


def test_vllm_decode_factory_returns_wired_runner(monkeypatch):
    """The vllm-decode factory MUST return a runner wired with the live engine and
    its Phase-4 A/B hooks.

    Regression guard: a refactor (#55, "real restart-apply") once dropped the
    trailing ``run.engine = ...`` wiring and ``return run``, so the factory
    returned ``None`` on the non-synthetic path. The loop then saw no runner,
    never ran the workload, and reported ``no_data`` unconditionally — and the
    structural-knob hooks (fp8 KV, quantization) were never attached. The whole
    suite stayed green because nothing exercised the factory's real return path
    (it needs ``vllm`` imported, which CPU-only CI skips). This injects a fake
    ``vllm`` module so that path is covered.
    """
    import sys
    import types

    from gitm.scheduler.loop import LoopConfig
    from gitm.workloads import get_factory

    class _Out:
        def __init__(self, n: int):
            self.outputs = [types.SimpleNamespace(token_ids=list(range(n)))]

    class _LLM:
        def __init__(self, model, **kwargs):
            self.model = model
            self.kwargs = kwargs

        def generate(self, prompts, params):
            return [_Out(params.max_tokens) for _ in prompts]

    class _SamplingParams:
        def __init__(self, max_tokens: int = 0, temperature: float = 0.0):
            self.max_tokens = max_tokens
            self.temperature = temperature

    fake = types.ModuleType("vllm")
    fake.LLM = _LLM
    fake.SamplingParams = _SamplingParams
    monkeypatch.setitem(sys.modules, "vllm", fake)
    monkeypatch.setenv("GITM_VLLM_PROMPTS", "3")
    monkeypatch.setenv("GITM_VLLM_MAX_TOKENS", "5")
    monkeypatch.delenv("GITM_VLLM_SYNTHETIC", raising=False)

    runner = get_factory("vllm-decode")(LoopConfig(workload="vllm-decode"))

    assert runner is not None, "factory returned None — engine wiring/return dropped"
    assert callable(runner)
    engine = getattr(runner, "engine", None)
    assert engine is not None, "run.engine not attached — loop stays predict-only"
    assert getattr(runner, "workload_id", None) == "vllm-decode"
    # Structural-knob hooks the loop reads off the engine (gitm.scheduler.loop
    # Phase 4). Without gitm_restart_fn, fp8/quant get rejected instead of measured.
    assert callable(getattr(engine, "gitm_throughput_fn", None))
    assert callable(getattr(engine, "gitm_restart_fn", None))
    # The observe runner produces a real token count, and the throughput probe
    # measures whatever engine it is handed (3 prompts x 5 tokens).
    assert runner()["generated_tokens"] == 3 * 5
    assert engine.gitm_throughput_fn(engine) > 0
    # The restart hook rebuilds a fresh engine with the structural knob applied.
    rebuilt = engine.gitm_restart_fn(engine, "kv_cache_dtype", "fp8")
    assert rebuilt is not engine and rebuilt.kwargs["kv_cache_dtype"] == "fp8"
