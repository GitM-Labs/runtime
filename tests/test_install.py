"""Host preparation planning.

``gitm install`` executes package installations and replaces torch with a
specific CUDA build. The plan is therefore constructed as data and rendered
before anything runs, so that what will be executed is reviewable and testable
without a GPU present. These tests exercise plan construction only; no step is
executed.
"""

from __future__ import annotations

import sys

import pytest

from gitm import cuda_env
from gitm.install import Plan, Step, build_plan, main


@pytest.fixture
def driver13(monkeypatch):
    """A host reporting a CUDA 13 driver, for which a pinned stack exists."""
    monkeypatch.setattr(cuda_env, "driver_cuda", lambda: (13, 0))
    return (13, 0)


@pytest.fixture
def no_apt(monkeypatch):
    """Suppress the apt step so plans are comparable across host images."""
    monkeypatch.setattr("gitm.install.shutil.which", lambda name: None)


# ── terminal conditions ─────────────────────────────────────────────────────


def test_absent_driver_is_refused(monkeypatch):
    """Preparation targets a CUDA host; without a driver there is nothing to do."""
    monkeypatch.setattr(cuda_env, "driver_cuda", lambda: None)
    with pytest.raises(RuntimeError, match="no NVIDIA driver"):
        build_plan()


def test_unpinned_driver_is_refused_with_the_supported_set_named(monkeypatch):
    """Proceeding would install an unverified combination.

    The resulting incompatibility surfaces inside ``torch._C._cuda_init``, long
    after the weight download, so the refusal is at plan time and names both the
    supported majors and the flag that bypasses the stack entirely.
    """
    monkeypatch.setattr(cuda_env, "driver_cuda", lambda: (11, 4))
    with pytest.raises(RuntimeError, match="no pinned stack"):
        build_plan()


def test_unpinned_driver_is_acceptable_when_the_stack_is_skipped(monkeypatch, no_apt):
    """Tracer preparation does not depend on the pinned stack."""
    monkeypatch.setattr(cuda_env, "driver_cuda", lambda: (11, 4))
    plan = build_plan(skip_stack=True)
    assert plan.stack is None
    assert any("_cupti.build" in " ".join(s.argv) for s in plan.steps)


def test_driverless_host_exits_two(monkeypatch, capsys):
    monkeypatch.setattr(cuda_env, "driver_cuda", lambda: None)
    assert main(["--dry-run"]) == 2
    assert "cannot prepare this host" in capsys.readouterr().out


# ── plan content ────────────────────────────────────────────────────────────


def test_cupti_major_follows_the_driver_not_torch(driver13, no_apt):
    """The Activity API requires libcupti's major to match the driver.

    A mismatch yields CUPTI_ERROR_NOT_COMPATIBLE at collection time. Torch may
    be built against a different CUDA minor and is not the reference here.
    """
    plan = build_plan()
    cupti = next(s for s in plan.steps if s.name == "CUPTI")
    assert "nvidia-cuda-cupti" in cupti.argv
    # The -cuNN variants are deprecated; their sdists refuse to build and abort
    # the step rather than installing anything.
    assert not any(a.startswith("nvidia-cuda-cupti-cu") for a in cupti.argv)


def test_vllm_is_installed_before_torch_on_a_pinned_driver(monkeypatch, no_apt):
    """Ordering is load-bearing where a pin exists.

    vLLM pins an exact torch version which pip resolves to the default CUDA
    build from PyPI. Installing torch first would have it overwritten; torch is
    therefore forced back afterwards. Only pinned drivers do this dance.
    """
    monkeypatch.setattr(cuda_env, "driver_cuda", lambda: (12, 8))
    names = [" ".join(s.argv) for s in build_plan().steps]
    vllm_at = next(i for i, n in enumerate(names) if "vllm==" in n)
    torch_at = next(i for i, n in enumerate(names) if "torch==" in n)
    assert vllm_at < torch_at


def test_an_unpinned_driver_never_names_a_version(driver13, no_apt):
    """The regression this exists for: `gitm install` downgraded a working host.

    A CUDA 13 box running vLLM 0.27.1 was moved back to 0.25.1 because the table
    named a version. An unpinned row must emit no `==` for either package.
    """
    joined = " ".join(" ".join(s.argv) for s in build_plan().steps)
    assert "vllm==" not in joined
    assert "torch==" not in joined
    assert "--force-reinstall" not in joined
    assert "pip install -U vllm" in joined.replace("  ", " ")


def test_the_shim_build_is_always_last_and_never_optional(driver13, no_apt):
    """Every preceding step exists to make this one succeed."""
    plan = build_plan()
    last = plan.steps[-1]
    assert "gitm.tracer._cupti.build" in last.argv
    assert last.optional is False


def test_pip_runs_through_the_active_interpreter(driver13, no_apt):
    """A bare ``pip`` resolves through PATH and, in a container with several
    interpreters, installs where the tracer will never import from."""
    for step in build_plan().steps:
        if "pip" in step.argv:
            assert step.argv[0] == sys.executable
            assert step.argv[1:3] == ["-m", "pip"]


def test_skip_stack_omits_framework_installation(driver13, no_apt):
    joined = " ".join(" ".join(s.argv) for s in build_plan(skip_stack=True).steps)
    assert "vllm==" not in joined
    assert "torch==" not in joined
    assert "_cupti.build" in joined


def test_gpu_extras_are_opt_in(driver13, no_apt):
    """RAPIDS is required by the HFT harness only; the capture path does not use it."""
    assert not any("cudf" in " ".join(s.argv) for s in build_plan().steps)
    assert any("cudf" in " ".join(s.argv) for s in build_plan(with_gpu_extras=True).steps)


def test_optional_steps_are_the_ones_that_may_fail(driver13, no_apt):
    """CUPTI wheels and RAPIDS may already be present or unavailable; neither
    should abort a run whose purpose is the shim build."""
    plan = build_plan(with_gpu_extras=True)
    optional = {s.name for s in plan.steps if s.optional}
    assert "CUPTI" in optional
    assert "RAPIDS (cuDF, CuPy)" in optional


# ── rendering ───────────────────────────────────────────────────────────────


def test_render_states_the_driver_the_stack_and_every_command(driver13, no_apt):
    """--dry-run output must be sufficient to review without reading the source."""
    text = build_plan().render()
    assert "CUDA 13.0" in text
    assert "unpinned" in text
    for step in build_plan().steps:
        assert " ".join(step.argv) in text


def test_dry_run_executes_nothing(driver13, no_apt, monkeypatch, capsys):
    def explode(*a, **k):
        raise AssertionError("--dry-run must not execute a step")

    monkeypatch.setattr("gitm.install.subprocess.call", explode)
    assert main(["--dry-run"]) == 0
    assert "dry run" in capsys.readouterr().out


def test_empty_plan_renders_without_raising():
    assert "unknown" in Plan().render()


def test_step_defaults_to_required():
    assert Step("x", ["true"], "because").optional is False
