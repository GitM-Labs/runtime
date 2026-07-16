"""Driver/stack CUDA compatibility (gitm.cuda_env).
Regression cover for the failure that ate three H100 runs: the pod was rescheduled
onto a host with a CUDA 12.8 driver while torch and vLLM were CUDA 13 builds, and
nothing said so until ~90s in, after the weight download, inside _cuda_init().
"""

from __future__ import annotations

import pytest

from gitm import cuda_env


@pytest.fixture
def host(monkeypatch):
    """Set the (driver, torch, vllm) CUDA triple this box reports."""

    def _set(driver, torch=None, vllm=None):
        monkeypatch.setattr(cuda_env, "driver_cuda", lambda: driver)
        monkeypatch.setattr(cuda_env, "torch_cuda", lambda: torch)
        monkeypatch.setattr(cuda_env, "vllm_cuda_major", lambda: vllm)

    return _set


def test_no_gpu_is_not_a_problem(host):
    host(driver=None, torch=(13, 0))
    assert cuda_env.check() == []


def test_matching_majors_pass(host):
    host(driver=(13, 0), torch=(13, 0), vllm=13)
    assert cuda_env.check() == []


def test_older_build_on_newer_driver_passes(host):
    """Drivers stay compatible with older toolkits: cu12 binaries on a CUDA 13 driver."""
    host(driver=(13, 0), torch=(12, 8), vllm=12)
    assert cuda_env.check() == []


def test_minor_version_compatibility_within_a_major(host):
    """A cu12.9 build on a 12.8 driver is fine — CUDA guarantees minor compat."""
    host(driver=(12, 8), torch=(12, 9), vllm=12)
    assert cuda_env.check() == []


def test_torch_built_for_a_newer_cuda_major_is_flagged(host):
    """The exact H100 failure: torch 2.11.0+cu130 on driver 570 (CUDA 12.8)."""
    host(driver=(12, 8), torch=(13, 0))

    problems = cuda_env.check()

    assert [p.component for p in problems] == ["torch"]
    assert "cu128" in problems[0].remediation  # points at the right wheel index


def test_vllm_kernels_built_for_a_newer_cuda_major_are_flagged(host):
    """Why a torch-only downgrade could never work: vLLM's own kernels were cu13."""
    host(driver=(12, 8), torch=(12, 8), vllm=13)

    problems = cuda_env.check()

    assert [p.component for p in problems] == ["vllm"]
    # No pip flag selects vLLM's CUDA build, so we must not suggest one.
    assert "--index-url" not in problems[0].remediation


def test_both_can_be_wrong_at_once(host):
    host(driver=(12, 8), torch=(13, 0), vllm=13)
    assert {p.component for p in cuda_env.check()} == {"torch", "vllm"}


def test_require_compatible_raises_before_an_expensive_build(host):
    """torch built past the driver — caught before the 90s engine build."""
    host(driver=(12, 8), torch=(13, 0), vllm=12)

    with pytest.raises(RuntimeError, match="incompatible with this host's driver"):
        cuda_env.require_compatible()


def test_require_compatible_is_silent_when_the_stack_fits(host):
    host(driver=(13, 0), torch=(13, 0), vllm=13)
    cuda_env.require_compatible()  # must not raise


@pytest.mark.parametrize(
    ("driver", "expected"),
    [
        ((12, 8), "https://download.pytorch.org/whl/cu128"),
        ((12, 6), "https://download.pytorch.org/whl/cu126"),
        ((13, 0), "https://download.pytorch.org/whl/cu130"),
        ((14, 0), "https://download.pytorch.org/whl/cu130"),  # newest we know of
        ((11, 8), None),                                      # nothing old enough
    ],
)
def test_torch_index_picks_the_newest_cuda_the_driver_can_run(driver, expected):
    assert cuda_env.torch_index_url(driver) == expected


# --------------------------------------------------------------------------- #
# pinned stacks — "for this driver, install this torch and this vllm"          #
# --------------------------------------------------------------------------- #
def test_cuda13_host_has_a_pinned_stack():
    stack = cuda_env.stack_for((13, 0))
    assert stack is not None
    assert stack.torch == "2.11.0"
    assert stack.vllm == "0.25.1"
    assert "cu130" in stack.torch_index


def test_pip_commands_install_vllm_before_torch():
    """Order is load-bearing: vLLM pins an exact torch and pip resolves it from PyPI,
    whose default build is cu130. Torch-first would be silently overwritten."""
    cmds = cuda_env.stack_for((12, 8)).pip_commands()
    assert cmds == [
        "pip install vllm==0.19.1",
        "pip install --force-reinstall --index-url "
        "https://download.pytorch.org/whl/cu128 torch==2.10.0",
    ]


def test_cuda12_host_has_a_stack_too():
    """Production fleets run CUDA 12. Refusing them is not an answer.
    0.19.1 is the newest vLLM whose wheels link libcudart.so.12 — verified with
    readelf against the wheels themselves, not inferred from the torch pin.
    """
    stack = cuda_env.stack_for((12, 8))
    assert stack is not None
    assert stack.vllm == "0.19.1"
    assert stack.torch == "2.10.0"
    assert "cu128" in stack.torch_index


def test_cuda12_row_warns_that_it_is_a_different_engine():
    """The rows pin different vLLMs, so results are NOT comparable across hosts.
    Every vLLM from 0.20.0 up is a CUDA 13 build, so a CUDA 12 host cannot run the
    same engine a CUDA 13 host runs. That has to be loud, not buried in a pin.
    """
    cuda12, cuda13 = cuda_env.stack_for((12, 8)), cuda_env.stack_for((13, 0))
    assert cuda12.vllm != cuda13.vllm
    assert "not comparable" in cuda12.note
    assert not cuda13.note


def test_unsupported_driver_major_is_still_rejected(host, monkeypatch):
    """A driver too old for ANY pinned row (e.g. CUDA 11) fails fast, with the fix."""
    monkeypatch.delitem(cuda_env.SUPPORTED_STACKS, 12)
    host(driver=(12, 8), torch=(12, 8), vllm=12)

    with pytest.raises(RuntimeError, match="unsupported host"):
        cuda_env.require_compatible()


def test_vllm_wheel_that_is_secretly_cuda13_is_caught_in_two_seconds(host):
    """The risk in the CUDA 12 row: vLLM's PyPI wheel is built against cu130 torch.
    If its extensions link libcudart.so.13 they cannot load on a 12.8 driver no matter
    what torch we install. We read that off the .so rather than trusting the pin.
    """
    host(driver=(12, 8), torch=(12, 8), vllm=13)

    problems = cuda_env.check()

    assert [p.component for p in problems] == ["vllm"]
