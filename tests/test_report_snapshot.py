"""Golden-file snapshot test for the report Jinja2 template.

The provenance report is the customer-visible artifact — every change to its
template or the summary logic should be a deliberate review, not a silent
diff. This test renders a frozen claims + provenance fixture and asserts the
output is byte-equal to ``tests/golden/report_basic.md``.

When an intentional template change lands, regenerate the golden:

    UPDATE_GOLDENS=1 .venv/bin/pytest tests/test_report_snapshot.py

then review the diff with ``git diff tests/golden/`` and commit both the
template change and the regenerated golden in the same PR.

All non-deterministic inputs (``time.time_ns``, ``git_sha``, etc.) are pinned
directly on the ``Provenance`` object — ``build_provenance`` is *not* used in
this test on purpose.
"""

from __future__ import annotations

import os
from pathlib import Path

from gitm.optimizer.report import Claim, Provenance, write_report

GOLDEN_PATH = Path(__file__).parent / "golden" / "report_basic.md"

# A fixed nanosecond timestamp the template's ``now_ns`` will be pinned to.
# Picked once, kept forever.
_PINNED_NS = 1_700_000_001_000_000_000


def _fixed_provenance() -> Provenance:
    """Pinned Provenance with no calls into the environment.

    Every non-deterministic field that would normally drift across runs
    (git_sha, gitm_version, ended_at_ns) is hard-coded here so the rendered
    output is byte-stable.
    """
    return Provenance(
        workload_id="vllm-decode",
        fingerprint="nvidia:0123456789abcdef",
        run_id="run_test_0001",
        git_sha="deadbeef",
        gitm_version="0.0.1-test",
        started_at_ns=1_700_000_000_000_000_000,
        ended_at_ns=_PINNED_NS,
        trace_path="/fixtures/trace.jsonl",
        rejected_candidates=["spec_block_size_32 (policy.skip_high_risk)"],
        rolled_back=[],
    )


def _fixed_claims() -> list[Claim]:
    return [
        Claim(
            summary="Set PagedAttention block size to 16",
            residual_invariant="kernel_time",
            residual_value=0.35,
            causal_evidence="mlp_gate_up→attn_score_value (p=0.02)",
            intervention_name="kv_cache_block_size_16",
            predicted_delta=0.05,
            measured_delta=0.048,
            rolled_back=False,
        ),
        Claim(
            summary="Raise GPU memory utilization to 0.92",
            residual_invariant="memory_traffic",
            residual_value=0.28,
            causal_evidence="paged_attention→attn_out_proj (p=0.04)",
            intervention_name="gpu_memory_utilization_092",
            predicted_delta=0.03,
            measured_delta=None,  # unverified
            rolled_back=False,
        ),
    ]


def test_report_renders_byte_equal_to_golden(monkeypatch):
    # Pin time.time_ns so the template's ``now_ns`` is deterministic.
    monkeypatch.setattr(
        "gitm.optimizer.report.time.time_ns", lambda: _PINNED_NS
    )

    rendered = write_report(
        claims=_fixed_claims(),
        provenance=_fixed_provenance(),
        qualification_diagnostic="",
    )

    # `UPDATE_GOLDENS=1 pytest tests/test_report_snapshot.py` writes the
    # current output as the new golden. Use after an intentional template
    # change; never commit the env var.
    if os.environ.get("UPDATE_GOLDENS") == "1":
        GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN_PATH.write_text(rendered, encoding="utf-8")

    assert GOLDEN_PATH.exists(), (
        f"golden file missing: {GOLDEN_PATH}\n"
        "Generate it with: UPDATE_GOLDENS=1 .venv/bin/pytest "
        "tests/test_report_snapshot.py"
    )
    expected = GOLDEN_PATH.read_text(encoding="utf-8")

    assert rendered == expected, (
        "report.md drifted from the golden snapshot. If the change was "
        "intentional, regenerate via:\n"
        "    UPDATE_GOLDENS=1 .venv/bin/pytest tests/test_report_snapshot.py\n"
        "and review the diff with `git diff tests/golden/` before committing."
    )
