"""The check that fails if the traffic library breaks.

One runnable thing, ``python -m gitm.traffic --selftest``, and the same functions
are the pytest cases in ``tests/test_traffic.py`` — no assertions written twice.

Every expected number below was measured on the committed fixtures and pinned
here. That is the point: a pinned count is a regression guard, an unpinned one is
a comment. The fixtures are **real published bytes** (the first 400 rows of each
source), so these numbers describe real production traffic, not a mock.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

from gitm.traffic.adapters import ADAPTERS, read_burstgpt, read_mooncake
from gitm.traffic.parameterize import fit, sample_trace
from gitm.traffic.regime import Regime, SourceKind
from gitm.traffic.replay import read_timed_trace, write_timed_trace
from gitm.traffic.schema import CanonicalRequest, DropReason, Trace, TraceMeta
from gitm.traffic.validate import REPLAY_THRESHOLDS, SAMPLED_THRESHOLDS, compare

#: Fixtures live beside the benchmark spec, not inside the package: they are
#: data, and the wheel ships ``gitm`` only. ``$GITM_TRAFFIC_FIXTURES`` overrides
#: for an installed checkout.
FIXTURES = Path(
    os.environ.get(
        "GITM_TRAFFIC_FIXTURES",
        Path(__file__).resolve().parents[2] / "benchmarks" / "traffic_replay" / "fixtures",
    )
)

# --- pinned on the committed fixtures ---------------------------------------
BURSTGPT_ROWS = 400
BURSTGPT_EMITTED = 383
BURSTGPT_DROPS = {"zero_input_tokens": 17}  # real: 4.3% of the slice is 0-in/0-out
BURSTGPT_LABEL = "prod/io1/in256/out128/burst-poisson/copen"

#: BurstGPT_3 (release v2.0) — the eight-column layout with Session ID and
#: Elapsed time inserted at positions 1 and 2.
BURSTGPT3_ROWS = 400
BURSTGPT3_EMITTED = 399
BURSTGPT3_DROPS = {"zero_input_tokens": 1}
BURSTGPT3_SESSION_ROWS = 393  # the rest are API-log rows with no conversation
BURSTGPT3_SESSIONS = 134
BURSTGPT3_LABEL = "prod/io2/in256/out64/burst-poisson/copen"

#: The v3 dirty fixture: 6 rows, 3 emitted, 3 defects, 1 junk Elapsed time.
BURSTGPT3_DIRTY_EMITTED = 3
BURSTGPT3_DIRTY_DROPS = {"malformed_row": 1, "non_monotonic_arrival": 1, "zero_input_tokens": 1}

MOONCAKE_ROWS = 400
MOONCAKE_EMITTED = 400
MOONCAKE_SPAN_S = 141.0
MOONCAKE_LABEL = "prod/io32/in8k/out256/burst-hi/copen"

#: GPT-4 rows in the BurstGPT slice, and the split that must reconcile.
BURSTGPT_GPT4_EMITTED = 77
BURSTGPT_GPT4_FILTERED = 320

#: Every defect reason, each firing exactly once across the two dirty fixtures.
DIRTY_ROWS = 9
DIRTY_EMITTED = 2


def _fixture(name: str) -> Path:
    p = FIXTURES / name
    if not p.exists():
        raise FileNotFoundError(
            f"fixture {p} not found. Fixtures live in benchmarks/traffic_replay/fixtures/; "
            "set $GITM_TRAFFIC_FIXTURES if this is an installed checkout."
        )
    return p


def check_burstgpt_fixture() -> None:
    """The BurstGPT adapter on real bytes, counts and regime pinned."""
    t = read_burstgpt(_fixture("burstgpt_slice.csv"))
    assert t.meta.rows_read == BURSTGPT_ROWS, t.meta.rows_read
    assert t.meta.rows_emitted == BURSTGPT_EMITTED, t.meta.rows_emitted
    assert t.meta.drops == BURSTGPT_DROPS, t.meta.drops
    assert t.meta.raw_time_unit == "s"
    assert not t.meta.has_prefix_identity  # BurstGPT has no hash_ids
    assert t.requests[0].arrival_s == 0.0  # clock anchored on the first row read
    assert Regime.from_trace(t).label() == BURSTGPT_LABEL, Regime.from_trace(t).label()


def check_burstgpt3_layout() -> None:
    """The eight-column BurstGPT_3 layout, read by column name rather than index.

    The two extra columns are *inserted* at positions 1 and 2, so a positional
    reader does not merely miss them — it reads Session ID as the model and
    Elapsed time as the request length. This is the check that the reader is
    name-based.
    """
    t = read_burstgpt(_fixture("burstgpt3_slice.csv"))
    assert t.meta.rows_read == BURSTGPT3_ROWS, t.meta.rows_read
    assert t.meta.rows_emitted == BURSTGPT3_EMITTED, t.meta.rows_emitted
    assert t.meta.drops == BURSTGPT3_DROPS, t.meta.drops

    # Session identity: present, and quantified. The flag alone would say "yes"
    # on a trace that is 90% single-shot API traffic, which the full v3 file is.
    assert t.meta.has_session_identity
    assert t.meta.session_rows == BURSTGPT3_SESSION_ROWS, t.meta.session_rows
    assert t.meta.sessions == BURSTGPT3_SESSIONS, t.meta.sessions
    assert 0 < t.meta.session_rows <= t.meta.rows_emitted

    # An empty Session ID is by design (API-log rows), never a drop.
    blank = [r for r in t.requests if r.session_id is None]
    assert blank, "no API-log rows survived; an empty session id must not drop a row"
    assert len(blank) == BURSTGPT3_EMITTED - BURSTGPT3_SESSION_ROWS

    # A real multi-turn conversation is visible, which is the point of the column.
    from collections import Counter

    turns = Counter(r.session_id for r in t.requests if r.session_id)
    assert max(turns.values()) > 1, "no multi-turn session in the fixture"

    # Elapsed time is carried, in seconds, and is not confused with a token count.
    latencies = [r.source_e2e_latency_s for r in t.requests if r.source_e2e_latency_s]
    assert latencies and all(0 <= v <= 3600 for v in latencies), latencies[:5]
    assert t.requests[0].source_e2e_latency_s == 43.0  # first row of the real file

    assert Regime.from_trace(t).label() == BURSTGPT3_LABEL, Regime.from_trace(t).label()


def check_burstgpt3_defects() -> None:
    """v3-specific handling: junk in an optional column does not lose the row."""
    t = read_burstgpt(_fixture("burstgpt3_dirty.csv"))
    assert t.meta.rows_emitted == BURSTGPT3_DIRTY_EMITTED, t.meta.rows_emitted
    assert t.meta.drops == BURSTGPT3_DIRTY_DROPS, t.meta.drops
    # An unparseable Elapsed time is read as absent and said out loud...
    assert any("unparseable" in n for n in t.meta.notes), t.meta.notes
    # ...and the request itself survives with its lengths intact.
    survived = [r for r in t.requests if r.source_e2e_latency_s is None and r.session_id]
    assert survived and survived[0].input_tokens == 120, survived


def check_burstgpt_layouts_are_read_by_name() -> None:
    """A six-column file and an eight-column file both load; junk does not."""
    v1 = read_burstgpt(_fixture("burstgpt_slice.csv"))
    v3 = read_burstgpt(_fixture("burstgpt3_slice.csv"))
    assert not v1.meta.has_session_identity and v3.meta.has_session_identity
    assert v1.meta.session_rows == 0 and v1.meta.sessions == 0
    # v1 is unchanged by v3 support — the regression this whole check exists for.
    assert v1.meta.rows_emitted == BURSTGPT_EMITTED and v1.meta.drops == BURSTGPT_DROPS
    # Every layout records the columns it actually saw.
    assert any(n.startswith("columns: ") for n in v3.meta.notes)
    # A file missing a core column is still rejected, and says which.
    try:
        read_burstgpt(_fixture("mooncake_slice.jsonl"))
    except ValueError as e:
        assert "missing column" in str(e) or "not a BurstGPT" in str(e), e
    else:
        raise AssertionError("a non-BurstGPT file was accepted as one")


def check_session_trace_replay_understates_reuse() -> None:
    """A session-aware source with no prefix hashes must say so on the plan.

    BurstGPT_3 knows which requests are turns of one conversation but not what
    they share. Synthesized unique blocks therefore understate real prefix reuse
    — the safe direction, but only if it is stated.
    """
    t = read_burstgpt(_fixture("burstgpt3_slice.csv"))
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "bg3.jsonl"
        plan = write_timed_trace(t, out)
        replayed = read_timed_trace(out)
        # The load shape crosses intact...
        assert compare(t, replayed, thresholds=REPLAY_THRESHOLDS).passed
        # ...but session identity does not, because timed_trace has no field for
        # it. Pinned so the limitation is a fact in the suite, not folklore.
        assert not any(r.session_id for r in replayed.requests)
        assert not any(r.source_e2e_latency_s for r in replayed.requests)
    assert plan.prefix_synthesized
    assert any("UNDERSTATED" in n for n in plan.notes), plan.notes
    assert any(f"{t.meta.sessions} sessions" in n for n in plan.notes), plan.notes
    assert any("NOT carried into the replay file" in n for n in plan.notes), plan.notes


def check_mooncake_fixture() -> None:
    """The Mooncake adapter, including the 512-token block reading."""
    t = read_mooncake(_fixture("mooncake_slice.jsonl"))
    assert t.meta.rows_read == MOONCAKE_ROWS, t.meta.rows_read
    assert t.meta.rows_emitted == MOONCAKE_EMITTED, t.meta.rows_emitted
    assert t.meta.drops == {}, t.meta.drops
    assert t.meta.raw_time_unit == "ms"
    assert abs(t.meta.span_s - MOONCAKE_SPAN_S) < 1e-9, t.meta.span_s
    assert t.meta.has_prefix_identity and t.meta.prefix_block_tokens == 512
    # The blocks tile the prompt exactly — the property that makes 512 the right
    # value and 16 (vLLM's default) a silent 32x truncation.
    for r in t.requests:
        assert len(r.prefix_blocks) * 512 >= r.input_tokens
        assert (len(r.prefix_blocks) - 1) * 512 < r.input_tokens
    assert Regime.from_trace(t).label() == MOONCAKE_LABEL, Regime.from_trace(t).label()


def check_every_drop_reason_fires() -> None:
    """Both dirty fixtures exercise all seven defect reasons, one row each."""
    bg = read_burstgpt(_fixture("burstgpt_dirty.csv"))
    mc = read_mooncake(_fixture("mooncake_dirty.jsonl"))
    defects = {r.value for r in DropReason} - {DropReason.FILTERED_OUT.value}
    for t in (bg, mc):
        assert t.meta.rows_read == DIRTY_ROWS, t.meta.rows_read
        assert t.meta.rows_emitted == DIRTY_EMITTED, t.meta.rows_emitted
        assert set(t.meta.drops) == defects, set(t.meta.drops)
        assert all(v == 1 for v in t.meta.drops.values()), t.meta.drops


def check_filtering_is_not_a_defect() -> None:
    """A caller-supplied filter counts separately from bad data, and reconciles."""
    t = read_burstgpt(_fixture("burstgpt_slice.csv"), model="GPT-4")
    assert t.meta.rows_emitted == BURSTGPT_GPT4_EMITTED, t.meta.rows_emitted
    assert t.meta.drops["filtered_out"] == BURSTGPT_GPT4_FILTERED, t.meta.drops
    assert t.meta.defects == t.meta.dropped - BURSTGPT_GPT4_FILTERED
    assert t.meta.rows_read == BURSTGPT_ROWS  # the whole file was still read


def check_provenance_must_reconcile() -> None:
    """A Trace whose counts do not add up cannot be constructed at all."""
    meta = TraceMeta(source="x", path="x", sha256="0" * 64, rows_read=5, rows_emitted=2)
    try:
        Trace(meta=meta, requests=[CanonicalRequest(0.0, 10, 10)] * 2)
    except ValueError as e:
        assert "unattributed" in str(e), e
    else:
        raise AssertionError("a trace with 3 unaccounted rows was accepted")


def check_replay_roundtrip() -> None:
    """The emitted timed_trace file reproduces the source, on both adapters.

    This is the brief's validation deliverable: the comparison is against the
    file bench-serve will actually read, so it is evidence about the artifact.
    """
    with tempfile.TemporaryDirectory() as td:
        for name, trace in (
            ("burstgpt", read_burstgpt(_fixture("burstgpt_slice.csv"))),
            ("mooncake", read_mooncake(_fixture("mooncake_slice.jsonl"))),
        ):
            out = Path(td) / f"{name}.jsonl"
            plan = write_timed_trace(trace, out)
            report = compare(trace, read_timed_trace(out), thresholds=REPLAY_THRESHOLDS)
            assert report.passed, f"{name}:\n{report.render()}"
            assert plan.self_timed and plan.sec_multiplier == 1.0
            assert plan.requests == len(trace)
            assert plan.source.sha256 == trace.meta.sha256  # provenance survives


def check_replay_refuses_to_truncate() -> None:
    """The two silent failures the replay path exists to make loud."""
    mc = read_mooncake(_fixture("mooncake_slice.jsonl"))
    with tempfile.TemporaryDirectory() as td:
        # vLLM's default chunk size against 512-token Mooncake blocks would
        # silently emit prompts 32x short. It must refuse instead.
        try:
            write_timed_trace(mc, Path(td) / "bad.jsonl", block_tokens=16)
        except ValueError as e:
            assert "truncated" in str(e), e
        else:
            raise AssertionError("a 32x prompt truncation was accepted")

        # A trace with no output lengths cannot be replayed as-is.
        meta = TraceMeta(source="x", path="x", sha256="0" * 64, rows_read=1, rows_emitted=1)
        no_out = Trace(meta=meta, requests=[CanonicalRequest(0.0, 128, None)])
        try:
            write_timed_trace(no_out, Path(td) / "noout.jsonl")
        except ValueError as e:
            assert "output length" in str(e), e
        else:
            raise AssertionError("a trace with no output lengths was replayed as-is")

        # BurstGPT has no prefix identity, so blocks are synthesized and said so.
        bg = read_burstgpt(_fixture("burstgpt_slice.csv"))
        plan = write_timed_trace(bg, Path(td) / "bg.jsonl")
        assert plan.prefix_synthesized
        assert any("SYNTHESIZED" in n for n in plan.notes)
        # ...and synthesized ids never collide, so no prefix sharing is invented.
        seen: set[int] = set()
        for req in read_timed_trace(Path(td) / "bg.jsonl").requests:
            assert not seen & set(req.prefix_blocks)
            seen |= set(req.prefix_blocks)


def check_regime_axes_separate_the_traces() -> None:
    """The axes have to tell the two real traces apart, or they are decoration."""
    bg = Regime.from_trace(read_burstgpt(_fixture("burstgpt_slice.csv")))
    mc = Regime.from_trace(read_mooncake(_fixture("mooncake_slice.jsonl")))
    assert mc.burstiness > 5.0 > bg.burstiness, (mc.burstiness, bg.burstiness)
    assert mc.input_p50 > 10 * bg.input_p50  # 9075 vs 353 tokens
    assert bg.label() != mc.label()
    # A scoreboard workload can never be read as production traffic.
    board = Regime.from_trace(
        read_burstgpt(_fixture("burstgpt_slice.csv")), source_kind=SourceKind.SCOREBOARD
    )
    assert board.label().startswith("board/") and bg.label().startswith("prod/")


def check_parameterized_envelope() -> None:
    """A sample matches the envelope it was fitted on; margin is labelled margin."""
    src = read_burstgpt(_fixture("burstgpt_slice.csv"))
    f = fit(src)
    inside, reg = sample_trace(f, seed=1)
    report = compare(src, inside, thresholds=SAMPLED_THRESHOLDS)
    assert report.passed, report.render()
    assert reg.in_envelope and "xenv" not in reg.label()

    outside, reg_out = sample_trace(f, rate_mult=4.0, burstiness=16.0, seed=2)
    assert not reg_out.in_envelope
    assert reg_out.label().endswith("/xenv")
    assert len(outside) > 3 * len(inside)  # 4x the rate really is 4x the load
    # The dispersion target is honoured, not merely requested.
    assert reg_out.burstiness > 4.0, reg_out.burstiness
    # A sample is reproducible from its parameters alone.
    again, _ = sample_trace(f, rate_mult=4.0, burstiness=16.0, seed=2)
    assert again.meta.sha256 == outside.meta.sha256


def check_bench_serve_argv() -> None:
    """The command line carries the flags that decide whether a replay is real."""
    mc = read_mooncake(_fixture("mooncake_slice.jsonl"))
    with tempfile.TemporaryDirectory() as td:
        plan = write_timed_trace(mc, Path(td) / "mc.jsonl")
        argv = plan.bench_serve_argv(model="Qwen/Qwen3.6-35B-A3B-FP8", max_concurrency=64)
    assert argv[:3] == ["vllm", "bench", "serve"]
    assert "--self-timed" in argv  # without this, timestamps are ignored
    assert argv[argv.index("--dataset-name") + 1] == "timed_trace"
    assert argv[argv.index("--timed-trace-chunk-hash-size") + 1] == "512"
    assert argv[argv.index("--timed-trace-sec-multiplier") + 1] == "1"
    assert argv[argv.index("--max-concurrency") + 1] == "64"
    assert argv[argv.index("--num-prompts") + 1] == str(MOONCAKE_EMITTED)


def check_banner_can_never_corrupt_stdout() -> None:
    """The banner's three guards. Cosmetic feature, real failure mode.

    A banner on stdout breaks `jq`, a `> results.json`, and every CI step that
    parses output — and it breaks them far from here, as a JSON parse error.
    """
    import io

    from gitm import _banner

    sink = io.StringIO()

    class _Tty(io.StringIO):
        def isatty(self):
            return True

    real_stdout, real_env = sys.stdout, os.environ.get(_banner.ENV_VAR)
    try:
        # not a TTY -> silent, which is the redirected/CI case
        sys.stdout = io.StringIO()
        os.environ.pop(_banner.ENV_VAR, None)
        assert _banner.show_banner(stream=sink) is False
        assert sink.getvalue() == ""

        # a TTY -> shown, and on the stream we were given, never on stdout
        sys.stdout = _Tty()
        assert _banner.show_banner(stream=sink) is True
        assert "GitM" in sink.getvalue() or "git machines" in sink.getvalue()
        assert sys.stdout.getvalue() == "", "banner reached stdout"

        # both escapes work even on a TTY
        sink.truncate(0), sink.seek(0)
        assert _banner.show_banner(suppressed=True, stream=sink) is False
        os.environ[_banner.ENV_VAR] = "1"
        assert _banner.show_banner(stream=sink) is False
        assert sink.getvalue() == ""
    finally:
        sys.stdout = real_stdout
        os.environ.pop(_banner.ENV_VAR, None)
        if real_env is not None:
            os.environ[_banner.ENV_VAR] = real_env

    # and the flag exists on every entry point, so one of them cannot drift
    import argparse

    p = argparse.ArgumentParser()
    _banner.add_banner_argument(p)
    assert p.parse_args(["--no-banner"]).no_banner is True
    assert p.parse_args([]).no_banner is False


def check_gui_refuses_paths_it_was_not_configured_for() -> None:
    """The viewer is browser-reachable, so its trace root is a trust boundary."""
    from gitm.traffic import gui

    root = FIXTURES.resolve()

    # a name from the server's own list resolves
    assert gui._resolve(root, "burstgpt_slice.csv") == root / "burstgpt_slice.csv"

    # anything that is not a bare name inside the root does not
    for bad in (
        "../../../etc/passwd",
        "..\\..\\windows\\win.ini",
        "/etc/passwd",
        "subdir/trace.csv",
        ".hidden",
        "",
        "does_not_exist.csv",
    ):
        try:
            gui._resolve(root, bad)
        except gui._Rejected:
            continue
        raise AssertionError(f"path traversal not refused: {bad!r}")

    # loopback only, and a module constant rather than a default someone can pass
    assert gui.HOST == "127.0.0.1"

    # the listing only offers files it knows an adapter for
    names = {t["name"] for t in gui._list_traces(root)}
    assert "burstgpt_slice.csv" in names and "mooncake_slice.jsonl" in names
    assert all(t["adapter"] in ADAPTERS for t in gui._list_traces(root))

    # a Host header that is not loopback is refused, so DNS rebinding cannot
    # reach the filesystem reader through a browser on another origin
    h = gui._Handler.__new__(gui._Handler)
    for host, ok in [("127.0.0.1:8765", True), ("localhost", True),
                     ("evil.example.com", False), ("", False)]:
        h.headers = {"Host": host}
        assert h._host_is_loopback() is ok, host


def check_version_guard_fires_before_launching() -> None:
    """Seam 2's whole point: a too-old vLLM must fail as a sentence, not argparse.

    Below the floor there is no ``timed_trace`` dataset and vLLM complains about
    an unknown dataset name — which reads like a typo in *our* command. The guard
    turns that into a message naming the version and the flag, before anything is
    launched.
    """
    from gitm.traffic import runner
    from gitm.traffic.replay import VLLM_MIN_VERSION

    # the release comparison, including the two ways a version string lies
    assert runner._release("0.23.0") == (0, 23, 0)
    assert runner._release("0.23.0+cu128") == (0, 23, 0)   # a build tag is not older
    assert runner._release("0.23.0rc1") == (0, 23, 0)      # nor an rc newer
    assert runner._release("0.22.1") < runner._release(VLLM_MIN_VERSION)
    assert runner._release("0.6") < runner._release(VLLM_MIN_VERSION)   # the old floor
    assert runner._release("1.0.0") > runner._release(VLLM_MIN_VERSION)

    real = runner.installed_vllm_version
    try:
        runner.installed_vllm_version = lambda: None
        try:
            runner.check_vllm()
        except runner.VllmUnavailable as e:
            assert "not installed" in str(e) and VLLM_MIN_VERSION in str(e)
        else:
            raise AssertionError("a missing vllm was not refused")

        runner.installed_vllm_version = lambda: "0.22.1"
        try:
            runner.check_vllm()
        except runner.VllmUnavailable as e:
            # the message has to name the flag, or it is the same puzzle as the
            # argparse error it exists to replace
            assert "timed_trace" in str(e) and "0.22.1" in str(e), e
        else:
            raise AssertionError("a too-old vllm was not refused")

        runner.installed_vllm_version = lambda: "0.23.0"
        assert runner.check_vllm() == "0.23.0"
    finally:
        runner.installed_vllm_version = real


def check_runner_builds_the_pinned_argv_and_keeps_provenance() -> None:
    """A dry run exercises everything except the subprocess — argv and shape."""
    from gitm.traffic import runner

    t = read_mooncake(_fixture("mooncake_slice.jsonl"))
    with tempfile.TemporaryDirectory() as d:
        plan = write_timed_trace(t, Path(d) / "replay.jsonl")
        res = runner.run_replay(plan, model="Qwen/Qwen3.6-35B-A3B-FP8",
                                result_dir=d, dry_run=True)

    assert res.ok and res.returncode == 0
    assert res.argv[:3] == ["vllm", "bench", "serve"]
    assert res.argv[res.argv.index("--dataset-name") + 1] == "timed_trace"
    assert "--self-timed" in res.argv
    # the 512-vs-16 finding survives into the command that actually runs
    assert res.argv[res.argv.index("--timed-trace-chunk-hash-size") + 1] == "512"
    assert "--save-result" in res.argv

    # provenance is carried, and it is the source trace's own, byte-identical
    assert res.source == plan.source == t.meta
    assert res.source.sha256 == t.meta.sha256 and res.source.raw_time_unit == "ms"
    # nothing was joined: seam 3 is not this module
    assert res.result is None
    assert any("dry run" in n for n in res.notes)


#: The REAL bench serve result JSON, from the 0.28.0 run in session 8. Committed
#: rather than mocked: the two fields the joiner drops are only wrong in a way a
#: mock would have gotten right by accident, and ``request_rate`` arrives as the
#: STRING "inf" because json.dumps cannot write a bare Infinity.
BENCHSERVE_RESULT = "benchserve_result.json"
REAL_RUN_REQUESTS = 40
REAL_RUN_SPAN_S = 12.0
REAL_RUN_INPUT_TOKENS = 506_280
REAL_RUN_DURATION_S = 12.00791824299995
#: The regime of the 40 rows that actually ran — NOT MOONCAKE_LABEL, which is
#: pinned for the whole 400-row fixture. A joined record must carry the regime of
#: the workload that ran, and a 40-row head of a trace is a different workload:
#: input p50 7,323 (in4k) against the full slice's 9,075 (in8k).
REAL_RUN_LABEL = "prod/io32/in4k/out256/burst-hi/copen"


def _real_run():
    """The committed result JSON, plus the plan and regime it came from."""
    result = json.loads(_fixture(BENCHSERVE_RESULT).read_text(encoding="utf-8"))
    t = read_mooncake(_fixture("mooncake_slice.jsonl"), max_rows=REAL_RUN_REQUESTS)
    with tempfile.TemporaryDirectory() as d:
        plan = write_timed_trace(t, Path(d) / "r.jsonl")
    return result, plan, Regime.from_trace(t), t


def check_join_drops_the_two_fields_that_are_wrong() -> None:
    """Seam 3's core claim, against the real result JSON.

    Under ``--self-timed`` the trace decides the schedule, so vLLM never consults
    ``request_rate`` or ``burstiness`` — but records them anyway, untouched, next
    to real metrics and on exactly the two axes the playbook keys on.
    """
    from gitm.traffic.results import MISLEADING_UNDER_SELF_TIMED, is_infinite, join_result

    result, plan, reg, _ = _real_run()
    # what the file actually says, before anything touches it
    assert result["request_rate"] == "inf"      # a STRING, not a float
    assert result["burstiness"] == 1.0
    assert is_infinite(result["request_rate"])

    run = join_result(result, plan, reg)

    # dropped, and visibly so
    assert set(run.dropped) == set(MISLEADING_UNDER_SELF_TIMED)
    assert "request_rate" not in run.metrics and "burstiness" not in run.metrics
    assert run.dropped_values["request_rate"] == "inf"
    assert run.dropped_values["burstiness"] == 1.0

    # and the truth is on the record instead, differing by a lot
    assert reg.burstiness > 5.0 and run.dropped_values["burstiness"] == 1.0
    assert reg.rate_rps > 1.0 and is_infinite(run.dropped_values["request_rate"])

    # nothing was lost: the raw JSON survives the join
    assert run.raw == result
    assert "request_rate" in run.raw


def check_join_attaches_the_identity_the_result_lacks() -> None:
    """D1's stated purpose: the regime label reaches a measured number."""
    from gitm.traffic.results import join_result

    result, plan, reg, t = _real_run()
    for k in ("regime", "regime_label", "trace", "sha256", "source"):
        assert k not in result, f"result JSON unexpectedly carries {k}"

    run = join_result(result, plan, reg)
    assert run.regime_label == reg.label() == REAL_RUN_LABEL, run.regime_label
    assert run.source.sha256 == t.meta.sha256 and len(run.source.sha256) == 64
    assert run.source.raw_time_unit == "ms"
    assert run.chunk_hash_size == 512
    assert run.config_capture == "pending-adit" and run.knobs == {}   # R1
    assert run.metrics["p99_ttft_ms"] == result["p99_ttft_ms"]


def check_join_reconciles_the_real_run() -> None:
    """The real run must pass every check; a mismatch means it is not evidence."""
    from gitm.traffic.results import join_result

    result, plan, reg, _ = _real_run()
    # the plan carries the totals a result is reconciled against
    assert plan.input_tokens_total == REAL_RUN_INPUT_TOKENS, plan.input_tokens_total
    assert abs(plan.span_s - REAL_RUN_SPAN_S) < 1e-9

    run = join_result(result, plan, reg)
    assert run.reconciled, run.render()
    assert run.promotable
    names = [c.name for c in run.checks]
    assert "input_tokens_match_trace" in names and "paced_to_trace_span" in names
    assert result["total_input_tokens"] == plan.input_tokens_total == REAL_RUN_INPUT_TOKENS
    assert abs(result["duration"] - REAL_RUN_DURATION_S) < 1e-9


def check_join_catches_the_32x_truncation_after_the_fact() -> None:
    """The emitter refuses to WRITE a truncating file; this catches one that ran.

    At vLLM's default 16-token blocks every prompt is 32x short while completed,
    duration, throughput and every percentile still read perfectly. Input tokens
    are the only number that moves.
    """
    from gitm.traffic.results import join_result

    result, plan, reg, _ = _real_run()
    truncated = dict(result, total_input_tokens=result["total_input_tokens"] // 32)
    run = join_result(truncated, plan, reg)
    assert not run.reconciled and not run.promotable
    bad = [c for c in run.failures() if c.name == "input_tokens_match_trace"]
    assert bad and "chunk-hash-size" in bad[0].detail, run.render()

    # every OTHER check still passes, which is exactly why this one is needed
    assert [c.name for c in run.failures()] == ["input_tokens_match_trace"]


def check_join_reads_both_directions_of_pacing_failure() -> None:
    """Too fast and too slow are different failures and must not be conflated."""
    from gitm.traffic.results import join_result

    result, plan, reg, _ = _real_run()

    def pacing(dur):
        run = join_result(dict(result, duration=dur), plan, reg)
        return next(c for c in run.checks if c.name == "paced_to_trace_span")

    fast = pacing(0.4)          # a client that ignored the timestamps
    assert not fast.ok and "FASTER" in fast.detail

    slow = pacing(40.0)         # a server that saturated
    assert not slow.ok and "drifted" in slow.detail

    assert pacing(REAL_RUN_DURATION_S).ok   # the real run

    # a run that was NOT self-timed has no schedule to hold, so no such check
    unpaced = plan.model_copy(update={"self_timed": False})
    run = join_result(result, unpaced, reg)
    assert "paced_to_trace_span" not in [c.name for c in run.checks]
    assert "NOT self-timed" in run.dropped["request_rate"]


def check_join_accounts_for_every_result_key() -> None:
    """A new vLLM field must be a decision, not a silent omission."""
    from gitm.traffic.results import unjoined_keys

    result = json.loads(_fixture(BENCHSERVE_RESULT).read_text(encoding="utf-8"))
    assert len(result) == 34, len(result)
    assert unjoined_keys(result) == [], unjoined_keys(result)


CHECKS = (
    check_burstgpt_fixture,
    check_burstgpt3_layout,
    check_burstgpt3_defects,
    check_burstgpt_layouts_are_read_by_name,
    check_session_trace_replay_understates_reuse,
    check_mooncake_fixture,
    check_every_drop_reason_fires,
    check_filtering_is_not_a_defect,
    check_provenance_must_reconcile,
    check_replay_roundtrip,
    check_replay_refuses_to_truncate,
    check_regime_axes_separate_the_traces,
    check_parameterized_envelope,
    check_bench_serve_argv,
    check_banner_can_never_corrupt_stdout,
    check_gui_refuses_paths_it_was_not_configured_for,
    check_version_guard_fires_before_launching,
    check_runner_builds_the_pinned_argv_and_keeps_provenance,
    check_join_drops_the_two_fields_that_are_wrong,
    check_join_attaches_the_identity_the_result_lacks,
    check_join_reconciles_the_real_run,
    check_join_catches_the_32x_truncation_after_the_fact,
    check_join_reads_both_directions_of_pacing_failure,
    check_join_accounts_for_every_result_key,
)


def run_all() -> int:
    for fn in CHECKS:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"selftest ok -- {len(CHECKS)} checks, 3 real traces, 7 drop reasons")
    return 0
