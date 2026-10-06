"""The loop, end to end: propose arms, run them, read the results back.

This is the epic's two missing clauses meeting in one test — *"emits experiment
specs for the runtime experiment harness"* and *"must ingest experiment results
and use them to re-rank/propose the next batch"*. Each had working code and no
caller; neither had ever been shown to connect to the other.

The harness itself is stubbed: it runs on a cluster and this does not. What is
stubbed is only the part that spends GPU time — taking an arm's server argv and
producing a capture directory in the shape `gitm capture serve` writes one. The
argv the stub receives is the argv `gitm propose` emitted, and the directories it
writes are read by `gitm ingest` unmodified, so everything between the two
commands is real.
"""

from __future__ import annotations

import json

from gitm.cli import main as cli_main
from gitm.optimizer.history import load_history, record_for

from .test_harness_results import LOAD, _arm

SKU = "AMD Instinct MI355X"


def _baseline(tmp_path, argv):
    """A traced baseline capture, the input `gitm propose` takes."""
    return _arm(tmp_path / "results", "baseline", argv=argv, rps=40.0)


def _run_arm(tmp_path, arm, rps):
    """Stand in for the harness: run one arm, leave a capture directory.

    Takes the emitted argv verbatim. If propose emitted an argv the reader
    cannot diff back to the lever, this is where it shows up.
    """
    return _arm(tmp_path / "results", arm["lever"], argv=arm["serve_argv"], rps=rps)


def test_propose_then_ingest_puts_measured_levers_in_the_history(tmp_path):
    base_argv = ["--tensor-parallel-size", "8", "--enforce-eager"]
    base = _baseline(tmp_path, base_argv)

    # 1. The loop proposes a batch from the baseline capture.
    rc = cli_main(["propose", "--baseline", str(base), "--max-arms", "4",
                   "--out", str(tmp_path / "experiments.json"), "--run-id", "batch1"])
    assert rc == 0

    sweep = json.loads((tmp_path / "experiments.json").read_text())
    assert sweep["baseline"]["serve_argv"] == base_argv
    assert sweep["load"] == LOAD          # carried from the baseline, not invented
    assert sweep["served_model"] == "Kimi-K2.5"
    arms = sweep["arms"]
    assert arms, "proposed nothing to run"

    # 2. The harness runs each arm. One wins, the rest are flat.
    ingestable = [a for a in arms if a["ingestable"]]
    assert ingestable, "every proposed arm was unattributable"
    dirs = [_run_arm(tmp_path, a, rps=59.6 if i == 0 else 40.2)
            for i, a in enumerate(ingestable)]

    # 3. The results come back in.
    argv = ["ingest", "--baseline", str(base), "--scratch", str(tmp_path / "scratch"),
            "--gpu-sku", SKU, "--fingerprint", "kimi-mi355x", "--run-id", "batch1"]
    for d in dirs:
        argv += ["--candidate", str(d)]
    assert cli_main(argv) == 0

    # 4. Every lever the sweep proposed is now a record the ranking reads, under
    #    the name the catalogue knows it by — which is the whole point: a name
    #    invented from a flag is a record nothing looks up.
    hist = load_history(tmp_path / "scratch" / "runs", gpu_sku=SKU)
    for a in ingestable:
        rec = record_for(hist, a["lever"], gpu_sku=SKU, fingerprint="kimi-mi355x")
        assert rec is not None, f"{a['lever']} proposed, measured, and then lost"

    # The winner came back as a win, with the delta it actually measured.
    won = record_for(hist, ingestable[0]["lever"], gpu_sku=SKU, fingerprint="kimi-mi355x")
    assert won.wins == 1 and won.mean_delta > 0.4


def test_the_next_batch_is_ranked_from_the_last_one(tmp_path):
    """The clause that makes it a loop rather than a pipeline: a measured result
    changes what the *next* propose call puts first.

    Asserting only that the record landed in history was the weaker claim, and
    the one this test used to make. A record nothing reads back is the same
    failure as no record, one layer in.
    """
    scratch = str(tmp_path / "scratch")
    base = _baseline(tmp_path, ["--tensor-parallel-size", "8", "--enforce-eager"])
    fp = "kimi-mi355x"

    def propose(out, **kw):
        argv = ["propose", "--baseline", str(base), "--gpu-sku", SKU,
                "--fingerprint", fp, "--scratch", scratch, "--out", str(out)]
        for k, v in kw.items():
            argv += [f"--{k.replace('_', '-')}"] + ([] if v is True else [str(v)])
        assert cli_main(argv) == 0
        return json.loads(out.read_text())

    first = propose(tmp_path / "b1.json")
    order_before = [a["lever"] for a in first["arms"]]
    assert len(order_before) > 1

    # Take a lever the catalogue ranked *last* and have it win big on the cluster.
    underdog = next(a for a in reversed(first["arms"]) if a["ingestable"])
    assert underdog["lever"] != order_before[0], "pick one the prior did not favour"

    won = _run_arm(tmp_path, underdog, rps=120.0)   # 40.0 -> 120.0, +200%
    assert cli_main(["ingest", "--baseline", str(base), "--candidate", str(won),
                     "--scratch", scratch, "--gpu-sku", SKU,
                     "--fingerprint", fp, "--run-id", "batch1"]) == 0

    # The measurement, not the estimate, now decides the order.
    second = propose(tmp_path / "b2.json")
    assert second["arms"][0]["lever"] == underdog["lever"], (
        f"a +200% measured win did not reach the front: {order_before[:3]} -> "
        f"{[a['lever'] for a in second['arms']][:3]}")
    assert "scored from measured results" in second["notes"]

    # And --no-history puts it back where the catalogue had it, so the change
    # above is the history being read and not something else moving.
    third = propose(tmp_path / "b3.json", no_history=True)
    assert [a["lever"] for a in third["arms"]] == order_before


def test_an_arm_the_baseline_already_runs_is_never_proposed(tmp_path):
    """Proposing it would spend a cluster job measuring the baseline against
    itself."""
    base = _baseline(tmp_path, ["--tensor-parallel-size", "8",
                                "--enable-chunked-prefill", "--enforce-eager"])
    cli_main(["propose", "--baseline", str(base), "--gpu-sku", SKU,
              "--out", str(tmp_path / "e.json")])
    doc = json.loads((tmp_path / "e.json").read_text())

    assert "enable_chunked_prefill" not in {a["lever"] for a in doc["arms"]}
    why = {u["lever"]: u["reason"] for u in doc["unreachable"]}
    assert "already runs" in why["enable_chunked_prefill"]


def test_a_lever_that_does_not_apply_to_this_box_is_not_proposed(tmp_path):
    """A cluster job is expensive, so a lever restricted to NVIDIA parts must not
    become an arm on an MI355X sweep."""
    base = _baseline(tmp_path, ["--tensor-parallel-size", "8"])
    cli_main(["propose", "--baseline", str(base), "--gpu-sku", SKU,
              "--dtype", "bf16", "--out", str(tmp_path / "e.json")])
    doc = json.loads((tmp_path / "e.json").read_text())

    assert "attention_backend_flashinfer" not in {a["lever"] for a in doc["arms"]}
    why = {u["lever"]: u["reason"] for u in doc["unreachable"]}
    assert "A100" in why["attention_backend_flashinfer"], why["attention_backend_flashinfer"]


def test_a_dtype_restricted_lever_is_not_proposed_when_the_dtype_is_unknown(tmp_path):
    """A capture does not record the serving dtype, so without --dtype the levers
    that need a specific one are held back rather than guessed at."""
    base = _baseline(tmp_path, ["--tensor-parallel-size", "8"])
    cli_main(["propose", "--baseline", str(base), "--gpu-sku", SKU,
              "--out", str(tmp_path / "e.json")])
    why = {u["lever"]: u["reason"]
           for u in json.loads((tmp_path / "e.json").read_text())["unreachable"]}
    assert "dtype" in why["attention_backend_flashinfer"]


def test_an_untraced_baseline_is_refused_rather_than_ranked_on_nothing(tmp_path):
    base = _arm(tmp_path / "results", "untraced", rps=40.0, trace=False)
    rc = cli_main(["propose", "--baseline", str(base), "--out", str(tmp_path / "e.json")])
    assert rc == 2
    assert not (tmp_path / "e.json").exists()


def test_the_loop_closes_on_the_commands_defaults_with_no_keys_supplied(tmp_path):
    """The failure this nearly shipped with: `propose` looked history up under
    the baseline's fingerprint while `ingest` filed under an accepted arm's, and
    a lever changes kernel shapes so those differ. The generated ingest command
    passed neither key, so an operator following it verbatim got results the next
    proposal could not find — and the loop silently did not close.

    Nothing here passes --fingerprint. That is the point.
    """
    scratch = str(tmp_path / "scratch")
    base = _baseline(tmp_path, ["--tensor-parallel-size", "8", "--enforce-eager"])

    def propose(out):
        assert cli_main(["propose", "--baseline", str(base), "--gpu-sku", SKU,
                         "--scratch", scratch, "--out", str(out)]) == 0
        return json.loads(out.read_text())

    first = propose(tmp_path / "b1.json")
    order_before = [a["lever"] for a in first["arms"]]

    # The file has to carry both keys, or the command it prints cannot close it.
    assert first["gpu_sku"] == SKU
    assert first["fingerprint"]
    cmd = first["ingest"]["command"]
    assert f"--gpu-sku '{SKU}'" in cmd
    assert f"--fingerprint {first['fingerprint']}" in cmd

    underdog = next(a for a in reversed(first["arms"]) if a["ingestable"])
    assert underdog["lever"] != order_before[0]

    # The arm is traced in its own right, so its kernel shapes — and therefore
    # its own digest — are not the baseline's. Ingest keys on the baseline.
    won = _run_arm(tmp_path, underdog, rps=120.0)
    assert cli_main(["ingest", "--baseline", str(base), "--candidate", str(won),
                     "--scratch", scratch, "--gpu-sku", SKU]) == 0

    second = propose(tmp_path / "b2.json")
    assert second["arms"][0]["lever"] == underdog["lever"], (
        "a measured win did not reach the next proposal on default keys")


def test_an_arm_that_was_ingested_is_not_reported_as_refused(tmp_path):
    """An arm with no trace can still be compared. Listing it under `refused`
    because it could not supply the workload digest tells the operator a
    measurement was dropped when it was kept."""
    scratch = str(tmp_path / "scratch")
    base = _baseline(tmp_path, ["--tensor-parallel-size", "8", "--enforce-eager"])
    untraced = _arm(tmp_path / "results", "untraced-arm",
                    argv=["--tensor-parallel-size", "8"], rps=70.0, trace=False)

    assert cli_main(["ingest", "--baseline", str(base), "--candidate", str(untraced),
                     "--scratch", scratch, "--gpu-sku", SKU, "--run-id", "one"]) == 0

    run = tmp_path / "scratch" / "runs" / "one"
    doc = json.loads((run / "verification.json").read_text())
    assert len(doc["results"]) == 1          # it was ingested
    assert doc["results"][0]["intervention_name"] == "cuda_graphs_enable"

    # The baseline supplied the digest, so nothing had to be reported at all.
    if (run / "ingest_refused.json").exists():
        side = json.loads((run / "ingest_refused.json").read_text())
        assert side["refused"] == [], side


def test_arms_proposed_from_an_attached_baseline_are_commands_that_run(tmp_path):
    """An attached capture's flags are what it is compared on, but an arm is a
    command the harness runs. Built from the flags alone, every arm started
    nothing."""
    from .test_harness_results import SERVE_CMD, _attach_arm

    base = _attach_arm(tmp_path / "results", "baseline", cmdline=SERVE_CMD,
                       model="Qwen/Qwen2.5-0.5B-Instruct")
    rc = cli_main(["propose", "--baseline", str(base), "--max-arms", "3",
                   "--out", str(tmp_path / "experiments.json")])
    assert rc == 0

    sweep = json.loads((tmp_path / "experiments.json").read_text())
    launch = ["vllm", "serve", "Qwen/Qwen2.5-0.5B-Instruct"]
    assert sweep["baseline"]["serve_argv"][:3] == launch
    assert sweep["arms"], "proposed nothing to run"
    for arm in sweep["arms"]:
        assert arm["serve_argv"][:3] == launch, arm["serve_argv"]


def test_an_attached_baseline_under_a_launcher_is_refused(tmp_path, capsys):
    """torchrun or a profiler in front of vLLM shaped how the server ran. An arm
    rebuilt without it would measure another layout, so propose says so and
    proposes nothing."""
    from .test_harness_results import _attach_arm

    base = _attach_arm(tmp_path / "results", "baseline", cmdline=[
        "torchrun", "--nproc-per-node", "2", "-m",
        "vllm.entrypoints.openai.api_server", "--model", "Kimi-K2.5"])
    out = tmp_path / "experiments.json"
    assert cli_main(["propose", "--baseline", str(base), "--out", str(out)]) == 2
    assert not out.exists()
    assert "launcher" in capsys.readouterr().err
