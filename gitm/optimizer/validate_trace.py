"""Offline attribution gate for a benchmark capture; never claims GPU validation.

Run ``python -m gitm.optimizer.validate_trace --help``. Reports device-time
coverage and raw NVTX/name disagreements for review before comparing timings.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gitm.optimizer.deviation import classify_op, observed_op
from gitm.tracer.kernel_taxonomy import NAME_MAX, classify_phase


def audit_trace(path: Path, graph, *, pid=None, device=None) -> dict:
    expected = {n.op for n in graph.nodes}
    expected_layers = {(n.op, n.layer) for n in graph.nodes}
    workers: dict[tuple, int] = {}
    ops: dict[str, list[int]] = {}
    disagreements: dict[tuple, int] = {}
    invalid = malformed = truncated = total = nvtx = covered = kernels = 0
    bad_layers = 0
    phases: dict[str, int] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                d = json.loads(line)
            except ValueError:
                malformed += 1
                continue
            if not isinstance(d, dict):
                malformed += 1
                continue
            if d.get("kind") != "kernel":
                continue
            scope = (d.get("pid"), d.get("device_id"))
            workers[scope] = workers.get(scope, 0) + 1
            if pid is not None and scope[0] != pid:
                continue
            if device is not None and scope[1] != device:
                continue
            start, end = d.get("start_ns"), d.get("end_ns")
            if (not isinstance(start, int) or not isinstance(end, int) or end <= start):
                invalid += 1
                continue
            name = d.get("name") or ""
            raw = d.get("range_op")
            if not isinstance(name, str) or (raw is not None and not isinstance(raw, str)):
                malformed += 1
                continue
            duration = end - start
            kernels += 1
            total += duration
            op = observed_op(name, raw)
            slot = ops.setdefault(op or "<unmodeled>", [0, 0])
            slot[0] += 1
            slot[1] += duration
            if op in expected:
                covered += duration
            if raw in expected:
                nvtx += duration
                layer = d.get("range_layer")
                if layer is not None and (raw, layer) not in expected_layers:
                    bad_layers += 1
            guess = classify_op(name)
            if raw and guess and raw != guess:
                key = (raw, guess, op, name)
                disagreements[key] = disagreements.get(key, 0) + duration
            if len(name.encode("utf-8")) >= NAME_MAX:
                truncated += 1
            phase = classify_phase(name) or "unknown"
            phases[phase] = phases.get(phase, 0) + duration
    selected = [s for s in workers if (pid is None or s[0] == pid)
                and (device is None or s[1] == device)]
    return {
        "kernels": kernels, "device_time_ns": total,
        "workers": [{"pid": p, "device": d, "kernels": n}
                    for (p, d), n in workers.items()],
        "selected_workers": len(selected),
        "modeled_time_share": covered / total if total else 0,
        "canonical_nvtx_time_share": nvtx / total if total else 0,
        "malformed_lines": malformed, "invalid_kernels": invalid,
        "truncated_names": truncated, "unexpected_layer_tags": bad_layers,
        "missing_ops": sorted(expected - ops.keys()),
        "ops": ops, "direct_phase_time_ns": phases,
        "nvtx_name_disagreements": [
            {"range_op": r, "name_op": g, "resolved_op": o, "name": n, "time_ns": t}
            for (r, g, o, n), t in sorted(disagreements.items(), key=lambda x: -x[1])[:50]
        ],
    }


def main(argv=None) -> int:
    from gitm.planner.registry import _hardware, _load, _predict
    from gitm.planner.roofline import BatchConfig, ShardingConfig

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("trace", type=Path)
    ap.add_argument("--model", default="glm-5.2-fp8")
    ap.add_argument("--gpu", default="H200")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--kv-len", type=int, default=8192)
    ap.add_argument("--tp", type=int, default=8)
    ap.add_argument("--ep", type=int, default=8)
    ap.add_argument("--pid", type=int)
    ap.add_argument("--device", type=int)
    ap.add_argument("--expected-workers", type=int, default=8)
    ap.add_argument("--min-coverage", type=float, default=0.9)
    ap.add_argument("--min-nvtx", type=float, default=0.8)
    args = ap.parse_args(argv)
    if not 0 <= args.min_coverage <= 1 or not 0 <= args.min_nvtx <= 1:
        ap.error("coverage thresholds must be between 0 and 1")
    try:
        spec, family, note = _load(args.model)
        if spec is None:
            raise ValueError("model has no config-backed graph")
        graph = _predict(spec, family, _hardware(args.gpu),
                         BatchConfig(batch=args.batch, kv_cache_len=args.kv_len),
                         ShardingConfig(tp=args.tp, ep=args.ep))
        report = audit_trace(args.trace, graph, pid=args.pid, device=args.device)
    except (OSError, ValueError) as e:
        print(json.dumps({"passed": False, "errors": [str(e)]}))
        return 2
    errors = []
    if len(report["workers"]) != args.expected_workers:
        errors.append("worker count differs from --expected-workers")
    if report["selected_workers"] != 1:
        errors.append("select exactly one worker using --pid and/or --device")
    for field in ("malformed_lines", "invalid_kernels", "truncated_names",
                  "unexpected_layer_tags"):
        if report[field]:
            errors.append(field)
    if not report["kernels"]:
        errors.append("no valid selected kernels")
    if report["modeled_time_share"] < args.min_coverage:
        errors.append("modeled device-time coverage below threshold")
    if report["canonical_nvtx_time_share"] < args.min_nvtx:
        errors.append("canonical NVTX device-time coverage below threshold")
    report.update(passed=not errors, errors=errors, model=args.model, provenance=note,
                  limitation="Attribution gate only. Verify scheduler step counts, batch, "
                  "KV lengths, phase, serving precision and backend separately.")
    print(json.dumps(report, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
