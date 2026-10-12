#!/usr/bin/env python3
"""A/A noise floor at the e8 headline point — laptop side, no GPU needed.

    python3 scripts/kimi_loop/analyze_aa.py <results_dir>

<results_dir> is a pulled copy of /mnt/shared/gitm/results/<run> holding
aa/guidellm_aa_r*.json: N back-to-back reps of rag 4096/512, c=64, 180 s on
ONE arm and ONE server session (no reloads between reps). Writes
<results_dir>/analysis/aa_noise.md:

  * per-metric mean, SD, CV, min/max, and a 95% CI on the mean across reps
  * the single-run-vs-single-run noise band (+/- t * SD * sqrt(2)): the
    smallest before/after delta e8's 1-vs-1 design could tell from noise
  * if e8/ is present, its throughput delta judged against that band

Scope: within-session noise only. Reload-to-reload variance is not sampled,
so the true noise floor for an arm switch can only be wider than this.
"""

from __future__ import annotations

import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_join import fmt, guidellm_point  # noqa: E402

METRICS = ("out_tok_s", "ttft_ms_p50", "itl_ms_p50", "itl_ms_p95")

# Two-sided 95% Student-t critical values by degrees of freedom; scipy is not
# a dependency of the laptop-side scripts.
T95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    12: 2.179,
    15: 2.131,
    20: 2.086,
    30: 2.042,
}


def problem(point: dict) -> str | None:
    """Why a parsed GuideLLM run cannot be used, or None if it can.

    One rule for A/A reps and e8 arms alike: a run with errored requests, no
    completed requests, or no throughput is a failed run, not a measurement.
    """
    if point.get("error"):
        return f"unreadable ({point['error']})"
    if point.get("errored"):
        return f"{point['errored']} errored requests"
    if point.get("successful") == 0:
        return "no completed requests"
    if not point.get("out_tok_s"):
        return "no throughput reported"
    return None


def t95(df: int) -> float:
    return T95.get(df) or T95[max(k for k in T95 if k <= df)]


def main() -> int:
    run = Path(sys.argv[1]).resolve()
    files = sorted(
        (run / "aa").glob("guidellm_aa_r*.json"),
        key=lambda f: int(re.search(r"_r(\d+)", f.stem).group(1)),
    )
    if len(files) < 2:
        print(f"need >= 2 A/A reps under {run / 'aa'}, found {len(files)}")
        return 1
    reps = [guidellm_point(f) for f in files]
    bad = [r for r in reps if problem(r)]

    lines = [
        f"# A/A noise floor: rag 4096/512, c=64, 180 s ({len(reps)} reps, one session)",
        "",
        "| rep | out tok/s | TTFT p50 | ITL p50 | ITL p95 | ok/err reqs |",
        "|---|---|---|---|---|---|",
    ]
    for r in reps:
        lines.append(
            f"| {r['file']} | "
            + " | ".join(fmt(r.get(m)) for m in METRICS)
            + f" | {r.get('successful', '?')}/{r.get('errored', '?')} |"
        )
    if bad:
        lines += [
            "",
            f"**{len(bad)} rep(s) errored or unparsed:** "
            + ", ".join(r["file"] for r in bad)
            + ". Excluded below.",
        ]

    lines += [
        "",
        "| metric | n | mean | SD | CV | min | max | 95% CI on mean | 1-vs-1 noise band |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    band = {}
    for m in METRICS:
        xs = [r[m] for r in reps if r not in bad and r.get(m) is not None]
        if len(xs) < 2:
            lines.append(f"| {m} | {len(xs)} | ? | | | | | | |")
            continue
        mu, sd = statistics.mean(xs), statistics.stdev(xs)
        t = t95(len(xs) - 1)
        half = t * sd / len(xs) ** 0.5
        band[m] = t * sd * 2**0.5 / mu  # relative, for one run vs one run
        lines.append(
            f"| {m} | {len(xs)} | {mu:.2f} | {sd:.2f} | {100 * sd / mu:.2f}% "
            f"| {min(xs):.2f} | {max(xs):.2f} | {mu - half:.2f} to {mu + half:.2f} "
            f"| ±{100 * band[m]:.2f}% |"
        )

    e8 = run / "e8"
    if e8.is_dir() and "out_tok_s" in band:
        before = guidellm_point(e8 / "guidellm_before.json")
        after = guidellm_point(e8 / "guidellm_after.json")
        lever = (e8 / "LEVER").read_text().strip() if (e8 / "LEVER").is_file() else "?"
        lines += ["", f"## e8 (`{lever}`) against this floor", ""]
        skipped = [
            (arm, why) for arm, pt in (("before", before), ("after", after)) if (why := problem(pt))
        ]
        if skipped:
            lines.append(
                "No verdict: failed the health check the A/A reps use: "
                + "; ".join(f"{arm} ({why})" for arm, why in skipped)
                + "."
            )
        else:
            b, a = before["out_tok_s"], after["out_tok_s"]
            d = (a - b) / b
            beyond = abs(d) > band["out_tok_s"]
            lines += [
                f"throughput delta {100 * d:+.2f}% vs 1-vs-1 noise band "
                f"±{100 * band['out_tok_s']:.2f}%: "
                + (
                    "**outside** the within-session band."
                    if beyond
                    else "**inside** the band, so it can't be told apart from noise."
                ),
                "",
                "Caveat: e8's arms are separated by a model reload, which this "
                "A/A does not sample, so the real band for e8 is at least this wide.",
            ]

    out = run / "analysis"
    out.mkdir(exist_ok=True)
    (out / "aa_noise.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n-> {out / 'aa_noise.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
