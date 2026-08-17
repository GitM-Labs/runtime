"""Compare two verification reports — "did we both get the same results?".

    python scripts/compare_results.py reference.json mine.json

Splits fields into two classes:

* **Must match exactly** — the reproducibility contract. Code (git_sha), pinned
  package versions, Python, and dataset manifest sha256s. Any mismatch here means
  you are not running the same thing, so results are not comparable.
* **Advisory** — GPU SKU / driver / CUDA. These don't have to match, but perf
  numbers are only comparable across the *same* GPU SKU, so a mismatch is flagged
  loudly.

Exit 0 if the exact-match contract holds, 1 otherwise. Performance numbers
themselves live in the per-benchmark BaselineRun JSONs and are gated separately
by ``gitm.bench`` (the <2% spread rule) — this tool checks you're on the same
software+data footing for that comparison to mean anything.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EXACT = ["schema", "git_sha", "gitm_version", "python", "dataset_manifests"]
EXACT_PKGS = ["pydantic", "numpy", "pandas", "pyarrow", "torch", "cudf-cu12"]
ADVISORY = ["gpu"]


def _load(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def compare(ref: dict, other: dict) -> tuple[list[str], list[str]]:
    """Return (mismatches, advisories)."""
    mismatches: list[str] = []
    advisories: list[str] = []

    for f in EXACT:
        ref_value = ref.get(f)
        other_value = other.get(f)
        if f not in ref or f not in other or ref_value is None or other_value is None:
            mismatches.append(
                f"{f}: required verification field is missing or unavailable "
                f"({ref_value!r} != {other_value!r})"
            )
        elif ref_value != other_value:
            mismatches.append(f"{f}: {ref_value!r} != {other_value!r}")

    rp, op = ref.get("packages"), other.get("packages")
    for pkg in EXACT_PKGS:
        ref_value = rp.get(pkg) if isinstance(rp, dict) else None
        other_value = op.get(pkg) if isinstance(op, dict) else None
        if (
            not isinstance(rp, dict)
            or not isinstance(op, dict)
            or pkg not in rp
            or pkg not in op
            or ref_value is None
            or other_value is None
        ):
            mismatches.append(
                f"packages.{pkg}: required version is missing or unavailable "
                f"({ref_value!r} != {other_value!r})"
            )
        elif ref_value != other_value:
            mismatches.append(f"packages.{pkg}: {ref_value} != {other_value}")

    if ref.get("git_dirty") or other.get("git_dirty"):
        advisories.append("a report was produced from a DIRTY git tree "
                          "(uncommitted changes) — not reproducible")

    for f in ADVISORY:
        if ref.get(f) != other.get(f):
            advisories.append(f"{f} differs: {ref.get(f)} vs {other.get(f)} "
                              "(perf numbers only comparable on the same GPU SKU)")
    return mismatches, advisories


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Compare two GITM verify reports.")
    p.add_argument("reference", type=Path)
    p.add_argument("other", type=Path)
    args = p.parse_args(argv)

    mismatches, advisories = compare(_load(args.reference), _load(args.other))

    for a in advisories:
        print(f"  ADVISORY: {a}")
    if mismatches:
        print(f"\n❌ not REPRODUCIBLE — {len(mismatches)} exact-match field(s) differ:")
        for m in mismatches:
            print(f"  - {m}")
        return 1
    print("\n✅ REPRODUCIBLE — code, deps, Python, and dataset manifests all match."
          + ("" if not advisories else " (see advisories above)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
