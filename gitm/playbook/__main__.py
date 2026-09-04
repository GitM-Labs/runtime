"""CLI for the playbook: inspect a file, measure a distance, run a lookup.

    python -m gitm.playbook --selftest
    python -m gitm.playbook --show benchmarks/playbook/examples.json
    python -m gitm.playbook --distance benchmarks/playbook/examples.json ex1-... ex2-...
    python -m gitm.playbook --lookup benchmarks/playbook/examples.json ex2-...

``--lookup`` takes a row id and asks the playbook what it would select *for that
row's own workload*, which is the honest way to demo a lookup without a live
server: the query is a real regime, and the answer is whatever the shipped policy
says. On the example file the answer is always "nothing" — every row is
illustrative — and that is the demonstration.

All CPU-only. Nothing here applies a knob to anything.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from gitm._banner import add_banner_argument, show_banner
from gitm.playbook._selftest import run_all
from gitm.playbook.match import UNCALIBRATED_POLICY, lookup, regime_distance
from gitm.playbook.schema import Playbook


def _load(path: str) -> Playbook:
    return Playbook.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


def _find(book: Playbook, row_id: str):
    for row in book.rows:
        if row.row_id == row_id:
            return row
    raise SystemExit(f"no row {row_id!r}; have: {', '.join(r.row_id for r in book.rows)}")


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    p = argparse.ArgumentParser(prog="python -m gitm.playbook")
    add_banner_argument(p)
    p.add_argument("--selftest", action="store_true", help="run every check and exit")
    p.add_argument("--show", metavar="PLAYBOOK")
    p.add_argument("--distance", nargs=3, metavar=("PLAYBOOK", "ROW_A", "ROW_B"))
    p.add_argument("--lookup", nargs=2, metavar=("PLAYBOOK", "ROW_ID"))
    a = p.parse_args(argv)
    show_banner(suppressed=a.no_banner)

    if a.selftest:
        return run_all()

    if a.show:
        book = _load(a.show)
        print(f"{a.show}: {len(book.rows)} rows, {len(book.selectable())} selectable")
        for row in book.rows:
            print(f"  {row.summary()}")
            for note in row.notes:
                print(f"      note: {note}")
        return 0

    if a.distance:
        path, a_id, b_id = a.distance
        book = _load(path)
        d = regime_distance(
            _find(book, a_id).identity.regime, _find(book, b_id).identity.regime, UNCALIBRATED_POLICY
        )
        print(f"{a_id} vs {b_id}")
        print(f"  {d.render()}")
        return 0

    if a.lookup:
        path, row_id = a.lookup
        book = _load(path)
        result = lookup(book, _find(book, row_id).identity, UNCALIBRATED_POLICY)
        print(result.render())
        print(f"\nroute_to_discovery: {result.route_to_discovery}")
        return 0

    p.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
