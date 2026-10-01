"""Validate experiment contracts from the command line.

    python -m gitm.experiment validate <contract.yaml> [...]

Exits 0 when every file validates, 1 otherwise, printing each file's errors.
"""

from __future__ import annotations

import sys

from pydantic import ValidationError

from gitm.experiment.contract import load_contract


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) < 2 or args[0] != "validate":
        print(__doc__.strip(), file=sys.stderr)
        return 2
    ok = True
    for path in args[1:]:
        try:
            c = load_contract(path)
        except (ValidationError, OSError, ValueError) as exc:
            ok = False
            print(f"FAIL {path}\n{exc}")
        else:
            print(f"ok   {path}  ({c.candidate_id}, sha256 {c.sha256()[:12]})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
