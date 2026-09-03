"""The ``GitM`` banner, and the three rules that stop it breaking a pipe.

Decoration on a CLI that also emits JSON and parseable tables is a hazard, not a
feature, so this module is mostly the guard rather than the art:

1. **stderr, never stdout.** Every byte on stdout belongs to whoever is reading
   it — `jq`, a CI step, a `> results.json`. A banner there corrupts the payload,
   and the failure shows up far from here as a JSON parse error.
2. **Only when stdout is a TTY.** Redirected stdout means a machine is reading,
   and a machine reading stdout usually means stderr is being captured into the
   same log. Gate on stdout even though we write to stderr: the question is "is a
   human watching this run", and stdout is where the answer is.
3. **Two escapes.** ``--no-banner`` for one invocation, ``GITM_NO_BANNER`` for a
   shell, a Makefile or a container that should never show it.

Cosmetic by intent. The point of writing the guard out is that the cosmetic part
is the part that is safe to change.
"""

from __future__ import annotations

import os
import sys

#: Set to anything non-empty to suppress the banner everywhere.
ENV_VAR = "GITM_NO_BANNER"

BANNER = r"""
   ___ _ _   __  __
  / __(_) |_|  \/  |   git machines
 | (_ | |  _| |\/| |   inference runtime
  \___|_|\__|_|  |_|
"""


def show_banner(*, suppressed: bool = False, stream=None) -> bool:
    """Write the banner to stderr. Returns whether it was actually written.

    ``suppressed`` is the parsed ``--no-banner`` flag. ``stream`` exists for the
    selftest; production callers never pass it, and it does not change the TTY
    gate — the gate asks about **stdout** regardless of where the banner goes,
    because the question is whether a human is watching, not where the bytes land.
    """
    if suppressed or os.environ.get(ENV_VAR):
        return False
    if not (hasattr(sys.stdout, "isatty") and sys.stdout.isatty()):
        return False
    print(BANNER.strip("\n"), file=stream if stream is not None else sys.stderr)
    return True


def add_banner_argument(parser) -> None:
    """Add ``--no-banner`` to an ``argparse`` parser. One line per entry point."""
    parser.add_argument(
        "--no-banner", action="store_true", help="suppress the startup banner"
    )
