"""Compile the ROCm collector against a ROCm release's real headers, no ROCm needed.

    python scripts/check_rocm_collector.py                  # rocm-7.2.3 (the MI355X box)
    python scripts/check_rocm_collector.py --ref develop    # what the next release will be
    python scripts/check_rocm_collector.py --ref rocm-7.2.3 --keep /tmp/rocm-inc

``rocm_inject.c`` reads rocprofiler-sdk record structs by field name. A renamed
field fails the build on the GPU box, after a provisioning round-trip; this
fails it on a laptop. It fetches the exact header tree of one ROCm release from
ROCm/rocm-systems (rocprofiler-sdk, HIP, the HSA runtime and libhsakmt — what
the sdk's own headers include), fills the two CMake-generated version headers,
and compiles with ``-Wall -Wextra -Werror``.

The compiler is ``$CC``, else ``cc``/``gcc``/``clang``, else ``python -m
ziglang cc -target x86_64-linux-gnu`` (``pip install ziglang``) — a
cross-compiler, so this also runs on macOS and Windows. Pass ``--keep DIR`` to
reuse the tree, and point ``GITM_ROCM_INCLUDE`` at it to run the compile test
in tests/test_rocm_collector_contract.py.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

REPO = "ROCm/rocm-systems"
SRC = Path(__file__).resolve().parents[1] / "gitm" / "tracer" / "_rocm" / "rocm_inject.c"

#: repo prefix -> destination under the include root.
TREES = {
    "projects/rocprofiler-sdk/source/include/": "",
    "projects/hip/include/": "",
    "projects/clr/hipamd/include/": "",
    "projects/rocr-runtime/libhsakmt/include/": "",
    "projects/rocr-runtime/runtime/hsa-runtime/inc/": "hsa/",
}

HIP_VERSION_H = """#pragma once
#define HIP_VERSION_MAJOR {major}
#define HIP_VERSION_MINOR {minor}
#define HIP_VERSION_PATCH 0
#define HIP_VERSION_GITHASH ""
#define HIP_VERSION_BUILD_ID 0
#define HIP_VERSION_BUILD_NAME ""
#define HIP_VERSION (HIP_VERSION_MAJOR * 10000000 + HIP_VERSION_MINOR * 100000)
"""


def _get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


def fetch(ref: str, root: Path) -> None:
    tree = json.loads(_get(f"https://api.github.com/repos/{REPO}/git/trees/{ref}?recursive=1"))
    if tree.get("truncated"):
        raise SystemExit(f"tree listing for {ref} is truncated; cannot select headers")
    jobs = []
    for e in tree["tree"]:
        path = e["path"]
        if e["type"] != "blob" or not path.endswith((".h", ".hpp", ".h.in")):
            continue
        for prefix, dest in TREES.items():
            if path.startswith(prefix):
                jobs.append((path, root / dest / path[len(prefix):]))

    def one(job):
        path, out = job
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(_get(f"https://raw.githubusercontent.com/{REPO}/{ref}/{path}"))

    with cf.ThreadPoolExecutor(16) as ex:
        list(ex.map(one, jobs))
    print(f"fetched {len(jobs)} headers for {ref}")


def configure(root: Path, ref: str) -> None:
    """Fill the CMake templates the way the sdk's build would."""
    m = re.match(r"rocm-(\d+)\.(\d+)", ref)
    major, minor = (m.group(1), m.group(2)) if m else ("99", "0")
    for tpl in root.rglob("*.h.in"):
        out = tpl.with_suffix("")
        if out.exists():
            continue
        text = tpl.read_text(encoding="utf-8")
        text = re.sub(r"#cmakedefine01 (\w+)", r"#define \1 1", text)
        text = re.sub(r"#cmakedefine (\w+) @\w+@", r"#define \1 1", text)
        text = re.sub(r"#cmakedefine (\w+)", r"#define \1 1", text)
        text = re.sub(r"@\w+_MAJOR@", "1", text)
        text = re.sub(r"@\w+_MINOR@", "0", text)
        text = re.sub(r"@\w+_PATCH@", "0", text)
        text = re.sub(r"@\w+@", "0", text)
        out.write_text(text, encoding="utf-8")
    hv = root / "hip" / "hip_version.h"
    if not hv.exists():
        hv.write_text(HIP_VERSION_H.format(major=major, minor=minor), encoding="utf-8")


def compiler() -> list[str]:
    import os

    if os.environ.get("CC"):
        return [os.environ["CC"]]
    for cc in ("cc", "gcc", "clang"):
        if shutil.which(cc):
            return [cc]
    try:
        import ziglang  # noqa: F401
    except ImportError as exc:
        raise SystemExit("no C compiler; pip install ziglang") from exc
    return [sys.executable, "-m", "ziglang", "cc", "-target", "x86_64-linux-gnu"]


def compile_collector(root: Path, out_dir: Path) -> int:
    obj = out_dir / "rocm_inject.o"
    cmd = [*compiler(), "-c", "-fPIC", "-O2", "-Wall", "-Wextra", "-Werror", "-pthread",
           "-D__HIP_PLATFORM_AMD__", "-isystem", str(root), str(SRC), "-o", str(obj)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    noise = "argument unused during compilation"
    err = "\n".join(line for line in r.stderr.splitlines() if noise not in line)
    if r.returncode != 0 or not obj.exists():
        print(err, file=sys.stderr)
        print("FAIL: rocm_inject.c does not compile against these headers", file=sys.stderr)
        return 1
    print(f"OK: rocm_inject.c compiles cleanly (-Wall -Wextra -Werror), {obj.stat().st_size} B")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ref", default="rocm-7.2.3", help="rocm-systems tag or branch")
    ap.add_argument("--keep", type=Path, help="header tree directory to create or reuse")
    args = ap.parse_args(argv)
    work = Path(tempfile.mkdtemp(prefix="gitm-rocm-"))
    root = args.keep or (work / "include")
    if not (root / "rocprofiler-sdk" / "rocprofiler.h").exists():
        fetch(args.ref, root)
    configure(root, args.ref)
    return compile_collector(root, work)


if __name__ == "__main__":
    raise SystemExit(main())
