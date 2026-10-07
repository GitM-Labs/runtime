"""Compile rocm_inject.c against a ROCm release's real headers, on any machine.

    python scripts/check_rocm_collector.py [--ref rocm-7.2.3|develop] [--keep DIR]

Fetches the release's rocprofiler-sdk, HIP, HSA and libhsakmt headers from
ROCm/rocm-systems, fills the CMake-generated version headers, and compiles with
-Wall -Wextra -Werror. Compiler: $CC, cc/gcc/clang, else ``python -m ziglang cc``
(``pip install ziglang``). Point GITM_ROCM_INCLUDE at a ``--keep`` tree to run
the compile test in tests/test_rocm_collector_contract.py.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

REPO = "ROCm/rocm-systems"
SRC = Path(__file__).resolve().parents[1] / "gitm" / "tracer" / "_rocm" / "rocm_inject.c"
TREES = {  # repo prefix -> destination under the include root
    "projects/rocprofiler-sdk/source/include/": "",
    "projects/hip/include/": "",
    "projects/clr/hipamd/include/": "",
    "projects/rocr-runtime/libhsakmt/include/": "",
    "projects/rocr-runtime/runtime/hsa-runtime/inc/": "hsa/",
}


def _get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


def fetch(ref: str, root: Path) -> None:
    tree = json.loads(_get(f"https://api.github.com/repos/{REPO}/git/trees/{ref}?recursive=1"))
    jobs = [(e["path"], root / dest / e["path"][len(prefix):])
            for e in tree["tree"] if e["type"] == "blob"
            and e["path"].endswith((".h", ".hpp", ".h.in"))
            for prefix, dest in TREES.items() if e["path"].startswith(prefix)]

    def one(job):
        path, out = job
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(_get(f"https://raw.githubusercontent.com/{REPO}/{ref}/{path}"))

    with cf.ThreadPoolExecutor(16) as ex:
        list(ex.map(one, jobs))
    print(f"fetched {len(jobs)} headers for {ref}")


def configure(root: Path, ref: str) -> None:
    for tpl in root.rglob("*.h.in"):
        out = tpl.with_suffix("")
        if not out.exists():
            text = re.sub(r"#cmakedefine(?:01)? (\w+)(?: @\w+@)?", r"#define \1 1",
                          tpl.read_text(encoding="utf-8"))
            out.write_text(re.sub(r"@\w+@", "0", text), encoding="utf-8")
    hv = root / "hip" / "hip_version.h"
    if not hv.exists():
        m = re.match(r"rocm-(\d+)\.(\d+)", ref)
        major, minor = m.groups() if m else ("99", "0")
        hv.write_text(f"#pragma once\n#define HIP_VERSION_MAJOR {major}\n"
                      f"#define HIP_VERSION_MINOR {minor}\n#define HIP_VERSION_PATCH 0\n"
                      f"#define HIP_VERSION (HIP_VERSION_MAJOR * 10000000 + "
                      f"HIP_VERSION_MINOR * 100000)\n", encoding="utf-8")


def compiler() -> list[str] | None:
    if os.environ.get("CC"):
        return [os.environ["CC"]]
    for cc in ("cc", "gcc", "clang"):
        if shutil.which(cc):
            return [cc]
    try:
        import ziglang  # noqa: F401
    except ImportError:
        return None
    return [sys.executable, "-m", "ziglang", "cc", "-target", "x86_64-linux-gnu"]


def compile_collector(include: Path, out_dir: Path) -> subprocess.CompletedProcess:
    cc = compiler()
    if cc is None:
        raise SystemExit("no C compiler; pip install ziglang")
    return subprocess.run(
        [*cc, "-c", "-fPIC", "-O2", "-Wall", "-Wextra", "-Werror", "-pthread",
         "-D__HIP_PLATFORM_AMD__", "-isystem", str(include), str(SRC),
         "-o", str(out_dir / "rocm_inject.o")], capture_output=True, text=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ref", default="rocm-7.2.3", help="rocm-systems tag or branch")
    ap.add_argument("--keep", type=Path, help="header tree directory to create or reuse")
    args = ap.parse_args(argv)
    work = Path(tempfile.mkdtemp(prefix="gitm-rocm-"))
    root = args.keep or work / "include"
    if not (root / "rocprofiler-sdk" / "rocprofiler.h").exists():
        fetch(args.ref, root)
    configure(root, args.ref)
    r = compile_collector(root, work)
    if r.returncode:
        print(r.stderr, file=sys.stderr)
        print("FAIL: rocm_inject.c does not compile against these headers", file=sys.stderr)
        return 1
    print("OK: rocm_inject.c compiles cleanly (-Wall -Wextra -Werror)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
