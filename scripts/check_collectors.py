"""Compile the native collectors against real vendor headers, on any machine.

    python scripts/check_collectors.py [--vendor amd|nvidia|all] [--ref rocm-7.2.3]
                                       [--cuda 12] [--keep DIR]

amd: fetches a ROCm release's rocprofiler-sdk/HIP/HSA/libhsakmt headers from
ROCm/rocm-systems and compiles rocm_inject.c. nvidia: pulls the CUPTI, CUDA
runtime and NVTX header wheels from PyPI and compiles cupti_core.c (with the
capture-time node map) and cupti_inject.c. Both with -Wall -Wextra -Werror.

Compiler: $CC, cc/gcc/clang, else ``python -m ziglang cc`` (``pip install
ziglang``). Point GITM_ROCM_INCLUDE / GITM_CUDA_INCLUDE at ``--keep`` trees to
run the compile tests in tests/test_*_collector_contract.py.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

TRACER = Path(__file__).resolve().parents[1] / "gitm" / "tracer"
REPO = "ROCm/rocm-systems"
TREES = {  # repo prefix -> destination under the include root
    "projects/rocprofiler-sdk/source/include/": "",
    "projects/hip/include/": "",
    "projects/clr/hipamd/include/": "",
    "projects/rocr-runtime/libhsakmt/include/": "",
    "projects/rocr-runtime/runtime/hsa-runtime/inc/": "hsa/",
}
#: Header wheels per CUDA major. crt/ (host_defines.h) ships with nvcc on 12 and
#: in its own wheel on 13, where the -cuXX suffix was dropped.
CUDA_WHEELS = {
    "12": ("nvidia-cuda-cupti-cu12", "nvidia-cuda-runtime-cu12", "nvidia-nvtx-cu12",
           "nvidia-cuda-nvcc-cu12"),
    "13": ("nvidia-cuda-cupti==13.*", "nvidia-cuda-runtime==13.*", "nvidia-nvtx==13.*",
           "nvidia-cuda-crt==13.*"),
}


def _get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


def fetch_rocm(ref: str, root: Path) -> None:
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
    for tpl in root.rglob("*.h.in"):  # fill the CMake templates
        out = tpl.with_suffix("")
        if not out.exists():
            text = re.sub(r"#cmakedefine(?:01)? (\w+)(?: @\w+@)?", r"#define \1 1",
                          tpl.read_text(encoding="utf-8"))
            out.write_text(re.sub(r"@\w+@", "0", text), encoding="utf-8")
    m = re.match(r"rocm-(\d+)\.(\d+)", ref)
    major, minor = m.groups() if m else ("99", "0")
    (root / "hip" / "hip_version.h").write_text(
        f"#pragma once\n#define HIP_VERSION_MAJOR {major}\n#define HIP_VERSION_MINOR {minor}\n"
        "#define HIP_VERSION_PATCH 0\n"
        "#define HIP_VERSION (HIP_VERSION_MAJOR * 10000000 + HIP_VERSION_MINOR * 100000)\n",
        encoding="utf-8")
    print(f"fetched {len(jobs)} ROCm headers for {ref}")


def fetch_cuda(major: str, root: Path) -> None:
    with tempfile.TemporaryDirectory() as d:
        platforms = [a for p in ("manylinux2014_x86_64", "manylinux_2_25_x86_64",
                                 "manylinux_2_27_x86_64",
                                 "manylinux_2_28_x86_64") for a in ("--platform", p)]
        subprocess.run([sys.executable, "-m", "pip", "download", "-q", "--no-deps",
                        "--only-binary=:all:", *platforms, "--python-version", "3.12",
                        "-d", d, *CUDA_WHEELS[major]], check=True)
        for whl in glob.glob(os.path.join(d, "*.whl")):
            with zipfile.ZipFile(whl) as z:
                for n in z.namelist():
                    if "/include/" in n and not n.endswith("/"):
                        out = root / n.split("/include/", 1)[1]
                        out.parent.mkdir(parents=True, exist_ok=True)
                        out.write_bytes(z.read(n))
    print(f"extracted CUDA {major} headers")


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


def compile_source(src: Path, include: Path, out_dir: Path, *flags: str
                   ) -> subprocess.CompletedProcess:
    cc = compiler()
    if cc is None:
        raise SystemExit("no C compiler; pip install ziglang")
    return subprocess.run(
        [*cc, "-c", "-fPIC", "-O2", "-Wall", "-Wextra", "-Werror", "-pthread", *flags,
         "-isystem", str(include), str(src), "-o", str(out_dir / (src.stem + ".o"))],
        capture_output=True, text=True)


def compile_rocm(include: Path, out_dir: Path) -> subprocess.CompletedProcess:
    return compile_source(TRACER / "_rocm" / "rocm_inject.c", include, out_dir,
                          "-D__HIP_PLATFORM_AMD__")


def compile_cupti(include: Path, out_dir: Path) -> list[subprocess.CompletedProcess]:
    here = TRACER / "_cupti"
    return [compile_source(here / f, include, out_dir, f"-I{here}")
            for f in ("cupti_core.c", "cupti_inject.c")]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--vendor", choices=("amd", "nvidia", "all"), default="all")
    ap.add_argument("--ref", default="rocm-7.2.3", help="rocm-systems tag or branch")
    ap.add_argument("--cuda", default="12", choices=sorted(CUDA_WHEELS),
                    help="CUDA major for the header wheels")
    ap.add_argument("--keep", type=Path, help="directory for the header trees (reused)")
    args = ap.parse_args(argv)
    work = Path(tempfile.mkdtemp(prefix="gitm-collectors-"))
    base = args.keep or work
    results: list[tuple[str, subprocess.CompletedProcess]] = []
    if args.vendor in ("amd", "all"):
        root = base / f"rocm-{args.ref}"
        if not (root / "rocprofiler-sdk" / "rocprofiler.h").exists():
            fetch_rocm(args.ref, root)
        results.append(("rocm_inject.c", compile_rocm(root, work)))
    if args.vendor in ("nvidia", "all"):
        root = base / f"cuda-{args.cuda}"
        if not (root / "cupti.h").exists():
            fetch_cuda(args.cuda, root)
        results += zip(("cupti_core.c", "cupti_inject.c"), compile_cupti(root, work), strict=True)
    failed = False
    for name, r in results:
        if r.returncode:
            failed = True
            print(r.stderr, file=sys.stderr)
        print(f"{'FAIL' if r.returncode else 'OK'}: {name}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
