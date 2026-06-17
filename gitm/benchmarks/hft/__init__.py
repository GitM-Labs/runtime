"""HFT LOB-replay benchmark, packaged so it ships in the wheel and is importable
from a pip install.

Two pieces live here because the runtime needs them: the cuDF/CuPy ``harness``
and the Parquet ``generate``-or (so the loop can auto-stage a smoke dataset with
no manual step). The manifest tooling (``gen_manifest.py``) and any staged
Parquet stay under the top-level ``benchmarks/hft/`` (repo-only, not shipped).
"""
