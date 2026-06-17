"""HFT LOB-replay benchmark — the cuDF/CuPy harness, packaged so it ships in the
wheel and is importable as ``gitm.benchmarks.hft.harness`` from a pip install.

The data *generators* (``generate.py``, ``gen_manifest.py``) and staged Parquet
remain under the top-level ``benchmarks/hft/`` (repo-only, not shipped) — only
the runtime harness lives here.
"""
