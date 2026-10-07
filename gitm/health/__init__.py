"""Pre-loop GPU collective readiness (NCCL / RCCL AllReduce)."""

from gitm.health.collective import HealthResult, detect_gpus, run_collective_health

__all__ = ["HealthResult", "detect_gpus", "run_collective_health"]
