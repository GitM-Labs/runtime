"""Deployment surface - Git.M attaches to a running job

Standalone attach lives here `gitm attach --job`
k8s operator and Slurm wrappers alongside it.
Every mode holds the same invariants: user spaces,
no root, no dirver replacement, no out-of-cluster cells, and fail-open
(our death must leave the workload untouched)
"""

from __future__ import annotations

from gitm.deploy.attach import AttachPlan, attach_job

__all__ = ["AttachPlan", "attach_job"]
