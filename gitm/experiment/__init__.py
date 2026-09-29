"""The experiment contract and the evaluator that turns one into a verdict.

See ``docs/autoresearch_experiment_contract.md``.
"""

from gitm.experiment.contract import (
    SCHEMA,
    VERDICT_SCHEMA,
    ExperimentContract,
    Verdict,
    load_contract,
)
from gitm.experiment.evaluate import Sample, Sampler, evaluate

__all__ = [
    "SCHEMA",
    "VERDICT_SCHEMA",
    "ExperimentContract",
    "Verdict",
    "load_contract",
    "Sample",
    "Sampler",
    "evaluate",
]
