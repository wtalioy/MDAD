from __future__ import annotations
from typing import TYPE_CHECKING, Any

from .base import BaseExperiment
from .config import ExperimentConfig

if TYPE_CHECKING:
    from .runner import ExperimentRunner

__all__ = ["Experiment1"]


_CONFIG = ExperimentConfig(
    train_datasets=["audiobook", "news"],
    val_datasets=["audiobook", "news"],
    test_sets={
        "InDomain": ["audiobook", "news"],
        "Spontaneous": ["interview", "podcast", "phonecall"],
        "RealWorld": ["movie", "publicfigure", "publicspeech"],
    },
)


class Experiment1(BaseExperiment):
    """Domain Generalization Stress Test (Scripted-to-Spontaneous)."""

    name = "expr1"

    @classmethod
    def run(cls, runner: ExperimentRunner) -> dict[str, Any]:
        """Execute Experiment 1 using *runner*."""
        return runner._run_experiment(cls.name, _CONFIG)
