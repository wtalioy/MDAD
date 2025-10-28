from __future__ import annotations
from typing import TYPE_CHECKING, Any

from .base import BaseExperiment
from .config import ExperimentConfig

if TYPE_CHECKING:
    from .runner import ExperimentRunner

__all__ = ["Experiment3"]

_CONFIG = ExperimentConfig(
    train_datasets=["interview", "podcast", "publicspeech"],
    val_datasets=["interview", "podcast", "publicspeech"],
    test_sets={
        "CleanFull": ["interview", "podcast", "publicspeech"],
        "Partial": ["partialfake"],
        "Noisy": ["noisyspeech"],
    },
)


class Experiment3(BaseExperiment):
    """Sensitivity vs Robustness Test."""

    name = "expr3"

    @classmethod
    def run(cls, runner: ExperimentRunner) -> dict[str, Any]:
        return runner._run_experiment(cls.name, _CONFIG)
