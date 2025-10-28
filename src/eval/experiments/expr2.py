from __future__ import annotations
from typing import TYPE_CHECKING, Any

from .base import BaseExperiment
from .config import ExperimentConfig

if TYPE_CHECKING:
    from .runner import ExperimentRunner

__all__ = ["Experiment2"]

_CONFIG = ExperimentConfig(
    train_datasets=[
        "audiobook", "interview", "movie", "news", "phonecall",
        "podcast", "publicfigure", "publicspeech",
    ],
    val_datasets=[
        "audiobook", "interview", "movie", "news", "phonecall",
        "podcast", "publicfigure", "publicspeech",
    ],
    test_sets={"Neutral": ["audiobook", "podcast"], "Emotional": ["emotional"]},
)


class Experiment2(BaseExperiment):
    """Emotional Prosody Uncanny Valley Test."""

    name = "expr2"

    @classmethod
    def run(cls, runner: ExperimentRunner) -> dict[str, Any]:
        return runner._run_experiment(cls.name, _CONFIG)
