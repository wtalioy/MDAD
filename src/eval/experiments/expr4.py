from __future__ import annotations
from typing import TYPE_CHECKING, Any

from .base import BaseExperiment
from .config import CrossLanguageExperimentConfig, LanguageConfig

if TYPE_CHECKING:
    from .runner import ExperimentRunner

__all__ = ["Experiment4"]

_CONFIG = CrossLanguageExperimentConfig(
    languages=[
        LanguageConfig(
            name="en",
            train_datasets=[
                "audiobook", "emotional", "interview", "movie", "podcast",
                "publicfigure", "publicspeech", "phonecall",
            ],
            val_datasets=[
                "audiobook", "emotional", "interview", "movie", "podcast",
                "publicfigure", "publicspeech", "phonecall",
            ],
            subset="en",
        ),
        LanguageConfig(
            name="zh",
            train_datasets=["news", "phonecall"],
            val_datasets=["news", "phonecall"],
            subset="zh-cn",
        ),
    ],
    test_sets={
        "en": [
            "audiobook", "emotional", "interview", "movie", "podcast",
            "publicfigure", "publicspeech", "phonecall",
        ],
        "zh": ["news", "phonecall"],
    },
)


class Experiment4(BaseExperiment):
    """Cross-Language Generalization Test."""

    name = "expr4"

    @classmethod
    def run(cls, runner: ExperimentRunner) -> dict[str, Any]:
        return runner._run_cross_language_experiment(cls.name, _CONFIG)
