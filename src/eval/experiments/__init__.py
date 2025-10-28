from __future__ import annotations

from .base import BaseExperiment
from .config import (
    CrossLanguageExperimentConfig,
    ExperimentConfig,
    LanguageConfig,
)
from .expr1 import Experiment1
from .expr2 import Experiment2
from .expr3 import Experiment3
from .expr4 import Experiment4

__all__ = [
    # Configs
    "ExperimentConfig",
    "LanguageConfig",
    "CrossLanguageExperimentConfig",
    # Experiments
    "BaseExperiment",
    "ALL_EXPERIMENTS",
    "Experiment1",
    "Experiment2",
    "Experiment3",
    "Experiment4",
]

ALL_EXPERIMENTS: list[type[BaseExperiment]] = [
    Experiment1,
    Experiment2,
    Experiment3,
    Experiment4,
]
