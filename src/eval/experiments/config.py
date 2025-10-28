from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

__all__ = ["ExperimentConfig", "LanguageConfig", "CrossLanguageExperimentConfig"]


@dataclass
class ExperimentConfig:
    """Configuration container for a single benchmark experiment."""

    train_datasets: List[str]
    val_datasets: List[str]
    test_sets: Dict[str, List[str]]
    subset: str | None = None


@dataclass
class LanguageConfig:
    """Configuration for an individual language-specific model."""

    name: str
    train_datasets: List[str]
    val_datasets: List[str]
    subset: str | None = None


@dataclass
class CrossLanguageExperimentConfig:
    """Configuration for experiments training models in multiple languages."""

    languages: List[LanguageConfig]
    test_sets: Dict[str, List[str]]
