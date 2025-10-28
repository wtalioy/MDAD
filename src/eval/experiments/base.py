from __future__ import annotations
import abc
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..experiment_runner import ExperimentRunner

__all__ = ["BaseExperiment"]


class BaseExperiment(abc.ABC):
    """Abstract base class for all benchmark experiments."""

    name: str

    @classmethod
    @abc.abstractmethod
    def run(cls, runner: ExperimentRunner) -> dict[str, Any]:
        """Execute the experiment using the provided runner."""
        raise NotImplementedError
