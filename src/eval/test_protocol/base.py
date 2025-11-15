from __future__ import annotations
import abc
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .runner import TestRunner

__all__ = ["BaseTest"]


class BaseTest(abc.ABC):
    """Abstract base class for all benchmark tests."""

    name: str

    @classmethod
    @abc.abstractmethod
    def run(cls, runner: TestRunner) -> dict[str, Any]:
        """Execute the test using the provided runner."""
        raise NotImplementedError
