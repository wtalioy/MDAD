from __future__ import annotations
from typing import TYPE_CHECKING, Any

from .base import BaseTest
from .config import TestConfig

if TYPE_CHECKING:
    from .runner import TestRunner

__all__ = ["Test1"]


_CONFIG = TestConfig(
    train_datasets=["audiobook", "news"],
    val_datasets=["audiobook", "news"],
    test_sets={
        "InDomain": ["audiobook", "news"],
        "Spontaneous": ["interview", "podcast", "phonecall"],
        "RealWorld": ["movie", "publicfigure", "publicspeech"],
    },
)


class Test1(BaseTest):
    """Domain Generalization Test (In-Domain vs. Spontaneous vs. Real-World)."""

    name = "test1"

    @classmethod
    def run(cls, runner: TestRunner) -> dict[str, Any]:
        return runner._run_test(cls.name, _CONFIG)
