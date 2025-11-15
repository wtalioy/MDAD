from __future__ import annotations
from typing import TYPE_CHECKING, Any

from .base import BaseTest
from .config import TestConfig

if TYPE_CHECKING:
    from .runner import TestRunner

__all__ = ["Test3"]

_CONFIG = TestConfig(
    train_subsets=["interview", "podcast", "publicspeech"],
    val_subsets=["interview", "podcast", "publicspeech"],
    test_subsets={
        "CleanFull": ["interview", "podcast", "publicspeech"],
        "Partial": ["partialfake"],
        "Noisy": ["noisyspeech"],
    },
)


class Test3(BaseTest):
    """Sensitivity vs Robustness Test."""

    name = "test3"

    @classmethod
    def run(cls, runner: TestRunner) -> dict[str, Any]:
        return runner._run_test(cls.name, _CONFIG)
