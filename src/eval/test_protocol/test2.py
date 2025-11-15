from __future__ import annotations
from typing import TYPE_CHECKING, Any

from .base import BaseTest
from .config import TestConfig

if TYPE_CHECKING:
    from .runner import TestRunner

__all__ = ["Test2"]

_CONFIG = TestConfig(
    train_subsets=[
        "audiobook", "interview", "movie", "news", "phonecall",
        "podcast", "publicfigure", "publicspeech",
    ],
    val_subsets=[
        "audiobook", "interview", "movie", "news", "phonecall",
        "podcast", "publicfigure", "publicspeech",
    ],
    test_subsets={"Neutral": ["audiobook", "podcast"], "Emotional": ["emotional"]},
)


class Test2(BaseTest):
    """Emotional Prosody Uncanny Valley Test."""

    name = "test2"

    @classmethod
    def run(cls, runner: TestRunner) -> dict[str, Any]:
        return runner._run_test(cls.name, _CONFIG)
