from __future__ import annotations

from .base import BaseTest
from .config import (
    CrossLanguageTestConfig,
    TestConfig,
    LanguageConfig,
)
from .test1 import Test1
from .test2 import Test2
from .test3 import Test3
from .test4 import Test4

__all__ = [
    # Configs
    "TestConfig",
    "LanguageConfig",
    "CrossLanguageTestConfig",
    # Tests
    "BaseTest",
    "ALL_TESTS",
    "Test1",
    "Test2",
    "Test3",
    "Test4",
]

ALL_TESTS: list[type[BaseTest]] = [
    Test1,
    Test2,
    Test3,
    Test4,
]
