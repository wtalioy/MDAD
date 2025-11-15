from __future__ import annotations
from typing import TYPE_CHECKING, Any

from .base import BaseTest
from .config import CrossLanguageTestConfig, TestConfig, LanguageConfig

if TYPE_CHECKING:
    from .runner import TestRunner

__all__ = ["Test4"]

# Dataset definitions
_EN_DATASETS = [
    "audiobook", "emotional", "interview", "movie",
    "podcast", "publicfigure", "publicspeech", "phonecall",
]
_ZH_DATASETS = ["news", "phonecall"]
_COMBINED_DATASETS = [
    "audiobook", "emotional", "interview", "movie",
    "podcast", "publicfigure", "publicspeech", "phonecall", "news",
]

# Cross-language generalization: separate models per language
_CONFIG_SEPARATE = CrossLanguageTestConfig(
    languages=[
        LanguageConfig(
            name="en",
            train_datasets=_EN_DATASETS,
            val_datasets=_EN_DATASETS,
            subset="en",
        ),
        LanguageConfig(
            name="zh",
            train_datasets=_ZH_DATASETS,
            val_datasets=_ZH_DATASETS,
            subset="zh-cn",
        ),
    ],
    test_sets={
        "en": _EN_DATASETS,
        "zh": _ZH_DATASETS,
    },
)

# Combined EN+ZH model tested on EN
_CONFIG_COMBINED_EN = TestConfig(
    train_datasets=_COMBINED_DATASETS,
    val_datasets=_COMBINED_DATASETS,
    test_sets={"en": _EN_DATASETS},
    subset=None,
)

# Combined EN+ZH model tested on ZH
_CONFIG_COMBINED_ZH = TestConfig(
    train_datasets=_COMBINED_DATASETS,
    val_datasets=_COMBINED_DATASETS,
    test_sets={"zh": _ZH_DATASETS},
    subset=None,
)


class Test4(BaseTest):
    """Cross-Language Generalization Test.
    
    Tests three scenarios:
    1. Separate models per language (trained on EN, trained on ZH)
    2. Combined multilingual model tested on EN
    3. Combined multilingual model tested on ZH
    """

    name = "test4"

    @classmethod
    def run(cls, runner: TestRunner) -> dict[str, Any]:
        results = {}
        
        # Scenario 1: Separate models per language
        results["separate_models"] = runner._run_cross_language_test(
            f"{cls.name}_separate", _CONFIG_SEPARATE
        )
        
        # Scenario 2: Combined model tested on EN
        results["combined_test_en"] = runner._run_test(
            f"{cls.name}_combined_en", _CONFIG_COMBINED_EN
        )
        
        # Scenario 3: Combined model tested on ZH
        results["combined_test_zh"] = runner._run_test(
            f"{cls.name}_combined_zh", _CONFIG_COMBINED_ZH
        )
        
        return results
