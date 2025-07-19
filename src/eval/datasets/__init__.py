from .base import BaseDataset
from .public import PublicFigures
from .news import News
from .podcast import Podcast
from .PartialFake import PartialFake
from .audiobook import Audiobook
from .noisyspeech import NoisySpeech

DATASET_MAP = {
    "public": PublicFigures,
    "news": News,
    "podcast": Podcast,
    "partial_fake": PartialFake,
    "audiobook": Audiobook,
    "noisy_speech": NoisySpeech,
}

__all__ = ["BaseDataset", "DATASET_MAP"]