from .base import BaseDataset
from .publicfigure import PublicFigure
from .news import News
from .podcast import Podcast
from .partialfake import PartialFake
from .audiobook import Audiobook
from .noisyspeech import NoisySpeech
from .phonecall import PhoneCall
from .interview import Interview
from .publicspeech import PublicSpeech

DATASET_MAP = {
    "publicfigure": PublicFigure,
    "news": News,
    "podcast": Podcast,
    "partial_fake": PartialFake,
    "audiobook": Audiobook,
    "noisy_speech": NoisySpeech,
    "phonecall": PhoneCall,
    "interview": Interview,
    "publicspeech": PublicSpeech,
}

__all__ = ["BaseDataset", "DATASET_MAP"]