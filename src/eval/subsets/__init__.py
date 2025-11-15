from .base import BaseSubset
from .publicfigure import PublicFigure
from .news import News
from .podcast import Podcast
from .partialfake import PartialFake
from .audiobook import Audiobook
from .noisyspeech import NoisySpeech
from .phonecall import PhoneCall
from .interview import Interview
from .publicspeech import PublicSpeech
from .movie import Movie
from .emotional import Emotional

SUBSET_MAP = {
    "publicfigure": PublicFigure,
    "news": News,
    "podcast": Podcast,
    "partialfake": PartialFake,
    "audiobook": Audiobook,
    "noisyspeech": NoisySpeech,
    "phonecall": PhoneCall,
    "interview": Interview,
    "publicspeech": PublicSpeech,
    "movie": Movie,
    "emotional": Emotional,
}

__all__ = ["BaseSubset", "SUBSET_MAP"]