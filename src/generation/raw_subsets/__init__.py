from .base import BaseRawSubset
from .news import News
from .podcast import Podcast
from .movie import Movie
from .phonecall import PhoneCall
from .interview import Interview
from .publicspeech import PublicSpeech
from .partialfake import PartialFake
from .noisyspeech import NoisySpeech

RAW_SUBSET_MAP = {
    "news": News,
    "podcast": Podcast,
    "movie": Movie,
    "phonecall": PhoneCall,
    "interview": Interview,
    "publicspeech": PublicSpeech,
    "partialfake": PartialFake,
    "noisyspeech": NoisySpeech,
}

__all__ = ["BaseRawSubset", "RAW_SUBSET_MAP"]