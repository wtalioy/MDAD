from .base import BaseRawDataset
from .news import News
from .podcast import Podcast
from .movie import Movie
from .phonecall import PhoneCall
from .interview import Interview
from .publicspeech import PublicSpeech
from .partialfake import PartialFake

RAWDATASET_MAP = {
    "news": News,
    "podcast": Podcast,
    "movie": Movie,
    "phonecall": PhoneCall,
    "interview": Interview,
    "publicspeech": PublicSpeech,
    "partialfake": PartialFake,
}

__all__ = ["BaseRawDataset", "RAWDATASET_MAP"]