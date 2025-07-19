from .base import BaseRawDataset
from .news import News
from .podcast import Podcast
from .movie import Movie

RAWDATASET_MAP = {
    "news": News,
    "podcast": Podcast,
    "movie": Movie,
}

__all__ = ["BaseRawDataset", "RAWDATASET_MAP"]