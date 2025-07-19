from .base import BaseRawDataset
from .news import News
from .podcast import Podcast

RAWDATASET_MAP = {
    "news": News,
    "podcast": Podcast,
}

__all__ = ["BaseRawDataset", "RAWDATASET_MAP"]