from .base import BaseDataset
from .public import PublicFigures
from .news import News
from .podcasts import Podcasts

DATASET_MAP = {
    "public": PublicFigures,
    "news": News,
    "podcasts": Podcasts,
}

__all__ = ["BaseDataset", "DATASET_MAP"]