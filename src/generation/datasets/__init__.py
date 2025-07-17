from .base import BaseRawDataset
from .news import News

RAWDATASET_MAP = {
    "news": News,
}

__all__ = ["BaseRawDataset", "RAWDATASET_MAP"]