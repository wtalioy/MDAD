from .base import BaseDataset
from .public import PublicFigures

DATASET_MAP = {
    "public": PublicFigures,
}

__all__ = ["BaseDataset", "DATASET_MAP"]