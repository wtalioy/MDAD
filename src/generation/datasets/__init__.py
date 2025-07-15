from .base import BaseRawDataset
from .cmlr import CMLR

RAWDATASET_MAP = {
    "cmlr": CMLR,
}

__all__ = ["BaseRawDataset", "RAWDATASET_MAP"]